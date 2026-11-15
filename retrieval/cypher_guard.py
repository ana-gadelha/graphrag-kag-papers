"""Parse a closed read-only Cypher language and compile bounded execution.

The parser's output is the only input accepted by the compiler. Generated text
is never sent to Neo4j, including when syntax or semantic validation fails.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

from lark import Lark, Tree
from lark.exceptions import LarkError

MAX_SEEDS = 8
MAX_NEIGHBORS = 20
MAX_CANDIDATES = 40
TIMEOUT_SECONDS = 5
LABELS = frozenset({"Paper", "Author", "Concept", "Method", "Dataset"})
RESERVED_ALIASES = frozenset(
    [
        "MATCH",
        "WHERE",
        "RETURN",
        "AS",
        "LIMIT",
        "DISTINCT",
        "AND",
        "OR",
        "NOT",
        "IN",
        "CALL",
        "WITH",
        "UNION",
        "CREATE",
        "DELETE",
        "DETACH",
        "SET",
        "REMOVE",
        "DROP",
        "LOAD",
        "CSV",
        "YIELD",
        "MERGE",
        "ORDER",
        "BY",
        "SKIP",
        "UNWIND",
        "OPTIONAL",
        "FOREACH",
        "CASE",
        "WHEN",
        "THEN",
        "ELSE",
        "END",
        "NULL",
        "TRUE",
        "FALSE",
    ]
)
RELATIONSHIPS = {
    "AUTHORED": ("Author", "Paper"),
    "HAS_CONCEPT": ("Paper", "Concept"),
    "PROPOSES": ("Paper", "Method"),
    "USES_METHOD": ("Paper", "Method"),
    "EVALUATED_ON": ("Paper", "Dataset"),
}


class InvalidCypher(ValueError):
    """A generated plan failed the syntax, schema or parameter allowlist."""


@dataclass(frozen=True)
class Predicate:
    alias: str
    property: str
    operator: str
    parameter: str


@dataclass(frozen=True)
class Edge:
    left: str
    right: str
    relationship: str
    direction: str
    match_group: int = 0


@dataclass(frozen=True)
class ValidatedQuery:
    """Validated AST, its canonical form, and the bounded execution query."""

    canonical: str
    executed: str
    parameters: dict[str, Any]
    original_parameters: dict[str, Any]
    return_alias: str
    root_alias: str
    nodes: dict[str, str]
    edges: tuple[Edge, ...]
    predicates: tuple[Predicate, ...]
    limit: int


@lru_cache(maxsize=1)
def _parser() -> Lark:
    return Lark.open(str(Path(__file__).with_name("cypher_subset.lark")), parser="lalr")


def _children(tree: Tree, kind: str) -> list[Tree]:
    return [
        child
        for child in tree.children
        if isinstance(child, Tree) and child.data == kind
    ]


def _identity_property(label: str) -> str:
    return "id" if label == "Paper" else "entity_id"


def _render_edge(
    edge: Edge,
    *,
    labels: dict[str, str] | None = None,
    relationship_alias: str = "",
) -> str:
    def node(alias: str) -> str:
        return f"({alias}:{labels[alias]})" if labels is not None else f"({alias})"

    relationship = f"[{relationship_alias}:{edge.relationship}]"
    if edge.direction == "incoming":
        return f"{node(edge.left)}<-{relationship}-{node(edge.right)}"
    arrow = "->" if edge.direction == "outgoing" else "-"
    return f"{node(edge.left)}-{relationship}{arrow}{node(edge.right)}"


def _render_predicate(predicate: Predicate) -> str:
    return f"{predicate.alias}.{predicate.property} {predicate.operator} ${predicate.parameter}"


def _validate_value(
    predicate: Predicate,
    label: str,
    value: Any,
    corpus: set[str],
    vocabulary: list[dict],
) -> None:
    prop, operator = predicate.property, predicate.operator
    allowed = {"id", "year"} if label == "Paper" else {"entity_id", "name"}
    if prop not in allowed:
        raise InvalidCypher("Propriedade fora do schema permitido.")
    if prop == "year":
        if (
            operator not in {"=", ">", "<", ">=", "<="}
            or type(value) is not int
            or not 1000 <= value <= 3000
        ):
            raise InvalidCypher("Ano ou operador de ano inválido.")
        return
    if operator not in {"=", "IN"}:
        raise InvalidCypher("Identificadores aceitam apenas igualdade ou IN.")
    values = value if operator == "IN" else [value]
    if not isinstance(values, list) or not 1 <= len(values) <= MAX_SEEDS:
        raise InvalidCypher("Lista de identificadores deve conter de 1 a 8 valores.")
    known = (
        corpus
        if label == "Paper"
        else {
            entry.get(prop)
            for entry in vocabulary
            if entry.get("label") == label and isinstance(entry.get(prop), str)
        }
    )
    if any(not isinstance(item, str) or item not in known for item in values):
        raise InvalidCypher("Identificador ou nome ausente no vocabulário autorizado.")


def validate_and_compile(
    query: str,
    params: dict[str, Any],
    *,
    seed_ids: list[str],
    vocabulary: list[dict],
    corpus_ids: list[str],
) -> ValidatedQuery:
    """Validate syntax and values, then impose bounds before each expansion.

    Names must match canonical vocabulary names exactly. Alias resolution belongs
    to the earlier stage, which supplies the model with stable entity IDs.
    """
    if not isinstance(query, str) or not query.strip() or len(query) > 12000:
        raise InvalidCypher("Cypher ausente ou muito extenso.")
    if not isinstance(params, dict) or any(
        not isinstance(key, str) or key.startswith("guard_") for key in params
    ):
        raise InvalidCypher("Parâmetros inválidos ou reservados.")
    try:
        tree = _parser().parse(query)
    except LarkError:
        raise InvalidCypher("Cypher fora da gramática de leitura permitida.") from None

    nodes: dict[str, str] = {}
    edges: list[Edge] = []
    predicates: list[Predicate] = []
    patterns: list[tuple[list[str], list[Edge], int]] = []
    for match_group, clause in enumerate(_children(tree, "match_clause")):
        for pattern in _children(clause, "pattern"):
            aliases: list[str] = []
            pattern_edges: list[Edge] = []
            pending: tuple[str, str] | None = None
            for item in pattern.children:
                if item.data != "node":
                    pending = (str(item.children[0]), str(item.data))
                    continue
                alias = str(item.children[0])
                if alias.startswith("guard_") or alias.upper() in RESERVED_ALIASES:
                    raise InvalidCypher("Alias reservado.")
                labels = _children(item, "label")
                if labels:
                    label = str(labels[0].children[0])
                    if label not in LABELS or (
                        alias in nodes and nodes[alias] != label
                    ):
                        raise InvalidCypher(
                            "Rótulo fora do schema ou alias inconsistente."
                        )
                    nodes[alias] = label
                elif alias not in nodes:
                    raise InvalidCypher(
                        "Todo nó precisa declarar seu rótulo na primeira ocorrência."
                    )
                for properties in _children(item, "properties"):
                    for pair in _children(properties, "property_pair"):
                        predicates.append(
                            Predicate(
                                alias,
                                str(pair.children[0]),
                                "=",
                                str(pair.children[1])[1:],
                            )
                        )
                if pending:
                    edge = Edge(aliases[-1], alias, *pending, match_group)
                    edges.append(edge)
                    pattern_edges.append(edge)
                aliases.append(alias)
            if len(pattern_edges) > 4:
                raise InvalidCypher("Caminhos limitados a quatro saltos explícitos.")
            patterns.append((aliases, pattern_edges, match_group))
    if not 1 <= len(patterns) <= 3 or len(edges) > 6 or len(nodes) > 7:
        raise InvalidCypher("Consulta excede três padrões ou seis relações.")
    for clause in _children(tree, "where_clause"):
        for item in _children(clause, "predicate"):
            prop, operator, parameter = item.children
            predicates.append(
                Predicate(
                    str(prop.children[0]),
                    str(prop.children[1]),
                    str(operator).upper(),
                    str(parameter)[1:],
                )
            )
    returns = _children(tree, "return_clause")[0]
    returned_property = _children(returns, "property")[0]
    returned_alias, returned_key = map(str, returned_property.children)
    if nodes.get(returned_alias) != "Paper" or returned_key != "id":
        raise InvalidCypher("Somente IDs de Paper podem ser retornados.")
    raw_limit = str(_children(tree, "limit_clause")[0].children[0])
    if len(raw_limit) > 2:
        raise InvalidCypher("LIMIT deve estar entre 1 e 40.")
    limit = int(raw_limit)
    if not 1 <= limit <= MAX_CANDIDATES:
        raise InvalidCypher("LIMIT deve estar entre 1 e 40.")
    corpus = set(corpus_ids)
    referenced: set[str] = set()
    for predicate in predicates:
        if predicate.alias not in nodes or predicate.parameter not in params:
            raise InvalidCypher("Alias ou parâmetro indefinido.")
        _validate_value(
            predicate,
            nodes[predicate.alias],
            params[predicate.parameter],
            corpus,
            vocabulary,
        )
        referenced.add(predicate.parameter)
    entity_constraints = {p.alias for p in predicates if nodes[p.alias] != "Paper"}
    if len(entity_constraints) > 3:
        raise InvalidCypher("Interseções limitadas a três entidades identificadas.")
    if referenced != set(params):
        raise InvalidCypher("Parâmetros extras não são permitidos.")
    for edge in edges:
        if edge.relationship not in RELATIONSHIPS or edge.left == edge.right:
            raise InvalidCypher("Relação fora do schema permitido.")
        expected = RELATIONSHIPS[edge.relationship]
        actual = (nodes[edge.left], nodes[edge.right])
        if edge.direction == "incoming":
            actual = actual[::-1]
        if actual != expected and not (
            edge.direction == "undirected" and actual[::-1] == expected
        ):
            raise InvalidCypher("Relação incompatível com os rótulos ou a direção.")
    reached = {returned_alias}
    while True:
        expanded = (
            reached
            | {edge.right for edge in edges if edge.left in reached}
            | {edge.left for edge in edges if edge.right in reached}
        )
        if expanded == reached:
            break
        reached = expanded
    if reached != set(nodes):
        raise InvalidCypher("Padrões desconectados não são permitidos.")
    # Bound the distance from every node to the returned Paper, even when the
    # model splits a longer path across several MATCH clauses.
    distances = {returned_alias: 0}
    for _ in nodes:
        for edge in edges:
            for source, target in ((edge.left, edge.right), (edge.right, edge.left)):
                if source in distances:
                    distances[target] = min(
                        distances.get(target, 99), distances[source] + 1
                    )
    if max(distances.values()) > 4:
        raise InvalidCypher("Distância máxima de quatro saltos excedida.")
    if not edges and not predicates:
        raise InvalidCypher("Consulta direta precisa restringir id ou ano.")

    by_alias = {alias: [p for p in predicates if p.alias == alias] for alias in nodes}
    anchored = [
        alias for alias in nodes if any(p.property != "year" for p in by_alias[alias])
    ]
    root = (
        min(
            anchored,
            key=lambda alias: (nodes[alias] == "Paper", list(nodes).index(alias)),
        )
        if anchored
        else returned_alias
    )
    execution_params = dict(params)
    seeds = list(dict.fromkeys(pid for pid in seed_ids if pid in corpus))[:MAX_SEEDS]
    if edges and not anchored and not seeds:
        raise InvalidCypher("Expansão requer sementes ou entidade identificada.")
    for alias, label in nodes.items():
        if label == "Paper":
            scope = (
                seeds if edges and not anchored and alias == root else sorted(corpus)
            )
        else:
            scope = sorted(
                {
                    entry["entity_id"]
                    for entry in vocabulary
                    if entry.get("label") == label
                    and isinstance(entry.get("entity_id"), str)
                }
            )
        execution_params[f"guard_scope_{alias}"] = scope

    def conditions(alias: str) -> str:
        key = _identity_property(nodes[alias])
        return " AND ".join(
            [
                f"{alias}.{key} IN $guard_scope_{alias}",
                *(_render_predicate(predicate) for predicate in by_alias[alias]),
            ]
        )

    def ordering(aliases: list[str]) -> str:
        return ", ".join(
            f"{alias}.{_identity_property(nodes[alias])}" for alias in aliases
        )

    parts = [
        f"MATCH ({root}:{nodes[root]})",
        f"WHERE {conditions(root)}",
        f"WITH DISTINCT {root} ORDER BY {ordering([root])} LIMIT {MAX_SEEDS if edges else limit}",
    ]
    bound = [root]
    bound_relationships: list[tuple[str, int]] = []
    pending_edges = list(enumerate(edges))
    while pending_edges:
        indexed_edge = next(
            item
            for item in pending_edges
            if item[1].left in bound or item[1].right in bound
        )
        pending_edges.remove(indexed_edge)
        edge_index, edge = indexed_edge
        relation_alias = f"guard_rel_{edge_index}"
        known = [*bound, *(alias for alias, _ in bound_relationships)]
        uniqueness = [
            f"{relation_alias} <> {alias}"
            for alias, group in bound_relationships
            if group == edge.match_group
        ]
        new_alias = edge.right if edge.left in bound else edge.left
        if new_alias in bound:
            parts.append(
                f"MATCH {_render_edge(edge, relationship_alias=relation_alias)}"
            )
            if uniqueness:
                parts.append("WHERE " + " AND ".join(uniqueness))
        else:
            # Carry relationship identities to preserve Cypher's no-reused-edge
            # rule within a MATCH clause, including across bounded subqueries.
            where = " AND ".join([conditions(new_alias), *uniqueness])
            parts.extend(
                [
                    f"CALL ({', '.join(known)}) {{",
                    f"  MATCH {_render_edge(edge, labels=nodes, relationship_alias=relation_alias)}",
                    f"  WHERE {where}",
                    f"  RETURN DISTINCT {new_alias}, {relation_alias} ORDER BY {ordering([new_alias])}, elementId({relation_alias}) LIMIT {MAX_NEIGHBORS}",
                    "}",
                ]
            )
            bound.append(new_alias)
        bound_relationships.append((relation_alias, edge.match_group))
        known = [*bound, *(alias for alias, _ in bound_relationships)]
        relation_order = ", ".join(
            f"elementId({alias})" for alias, _ in bound_relationships
        )
        parts.append(
            f"WITH DISTINCT {', '.join(known)} ORDER BY {ordering(bound)}, {relation_order} LIMIT {MAX_CANDIDATES}"
        )
    parts.append(
        f"RETURN DISTINCT {returned_alias}.id AS paper_id ORDER BY paper_id LIMIT {limit}"
    )
    canonical_patterns: dict[int, list[str]] = {}
    for aliases, pattern_edges, match_group in patterns:
        if not pattern_edges:
            alias = aliases[0]
            canonical_patterns.setdefault(match_group, []).append(
                f"({alias}:{nodes[alias]})"
            )
        else:
            text = _render_edge(pattern_edges[0], labels=nodes)
            for edge in pattern_edges[1:]:
                text += _render_edge(edge, labels=nodes).split(")", 1)[1]
            canonical_patterns.setdefault(match_group, []).append(text)
    canonical = " ".join(
        "MATCH " + ", ".join(group) for group in canonical_patterns.values()
    )
    if predicates:
        canonical += " WHERE " + " AND ".join(map(_render_predicate, predicates))
    canonical += f" RETURN DISTINCT {returned_alias}.id AS paper_id LIMIT {limit}"
    return ValidatedQuery(
        canonical,
        "\n".join(parts),
        execution_params,
        dict(params),
        returned_alias,
        root,
        nodes,
        tuple(edges),
        tuple(predicates),
        limit,
    )
