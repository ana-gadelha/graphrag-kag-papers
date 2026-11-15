"""Interface Streamlit para consulta aos artigos de GraphRAG/KAG."""

from __future__ import annotations

from typing import Any

import streamlit as st

from app_integration import consultar, url_http_valida
from graph_component import build_graph_payload, render_graph
from retrieval.config import RetrievalError

st.set_page_config(page_title="GraphRAG/KAG - Consulta a artigos", layout="wide")

st.title("Consulta a artigos de GraphRAG / KAG")
st.caption("Busca vetorial + grafo (Neo4j) + LLM")

pergunta = st.text_input(
    "Pergunta",
    placeholder="Ex.: Quais artigos utilizam GraphRAG e quais datasets foram utilizados?",
)

if st.button("Consultar", type="primary"):
    st.session_state.pop("resultado", None)
    for state_key in list(st.session_state):
        if isinstance(state_key, str) and state_key.startswith("evidence-graph-"):
            st.session_state.pop(state_key, None)
    if not pergunta.strip():
        st.warning("Digite uma pergunta antes de consultar.")
    else:
        with st.spinner("Consultando o grafo e gerando a resposta..."):
            try:
                st.session_state["resultado"] = consultar(pergunta)
            except RetrievalError as error:
                st.error(f"Não foi possível consultar: {error}")
            except Exception:  # noqa: BLE001 - avoid exposing provider/database secrets
                st.error("Não foi possível consultar agora. Tente novamente.")


def _query_text(query: Any, *, generated: bool = False) -> str:
    if isinstance(query, str):
        return query
    if isinstance(query, dict):
        keys = (
            ("original", "query", "cypher", "canonical")
            if generated or query.get("executed") is False
            else ("cypher", "canonical", "query", "original")
        )
        for key in keys:
            value = query.get(key)
            if isinstance(value, str) and value.strip():
                return value
    return "Consulta sem texto registrado"


def _render_query(query: Any, position: int, *, generated: bool = False) -> None:
    if isinstance(query, dict):
        stage = query.get("stage") or (
            "Planejada" if generated else f"Consulta {position}"
        )
        executed = query.get("executed")
        if executed is False:
            stage = f"{stage} · não executada"
        st.write(f"**{position}. {stage}**")
        st.code(_query_text(query, generated=generated), language="cypher")
        params = query.get("original_parameters" if generated else "parameters")
        if isinstance(params, dict):
            st.json(params)
        if generated and isinstance(query.get("canonical"), str) and query["canonical"]:
            st.caption("Forma validada:")
            st.code(query["canonical"], language="cypher")
        if generated and isinstance(query.get("cypher"), str) and query["cypher"]:
            st.caption("Forma limitada enviada ao Neo4j:")
            st.code(query["cypher"], language="cypher")
    else:
        st.write(
            f"**{position}. {'Planejada' if generated else 'Consulta executada'}**"
        )
        st.code(_query_text(query, generated=generated), language="cypher")


def _render_claims(claims: list[dict[str, Any]]) -> None:
    if not claims:
        return
    st.subheader("Afirmações estruturadas")
    for claim in claims:
        kind = claim.get("kind") or "Afirmação"
        statement = claim.get("text") or "Texto da afirmação não disponível."
        st.text(f"{kind} — {statement}")
        metadata = []
        source_ids = claim.get("source_ids")
        path_ids = claim.get("path_ids")
        if isinstance(source_ids, list) and source_ids:
            metadata.append(f"Fontes: {', '.join(map(str, source_ids))}")
        if isinstance(path_ids, list) and path_ids:
            metadata.append(f"Caminhos: {', '.join(map(str, path_ids))}")
        if metadata:
            st.caption(" · ".join(metadata))
        quotes = claim.get("quotes")
        if isinstance(quotes, list):
            for quote in quotes:
                if isinstance(quote, dict):
                    quote_text = quote.get("quote")
                    if not isinstance(quote_text, str):
                        continue
                    quote_source = quote.get("source_id", "")
                    quote_field = quote.get("field", "")
                    quote_range = "–".join(
                        str(quote[key])
                        for key in ("start", "end")
                        if quote.get(key) is not None
                    )
                    provenance = " · ".join(
                        str(value)
                        for value in (quote_source, quote_field, quote_range)
                        if value
                    )
                    st.text(
                        f"Trecho{f' ({provenance})' if provenance else ''}: {quote_text}"
                    )
                elif isinstance(quote, str):
                    st.text(f"Trecho: {quote}")


resultado = st.session_state.get("resultado")

if resultado:
    st.subheader("Resposta")
    st.markdown(resultado.get("resposta", ""))

    confidence = resultado.get("confidence", {})
    confidence = confidence if isinstance(confidence, dict) else {}
    level = {"low": "Baixa", "medium": "Média", "high": "Alta"}.get(
        confidence.get("level")
    )
    st.subheader("Confiança heurística nas evidências")
    if level:
        reason = confidence.get("reason") or "Motivo não informado."
        st.write(f"**{level}** — {reason}")
    else:
        st.info("Nível de confiança não disponível para este resultado.")
    st.caption(
        "Indica convergência dos canais nas fontes citadas; não representa "
        "probabilidade de acerto nem avaliação da veracidade da resposta."
    )

    for warning in resultado.get("warnings", []):
        st.warning(warning)

    planner = resultado.get("planner", {})
    planner = planner if isinstance(planner, dict) else {}
    planner_status = str(planner.get("status", "")).strip()
    planner_reason = str(planner.get("reason", "")).strip()
    if planner_status:
        st.caption(f"Planejador de consultas: **{planner_status}**")
    if planner_status.lower() not in {"", "executed", "empty", "disabled"}:
        st.warning(
            planner_reason
            or "O planejador ficou indisponível; a consulta seguiu pelo fluxo de recuperação de contingência."
        )

    graph_payload = build_graph_payload(
        resultado.get("nos", []),
        resultado.get("arestas", []),
        resultado.get("paths", []),
        resultado.get("claims", []),
    )
    st.subheader("Subgrafo recuperado")
    st.caption(
        "O grafo mantém a direção das relações. O caminho P1 fica destacado "
        "quando disponível; arraste, mova e amplie o grafo para inspecionar os dados."
    )
    render_graph(graph_payload)

    col_fontes, col_conexoes = st.columns([1.15, 1])
    with col_fontes:
        st.subheader("Fontes recuperadas")
        st.caption(
            "As fontes fornecem contexto; nem todas são necessariamente citadas na resposta."
        )
        fontes = resultado.get("fontes", [])
        if fontes:
            for fonte in fontes:
                citation_id = fonte.get("citation_id", "S?")
                title = fonte.get("titulo", "Artigo sem título")
                url = fonte.get("url")
                if url_http_valida(url):
                    st.link_button(f"[{citation_id}] {title}", url)
                else:
                    st.write(f"[{citation_id}] {title}")
                metadata = f"{fonte.get('autores', 'Autoria não informada')} · {fonte.get('ano') or 'Ano não informado'}"
                st.caption(metadata)
                if isinstance(fonte.get("score"), (int, float)):
                    st.caption(f"Relevância relativa (RRF): {fonte['score']:.6f}")
                channels = [
                    {"vector": "vetorial", "graph": "grafo"}.get(channel, channel)
                    for channel in fonte.get("retrieved_by", [])
                ]
                if channels:
                    st.caption(f"Canais de recuperação: {', '.join(channels)}")
        else:
            st.info("Nenhuma fonte foi recuperada para esta pergunta.")

    with col_conexoes:
        st.subheader("Conexões entre as fontes")
        st.caption(
            "Caminhos mínimos de até quatro saltos entre fontes e fatos de uma relação. "
            "Artigos conectores são apenas intermediários no grafo."
        )
        paths = resultado.get("paths", [])
        shortest_paths = [path for path in paths if path.get("kind") != "fact"]
        fact_paths = [path for path in paths if path.get("kind") == "fact"]
        for position, path in enumerate(shortest_paths, start=1):
            path_id = path.get("path_id", f"P{position}")
            st.write(
                f"**{path_id}** · [{path.get('source', '?')}] ↔ "
                f"[{path.get('target', '?')}] · {path.get('length', '?')} saltos"
            )
            sequence = []
            for node in path.get("nodes", []):
                if not isinstance(node, dict):
                    sequence.append(str(node))
                    continue
                properties = node.get("properties", {})
                properties = properties if isinstance(properties, dict) else {}
                name = (
                    properties.get("title")
                    or properties.get("name")
                    or node.get("id", "nó")
                )
                sequence.append(
                    f"{name} (conector)" if node.get("connector") else str(name)
                )
            if sequence:
                st.text(" → ".join(sequence))
        if fact_paths:
            st.write("**Fatos de uma relação**")
            with st.expander(f"Ver {len(fact_paths)} fatos citáveis"):
                for position, path in enumerate(fact_paths, start=1):
                    path_id = path.get("path_id", f"F{position}")
                    st.write(
                        f"{path_id} · [{path.get('source', '?')}] — "
                        f"{path.get('length', 1)} salto"
                    )
        if not shortest_paths and not fact_paths:
            if len(resultado.get("fontes", [])) < 2:
                st.info(
                    "São necessárias pelo menos duas fontes para calcular conexões."
                )
            else:
                st.info(
                    "Nenhuma conexão encontrada entre as fontes no limite de quatro saltos."
                )

    _render_claims(resultado.get("claims", []))

    with st.expander("Planejamento e consultas Cypher"):
        if planner_status:
            st.write(f"**Status do planejador:** {planner_status}")
        if planner_reason:
            st.caption(planner_reason)
        raw_generated = planner.get("generated_queries", [])
        generated_queries = raw_generated if isinstance(raw_generated, list) else []
        st.write("**Consulta planejada**")
        if generated_queries:
            for position, query in enumerate(generated_queries, start=1):
                _render_query(query, position, generated=True)
        else:
            st.info("Nenhuma consulta gerada pelo planejador foi registrada.")

        queries = resultado.get("queries", [])
        st.write("**Consultas executadas e registradas**")
        if queries:
            st.caption(
                "Parâmetros de diagnóstico: valores de embeddings são omitidos pelo backend."
            )
            for position, query in enumerate(queries, start=1):
                _render_query(query, position)
        else:
            st.info("Nenhum registro de consulta disponível para este resultado.")

    with st.expander("Nós e relações consultados"):
        st.write("**Nós**")
        st.dataframe(resultado.get("nos", []), width="stretch")
        st.write("**Relações**")
        st.dataframe(resultado.get("arestas", []), width="stretch")
