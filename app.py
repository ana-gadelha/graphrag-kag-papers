import streamlit as st

st.set_page_config(page_title="GraphRAG/KAG - Consulta a artigos", layout="wide")


# ═══════════════════════════════════════════════════════════════
# BACKEND (SIMULADO)
# Quando o grupo terminar Neo4j + busca híbrida + LLM, só esta
# função muda. O resto da interface continua igual.
#
# Contrato: consultar(pergunta) devolve um dict com
#   resposta : str
#   fontes   : lista de {titulo, autores, ano}
#   nos      : lista de {id, label, tipo}
#   arestas  : lista de {origem, destino, relacao}
# ═══════════════════════════════════════════════════════════════
def consultar(pergunta: str) -> dict:
    return {
        "resposta": (
            "[RESPOSTA DE EXEMPLO] Os artigos A e B utilizam GraphRAG. "
            "O artigo A usa o dataset X e o artigo B usa o dataset Y."
        ),
        "fontes": [
            {"titulo": "Artigo de exemplo A", "autores": "Autor 1, Autor 2", "ano": 2024},
            {"titulo": "Artigo de exemplo B", "autores": "Autor 3", "ano": 2025},
        ],
        "nos": [
            {"id": "m1", "label": "GraphRAG", "tipo": "Metodo"},
            {"id": "p1", "label": "Artigo A", "tipo": "Paper"},
            {"id": "p2", "label": "Artigo B", "tipo": "Paper"},
            {"id": "d1", "label": "Dataset X", "tipo": "Dataset"},
            {"id": "d2", "label": "Dataset Y", "tipo": "Dataset"},
        ],
        "arestas": [
            {"origem": "p1", "destino": "m1", "relacao": "USA"},
            {"origem": "p2", "destino": "m1", "relacao": "USA"},
            {"origem": "p1", "destino": "d1", "relacao": "UTILIZA_DATASET"},
            {"origem": "p2", "destino": "d2", "relacao": "UTILIZA_DATASET"},
        ],
    }


# ═══════════════════════════════════════════════════════════════
# SUBGRAFO: converte nós e arestas em texto DOT (Graphviz)
# ═══════════════════════════════════════════════════════════════
CORES = {"Paper": "#cfe8ff", "Dataset": "#d5f5d5", "Metodo": "#ffe8b3"}


def montar_dot(nos: list, arestas: list) -> str:
    def limpa(texto) -> str:
        return str(texto).replace('"', "'")

    linhas = [
        "digraph G {",
        "rankdir=LR;",
        'node [shape=box, style="rounded,filled", fontname="Helvetica"];',
    ]
    for n in nos:
        cor = CORES.get(n["tipo"], "#eeeeee")
        linhas.append(f'"{limpa(n["id"])}" [label="{limpa(n["label"])}", fillcolor="{cor}"];')
    for a in arestas:
        linhas.append(
            f'"{limpa(a["origem"])}" -> "{limpa(a["destino"])}" [label="{limpa(a["relacao"])}"];'
        )
    linhas.append("}")
    return "\n".join(linhas)


# ═══════════════════════════════════════════════════════════════
# INTERFACE
# ═══════════════════════════════════════════════════════════════
st.title("Consulta a artigos de GraphRAG / KAG")
st.caption("Busca vetorial + grafo (Neo4j) + LLM")

# 1) Campo de pergunta
pergunta = st.text_input(
    "Pergunta",
    placeholder="Ex.: Quais artigos utilizam GraphRAG e quais datasets foram utilizados?",
)

# 2) Botão consultar
if st.button("Consultar", type="primary"):
    if not pergunta.strip():
        st.warning("Digite uma pergunta antes de consultar.")
    else:
        with st.spinner("Consultando o grafo e gerando a resposta..."):
            try:
                # session_state guarda o resultado entre as re-execuções do script
                st.session_state["resultado"] = consultar(pergunta)
            except Exception as erro:
                st.session_state.pop("resultado", None)
                st.error(f"Não foi possível consultar: {erro}")

resultado = st.session_state.get("resultado")

if resultado:
    # 3) Resposta
    st.subheader("Resposta")
    st.write(resultado["resposta"])

    col_fontes, col_grafo = st.columns(2)

    # 4) Fontes utilizadas
    with col_fontes:
        st.subheader("Fontes utilizadas")
        for i, fonte in enumerate(resultado["fontes"], start=1):
            st.markdown(f"**[{i}] {fonte['titulo']}**  \n{fonte['autores']} ({fonte['ano']})")

    # 5) Subgrafo utilizado
    with col_grafo:
        st.subheader("Subgrafo utilizado")
        st.graphviz_chart(montar_dot(resultado["nos"], resultado["arestas"]))

    # Nós consultados (item 3 do que mostrar na defesa)
    with st.expander("Nós e relações consultados"):
        st.write("**Nós**")
        st.dataframe(resultado["nos"], width="stretch")
        st.write("**Relações**")
        st.dataframe(resultado["arestas"], width="stretch")
