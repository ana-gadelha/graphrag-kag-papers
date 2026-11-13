"""Interface Streamlit para consulta aos artigos de GraphRAG/KAG."""

from __future__ import annotations

import streamlit as st

from app_integration import consultar, montar_dot, url_http_valida
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
    if not pergunta.strip():
        st.warning("Digite uma pergunta antes de consultar.")
    else:
        with st.spinner("Consultando o grafo e gerando a resposta..."):
            try:
                st.session_state["resultado"] = consultar(pergunta)
            except RetrievalError as error:
                st.error(f"Não foi possível consultar: {error}")
            except Exception:
                st.error("Não foi possível consultar agora. Tente novamente.")

resultado = st.session_state.get("resultado")

if resultado:
    st.subheader("Resposta")
    st.markdown(resultado["resposta"])

    for warning in resultado["warnings"]:
        st.warning(warning)

    col_fontes, col_grafo = st.columns(2)

    with col_fontes:
        st.subheader("Fontes recuperadas")
        st.caption(
            "As fontes fornecem contexto; nem todas são necessariamente citadas na resposta."
        )
        if resultado["fontes"]:
            for fonte in resultado["fontes"]:
                citation_id = fonte["citation_id"]
                title = fonte["titulo"]
                url = fonte.get("url")
                if url_http_valida(url):
                    st.link_button(f"[{citation_id}] {title}", url)
                else:
                    st.write(f"[{citation_id}] {title}")
                metadata = f"{fonte['autores']} · {fonte['ano'] or 'Ano não informado'}"
                st.caption(metadata)
        else:
            st.info("Nenhuma fonte foi recuperada para esta pergunta.")

    with col_grafo:
        st.subheader("Subgrafo recuperado")
        st.graphviz_chart(montar_dot(resultado["nos"], resultado["arestas"]))

    with st.expander("Nós e relações consultados"):
        st.write("**Nós**")
        st.dataframe(resultado["nos"], width="stretch")
        st.write("**Relações**")
        st.dataframe(resultado["arestas"], width="stretch")
