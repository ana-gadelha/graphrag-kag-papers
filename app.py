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

    confidence = resultado["confidence"]
    level = {"low": "Baixa", "medium": "Média", "high": "Alta"}.get(confidence.get("level"))
    st.subheader("Confiança heurística nas evidências")
    if level:
        st.write(f"**{level}** — {confidence['reason']}")
    else:
        st.info("Nível de confiança não disponível para este resultado.")
    st.caption(
        "Indica convergência dos canais nas fontes citadas; não representa "
        "probabilidade de acerto nem avaliação da veracidade da resposta."
    )

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

    with col_grafo:
        st.subheader("Subgrafo recuperado")
        st.graphviz_chart(montar_dot(resultado["nos"], resultado["arestas"]))

    st.subheader("Conexões entre as fontes")
    st.caption(
        "Caminhos mínimos de até quatro saltos entre as primeiras seis fontes. "
        "As setas abaixo indicam a ordem do percurso; a direção das relações "
        "é preservada no grafo. Artigos conectores não são fontes adicionais."
    )
    for path in resultado["paths"]:
        st.write(f"[{path['source']}] ↔ [{path['target']}] · {path['length']} saltos")
        sequence = []
        for node in path["nodes"]:
            properties = node["properties"]
            name = properties.get("title") or properties.get("name") or node["id"]
            sequence.append(f"{name} (conector)" if node.get("connector") else name)
        st.text(" → ".join(sequence))
    if not resultado["paths"]:
        if len(resultado["fontes"]) < 2:
            st.info("São necessárias pelo menos duas fontes para calcular conexões.")
        else:
            st.info("Nenhuma conexão encontrada entre as fontes no limite de quatro saltos.")

    with st.expander("Consultas Cypher executadas"):
        st.caption("Parâmetros de diagnóstico: os valores do vetor da pergunta são omitidos.")
        for position, query in enumerate(resultado["queries"], start=1):
            st.write(f"**{position}. {query['stage']}**")
            st.code(query["cypher"], language="cypher")
            st.json(query["parameters"])
        if not resultado["queries"]:
            st.info("Nenhum registro de consulta disponível para este resultado.")

    with st.expander("Nós e relações consultados"):
        st.write("**Nós**")
        st.dataframe(resultado["nos"], width="stretch")
        st.write("**Relações**")
        st.dataframe(resultado["arestas"], width="stretch")
