# Subgrafo interativo

Componente local Streamlit v2 + Cytoscape.js. Recebe somente o subgrafo da resposta,
com nós, arestas, caminhos e afirmações; não conecta ao Neo4j nem ao provedor LLM.

Funcionalidades: pan/zoom/drag, ajustar/restaurar visualização, selecionar nós e
arestas, consultar propriedades/proveniência e destacar caminhos `[Px]`. As
relações mantêm direção. Artigos conectores têm marcação própria. Fatos de um
salto são identificados separadamente dos caminhos mínimos entre fontes.

Os dados entram pela API JSON do componente. Textos são inseridos com
`textContent`, sem interpretar HTML dos títulos/abstracts. A distribuição da
biblioteca fica em `vendor/cytoscape.min.js`; não há CDN em tempo de execução.

Origem: [Cytoscape.js v3.35.0](https://github.com/cytoscape/cytoscape.js/tree/v3.35.0),
Copyright The Cytoscape Consortium, licença MIT em
[`vendor/CYTOSCAPE_LICENSE.txt`](vendor/CYTOSCAPE_LICENSE.txt).

## Verificação sem backend

```bash
PYTHONPATH=. python -m streamlit run tests/graph_fixture_app.py --server.address 127.0.0.1
python -m unittest tests.test_graph_component tests.test_app -v
```

A fixture é sintética e nunca chama retrieval, Neo4j ou LLM. Os testes Python
cobrem transformação dos dados, vínculo de caminhos e persistência do resultado.
A interação JavaScript também deve ser conferida em navegador.
