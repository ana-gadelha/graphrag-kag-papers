# Interface de consulta

```bash
python -m streamlit run app.py
```

A aplicação chama `retrieval.answer_question` ao clicar **Consultar** e mantém a
resposta em `st.session_state`. Explorar o subgrafo, selecionar elementos e
inspecionar evidências reaproveita esse resultado, sem nova inferência.

A tela mostra resposta, fontes `[Sx]`, fatos/caminhos `[Px]`, trechos por afirmação,
RRF, confiança heurística, estado do planejador e consultas Cypher. É possível
comparar o texto gerado com a forma validada/limitada efetivamente executada.
Consultas rejeitadas ficam identificadas como não executadas. Falhas do planejador
podem resultar em recuperação degradada, sempre sinalizada.

O [componente Cytoscape](../graph_component/README.md) permite pan, zoom, arraste,
seleção de nós/arestas e destaque dos caminhos citados. As tabelas de nós e relações
continuam disponíveis. Conectores não são promovidos a fontes de conteúdo.

Confiança indica convergência dos canais nas fontes citadas, não probabilidade de
acerto. Trechos e caminhos válidos confirmam rastreabilidade; avaliação semântica
permanece no [benchmark](../evaluation/README.md).

Configure o banco e o modelo conforme o [README principal](../README.md). Ao fazer
uma pergunta real, o provedor recebe a pergunta e as evidências selecionadas. A
fixture `tests/graph_fixture_app.py` permite testar a visualização com dados
sintéticos, sem serviços externos.
