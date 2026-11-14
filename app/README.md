# Interface de consulta

A interface Streamlit consulta os artigos pelo contrato público
`retrieval.answer_question(question, *, top_k=6)`. O módulo
`app_integration.py` adapta a resposta para os campos usados pela interface e
converte o subgrafo em DOT com a biblioteca Graphviz.

## Executar

Na raiz do repositório, com as dependências instaladas e as configurações de
retrieval disponíveis no ambiente:

```bash
python -m streamlit run app.py
```

A consulta ao banco e ao gerador só acontece ao pressionar **Consultar**. Uma
pergunta vazia não chama o backend. Os resultados ficam no estado da sessão
durante atualizações normais da página e são removidos quando uma nova consulta
falha.

## O que a tela mostra

- A resposta do backend, incluindo citações como `[S1]`.
- As fontes recuperadas, seus identificadores de citação, título, autoria, ano
  e link HTTP/HTTPS quando disponível. As fontes fornecem contexto e podem não
  ser todas citadas na resposta.
- O subgrafo direcionado, com cores para `Paper`, `Author`, `Concept`, `Method`
  e `Dataset`.
- Avisos sobre limites das evidências recuperadas.
- O nível de **confiança heurística nas evidências**, sua justificativa e a
  indicação de que ele não representa probabilidade de acerto. Usa apenas as
  fontes distintas citadas: alta com duas ou mais, todas nos dois canais; média
  com uma nos dois canais ou duas ou mais citadas; baixa nos demais casos e na
  abstenção. As regras completas estão no [retrieval](../retrieval/README.md#confiança-heurística-nas-evidências).
- A relevância relativa RRF de cada fonte e seus canais de recuperação.
- **Conexões entre as fontes**, calculadas com Shortest Path entre as primeiras
  seis fontes, em até quatro saltos. Artigos intermediários aparecem como
  conectores, sem se tornarem fontes adicionais. As setas na sequência indicam
  a ordem do percurso; o desenho do grafo preserva a direção das relações.
- O expansor **Consultas Cypher executadas**, incluindo verificação do índice,
  parâmetros de diagnóstico e consulta dos caminhos quando executada. O vetor
  da pergunta aparece apenas com dimensão e indicação de valores omitidos.

Com menos de duas fontes, a tela informa que não há fontes suficientes para
calcular conexões. Caso contrário, a ausência de caminhos é informada com o
limite de quatro saltos. A abstenção mantém o diagnóstico disponível.

Erros de recuperação conhecidos são apresentados com a mensagem segura do
backend. Erros inesperados exibem uma mensagem genérica para não revelar dados
internos.

## Testes da interface

Os testes usam `streamlit.testing.v1.AppTest` e substituem `answer_question`
por dados locais; não chamam Neo4j nem um provedor de linguagem:

```bash
.venv/bin/python -m unittest tests.test_app -v
```
