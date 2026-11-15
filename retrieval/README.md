# Recuperação híbrida e respostas com evidências

## Configuração

Use Python 3.10+ (CI 3.13), Neo4j 5.26+ e as dependências da raiz. O `.env` é
carregado sem substituir variáveis exportadas. Veja [`.env.example`](../.env.example).

`RETRIEVAL_LLM_*` controla a resposta. `EXTRACTION_LLM_*` e `CYPHER_LLM_*` permitem
sobrescrever modelo, endpoint, chave, protocolo, timeout, limite de saída e retries
por etapa. Após editar o `.env` com a aplicação aberta, reinicie o processo
Streamlit; variáveis já exportadas continuam tendo precedência.
Extração/planejamento podem herdar o modelo de retrieval explicitamente
configurado; não usam o modelo OpenAI implícito. Um novo endpoint não herda uma
chave de outro. Endpoint alternativo sem chave recebe `local`, útil para servidores
locais, mas insuficiente para OpenRouter/Groq ou outro serviço autenticado.

Protocolos aceitos: `chat_completions` e `responses`. Não há fallback automático
entre provedores/modelos. A compatibilidade do SDK não garante disponibilidade,
quota, suporte a parâmetros ou aderência do modelo ao contrato JSON.

O limite `*_MAX_OUTPUT_TOKENS` deve acomodar JSON e raciocínio quando o provedor
contabiliza ambos. Truncamento é erro, não resposta parcial aprovada. O padrão
legado de resposta é 2000; o exemplo gratuito configura 8192 explicitamente.

## Uso

Prepare o grafo conforme o README da raiz e execute:

```bash
python -m retrieval "Quais trabalhos usam HotpotQA?"
python -m retrieval "Quais trabalhos usam HotpotQA?" --mode vector --top-k 6
```

Uma pergunta real envia contexto ao provedor configurado. Os testes usam transporte
em memória e banco isolado; não fazem inferência. Sucesso do CLI produz JSON em
stdout; falha produz mensagem sanitizada em stderr e código de saída 1.

```python
from retrieval import answer_question
result = answer_question("Quais trabalhos usam HotpotQA?", top_k=6, mode="hybrid")
```

Perguntas: texto não vazio até 4000 caracteres. `top_k`: inteiro de 1 a 20.
`mode`: `hybrid` (padrão) ou `vector` (baseline de título/abstract).

## Recuperação e Cypher gerado

1. Confere índice e embeddings; codifica a pergunta localmente com
   `paraphrase-multilingual-MiniLM-L12-v2`, 384 dimensões/cosseno.
2. Recupera oito candidatos vetoriais; no híbrido também encontra entidades por
   nome/alias e expande Paper–entidade–Paper.
3. O planejador recebe pergunta, schema, vocabulário e IDs. Gera Cypher real e
   parâmetros JSON; não recebe credenciais do Neo4j.
4. Lark analisa uma gramática fechada. O validador confere tipos, direções,
   propriedades, parâmetros e corpus. Escrita, procedimentos, comentários,
   consultas múltiplas, expressões e padrões fora do subconjunto são rejeitados.
5. O AST é compilado para Cypher com limites antes das expansões: oito raízes,
   vinte vizinhos por etapa, quarenta candidatos e timeout transacional de cinco
   segundos. Mantém filtros e unicidade de relações do MATCH original. Resultados
   podem ser parciais; limites são avisados.
6. Os IDs válidos são incorporados ao ranking do grafo, combinado ao vetor por
   RRF. `graph_origins` distingue `fixed_graph` e `llm_cypher` sem tratá-los como
   evidências independentes. Candidatos do Cypher validado têm prioridade de
   seleção; em seguida vêm correspondências diretas e o desempate pelo RRF,
   evitando que vizinhos genéricos eliminem os artigos filtrados pela pergunta.
7. Materializa fatos e caminhos mínimos até quatro saltos entre as primeiras
   seis fontes. Artigos intermediários recebem `connector`, sem virar fontes.

O subconjunto aceita até três padrões conectados, filtros de ID/ano e entidades,
interseções e caminhos explícitos de até quatro saltos. Não é um console Cypher
geral. O modo de leitura do driver é roteamento; a proteção obrigatória está no
validador e nos limites. Use credenciais de leitura quando disponíveis.

`retrieval.hybrid.retrieve(..., use_planner=False)` permanece uma operação sem LLM
por padrão, apropriada ao smoke. `answer_question` ativa o planejador no híbrido.

Falhas do planejador preservam status (`invalid`, `quota`, `timeout`, erro de
configuração etc.) e modo degradado com recuperação fixa. `empty` significa
consulta válida sem candidatos, não prova de ausência no corpus.

## Contrato de resposta

Os campos anteriores são preservados, com extensões:

| Campo | Conteúdo |
|---|---|
| `question`, `mode` | Entrada normalizada e modo usado |
| `answer`, `abstained` | Texto renderizado e abstenção explícita |
| `claims` | Afirmações aprovadas, fontes, caminhos, trechos e offsets |
| `sources` | IDs `[Sx]`, títulos, abstracts, metadados e ranking |
| `subgraph` | Nós por ID canônico, arestas dirigidas com ID e proveniência |
| `paths` | `path_id` `[Px]`, tipo `shortest` ou `fact`, extremos, nós e arestas |
| `planner` | Status, candidatos, diagnóstico e recibo da geração de Cypher |
| `queries` | Consultas fixas e diagnósticos do Cypher original/canônico/executado |
| `confidence` | Heurística de convergência dos canais nas fontes citadas |
| `generation` | Recibo: modelo, cache, uso de tokens quando fornecido e duração |
| `warnings` | Limitações do corpus e da operação |

O modelo deve devolver `abstained` e `claims`. Cada claim contém `kind`, `text`,
`source_ids`, `path_ids` e `quotes`. Afirmações científicas (`document`) exigem
um trecho literal de título/abstract para cada fonte citada. Conexões (`graph`)
exigem caminhos válidos; seu texto é renderizado diretamente das arestas, sem
publicar conclusões livres do modelo sobre equivalência de resultados.

Caminhos `fact` representam relações de um salto e podem ter a mesma fonte nos
extremos. `shortest` conecta fontes distintas. Nós e arestas mantêm a direção
original, inclusive Author → Paper. IDs de entidade usam `entity_id`; leituras
legadas usam nome como fallback até a migração. IDs `[Sx]` e `[Px]` pertencem a
uma resposta específica. IDs internos do banco não são expostos.

O validador rejeita IDs ausentes, trechos inventados e caminhos descontínuos. Um
texto que cita um trecho real ainda pode interpretá-lo incorretamente; a revisão
humana no [benchmark](../evaluation/README.md) mede essa dimensão. Sem evidência,
a abstenção não exige LLM. A confiança não é probabilidade de acerto: continua
considerando apenas fontes citadas e convergência vetor/grafo, com canais que
compartilham sementes. RRF é relevância por fonte, não confiança factual.

## Cache, orçamento e reprodução

`retrieval.llm` compartilha transporte e tratamento de erros entre as etapas.
Somente JSON semanticamente aprovado pela aplicação entra no cache reutilizável.
Respostas inválidas ficam como diagnóstico. O hash considera modelo, endpoint,
credencial hasheada, conteúdo, prompt, versão e parâmetros. As chaves nunca são
persistidas em texto. Cache e ledger são privados/ignorados pelo Git.

`LLM_DAILY_REQUEST_BUDGET` (40) e `LLM_REQUESTS_PER_MINUTE` (10) limitam tentativas
locais, inclusive falhas. O dia segue UTC. Não representam quota oficial ou outras
aplicações na conta. `*_MAX_RETRIES` vale zero por padrão e aceita até três retries
explícitos; cada tentativa é contabilizada. Falhas de quota não são repetidas
automaticamente. Ao atingir limite, retome mais tarde; resultados válidos em cache
não gastam novas chamadas. `LLM_CACHE_ENABLED=0` desliga cache/ledger para testes.

Trocar gerador não exige reindexar. Alterar título, abstract ou conceitos exige
reindexação; o retrieval rejeita embeddings incompatíveis/desatualizados. O
indexador reaproveita vetores válidos e usa `--force` para troca explícita de modelo.

## Verificação

```bash
python -m unittest discover -s tests -v
```

`TEST_NEO4J_*` habilita integração optativa em instância isolada. A suíte cobre o
SDK real com HTTP em memória, geração simulada de Cypher executada no Neo4j,
validação adversarial, citações, cache e retomada. Inferência e avaliação humana
são verificações separadas; testes passando não demonstram redução de alucinações.
