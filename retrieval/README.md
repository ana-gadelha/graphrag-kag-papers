# Busca híbrida e respostas (GraphRAG)

Esta etapa implementa busca vetorial, consultas Cypher, combinação
dos resultados e envio do contexto para um LLM. A função `answer_question()`
devolve resposta com citações, fontes e subgrafo para a interface Streamlit.

## Preparação

Use Neo4j 5.23 ou superior (incluindo Aura/Neo4j 2026). Execute a partir da raiz
do repositório, com Python 3.10 ou superior:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

No PowerShell, ative o ambiente com `.\.venv\Scripts\Activate.ps1`.
Copie `.env.example` para `.env` e preencha os valores locais:

```dotenv
NEO4J_URI=neo4j+s://SEU_ID.databases.neo4j.io
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=SUA_SENHA
NEO4J_DATABASE=neo4j
OPENAI_API_KEY=SUA_CHAVE_OPENAI
OPENAI_MODEL=gpt-4.1-mini
```

Use o nome de usuário e de banco efetivamente fornecidos pela instância do
grupo. O `.env` da raiz é carregado sem substituir variáveis já definidas no
ambiente. Não versione credenciais.

### 1. Importar o corpus e aplicar o schema

Se o banco ainda não estiver preparado, execute o pipeline existente:

```bash
python extraction/load_graph.py
```

Veja os detalhes em [extraction](../extraction/README.md) e
[schema](../schema/README.md). A consulta não executa importação nem modifica o
schema automaticamente.

### 2. Gerar os embeddings

```bash
python -m retrieval.index_embeddings
```

O indexador lê título, abstract e nomes dos conceitos de cada `Paper` no Neo4j.
Usa `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`, com 384
dimensões e vetores normalizados, para permitir perguntas em português sobre
abstracts em inglês. O mesmo modelo codifica documentos e perguntas.

O primeiro uso baixa o modelo público do Hugging Face; os seguintes aproveitam
o cache local. A geração de embeddings não usa a API OpenAI. O modelo usa uma
janela limitada de tokens e pode truncar textos longos; o contexto de resposta
continua incluindo os abstracts armazenados.

O índice `paper_embedding` deve estar `ONLINE`, sobre `Paper.embedding`, com
384 dimensões e similaridade `cosine`. O indexador verifica essas condições e
informa ajustes necessários, sem apagar ou recriar o índice.

Além de `embedding`, são gravados `embedding_model` e `embedding_hash` nos
papers. Registros com o mesmo modelo, conteúdo e vetor válido são ignorados;
novos registros e conteúdos alterados são recalculados. Para substituir
explicitamente embeddings de outro modelo, mantendo a dimensão do índice:

```bash
python -m retrieval.index_embeddings --force
```

Execute novamente o indexador depois de alterar títulos, abstracts ou conceitos.
A consulta rejeita embeddings ausentes, inválidos, incompatíveis ou
desatualizados em qualquer paper do corpus, evitando misturar espaços vetoriais.

### 3. Fazer uma pergunta

```bash
python -m retrieval "Quais artigos utilizam GraphRAG e quais datasets foram utilizados nesses trabalhos?"
python -m retrieval "O que os papers dizem sobre KAG?" --top-k 4
```

A saída de sucesso é JSON em `stdout`; falhas são informadas em `stderr`, com
código de saída 1. Para salvar a saída:

```bash
python -m retrieval "Quais artigos propõem métodos multi-hop?" > /tmp/resposta.json
```

## Usar um modelo local: LM Studio, Ollama ou llama.cpp

O padrão continua sendo a OpenAI com a Responses API. Para usar outro servidor,
configure apenas as variáveis `RETRIEVAL_LLM_*`. Elas são exclusivas da geração
de respostas e não alteram a configuração de extração dos outros integrantes.
A assinatura e o retorno de `answer_question()` continuam iguais, assim como
a busca no Neo4j e os embeddings locais. Não é necessário reindexar os papers
ao trocar apenas o modelo gerador.

### Exemplo com LM Studio

Carregue um modelo com suporte a chat no LM Studio e inicie seu servidor na
seção **Developer**. Use o endereço e o identificador exibidos pelo servidor.
O endereço habitual é `http://localhost:1234/v1`; os modelos disponíveis podem
ser consultados com:

```bash
curl http://localhost:1234/v1/models
```

No `.env` da raiz:

```dotenv
RETRIEVAL_LLM_BASE_URL=http://localhost:1234/v1
RETRIEVAL_LLM_MODEL=IDENTIFICADOR_EXATO_DO_MODELO
RETRIEVAL_LLM_API=chat_completions
RETRIEVAL_LLM_TIMEOUT=120
```

Se o servidor exigir autenticação, acrescente `RETRIEVAL_LLM_API_KEY` com a chave
dele. Sem essa variável, um endereço próprio recebe a chave fictícia `local`,
para atender ao SDK; ela não autentica servidores que exigem uma chave real.
A chave configurada em `OPENAI_API_KEY` **não é enviada ao endereço alternativo**.

Execute a consulta normalmente:

```bash
python -m retrieval "Quais artigos utilizam GraphRAG e quais datasets foram utilizados?"
```

### Outros servidores compatíveis

| Servidor | Exemplo de `RETRIEVAL_LLM_BASE_URL` | `RETRIEVAL_LLM_MODEL` |
|---|---|---|
| LM Studio | `http://localhost:1234/v1` | Identificador do modelo carregado |
| Ollama | `http://localhost:11434/v1` | Nome e tag exatos do modelo instalado |
| `llama-server` do llama.cpp | `http://localhost:8080/v1` | Identificador/alias anunciado pelo servidor |

As portas são exemplos e devem coincidir com as do seu servidor. Llama é uma
família de modelos que pode ser servida por essas ferramentas. O programa não
instala, inicia ou baixa automaticamente o modelo gerador.

Se a aplicação Python rodar em outro computador ou contêiner, `localhost`
representará esse ambiente; use o endereço pelo qual ele alcança o servidor.

### Configuração e escolha do endpoint

| Variável | Comportamento |
|---|---|
| `RETRIEVAL_LLM_BASE_URL` | URL HTTP(S) com o prefixo da API, normalmente `/v1`; sem ela, usa a OpenAI oficial |
| `RETRIEVAL_LLM_MODEL` | Obrigatória quando há endereço próprio; sem ele, pode sobrescrever `OPENAI_MODEL` |
| `RETRIEVAL_LLM_API` | `chat_completions` ou `responses`; padrão: chat com endereço próprio, Responses com OpenAI |
| `RETRIEVAL_LLM_API_KEY` | Chave específica; sem ela, usa `local` com endereço próprio ou `OPENAI_API_KEY` no padrão OpenAI |
| `RETRIEVAL_LLM_TIMEOUT` | Timeout positivo em segundos; padrão: 120 com endereço próprio e 30 com OpenAI |

O timeout é aplicado pelo cliente HTTP às operações da requisição, incluindo
esperar a resposta; o SDK também permite até duas tentativas adicionais em
erros transitórios. Portanto, não representa um limite total de execução.

`chat_completions` usa `/chat/completions`, mensagens `system`/`user` e
`max_tokens=2000`. `responses` usa `/responses`, `instructions`,
`max_output_tokens=2000` e `store=False`. Para um servidor local que implemente
Responses, altere explicitamente:

```dotenv
RETRIEVAL_LLM_API=responses
```

A compatibilidade é por endpoint e pelos parâmetros aceitos. Não há troca
automática de protocolo ou retorno à nuvem em caso de falha. O cliente também
fixa o endereço explicitamente: `OPENAI_BASE_URL` não controla mais retrieval;
use `RETRIEVAL_LLM_BASE_URL` para isolar essa etapa da extração.

Se a resposta for truncada, bloqueada, vazia ou tiver citações inválidas, a
função sinaliza erro. Para reduzir o contexto enviado a um modelo com janela
menor, use, por exemplo, `--top-k 3`. Configure também uma janela adequada no
servidor; a aplicação não altera essa configuração por conta própria.

Para voltar à OpenAI, remova as variáveis `RETRIEVAL_LLM_*` e mantenha
`OPENAI_MODEL` e `OPENAI_API_KEY`. O indexador e a abstenção sem evidências
funcionam mesmo quando a configuração do modelo gerador está incompleta.

Documentação dos servidores: [LM Studio](https://lmstudio.ai/docs/developer/openai-compat),
[Ollama](https://docs.ollama.com/api/openai-compatibility) e
[llama.cpp](https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md).

## Como funciona

1. **Busca vetorial:** codifica a pergunta localmente e consulta os oito papers
   mais próximos no índice do Neo4j.
2. **Busca no grafo:** identifica nomes de autores, conceitos, métodos e datasets
   mencionados na pergunta, independentemente dos resultados vetoriais. Tolera
   diferenças de caixa e acentuação, além de aliases mínimos de GraphRAG/KAG;
   não interpreta filtros ou consultas em linguagem natural arbitrária.
3. **Expansão:** percorre até dois saltos, `Paper → entidade ← Paper`, também
   respeitando a direção inversa das relações com autores. Busca até cinco
   vizinhos por paper inicial e limita o ranking do grafo a vinte papers.
   Usa `AUTHORED`, `HAS_CONCEPT`, `PROPOSES`, `USES_METHOD` e `EVALUATED_ON`.
4. **Combinação:** aplica Reciprocal Rank Fusion, somando `1 / (60 + posição)`
   em cada canal. Na seleção final, prioriza papers ligados a mais entidades
   explicitamente citadas na pergunta; entre papers com a mesma quantidade de
   correspondências diretas, ordena por RRF e desempata pelo ID. Isso evita que
   vizinhos encontrados nos dois canais excluam os artigos de um autor citado.
   Remove duplicatas pelo ID. O campo `score` continua sendo a pontuação RRF,
   representa relevância relativa e pode não ser decrescente na ordem final.
5. **Evidências:** seleciona seis papers por padrão e recupera seus metadados,
   abstracts e relações. Esses mesmos dados formam o contexto e o subgrafo.
6. **Resposta:** envia pergunta e evidências ao LLM configurado, pedindo
   resposta em português com referências `[S1]`, `[S2]`, etc. Abstracts e pergunta
   são tratados como dados, sem autorização para executar suas instruções.

As consultas Cypher são fixas e parametrizadas. O LLM não gera Cypher nem recebe
credenciais do Neo4j. `answer_question()` realiza apenas leituras no banco;
somente o comando de indexação escreve os embeddings.

## Integração com a interface

```python
from retrieval import answer_question

result = answer_question(
    "Quais artigos utilizam GraphRAG e quais datasets foram utilizados?",
    top_k=6,
)

print(result["answer"])
for source in result["sources"]:
    print(source["citation_id"], source["title"], source["url"])

nodes = result["subgraph"]["nodes"]
edges = result["subgraph"]["edges"]
```

Assinatura pública:

```python
answer_question(question: str, *, top_k: int = 6) -> dict
```

A pergunta deve ser um texto não vazio de até 4.000 caracteres. `top_k` deve ser
um inteiro de 1 a 20. O retorno contém apenas valores serializáveis em JSON:

| Campo | Conteúdo |
|---|---|
| `question` | Pergunta recebida, sem espaços nas extremidades |
| `answer` | Texto em português com citações locais à resposta |
| `sources` | Evidências enviadas ao LLM, inclusive aquelas que ele não citou |
| `subgraph.nodes` | Nós com `id`, `label` e `properties` |
| `subgraph.edges` | Arestas com `source`, `target` e `type` |
| `warnings` | Limitações do material recuperado |

Cada fonte inclui `citation_id`, `paper_id`, `title`, `authors`, `year`, `url`,
`abstract`, `relations`, `retrieved_by`, `ranks` e `score`. Cada relação tem
`type`, `entity_label` e `entity_name`. Os IDs de citação são atribuídos depois
da ordenação, começando por `S1`; devem ser associados à resposta correspondente.

Os IDs do subgrafo usam o tipo e o identificador/nome com escape de URL, por
exemplo `paper:2404.16130` e `method:GraphRAG`. Autores, conceitos, métodos e
datasets com nomes iguais continuam sendo nós distintos por tipo. Arestas
preservam a direção do banco, inclusive `Author → Paper`. Não são retornados
embeddings, credenciais ou objetos do driver Neo4j.

Para tratar erros na interface:

```python
from retrieval.config import RetrievalError

try:
    result = answer_question("Quais datasets aparecem nos trabalhos sobre GraphRAG?")
except RetrievalError as exc:
    print(str(exc))
```

Sem evidências, a função retorna uma abstenção com fontes e subgrafo vazios,
sem chamar o LLM. Com evidências, a geração usa a configuração descrita acima;
o padrão OpenAI exige uma chave de API. Falhas de
conexão, indexação ou geração são sinalizadas como erro; não são transformadas
em respostas científicas.

## Validação e limites

```bash
python -m unittest discover -s tests -v
```

Os testes automatizados usam drivers e modelos simulados: cobrem validação de
entrada, indexação incremental, configuração do índice, busca independente pelo
grafo, expansão, fusão, fontes, subgrafo, citações e falhas dos serviços, sem
precisar de credenciais ou chamadas de rede. Os testes dos dois protocolos
também usam o SDK OpenAI real com transporte HTTP em memória: verificam endpoint,
modelo, autenticação, timeout, payload e leitura da resposta. A execução real
contra Neo4j e o servidor de geração deve ser conferida separadamente pelos
comandos acima. Esta adição não foi executada contra um modelo carregado no
LM Studio, Ollama ou llama.cpp.

Na validação real do Aura, o banco inicialmente vazio foi reconstruído com
302 nós e 335 relacionamentos. A comparação com os dados locais não encontrou
nós ou relações ausentes, diferenças de propriedades ou relações duplicadas.
As cinco constraints e os quatro índices explícitos foram conferidos, com
todos os índices `ONLINE`.

Foram gravados 26 embeddings MiniLM reais de 384 dimensões. A segunda execução
reaproveitou todos os vetores, sem reprocessamento. As consultas de GraphRAG,
HotpotQA e autor compartilhado passaram no banco real, incluindo conferência
das fontes e da direção/completude das relações do subgrafo. Esse teste revelou
e corrigiu a perda de artigos do autor citado durante a fusão dos rankings.
O conjunto automatizado passou com 44 testes.

A chave OpenAI autenticou e consultou o modelo, mas a tentativa de geração
real retornou HTTP 429, código `credit_balance_exhausted`. Portanto, a geração
completa de respostas por `answer_question()` e pelo CLI permanece pendente de
saldo no provedor ou de um servidor local disponível. Os testes de citações e
dos protocolos permanecem cobertos com transporte HTTP simulado.

- O MVP usa metadados e abstracts, sem processar os PDFs completos. A curadoria
  atual identifica datasets em apenas cinco dos 26 papers.
- Ausência de `EVALUATED_ON` significa **dataset não identificado nas evidências**,
  sem demonstrar que o trabalho não usou datasets.
- `USES_METHOD` inclui uso, comparação ou baseline; não comprova que o paper
  implementou o método. `PROPOSES` é uma relação distinta.
- Os resultados dependem do corpus e do limite de recuperação. Respostas não
  devem ser interpretadas como levantamento exaustivo de toda a literatura.
- A validação exige ao menos uma citação e rejeita IDs não fornecidos ao LLM.
  Isso confirma a referência ao contexto, mas não prova que cada afirmação seja
  sustentada por ele. Revise as respostas usadas na apresentação.
- O servidor configurado recebe a pergunta e as evidências selecionadas. No
  padrão OpenAI pode haver custos de API. A Responses API usa `store=False`;
  retenção e logs de servidores locais dependem da configuração deles.

Referências de implementação: [modelo multilíngue](https://huggingface.co/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2),
[índices vetoriais do Neo4j](https://neo4j.com/docs/cypher-manual/current/indexes/semantic-indexes/vector-indexes/)
e [geração de texto pela OpenAI](https://developers.openai.com/api/docs/guides/text).
