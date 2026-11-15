# GraphRAG e KAG: Recuperação Híbrida em Papers Científicos (2023–2026)

Projeto Final da disciplina **Tendências em Ciência da Computação** – UNIPÊ
Professor: **Ricardo Roberto de Lima** | Semestre: 2026 (Outubro – Dezembro)

> **Status:** implementação de extração automática, entidades canônicas, Cypher gerado com validação, respostas com evidências e grafo interativo. Avaliação empírica e revisão humana permanecem etapas explícitas; veja a [matriz de requisitos](docs/ADEQUACAO_REQUISITOS.md).

---

## Sobre o projeto

Este projeto transforma artigos científicos (2023–2026) sobre **GraphRAG**, **recuperação híbrida** e **Knowledge-Augmented Generation (KAG)** em um **grafo de conhecimento** no Neo4j.

Quando o usuário faz uma pergunta, o sistema:

1. Procura textos parecidos com a pergunta (**busca vetorial**);
2. Percorre o grafo para encontrar as conexões entre artigos, autores, métodos e datasets (**busca no grafo com Cypher, multi-hop**);
3. Envia esse contexto para um LLM, que monta uma resposta **citando de onde veio cada informação** (*provenance*).

O objetivo é **reduzir alucinações** e deixar as respostas **rastreáveis**.

---

## Equipe

| Integrante | Função principal |
|---|---|
| Ana Maria Lúcio de Sousa Dantas | Coordenação + Documentação + Artigo |
| Marlon Rufino do Nascimento | Coleta de Papers e Dados |
| Gilverson Rodrigues da Silva | Modelagem do Grafo (Neo4j) |
| Mirela Ferraz de A. G. de Souza | Extração de Informações (Text-to-Graph) |
| João Victor Barbosa Perazzo | Busca Híbrida + Respostas (GraphRAG) |
| Ramon Fhelipe Vieira Marques | Interface + Visualização do Grafo |

---

## Arquitetura (visão geral)

```
Papers  →  1. Extração (NER/RE)  →  2. Grafo no Neo4j
                                              ↓
          4. Interface (Streamlit)  ←  3. Busca híbrida + LLM
```

---

## Tecnologias

- **Python** – linguagem principal
- **Neo4j** (Desktop ou Aura Free) – banco de dados de grafos
- **Driver Neo4j + Cypher parametrizado** – recuperação fixa e Cypher gerado pelo LLM, validado por gramática Lark antes de executar
- **Sentence Transformers** – embeddings locais multilíngues de 384 dimensões
- **Streamlit + Cytoscape.js** – interface e exploração interativa do subgrafo
- **arXiv API + artigo SBC** – corpus versionado com 26 papers
- **OpenAI SDK** – geração via Responses ou Chat Completions; aceita servidores locais compatíveis

---

## Estrutura do repositório

```
├── data/          # Corpus JSON, títulos/abstracts e aliases
├── schema/        # Script Cypher do grafo (constraints e índices)
├── extraction/    # Pipeline Text-to-Graph (NER/RE)
├── retrieval/     # Busca híbrida e geração de respostas
├── app.py         # Interface Streamlit integrada ao retrieval
├── app/           # Documentação da interface
├── tests/         # Testes unitários, da interface e de integração
├── docs/          # Decisões, atas e matriz de requisitos
└── README.md
```

---

## Como instalar e rodar

Execute os comandos a partir da raiz. Use Python 3.10+ (validação local e CI com
3.13), Neo4j 5.26+ e um LLM disponível. A integração com o banco é testada no
Neo4j Community 5.26. O primeiro uso baixa o encoder público do Hugging Face.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Se ainda não tiver `.env`, copie `.env.example` e preencha `NEO4J_URI`,
`NEO4J_USERNAME`, `NEO4J_PASSWORD`, `NEO4J_DATABASE` e `RETRIEVAL_LLM_*`. O
exemplo usa endpoint OpenAI-compatible com modelo gratuito explícito. Veja a
[documentação de retrieval](retrieval/README.md). Não versione credenciais.

```bash
# Extração automática: envia títulos/abstracts ao provedor configurado.
# Execute somente após autorizar esse envio e conferir o modelo gratuito.
python run_pipeline.py --provider openai

# Banco existente: confira a prévia de migração antes de aplicar.
python -m src.migrate_entities

# Fluxo completo autorizado: migra/importa e indexa o banco configurado.
python run_pipeline.py --provider openai --import-neo4j --index-embeddings

# Reprodução offline da curadoria, sem inferência de extração:
# python run_pipeline.py --provider curated --import-neo4j --index-embeddings

# Abrir a aplicação; as perguntas fazem somente leituras no Neo4j.
python -m streamlit run app.py

# Alternativa pela linha de comando.
python -m retrieval "Quais trabalhos usam HotpotQA?"
```

A interface chama `answer_question()` diretamente no processo Python. Mostra
as afirmações com fontes `[S1]`, fatos/caminhos `[P1]`, pontuação RRF, confiança
heurística, avisos e status do planejador. O grafo permite zoom, deslocamento,
arraste, seleção de nós/arestas e destaque de caminhos sem nova inferência.

A extração automática é o padrão. Os modos `curated` e `heuristic` continuam
explícitos para comparação/reprodução; seu resultado não é apresentado como
extração por LLM. Relações automáticas precisam de evidência literal no texto.
Autores e palavras-chave mantêm proveniência de metadados. A identidade de autores
por nome é conservadora e não resolve homônimos sem fontes adicionais.

A migração faz prévia e bloqueia conflitos; alterações só acontecem com `--apply`
ou pelo pipeline explicitamente executado com `--import-neo4j`. Preserve um backup
do banco antes de uma migração real. Não há limpeza global do grafo. A importação
reconcilia cada Paper em uma transação, mantendo dados externos; falhas podem deixar
papers anteriores confirmados, por isso a execução é retomável/idempotente.

Extração, planejamento e resposta compartilham transporte com configuração por
etapa. `EXTRACTION_LLM_*` e `CYPHER_LLM_*` sobrepõem `RETRIEVAL_LLM_*` explicitamente
configurado. Outro endpoint exige sua própria chave. Não existe fallback para um
modelo pago. O limite de geração inclui raciocínio quando o provedor assim o
contabiliza. Truncamento, quota e falhas do planejador são diagnosticados; a
recuperação fixa pode continuar em modo degradado claramente indicado.

O cache em `output/llm_cache/` só reutiliza respostas aprovadas pelo validador da
etapa. Modelo, endpoint, prompt, parâmetros e conteúdo integram sua identidade.
O ledger conta tentativas malsucedidas: padrão de 40 chamadas/dia UTC e 10/minuto,
configurável no `.env`. São limites locais, não a quota oficial. O padrão é zero
retries automáticos; não são feitas chamadas de reparo ilimitadas.

Se o banco já estiver preparado, inicie apenas a interface. Quando título,
abstract ou conceitos mudarem, execute novamente o pipeline completo ou o
indexador; o retrieval rejeita embeddings desatualizados. Trocar somente o LLM
gerador não exige reindexação.

Detalhes: [extração](extraction/README.md), [schema](schema/README.md),
[retrieval](retrieval/README.md) e [interface](app/README.md).

## Testes

```bash
python -m unittest discover -s tests -v
python -m compileall -q src retrieval tests app.py app_integration.py run_pipeline.py extraction
git diff --check
```

Os testes unitários e da interface não acessam serviços externos. Os testes
optativos de banco exigem `TEST_NEO4J_URI`, `TEST_NEO4J_USERNAME`,
`TEST_NEO4J_PASSWORD` e, opcionalmente, `TEST_NEO4J_DATABASE`. Aponte essas
variáveis para um **Neo4j isolado de testes**; não existe fallback para `.env`.
Os testes criam e removem seus próprios registros nesse serviço.

A CI fornece esse Neo4j, executa testes e prepara os 26 papers com embeddings
reais. Depois roda `python tests/smoke_retrieval.py`, que confere reindexação sem
alterações, perguntas GraphRAG/HotpotQA/autor e direções do subgrafo. Para repetir
localmente, prepare o mesmo corpus no serviço isolado e configure `TEST_NEO4J_*`.
O smoke pressupõe esse corpus fixo; o validador geral aceita entradas de outros
tamanhos. Nenhuma chave LLM é necessária na CI.

A geração real é uma verificação separada com `python -m retrieval "..."` e pela
interface. Erros de saldo/quota ou indisponibilidade do provedor impedem confirmar
essa etapa mesmo quando os testes automatizados passam.

---

## Cronograma

| Semana | Foco | Entregável |
|---|---|---|
| 1 | Proposta + schema do grafo + coleta inicial | Schema documentado + 26 papers |
| 2 | Extração de entidades/relações + ingestão no Neo4j | Grafo populado com dados reais |
| 3 | Busca híbrida + multi-hop + geração de respostas | Sistema respondendo com provenance |
| 4 | Interface final + artigo + apresentação | Código + artigo + demo pronta |

---

## Avaliação e entrega

Veja [evaluation/README.md](evaluation/README.md) para preparar anotações dos 26
abstracts, revisar seis perguntas de desenvolvimento e dezesseis de avaliação,
comparar baseline vetorial e híbrido, retomar execuções e gerar o relatório.
Preparar/conferir/relatar é offline. Executar inferência é uma etapa explícita.
A referência e as afirmações precisam da revisão de dois integrantes; dados
preliminares e resultados incompletos são identificados como inconclusivos.

A [matriz de conformidade](docs/ADEQUACAO_REQUISITOS.md) liga cada requisito às
funcionalidades e aos testes. Artigo SBC/IEEE e defesa ficam com a equipe. O
repositório pode permanecer privado, desde que o docente tenha acesso; isso deve
ser conferido pela equipe antes da entrega.

## Créditos

- [Neo4j](https://neo4j.com/) e [driver Python](https://github.com/neo4j/neo4j-python-driver): LPG, Cypher e acesso ao banco.
- [Sentence Transformers](https://www.sbert.net/) e [paraphrase-multilingual-MiniLM-L12-v2](https://huggingface.co/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2): embeddings locais.
- [Streamlit](https://streamlit.io/): interface e componentes.
- [Cytoscape.js](https://js.cytoscape.org/): visualização interativa; distribuição local e licença MIT incluídas no componente.
- [Lark](https://github.com/lark-parser/lark): gramática e análise do subconjunto de Cypher.
- [OpenAI Python SDK](https://github.com/openai/openai-python): transporte para APIs compatíveis; o uso do SDK não implica uso de modelo OpenAI.
- [python-dotenv](https://github.com/theskumar/python-dotenv), [arxiv.py](https://github.com/lukasschwab/arxiv.py) e [Graphviz Python](https://github.com/xflr6/graphviz): configuração, coleta e utilitário de visualização legado.
- Corpus e fontes: [data/README.md](data/README.md). Não redistribuir textos integrais ou atribuir resultados ausentes nos abstracts.
