# GraphRAG e KAG: Recuperação Híbrida em Papers Científicos (2023–2026)

Projeto Final da disciplina **Tendências em Ciência da Computação** – UNIPÊ
Professor: **Ricardo Roberto de Lima** | Semestre: 2026 (Outubro – Dezembro)

> **Status:** corpus, pipeline, retrieval e interface implementados. A geração depende de um provedor LLM disponível e com quota.

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
- **Driver Neo4j + Cypher parametrizado** – recuperação e expansão do grafo
- **Sentence Transformers** – embeddings locais multilíngues de 384 dimensões
- **Streamlit** – interface web
- **arXiv API + artigo SBC** – corpus versionado com 26 papers
- **OpenAI SDK** – geração via Responses ou Chat Completions; aceita servidores locais compatíveis

---

## Estrutura do repositório

```
├── data/          # Papers coletados (JSON, PDFs, abstracts)
├── schema/        # Script Cypher do grafo (constraints e índices)
├── extraction/    # Pipeline Text-to-Graph (NER/RE)
├── retrieval/     # Busca híbrida e geração de respostas
├── app.py         # Interface Streamlit integrada ao retrieval
├── app/           # Documentação da interface
├── tests/         # Testes unitários, da interface e de integração
├── docs/          # Decisões do grupo, atas e artigo técnico
└── README.md
```

---

## Como instalar e rodar

Execute os comandos a partir da raiz. Use Python 3.10+ (validação local e CI com
3.13), Neo4j 5.23+ e um LLM disponível. A integração com o banco é testada no
Neo4j Community 5.26. O primeiro uso baixa o encoder público do Hugging Face.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Se ainda não tiver `.env`, copie `.env.example` e preencha `NEO4J_URI`,
`NEO4J_USERNAME`, `NEO4J_PASSWORD`, `NEO4J_DATABASE`, `OPENAI_API_KEY` e
`OPENAI_MODEL`. Para modelos locais, veja as variáveis `RETRIEVAL_LLM_*` na
[documentação de retrieval](retrieval/README.md). Não versione credenciais.

```bash
# Preparação completa; escreve apenas no Neo4j configurado.
python run_pipeline.py --provider curated --import-neo4j --index-embeddings

# Abrir a aplicação; as perguntas fazem somente leituras no Neo4j.
python -m streamlit run app.py

# Alternativa pela linha de comando.
python -m retrieval "Quais trabalhos usam HotpotQA?"
```

A interface chama `answer_question()` diretamente no processo Python. Mostra a
resposta, citações `[S1]`, fontes com links, avisos e o subgrafo recuperado. Não há
servidor HTTP separado. Consultas sem evidência suficiente retornam abstenção;
problemas de conexão, configuração ou geração são apresentados como erros.

O pipeline usa o corpus versionado e a curadoria existente. Sem `--import-neo4j`,
executa apenas validação e extração offline. `--index-embeddings` requer importação.
Falhas interrompem as etapas seguintes. A importação atualiza as relações de cada
Paper em uma transação, preservando outros artigos, entidades compartilhadas e
relações fora do modelo. Uma falha operacional pode deixar artigos anteriores já
confirmados; corrija o problema e execute novamente. Não há limpeza global do banco.

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
| 1 | Proposta + schema do grafo + coleta inicial | Schema documentado + ~30 papers |
| 2 | Extração de entidades/relações + ingestão no Neo4j | Grafo populado com dados reais |
| 3 | Busca híbrida + multi-hop + geração de respostas | Sistema respondendo com provenance |
| 4 | Interface final + artigo + apresentação | Código + artigo + demo pronta |

---

## Créditos

As bibliotecas de código aberto utilizadas serão listadas aqui, com os devidos créditos, conforme exigido pelas regras do projeto.
