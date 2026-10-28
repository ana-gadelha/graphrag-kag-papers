# GraphRAG e KAG: Recuperação Híbrida em Papers Científicos (2023–2026)

Projeto Final da disciplina **Tendências em Ciência da Computação** – UNIPÊ
Professor: **Ricardo Roberto de Lima** | Semestre: 2026 (Outubro – Dezembro)

>  **Status:** em desenvolvimento – Fase 1 (Proposta e Modelagem)

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
- **LangChain / LlamaIndex / neo4j-graphrag** – integração LLM + grafo
- **Streamlit** – interface web
- **arXiv API / Semantic Scholar API** – coleta dos papers
- **LLM** (OpenAI, Groq ou outro) – extração e geração de respostas

---

## Estrutura do repositório (planejada)

```
├── data/          # Papers coletados (JSON, PDFs, abstracts)
├── schema/        # Script Cypher do grafo (constraints e índices)
├── extraction/    # Pipeline Text-to-Graph (NER/RE)
├── retrieval/     # Busca híbrida e geração de respostas
├── app/           # Interface Streamlit
├── docs/          # Decisões do grupo, atas e artigo técnico
└── README.md
```

---

## Como instalar e rodar

> Esta seção será preenchida à medida que o projeto avançar.

**Pré-requisitos (previstos):** Python 3.10+, Neo4j, chave de API de um LLM.

```bash
# 1. Clonar o repositório
git clone <URL-DO-REPOSITÓRIO>

# 2. Instalar as dependências
pip install -r requirements.txt

# 3. Configurar as variáveis de ambiente (.env)
# NEO4J_URI=...
# NEO4J_USER=...
# NEO4J_PASSWORD=...
# LLM_API_KEY=...

# 4. Rodar a interface
streamlit run app/main.py
```

> **Nunca** coloque senhas ou chaves de API diretamente no código ou no repositório. Use o arquivo `.env` (que fica fora do Git).

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
