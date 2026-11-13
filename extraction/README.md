# Extração e carregamento no Neo4j

Execute a partir da raiz com as dependências de `requirements.txt` instaladas.
O pipeline usa os 26 artigos de `data/papers.json` e a curadoria versionada em
`data/curated_entities.json`; não refaz a coleta dos artigos.

## Configuração e conexão

Crie `.env` a partir de `.env.example`, caso ainda não exista:

```dotenv
NEO4J_URI=neo4j+s://SEU_ID.databases.neo4j.io
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=SUA_SENHA
NEO4J_DATABASE=neo4j
```

Use as credenciais e o nome do banco fornecidos pela instância. O identificador
da instância Aura não determina o nome do banco. Preserve outras configurações
do `.env` e não publique credenciais.

```bash
python extraction/test_connection.py
```

Esse teste não modifica o banco. A aplicação usa o protocolo do Neo4j; não exige
variáveis OAuth ou credenciais da API administrativa do Aura.

## Preparar a aplicação completa

```bash
python run_pipeline.py --provider curated --import-neo4j --index-embeddings
# Atalho equivalente:
python extraction/load_graph.py
```

O fluxo valida os dados, extrai entidades, aplica o schema com `IF NOT EXISTS`,
importa os artigos, aguarda os índices, confere o grafo, gera embeddings locais
e faz a validação final. Não é necessário aplicar o schema manualmente antes.
O schema existente não é apagado nem recriado para corrigir incompatibilidades;
elas são informadas pela conferência. Veja [schema](../schema/README.md) e
[retrieval](../retrieval/README.md) para detalhes.

Sem acesso ao banco, prepare os arquivos offline:

```bash
python run_pipeline.py --provider curated
```

Saídas em `output/`: `papers_normalized.json`, `validation_report.json`,
`papers_enriched.json`, `extraction_report.json` e, ao usar o banco,
`graph_validation_report.json`. Falhas encerram o comando com código diferente
de zero e interrompem as etapas seguintes.

## Validação e extração

Os metadados obrigatórios devem ter os tipos corretos e textos não vazios.
Autores são obrigatórios; palavras-chave vazias geram aviso. Anos devem ser
inteiros de 2023 a 2026, conforme o corpus da disciplina.

Os provedores disponíveis são `curated` (padrão), `heuristic` e `openai`.
Os dois últimos exigem revisão das entidades conforme o relatório; a extração
OpenAI usa `OPENAI_API_KEY` e `OPENAI_MODEL`, separadamente do gerador do retrieval.
As três listas `proposed_methods`, `used_methods` e `datasets` devem existir e
conter textos. Uma lista vazia válida significa ausência de menção explícita.
Falhas de extração não viram listas vazias: ficam no relatório, invalidam o
lote e impedem sobrescrever uma saída enriquecida anterior válida.

## Atualização do grafo

Cada Paper é atualizado em uma única transação com seus metadados e relações:

- `Author → AUTHORED → Paper`;
- `Paper → HAS_CONCEPT → Concept`;
- `Paper → PROPOSES/USES_METHOD → Method`;
- `Paper → EVALUATED_ON → Dataset`.

A reimportação sincroniza essas relações com a entrada, inclusive remoções e
trocas entre método proposto e utilizado. Preserva nós compartilhados, artigos
ausentes da entrada e relações fora desses padrões. Não apaga nós órfãos nem
faz limpeza global. Se um artigo falhar, sua transação é desfeita; artigos
anteriores podem já estar confirmados. Corrija a causa e execute novamente.

Também é possível executar etapas separadas:

```bash
python -m src.validate_data
python -m src.extract_entities --provider curated
python -m src.import_graph
python -m src.validate_graph
python -m retrieval.index_embeddings
python -m src.validate_graph --check-embeddings
```

O validador compara propriedades e relações com `output/papers_enriched.json`,
verifica constraints e índices e sinaliza falha para divergências ou banco
vazio. `--input` permite conferir outro arquivo enriquecido. Artigos extras no
banco são preservados. `--check-embeddings` também confere dimensão, valores,
modelo e hash de conteúdo. Consultas da interface não executam essas escritas.

## Testes

```bash
python -m unittest tests.test_pipeline tests.test_graph_import -v
```

O teste `tests.test_graph_integration` requer `TEST_NEO4J_URI`,
`TEST_NEO4J_USERNAME` e `TEST_NEO4J_PASSWORD` para um serviço isolado. Ele verifica
reimportação, remoção de relações obsoletas, preservação de relações alheias e
rollback. Nunca use as credenciais do banco compartilhado nesses testes.
