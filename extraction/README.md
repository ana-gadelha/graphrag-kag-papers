# Extração e carregamento no Neo4j Aura

Esta pasta contém os pontos de entrada da etapa Text-to-Graph. O comando de
carregamento valida e enriquece o corpus automaticamente antes de acessar o
Neo4j.

## 1. Preparar o ambiente

No PowerShell, a partir da raiz do projeto:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## 2. Criar o `.env`

Copie `.env.example`, renomeie a cópia para `.env` e preencha com as
credenciais baixadas no momento da criação da instância Aura:

```env
NEO4J_URI=neo4j+s://SEU_ID.databases.neo4j.io
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=SUA_SENHA
NEO4J_DATABASE=NOME_EXATO_DO_ARQUIVO_DE_CREDENCIAIS
```

No banco criado para o grupo, confira se o arquivo de credenciais informa
`20fda131` como nome do banco. O `.env` está bloqueado no `.gitignore`. Nunca
envie esse arquivo ao GitHub.

## 3. Testar sem modificar o banco

```powershell
python extraction/test_connection.py
```

## 4. Aplicar o schema manualmente no Aura

Abra **Query**, conecte-se à instância e execute todo o conteúdo de
`schema/schema.cypher`. Depois confira:

```cypher
SHOW CONSTRAINTS;
SHOW INDEXES;
```

Devem existir 5 constraints e os índices `paper_year_range`,
`paper_fulltext`, `entity_name_fulltext` e `paper_embedding`, além dos índices
criados automaticamente pelas constraints.

## 5. Carregar os dados

```powershell
python extraction/load_graph.py
```

O script executa todo o fluxo: valida `data/papers.json`, gera os arquivos em
`output/`, reaplica o schema com segurança (`IF NOT EXISTS`), importa os 26
papers com `MERGE` e mostra o relatório final de nós, relações e lacunas.

## 6. Conferir no Aura Query

```cypher
MATCH (n)
UNWIND labels(n) AS tipo
RETURN tipo, count(*) AS quantidade
ORDER BY tipo;
```

```cypher
MATCH ()-[r]->()
RETURN type(r) AS relacionamento, count(*) AS quantidade
ORDER BY relacionamento;
```

```cypher
MATCH (a:Author)-[:AUTHORED]->(p:Paper)
OPTIONAL MATCH (p)-[:HAS_CONCEPT]->(c:Concept)
OPTIONAL MATCH (p)-[:PROPOSES|USES_METHOD]->(m:Method)
OPTIONAL MATCH (p)-[:EVALUATED_ON]->(d:Dataset)
RETURN a, p, c, m, d
LIMIT 100;
```

## Segurança e compartilhamento

- compartilhe URI e credenciais somente em canal privado;
- não coloque senha em código, print, README, commit ou captura de tela;
- a senha divulgada anteriormente deve ser trocada;
- mantenha uma cópia segura do arquivo de credenciais do Aura;
- para interromper o acesso de alguém, altere a senha ou crie usuários
  separados, conforme as permissões disponíveis no projeto.

## Pipeline Text-to-Graph

### Preparação

Na raiz do projeto:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
