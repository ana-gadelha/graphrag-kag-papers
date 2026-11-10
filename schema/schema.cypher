// =====================================================================
// Schema do grafo – GraphRAG e KAG em Papers Científicos (2023–2026)
// Banco: Neo4j 5 (AuraDB Free ou Neo4j Desktop)
//
// Como usar: copie e execute este arquivo inteiro na ferramenta de
// consulta do Neo4j (Query / Neo4j Browser). Pode rodar mais de uma vez:
// o "IF NOT EXISTS" evita erro se a regra ou o índice já existir.
// =====================================================================


// ---------------------------------------------------------------------
// 1. CONSTRAINTS DE UNICIDADE
//    Impedem nós duplicados. Cada constraint de unicidade também cria,
//    automaticamente, um índice RANGE na mesma propriedade.
// ---------------------------------------------------------------------

// Cada paper tem um id único (arXiv ID ou "doi:..." para o artigo da SBC)
CREATE CONSTRAINT paper_id_unique IF NOT EXISTS
FOR (p:Paper) REQUIRE p.id IS UNIQUE;

// Cada autor aparece uma única vez (identificado pelo nome)
CREATE CONSTRAINT author_name_unique IF NOT EXISTS
FOR (a:Author) REQUIRE a.name IS UNIQUE;

// Cada conceito/palavra-chave aparece uma única vez
CREATE CONSTRAINT concept_name_unique IF NOT EXISTS
FOR (c:Concept) REQUIRE c.name IS UNIQUE;

// Cada método (ex.: LightRAG, HippoRAG) aparece uma única vez
CREATE CONSTRAINT method_name_unique IF NOT EXISTS
FOR (m:Method) REQUIRE m.name IS UNIQUE;

// Cada dataset aparece uma única vez
CREATE CONSTRAINT dataset_name_unique IF NOT EXISTS
FOR (d:Dataset) REQUIRE d.name IS UNIQUE;


// ---------------------------------------------------------------------
// 2. ÍNDICE RANGE (equivalente ao antigo "BTree")
//    No Neo4j 5 o tipo BTREE foi removido e substituído pelo RANGE.
//    Acelera filtros como "papers de 2025" ou "papers depois de 2024".
// ---------------------------------------------------------------------

CREATE INDEX paper_year_range IF NOT EXISTS
FOR (p:Paper) ON (p.year);


// ---------------------------------------------------------------------
// 3. ÍNDICES FULLTEXT (busca por palavras, como um "Ctrl+F")
// ---------------------------------------------------------------------

// Busca por palavras no título e no resumo dos papers
CREATE FULLTEXT INDEX paper_fulltext IF NOT EXISTS
FOR (p:Paper) ON EACH [p.title, p.abstract];

// Busca por nome de conceitos, métodos e datasets
CREATE FULLTEXT INDEX entity_name_fulltext IF NOT EXISTS
FOR (n:Concept|Method|Dataset) ON EACH [n.name];


// ---------------------------------------------------------------------
// 4. ÍNDICE VETORIAL (busca por significado / similaridade de cosseno)
//    ATENÇÃO: vector.dimensions precisa ser IGUAL ao tamanho do vetor
//    gerado pelo modelo de embeddings escolhido pelo João Victor.
//      - all-MiniLM-L6-v2 ou paraphrase-multilingual-MiniLM-L12-v2 -> 384
//      - OpenAI text-embedding-3-small -> 1536
// ---------------------------------------------------------------------

CREATE VECTOR INDEX paper_embedding IF NOT EXISTS
FOR (p:Paper) ON (p.embedding)
OPTIONS {indexConfig: {
  `vector.dimensions`: 384,
  `vector.similarity_function`: 'cosine'
}};


// ---------------------------------------------------------------------
// 5. CONFERÊNCIA (rode depois dos comandos acima)
// ---------------------------------------------------------------------
// SHOW CONSTRAINTS;
// SHOW INDEXES;
