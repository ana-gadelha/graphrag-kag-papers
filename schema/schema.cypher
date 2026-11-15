// GraphRAG/KAG paper graph, compatible with Neo4j 5.26 and Aura.
// Run `python -m src.migrate_entities --apply` once on an existing graph first.

// Papers keep their existing external ID.
CREATE CONSTRAINT paper_id_unique IF NOT EXISTS
FOR (p:Paper) REQUIRE p.id IS UNIQUE;

// Global entity identity is independent from display names and aliases.
CREATE CONSTRAINT author_entity_id_unique IF NOT EXISTS
FOR (a:Author) REQUIRE a.entity_id IS UNIQUE;

CREATE CONSTRAINT concept_entity_id_unique IF NOT EXISTS
FOR (c:Concept) REQUIRE c.entity_id IS UNIQUE;

CREATE CONSTRAINT method_entity_id_unique IF NOT EXISTS
FOR (m:Method) REQUIRE m.entity_id IS UNIQUE;

CREATE CONSTRAINT dataset_entity_id_unique IF NOT EXISTS
FOR (d:Dataset) REQUIRE d.entity_id IS UNIQUE;

// Year filters and paper retrieval.
CREATE INDEX paper_year_range IF NOT EXISTS
FOR (p:Paper) ON (p.year);

CREATE FULLTEXT INDEX paper_fulltext IF NOT EXISTS
FOR (p:Paper) ON EACH [p.title, p.abstract];

CREATE FULLTEXT INDEX entity_name_fulltext IF NOT EXISTS
FOR (n:Concept|Method|Dataset) ON EACH [n.name];

// The retrieval encoder uses 384-dimensional multilingual MiniLM vectors.
CREATE VECTOR INDEX paper_embedding IF NOT EXISTS
FOR (p:Paper) ON (p.embedding)
OPTIONS {indexConfig: {
  `vector.dimensions`: 384,
  `vector.similarity_function`: 'cosine'
}};
