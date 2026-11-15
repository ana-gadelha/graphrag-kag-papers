# Text-to-Graph: extração, evidência e identidade

O corpus contém 26 títulos/abstracts, não textos integrais. Autores e palavras-chave
vêm dos metadados; Method e Dataset são extraídos do texto. A origem é registrada
para evitar apresentar metadados como resultado de NER.

```bash
# Automático: usa o modelo EXTRACTION_LLM_* ou RETRIEVAL_LLM_* explícito.
python run_pipeline.py --provider openai

# Modos sem inferência, úteis para reprodução e comparação.
python run_pipeline.py --provider curated
python run_pipeline.py --provider heuristic
```

O padrão `auto` usa extração por LLM. Esse comando envia títulos/abstracts ao
provedor configurado; confira previamente autorização, modelo e quota. A
configuração está no [`.env.example`](../.env.example). Não há fallback para API
paga. Sem `--import-neo4j`, não escreve no banco. O modo curado/heurístico continua
explicitamente identificado e não comprova funcionamento da extração por LLM.

## Contrato e validação

A resposta do modelo contém `mentions`: forma de superfície, label, relação,
campo `title`/`abstract` e trecho literal. O programa localiza o trecho, calcula
o intervalo `[start,end)` da menção e confere tipo/relação e indícios de suporte.
Relações possíveis: Method–PROPOSES, Method–USES_METHOD, Dataset–EVALUATED_ON.
Uma menção sem evidência de relação não produz uma aresta científica.

As menções validadas recebem `entity_id` e nome canônico. As três listas legadas
`proposed_methods`, `used_methods` e `datasets` são derivadas dessas menções.
`metadata_entities` mantém autores/conceitos com proveniência própria.

Uma lista de menções vazia é um resultado válido sem relações identificadas.
Falha de API/JSON, trecho ausente, offsets inconsistentes ou alias ambíguo são
erros. A validação integral antecede qualquer importação: lote inválido não
atualiza o grafo. Importação reconcilia cada paper em transação, preservando
embeddings e relações fora das formas gerenciadas pelo pipeline.

O registro global resolve aliases exatos, sem merge fuzzy. Autores identificados
por nome não têm desambiguação autoritativa de homônimos. Revise colisões e
aliases novos; não transforme equivalência textual em certeza sobre pessoas.

## Retomada e avaliação

O transporte compartilhado guarda resultados por hash do conteúdo, prompt,
configuração e versão. Apenas respostas aprovadas entram no cache reutilizável;
falhas permanecem auditáveis. Quota e falhas também consomem o orçamento local.
Retome a execução quando o limite renovar; resultados válidos serão reutilizados.

Os relatórios ficam em `output/`. Para migrar/importar/indexar:

```bash
python -m src.migrate_entities
python run_pipeline.py --provider openai --import-neo4j --index-embeddings
```

O pipeline com `--import-neo4j` aplica a migração após validar a extração. Use banco
isolado para testes. A curadoria atual é um rascunho para avaliação: revisão de
dois integrantes deve confirmar entidades, relações e evidências, conforme
[evaluation/README.md](../evaluation/README.md). Trechos válidos e indícios
linguísticos não substituem avaliação semântica humana.
