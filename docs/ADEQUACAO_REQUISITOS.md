# Adequação ao enunciado — software e avaliação

Escopo: 26 títulos/abstracts de `data/papers.json`; autores e palavras-chave são
metadados. PDFs completos, artigo SBC/IEEE e preparação da defesa não fazem parte
desta implementação. Este arquivo é um mapa de comprovação, não um atestado de
resultados de inferência que ainda não foram medidos.

Atualização em 15/11/2026: a migração foi aplicada no Aura e a rodada com o modelo
gratuito foi executada. **O atendimento integral ainda não foi comprovado.**
Resultados, falhas e pendências estão no
[relatório da validação real](RELATORIO_VALIDACAO_REAL_2026-11-15.md).

| Requisito | Implementação | Evidência/aceite |
|---|---|---|
| §3.1 NER/RE automático | Extração OpenAI-compatible com menções, relações, trechos e offsets | Rodada real: 26 processados, 18 aceitos e oito rejeitados pela validação; qualidade semântica ainda requer revisão |
| §3.1 Resolução global | Registro de aliases e IDs canônicos, importação por identidade | Migração real: 276 entidades com ID; 302 nós, 335 relações e 26 embeddings preservados |
| §3.2 LPG e índices | Cinco labels, cinco relações, unicidade por IDs; RANGE, Fulltext e Vector | Validação do schema e testes em Neo4j isolado; RANGE atende ao antigo BTree |
| §3.3 Híbrido/multi-hop | Vetores + consultas fixas + Cypher LLM validado; fusão RRF | Seis perguntas nos dois modos e dois retestes híbridos; comparação científica ainda pendente |
| §3.3 Algoritmo | Shortest Path até quatro saltos entre fontes | Reteste real: GFM-RAG → autor Linhao Luo → Youtu-GraphRAG, dois saltos, citação `[P7]` |
| Fase 3 Cypher via LLM | Gramática fechada, AST, valores permitidos e compilação limitada | Cypher gerado pelo modelo e executado no Aura; rodada inicial teve truncamento e erro externo, retestes executaram |
| §2 Grounding explícito | Afirmações com trechos `[Sx]` e fatos/caminhos `[Px]` | Parcial: caminhos citados em respostas reais, mas a coautoria foi apresentada como afirmação documental com trechos sem suporte semântico |
| §3.4 Visualização interativa | Cytoscape.js local no Streamlit | Verificação real no navegador após corrigir eventos de mouse: seleção, pan/zoom/drag, inspeção e destaque de caminhos |
| §3.4 Rastreabilidade | Fontes, nós, relações, Cypher original/executado, status e confiança | Testes da interface e do contrato público |
| Rubrica: mitigação de alucinações | Avaliação pareada com casos sem resposta e armadilhas | Revisão de dois integrantes; relatório inconclusivo até concluir dados e avaliação |
| Qualidade/reprodução | Testes operacionais, configuração isolada, cache, limites e documentação | 199 testes passaram sem pulos, incluindo Neo4j isolado; cobertura de 79%; instruções no README |
| Créditos | Referências e licenças das dependências, Cytoscape vendorizado | README e licença no diretório do componente |
| Entrega no GitHub | Repositório privado existente | Seis colaboradores identificados como equipe; acesso do docente não confirmado; estado local completo ainda não publicado |
| Artigo e defesa | Responsabilidade da equipe | Pendentes de produção/revisão pela equipe |

## Limites das garantias

Uma citação válida demonstra a existência da referência; um trecho literal demonstra
que o texto aparece na fonte. Nenhum desses testes, sozinho, comprova que a
conclusão científica seja sustentada. A revisão humana mede essa dimensão.
A confiança exibida é uma heurística de recuperação, não uma probabilidade.

A compilação do Cypher admite somente o subconjunto documentado. Os limites
intermediários podem produzir resultados parciais. Quota, consulta inválida e
execução degradada permanecem visíveis e não contam como sucesso do planejador.

Não há promessa de redução de alucinações antes da avaliação. Se o resultado
comparativo for nulo, desfavorável ou incompleto, o relatório deve dizer isso.
