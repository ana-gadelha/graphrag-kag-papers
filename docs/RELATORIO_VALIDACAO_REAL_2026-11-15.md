# Validação real e conferência do enunciado — 15/11/2026

## Conclusão

A migração no Neo4j Aura foi aplicada e a rodada real foi concluída com
`nvidia/nemotron-3.5-super-vl:free`, via OpenRouter. Há evidência real de recuperação
híbrida, Cypher gerado por LLM, caminhos de dois saltos, abstenção e interface
interativa. **Ainda não é possível declarar atendimento integral ao documento.**
Permanecem falhas na extração, uma lacuna de suporte semântico nas citações de
uma resposta e a avaliação humana necessária para comprovar mitigação de
alucinações. Artigo, defesa e entrega completa no GitHub também exigem conclusão
ou confirmação pela equipe.

Esta é uma validação operacional. Ela não substitui o benchmark com referências
revisadas e não constitui uma medida de precisão científica do sistema.

## Documento e escopo conferidos

Fonte: `Projeto Final_ GraphRAG e KAG em Ciência da Computação (1).docx`, em
`C:\Users\João Victor\Downloads`. SHA-256:
`22fa7570d3d2a32eec327a7e61c5131f9309e5cbcc99d1eac06df48120b2fe03`.

Corpus: os 26 títulos, abstracts e metadados de `data/papers.json`. Não foram
usados PDFs completos. A conferência contempla os quatro pilares, as fases,
a rubrica e as condições de entrega do enunciado. O desenvolvimento anterior
estava limitado ao software e à avaliação; artigo e defesa continuam sendo
entregas acadêmicas da equipe.

## Migração aplicada no Aura

Foi salvo um snapshot lógico de recuperação, incluindo nós, relações,
constraints, índices e o registro anterior de entidades, antes da escrita:

- Backup local: `output/aura_backup_20261115T170128Z.json`, permissão `0600`.
- SHA-256: `fdaa49cf8a695725ad1298909094a2c16e482fd12d09bc708aeeb0566388a7ca`.
- Relatório: `output/aura_migration_applied_20261115T170128Z.json`.

O snapshot é uma exportação lógica local; não é um snapshot nativo do serviço
Aura, e não foi necessário executar uma restauração.

| Verificação após aplicação | Resultado |
|---|---|
| Nós preservados | 302 |
| Relações, direção e propriedades preservadas | 335 |
| Papers e suas propriedades preservados | 26 |
| Embeddings preservados | 26 |
| Entidades com identidade canônica | 276 |
| Constraints de unicidade por ID | Paper, Author, Concept, Dataset e Method |
| Estado dos índices | Todos ONLINE |
| Prévia repetida depois da migração | Válida, sem merges, transferências ou conflitos |

O registro `data/entity_registry.json` passou a incluir as entidades observadas
no Aura. As extrações novas e ainda não revisadas do modelo ficaram em artefatos
separados e não substituíram as relações existentes no Aura.

## Extração real dos 26 registros

O modelo configurado foi mantido, sem fallback pago. Os 26 registros passaram
pelo caminho normal de extração. Houve 27 reservas no ledger: uma execução foi
interrompida após 23 registros e o registro seguinte, sem resposta salva, precisou
ser retomado. Os 23 já concluídos não foram enviados novamente.

As respostas salvas foram revalidadas offline com o mesmo validador final:

| Resultado | Quantidade |
|---|---|
| Registros processados | 26 |
| Respostas aceitas na validação final | 18 |
| Respostas rejeitadas por evidência insuficiente para a relação | 8 |
| Respostas recuperadas por correções do validador, sem nova chamada | 3 |

Foram corrigidos falsos negativos para expressões de integração, comparação e
desempenho em datasets. Também foi corrigido o reconhecimento heurístico de
aliases sobrepostos: `RAG` dentro de `Graph RAG` não cria uma segunda menção,
mas uma ocorrência independente de `RAG` continua reconhecida.

Rejeições remanescentes: `2501.13958`, `2405.14831`, `2412.18644`, `2503.13804`,
`2508.19855`, `2502.09304`, `2608.28766` e `doi:10.5753/reic.2026.8503`.
Entre os motivos estão nomes isolados sem contexto de proposta/uso e descrições
de trabalhos anteriores tratadas como uso pelo artigo. A validação é conservadora:
esses oito resultados exigem análise, e não demonstram que todas as suas menções
estejam erradas. Uma menção inválida impede a aceitação do registro inteiro.

Os 18 aceitos significam aprovação pelas regras automáticas, não aprovação humana
nem precisão de 69,2%. As 26 respostas completas somaram 101.420 tokens informados
pelo provedor, incluindo raciocínio; a tentativa interrompida não tem recibo salvo.
O hash do validador usado na revalidação está no relatório de extração.

Artefatos em `output/real_validation_20261115/`: `extraction_report.json`,
`papers_enriched_llm.json`, `paper_diagnostics.json`,
`offline_validation_audit.json` e o registro de entidades separado da rodada.
O arquivo de diagnóstico enriquecido não é um lote aprovado para importação.

## Consultas reais e retestes

Foram executadas seis perguntas em ambos os modos, híbrido e vetorial, pelo
contrato público `answer_question()`. As 12 execuções retornaram resultados
válidos para a aplicação; isso inclui abstenções e execução degradada do
planejador, que não contam como respostas corretas nem sucesso do Cypher.

| Caso | Híbrido, rodada inicial | Vetorial, rodada inicial |
|---|---|---|
| Artigos avaliados em HotpotQA | Duas afirmações; Cypher executado | Uma afirmação |
| Métodos propostos/usados por HippoRAG | Quatro afirmações, duas de grafo com `[P25]`/`[P26]`; Cypher executado | Abstenção |
| Autores comuns de 2407.10805 e 2509.22009 | Abstenção; Cypher truncado pelo provedor | Abstenção |
| Conexão entre 2502.01113 e 2508.19855 | Abstenção; erro do provedor no planejador | Abstenção |
| GPU/VRAM não informadas no abstract de LightRAG | Abstenção; Cypher executado | Abstenção |
| Limitações e mecanismos descritos em PathRAG | Cinco afirmações; Cypher válido sem candidatos | Uma afirmação |

No truncamento, o provedor informou 8.192 tokens de conclusão, todos usados em
raciocínio, sem conteúdo final. Isso é uma limitação real dessa geração.
Separadamente, a investigação revelou um problema de recuperação no código:
os artigos explicitamente identificados nas perguntas podiam ficar fora do
contexto quando o planejador falhava. A recuperação híbrida agora reserva esses
artigos antes dos vizinhos genéricos e os inclui nos seeds. O modo vetorial
permaneceu com sua definição original para a comparação futura.

Após a correção foram realizados dois novos testes híbridos:

- **Coautoria:** Cypher executado e resposta com Cehao Yang, Chengjin Xu, Jian Guo
  e Xuhui Jiang, conferidos nos metadados locais; 53,494 segundos.
- **Multi-hop:** Cypher executado e resposta GFM-RAG ← AUTHORED — Linhao Luo —
  AUTHORED → Youtu-GraphRAG, com `[S1] [S2] [P7]`; caminho validado de dois saltos;
  75,447 segundos. O autor compartilhado foi conferido nos metadados locais.

**Lacuna encontrada na coautoria:** os nomes estão corretos, porém o modelo os
classificou como afirmação documental e citou frases introdutórias dos abstracts
que não sustentam esses nomes. O validador confirmou a literalidade dos trechos,
mas não sua implicação semântica. Essa resposta deveria citar os caminhos de
autoria. O caso permanece pendente; a capacidade de citar caminhos existe, mas
o requisito não é garantido em todas as respostas reais.

As fases estão salvas separadamente em `operational_queries.json` e
`operational_recovery_queries.json`. Não foram misturadas como um benchmark
controlado, pois a implementação mudou entre elas. O hash do grafo antes/depois
foi constante nas duas fases:
`dafb0fcc5cd852017e4cd13e9471c78d63ce84a765cf3e290f88c51b9df47cb1`.

O limite de saída permaneceu em 8.192 tokens, sem retries automáticos. O orçamento
local diário foi elevado somente nos processos de teste para 60, pois o ledger
também registra tentativas anteriores que falharam antes de chegar ao provedor.
O `.env` não foi alterado. Na consulta final à API de conta, às 17:36:44 UTC,
o OpenRouter informou `usage_daily=0`, conta gratuita e 47/50 solicitações
gratuitas usadas, com três restantes. Esses são os contadores informados naquele
instante, não uma previsão nem uma equivalência com reservas do ledger local.

## Interface e testes automatizados

A interface foi aberta no navegador com uma resposta real salva, de 84 nós,
88 relações e 99 caminhos. O replay não realizou novas inferências.

Uma falha de interação foi reproduzida com cliques reais: o isolamento de estilos
do componente Streamlit alterava o alvo dos eventos nativos do Cytoscape. A
configuração `isolate_styles=False` corrigiu esse comportamento. Foram verificados
clique em nó e relação dirigida, inspeção, pan, arraste de nó, zoom com a roda,
restauração da visualização, persistência após rerun e destaque de `[P1]`/`[P99]`.
O console do navegador ficou sem erros ou avisos. Consultas, fontes e confiança
foram exibidas nos painéis. A confiança é heurística, não probabilidade calibrada.

Evidências: `output/real_validation_20261115/ui_real.png` e
`ui_verification.json`.

A suíte completa passou com **199 testes, sem falhas nem pulos**, usando Neo4j
Community 5.26 em `bolt://127.0.0.1:17688`, separado do Aura. Cobertura de 79%
com branches. Nenhuma inferência externa foi permitida na suíte. A verificação
de sintaxe JavaScript, o lint direcionado e `git diff --check` passaram.

Depois da migração, foram atualizados somente os hashes do registro canônico
nos dois arquivos de rascunho da avaliação: os hashes de papers e entidades
curadas foram conferidos e permaneceram iguais. Todos os status humanos ficaram
`PENDING`. `evaluation check` agora retorna `valid_with_pending_review`.
Os 28 testes da avaliação também passaram após essa atualização.

## Situação de cada condição do enunciado

| Condição | Situação verificada |
|---|---|
| §3.1 Extração automática de documentos não estruturados | Implementada e executada; parcial na robustez: oito dos 26 registros ainda não aceitos |
| §3.1 Resolução de entidades e schema de domínio | Implementados; migração aplicada e validada no Aura |
| §3.2 LPG, labels, relações, constraints e índices | Atendidos no banco verificado; RANGE, Fulltext e Vector disponíveis |
| §3.3 Vetores/cosseno + Cypher e fusão híbrida | Implementados e exercitados com modelo e banco reais |
| §3.3 Travessia de dois ou mais saltos e algoritmo de grafo | Exercitados; Shortest Path e caminho real de dois saltos citado |
| Fase 3: geração de Cypher via LLM | Exercitada; consultas validadas e executadas, com falhas iniciais preservadas nos relatórios |
| §2/§3.4 Procedência, caminhos, Cypher, nós e confiança | Painéis e citações implementados; suporte semântico das citações permanece parcial, conforme o caso de coautoria |
| §3.4 Interface funcional e grafo interativo | Confirmados no navegador com resposta real salva |
| Rubrica: mitigação comprovada de alucinações | Pendente: não há comparação final com revisão humana |
| Qualidade, testes e instruções de reprodução/deployment | Suíte validada e instruções no README; qualidade da entrega acadêmica não é determinada pela cobertura |
| Créditos das bibliotecas e licença do Cytoscape | Presentes no README e nos arquivos do componente |
| Grupo de quatro a seis alunos | README e colaboradores mostram seis integrantes; participação e presença na defesa dependem da equipe |
| Código completo via GitHub | Pendente: repositório privado, `main` local três commits à frente da referência local de `origin/main`, além de alterações sem commit; não foi publicado nesta rodada |
| Acesso do docente ao repositório | Não confirmado: lista de seis colaboradores consultada, sem identificação do docente; nenhum convite foi enviado |
| Artigo técnico no template SBC/IEEE | Pendente; nenhum artigo final identificado na entrega inspecionada |
| Apresentação presencial, prazo em relação à AV2 | Dependem da equipe/calendário; realização e cumprimento não comprovados |

O repositório remoto conferido é
`ana-gadelha/graphrag-kag-papers`, privado. Os colaboradores retornados foram
MirelaFerraz, ana-gadelha, GilversonR, Perazzojoao, CC-UNIPE-MARLONRN e RamonFVM.
Isso não comprova ausência absoluta de outro meio de acesso do professor.

## Pendências para atestar atendimento integral

1. Revisar as oito extrações rejeitadas e corrigir a produção de evidências sem
   transformar descrições, títulos ou nomes isolados em relações não sustentadas.
2. Tratar o caso de afirmações sobre autoria/estrutura apresentadas como texto
   documental, garantindo a procedência adequada do grafo.
3. Dois integrantes devem revisar as 22 perguntas de desenvolvimento/avaliação
   e as 26 anotações de extração. Depois, executar o benchmark de avaliação
   congelado e revisar as afirmações geradas. Atualmente são zero aprovações.
4. Produzir a conclusão comparativa de alucinações apenas se as métricas e a
   revisão a sustentarem; a conclusão também pode ser inconclusiva ou desfavorável.
5. Preparar a entrega integral no GitHub, confirmar acesso do docente e concluir
   artigo/defesa conforme calendário da disciplina.

Os artefatos de runtime, cache e backup continuam locais e ignorados pelo Git.
Este relatório documenta os resultados necessários para revisão sem publicar
chaves, mensagens brutas do provedor ou o backup do banco no repositório.
