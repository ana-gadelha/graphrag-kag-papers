# Corpus de artigos

`papers.json` contém 26 registros selecionados para o MVP: 25 artigos do arXiv
e um artigo de autores brasileiros publicado pela Sociedade Brasileira de
Computação (SBC). O estudo da SBC aborda RAG multimodal com busca vetorial e
travessia de grafo de conhecimento em manuais de manutenção em português.

Cada item segue o esquema acordado pela equipe:

- `id`: identificador único do paper (arXiv ID, ou `doi:...` para o trabalho da SBC);
- `source`: origem do registro (`arXiv` ou `SBC`);
- `url`: link para o artigo original (usado na provenance das respostas);
- `title`: título do artigo;
- `authors`: lista de autores;
- `year`: ano de publicação/submissão;
- `abstract`: resumo;
- `keywords`: termos temáticos organizados para o projeto (não necessariamente
	palavras-chave fornecidas pelos autores).

Os metadados dos trabalhos do arXiv são coletados pela API arXiv. Para refazer
o arquivo, instale as dependências na raiz do repositório (`pip install -r
requirements.txt`) e execute `python data/build_papers_json.py`. O trabalho da
SBC está incluído com DOI `10.5753/reic.2026.8503`, pois não possui arXiv ID.
