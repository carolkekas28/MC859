# Correlação Estrutura-Semântica em Radicais de Caracteres Chineses

Este projeto realiza uma investigação quantitativa sobre a **transparência semântica** de ideogramas chineses. Utilizamos uma abordagem híbrida que combina **Teoria de Redes Complexas** e **embeddings semânticos** para mapear como os componentes estruturais (radicais) se relacionam com o significado moderno dos caracteres.

Projeto desenvolvido para a disciplina **MC859 - Projeto em Teoria da Computação**.

## Autores

1. **Ana Carolina de Almeida Cardoso** (RA: 246914)
2. **Pedro Damasceno Vasconcellos** (RA: 260640)

---

## Descrição do Projeto

O objetivo central é quantificar o grau de preservação ou opacidade semântica resultante da evolução linguística. 

Por exemplo, o caractere **网 (wǎng)** representava originalmente uma rede de pesca física, mas hoje expandiu-se para o conceito de "Internet". Buscamos medir se a estrutura morfológica original ainda exerce influência estatística na representação vetorial (*embeddings*) desses conceitos no chinês contemporâneo.

## Objetivos

### Geral

Quantificar a influência semântica dos radicais na construção do significado dos ideogramas através de métricas de centralidade em um **grafo bipartido ponderado**.

### Específicos

1. Construir um grafo relacionando radicais e ideogramas (via base **Unihan**).
2. Computar *embeddings* semânticos utilizando o modelo `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`.
3. Comparar métricas de centralidade (Degree, Betweenness, Closeness, Eigenvector e PageRank).

## Metodologia

A execução técnica divide-se em três fases:

1. **Modelagem do Grafo:** Construção de um grafo bipartido $G = (R \cup C, E)$ com decomposição recursiva via IDS até folhas Kangxi.
2. **Análise de Centralidade:** Identificação da dominância semântica e posição topológica dos radicais, utilizando 5 métricas de centralidade distintas.
3. **Baselines Estatísticos:** Comparação com grafos de pesos permutados e modelos aleatórios para garantir que as correlações encontradas não sejam fruto do acaso.

## Origem dos Dados

**Estrutura:** Unihan Database (`Unihan_IRGSources.txt`, `ids.txt` e `Unihan_Readings.txt`).  
**Referência de radicais:** `data/reference/kangxi_radicals.csv` (214 radicais Kangxi, com variantes simplificadas onde necessário).  
**Semântica:** `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` via Hugging Face.

## O que já foi implementado

1. **Recorte de caracteres:** filtro de faixa Unicode `U+3400..U+9FFF` (CJK Extension A + bloco principal), aplicado de forma consistente nos scripts de extração. Desta forma, apenas caracteres comuns cuja tokenização é coerente são usados.
2. **Tabela de caracteres:** definições (`kDefinition` em `Unihan_Readings.txt`) e cobertura em `ids.txt` → `data/processed/characters.csv` (inclui coluna `semantic_text` após rodar o script de embeddings).
3. **IDS:** uma única decomposição por caractere, com precedência regional `G > T > J > K > V > X` (cada letra representa uma região; ex: G = China continental), limpeza de tags regionais nas colunas de expressão, e remoção de numerais circulados (placeholders gráficos) na expressão usada para o grafo.
4. **Decomposição recursiva:** parsing de operadores IDS, folhas mapeadas aos 214 radicais Kangxi (mapeamento de variantes, desambiguação contextual de `阝` e `月`, fallback via `kRSUnicode` em `Unihan_IRGSources.txt` quando necessário) → `ids_recursive_kangxi.csv` e `radical_character_edges.csv` (sem arestas duplicadas para o mesmo par radical–caractere).
5. **Embeddings semânticos:** vetores L2-normalizados por caractere, texto `character: …. definition: …` ou `character: …` só; arquivos `semantic_embeddings.npy`, `semantic_embeddings_index.csv`, `semantic_embeddings_meta.json` (o `.npy` é grande e está listado no `.gitignore`).
6. **Grafo semântico ponderado:** para cada aresta do grafo estrutural, pesos `weight_cosine` (similaridade de cosseno) e `weight_norm_01` = (cosine + 1) / 2, usando embeddings *puros* do radical (`character: {radical}.`) e do caractere, sem média de vizinhança.
7. **Exportação GraphML:** instâncias não direcionadas com atributos de nó e aresta em `data/graphs/`; cópias para entrega em `entrega_parcial/`.
8. **Análise inicial (entrega parcial):** distribuição de graus (agrupando graus ≥ 7 em `7+`) e distribuição de `weight_norm_01`, com PNG + CSVs gerados a partir de scripts em `entrega_parcial/python/`.

Arquivos intermediários e finais principais (`data/processed/`):


| Arquivo                                | Conteúdo                                               |
| -------------------------------------- | ------------------------------------------------------ |
| `characters.csv`                       | metadados por caractere, definição, `semantic_text`    |
| `ids_selected.csv`                     | IDS selecionado e expressão normalizada para o grafo   |
| `ids_recursive_kangxi.csv`             | trilha da recursão e status de expansão                |
| `radical_character_edges.csv`          | arestas bipartidas radical Kangxi → caractere          |
| `radical_character_edges_weighted.csv` | mesmas arestas + colunas de peso e metadados do modelo |


Instâncias de grafo (GraphML):

1. `data/graphs/graph_structural.graphml` — arestas unitárias (visão puramente estrutural).
2. `data/graphs/graph_semantic_weighted.graphml` — mesma topologia com pesos semânticos nas arestas.

**Entrega parcial (`entrega_parcial/`):**

1. `graph_structural.graphml`, `graph_semantic_weighted.graphml` — instâncias para divulgação.
2. `plots/` — `degree_distribution.png`, `weight_norm_distribution.png`.
3. `csvs/` — dados tabulares usados ou derivados dos gráficos (`degree_distribution.csv`, `weight_norm_distribution_bins.csv`, `weight_norm_summary.csv`).
4. `python/` — scripts que geram os plots a partir de GraphML / CSV de arestas.

*Ainda em planejamento para a entrega final:* métricas de centralidade em lote, comparação de agrupamentos (estrutural *vs.* semântico) e baselines estatísticos, conforme a metodologia e o cronograma abaixo.

## Tutorial: como reproduzir o pipeline

**Pré-requisitos:** [uv](https://github.com/astral-sh/uv) instalado, Python ≥ 3.10, e a pasta `Unihan/` com pelo menos `Unihan_Readings.txt`, `Unihan_IRGSources.txt` e `ids.txt` (a partir da raiz do repositório, nos caminhos padrão dos scripts).

**Linhagem das tabelas:** em cada passo: **entradas**, **saídas**, **propósito** e **script** (caminhos relativos à raiz do repositório).

1. **Script:** `scripts/build_characters_base.py`  
   **Entradas:**  
   1. `Unihan/Unihan_Readings.txt` (campo `kDefinition`)  
   2. `Unihan/ids.txt` (universo e cobertura)  
   **Saídas:**  
   1. `data/processed/characters.csv`  
   **Propósito:** base de ideogramas no recorte Unicode (símbolo, definição, flags); depois o passo 4 acrescenta `semantic_text` neste mesmo arquivo.

2. **Script:** `scripts/build_ids_selected.py`  
   **Entradas:**  
   1. `Unihan/ids.txt`  
   **Saídas:**  
   1. `data/processed/ids_selected.csv`  
   **Propósito:** uma decomposição em IDS por caractere (precedência regional), expressões bruta, limpa e a usada no grafo (sem placeholders circulados). Aqui um ideograma ainda pode aparecer dentro do outro, e isto será resolvido no próximo passo

3. **Script:** `scripts/build_recursive_kangxi.py`  
   **Entradas:**  
   1. `data/processed/ids_selected.csv`  
   2. `data/reference/kangxi_radicals.csv`  
   3. `Unihan/Unihan_IRGSources.txt` (fallback `kRSUnicode`)  
   **Saídas:**  
   1. `data/processed/ids_recursive_kangxi.csv`  
   2. `data/processed/radical_character_edges.csv`  
   **Propósito:** *ids_recursive_kangxi* — registro da recursão até folhas Kangxi, onde teremos os radicais puros. *radical_character_edges* — arestas do grafo bipartido estrutural (peso unitário).

4. **Script:** `scripts/build_semantic_embeddings.py`  
   **Entradas:**  
   1. `data/processed/characters.csv` (lido e reescrito com `semantic_text`)  
   2. modelo `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` (download no Hugging Face na primeira execução)  
   **Saídas:**  
   1. `data/processed/characters.csv` (atualizado)  
   2. `data/processed/semantic_embeddings.npy`  
   3. `data/processed/semantic_embeddings_index.csv`  
   4. `data/processed/semantic_embeddings_meta.json`  
   **Propósito:** texto para o encoder; matriz de embeddings por caractere; índice e metadados do modelo.

5. **Script:** `scripts/build_weighted_edges_from_embeddings.py`  
   **Entradas:**  
   1. `data/processed/radical_character_edges.csv`  
   2. `data/processed/semantic_embeddings.npy`  
   3. `data/processed/semantic_embeddings_index.csv`  
   4. o mesmo modelo gera embeddings "puros" para cada radical  
   **Saídas:**  
   1. `data/processed/radical_character_edges_weighted.csv`  
   **Propósito:** reutilizar a topologia estrutural e acrescentar `weight_cosine` e `weight_norm_01` por aresta.

6. **Script:** `scripts/export_graphml.py` (um único comando gera as duas saídas abaixo)  
   **Entradas:**  
   1. `data/processed/radical_character_edges.csv` (para o grafo estrutural)  
   **Saídas:**  
   1. `data/graphs/graph_structural.graphml`  
   **Propósito:** instância GraphML só com topologia (pesos 1 nas arestas), para análise e visualização.

7. **Script:** *idem* `scripts/export_graphml.py` (mesma execução do passo 6)  
   **Entradas:**  
   1. `data/processed/radical_character_edges_weighted.csv` (para o grafo semântico)  
   **Saídas:**  
   1. `data/graphs/graph_semantic_weighted.graphml`  
   **Propósito:** mesma topologia de nós, com pesos semânticos nas arestas.

**1. Ambiente e dependências**

```bash
cd /caminho/para/MC859
uv sync
```

**2. Dados processados (ordem sugerida)**

Execute a partir da raiz do repositório; os caminhos padrão apontam para `Unihan/` e `data/processed/`.

```bash
uv run python scripts/build_characters_base.py
uv run python scripts/build_ids_selected.py
uv run python scripts/build_recursive_kangxi.py
```

**3. Embeddings (pode levar um bom tempo: download do modelo e codificação em lote)**

```bash
uv run python scripts/build_semantic_embeddings.py
```

Gera `data/processed/semantic_embeddings.npy` (não versionado no Git) e metadados associados, e atualiza a coluna `semantic_text` em `characters.csv`.

**4. Arestas ponderadas e exportação GraphML**

```bash
uv run python scripts/build_weighted_edges_from_embeddings.py
uv run python scripts/export_graphml.py
```

**5. Gráficos da entrega parcial**

Lê, por padrão, o grafo estrutural em `data/graphs/graph_structural.graphml` e grava CSV e PNG em `entrega_parcial/csvs/` e `entrega_parcial/plots/`. Use `--graphml` se o `.graphml` estiver em outro caminho (por exemplo, só em `entrega_parcial/`).

```bash
uv run python entrega_parcial/python/plot_degree_distribution.py
uv run python entrega_parcial/python/plot_weight_norm_distribution.py
```

**6. Copiar instâncias para `entrega_parcial/` (se necessário)**

```bash
cp data/graphs/graph_structural.graphml entrega_parcial/
cp data/graphs/graph_semantic_weighted.graphml entrega_parcial/
```

**Notas:** na primeira vez, o `uv sync` baixa as dependências do `pyproject.toml` (incluindo `sentence-transformers` e o renderizador de imagem estática do Plotly, `kaleido`). A primeira execução do *sentence-transformers* também baixa os pesos do modelo no Hugging Face.

## Cronograma e Planejamento


| Etapa                              | Prazo | Status |
| ---------------------------------- | ----- | ------ |
| Proposta do Projeto                | 20/03 | ✅      |
| Coleta e Extração de Embeddings    | 10/04 | ✅      |
| Construção do Grafo                | 26/04 | ✅      |
| Análise de Resultados              | 25/05 | ⏳      |
| Entrega Final (Código + Relatório) | 08/06 | ⏳      |


