# Correlação Estrutura-Semântica em Radicais de Caracteres Chineses

Este projeto realiza uma investigação quantitativa sobre a **transparência semântica** de ideogramas chineses. Utilizamos uma abordagem híbrida que combina **Teoria de Redes Complexas** e **Modelos de Linguagem de Grande Escala (LLMs)** para mapear como os componentes estruturais (radicais) se relacionam com o significado moderno dos caracteres.

Projeto desenvolvido para a disciplina **MC859 - Projeto em Teoria da Computação**.

## Autores

- **Ana Carolina de Almeida Cardoso** (RA: 246914)
- **Pedro Damasceno Vasconcellos** (RA: 260640)

---

## Descrição do Projeto

O objetivo central é quantificar o grau de preservação ou opacidade semântica resultante da evolução linguística. 

Por exemplo, o caractere **网 (wǎng)** representava originalmente uma rede de pesca física, mas hoje expandiu-se para o conceito de "Internet". Buscamos medir se a estrutura morfológica original ainda exerce influência estatística na representação vetorial (*embeddings*) desses conceitos no chinês contemporâneo.

## Objetivos

### Geral

Quantificar a influência semântica dos radicais na construção do significado dos ideogramas através de métricas de centralidade em um **grafo bipartido ponderado**.

### Específicos

- Construir um grafo relacionando radicais e ideogramas (via base **Unihan**).
- Computar *embeddings* semânticos utilizando o modelo `bert-base-chinese`.
- Comparar métricas de centralidade (Degree, Betweenness, Closeness, Eigenvector e PageRank).
- Validar a significância dos resultados contra modelos de grafos aleatórios (**Erdos-Rényi**).

## Metodologia

A execução técnica divide-se em três fases:

1. **Modelagem do Grafo:** Construção de um grafo $G = (U \cup W, E)$ onde o peso das arestas é definido pela similaridade de cosseno entre os vetores do radical e do caractere.
2. **Análise de Centralidade:** Identificação da dominância semântica e posição topológica dos radicais, utilizando 5 métricas de centralidade distintas.
3. **Baselines Estatísticos:** Comparação com grafos de pesos permutados e modelos aleatórios para garantir que as correlações encontradas não sejam fruto do acaso.

## Origem dos Dados

- **Estrutura:** Unihan Database (`Unihan_IRGSources.txt`, `IDS.txt` e `Unihan_Readings.txt`).
- **Semântica:** `bert-base-chinese` via Hugging Face.

## Cronograma e Planejamento


| Etapa                              | Prazo | Status |
| ---------------------------------- | ----- | ------ |
| Proposta do Projeto                | 20/03 | ✅      |
| Coleta e Extração de Embeddings    | 10/04 | ✅      |
| Construção do Grafo                | 26/04 | ✅      |
| Análise de Resultados              | 25/05 | ⏳      |
| Entrega Final (Código + Relatório) | 08/06 | ⏳      |


