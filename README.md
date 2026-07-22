# Dimensionality Reduction Benchmark (Databricks)

Recriação de um projeto original do MBA em Ciência de Dados (UNIFOR), reestruturado como
projeto de portfólio: comparação de técnicas de **seleção de features** e de **projeção**
sobre dois datasets com perfis de dimensionalidade opostos.

## Motivação

O material original (notebooks de exercício da disciplina) tinha o conteúdo técnico certo,
mas com problemas de metodologia e organização: vazamento de dado em uma das comparações,
splits de treino/teste refeitos a cada iteração de loop, notebooks monolíticos sem
documentação. Este projeto refaz a análise com o mesmo objetivo pedagógico, corrigindo esses
pontos e organizando o código em notebooks Databricks versionáveis.

## Datasets

| Dataset | Linhas | Features | Cenário | Fonte |
|---|---|---|---|---|
| `DESCRITORESMATH.csv` | 2.472 | 24 | baixa dimensionalidade | dados de desempenho escolar, Ceará |
| `Swarm_Behaviour.csv` | 23.309 | 2.400 | altíssima dimensionalidade | comportamento de locomoção em grupo (boids) |

## Estrutura

```
notebooks/
  00_ingestion_eda.py            # carga (bronze) + EDA + limpeza (silver) dos 2 datasets
  01_feature_selection_methods.py # aleatório vs. variância vs. árvore (DESCRITORESMATH)
  02_projection_methods.py        # t-SNE, PCA, Kernel PCA, Factor Analysis (Swarm)
```

Cada notebook está no formato "Databricks notebook source" (`# Databricks notebook source`
+ `# COMMAND ----------`), então pode ser importado direto em um Workspace, ou usado com o
Databricks CLI / Repos.

## Como rodar

1. Suba `DESCRITORESMATH.csv` e `Swarm_Behaviour.csv` no volume `reducao_dimensionalidade.default.raw_data`.
2. Os widgets `volume_path` (`/Volumes/reducao_dimensionalidade/default/raw_data`) e `schema` (`reducao_dimensionalidade.default`) já vêm com esse default; ajuste só se usar outro catálogo.
3. Rode `00_ingestion_eda.py` (cria as tabelas bronze/silver).
4. Rode `01_feature_selection_methods.py` e `02_projection_methods.py` (independentes entre si).

Funciona no Databricks Free Edition (clusters serverless / single-node).

## O que foi corrigido em relação ao notebook original

- **Vazamento de dado no PCA** (exercício de projeção): o classificador era treinado em
  `Xpca` completo (treino+teste) e avaliado no teste — métrica inflada artificialmente.
  Agora `fit` acontece só em `X_train`, `transform`/`predict` em `X_test`.
- **Split refeito a cada iteração** (exercício de seleção de features): o notebook original
  reamostrava treino/teste dentro do loop que testava 1..N features, adicionando ruído à
  comparação de tempo/performance. Agora o split é único, feito uma vez no início.
- **Normalização vazando teste**: `MinMaxScaler`/`StandardScaler` agora são ajustados (`fit`)
  só em `X_train`.
- **Desbalanceamento não tratado**: a classe minoritária do dataset escolar (`Adequado`, 1,3%
  do dado) não era considerada na escolha de métrica. Agora acurácia é reportada ao lado de
  F1, com `class_weight="balanced"` nos modelos.
- **Amostragem perdendo dado real**: o notebook original do Swarm reduzia o dataset para 20%
  e depois subamostrava a classe majoritária para balancear — descartando dado real sem
  necessidade, já que o desbalanceamento ali é moderado (66/34). Mantido o dataset completo.

## Principais achados

_(preencher após rodar no cluster — os notebooks já geram as tabelas `gold_feature_selection_results`
e `gold_projection_results` com os números para consolidar aqui)._

## Próximos passos possíveis

- Notebook de consolidação lendo as duas tabelas `gold_*` e montando um comparativo único.
- Testar LDA como alternativa supervisionada de projeção no dataset Swarm.
- Publicar como artigo no LinkedIn, no mesmo formato do artigo sobre o pipeline de
  crédito/RAG.
