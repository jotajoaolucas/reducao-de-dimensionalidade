# Databricks notebook source
# MAGIC %md
# MAGIC # 02 · Métodos de Projeção (Swarm Behaviour, 2.400 features)
# MAGIC
# MAGIC Aqui a seleção de features (escolher um subconjunto das colunas originais) faz menos sentido —
# MAGIC as 2.400 colunas são posições/velocidades de 200 "boids", altamente correlacionadas entre si.
# MAGIC Faz mais sentido **projetar** o espaço original em poucas dimensões novas: PCA, Kernel PCA,
# MAGIC Factor Analysis. t-SNE entra só como ferramenta de visualização (não gera `transform()` para
# MAGIC dado novo, então não pode virar feature de um pipeline de classificação).
# MAGIC
# MAGIC **Bug corrigido em relação ao notebook original do MBA:** no PCA, o classificador era treinado
# MAGIC com `Xpca` (treino+teste juntos) e avaliado em `X_test` — vazamento de dado que infla a métrica
# MAGIC artificialmente. Aqui, o `fit` de cada método de projeção e do classificador usa **só X_train**.

# COMMAND ----------

dbutils.widgets.text("schema", "reducao_dimensionalidade.default", "Catalog.schema")
SCHEMA = dbutils.widgets.get("schema")

# COMMAND ----------

import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA, KernelPCA, FactorAnalysis
from sklearn.metrics import accuracy_score, f1_score

df = spark.table(f"{SCHEMA}.silver_swarm_behaviour").toPandas()
features_all = [c for c in df.columns if c != "Swarm_Behaviour"]
X = df[features_all]
y = df["Swarm_Behaviour"].astype(int)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

scaler = StandardScaler().fit(X_train)
X_train_s = scaler.transform(X_train)
X_test_s = scaler.transform(X_test)

# COMMAND ----------

# MAGIC %md ## 1. t-SNE (só visualização, 20% amostra por custo computacional)

# COMMAND ----------

from sklearn.manifold import TSNE

sample_idx = X_train.sample(frac=0.2, random_state=42).index
tsne = TSNE(n_components=2, random_state=42, init="pca", learning_rate="auto")
X_tsne = tsne.fit_transform(scaler.transform(X_train.loc[sample_idx]))

fig = plt.figure(figsize=(8, 6))
plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=y_train.loc[sample_idx], cmap="coolwarm", alpha=0.6)
plt.xlabel("t-SNE 1"); plt.ylabel("t-SNE 2"); plt.colorbar(label="Swarm_Behaviour")
display(fig)

# COMMAND ----------

# MAGIC %md ## 2. Benchmark: PCA vs Kernel PCA vs Factor Analysis
# MAGIC
# MAGIC Para cada método: fit em `X_train_s`, transform em `X_test_s`, treina `LogisticRegression`
# MAGIC nos dados projetados, avalia em teste projetado. `n_components=10` fixo para comparar as
# MAGIC 3 técnicas em igualdade de condições (equivalente ao notebook original).

# COMMAND ----------

def bench_projection(name, projector, X_train_s, X_test_s, y_train, y_test):
    start = time.time()
    Xp_train = projector.fit_transform(X_train_s)
    Xp_test = projector.transform(X_test_s)
    clf = LogisticRegression(max_iter=2000, random_state=42, class_weight="balanced")
    clf.fit(Xp_train, y_train)
    y_pred = clf.predict(Xp_test)
    elapsed = time.time() - start
    return {
        "method": name,
        "accuracy": accuracy_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
        "time_s": elapsed,
    }, Xp_train

# COMMAND ----------

results = []

pca = PCA(n_components=10, random_state=42)
res_pca, Xp_pca = bench_projection("PCA", pca, X_train_s, X_test_s, y_train, y_test)
results.append(res_pca)

# COMMAND ----------

kpca = KernelPCA(n_components=10, kernel="rbf", random_state=42)
res_kpca, _ = bench_projection("KernelPCA", kpca, X_train_s, X_test_s, y_train, y_test)
results.append(res_kpca)

# COMMAND ----------

fa = FactorAnalysis(n_components=10, random_state=42)
res_fa, _ = bench_projection("FactorAnalysis", fa, X_train_s, X_test_s, y_train, y_test)
results.append(res_fa)

# COMMAND ----------

results_df = pd.DataFrame(results)
spark.createDataFrame(results_df).write.mode("overwrite").saveAsTable(
    f"{SCHEMA}.gold_projection_results"
)
results_df

# COMMAND ----------

# MAGIC %md ## 3. Variância explicada (PCA)

# COMMAND ----------

fig = plt.figure(figsize=(8, 5))
plt.bar(range(1, 11), pca.explained_variance_ratio_)
plt.xlabel("componente"); plt.ylabel("variância explicada")
plt.title(f"soma dos 10 primeiros componentes: {pca.explained_variance_ratio_.sum():.2%}")
display(fig)

# COMMAND ----------

fig = plt.figure(figsize=(8, 6))
plt.scatter(Xp_pca[:, 0], Xp_pca[:, 1], c=y_train, cmap="coolwarm", alpha=0.5)
plt.xlabel("PC1"); plt.ylabel("PC2"); plt.colorbar(label="Swarm_Behaviour")
display(fig)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Conclusão
# MAGIC
# MAGIC - Com o vazamento de dado corrigido, os números de F1/acurácia tendem a ficar **mais baixos e
# MAGIC   mais realistas** que os do notebook original — isso é o resultado esperado, não um problema.
# MAGIC - De 2.400 features para 10 componentes (redução de 99,6%), compare qual método manteve melhor
# MAGIC   a separabilidade das classes (F1) e qual foi mais rápido — Kernel PCA tende a ser
# MAGIC   consideravelmente mais lento que PCA/FA por calcular a matriz de kernel completa.
# MAGIC - O t-SNE dá uma leitura visual de quão separáveis as classes são no espaço original — útil como
# MAGIC   diagnóstico antes de escolher método de projeção, mesmo não entrando no benchmark quantitativo.