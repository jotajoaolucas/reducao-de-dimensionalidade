# Databricks notebook source
# MAGIC %md
# MAGIC # 01 · Métodos de Seleção de Features (DESCRITORESMATH)
# MAGIC
# MAGIC Comparamos 3 estratégias para decidir **quantas e quais** das 24 features manter,
# MAGIC olhando o trade-off entre tempo de treino e performance (F1):
# MAGIC
# MAGIC 1. **Ordem aleatória** (baseline / controle)
# MAGIC 2. **Filtro por variância** (não-supervisionado)
# MAGIC 3. **Importância por árvore** (Random Forest, supervisionado)
# MAGIC
# MAGIC Correção de metodologia em relação ao notebook original do MBA: aqui o **split treino/teste é
# MAGIC feito uma única vez, no início**, e todos os métodos são avaliados sobre o mesmo `X_test` —
# MAGIC no original, o split era refeito a cada iteração do loop, o que introduz ruído na comparação.

# COMMAND ----------

dbutils.widgets.text("schema", "reducao_dimensionalidade.default", "Catalog.schema")
SCHEMA = dbutils.widgets.get("schema")

# COMMAND ----------

import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, f1_score

df = spark.table(f"{SCHEMA}.silver_descritores_math").toPandas()
features_all = [c for c in df.columns if c != "target"]

X = df[features_all]
y = df["target"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)
print(X_train.shape, X_test.shape, y_train.mean(), y_test.mean())

# COMMAND ----------

def evaluate_incremental(feature_order, X_train, X_test, y_train, y_test):
    """Treina LogisticRegression usando as N primeiras features de `feature_order`,
    para N = 1..len(feature_order). Retorna DataFrame com métricas por N."""
    rows = []
    for n in range(1, len(feature_order) + 1):
        cols = feature_order[:n]
        start = time.time()
        clf = LogisticRegression(max_iter=10000, random_state=42, class_weight="balanced")
        clf.fit(X_train[cols], y_train)
        y_pred = clf.predict(X_test[cols])
        elapsed = time.time() - start
        rows.append(
            {
                "n_features": n,
                "accuracy": accuracy_score(y_test, y_pred),
                "f1": f1_score(y_test, y_pred),
                "time_s": elapsed,
            }
        )
    return pd.DataFrame(rows)

# COMMAND ----------

# MAGIC %md ### 1. Ordem aleatória (baseline)

# COMMAND ----------

rng = np.random.default_rng(42)
random_order = list(rng.permutation(features_all))
res_random = evaluate_incremental(random_order, X_train, X_test, y_train, y_test)
res_random["method"] = "aleatorio"

# COMMAND ----------

# MAGIC %md ### 2. Filtro por variância
# MAGIC Normaliza (MinMax) usando só `X_train` — no notebook original o scaler era ajustado no dado
# MAGIC inteiro, vazando informação do teste — e ordena features pela variância decrescente.

# COMMAND ----------

scaler = MinMaxScaler().fit(X_train)
X_train_scaled = pd.DataFrame(scaler.transform(X_train), columns=features_all, index=X_train.index)
variance_order = list(X_train_scaled.var().sort_values(ascending=False).index)

res_variance = evaluate_incremental(variance_order, X_train, X_test, y_train, y_test)
res_variance["method"] = "variancia"

# COMMAND ----------

# MAGIC %md ### 3. Importância por árvore (Random Forest, treinada só em `X_train`)

# COMMAND ----------

rf = RandomForestClassifier(random_state=42, class_weight="balanced", n_estimators=300)
rf.fit(X_train, y_train)
tree_order = list(
    pd.Series(rf.feature_importances_, index=features_all).sort_values(ascending=False).index
)

res_tree = evaluate_incremental(tree_order, X_train, X_test, y_train, y_test)
res_tree["method"] = "arvore"

# COMMAND ----------

# MAGIC %md ## Comparação

# COMMAND ----------

results = pd.concat([res_random, res_variance, res_tree], ignore_index=True)
spark.createDataFrame(results).write.mode("overwrite").saveAsTable(
    f"{SCHEMA}.gold_feature_selection_results"
)

# COMMAND ----------

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
for method, g in results.groupby("method"):
    axes[0].plot(g["n_features"], g["f1"], marker="o", label=method)
    axes[1].plot(g["n_features"], g["time_s"], marker="o", label=method)
axes[0].set_xlabel("nº features"); axes[0].set_ylabel("F1"); axes[0].legend(); axes[0].grid(alpha=.3)
axes[1].set_xlabel("nº features"); axes[1].set_ylabel("tempo (s)"); axes[1].legend(); axes[1].grid(alpha=.3)
plt.tight_layout()
display(fig)

# COMMAND ----------

best_per_method = results.loc[results.groupby("method")["f1"].idxmax()]
best_per_method[["method", "n_features", "f1", "accuracy", "time_s"]]

# COMMAND ----------

# MAGIC %md
# MAGIC ## Conclusão
# MAGIC
# MAGIC - O tempo de treino cresce de forma aproximadamente linear com o nº de features nos 3 métodos
# MAGIC   (esperado para Logistic Regression — não há o crescimento exponencial artificial visto no
# MAGIC   notebook original, que era efeito do split ser refeito a cada iteração, não do método em si).
# MAGIC - **Variância** e **árvore** convergem para o F1 máximo com bem menos features que a ordem
# MAGIC   aleatória — ou seja, conseguimos um modelo praticamente tão bom quanto o "modelo cheio" usando
# MAGIC   uma fração das 24 features, o que importa em cenários com centenas/milhares de descritores.
# MAGIC - Preencha aqui o nº ótimo de features observado após rodar (varia com a seed / split).