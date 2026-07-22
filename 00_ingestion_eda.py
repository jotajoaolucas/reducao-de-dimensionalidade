# Databricks notebook source
# MAGIC %md
# MAGIC # 00 · Ingestão e EDA
# MAGIC
# MAGIC Projeto: **Comparação de Técnicas de Redução de Dimensionalidade**
# MAGIC (recriação de um projeto original do MBA em Ciência de Dados, com dados e metodologia revisados).
# MAGIC
# MAGIC Dois datasets, dois cenários de dimensionalidade:
# MAGIC
# MAGIC | Dataset | Linhas | Features | Cenário |
# MAGIC |---|---|---|---|
# MAGIC | `DESCRITORESMATH.csv` | ~2.472 | 24 | baixa dimensionalidade → **seleção de features** |
# MAGIC | `Swarm_Behaviour.csv` | ~23.309 | 2.400 | altíssima dimensionalidade → **métodos de projeção** |
# MAGIC
# MAGIC Suba os dois arquivos em um Volume do Unity Catalog antes de rodar (ajuste o widget abaixo).

# COMMAND ----------

dbutils.widgets.text("volume_path", "/Volumes/reducao_dimensionalidade/default/raw_data", "Volume path (raw CSVs)")
dbutils.widgets.text("schema", "reducao_dimensionalidade.default", "Catalog.schema para as tabelas Delta")

VOLUME_PATH = dbutils.widgets.get("volume_path")
SCHEMA = dbutils.widgets.get("schema")

spark.sql(f"CREATE SCHEMA IF NOT EXISTS {SCHEMA}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Ingestão → Bronze (Delta)

# COMMAND ----------

import re


def sanitize_columns(sdf):
    """Delta não aceita espaço/caractere especial em nome de coluna. Troca por '_'."""
    renamed = sdf
    for c in sdf.columns:
        clean = re.sub(r"[ ,;{}()\n\t=]+", "_", c).strip("_")
        if clean != c:
            renamed = renamed.withColumnRenamed(c, clean)
    return renamed

# COMMAND ----------

df_desc_raw = sanitize_columns(
    spark.read.option("header", True).option("inferSchema", True)
    .csv(f"{VOLUME_PATH}/DESCRITORESMATH.csv")
)
df_desc_raw.write.mode("overwrite").saveAsTable(f"{SCHEMA}.bronze_descritores_math")

df_swarm_raw = sanitize_columns(
    spark.read.option("header", True).option("inferSchema", True)
    .csv(f"{VOLUME_PATH}/Swarm_Behaviour.csv")
)
df_swarm_raw.write.mode("overwrite").saveAsTable(f"{SCHEMA}.bronze_swarm_behaviour")

print("descritores:", df_desc_raw.count(), len(df_desc_raw.columns))
print("swarm:", df_swarm_raw.count(), len(df_swarm_raw.columns))
print(df_desc_raw.columns)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. EDA — DESCRITORESMATH (desempenho escolar, Ceará)
# MAGIC
# MAGIC 24 descritores numéricos (notas por competência) + `Indicação do Padrão de Desempenho`
# MAGIC (4 classes) + identificadores de `Município`/`Escola`.

# COMMAND ----------

import pandas as pd 

pdf_desc = df_desc_raw.toPandas()
display(pdf_desc.describe())

# COMMAND ----------

pdf_desc["Indicação_do_Padrão_de_Desempenho"].value_counts()

# COMMAND ----------

# MAGIC %md
# MAGIC **Achado importante:** a classe `Adequado` tem só 33 registros em 2.472 (1,3%) — desbalanceamento
# MAGIC severo se tratarmos como problema de 4 classes. Assim como no projeto original, vamos binarizar
# MAGIC (`Crítico`/`Muito Crítico` → 0, `Intermediário`/`Adequado` → 1), mas diferente do original, vamos:
# MAGIC - manter o desbalanceamento resultante (~86% / ~14%) documentado e visível;
# MAGIC - usar **F1 e recall da classe minoritária** como métricas principais, não só acurácia;
# MAGIC - usar `class_weight="balanced"` nos modelos.

# COMMAND ----------

def to_binary(x):
    return 0 if x in ("Crítico", "Muito Crítico") else 1

pdf_desc["target"] = pdf_desc["Indicação_do_Padrão_de_Desempenho"].apply(to_binary)
pdf_desc["target"].value_counts(normalize=True)

# COMMAND ----------

pdf_desc_clean = pdf_desc.drop(columns=["Município", "Escola", "Indicação_do_Padrão_de_Desempenho"])
pdf_desc_clean.isna().sum().sum()  # checar nulos

# COMMAND ----------

spark.createDataFrame(pdf_desc_clean).write.mode("overwrite").saveAsTable(
    f"{SCHEMA}.silver_descritores_math"
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. EDA — Swarm Behaviour (locomoção de grupos de animais)
# MAGIC
# MAGIC 200 "boids" simulados × 12 atributos cinemáticos cada (posição, velocidade, aceleração,
# MAGIC separação/coesão/alinhamento, nº de vizinhos) = 2.400 features + `Swarm_Behaviour` (0/1).

# COMMAND ----------

pdf_swarm = df_swarm_raw.toPandas()
pdf_swarm["Swarm_Behaviour"].value_counts(normalize=True)

# COMMAND ----------

# MAGIC %md
# MAGIC Desbalanceamento moderado (~66% / ~34%), bem menos crítico que o dataset escolar — não precisa
# MAGIC de reamostragem agressiva como no notebook original (que reduzia o dado para 20% e depois
# MAGIC balanceava por subamostragem, jogando fora dado real sem necessidade).

# COMMAND ----------

pdf_swarm.isna().sum().sum()

# COMMAND ----------

spark.createDataFrame(pdf_swarm).write.mode("overwrite").saveAsTable(
    f"{SCHEMA}.silver_swarm_behaviour"
)