import numpy as np
import pandas as pd
from pyspark.sql import SparkSession
from pyspark.sql import DataFrame
from pyspark.storagelevel import StorageLevel
from pyspark.sql.functions import col
import pyspark.sql.types as pstype
import pyspark.sql.functions as F
import pyspark as ps
from statsmodels.tsa.vector_ar.var_model import VAR
from statsmodels.tsa.stattools import adfuller

ps_conf = ps.SparkConf().set("spark.logConf", "false")\
            .set("spark.executor.memory", "12g")\
            .set("spark.driver.memory", "4g")\
            .set("spark.executor.cores", "7")\
            .set("spark.sql.shuffle.partitions", "500")\
            .set("spark.executor.extraJavaOptions", "-XX:+UseG1GC -XX:+UseStringDeduplication")\
            .set("spark.eventLog.gcMetrics.youngGenerationGarbageCollectors", "G1 Young Generation")\
            .set("spark.eventLog.gcMetrics.oldGenerationGarbageCollectors", "G1 Old Generation")\
			.set("spark.logConf", "false")
spark = SparkSession.builder.config(conf=ps_conf).getOrCreate()

SPECIFIED_PATH = "csv_data/"
SPECIFIED_DATE = "20240918"
SPECIFIED_CSV  = SPECIFIED_PATH + SPECIFIED_DATE

input_path = SPECIFIED_CSV + "_c-united_config.csv"
df_config  = spark.read.option("inferSchema", "True").option("header", "True").csv(input_path)
df_config.persist(StorageLevel.MEMORY_AND_DISK_DESER)

utid_list = df_config.select("unit_id").drop_duplicates().rdd.flatMap(lambda x: x).collect()
spid_list = df_config.select("shop_id").drop_duplicates().rdd.flatMap(lambda x: x).collect()

#pos data 前処理
input_path  = SPECIFIED_CSV + "_pos_data_table.csv"
df_pos_data = spark.read.option('inferSchema', 'True').option('header', 'True').csv(input_path)\
				.select(
					"shop_id",
                    "レシートＮｏ．",
                    "商品種別",
                    "商品コード",
                    F.regexp_replace(col("商品名称（または券名称）"), "[ 　]", "").alias("商品名称（または券名称）"),
                    "オーダー時刻",
                    "単価",
                    "数量",
                    "合計金額",
                    "date"
				)\
				.filter(col("商品名称（または券名称）") != "")\
				.groupBy("shop_id", "date", "レシートＮｏ．").agg(
                    F.last("オーダー時刻").alias("オーダー時刻"),
                    F.sum(F.when(col("商品種別") == "Y", 1).otherwise(0)).alias("レシートあたりのセット商品の数"),
                    F.sum("数量").alias("総売上点数"),
                    F.sum("合計金額").alias("総売上"),
				)\
                .withColumn("レシートあたりのセット商品の数", F.when(col("レシートあたりのセット商品の数") == 0, 1)
                            								.otherwise(col("レシートあたりのセット商品の数")))\
                .withColumn("オーダー時刻", (F.col("オーダー時刻") / 100).cast("int"))\
                .withColumnRenamed("レシートあたりのセット商品の数", "来店者数")\
                .withColumnRenamed("オーダー時刻", "hour")
df_pos_data = df_pos_data.groupBy("shop_id", "date", "hour").agg(
                    F.sum("来店者数").alias("来店者数"),
                    F.sum("総売上点数").alias("総売上点数"),
                    F.sum("総売上").alias("総売上"),
				)\
                .select(["shop_id", "date", "hour", "来店者数", "総売上点数", "総売上"])\
                .orderBy(col("shop_id").asc(), col("date").asc(), col("hour").asc())

df_pos_data = df_pos_data\
    				.withColumn("date", F.from_unixtime(F.unix_timestamp("date") + F.col("hour") * 3600))\
                    .drop("hour")\
                    .orderBy(col("shop_id").asc(), col("date").asc())

df_pos_data = df_pos_data\
					.join(df_config.select(["shop_id", "caption"]), on="shop_id", how="inner")\
                    .select(["shop_id", "caption", "date", "来店者数", "総売上点数", "総売上"])\
                    .orderBy(col("shop_id").asc(), col("date").asc())

pd_pos_data  = df_pos_data.select(["shop_id", "caption", "date", "来店者数"]).toPandas()
pd_tmp_data1 = pd_pos_data[pd_pos_data["shop_id"] == 1189] # カフェ・ド・クリエグランサンシャイン通り店
pd_tmp_data2 = pd_pos_data[pd_pos_data["shop_id"] == 1616] # カフェ・ド・クリエ日比谷通り内幸町店
pd_tmp_data3 = pd_pos_data[pd_pos_data["shop_id"] == 1428] # カフェ・ド・クリエ札幌オーロラタウン店
pd_tmp_data4 = pd_pos_data[pd_pos_data["shop_id"] == 1550] # カフェ・ド・クリエ博多大博通店
pd_pos_data  = pd.merge(pd_tmp_data1, pd_tmp_data2, on="date", how="inner", suffixes=['_1', '_2'])
pd_pos_data  = pd.merge(pd_pos_data,  pd_tmp_data3, on="date", how="inner", suffixes=['_2', '_3'])
pd_pos_data  = pd.merge(pd_pos_data,  pd_tmp_data4, on="date", how="inner", suffixes=['_3', '_4'])
pd_pos_data  = pd_pos_data[["date", "来店者数_1", "来店者数_2", "来店者数_3", "来店者数_4"]]
pd_pos_data  = pd_pos_data.rename(columns={
    								"来店者数_1": "カフェ・ド・クリエグランサンシャイン通り店",
                                    "来店者数_2": "カフェ・ド・クリエ日比谷通り内幸町店",
                                    "来店者数_3": "カフェ・ド・クリエ札幌オーロラタウン店",
                                    "来店者数_4": "カフェ・ド・クリエ博多大博通店"
                                })
pd_pos_data  = pd_pos_data[[
    				"date",
                    "カフェ・ド・クリエグランサンシャイン通り店",
                    "カフェ・ド・クリエ日比谷通り内幸町店",
                    "カフェ・ド・クリエ札幌オーロラタウン店",
                    "カフェ・ド・クリエ博多大博通店"
                ]]
del pd_tmp_data1
del pd_tmp_data2
del pd_tmp_data3
del pd_tmp_data4

x_data = pd_pos_data[["カフェ・ド・クリエグランサンシャイン通り店", "カフェ・ド・クリエ日比谷通り内幸町店", "カフェ・ド・クリエ札幌オーロラタウン店", "カフェ・ド・クリエ博多大博通店"]].values.tolist()
x_train, x_test = x_data[0:3600], x_data[3600:]

# VARモデルの作成とフィット
var_model = VAR(x_test)

var_fit = var_model.fit(4)
var_model.select_order(maxlags=20)
IRF  = var_fit.irf(1)
FEVD = var_fit.fevd(30)

s = var_fit.sigma_u
p = np.linalg.cholesky(var_fit.sigma_u)
print(IRF.orth_irfs.round(4))
print((IRF.orth_irfs / np.diag(p)).round(4))
#result = adfuller(pd_pos_data["カフェ・ド・クリエグランサンシャイン通り店"].to_list(), autolag="BIC", regression="ct")
#print(result)
