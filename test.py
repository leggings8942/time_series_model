import datetime
import numpy as np
import pandas as pd
from dateutil.relativedelta import relativedelta
from tqdm.notebook import tqdm
from tqdm.contrib import tzip, tenumerate, tmap

from pyspark.sql import SparkSession
from pyspark.sql import DataFrame
from pyspark.storagelevel import StorageLevel
from pyspark.sql.functions import col
import pyspark.sql.types as pstype
import pyspark.sql.functions as F
import pyspark as ps

import matplotlib as mlt
import matplotlib.pyplot as plt
import japanize_matplotlib

from time_series_model import *

from sklearn.linear_model import ElasticNet

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
SPECIFIED_DATE = "20241120"
SPECIFIED_CSV  = SPECIFIED_PATH + SPECIFIED_DATE
input_path  = SPECIFIED_CSV + "_raw_data_sumida_bunka.csv"
df_raw_data = spark.read.option("inferSchema", "True").option("header", "True").csv(input_path)
df_raw_data.persist(StorageLevel.MEMORY_AND_DISK_DESER)

utid_list = sorted(df_raw_data.select("unit_id").drop_duplicates().rdd.flatMap(lambda x: x).collect())

SPECIFIED_PAST = (datetime.datetime.strptime(SPECIFIED_DATE, '%Y%m%d') - relativedelta(days=1) + datetime.timedelta(hours=9)).strftime('%Y-%m-%d')

start_time = f'{SPECIFIED_PAST} 00:00:00'
end_time = f'{SPECIFIED_PAST} 23:59:00'
# 1分単位の時間列を作成
time_range = pd.date_range(start=start_time, end=end_time, freq='min')
# DataFrameに変換
df_time = pd.DataFrame(time_range, columns=['datetime'])
df_time = spark.createDataFrame(df_time)\
				.withColumn('hour', F.hour(col('datetime')))

df_by1min = spark.read\
				.option('header', True)\
				.option('inferSchema', True)\
           		.csv(SPECIFIED_CSV + "_1min_data_sumida_bunka.csv")

for unit_id in utid_list:
    df_tmp  = df_by1min\
        		.filter(col('unit_id') == unit_id)\
          		.select(['minute', '1min_count'])\
            	.withColumnRenamed('minute',     'datetime')\
                .withColumnRenamed('1min_count', unit_id)
    df_time = df_time.join(df_tmp, on='datetime', how='left')
df_time = df_time\
			.fillna(0)\
    		.orderBy('datetime')
df_time.persist(StorageLevel.MEMORY_AND_DISK_DESER)

SPECIFIED_HOUR = 13

pd_data = df_time.toPandas()
pd_data = pd_data[pd_data['hour'] == SPECIFIED_HOUR]
pd_data = pd_data[utid_list]
pd_data

test_data = pd_data.values.tolist()
test_data

model = ElasticNet(alpha=1, l1_ratio=0.1, max_iter=1000000, tol=1e-6)

lag   = 4
x_tmp = np.array([np.array(test_data)[t-lag : t][::-1].ravel() for t in range(lag, len(test_data))])
y_tmp = test_data[lag:]
model.fit(x_tmp, y_tmp)
mean  = model.predict(x_tmp)

mse = np.sum((y_tmp - mean) ** 2) / len(y_tmp)
mse

