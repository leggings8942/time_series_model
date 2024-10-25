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
from time_series_model import Dickey_Fuller_Test

SPECIFIED_PATH = "csv_data/"
SPECIFIED_DATE = "20240918"
SPECIFIED_CSV  = SPECIFIED_PATH + SPECIFIED_DATE

pd_data = pd.read_csv(SPECIFIED_CSV + "_urp_data.csv")
result = adfuller(pd_data.to_numpy(), maxlag=1, autolag=None, regression="ct")

print("ADF Statistic: %f" % result[0])
print("p-value: %f" % result[1])
print("used lag: %d" % result[2])
print("data num: %d" % result[3])
for key, value in result[4].items():
    print('\t%s: %.3f' % (key, value))
#print("ic best: %f" % result[5])


dfruler = Dickey_Fuller_Test(pd_data, regression="ct")
dfruler.fit()
result = dfruler.dfRuller(qlist=[1, 5, 10])

print("DF Statistic: ", result[0])
print("p-value: ", result[1])
print("used lag: ", result[2])
print("data num: ", result[3])
for key, value in result[4].items():
    print('\t%s: %.3f' % (key, value))

