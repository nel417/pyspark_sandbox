import pyspark
import pandas as pd
from pyspark.sql import SparkSession
from pyspark.ml.feature import Imputer

df = pd.read_csv('test.csv')

spark = SparkSession.builder.appName('sparksandbox').getOrCreate()

# pyspark_data = spark.read.csv('test.csv')

# cleaner version. use .show() instead to show tables and columns
pyspark_data = spark.read.option('header', 'true').csv('test.csv', inferSchema=True)

# types
# print(type(pyspark_data))
# # first x items
# print(pyspark_data.head(5))
# # print schema
# print(pyspark_data.printSchema())
################################################################

# print(pyspark_data.columns)
# print(pyspark_data.select('experience').show())
# print(pyspark_data.select(['name', 'experience']).show())
# print(pyspark_data.dtypes)
# print(pyspark_data.describe().show())
################################################################

# pyspark_data = pyspark_data.withColumn('Raise', pyspark_data['salary'] * 2)
# print(pyspark_data.show())
# print(pyspark_data.printSchema())
#
# pyspark_data = pyspark_data.drop('raise')
#
# print(pyspark_data.printSchema())
################################################################
# pyspark_data = pyspark_data.withColumnRenamed('name', 'employeeName')
pyspark_data_missing_data = spark.read.csv('test_missingValues.csv', header=True, inferSchema=True)

print(pyspark_data_missing_data.show())

# drop nulls
# how = any or all
# threshold is drop if x null values exist
# subset drop null values from specified column
# pyspark_data_cleaned = pyspark_data_missing_data.na.drop()
# print(pyspark_data_cleaned.show())

# pyspark_data_missing_data.nam.fill('stuff to fill in missing').show()
# pyspark_data_missing_data.nam.fill('stuff to fill in missing', ['column', 'column']).show()

# moving back dirty data and replace missing with mean

imputed_data = Imputer(
    inputCols=['age', 'experience', 'salary'],
    outputCols=["{}_imputed_data".format(c) for c in ['age', 'experience', 'salary']],
).setStrategy('mean')

new_clean_data = \
    imputed_data.fit(pyspark_data_missing_data).transform(pyspark_data_missing_data)

print(new_clean_data.show())
