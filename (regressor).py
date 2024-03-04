# coding: utf-8
regressor.save(tips)
from pyspark.sql import SparkSession
spark=SparkSession.builder.appName('Lr').getOrCreate()
# File location and type
file_location = "tips.csv"
file_type = "csv"


# The applied options are for CSV files. For other file types, these will be ignored.
df = spark.read.csv(file_location,header=True,inferSchema=True)
# File location and type
file_location = "tip/tips.csv"
file_type = "csv"


# The applied options are for CSV files. For other file types, these will be ignored.
df = spark.read.csv(file_location,header=True,inferSchema=True)
# Create a view or table
df.show()
df.printSchema()
df.columns
### Handling Categorical Features
from pyspark.ml.feature import StringIndexer
indexer = StringIndexer(inputCol="sex",outputCol="sex_index")
df_r=indexer.fit(df).transform(df)
df_r.show()
indexer = StringIndexer(inputCols=["smoker","day","time"],outputCols=["smoker_index","day_index","time_index"])
df_r=indexer.fit(df_r).transform(df_r)
df_r.show()
df_r.columns
#vectorassembler
from pyspark.ml.feature import VectorAssembler
featureassembler=VectorAssembler(inputCols=['tip','size','sex_index','smoker_index','day_index','time_index'], outputCol="Independent Features")
output = featureassembler.transform(df_r)
output.select('Independent Features').show()
finalized_data = output.select("Independent Features", "total_bill")
finalized_data.show()
#inplement ML Linear Regression 
from pyspark.ml.regression import LinearRegression
#train test split 
train_data,test_data=finalized_data.randomSplit([0.75,0.25])
regressor=LinearRegression(featuresCol="Independent Features", labelCol="total_bill")
regressor=regressor.fit(train_data)
regressor.coefficients
regressor.intercept
#prediction
pred_results=regressor.evaluate(test_data)
#final comparition
pred_results.predictions.show()
pred_results.r2
#performance metrics
pred_results.meanAbsoluteError, pred_results.meanSquaredError
regressor.save(tips)
regressor.save("tips")
