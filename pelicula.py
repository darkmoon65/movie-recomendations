import pandas as pd
from pyspark.sql import SparkSession
from pyspark.ml.recommendation import ALS
from pyspark.ml.feature import StringIndexer
from pyspark.ml.evaluation import RegressionEvaluator

#####################
# Create a Spark session
spark = SparkSession.builder.appName("CSV Reader").getOrCreate()

# Load the dataset
data_path = "data.zip"
pdf = pd.read_csv('data.zip')

df = spark.createDataFrame(pdf)

#####################
# Seleccionamos los campos que nos interesan

df = df.select("userId", "movieId", "rating")

# Casteamos las columnas
df = df.withColumn("rating", df["rating"].cast("float"))
df = df.withColumn("movieId", df["movieId"].cast("int"))
df = df.withColumn("userId", df["userId"].cast("int"))

# Filtramos
df = df.filter(df["userId"].isNotNull() & df["movieId"].isNotNull() & df["rating"].isNotNull())


# Split the data into training and testing sets (80% for training, 20% for testing)
train_data, test_data = df.randomSplit([0.8, 0.2], seed=42)
print(test_data.head(3))


# Create StringIndexers for the user and movie columns
user_indexer = StringIndexer(inputCol="userId", outputCol="userIndex")
movie_indexer = StringIndexer(inputCol="movieId", outputCol="movieIndex")

# Fit StringIndexers and transform the data
indexed_data = user_indexer.fit(train_data).transform(train_data)
indexed_data_final = movie_indexer.fit(indexed_data).transform(indexed_data)

# Create an ALS recommender model
als = ALS(userCol="userIndex", itemCol="movieIndex", ratingCol="rating", nonnegative=True)

# Fit the model to the training data
model = als.fit(indexed_data_final)

user_indexer = StringIndexer(inputCol="userId", outputCol="userIndex")
movie_indexer = StringIndexer(inputCol="movieId", outputCol="movieIndex")

# Fit StringIndexers and transform the data
indexed_test_data = user_indexer.fit(test_data).transform(test_data)
indexed_test_data = movie_indexer.fit(indexed_test_data).transform(indexed_test_data)

#Probamos el modelo mediante RMSE
predictions = model.transform(indexed_test_data)
evaluator = RegressionEvaluator(metricName="rmse", labelCol="rating", predictionCol="prediction")
rmse = evaluator.evaluate(predictions)

# Generamos top 3 recomendaciones para cada usuario en el test data
recommendations = model.recommendForUserSubset(indexed_test_data, 3)
print("Generando 5 recomendaciones por usuario")
print(recommendations.show(10))
print("Error Cuadratico= " + str(rmse))
