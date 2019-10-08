from pyspark.sql import SparkSession
from pyspark.ml.feature import StandardScaler
from pyspark.ml.linalg import Vectors
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.clustering import KMeans
import sys
import os

#Creating Spark Session
spark = SparkSession.builder.appName('hacking-analysis').getOrCreate()

#Getting K as variable from user
n = int(sys.arg[1])

#Getting Current Path
cwd = os.getcwd()
path = 'file:///'+cwd+'hack.csv'

#Reading data From HDFS
data = spark.read.csv(path, header=True, inferSchema=True)

#Renaming the Column To Match File
data = data.withColumnRenamed('Bytes Transferred', 'Bytes_Transferred')

cols = ['Session_Connection_Time', 'Bytes Transferred', 'Kali_Trace_Used', 'Servers_Corrupted', 
'Pages_Corrupted', 'WPM_Typing_Speed']

#Assembling The Features
assembler = VectorAssembler(inputCols=cols, outputCol='features')

#Creating the new Dataframe with Features
assembled_data = assembler.transform(data)

#Scaling the Features
scaler = StandardScaler(inputCol='features', outputCol='scaledFeatures')

scaler_model = scaler.fit(assembled_data)

scaled_data = scaler_model.transform(assembled_data)

#Creating the Model
k_means = KMeans(featuresCol='scaledFeatures', k=n)

#Training The Model
model = k_means.fit(scaled_data)

#Prediction
model_data = model.transform(scaled_data)

#Grouping and Displaying By Cluster
model_data.groupBy('prediction').count().show()
