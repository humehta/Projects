#/bin/bash

#numExecutor=10
#coresPerWorker=6
#memExecutor=20G

#data=/data/kmeans-P200000-D10.txt
#centroids=10
#iteration=10
spark-submit --master local[*] --packages graphframes:graphframes:0.6.0-spark2.3-s_2.11 --driver-class-path /home/hduser/.ivy2/jars/com.typesafe.scala-logging_scala-logging-api_2.11-2.1.2.jar:/home/hduser/.ivy2/jars/com.typesafe.scala-logging_scala-logging-slf4j_2.11-2.1.2.jar ./lpa.py

