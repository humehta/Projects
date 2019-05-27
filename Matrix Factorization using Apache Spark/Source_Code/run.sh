#!/bin/bash

numExecutor=10
coresPerWorker=6
memExecutor=20G


spark-submit --deploy-mode client --conf spark.driver.maxResultSize=15g --driver-memory 20g  --num-executors ${numExecutor} --executor-cores ${coresPerWorker} --executor-memory ${memExecutor} ./classify.py $1 $2

