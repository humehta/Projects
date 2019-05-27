import sys
import pandas as pd
import numpy as np
import os,random
from scipy.sparse.linalg import svds
from surprise import Reader, Dataset, SVD, evaluate
from surprise.model_selection import cross_validate,GridSearchCV
import time
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from pyspark import SparkContext,SparkConf
from pyspark.mllib.linalg import Vectors,Matrices
from pyspark.mllib.linalg.distributed import RowMatrix,CoordinateMatrix, MatrixEntry,BlockMatrix
from pyspark.sql import SparkSession
from pyspark.mllib.recommendation import ALS
import math
from sklearn.model_selection import train_test_split
from pyspark.sql import SQLContext
from pyspark import sql
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.functions import col,row_number
from memory_profiler import profile
from pyspark.mllib.util import MLUtils
from pyspark.mllib.evaluation import MulticlassMetrics

start = time.time()





if __name__ == "__main__":
	# set up Spark environment
	APP_NAME = "Collaboratove filtering for movie recommendation"
	conf = (SparkConf().set("spark.dynamicAllocation.enabled", "false").set("spark.shuffle.service.enabled", "false").setAppName(APP_NAME))
#	conf = conf.setMaster('spark://ukko160:7077')
	sc = SparkContext(conf=conf)
	sqlC = sql.SQLContext(sc)
	spark = SparkSession(sc)




def svd_python(final_ratings):
	reader = Reader()

	# get just top 100K rows for faster run time
	data = Dataset.load_from_df(final_ratings[['UserID', 'MovieID', 'Rating']], reader)
	#data.split(n_folds=3,c)

	# Use the famous SVD algorithm.
	algo = SVD()
	#print (evaluate(algo, data, measures=['RMSE', 'MAE']))
	#Run 5-fold cross-validation and print results.
	cross_validate(algo, data, measures=['RMSE', 'MAE'], cv=5, verbose=True)

	param_grid = {'n_epochs': [5, 10], 'lr_all': [0.002, 0.005],'reg_all': [0.4, 0.6]}
	gs = GridSearchCV(SVD, param_grid, measures=['rmse', 'mae'], cv=3)
	gs.fit(data)
	# best RMSE score
	print("Best RMSE Score:",best_score['rmse'])
	results_df = pd.DataFrame.from_dict(gs.cv_results)
	print ("Results:",results_df)
	# combination of parameters that gave the best RMSE score
	print("Best Parameters::",gs.best_params['rmse'])
	print("Detailed_Results-Stored in svd.txt")
	results_df.to_csv('svd.txt', header=True, index=False, sep='\t', mode='a')












def spark_svd(pivot_table,row_index,headers):
	global preds
	start = time.ctime()
	pivot_table[np.isnan(pivot_table)] = 0
	data = []
	for i in range(pivot_table.shape[0]):
		nonzeroind = np.nonzero(pivot_table[i,:])[0] # the return is a little funny so I use the [0]
		kiki = list(pivot_table[i,:][nonzeroind])
		data.append(Vectors.sparse(pivot_table.shape[1], nonzeroind, kiki))
	partitionNum = 1000
	rows = sc.parallelize(data,partitionNum)
	mat = RowMatrix(rows)

	# Compute the top 2 singular values and corresponding singular vectors.
	svd = mat.computeSVD(3, computeU=True)
	U = svd.U       # The U factor is a RowMatrix.
	s = svd.s       # The singular values are stored in a local dense vector.
	V = svd.V       # The V factor is a local dense matrix.
	s = np.diag(s) 
	s = Matrices.dense(s.shape[0],s.shape[1],s.reshape((s.shape[0]*s.shape[1])))

	x = U.multiply(s)
	V = V.toArray()
	V = Matrices.dense(V.shape[1],V.shape[0],V.reshape((V.shape[0]*V.shape[1],1)))
	
	fin = x.multiply(V)
	fifi = fin.rows
	gg = fin.rows.map(lambda x: (x, )).zipWithIndex().toDF()
	print("pivot banaya")
	
	testing = np.random.choice(len(row_index), 500, replace=False)
	print(testing.shape)
	
	print("Start time :::",start)
	print("End Time :::",time.ctime())
	sys.exit(0)
	right = 0
	for i in testing:
		pr = np.abs(np.round(np.array(gg.where(gg._2 == i).select('_1').take(1))[0][0][0]))
		if np.array_equal(pivot_table[i,:],pr):
			right += 1
	print("Accuracy: ",float(right)/len(testing)*100)
	print("Start time :::",start)
        print("End Time :::",time.ctime())
	sqlC.clearCache()
	sc.stop()	
	


def test_ALS(ratings_data,best_rank):
	seed = 5L
        iterations = 10
        regularization_parameter = 0.1
        rank = best_rank
        errors = [0, 0, 0]
        err = 0
        tolerance = 0.02
	print "There are %s recommendations in the complete dataset" % (ratings_data.count())
	training_RDD, test_RDD = ratings_data.randomSplit([7, 3], seed=0L)

	complete_model = ALS.train(training_RDD, rank, seed=seed, 
				   iterations=iterations, lambda_=regularization_parameter)

	test_for_predict_RDD = test_RDD.map(lambda x: (x[0], x[1]))

	predictions = complete_model.predictAll(test_for_predict_RDD).map(lambda r: ((r[0], r[1]), r[2]))
	rates_and_preds = test_RDD.map(lambda r: ((int(r[0]), int(r[1])), float(r[2]))).join(predictions)
	error = math.sqrt(rates_and_preds.map(lambda r: (r[1][0] - r[1][1])**2).mean())
	    
	print 'For testing data the RMSE is %s' % (error)
	sc.stop()
	sys.exit(0)









def implement_ALS(traing_RDD,validation_RDD,test_RDD,validation_for_predict_RDD,test_for_predict_RDD):
	seed = 5L
	iterations = 10
	regularization_parameter = 0.1
	ranks = [1,2,3,4]
	errors = [0, 0, 0]
	err = 0
	tolerance = 0.02

	min_error = float('inf')
	best_rank = -1
	best_iteration = -1
	for rank in ranks:
	    model = ALS.train(training_RDD, rank, seed=seed, iterations=iterations,
			      lambda_=regularization_parameter)
	    predictions = model.predictAll(validation_for_predict_RDD).map(lambda r: ((r[0], r[1]), r[2]))
            rates_and_preds = validation_RDD.map(lambda r: ((int(r[0]), int(r[1])), float(r[2]))).join(predictions)
	    error = math.sqrt(rates_and_preds.map(lambda r: (r[1][0] - r[1][1])**2).mean())
	    errors[err] = error
	    err += 1
	    print ('For rank %s the RMSE is %s' % (rank, error))
	    if error < min_error:
		min_error = error
		best_rank = rank
	print ('The best model was trained with rank %s' % best_rank)
	return (best_rank)	




method = sys.argv[1]
als_type = sys.argv[2]


write_data = True
if write_data:
	count = 0
	list_of_files = os.listdir("Input_Data")
	for filename in list_of_files[0:2]:
	    count += 1
            print("Pre-processing files!!!!")
	    print ("Filename - ",filename)
	    with open("Input_Data/" + filename) as myfile:
		file = myfile.readlines()
	    ratings = []
	    for line in file:
		line = line.strip()
		if ":" in line:
		    movie_id = line.split(":")[0]
		    continue
		line = line.split(",")
		ratings.append([line[0],movie_id,line[1]])

	    ratings_df = pd.DataFrame(ratings, columns = ['UserID', 'MovieID', 'Rating'], dtype = int)
	    if count == 1:
		final_ratings = ratings_df.copy()
	    else:
		final_ratings = pd.concat([final_ratings,ratings_df])
	final_ratings = final_ratings.sort_values("UserID")
	if method == "SVD_Python":
		svd_python(final_ratings)
	if method == "ALS":
		final_mat = final_ratings.values
		with open("sparse_data.csv",'w') as d:
			np.savetxt(d,final_mat,delimiter=",",fmt='%i,%i,%i')

	if method == "Spark_SVD":
		final_mat = final_ratings.pivot(index = "UserID",columns="MovieID",values="Rating").values
		row_index = final_ratings['UserID'].unique().tolist()
		headers = final_ratings['MovieID'].unique().tolist()
		spark_svd(final_mat,row_index,headers)	

if method == "ALS":	
	# read input text file to RDD
	rat_data = sc.textFile("file:///sparse_data.csv")

	ratings_data = rat_data.map(lambda line: line.split(",")).map(lambda tokens: (tokens[0],tokens[1],tokens[2])).cache()

	training_RDD, validation_RDD, test_RDD = ratings_data.randomSplit([6, 2, 2], seed=0L)
	validation_for_predict_RDD = validation_RDD.map(lambda x: (x[0], x[1]))
	test_for_predict_RDD = test_RDD.map(lambda x: (x[0], x[1]))
	if als_type == "cross_validate_ALS":
		start = time.ctime()
		best_rank = implement_ALS(training_RDD,validation_RDD,test_RDD,validation_for_predict_RDD,test_for_predict_RDD)
		test_ALS(ratings_data,best_rank)
		print("Start:",start)
		end = time.ctime()
		print ("Time:",end)
	else:
		start = time.ctime()
		test_ALS(ratings_data,3)
		print("Start Time :: ",start)
		end = time.ctime()
		print ("End Time:",end)
		exit()










