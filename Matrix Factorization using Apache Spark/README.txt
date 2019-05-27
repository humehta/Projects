We have three implementations for our project. Below are the instructions to run them on your machine.

1.) To implement the Python Singular Value Decomposition Non-Ditributed Implementation( Takes a lot of time to run because non-distributed) run the follwing command:

	Command :: ./run.sh SVD_Python None


2.) To implement the Spark Alternating Least Squares Implementation run the follwing command:

	To run the ALS implementation with Cross-Validation and Testing:

	Command :: ./run.sh ALS cross_validate_ALS


	To run the ALS implementation with only Final Model:

	Command :: ./run.sh ALS test



3.) To implement the Spark Singular Value Decompositon Implementation run the follwing command:

	Command :: ./run.sh Spark_SVD None
