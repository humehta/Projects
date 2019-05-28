from pyspark import sql
from pyspark import SparkContext,SparkConf
from pyspark.sql import SparkSession
from graphframes import *
from neo4j.v1 import GraphDatabase
import pandas as pd


df = pd.read_csv('kaka.csv',sep='\t')
print(df['id'].nunique())
exit()


##### Creates the Spark Context###

if __name__ == "__main__":

	spark = SparkSession.builder.appName('LPA Graphframes').getOrCreate()
        conf = SparkConf()
        sqlc = sql.SQLContext(spark)

### Neo4j Driver that connects to Neo4j Database ####
driver = GraphDatabase.driver('bolt://localhost:7687', auth=('neo4j', 'test'))


ne_dict = {}

#### This Method Loads the your graph data into Neo4j ####

load_neo = False ### True to execute this method else False ####
if load_neo:
	data = open("facebook_graph.txt","r")
	for i in data:
		if not i.strip():
			break
		n = i.split()
		if n[0] not in ne_dict:
			ne_dict[n[0]] = []
		ne_dict[n[0]].append(n[1])

	node_tuples=[]
	for node_id in ne_dict.keys():
		node_tuples.append((node_id,'1'))
		cmd = 'CREATE (n:Person {name: $name})'
		with driver.session() as session:
			session.run(cmd, name = node_id)

	node_edge = []
	for key,value in ne_dict.items():
		for k in value:
			node_edge.append((key,k,'1'))	
			cmd = 'MERGE (n_1:Person {name : $k1}) MERGE (n_2:Person {name : $k2}) MERGE (n_1)<-[r:HAS_CONNNECTION]-(n_2);'
			with driver.session() as session:
				session.run(cmd, k1 = key,k2 = k)


### This fetches each record from Neo4j Database ####
cmd = 'MATCH (n1:Person) - [r:HAS_CONNNECTION] -> (n2:Person) RETURN n1,n2;'
with driver.session() as session:
	query_results = session.run(cmd)

#### You can directly use the query_results variable above to load into Graphframes I have below saved the query_results into data.txt to save time ######

##### The following tajes each record from Neo4j query_results and loads into Graphframes dataframe ######
node_tuples = []
node_edge = []
with open("data.txt","r") as f:
	for i in range(15000):
		l = f.next().strip().split(",")
		node_tuples.append((l[0],1))
		node_tuples.append((l[1],1))
		node_edge.append((l[0],l[1],'has_connection'))
v = sqlc.createDataFrame(node_tuples,["id","node_weight"])
e = sqlc.createDataFrame(node_edge,['src','dst','edge_wt'])
g = GraphFrame(v,e)

verticesDF=g.vertices 
edgesDF=g.edges 

verticesDF.show()
result = g.labelPropagation(maxIter=5)
result.sort(['label'],ascending=[0])
### The below line saves the output to a csv file ####
result.toPandas().to_csv("kaka.csv", header=True,sep="\t")
