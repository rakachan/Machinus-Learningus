import re
import numpy as np
import scipy
import scipy.sparse as sp

import findspark
findspark.init()

import pyspark
sc = pyspark.SparkContext(master="local[3]", appName="ML project 2")

from pyspark.mllib.recommendation import ALS, MatrixFactorizationModel, Rating

from helpers import row_col_spark, load_data, create_csv_submission

r = re.compile(r'r(\d+)_c(\d+)')
	
# Load and parse the data
data2 = sc.textFile("data_train.csv")
header = data2.first() #extract header
data2 = data2.filter(lambda row: row != header) 

ratings = data2.map(lambda l: l.split(','))
ratings = ratings.map(lambda l: Rating(*row_col_spark(l[0], r), float(l[1])))
train2, test2 = ratings.randomSplit([0.9, 0.1], seed=4242)

#Need to cache the data to speed up training
train2.cache()
test2.cache()

sc.setCheckpointDir('checkpoint/')

l_s = 0.09
r_s = 100

print("""
============================
lambda = {}   rank = {}
============================
""".format(l_s, r_s))

# Build the recommendation model using Alternating Least Squares
rank = r_s
numIterations = 250
lambda_spark = l_s
model = ALS.train(train2, rank, numIterations, lambda_=lambda_spark, nonnegative=True, seed=459832632)

#TRAIN RMSE
train_input = train2.map(lambda x:(x[0],x[1]))   
pred_train = model.predictAll(train_input) 

true_train = train2.map(lambda x:((x[0],x[1]), x[2]))
pred_train = pred_train.map(lambda x:((x[0],x[1]), x[2]))

true_pred = true_train.join(pred_train)

MSE_train = true_pred.map(lambda r: (r[1][0] - r[1][1])**2).mean()
RMSE_train = np.sqrt(MSE_train)

#TEST RMSE
test_input = test2.map(lambda x:(x[0],x[1])) 
pred_test = model.predictAll(test_input)

true_test = test2.map(lambda x:((x[0],x[1]), x[2]))
pred_test = pred_test.map(lambda x:((x[0],x[1]), x[2]))

true_pred = true_test.join(pred_test)

MSE_test = true_pred.map(lambda r: (r[1][0] - r[1][1])**2).mean()
RMSE_test = np.sqrt(MSE_test)

print("Train rmse : ", RMSE_train)
print("Test rmse : ", RMSE_test)

# Generate the submission
testdata = sc.textFile("sample_submission.csv")
testheader = testdata.first() #extract header
testdata = testdata.filter(lambda row: row != testheader) 
testdata = testdata.map(lambda l: l.split(','))
testdata = testdata.map(lambda l: (row_col_spark(l[0], r)))

predictions = model.predictAll(testdata).map(lambda r: ((r[0], r[1]), r[2]))
pred = predictions.collect()

matrix_pred = sp.dok_matrix((10000, 1000), dtype=np.float32)

for row in pred:
    matrix_pred[row[0][0], row[0][1]] = row[1]

path_dataset2 = "sample_submission.csv"
sub_ex = load_data(path_dataset2)	
	
create_csv_submission(list(zip(*sub_ex.nonzero())), matrix_pred, 'submission.csv')