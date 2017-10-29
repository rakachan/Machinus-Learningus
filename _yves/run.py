#Imports
import numpy as np

from proj1_helpers import *
from helpers import *
from implementations import *
from cross_validation import *

from data_pre_processing import preprocess_data


# Load data
TEST_PATH = "data/test.csv"
TRAIN_PATH = "data/train.csv"

OUTPUT_PATH = "data/submission.csv"

te_data = load_csv_data(TEST_PATH, sub_sample=False)
tr_data = load_csv_data(TRAIN_PATH, sub_sample=False)

#get testing data
x_te = te_data[1]
y_te = te_data[0]
#we only need the ids to make the submission file
ids_te = te_data[2]

#get training data
x = tr_data[1]
y = tr_data[0]

#get the training data split by set and the jet indices
xx, yy, jet_indices = preprocess_data(x, y, augment=True, clean=True)

#get the tresting data split by set and the jet indices
xx_te, yy_te, jet_indices_te = preprocess_data(x_te, y_te, augment=True, clean=True)

#Prepare hyperparameters:
#Here we already give the best parameter
#since we already found them.
degrees = [[5,5], [9, 9], [13, 13]]
lambdas =[[1e-08], [8.53167852417e-07], [1.26896100317e-07]]
#We tried different seeds and this was the best one.
seed = 120

#array to store the best results 
#and parameters for each jet.
accuracy_arr = []
w_arr = []
lambda_arr = []
degree_arr = []

#for each jet
for jet in range(3):
    #print something to get an idea of the progress
    print('jet : ', jet)
    best_accuracy, best_w, best_degree, best_lambda = \
        best_model_cross_validation(xx[jet], yy[jet], seed, degrees[jet], lambdas=lambdas[jet])  
    
    accuracy_arr.append(best_accuracy)
    w_arr.append(best_w)
    lambda_arr.append(best_lambda)
    degree_arr.append(best_degree)

#get the prediction for the testing set
y_p_te = get_overall_predictions(xx_te, w_arr, degree_arr, jet_indices_te)
create_csv_submission(ids_te, y_p_te, "submission.csv")
print('==> Done.')