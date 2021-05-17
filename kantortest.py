#Note: : decision trees do not require normalization of their inputs; 
# since XGBoost is like an ensemble algorithm comprised of decision trees,
# it does not require normalization for the inputs either.

# count of each non A-Z a-z character (ie. special character) in file (use ascii) 
# should probably exclude linefeeds and spaces as features as these are irrelevant
# count of each token in the file

import sqlite3
import pandas as pd
import random
import numpy as np


import xgboost as xgb
from xgboost import XGBClassifier
from skopt import gp_minimize
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
sql_connect = sqlite3.connect('snippets-dev/snippets-dev.db')
cursor = sql_connect.cursor()
query = "select language, snippet FROM snippets;"
results = cursor.execute(query).fetchmany(size=10000)



def train_test_split(percentage_traindata, data):
	random.shuffle(data)
	c = int(percentage_traindata * len(data))
	train = []
	for i in range(0, c):
		train.append(data[i])
	test = []
	for i in range(c, len(data)):
		test.append(data[i])

	return train, test


train_raw, test_raw = train_test_split(0.85, results)

def tokenise(codestring):
	#Create list containing all tokens in the code :)
	normalChars = "abcdefghijklmnopqrstuvwxyz"
	normalChars = normalChars + normalChars.upper()
	curToken = ""
	tokenList = []
	for c in codestring:
		if c in normalChars:
			normaltoken = True
			for tc in curToken:
				if tc not in normalChars:
					normaltoken = False
			if normaltoken:
				curToken = curToken + c
			else:
				#new token!
				if curToken != "":
					tokenList.append(curToken)
				curToken = c
		else:
			if curToken != "":
				tokenList.append(curToken)
			curToken = c
	if curToken != "":
		tokenList.append(curToken)
	return tokenList


#curate list of all languages using all data - this is fair practice
#Note: 'UNKNOWN' means we don't know which one it is and that is included in the data

print ("1. Retrieve list of all labels (programming languages)")
knownLabels = set([])
for (y_raw, x_raw) in results:
	knownLabels.add(y_raw)
knownLabels = list(knownLabels)
print (knownLabels)

label_to_int = {}

for i in range(0, len(knownLabels)):
	curToken = knownLabels[i]
	label_to_int[curToken] = i



print ("2. Getting all tokens (incl special characters) in training data")

#Collect and form a list of known tokens!
#Note this should include special characters as tokens
#To maintain fairness in evaluation, we only include tokens in training data.

knownTokens = set([])
for (y_raw, x_raw) in train_raw:
	x_tokenized = tokenise(x_raw)
	for token in x_tokenized:
		knownTokens.add(token)


knownTokens = list(knownTokens)

print ("Result: " + str(len(knownTokens)) + " unique tokens identified in training data,\nand this will be used for our feature engineering")

#Map the tokens to a unique position in our feature vector :)
token_to_vector_position = {}

for i in range(0, len(knownTokens)):
	curToken = knownTokens[i]
	token_to_vector_position[curToken] = i

#If we wanted to, we could add an unknown feature too at the end of our feature vector 
#token_to_vector_position[None] = len(knownTokens)


def createFeatureVectors(data):
	final_x = np.zeros((len(data), len(knownTokens)))
	final_y = np.zeros((len(data), 1))

	i = 0
	for (y_raw, x_raw) in data:
		x_tokenized = tokenise(x_raw)
		for t in x_tokenized:
			#if not an unknown feature
			if t in token_to_vector_position:
				index = token_to_vector_position[t]
				#increment count of specific feature
				final_x[i, index] = final_x[i, index] + 1
		final_y[i] = label_to_int[y_raw]
		i = i + 1
	return final_x, final_y





#computee X_tr
X_tr,Y_tr = createFeatureVectors(train_raw) 
X_te,Y_te = createFeatureVectors(test_raw)


print ("TRAINING data")
print ("X_tr: ", X_tr.shape)
print ("Y_tr: ", Y_tr.shape)
print ("TESTING data")
print ("X_te: ", X_te.shape)
print ("Y_te: ", Y_te.shape)
#to compute mean accuracy during XGBoost testing
accArr = []


def getBestXGB(params):
	'''
	Performs k fold validation on XGboost with given params, by splitting into 10 train and validation sets.
	At each train and validation set, we train a XGB classifier using params on train data.
	We then measure the weighted accuracy of the predictions for the given XGB classifier.
	We take the average to get a score for the XGB model with params accuracy 
	INPUT:
		params: a list of floats contain params [learning_rate, n_estimators, max_depth, min_child_weight, subsample, colsample_bytree]
	OUTPUT:
		 -acc: a float that is smallest at greatest accuracy. This enables gp_minimize to find the highest accuracy
	'''
	global accArr

	#Hyperparameters to be optimized
	print (params)
	learning_rate = params[0] 
	n_estimators = params[1] 
	max_depth = params[2]
	min_child_weight = params[3]
	gamma = 0
	subsample = params[4]
	colsample_bytree = params[5]

	k = 2
	kf = KFold(n_splits=k)
	acc = 0
	print ("ST")
	for TrIndex, TeIndex in kf.split(X_tr):
		print ("splitted")
		xTr, xVal = X_tr[TrIndex], X_tr[TeIndex]
		print ("initialised")
		mdl = xgb.XGBClassifier(num_class = len(knownLabels), tree_method='hist', n_jobs=4, early_stopping_rounds=5, use_label_encoder=False, learning_rate = learning_rate, n_estimators = n_estimators, max_depth = max_depth, min_child_weight = min_child_weight, gamma = gamma, subsample = subsample, colsample_bytree = colsample_bytree, seed = 42, verbosity = 0)       
		print ("fitted")
		mdl.fit(xTr, Y_tr[TrIndex])
		print ("predicted")
		y_pred = mdl.predict(xVal)
		print ("acc add")
		acc += accuracy_score(y_pred, Y_tr[TeIndex]) / k
		print ("HERE:", acc)
	accArr.append(acc)
	print ("Overall mean accuracy: ", np.mean(accArr))
	return -acc

# Creating a sample space in which the initial randomic search should be performed
space = [(1e-3, 1e-1, 'log-uniform'), # learning rate
          (20, 100), # n_estimators
          (1, 10), #(1, 10) max_depth 
          (1, 6.), # min_child_weight (1, 6.)
          #(0, 1),  gamma 
          (0.5, 1.), # subsample 
          (0.5, 1.)] # colsample_bytree 

train_x, train_y = createFeatureVectors(train_raw)
test_x, test_y = createFeatureVectors(test_raw)


result = gp_minimize(getBestXGB, space, random_state = 42, n_random_starts = 150, n_calls  = 220, verbose = 1)
# Train final classifier
params = result["x"]
print ("Training on best params: ")
print (params)




