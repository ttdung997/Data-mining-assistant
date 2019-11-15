import csv
import numpy as np
import pandas as pd
import math
import pickle
from time import gmtime, strftime
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn import svm

from sklearn.ensemble import IsolationForest
from sklearn.metrics import average_precision_score, recall_score, precision_score, f1_score,accuracy_score
from sklearn import metrics
import time
def report_evaluation_metrics(y_true, y_pred):
	average_precision = average_precision_score(y_true, y_pred)
	precision = precision_score(y_true, y_pred, labels=[0, 1], pos_label=1)
	recall = recall_score(y_true, y_pred, labels=[0, 1], pos_label=1)
	f1 = f1_score(y_true, y_pred, labels=[0, 1], pos_label=1)
	acc = accuracy_score(y_true,y_pred)
	print('Average precision-recall score: {0:0.2f}'.format(average_precision))
	print('Precision: {0:0.4f}'.format(precision))
	print('Recall: {0:0.4f}'.format(recall))
	print('F1: {0:0.4f}'.format(f1))
	print('acc: {0:0.4f}'.format(acc))

def get_data(dataset): 
	data= []
	with open(dataset, "r") as f:
		reader = csv.reader(f)
		for row in reader:
			data.append(row) 
		data =  np.array(data)
	return data

def main():

	dataset = "data.csv"

	# Read data from training set and testing set
	print(dataset)
	data = get_data(dataset)

	anomaly_test_data = []
	anomaly_test_label = []
	nomaly_data =[]
	nomaly_label = []
	for x in data:
		if float(x[-1]) == 1:
			anomaly_test_data.append(x[:-1])
			anomaly_test_label.append(1)
		else:
			nomaly_data.append(x[:-1])
			nomaly_label.append(0)

	nomaly_data = np.array(nomaly_data)
	nomaly_label = np.array(nomaly_label)
	anomaly_test_data = np.array(anomaly_test_data)
	anomaly_test_label = np.array(anomaly_test_label)


	split_train_data = StratifiedShuffleSplit(n_splits=1, test_size=0.3,random_state=0)
	for train, test in split_train_data.split(nomaly_data, nomaly_label):
		X_train, nomaly_test_data, y_train, nomaly_test_label = nomaly_data[train], nomaly_data[test], nomaly_label[train], nomaly_label[test]
	print(nomaly_test_data)
	print(anomaly_test_data)

	nomaly_test_data=nomaly_test_data.astype(float)
	anomaly_test_data=anomaly_test_data.astype(float)
	nomaly_test_label=nomaly_test_label.astype(int)
	anomaly_test_label=anomaly_test_label.astype(int)

	X_test = np.concatenate((nomaly_test_data,anomaly_test_data), axis=0)
	y_test = np.concatenate((nomaly_test_label,anomaly_test_label), axis=0)


	# Extract data and labels
	X_train =np.array(X_train)
	X_test =np.array(X_test)
	y_test =np.array(y_test)
	X_train=X_train.astype(float)
	X_test=X_test.astype(float)
	y_test=y_test.astype(int)

	print("Starting training!!")
	print("Starting training!!")
	clf = svm.OneClassSVM(nu=0.3 , kernel="rbf",verbose = True)
	clf.fit(X_train)


	# save the model to disk
	filename = 'models/svm_model.sav'
	pickle.dump(clf, open(filename, 'wb'))

	# some time later...

	# load the model from disk
	clf = pickle.load(open(filename, 'rb'))
	# print(clf.predict(X_test))
	# print(y_test)

	start_time = time.time()

	y_predict = clf.predict(X_test)
	print("--- request/seconds ---" +str(float(time.time() - start_time)/len(X_test)))
	
	y_result = [0 if( x > 0 ) else 1 for x in y_predict]
	# print(y_result)
	nomalyPercent = y_result.count(0)
	anomalyPercent = y_result.count(1)
	print("nomaly percent :"+str(float(nomalyPercent)/(anomalyPercent + nomalyPercent)))
	
	report_evaluation_metrics(y_test,y_result)

	print("Final report!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
	report = metrics.classification_report(y_test,y_result,digits=4)
	print '\n clasification report:\n', report
main()
