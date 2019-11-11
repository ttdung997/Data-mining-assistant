from __future__ import print_function
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
import random

def preProcessingData( data):
    minValue = data.min(axis=0)
    maxValue = data.max(axis=0)
    diff = maxValue - minValue
    # normalization
    mindata = np.tile(minValue, (data.shape[0], 1))
    normdata = (data - mindata) / np.tile(diff, (data.shape[0], 1))
    return minValue,maxValue,normdata
    
# Khời tạo tâm cụm ban đầu
def kmeans_init_centroids(X, k):
	return X[np.random.choice(X.shape[0], k, replace=False)]

# tính toán điểm thuộc cụm
def kmeans_assign_labels(X, centroids):
	D = cdist(X, centroids)
	return np.argmin(D, axis = 1)

# kiểm tra điều kiện thoát
def has_converged(centroids, new_centroids):
	return (set([tuple(a) for a in centroids]) ==
	set([tuple(a) for a in new_centroids]))

def kmeans_update_centroids(X, labels, K):
	centroids = np.zeros((K, X.shape[1]))
	# Tính tâm cụm bằng trung bình n cụm
	for k in range(K):
		Xk = X[labels == k, :]
		centroids[k,:] = np.mean(Xk, axis = 0)
	return centroids

def kmeans(X, K):
	# hàm xây dựng thuật toán
	centroids = [kmeans_init_centroids(X, K)]
	labels = []
	while True:
		labels.append(kmeans_assign_labels(X, centroids[-1]))
		new_centroids = kmeans_update_centroids(X, labels[-1], K)
		if has_converged(centroids[-1], new_centroids):
			break
		centroids.append(new_centroids)
	return (centroids, labels)




data = pd.read_csv("data.csv").values

minValue,maxValue,data = preProcessingData(data)

K = 3 # 3 clusters

(centroids, labels) = kmeans(data, K)
print(labels[-1])
print('Centers found by algorithm:')

clusters = centroids[-1]


for cluster in  clusters:
	cluster = cluster*(maxValue - minValue)+minValue 
	cluster = [round(float(x),3) for x in cluster]
	print(cluster)

test_data = [550,3.3,2,2,2]
test_data = (test_data - minValue)/(maxValue - minValue)

print(kmeans_assign_labels([test_data],centroids[-1]))