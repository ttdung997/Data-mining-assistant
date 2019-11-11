from Cluster import DBSCAN as dbscan
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import datasets

data=datasets.load_iris().data



clf2 = dbscan()
pred = clf2.train(data)
clf2.infomation()
clf2.predict([8,2,2,2])
