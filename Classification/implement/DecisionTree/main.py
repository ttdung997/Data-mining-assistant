#!/usr/bin/env python
# -*- coding: utf-8 -*-
import pdb
from c45 import C45

# khởi tạo cây quyết định bằng thuật toán ID3
c1 = C45("data/train_data.csv", "data/metadata")

# Đọc dữ liệu
c1.fetchData()

# Tiền xử lý dữ liệu
c1.preprocessData()

# sinh cây quyết định
c1.generateTree()

# In cây quyết định
c1.printTree()
print("_______________")
print("result: ")
# Dự đoán kết quả cây

data = [line.rstrip('\n') for line in open('data/test_data.csv')]

X = []
y = []
for line in data:
	line= line.split(",")
	X.append(line[:-1])
	y.append(line[-1])


c1.metric(X,y)	