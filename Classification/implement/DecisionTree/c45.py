#!/usr/bin/env python
# -*- coding: utf-8 -*-
import math
from sklearn import metrics
class C45:

	""" khởi tạo một cây quyết định """
	def __init__(self, pathToData,pathToNames):
		self.filePathToData = pathToData
		self.filePathToNames = pathToNames
		self.data = []
		self.classes = []
		self.numAttributes = -1 
		self.attrValues = {}
		self.attributes = []
		self.tree = None

	# lọc dữ liệu từ file dữ liệu đầu vào
	def fetchData(self):
		# Đọc dữ liệu meta-data
		with open(self.filePathToNames, "r") as file:
			classes = file.readline()
			self.classes = [x.strip() for x in classes.split(",")]
			# Thêm lần lượt các thuộc tính vào chuỗi
			for line in file:
				[attribute, values] = [x.strip() for x in line.split(":")]
				values = [x.strip() for x in values.split(",")]
				# thêm trạng thái của thuộc tính
				self.attributes.append(attribute)
		self.numAttributes = len(self.attrValues.keys())
		# self.attributes = list(self.attrValues.keys())

		# Đọc các bản ghi dữ liệu khác
		with open(self.filePathToData, "r") as file:
			for line in file:
				row = [x.strip() for x in line.split(",")]
				if row != [] or row != [""]:
					self.data.append(row)
	# Tiền xử lý dữ liệu, quy đổi các thuộc tính continuous về dạng thực (float)
	def preprocessData(self):
		for index,row in enumerate(self.data):
			for attr_index in range(self.numAttributes):
				if(not self.isAttrDiscrete(self.attributes[attr_index])):
					self.data[index][attr_index] = float(self.data[index][attr_index])

	# In cây quyết định bằng cách gọi các nút
	def printTree(self):
		self.printNode(self.tree)

	def printNode(self, node, indent=""):
		if not node.isLeaf:
			#numerical
			leftChild = node.children[0]
			rightChild = node.children[1]
			if leftChild.isLeaf:
				print(indent + str(node.label) + " <= " + str(node.threshold) + " : " + leftChild.label)
			else:
				print(indent + str(node.label) + " <= " + str(node.threshold)+" : ")
				self.printNode(leftChild, indent + "	")

			if rightChild.isLeaf:
				print(indent + str(node.label) + " > " + str(node.threshold) + " : " + rightChild.label)
			else:
				print(indent + str(node.label) + " > " + str(node.threshold) + " : ")
				self.printNode(rightChild , indent + "	")


	# Tạo cây quyết định
	def generateTree(self):
		self.tree = self.recursiveGenerateTree(self.data, self.attributes)

	def recursiveGenerateTree(self, curData, curAttributes):
		# kiểu tra dữ liệu đầu vào
		allSame = self.allSameClass(curData)
		
		if len(curData) == 0:
			#Fail
			return Node(True, "Fail", None)
		elif allSame is not False:
			#return a node with that class
			return Node(True, allSame, None)
		elif len(curAttributes) == 0:
			#return a node with the majority class
			majClass = self.getMajClass(curData)
			return Node(True, majClass, None)
		else:
			# Tính toán ngưỡng và đệ quy
			(best,best_threshold,splitted) = self.splitAttribute(curData, curAttributes)
			# Lọc thuộc tính đã chia của cây
			remainingAttributes = curAttributes[:]
			remainingAttributes.remove(best)
			node = Node(False, best, best_threshold)
			node.children = [self.recursiveGenerateTree(subset, remainingAttributes) for subset in splitted]
			return node

	# tìm lớp ứng với nút lá
	def getMajClass(self, curData):
		freq = [0]*len(self.classes)
		for row in curData:
			index = self.classes.index(row[-1])
			freq[index] += 1
		maxInd = freq.index(max(freq))
		return self.classes[maxInd]

	#kiểm tra dữ liệu có thuộc 1 lớp k
	def allSameClass(self, data):
		for row in data:
			if row[-1] != data[0][-1]:
				return False
		return data[0][-1]

	# kiểm tra sự tồn tại và liên tục của thuộc tính
	def isAttrDiscrete(self, attribute):
		if attribute not in self.attributes:
			raise ValueError("Attribute not listed")
		return False

	# Xây dựng cây quyết định them thuộc tính
	def splitAttribute(self, curData, curAttributes):
		splitted = []
		maxEnt = -1*float("inf")
		best_attribute = -1
		best_threshold = None
		for attribute in curAttributes:
			indexOfAttribute = self.attributes.index(attribute)
			# Sắp xếp dữ liệu, duyệt ngưỡng trung bình, tính toán entropy và điểm thưởng
			curData.sort(key = lambda x: x[indexOfAttribute])
			# print(curData)
			# quit()
			for j in range(0, len(curData) - 1):
				# Tính ngưỡng và chia dữ liệu theo ngưõng

				if curData[j][indexOfAttribute] != curData[j+1][indexOfAttribute]:
					threshold = (float(curData[j][indexOfAttribute]) + float(curData[j+1][indexOfAttribute])) / 2
					less = []
					greater = []
					for row in curData:
						if(float(row[indexOfAttribute]) > threshold):
							greater.append(row)
						else:
							less.append(row)
					# Tính toán điểm thưởng, nếu điểm thưởng lớn hớn
					# cập nhật gía trị tốt nhất
					e = self.gain(curData, [less, greater])
					if e >= maxEnt:
						splitted = [less, greater]
						maxEnt = e
						best_attribute = attribute
						best_threshold = threshold
		return (best_attribute,best_threshold,splitted)

	def gain(self,unionSet, subsets):
		#input : Toàn bộ dữ liệu và dữ liệu phân hoạch them ngưỡng
		#output : Điểm thưởng
		S = len(unionSet)

		# Tính entropy trước khi phân hoạch
		impurityBeforeSplit = self.entropy(unionSet)
		
		# Tình entropy sau khi phân hoạch
		
		weights = [len(subset)/float(S) for subset in subsets]
		
		impurityAfterSplit = 0
		for i in range(len(subsets)):
			impurityAfterSplit += weights[i]*self.entropy(subsets[i])
		
		# Tình hiệu số entropy, coi là điểm thưởng
		totalGain = impurityBeforeSplit - impurityAfterSplit
		return totalGain

	# Hàm tính entropy
	def entropy(self, dataSet):
		S = len(dataSet)
		# print(S)
		if S == 0:
			return 0
		num_classes = [0 for i in self.classes]
		for row in dataSet:
			classIndex = list(self.classes).index(row[-1])
			num_classes[classIndex] = num_classes[classIndex]+1
		num_classes = [float(x)/S for x in num_classes]
		ent = 0
		for num in num_classes:
			ent += num*self.log(num)
		return ent*-1


	#hàm tính log
	def log(self, x):
		if x == 0:
			return 0
		else:
			return math.log(x,2)

	# hàm dự đoán gía trị
	def metric(self,data,label):
		label = [self.classes.index(x) for x in label]
		predict = []
		for x in data:
			predict.append(self.classes.index(self.predict(x)))
		report = metrics.classification_report(label,predict,digits=4) 
		print(report)


	def predict(self,data):
		return self.findclass(self.tree,data)

	def findclass(self, node, data):
		if not node.isLeaf:
			leftChild = node.children[0]
			rightChild = node.children[1]
			index = self.attributes.index(node.label)

			if(float(data[index]) > node.threshold):
				classtify =  self.findclass(rightChild ,data)
			else:
				classtify = self.findclass(leftChild ,data)
			return classtify
		else:
			return node.label

# Nút và các tham số
class Node:
	def __init__(self,isLeaf, label, threshold):
		self.label = label
		self.threshold = threshold
		self.isLeaf = isLeaf
		self.children = []


