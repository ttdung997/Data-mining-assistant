
import sys
import numpy as np
import pickle
import random
import matplotlib.pyplot as plt
import operator as op

class DBSCAN:
    def __init__(self, norm_type="Normalization", distance_type="Euclidean", eps=0.2, m=30):
        self.norm_type = norm_type
        self.distance_type = distance_type
        self.eps = eps          # khoảng cách tối thiểu để 1 điểm được coi là hàng xóm
        self.m = m              # số lượng hàng xóm
        self.label = None
        self.neighbor = None
        self.cluster_info = None
        self.cluster_center = None
        self.minValue = None
        self.maxValue = None
        self.data = None

    # Hàm chuẩn hóa dữ liệu [0,1]
    def preProcessingData(self, data):
        minValue = data.min(axis=0)
        maxValue = data.max(axis=0)
        self.minValue = minValue
        self.maxValue = maxValue
        diff = maxValue - minValue
        # normalization
        mindata = np.tile(minValue, (data.shape[0], 1))
        normdata = (data - mindata) / np.tile(diff, (data.shape[0], 1))
        return normdata

    # Tính khoảng cách theo một số chuẩn thông thường
    def calculateDistance(self, x1, x2):
        if self.distance_type == "Euclidean":
            try: 
                d = np.sqrt(np.sum(np.power(x1 - x2, 2), axis=1))
            except:
                d = np.sqrt(np.sum(np.power(x1 - x2, 2)))
            #d = np.sqrt(np.sum(np.power(x1 - x2, 2)))
        elif self.distance_type == "Cosine":
            d = np.dot(x1, x2)/(np.linalg.norm(x1)*np.linalg.norm(x2))
        elif self.distance_type == "Manhattan":
            d = np.sum(x1 - x2)
        else:
            print("Error Type!")
            sys.exit()
        return d

    '''
  
      Input:  train_data      Dữ liệu đầu vào
      Output: centers         Tâm cụm
              distances       Khoảng cách với tâm cụm
    '''
    def train(self, train_data, display="True"):
        # Khởi tạo tâm cụm ngẫu nhiên
        train_data = self.preProcessingData(train_data)
        centers = self.getCenters(train_data)
        self.data = train_data
        label = {}
        sample_num = len(train_data)

        # gĩư các tâm cụm ban đầu để so sánh
        initial_centers = centers.copy()

        k = 0
        # tạo các điểm chưa đến thăm
        unvisited = list(range(sample_num))  
        while len(centers) > 0:
            # tạo danh sách các điểm đến thăm
            visited = []
            visited.extend(unvisited)
            cores = list(centers.keys())
            # lựa chọn tâm bất kì
            randNum = np.random.randint(0, len(cores))
            core = cores[randNum]
            # thêm tâm đã chọn vào danh sách xét
            # xóa tâm đã chọn trong danh sách chưa thăm
            core_neighbor = []                          
            core_neighbor.append(core)
            unvisited.remove(core)

            
            # vòng lặp phân cụm
            while len(core_neighbor) > 0:
                # chọn tâm cụm đầu trong danh sách
                Q = core_neighbor[0]
                del core_neighbor[0]
                # lọc với tâm cụm đã chọn
                if Q in initial_centers.keys():
                    # chọn các điểm là hàng xóm của tâm cụm
                    diff = [sample for sample in initial_centers[Q] if sample in unvisited]
                    # thêm các điểm đã chọn vào danh sách
                    core_neighbor.extend(diff)
                    unvisited = [sample for sample in unvisited if sample not in diff]
            k += 1
            # lấy các điểm thuộc 1 cụm và xóa khỏi danh sách xét
        
            label[k] = [val for val in visited if val not in unvisited]
            for index in label[k]:
                if index in centers.keys():
                    del centers[index]

        # đánh cụm cho các điểm trong danh sahcs
        labels = np.zeros([sample_num])
        for i in range(1, len(label)):
            index = label[i]
            labels[index] = i
        self.label = labels

        # vẽ hình biểu thị
        # if display:
        #     self.plotResult(train_data)

        # print(labels)
        self.cluster_info = label
        return label

    # lấy các tâm cụm
    def getCenters(self, train_data):
        neighbor = {}
        for i in range(len(train_data)):
            # lấy các hàng xóm của điểm
            distance = self.calculateDistance(train_data[i], train_data)
            # print(distance)
            # quit()
            # print(distance)
            index = np.where(distance <= self.eps)[0]
            # Kiểm tra mật độ để quyết định
            if len(index) > self.m:
                neighbor[i] = index
        return neighbor
        
    # hiển thị thông tin phân cụm
    def infomation(self):
        print("number of cluster:", len(self.cluster_info))
        print(self.cluster_info)

        cluster_center = []
        for cluster in self.cluster_info.values():
            sumValue  = np.zeros([len(self.data[0])])
            count = 0
            for i in (cluster):
                count = count + 1
                sumValue = sumValue + self.data[i] 
            cluster_center.append(sumValue/count)
        self.cluster_center = cluster_center
        cluster_info = [x*(self.maxValue-self.minValue)+self.minValue for x in cluster_center] 
        print("Cluster infomation: ")
        print(cluster_info)

    #dự đoán cụm mới
    def predict(self,data):
        data = (data - self.minValue)/(self.maxValue - self.minValue)
        print(data)
        cluster_class = -1
        min_distance = 100
        count = 0
        for cluster in self.cluster_center:
            distance = self.calculateDistance(data,cluster)
            if distance < min_distance:
                min_distance = distance
                cluster_class = count

            count = count +1
        print(cluster_class)



    # vẽ và biểu diễn đồ thị (dữ liệu 2 chiều)
    def plotResult(self, train_data):
        plt.scatter(train_data[:, 0], train_data[:, 1], c=self.label)
        plt.title('DBSCAN')
        plt.show()



