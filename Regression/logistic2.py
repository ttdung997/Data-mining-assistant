import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
# Mo ta bo du lieu

# bai toan tuyen dung, Tu du lieu cua sinh vien bao gom
# toiec: diem toeic cua sinh vien
# gpa: diem chuan dau ra cua sinh vien
# work_experience: thoi gian lam viec cua sinh vien

# tu do thu nhan duoc ket qua tuyen dung trong admitted
# admitted = 1: co viec lam
# admitted = 0: that nghiep

# day la du lieu dung tu dien
candidates = {'toeic': [780,750,690,710,680,730,690,720,740,690,610,690,710,680,770,610,580,650,540,590,620,600,550,550,570,670,660,580,650,660,640,620,660,660,680,650,670,580,590,690],
              'gpa': [4,3.9,3.3,3.7,3.9,3.7,2.3,3.3,3.3,1.7,2.7,3.7,3.7,3.3,3.3,3,2.7,3.7,2.7,2.3,3.3,2,2.3,2.7,3,3.3,3.7,2.3,3.7,3.3,3,2.7,4,3.3,3.3,2.3,2.7,3.3,1.7,3.7],
              'work_experience': [3,4,3,5,4,6,1,4,5,1,3,5,6,4,3,1,4,6,2,3,2,1,4,1,2,6,4,2,6,5,1,2,4,6,5,1,2,1,4,5],
              'admitted': [1,1,1,1,1,1,0,1,1,0,0,1,1,1,1,0,0,1,0,0,0,0,0,0,0,1,1,0,1,1,0,0,1,1,1,0,0,0,0,1]
              }

# chuyen du lieu tu dien ve mang data
df = pd.DataFrame(candidates,columns= ['toeic', 'gpa','work_experience','admitted'])


# du lieu x gom 3 truong, du lieu y
X = df[['toeic', 'gpa','work_experience']]
y = df['admitted']  

# chia du lieu thanh 2 phan: train va test
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.25,random_state=0)  #train is based on 75% of the dataset, test is based on 25% of dataset

# Tao va huan luyen mo hinh voi phan train
logistic_regression= LogisticRegression()
logistic_regression.fit(X_train,y_train)

# Kiem thu ket qua voi tap test
y_pred=logistic_regression.predict(X_test)

# load metrics de danh gia do chinh xac cua mo hinh 

report = metrics.classification_report(y_test,y_pred,digits=4) 
# classification_report cho phep danh gia ket qua cho bai toan phan lop
# Cac ban tim doc cac chi so precision, recall (tim thong qua google)
# de hien ro hon ve bai toan phan lop
print (report)