
import pandas as pd

# su dung ham SVM tu sklearn
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
# su dung ham export_text de tao cac luat trong nhom
from sklearn import metrics
from sklearn.model_selection import train_test_split




candidates = {'toeic': [780,750,690,710,680,730,690,720,740,690,610,690,710,680,770,610,580,650,540,590,620,600,550,550,570,670,660,580,650,660,640,620,660,660,680,650,670,580,590,690],
              'gpa': [4,3.9,3.3,3.7,3.9,3.7,2.3,3.3,3.3,1.7,2.7,3.7,3.7,3.3,3.3,3,2.7,3.7,2.7,2.3,3.3,2,2.3,2.7,3,3.3,3.7,2.3,3.7,3.3,3,2.7,4,3.3,3.3,2.3,2.7,3.3,1.7,3.7],
              'work_experience': [3,4,3,5,4,6,1,4,5,1,3,5,6,4,3,1,4,6,2,3,2,1,4,1,2,6,4,2,6,5,1,2,4,6,5,1,2,1,4,5],
              'admitted': [1,1,1,1,1,1,0,1,1,0,0,1,1,1,1,0,0,1,0,0,0,0,0,0,0,1,1,0,1,1,0,0,1,1,1,0,0,0,0,1]
              }

# Chuyen du lieu tu dien ve mang data
df = pd.DataFrame(candidates,columns= ['toeic', 'gpa','work_experience','admitted'])


# Du lieu x gom 3 truong, du lieu y
X = df[['toeic', 'gpa','work_experience']]
y = df['admitted'] 

# chia du lieu thanh 2 phan: train va test
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.25,random_state=0)  #train is based on 75% of the dataset, test is based on 25% of dataset

# hoc mo hinh SVC
clf = SVC(gamma='auto')
clf.fit(X, y)  


# hoc mo hinh SVC tuyen tinh
# clf = LinearSVC(random_state=0)


## in cac tham so cua mo hinh SVM
# print(clf.coef_)

# print(clf.intercept_)



# danh gia lai mo hinh
y_pred = clf.predict(X_test)

report = metrics.classification_report(y_test,y_pred,digits=4) 

print (report)