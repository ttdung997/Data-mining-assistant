import pandas as pd

# su dung ham cay quyet dinh tu sklearn
>>> from sklearn.tree import DecisionTreeClassifier

# su dung ham export_text de tao cac luat trong nhom
from sklearn import metrics
from sklearn.tree.export import export_text
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

# hoc mo hinh
# cac tham so :random_state , max_depth: do sau cua cay quyet dinh
decision_tree = DecisionTreeClassifier(random_state=0, max_depth=3)
decision_tree = decision_tree.fit(X_train, y_train)

# in ra tap luat
r = export_text(decision_tree,feature_names=['toeic','gpa','exp'])
print(r)

# |--- gpa <= 3.15
# |   |--- class: 0
# |--- gpa >  3.15
# |   |--- exp <= 2.50
# |   |   |--- class: 0
# |   |--- exp >  2.50
# |   |   |--- class: 1


# Phan tich: mot sinh vien co cpa 3.15 va hon 2.5 nam kinh ngiep se duoc nhan viec
# Cac truong hop con lai khong duoc nhan
# Diem toiec cua sinh vien khong lien quan den kha nang duoc nhan viec
# , dung la du lieu bia :v


# danh gia lai mo hinh
y_pred = decision_tree.predict(X_test)

report = metrics.classification_report(y_test,y_pred,digits=4) 

print (report)