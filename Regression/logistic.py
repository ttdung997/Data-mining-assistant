from sklearn.datasets import load_iris #load bo du lieu mau cua sklearn
from sklearn.linear_model import LogisticRegression # load mo hinh Logistic


# doc du llieu va nhan
X, y = load_iris(return_X_y=True)


# goi va huan luyen mo hinh
clf = LogisticRegression(random_state=0, solver='lbfgs',
                         multi_class='multinomial')
clf.fit(X, y)

# danh gia do chinh xac cua mo hinh
print(clf.score(X, y))	
