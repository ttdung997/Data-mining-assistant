#import cac thu vien python co ban
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#import thu vien hoc may sklearn
from sklearn import linear_model
from sklearn import metrics

# Doc du lieu tu file csv:
data = pd.read_csv("data.csv")

# Load cac du lieu ve chieu cao va can nang tu file
X = np.array([data['weight']]).T
y = np.array(data['height'])

print(X)
print(y)

# Tao mo hinh hoi quy tuyen hinh va huan luyen
regr = linear_model.LinearRegression()
regr.fit(X, y)

# Cac he so hoi quy thu duoc cua mo hinh
print("w_1 = ", regr.coef_[0], "w_0 = ", regr.intercept_)

# Kiem thu lai dieu kien

# Tinh y~
y_pred = regr.predict(X)

# Su dung metrics de tinh toan cac sai so cua mo hinh
print('Mean Absolute Error:', metrics.mean_absolute_error(y, y_pred))  #sai so tuyet doi
print('Mean Squared Error:', metrics.mean_squared_error(y, y_pred))  #sai so binh phuong trung binh

# Hien thi du lieu

# Lay 2 diem dau cuoi de ve do thi
x0 = np.array([[140,182]]).T
y0 = regr.predict(x0)


# Ve do thi su dung thu vien matplotlib
plt.plot(X, y.T, 'ro')     # Bieu thi cac diem du lieu
plt.plot(x0, y0)               # Bieu thi duong thang hoi quy
plt.axis([140, 190, 45, 75])
plt.xlabel('Height (cm)')
plt.ylabel('Weight (kg)')
plt.show()
