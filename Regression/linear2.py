#import cac thu vien python co ban
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Thu vien math dung de goi cac ham tinh toan co ban (binh phuong, khai can)
from math import sqrt

#import thu vien hoc may sklearn
from sklearn import linear_model
from sklearn import metrics



X = np.array([[147, 150, 153, 158, 163, 165, 168, 170, 173, 175, 178, 180, 183]]).T
# y: weight (kg)
y = np.array([ 49, 50, 51, 54, 58, 59, 60, 62, 63, 64, 66, 67, 68])

# Tao mo hinh hoi quy tuyen hinh va huan luyen
regr = linear_model.LinearRegression()
regr.fit(X, y)

# Cac he so hoi quy thu duoc cua mo hinh
print("w_1 = ", regr.coef_[0], "w_0 = ", regr.intercept_)

# Kiem thu lai dieu kien

# Tinh y~
y_pred = regr.predict(X)

# Su dung metrics de tinh toan cac sai so cua mo hinh

# Sai so tuyet doi
print('Mean Absolute Error:', metrics.mean_absolute_error(y, y_pred)) 

# Sai so binh phuong trung binh 
print('Mean Squared Error:', metrics.mean_squared_error(y, y_pred))  

# Sai so can bac 2 trung binh
print('Root Mean Squared Error:', sqrt(metrics.mean_squared_error(y, y_pred)))

# He so xac dinh : coefficient of determination (R2)
print('R^2 score:', (metrics.r2_score(y, y_pred)))

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
