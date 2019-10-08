# ap dung bai toan hoi quy da bien de du doan chung khoan

from pandas import DataFrame

# Thu vien math dung de goi cac ham tinh toan co ban (binh phuong, khai can)
from math import sqrt

#import thu vien hoc may sklearn
from sklearn import linear_model
from sklearn import metrics

# thu vien thong ke
import statsmodels.api as sm


# du lieu dau vao

Stock_Market = {'Year': [2017,2017,2017,2017,2017,2017,2017,2017,2017,2017,2017,2017,2016,2016,2016,2016,2016,2016,2016,2016,2016,2016,2016,2016],
                'Month': [12, 11,10,9,8,7,6,5,4,3,2,1,12,11,10,9,8,7,6,5,4,3,2,1],
                'Interest_Rate': [2.75,2.5,2.5,2.5,2.5,2.5,2.5,2.25,2.25,2.25,2,2,2,1.75,1.75,1.75,1.75,1.75,1.75,1.75,1.75,1.75,1.75,1.75],
                'Unemployment_Rate': [5.3,5.3,5.3,5.3,5.4,5.6,5.5,5.5,5.5,5.6,5.7,5.9,6,5.9,5.8,6.1,6.2,6.1,6.1,6.1,5.9,6.2,6.2,6.1],
                'Stock_Index_Price': [1464,1394,1357,1293,1256,1254,1234,1195,1159,1167,1130,1075,1047,965,943,958,971,949,884,866,876,822,704,719]        
                }

#doc du lieu va chia du lieuj thanh 2 truoung 
df = DataFrame(Stock_Market,columns=['Year','Month','Interest_Rate','Unemployment_Rate','Stock_Index_Price'])


X = df[['Interest_Rate','Unemployment_Rate']] 
Y = df['Stock_Index_Price']
 
 #hoc mo hinh
regr = linear_model.LinearRegression()
regr.fit(X, Y)

print('Intercept: \n', regr.intercept_)
print('Coefficients: \n', regr.coef_)


# du doan thu
New_Interest_Rate = 2.75
New_Unemployment_Rate = 5.3
print ('Predicted Stock Index Price: \n', regr.predict([[New_Interest_Rate ,New_Unemployment_Rate]]))

y_pred = regr.predict(X)

# Su dung metrics de tinh toan cac sai so cua mo hinh

# Sai so tuyet doi
print('Mean Absolute Error:', metrics.mean_absolute_error(Y, y_pred)) 

# Sai so binh phuong trung binh 
print('Mean Squared Error:', metrics.mean_squared_error(Y, y_pred))  

# Sai so can bac 2 trung binh
print('Root Mean Squared Error:', sqrt(metrics.mean_squared_error(Y, y_pred)))

# He so xac dinh : coefficient of determination (R2)
print('R^2 score:', (metrics.r2_score(Y, y_pred)))

# Hien thi du lieu

# phan tich thong ke
X = sm.add_constant(X) # adding a constant
 
model = sm.OLS(Y, X).fit()
predictions = model.predict(X) 
 
print_model = model.summary()
print(print_model)