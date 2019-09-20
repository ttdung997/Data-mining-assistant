
import numpy as np

from sklearn import datasets, linear_model

# X: height (cm), 
X = np.array([[147, 150, 153, 158, 163, 165, 168, 170, 173, 175, 178, 180, 183]]).T
# y: weight (kg)
y = np.array([ 49, 50, 51, 54, 58, 59, 60, 62, 63, 64, 66, 67, 68])
x1 = [147, 150, 153, 158, 163, 165, 168, 170, 173, 175, 178, 180, 183]



# fit the model by Linear Regression
regr = linear_model.LinearRegression()
regr.fit(X, y) 

# solution
print(" w_1 = ", regr.coef_[0], "w_0 = ", regr.intercept_)


