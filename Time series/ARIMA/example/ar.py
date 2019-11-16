# AR example
from statsmodels.tsa.ar_model import AR
from random import random
# contrived dataset
data = [x + random() for x in range(1, 100)]
print(data)
# fit model
model = AR(data)
model_fit = model.fit()

# make prediction
yhat = model_fit.predict(90, 100)
print(yhat)