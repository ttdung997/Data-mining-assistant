# -*- coding: utf-8 -*-
import pandas as pd
import time
import seaborn as sns
import matplotlib.pyplot as plt
import datetime
import pickle
import numpy as np
import sys
from PIL import Image
import io
from keras.models import Sequential
from keras.layers import Activation, Dense
from keras.layers import LSTM
from keras.layers import Dropout
from keras.models import model_from_json
from keras.preprocessing import sequence
from scipy import interp
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
from mpl_toolkits.axes_grid1.inset_locator import mark_inset
from sklearn.linear_model import LinearRegression
from matplotlib.dates import (YEARLY, DateFormatter,
                              rrulewrapper, RRuleLocator, drange)
def mean(lst): 
    return sum(lst) / len(lst) 
TRAINING_FLAG = 0

def build_LSTM_model(inputs, output_size, neurons, activ_func="linear",
                dropout=0.25, loss="mae", optimizer="adam"):
    model = Sequential()

    model.add(LSTM(neurons, input_shape=(inputs.shape[1], inputs.shape[2])))
    model.add(Dropout(dropout))
    model.add(Dense(units=output_size))
    model.add(Activation(activ_func))

    model.compile(loss=loss, optimizer=optimizer)
    return model

def build_feedforward_model(inputs, output_size, neurons, loss="mae", optimizer="adam"):
    # create model
    model = Sequential()
    model.add(Dense(12, input_dim=inputs.shape[1], activation='relu'))
    model.add(Dense(inputs.shape[1], activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss=loss, optimizer=optimizer, metrics=['accuracy'])
    return model


tickerList =['VCB','BID','VPB','ACB','SHB']

for ticker in tickerList:
    print(ticker)
    myDf = pd.read_csv('data/'+ticker+'.csv')


    #print(myDf) 

    myDf = myDf.sort_values(by='Date')
    split_date = 20180815
    training_set, test_set = myDf[myDf['Date']<split_date], myDf[myDf['Date']>=split_date]

    training_set = training_set.drop('Date', 1)
    test_set = test_set.drop('Date', 1)
    training_set=training_set.astype('float')

    test_set=test_set.astype('float')

    kwargs = { 'Close_off_high': lambda x: 2*(x['High']- x['Close'])/(x['High']-x['Low'])-1,
            'Volatility': lambda x: (x['High']- x['Low'])/(x['Open'])}
    training_set = training_set.assign(**kwargs)
    test_set = test_set.assign(**kwargs)

    #need to reverse the data frame so that subsequent rows represent later timepoints
    training_set = training_set.drop('High', 1)
    training_set = training_set.drop('Low', 1)
    training_set = training_set.drop('Open', 1)

    test_set = test_set.drop('High', 1)
    test_set = test_set.drop('Low', 1)
    test_set = test_set.drop('Open', 1)
    # test_set.to_csv("test_set.csv")

    close_train = np.array(training_set['Close'])


    close_test = np.array(test_set['Close'])

    # close_all = close_train + close_test
    close_all =  np.concatenate((close_train, close_test), axis=None)


    close_label = close_all[len(close_train)-1:len(close_all)-1]

    from statsmodels.tsa.ar_model import AR

    # # AR example
    print("AR model: ")

    model = AR(close_train)
    model_fit = model.fit()

    # make prediction
    yhat = model_fit.predict(len(close_train), len(close_all)-1)

    true_predict = 0
    for i in range(0,len(yhat)-1):
        # print((yhat[i+1]-close_label[i])*(close_label[i+1]-close_label[i]))
        if (yhat[i+1]-close_label[i])*(close_label[i+1]-close_label[i]) >= 0:
           true_predict = true_predict + 1
    print("Trend accuracy: ", str(true_predict/len(close_label)))

    error = [(x - y)*(x-y) for x, y in zip(yhat, close_label)]
    print("MSE: ", str(mean(error)))
    print("___________________________________________")

    # ARIMA example
    from statsmodels.tsa.arima_model import ARMA
    print("MA model: ")
    # contrived dataset
    model = ARMA(close_train, order=(0, 1))
    # model = ARMA(close_train, order=(2, 1))
    model_fit = model.fit(disp=False)
    # make prediction
    yhat = model_fit.predict(len(close_train), len(close_all)-1)
    true_predict = 0
    for i in range(0,len(yhat)-1):
        # print((yhat[i+1]-close_label[i])*(close_label[i+1]-close_label[i]))
        if (yhat[i+1]-close_label[i])*(close_label[i+1]-close_label[i]) >= 0:
           true_predict = true_predict + 1
    print("Trend accuracy: ", str(true_predict/len(close_label)))

    error = [(x - y)*(x-y) for x, y in zip(yhat, close_label)]
    print("MSE: ", str(mean(error)))
    print("___________________________________________")

    from statsmodels.tsa.arima_model import ARMA
    print("ARMA model: ")
    # contrived dataset
    # model = ARMA(close_train, order=(0, 1))
    model = ARMA(close_train, order=(2, 1))
    model_fit = model.fit(disp=False)
    # make prediction
    yhat = model_fit.predict(len(close_train), len(close_all)-1)
    true_predict = 0
    for i in range(0,len(yhat)-1):
        # print((yhat[i+1]-close_label[i])*(close_label[i+1]-close_label[i]))
        if (yhat[i+1]-close_label[i])*(close_label[i+1]-close_label[i]) >= 0:
           true_predict = true_predict + 1
    print("Trend accuracy: ", str(true_predict/len(close_label)))

    error = [(x - y)*(x-y) for x, y in zip(yhat, close_label)]
    print("MSE: ", str(mean(error)))
    print("___________________________________________")

    # SES example
    print("SES model")
    from statsmodels.tsa.holtwinters import SimpleExpSmoothing
    from random import random
    # contrived dataset
    model = SimpleExpSmoothing(close_train)
    model_fit = model.fit()
    # make prediction
    yhat = model_fit.predict(len(close_train), len(close_all)-1)
    true_predict = 0
    for i in range(0,len(yhat)-1):
        # print((yhat[i+1]-close_label[i])*(close_label[i+1]-close_label[i]))
        if (yhat[i+1]-close_label[i])*(close_label[i+1]-close_label[i]) >= 0:
           true_predict = true_predict + 1
    print("Trend accuracy: ", str(true_predict/len(close_label)))

    error = [(x - y)*(x-y) for x, y in zip(yhat, close_label)]
    print("MSE: ", str(mean(error)))
    print("___________________________________________")


    # HWES example
    print("HWES model")
    from statsmodels.tsa.holtwinters import ExponentialSmoothing
    from random import random
    model = ExponentialSmoothing(close_train)
    model_fit = model.fit()
    # make prediction
    yhat = model_fit.predict(len(close_train), len(close_all)-1)
    # print(yhat)
    true_predict = 0
    for i in range(0,len(yhat)-1):
        # print((yhat[i+1]-close_label[i])*(close_label[i+1]-close_label[i]))
        if (yhat[i+1]-close_label[i])*(close_label[i+1]-close_label[i]) >= 0:
           true_predict = true_predict + 1
    print("Trend accuracy: ", str(true_predict/len(close_label)))

    error = [(x - y)*(x-y) for x, y in zip(yhat, close_label)]
    print("MSE: ", str(mean(error)))
    print("___________________________________________")



# rule = rrulewrapper(YEARLY, byweekday=1, interval=5)
# loc = RRuleLocator(rule)
# formatter = DateFormatter('%m/%d/%y')
# date1 = datetime.date(2019, 11, 27)
# date2 = datetime.date(2019, 12, 30)
# delta = datetime.timedelta(days=1)

# dates = drange(date1, date2, delta)


# fig, ax1 = plt.subplots(1,1)
# # ax1.plot(dates,
#          close_label, label='Actual')
# ax1.plot(dates,
#          yhat, 
#          label='Predicted')
# ax1.annotate('MAE: %.4f'%mean(error), 
#              xy=(0.75, 0.9),  xycoords='axes fraction',
#             xytext=(0.75, 0.9), textcoords='axes fraction')
# ax1.set_title('Actual and Predict price',fontsize=13)
# ax1.set_ylabel('VND',fontsize=12)
# ax1.xaxis.set_major_locator(loc)
# ax1.xaxis.set_major_formatter(formatter)
# ax1.xaxis.set_tick_params(rotation=10, labelsize=10)
# ax1.set_ylim(bottom=60)
# ax1.set_ylim(top=67)
# plt.show()