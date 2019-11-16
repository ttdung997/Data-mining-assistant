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

myDf = pd.read_csv('data/vcb.csv')


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
test_set.to_csv("test_set.csv")

window_len = 10

norm_cols = ['Close','Volume']

LSTM_training_inputs = []
for i in range(len(training_set)-window_len):
    temp_set = training_set[i:(i+window_len)].copy()
    for col in norm_cols:
        temp_set.loc[:, col] = temp_set[col]/temp_set[col].iloc[0] - 1
        #print(temp_set)
    LSTM_training_inputs.append(temp_set)

LSTM_training_outputs=LSTM_training_inputs

LSTM_test_inputs = []
for i in range(len(test_set)-window_len):
    temp_set = test_set[i:(i+window_len)].copy()
    for col in norm_cols:
        temp_set.loc[:, col] = temp_set[col]/temp_set[col].iloc[0] - 1
    # print(temp_set)
    LSTM_test_inputs.append(temp_set)
    LSTM_last_input=temp_set
    
LSTM_test_outputs = (test_set['Close'][window_len:].values/test_set['Close'][:-window_len].values)-1

LSTM_training_inputs = [np.array(LSTM_training_input) for LSTM_training_input in LSTM_training_inputs]
LSTM_training_inputs = np.array(LSTM_training_inputs)

LSTM_test_inputs = [np.array(LSTM_test_inputs) for LSTM_test_inputs in LSTM_test_inputs]
LSTM_test_inputs = np.array(LSTM_test_inputs)
print(LSTM_last_input.shape)
LSTM_last_input.to_csv("lastdata.csv")
LSTM_last_input = LSTM_test_inputs[-1]
LSTM_last_input.shape = (1,10,4)
np.random.seed(202)
print(LSTM_training_inputs)
LSTM_training_Close_outputs = (training_set['Close'][window_len:].values/training_set['Close'][:-window_len].values)-1
    
if TRAINING_FLAG == 1:
    print("this isc Close model")
    Close_model =build_LSTM_model(LSTM_training_inputs, output_size=1, neurons = 50)
    Close_model.fit(LSTM_training_inputs, LSTM_training_Close_outputs, 
                                epochs=20, batch_size=1, verbose=1, shuffle=True)


    model_json =  Close_model.to_json()
    model_output = "model/Close_model.json"
    weight_output = "model/Close_model.h5"
    with open(model_output, "w") as json_file:
            json_file.write(model_json)
            # serialize weights to HDF5
            Close_model.save_weights(weight_output)

   
model_output = 'model/Close_model.json'
weight_output = 'model/Close_model.h5'
json_file = open(model_output, 'r')
loaded_model_json = json_file.read()
json_file.close()
Close_model = model_from_json(loaded_model_json)
Close_model.load_weights(weight_output)
C = 0
RC = 0
predict = Close_model.predict(LSTM_test_inputs)
for i in range(0,len(LSTM_test_outputs)):
    try:
        if (predict[i] - predict[i+5])*(LSTM_test_outputs[i]-LSTM_test_outputs[i+5]) > 0:
            RC = RC +  1
        C = C+1
    except:
        continue

print(float(RC)/float(C))
# #build fw model by gold and bank data, and sectiment

# sectimentData = pd.read_csv("data/sectiment.csv")

# sectiment = [x[2:] for x in sectimentData.values]
# # print(sectiment)

# count = 0
# SecCount = []
# SecValue = []
# while count < len(sectiment):
#     try:
#         temCount = 0
#         temValue = 0
#         for i in range(count,count+30):
#             temValue = temValue + sectiment[i][0] - sectiment[i][1]
#             temCount = temCount +  sectiment[i][0]+sectiment[i][1]+sectiment[i][2]
#         SecCount.append(temCount)
#         SecValue.append(temValue)
#     except:
#         break
#     count = count +1

# SecCountTrain = SecCount[:(len(training_set)+1)] 
# SecCountTest = SecCount[-(len(test_set)+1):]   
# SecValueTrain = SecValue[:(len(training_set)+1)] 
# SecValueTest = SecValue[-(len(test_set)+1):]   
# sec_training_outputs = []


# for i in range(0,len(LSTM_training_Close_outputs)):
#     try:
#         sec_training_outputs.append(LSTM_training_Close_outputs[i+1]-LSTM_training_Close_outputs[i])
#     except:
#         break



# max_SC = max(SecCount)
# min_SC = min(SecCount)
# max_SV = max(SecValue)
# min_SV = min(SecValue)

# secTrain = []
# secTest = []

# for i in range(0,len(SecCountTrain)):
#     secTrain.append(np.array([SecCountTrain[i],SecValueTrain[i]]))

# for i in range(0,len(SecCountTest)):
#     secTest.append(np.array([SecCountTest[i],SecValueTest[i]]))

# secTrain =np.asarray(secTrain)
# secTest = np.asarray(secTest)


# json_file = open(model_output, 'r')
# loaded_model_json = json_file.read()
# json_file.close()
# Close_model = model_from_json(loaded_model_json)
# Close_model.load_weights(weight_output)


# # reg = LinearRegression()
# # reg.fit(secTrain[:len(sec_training_outputs)], sec_training_outputs)
# # filename = 'model/linear_model.sav'
# # pickle.dump(reg, open(filename, 'wb'))

# filename = 'model/linear_model.sav'
# reg = pickle.load(open(filename, 'rb'))

# sec_predict =  reg.predict(secTest)
# print(sec_predict)

# rule = rrulewrapper(YEARLY, byweekday=1, interval=5)
# loc = RRuleLocator(rule)
# formatter = DateFormatter('%m/%d/%y')
# date1 = datetime.date(2018, 12, 7)
# date2 = datetime.date(2018, 12, 30)
# delta = datetime.timedelta(days=1)

# dates = drange(date1, date2, delta)

# predict = [x + y for x, y in zip(sec_predict, Close_model.predict(LSTM_test_inputs))]

# fig, ax1 = plt.subplots(1,1)
# # ax1.plot(dates,
# #          test_set['Close'][window_len:], label='Actual')
# ax1.plot(dates,
#          ((np.transpose(predict)+1) * test_set['Close'].values[:-window_len])[0], 
#          label='Predicted')
# ax1.annotate('MAE: %.4f'%np.mean(np.abs((np.transpose(predict)+1)-\
#             (test_set['Close'].values[window_len:])/(test_set['Close'].values[:-window_len]))), 
#              xy=(0.75, 0.9),  xycoords='axes fraction',
#             xytext=(0.75, 0.9), textcoords='axes fraction')
# ax1.set_title('Dự đoán gía chứng VCB quán tháng 12/2018',fontsize=13)
# ax1.set_ylabel('gía cổ phiếu (VND)',fontsize=12)
# ax1.xaxis.set_major_locator(loc)
# ax1.xaxis.set_major_formatter(formatter)
# ax1.xaxis.set_tick_params(rotation=10, labelsize=10)
# ax1.set_ylim(bottom=50)
# ax1.set_ylim(top=80)
# plt.show()