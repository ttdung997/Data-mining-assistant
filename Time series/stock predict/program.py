import pandas as pd
import time
import seaborn as sns
import matplotlib.pyplot as plt
import datetime
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


Close_model_id = 1
Volume_model_id = 2
Close_off_high_model_id =3
Volatility_model_id = 4
class BtcPredict():
    def _init_(self):
        print("This is bitcoin predict!")
    #Load Model
    def loadModel(self,model,weight,id): 
        # load json and create model
        json_file = open(model, 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        if(id == Close_model_id):
            self.Close_model = model_from_json(loaded_model_json)
            # load weights into new model
            self.Close_model.load_weights(weight) 
        elif(id==Volume_model_id):
            self.Volume_model = model_from_json(loaded_model_json)
            # load weights into new model
            self.Volume_model.load_weights(weight) 
        elif(id==Close_off_high_model_id):
            self.Close_off_high_model = model_from_json(loaded_model_json)
            # load weights into new model
            self.Close_off_high_model.load_weights(weight)
        elif(id== Volatility_model_id):
            self.Volatility_model = model_from_json(loaded_model_json)
            # load weights into new model
            self.Volatility_model.load_weights(weight) 
      
        # print("Loaded model from disk")
    def predict(self,data,id):
        if(id == Close_model_id):
            return self.Close_model.predict(data)
        elif(id==Volume_model_id):
            return self.Volume_model.predict(data)
        elif(id==Close_off_high_model_id):
            return self.Close_off_high_model.predict(data)
        elif(id==Volatility_model_id):
            return self.Volatility_model.predict(data)
if __name__ == "__main__":
    Close_model = "model/Close_model.json"
    Close_model_weight = "model/Close_model.h5"

    Volume_model = "Volume_model.json"
    Volume_model_weight = "Volume_model.h5"

    close_off_high_model ="Close_off_high_model.json"
    close_off_high_model_weight ="Close_off_high_model.h5"

    volatility_model = "Volatility_model.json"
    volatility_model_weight = "Volatility_model.h5"


    btc_predict =BtcPredict()

    btc_predict.loadModel(Close_model,Close_model_weight,Close_model_id)
    btc_predict.loadModel(Volume_model,Volume_model_weight,Volume_model_id)
    btc_predict.loadModel(close_off_high_model,close_off_high_model_weight,Close_off_high_model_id)
    btc_predict.loadModel(volatility_model,volatility_model_weight,Volatility_model_id)
    
    data = pd.read_csv("lastdata.csv")   
    data=data.astype('float')
    data = np.array(data)
    data.shape = (1,10,4)
    print(data)
    window_len = 10
    predict_date=pd.read_csv('date.csv')

    test_set = pd.read_csv("test_set.csv")
    predict = []
    predict_price=[]
    i=0
    while i<152:
        Close_predict = btc_predict.predict(data,Close_model_id)
        Volume_predict = btc_predict.predict(data,Volume_model_id)
        Close_off_high_predict = btc_predict.predict(data,Close_off_high_model_id)
        Volatility_predict = btc_predict.predict(data,Volatility_model_id)
        price_predict = ((np.transpose(Close_predict)+1.021) * test_set['Close'].values[:-window_len])[0][-1]
        predict.append([Close_predict[0][0]])
        predict_price.append(price_predict)
        test_set['Close'].values[:-window_len][-1] = price_predict
        data= np.delete(data[0], 0,0)
        last_row = np.array([Close_predict[0][0], Volume_predict[0][0] 
                            , Close_off_high_predict[0][0] , Volatility_predict[0][0] ])
        data = np.concatenate((data, [last_row]), axis=0)
        data.shape = (1,10,4)
        i = i+1
    print(predict)

    print(len(predict_price))
    fig, ax1 = plt.subplots(1,1)
    ax1.set_xticks([datetime.date(2018,i+1,1) for i in range(12)])
    ax1.set_xticklabels([datetime.date(2018,i+1,1).strftime('%b %d %Y')  for i in range(12)])
    ax1.plot(predict_date.head(152)['date'].astype(datetime.datetime),
             predict_price, label='Predicted')
    
    ax1.set_title('Test Set: Single Timepoint Prediction',fontsize=13)
    ax1.set_ylabel('Price ($)',fontsize=12)
    ax1.set_xlabel('                20/4                20/5                20/6                20/7                20/8')
    ax1.legend(bbox_to_anchor=(0.1, 1), loc=2, borderaxespad=0., prop={'size': 14})
    plt.show() 