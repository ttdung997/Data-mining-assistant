import time
import itertools
import os, re
import matplotlib
import numpy as np
from multiprocessing import Process, Manager
from keras.models import model_from_json
from keras.preprocessing import sequence
from scipy import interp
from datetime import datetime
from library.regularizeddeepautoencoder import RegularizedDeepAutoencoder
from time import gmtime, strftime
import tensorflow as tf

class HttpPredict():
    def _init_(self):
       print("loaded model")
    #Load Model
    def loadModelBinary(self,model): 
        print(strftime("%Y-%m-%d %H:%M:%S", gmtime()))
        self.ae = RegularizedDeepAutoencoder()
        self.ae.load_model(model)
        self.graph = tf.get_default_graph()
        print("Loaded model from disk")
        self.ae.threshold = 0.6817
    #Preprocess  

    def SplitParam(self,s):
        pos = 1
        while pos < len(s):
            if s[pos].isalpha() or unicode(s[pos]).isnumeric():
                pos= pos + 1
            else:
                break
        if pos >= len(s):
            return(-1,-1,-1,s)
        return (pos,s[pos],s[:pos], s[pos+1:])

    def tokenurl(self, request):
        try:
            postdata = 1
            # print(request)
            try:
                postdata = request.split("--START-BODY--", 1)[1]
                request = request.split("--START-BODY--", 1)[0]
                # print("how about here")
            except:
                postdata = 1
            method = request.split("/", 1)[0]
            request = request.split("/", 1)[1]
            upperLetters = sum(1 for c in request if c.isupper())
            request = request.lower()
            getinpost = 0

            # print("it here1")
            param = ''
            try:
                path = request.split("?", 1)[0]
                param = request.split("?", 1)[1]
            except:
                if 'GET' in method: 
                    fullURL = (method+"/"+request)
                else:
                    getinpost =1

            newparam = ''
            pos = 0
            if getinpost ==0:
                while pos <len(param):
                    lentem = len(param)
                    pos,anphakey,temparam, param = self.SplitParam(param)
                    if pos != -1:
                        pos = 0 
                        if anphakey == '=':
                            newparam = newparam + temparam + "="
                        else:
                            try:
                                a = float(temparam)
                                newparam = newparam + "<NV>"+ anphakey
                            except:
                                newparam = newparam + "<SV>"+ anphakey
                    else:
                        try:
                            a = float(param)
                            newparam = newparam + "<NV>"
                        except:
                            newparam = newparam + "<SV>"
                        break
            pos =0
            # print("it here")
            if postdata != 1:
                #print("it here")
                newparam = newparam + "<BD>" 
                while pos <len(postdata):
                    lentem = len(postdata)
                    pos,anphakey,temparam, postdata = self.SplitParam(postdata)
                    if pos != -1:
                        pos = 0 
                        if anphakey == '=':
                            newparam = newparam + temparam + "="
                        else:
                            try:
                                a = float(temparam)
                                newparam = newparam + "<NV>"+ anphakey
                            except:
                                newparam = newparam + "<SV>"+ anphakey
                    else:
                        try:
                            a = float(postdata)
                            newparam = newparam + "<NV>"
                        except:
                            newparam = newparam + "<SV>"
                        break
            fullURL = (method+"/"+path+"?"+newparam)
                #break
        except:
            fullURL = 0
        return fullURL

    def preprocess(self, data):
        try:
            # self.tokenUrl(data)
            data = self.tokenurl(data) 
            print(data)
            if data == 0:
                print("this is error!")
                return 0

            x = [[float(self.valid_chars[i])/127 for i in data]]
            x = sequence.pad_sequences(x, maxlen=self.maxlen,dtype='float')
        except:
            x = None
        
        return x

    #Preprocess 
    def predict(self, data):
            # print(data)
        try:
            print(self.ae.predict(data))

            anomaly_predict = self.ae.predict(data)[0]
            # print(anomaly_predict)
            # print(anomaly_predict)
            print("threeholsd: ",self.ae.threshold)
            # print(anomaly_predict)
            # print(self.ae.threshold)
            if anomaly_predict > self.ae.threshold:
                predict_score = 1
            else:
                predict_score = 0

        except:
            print("error")
            predict_score = 1
        # # print("score: ", predict_score)
        return predict_score
   
    def loadModelInit(self):
        self.valid_chars =   valid_chars = {'\x00': 1, '\x04': 5, '\x08': 9,
			'\x0c': 13, '\x10': 17, '\x14': 21, '\x18': 25, '\x1c': 29, ' ': 33,
			'$': 37, '(': 41, ',': 45, '0': 49, '4': 53, '8': 57, '<': 61,
			'@': 65, 'D': 69, 'H': 73, 'L': 77, 'P': 81, 'T': 85, 'X': 89,
			'\\': 93, '`': 97, 'd': 101, 'h': 105, 'l': 109, 'p': 113,
			't': 117, 'x': 121, '|': 125, '\x03': 4, '\x07': 8, '\x0b': 12,
			'\x0f': 16, '\x13': 20, '\x17': 24, '\x1b': 28, '\x1f': 32,
			'#': 36, "'": 40, '+': 44, '/': 48, '3': 52, '7': 56, ';': 60,
			'?': 64, 'C': 68, 'G': 72, 'K': 76, 'O': 80, 'S': 84, 'W': 88,
			'[': 92, '_': 96, 'c': 100, 'g': 104, 'k': 108, 'o': 112,
			's': 116, 'w': 120, '{': 124, '\x7f': 128, '\x02': 3,
			'\x06': 7, '\n': 11, '\x0e': 15, '\x12': 19, '\x16': 23,
			'\x1a': 27, '\x1e': 31, '"': 35, '&': 39, '*': 43,
			'.': 47, '2': 51, '6': 55, ':': 59, '>': 63,
			'B': 67, 'F': 71, 'J': 75, 'N': 79, 'R': 83,
			'V': 87, 'Z': 91, '^': 95, 'b': 99, 'f': 103, 'j': 107,
			'n': 111, 'r': 115, 'v': 119, 'z': 123, '~': 127,
			'\x01': 2, '\x05': 6, '\t': 10, '\r': 14, '\x11': 18,
			'\x15': 22, '\x19': 26, '\x1d': 30, '!': 34, '%': 38,
			')': 42, '-': 46, '1': 50, '5': 54, '9': 58, '=': 62,
			'A': 66, 'E': 70, 'I': 74, 'M': 78, 'Q': 82, 'U': 86,
			'Y': 90, ']': 94, 'a': 98, 'e': 102, 'i': 106, 'm': 110,
			'q': 114, 'u': 118, 'y': 122, '}': 126}


        self.maxlen = 320
        
        basePath="."
        model_dir_path = './models'
        self.loadModelBinary(model_dir_path)
  
           
                
           

                    

                        
        			
                    

        			 
    
 
