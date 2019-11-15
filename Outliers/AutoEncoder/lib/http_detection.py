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

    def urltoken(self,request):
        try:
            postdata = 1

            #Split data into URL, BODY, HEADER
            try:
                headerfield = request.split("--START-HEADER--:", 1)[1]
                request = request.split("--START-HEADER--:", 1)[0]
            except:
                headerfield = 1

            try:
                postdata = request.split("--START-BODY--:", 1)[1]
                request = request.split("--START-BODY--:", 1)[0]
            except:
                postdata = 1


            #RAW_header

            #export mothod 
            method = request.split("/", 1)[0]
            request = request.split("/", 1)[1]
            upperLetters = sum(1 for c in request if c.isupper())
            request = request.lower()


            getinpost = 0
            newheader = ""
            if headerfield != 1:
                fieldArr = headerfield.replace(" ", "").split("<HD>")
                # print(fieldArr)
                # print("________________")
                newheader = newheader + "<HEADER>"

                pos =0
                while pos <len(fieldArr[0]):
                    lentem = len(fieldArr[0])
                    pos,anphakey,temparam, fieldArr[0] = self.SplitParam(fieldArr[0])
                    if pos != -1:
                        pos = 0 
                        if anphakey not in "/.(;)":
                            newheader = newheader + anphakey
                    else:
                        break

                newheader = newheader +"<HD>" 
                pos =0
                while pos <len(fieldArr[1]):
                    lentem = len(fieldArr[1])
                    pos,anphakey,temparam, fieldArr[1] = self.SplitParam(fieldArr[1])
                    if pos != -1:
                        pos = 0 
                        try:
                            a = float(temparam)
                            newheader = newheader + "<NV>"+ anphakey
                        except:
                            newheader = newheader + "<SV>"+ anphakey
                    else:
                        try:
                            a = float(fieldArr[1])
                            newheader = newheader + "<NV>"
                        except:
                            newheader = newheader + "<SV>"
                        break
                newheader = newheader +"<HD>" 
                if "http" in fieldArr[2] or "Http" in fieldArr[2]:
                    newheader = newheader + "<HTTP>"
                else:
                    newheader = newheader + fieldArr[2] + ""

            try:
                path = request.split("?", 1)[0]
                param = request.split("?", 1)[1]

            except:
                # return (method+"/"+request+newheader).replace(",","") 
                return (method+newheader).replace(",","")   
            newparam = ''
            pos = 0
            while pos <len(param):
                lentem = len(param)
                pos,anphakey,temparam, param = self.SplitParam(param)
                if pos != -1:
                    pos = 0 
                    if '=' in anphakey:
                        newparam = newparam + temparam + anphakey
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
            if postdata != 1:
                newparam = newparam + "<BODY>" 
                while pos <len(postdata):
                    lentem = len(postdata)
                    pos,anphakey,temparam, postdata = self.SplitParam(postdata)
                    if pos != -1:
                        pos = 0 
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
                #break

            
            return (method+"/"+newparam+newheader).replace(",","")  
        except:
            return 0

    def preprocess(self, data):
        try:
            # self.tokenUrl(data)
            data = self.urltoken(data) 
            # print(data)
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
            # print(self.ae.predict(data))
            anomaly_predict = self.ae.predict(data)[0]
            # print(anomaly_predict)
            # print(anomaly_predict)
            # print("threeholsd: ",self.ae.threshold)
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
        self.valid_chars =   {'\x00': 1, '\x02': 3, '\x04': 5, '\x06': 7, '\x08': 9,
        '\n': 11, '\x0c': 13, '\x0e': 15, '\x10': 17, '\x12': 19, '\x14': 21, '\x16': 23,
        '\x18': 25, '\x1a': 27, '\x1c': 29, '\x1e': 31, ' ': 33, '"': 35, '$': 37, '&': 39,
        '(': 41, '*': 43, ',': 45, '.': 47, '0': 49, '2': 51, '4': 53, '6': 55, '8': 57,
        ':': 59, '<': 61, '>': 63, '@': 65, 'B': 67, 'D': 69, 'F': 71, 'H': 73, 'J': 75,
        'L': 77, 'N': 79, 'P': 81, 'R': 83, 'T': 85, 'V': 87, 'X': 89, 'Z': 91, '\\': 93,
        '^': 95, '`': 97, 'b': 99, 'd': 101, 'f': 103, 'h': 105, 'j': 107, 'l': 109, 
        'n': 111, 'p': 113, 'r': 115, 't': 117, 'v': 119, 'x': 121, 'z': 123, '|': 125,
        '~': 127, '\x01': 2, '\x03': 4, '\x05': 6, '\x07': 8, '\t': 10, '\x0b': 12, 
        '\r': 14, '\x0f': 16, '\x11': 18, '\x13': 20, '\x15': 22, '\x17': 24, '\x19': 26,
        '\x1b': 28, '\x1d': 30, '\x1f': 32, '!': 34, '#': 36, '%': 38, "'": 40, ')': 42,
        '+': 44, '-': 46, '/': 48, '1': 50, '3': 52, '5': 54, '7': 56, '9': 58, ';': 60,
        '=': 62, '?': 64, 'A': 66, 'C': 68, 'E': 70, 'G': 72, 'I': 74, 'K': 76, 'M': 78,
        'O': 80, 'Q': 82, 'S': 84, 'U': 86, 'W': 88, 'Y': 90, '[': 92, ']': 94, '_': 96,
        'a': 98, 'c': 100, 'e': 102, 'g': 104, 'i': 106, 'k': 108, 'm': 110, 'o': 112,
        'q': 114, 's': 116, 'u': 118, 'w': 120, 'y': 122, '{': 124, '}': 126, 
        '\x7f': 128}

        self.maxlen = 396
        
        basePath="."
        model_dir_path = './models'
        self.loadModelBinary(model_dir_path)
  
           
                
           

                    

                        
        			
                    

        			 
    
 
