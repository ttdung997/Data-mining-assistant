# -*- coding: utf-8 -*-
import pandas as pd
import time
import numpy as np

#import linalg de tinh tri rieng va vecto tieng
from numpy import linalg as LA


# #load excel dataframe
# fields =['Power (W)','Laser Speed (mm/s)','Hatch Spacing (mm)','Layer Thickness (mm)','Energy Density (J/mm^3)','XY/YZ Density Avg']


fields =['Power (W)','Laser Speed (mm/s)','Hatch Spacing (mm)','Layer Thickness (mm)','Energy Density (J/mm^3)']
# doc du lieu tu file excel
myDf = pd.read_excel('Data.xlsx')

#load cac cot du lieu tu array
myDf = myDf[fields]

#goi cac cau lenh tinh gia tri rieng va phuong sai
print("Ma trận hiệp phương sai: ")
cov =np.array(myDf.cov())

print(myDf.cov())
print("____________________________________________________")
print("Các cặp trị riêng và vecto riêng tương ứng: ")
w, v = LA.eig(cov)

#tinh thanh phan chinh cua chuoi

for i in range(0,len(w)):
	print("w_"+str(i+1)+" = "+str(round(w[i],2))+",v_"+str(i+1)+" = "+str(np.around(v[i],decimals=1)))


print("Các thành phần chính tương ứng: ")
for i in range(0,len(v)):
	output = "Y_"+str(i+1)+" = "
	for j in range(0,len(v[0])):
		if j == 0 or '-' in str(round(v[i][j],2)):
			output = output + str(round(v[i][j],2)) + "X_"+str(j+1)
		else:
			output = output +" + "+ str(round(v[i][j],2)) + "X_"+str(j+1)

	print(output)
w2 = w/sum(w)


print("Độ sai số tương ứng: ")
print(np.around(w2, decimals=2))

# tinh he so tuong quan
print("________________________________________________________")
print(myDf.corr())