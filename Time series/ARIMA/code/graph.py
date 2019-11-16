import matplotlib.pyplot as plt
import pandas as pd
import time
import datetime
from matplotlib.dates import (YEARLY, DateFormatter,
                              rrulewrapper, RRuleLocator, drange)
# tickerList =['VCB','BID','CTG','STB','EIB','VPB','TCB','MBB','ACB','SHB']


tickerList =['VCB']
for ticker in tickerList:
	dataFrame = pd.read_excel("data/excel/"+ticker+".xlsx")
	data = dataFrame.values

	time = [x[1] for x in data]
	close = [x[5] for x in data]
	high = [x[3] for x in data]
	low =[x[4] for x in data]

	ema =  pd.read_csv("tool/EMA_"+ticker+".csv")
	macd =  pd.read_csv("tool/MACD_"+ticker+".csv")
	rule = rrulewrapper(YEARLY, bymonthday=1, interval=1)
	loc = RRuleLocator(rule)
	formatter = DateFormatter('%m/%d/%y')
	fig, ax1 = plt.subplots(1,1,figsize=(12,6))

	# print(time)
	date2 = datetime.datetime.strptime(str(time[0]),'%Y%m%d')
	date1 = datetime.datetime.strptime(str(time[-1]),'%Y%m%d')
	delta = datetime.timedelta(days=1)

	dates = drange(date1, date2, delta)

	ax1.plot(dates[-len(high):],
			 (high),
			  label='High')
	ax1.plot(dates[-len(high):],
			 (close),
			  label='Price')

	ax1.plot(dates[-len(high):],
			 (low),
			  label='low')
	ax1.plot(dates[-len(high):],
			 (ema[-len(high):]),
			  label='EMA')
	ax1.plot(dates[-len(high):],
			 (macd[-len(high):]),
			  label='MACD')
	ax1.set_title('History price',fontsize=13)
	ax1.set_ylabel('Price (VND)',fontsize=12)
	ax1.legend(bbox_to_anchor=(0.1, 1), loc=2, borderaxespad=0., prop={'size': 14})

	ax1.xaxis.set_major_locator(loc)
	ax1.xaxis.set_major_formatter(formatter)
	ax1.xaxis.set_tick_params(rotation=10, labelsize=10)

	# plt.show()
	plt.savefig("public/img/graph/"+ticker+".png")