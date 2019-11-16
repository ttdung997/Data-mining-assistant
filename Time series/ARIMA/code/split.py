import pandas as pd

# Tên ngân hàng - Mã cô phiêu
# BIDV 		BID
# Vietinbank	CTG
# Sacobank	STB
# Eximbank	EIB
# VPBANK		VPB
# Techcombank	TCB
# MBbank		MBB
# ACB		ACB
# SHB		SHB

# tickerList =['VCB','BID','CTG','STB','EIB','VPB','TCB','MBB','ACB','SHB']

# tickerList =['ACB','SHB']


tickerList =['VCB']
data =pd.read_csv("data1.csv")
print(list(data))


for ticker in tickerList:

	splitData = data[(data['<Ticker>'] == ticker )]

	splitData = splitData[(splitData['<DTYYYYMMDD>'] >20160000 )]
	splitData.to_csv("vcb.csv")
	# writer = pd.ExcelWriter('data/excel/'+ticker+'.xlsx')


	# splitData.to_excel(writer,'Sheet1')
	# writer.save()
