import pandas as pd
import time

# Doc du lieu bang tu website tu bang pandas
bitcoin_market_info = pd.read_html("https://coinmarketcap.com/currencies/bitcoin/historical-data/?start=20130428&end="+time.strftime("%Y%m%d"))[0]

# Luu du lieu su dung pandas
bitcoin_market_info.to_csv("btc.csv")

# Dung lenh assign de thay doi mot cot du lieu trong pandas, o day quy doi truong thoi gian ve
# dinh dang time trong python
bitcoin_market_info = bitcoin_market_info.assign(Date=pd.to_datetime(bitcoin_market_info['Date']))

# Chuyen cac gia tri "-" ve 0
bitcoin_market_info.loc[bitcoin_market_info['Volume']=="-",'Volume']=0

# Chuyen co ve kien int
bitcoin_market_info['Volume'] = bitcoin_market_info['Volume'].astype('int64')

# Dung lenh head() de in ra cac dong dau cua du lieu

print(bitcoin_market_info.head())

#   Date     Open*      High       Low   Close**       Volume    Market Cap
# 0 2019-09-19  10200.50  10295.67   9851.69  10266.41  19937691247  184240949577
# 1 2019-09-18  10247.80  10275.93  10191.47  10198.25  16169268880  182998899156
# 2 2019-09-17  10281.51  10296.77  10199.74  10241.27  15304603363  183748515828
# 3 2019-09-16  10347.22  10386.87  10189.74  10276.79  15160167779  184366827555
# 4 2019-09-15  10356.47  10387.03  10313.09  10347.71  12043433567  185618174384



# dung lenh describe() de in ra thong so thong ke cua du lieu
print(bitcoin_market_info.describe())

#               Open*          High  ...        Volume    Market Cap
# count   2336.000000   2336.000000  ...  2.336000e+03  2.336000e+03
# mean    2843.913502   2925.563399  ...  3.062316e+09  4.814084e+10
# std     3696.624332   3822.722354  ...  5.814256e+09  6.410293e+10
# min       68.500000     74.560000  ...  0.000000e+00  7.784112e+08
# 25%      364.975000    377.010000  ...  2.120968e+07  5.016256e+09
# 50%      664.045000    675.330000  ...  7.796925e+07  9.990399e+09
# 75%     4586.237500   4664.682500  ...  4.163460e+09  7.609100e+10
# max    19475.800000  20089.000000  ...  4.510573e+10  3.265025e+11