import ta
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from datetime import timedelta
import yfinance as yf
#all imports
ticker = "TSLA"
#price = yf.Ticker('TSLA').history(period='1y').reset_index()[['Date', 'Close']]
data = yf.download(ticker,start="2024-10-05",end="2024-11-05", interval="5m")
data["MACD"] = ta.trend.MACD(data["Close"]).macd()
data["MACD_Signal"] = ta.trend.MACD(data["Close"]).macd_signal()
macd_data = data[["MACD","MACD_Signal","MACD_Hist"]].dropna()
print(macd_data)
#for x in range(0,len(macd_data))

#plt.plot(price['Date'], price['Close'])
