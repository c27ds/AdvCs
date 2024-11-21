import talib as ta
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from datetime import timedelta
import yfinance as yf
import mplfinance as mpf
import talib as ta

price = yf.Ticker('AAPL').history(period="5d", interval="1m")
price.index = pd.to_datetime(price.index)
price['SMA_14'] = ta.SMA(price['Close'], timeperiod=14)
price['SMA_28'] = ta.SMA(price['Close'], timeperiod=28)
price['SMA_50'] = ta.SMA(price['Close'], timeperiod=50)

price['MACD'], price['MACD_signal'], price['MACD_hist'] = ta.MACD(price['Close'], fastperiod=12, slowperiod=26, signalperiod=9)

price['ADX'] = ta.ADX(price['High'], price['Low'], price['Close'], timeperiod=14)

price['BB_upper'], price['BB_middle'], price['BB_lower'] = ta.BBANDS(price['Close'], timeperiod=56, nbdevup=2, nbdevdn=2, matype=0)

price['RSI'] = ta.RSI(price['Close'], timeperiod=14)
rsi_above_70 = price['RSI'].where(price['RSI'] >= 70)
rsi_below_30 = price['RSI'].where(price['RSI'] <= 30)
rsi_between_30_70 = price['RSI'].where((price['RSI'] >= 30) & (price['RSI'] <= 70))

# Prepare additional plots
plots = [
    mpf.make_addplot(price['SMA_14'], color='blue'),
    mpf.make_addplot(price['SMA_28'], color='red'),
    mpf.make_addplot(price['SMA_50'], color='white'),
    mpf.make_addplot(price['BB_upper'], color='gray'),
    mpf.make_addplot(price['BB_middle'], color='gray'),
    mpf.make_addplot(price['BB_lower'], color='gray'),
    mpf.make_addplot(price['MACD'], panel=1, color='aqua'),
    mpf.make_addplot(price['MACD_signal'], panel=1, color='purple'),
    mpf.make_addplot(price['ADX'], panel=2, color='teal'),
    mpf.make_addplot(rsi_above_70, panel=3, color='green'),
    mpf.make_addplot(rsi_below_30, panel=3, color='red'),
    mpf.make_addplot(rsi_between_30_70, panel=3, color='gray')
]

# Plot the data
mpf.plot(price, type='candle', style='yahoo', title='AAPL Candlestick Chart', volume=True, addplot=plots)