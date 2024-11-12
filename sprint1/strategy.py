from math import *
from ta import *
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


plots = [
    mpf.make_addplot(price['SMA_14'], color='blue', title='SMA 14'),
    mpf.make_addplot(price['SMA_28'], color='red', title='SMA 28'),
    mpf.make_addplot(price['SMA_50'], color='white', title='SMA 50'),
    mpf.make_addplot(price['BB_upper'], color='gray', title='BB Upper'),
    mpf.make_addplot(price['BB_middle'], color='gray', title='BB Middle'),
    mpf.make_addplot(price['BB_lower'], color='gray', title='BB Lower'),
    mpf.make_addplot(price['MACD'], panel=1, color='aqua', title='MACD'),
    mpf.make_addplot(price['MACD_signal'], panel=1, color='purple', title='MACD Signal'),
    mpf.make_addplot(price['ADX'], panel=2, color='teal', title='ADX'),
    mpf.make_addplot(price['RSI'], panel=3, color='yellow', title='RSI')
]


mpf.plot(price, type='candle', style='yahoo', title='AAPL Candlestick Chart', volume=True, addplot=apds)