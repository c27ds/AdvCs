import ta
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime
import yfinance as yf
import backtrader as bt
import backtrader.feeds as btfeeds

class Strategy(bt.Strategy):
    def __init__(self):
        self.rsi = bt.indicators.RSI(self.data.close,period=14)
        self.ema12 = bt.indicators.EMA(self.data.close,period=12)
        self.ema26 = bt.indicators.EMA(self.data.close,period=26)
        self.macd = self.ema12 - self.ema26
        self.adx = bt.indicators.AverageDirectionalIndex
    def next(self):
        self.log('Close, %.2f' % self.dataclose[0])

        if self.rsi < 30 and self.macd > 0 and self.position != True and self.adx > 25:
            self.buy()
        elif self.rsi >70 and self.macd < 0 and self.position == True and self.adx > 25:
            self.sell()

def backtest(ticker):
    cerebro = bt.Cerebro()
    cerebro.addstrategy(Strategy)
    cerebro.broker.setcash(100000.0)
    data = btfeeds.YahooFinanceCSVData(dataname='yfdata.csv',fromdate=datetime.datetime(2024, 10, 9), todate=datetime.datetime(2024, 11, 9))
    cerebro.adddata(data)
    print('Starting Portfolio Value: %.2f' % cerebro.broker.getvalue())
    cerebro.run()
    print('Final Portfolio Value: %.2f' % cerebro.broker.getvalue())
backtest("AAPL")


# def ema(data,length):
#     return data.ewm(span=length).mean()

# def get_macd(data):
#     data['EMA_12'] = ema(data['Close'], 12)
#     data['EMA_26'] = ema(data['Close'], 26)
#     data['MACD'] = data['EMA_12'] - data['EMA_26']
#     data = data.drop(columns=['EMA_12', 'EMA_26'])
#     return data['MACD']

# def get_rsi(data):
#     changes = data['Close'].diff()
#     avg_gains = (changes.where(changes > 0, 0)).rolling(window=14).mean()
#     avg_losses = (abs(changes).where(changes < 0, 0)).rolling(window=14).mean()
#     rs = avg_gains/avg_losses
#     rsi = 100 - (100 / (1 + rs))
#     return rsi


# def calculate_indicators_5min(ticker):
#     data = yf.download(ticker, start="2024-10-05", end="2024-11-05", interval="5m")

#     data['MACD'] = get_macd(data)
#     data['RSI'] = get_rsi(data)

#     data = data.dropna()

#     signals = []
#     for i in range(len(data)):
#         macd = data['MACD'].iloc[i]
#         rsi = data['RSI'].iloc[i]

#         if rsi < 30 and macd > 0:
#             signals.append((data.index[i], "Buy",rsi,macd))
#         elif rsi > 70 and macd < 0:
#             signals.append((data.index[i], "Sell",rsi,macd))
#     for signal in signals:
#         print(f"{signal[0]} - {signal[1]} because RSI = {signal[2]} and MACD = {signal[3]}")
    
#     return signals

# ticker = "TSLA"
# calculate_indicators_5min(ticker)

