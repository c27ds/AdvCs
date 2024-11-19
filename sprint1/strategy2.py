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