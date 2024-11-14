import ta
import numpy as np
import pandas as pd
import yfinance as yf
import backtrader as bt

class Strategy(bt.Strategy):
    def __init__(self):
        self.rsi = bt.indicators.RSI(self.data.close, period=14)
        self.ema12 = bt.indicators.EMA(self.data.close, period=12)
        self.ema26 = bt.indicators.EMA(self.data.close, period=26)
        self.macd = self.ema12 - self.ema26
        self.bbands = bt.indicators.BollingerBands(self.data.close,period=20,devfactor=2.0)
        self.adx = bt.indicators.AverageDirectionalMovementIndex(self.data)
        self.atr = bt.indicators.AverageTrueRange(self.data,period=14)
    def determine_position_size(self):
        pass
    # Add our function for determining position size

    def next(self):
        # position_size = self.determine_position_size()
        self.debug(f'Close: {self.data.close[0]:.2f}, RSI: {self.rsi[0]:.2f}, MACD: {self.macd[0]:.2f}, ADX: {self.adx[0]:.2f}')
        if self.rsi < 40 and self.macd > 0 and not self.position and self.adx > 25:
            self.debug('Buy Signal')
            self.buy() # Later self.buy(size = position.size)
        elif self.rsi > 60 and self.macd < 0 and self.position and self.adx > 25:
            self.debug('Sell Signal')
            self.sell() # Later self.sell(size = position.size)
        else:
            pass

    def debug(self, text):
        print(text)

def backtest(stock):
    cerebro = bt.Cerebro()
    cerebro.addstrategy(Strategy)
    cerebro.broker.setcash(100000.0)
    price = yf.Ticker(stock).history(period="5d", interval="1m")
    price.index = pd.to_datetime(price.index)
    data = bt.feeds.PandasData(dataname=price)
    cerebro.adddata(data)
    print('Starting Portfolio Value: %.2f' % cerebro.broker.getvalue())
    cerebro.run()
    print('Final Portfolio Value: %.2f' % cerebro.broker.getvalue())

stock = "TSLA"
backtest(stock)