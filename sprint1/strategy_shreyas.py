import ta
import numpy as np
import pandas as pd
import yfinance as yf
import backtrader as bt
from datetime import datetime
from datetime import timedelta
import mplfinance as mpf
import talib as ta
import matplotlib.pyplot as plt


class Strategy(bt.Strategy):
    def __init__(self):
        self.rsi = bt.indicators.RSI(self.data.close, period=14)
        self.ema12 = bt.indicators.EMA(self.data.close, period=12)
        self.ema26 = bt.indicators.EMA(self.data.close, period=26)
        self.macd = self.ema12 - self.ema26
        self.bbands = bt.indicators.BollingerBands(self.data.close,period=20,devfactor=2.0)
        self.adx = bt.indicators.AverageDirectionalMovementIndex(self.data)
        self.atr = bt.indicators.AverageTrueRange(self.data,period=14)
        self.bbdif = self.bbands.lines.top - self.bbands.lines.bot
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

class PerformanceAnalyzer(bt.Analyzer):
    def __init__(self):
        self.returns = []
        self.win_trades = 0
        self.loss_trades = 0
        self.total_profit = 0
        self.total_loss = 0

    def notify_trade(self, trade):
        if trade.isclosed:
            pnl = trade.pnl
            self.returns.append(pnl)
            if pnl > 0:
                self.win_trades += 1
                self.total_profit += pnl
            else:
                self.loss_trades += 1
                self.total_loss += abs(pnl)

    def get_analysis(self):
        total_trades = self.win_trades + self.loss_trades
        win_rate = (self.win_trades / total_trades * 100) if total_trades > 0 else 0
        profit_factor = (self.total_profit / self.total_loss) if self.total_loss > 0 else 0
        sharpe_ratio = self.strategy.analyzers.sharpe.get_analysis().get("sharperatio")
        max_drawdown = (self.strategy.analyzers.drawdown.get_analysis().get("max").get("drawdown"))*100

        return {
            'Total Trades': total_trades,
            'Win Rate (%)': win_rate,
            'Profit Factor': profit_factor,
            'Sharpe Ratio': sharpe_ratio,
            'Max Drawdown': max_drawdown,
        }

def backtest():
    cerebro = bt.Cerebro()
    cerebro.addstrategy(Strategy)
    cerebro.broker.setcash(100000.0)
    price_data = yf.Ticker('TSLA').history(period="5d", interval="1m")
    price_data.index = pd.to_datetime(price_data.index)
    data = bt.feeds.PandasData(dataname=price_data)
    cerebro.adddata(data)
    cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='sharpe', timeframe=bt.TimeFrame.Days, annualize=True)
    cerebro.addanalyzer(bt.analyzers.DrawDown, _name='drawdown')
    cerebro.addanalyzer(PerformanceAnalyzer, _name="performance")
    print(f'Starting Portfolio Value: {cerebro.broker.getvalue()}')
    results = cerebro.run()
    print(f'Final Portfolio Value: {cerebro.broker.getvalue()}')
    performance = results[0].analyzers.performance.get_analysis()
    print("Analysis:")
    for i in performance:
        print(f"{i}: {performance[i]}")
    cerebro.plot()
backtest()
plt.show()