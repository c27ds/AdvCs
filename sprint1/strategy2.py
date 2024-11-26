import numpy as np
import pandas as pd
import yfinance as yf
from matplotlib import pyplot as plt
import backtrader as bt
import backtrader.feeds as btfeeds
import collections

MAINSIGNALS = collections.OrderedDict(
    (('longshort', bt.SIGNAL_LONGSHORT),
     ('longonly', bt.SIGNAL_LONG),
     ('shortonly', bt.SIGNAL_SHORT),)
)


EXITSIGNALS = {
    'longexit': bt.SIGNAL_LONGEXIT,
    'shortexit': bt.SIGNAL_LONGEXIT,
}


class Strategy(bt.Strategy):
    def __init__(self):
        self.rsi = bt.indicators.RSI(self.data.close, period=14)
        self.ema12 = bt.indicators.EMA(self.data.close, period=12)
        self.ema26 = bt.indicators.EMA(self.data.close, period=26)
        self.macd = self.ema12 - self.ema26
        self.macd_signal = bt.indicators.EMA(self.macd, period=9)
        self.bbands = bt.indicators.BollingerBands(self.data.close, period=56, devfactor=2.0)
        self.adx = bt.indicators.AverageDirectionalMovementIndex(self.data)
        self.atr = bt.indicators.AverageTrueRange(self.data, period=14)
        self.bbdiff = (self.bbands.lines.top - self.bbands.lines.bot) * 100 / self.data.close[0]
        self.size = 10
        self.b = 0
        self.s = 0

    def next(self):
        self.debug(f'Close: {self.data.close[0]:.2f}, RSI: {self.rsi[0]:.2f}, MACD: {self.macd[0]:.2f}, ADX: {self.adx[0]:.2f}')
        
        # Buy signal
        if (self.rsi[0] > 70 and self.macd_signal[0] > 0.1 and self.adx[0] > 27.5 and (self.bbdiff[0] < 15 and self.bbdiff[0] > 3)):
            if self.position.size < 0:  # If in a short position, close it first
                self.close()
                self.debug("Closing Short Position")
            self.b += 1
            self.debug(f'Buy Signal #{self.b}')
            self.buy(size=self.size)  # Example size; adjust as needed

        # Sell signal and shorting logic
        elif (self.rsi[0] < 30 and self.macd_signal[0] < -0.1 and self.adx[0] > 27.5 and (self.bbdiff[0] < 15 and self.bbdiff[0] > 3)):
            if self.position.size > 0:  # If in a long position, close it first
                self.close()
                self.debug("Closing Long Position")
            elif self.position.size == 0:  # Open a short position if no position exists
                self.s += 1
                self.debug(f'Short Signal #{self.s}')
                self.sell(size=self.size)

    def debug(self, text):
        print(text)
        
    def totaltrades(self):
        return self.b + self.s,

class PerformanceAnalyzer(bt.Analyzer):
    def __init__(self):
        self.returns = []
        self.win_trades = 0
        self.loss_trades = 0
        self.total_profit = 0
        self.total_loss = 0

    def notify_trade(self, trade):
        if trade.isclosed:  # Only consider fully closed trades
            pnl = trade.pnl  # Profit and loss from the trade
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

        return {
            'Total Trades': total_trades,
            'Win Rate (%)': win_rate,
            'Profit Factor': profit_factor,
            'Total Profit': self.total_profit,
            'Total Loss': self.total_loss,
        }
    def notify_order(self, order):
        if order.status in [order.Completed]:
            if order.isbuy():
                self.debug(f"BUY EXECUTED: Price={order.executed.price:.2f}, Size={order.executed.size}")
            elif order.issell():
                self.debug(f"SELL EXECUTED: Price={order.executed.price:.2f}, Size={order.executed.size}")
        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            self.debug("Order Canceled/Margin/Rejected")


def backtest(stock):
    cerebro = bt.Cerebro()
    cerebro.addstrategy(Strategy)
    cerebro.broker.setcash(100000.0)
    price_data = yf.Ticker(stock).history(period="5d", interval="1m")
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
    print(performance)
    print("Analysis:")
    for i in performance:
        print(f"{i}: {performance[i]}")
    cerebro.plot()
stock = "TSLA"
backtest(stock)