import numpy as np
import pandas as pd
import yfinance as yf
from matplotlib import pyplot as plt
import backtrader as bt
import backtrader.feeds as btfeeds

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
        self.debug(f'Close: {self.data.close[0]:.2f}, RSI: {self.rsi[0]:.2f}, MACD: {self.macd[0]:.2f}, ADX: {self.adx[0]:.2f}, BBandsDIFF: {self.bbdiff[0]:.2f}')
        if (self.rsi[0] > 70 and self.macd_signal[0] > 0.1 and self.adx[0] > 27.5 and (self.bbdiff[0] < 5 and self.bbdiff[0] > 2)):
            if self.position.size < 0:
                self.close()
                self.debug("Closing Short Position")
                cash = self.broker.getvalue()
                print(cash)
            else:
                self.b += 1
                self.debug(f'Buy Signal #{self.b}')
                self.buy(size=10)
                cash = self.broker.getvalue()
                print(cash)

        elif (self.rsi[0] < 30 and self.macd_signal[0] < -0.1 and self.adx[0] > 27.5 and (self.bbdiff[0] < 5 and self.bbdiff[0] > 2)):
            if self.position.size > 0:
                self.close()
                self.debug("Closing Long Position")
                cash = self.broker.getvalue()
                print(cash)
            else:
                self.s += 1
                self.debug(f'Short Signal #{self.s}')
                self.sell(size=10)
                cash = self.broker.getvalue()
                print(cash)
        

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
        if trade.isclosed:
            pnl = trade.pnl
            self.returns.append(pnl)
            print(pnl)
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
            'Total Profit': self.total_profit,
            'Total Loss': self.total_loss,
            'Sharpe Ratio': sharpe_ratio,
            'Max Drawdown': max_drawdown
        }
    
    def notify_order(self, order):
        if order.status in [order.Completed]:
            if order.isbuy():
                print(f"BUY EXECUTED: Price={order.executed.price:.2f}, Size={order.executed.size}")
            elif order.issell():
                print(f"SELL EXECUTED: Price={order.executed.price:.2f}, Size={order.executed.size}")
        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            print("Order Canceled/Margin/Rejected")


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