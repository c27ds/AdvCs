import backtrader as bt
import yfinance as yf
import pandas as pd

class Strategy(bt.Strategy):
    def __init__(self):
        # Indicators
        self.rsi = bt.indicators.RSI(self.data.close, period=14)
        self.ema12 = bt.indicators.EMA(self.data.close, period=12)
        self.ema26 = bt.indicators.EMA(self.data.close, period=26)
        self.macd = self.ema12 - self.ema26
        self.macd_signal = bt.indicators.EMA(self.macd, period=9)
        self.bbands = bt.indicators.BollingerBands(self.data.close, period=56, devfactor=2.0)
        self.adx = bt.indicators.AverageDirectionalMovementIndex(self.data)
        self.sma14 = bt.indicators.SimpleMovingAverage(self.data.close, period=14)
        self.sma28 = bt.indicators.SimpleMovingAverage(self.data.close, period=28)
        
        # Position sizing
        self.base_size = 180  # This corresponds to qty1 in Pine Script
        self.size = 0
        self.b = 0  # Buy counter
        self.s = 0  # Short counter
    
    def next(self):
        current_price = self.data.close[0]
        
        # Calculate Bollinger Bands percentage difference (filter)
        bbdiff = (self.bbands.lines.top[0] - self.bbands.lines.bot[0]) * 100 / current_price

        # Calculate dynamic position size based on ADX
        dynamic_size = int(self.base_size * (self.adx[0] / 25))

        # Long Entry Conditions
        if self.macd_signal[0] > 0.1 and self.rsi[0] > 70 and self.adx[0] > 27.5 and 2 < bbdiff < 5:
            if self.position.size <= 0:  # Close short if any, then go long
                self.close()
                self.s += 1
                self.debug("Closing Short Position")
            self.b += 1
            self.size = dynamic_size  # Adjust position size
            self.buy(size=self.size)
            self.debug(f'Buy Signal #{self.b} | Size: {self.size}')
        
        # Short Entry Conditions
        elif self.macd_signal[0] < -0.1 and self.rsi[0] < 30 and self.adx[0] > 27.5 and 2 < bbdiff < 5:
            if self.position.size >= 0:  # Close long if any, then go short
                self.close()
                self.b += 1
                self.debug("Closing Long Position")
            self.s += 1
            self.size = dynamic_size  # Adjust position size
            self.sell(size=self.size)
            self.debug(f'Short Signal #{self.s} | Size: {self.size}')
        
        # Exiting long position based on SMA crossover (14-period SMA crossing below 28-period SMA)
        if self.position.size > 0 and self.sma14[0] < self.sma28[0]:
            self.close()
            self.debug("Closing Long Position based on SMA crossover")
        
        # Exiting short position based on SMA crossover (14-period SMA crossing above 28-period SMA)
        elif self.position.size < 0 and self.sma14[0] > self.sma28[0]:
            self.close()
            self.debug("Closing Short Position based on SMA crossover")
    
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
        return {
            'Total Trades': total_trades,
            'Win Rate (%)': win_rate,
            'Profit Factor': profit_factor,
            'Total Profit': self.total_profit,
            'Total Loss': self.total_loss
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
    
    price_data = yf.Ticker(stock).history(period="1mo", interval="60m")
    price_data.index = pd.to_datetime(price_data.index)
    data = bt.feeds.PandasData(dataname=price_data)
    cerebro.adddata(data)
    
    # Add analyzers for performance
    cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='sharpe', timeframe=bt.TimeFrame.Days, annualize=True)
    cerebro.addanalyzer(bt.analyzers.DrawDown, _name='drawdown')
    cerebro.addanalyzer(PerformanceAnalyzer, _name="performance")
    
    print(f'Starting Portfolio Value: {cerebro.broker.getvalue()}')
    results = cerebro.run()
    print(f'Final Portfolio Value: {cerebro.broker.getvalue()}')
    
    # Print performance analysis
    performance = results[0].analyzers.performance.get_analysis()
    print("Performance Analysis:")
    for key, value in performance.items():
        print(f"{key}: {value}")
    
    cerebro.plot()

# Running the backtest with TSLA as an example
backtest("TSLA")
