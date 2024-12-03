import backtrader as bt
import datetime 
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import mplfinance as mpf

import mplfinance as mpf

class Strategy(bt.Strategy):
    params = (
        ('base_size', 180),  # Base quantity for trades
    )

    def __init__(self):
        # Indicators
        self.rsi = bt.indicators.RSI(self.data.close, period=14)
        self.ema12 = bt.indicators.EMA(self.data.close, period=12)
        self.ema26 = bt.indicators.EMA(self.data.close, period=26)
        self.macd = self.ema12 - self.ema26
        self.macd_signal = bt.indicators.EMA(self.macd, period=9)
        self.bbands = bt.indicators.BollingerBands(self.data.close, period=56, devfactor=2.0)
        self.adx = bt.indicators.AverageDirectionalMovementIndex(self.data, period=14)
        self.sma14 = bt.indicators.SimpleMovingAverage(self.data.close, period=14)
        self.sma28 = bt.indicators.SimpleMovingAverage(self.data.close, period=28)

        # Record storage
        self.data_records = []
        self.buy_events = []   # Store buy events (datetime and price)
        self.sell_events = []  # Store sell events (datetime and price)
        self.close_events = [] # Store close events (datetime and price)

    def next(self):
        # Record current price and indicator values
        current_record = {
            'datetime': self.data.datetime.datetime(0),
            'price': self.data.close[0],
            'rsi': self.rsi[0],
            'macd': self.macd[0],
            'macd_signal': self.macd_signal[0],
            'adx': self.adx[0],
            'bb_upper': self.bbands.lines.top[0],
            'bb_lower': self.bbands.lines.bot[0],
            'sma14': self.sma14[0],
            'sma28': self.sma28[0]
        }
        self.data_records.append(current_record)

        # Calculate Bollinger Band width as a percentage
        bb_width_pct = (current_record['bb_upper'] - current_record['bb_lower']) / current_record['price'] * 100

        # Determine dynamic trade size based on ADX
        dynamic_size = int(self.params.base_size * (self.adx[0] / 25))

        # Long entry condition
        if self.macd_signal[0] > 0.1 and self.rsi[0] > 70 and self.adx[0] > 27.5 and 2 < bb_width_pct < 5:
            if not self.position:  # No position
                self.buy(size=dynamic_size)
                self.buy_events.append((self.data.datetime.datetime(0), self.data.close[0]))
                print(f"LONG: Price={self.data.close[0]:.2f}, Size={dynamic_size}")
            elif self.position.size < 0:  # If short, close and go long
                self.close()
                self.buy(size=dynamic_size)
                self.buy_events.append((self.data.datetime.datetime(0), self.data.close[0]))
                print(f"CLOSE SHORT, GO LONG: Price={self.data.close[0]:.2f}, Size={dynamic_size}")

        # Short entry condition
        elif self.macd_signal[0] < -0.1 and self.rsi[0] < 30 and self.adx[0] > 27.5 and 2 < bb_width_pct < 5:
            if not self.position:  # No position
                self.sell(size=dynamic_size)
                self.sell_events.append((self.data.datetime.datetime(0), self.data.close[0]))
                print(f"SHORT: Price={self.data.close[0]:.2f}, Size={dynamic_size}")
            elif self.position.size > 0:  # If long, close and go short
                self.close()
                self.sell(size=dynamic_size)
                self.sell_events.append((self.data.datetime.datetime(0), self.data.close[0]))
                print(f"CLOSE LONG, GO SHORT: Price={self.data.close[0]:.2f}, Size={dynamic_size}")

        # Exit conditions
        if self.position.size > 0 and self.sma14[0] < self.sma28[0]:  # Exit long
            self.close()
            self.close_events.append((self.data.datetime.datetime(0), self.data.close[0]))
            print("EXIT LONG: Price below SMA crossover")

        elif self.position.size < 0 and self.sma14[0] > self.sma28[0]:  # Exit short
            self.close()
            self.close_events.append((self.data.datetime.datetime(0), self.data.close[0]))
            print("EXIT SHORT: Price above SMA crossover")

    def stop(self):
        # Save data to CSV
        df = pd.DataFrame(self.data_records)
        df['datetime'] = pd.to_datetime(df['datetime'])
        df.sort_values('datetime', inplace=True)
        df.set_index('datetime', inplace=True)
        df.reset_index(inplace=True)
        df.to_csv('recorded_data.csv', index=False)
        print("Data has been recorded to 'recorded_data.csv'.")

        # Plot the data
        self.plot_data(df)

    def plot_data(self, df):
        # Ensure datetime is converted to pandas datetime
        df['datetime'] = pd.to_datetime(df['datetime'])
        df.sort_values('datetime', inplace=True)
        df.interpolate(method='linear', inplace=True)
        df.set_index('datetime', inplace=True)
        df.reset_index(inplace=True)

        # Convert the index to DatetimeIndex for mplfinance
        df.index = pd.to_datetime(df.index)

        # Prepare the data in OHLC format for mplfinance
        ohlc_data = df[['datetime', 'price', 'bb_upper', 'bb_lower']].copy()
        ohlc_data['open'] = df['price']
        ohlc_data['high'] = df['bb_upper']
        ohlc_data['low'] = df['bb_lower']
        ohlc_data['close'] = df['price']

        # Plot using mplfinance
        mpf.plot(ohlc_data, type='candle', style='charles', volume=False, ylabel='Price',
                 title='Candlestick Chart', figratio=(12,8), figsize=(12, 8),
                 addplot=[
                     mpf.make_addplot(df['rsi'], panel=1, color='purple', title='RSI'),
                     mpf.make_addplot(df['macd'], panel=2, color='orange', title='MACD'),
                     mpf.make_addplot(df['macd_signal'], panel=2, color='red', linestyle='--', title='MACD Signal')
                 ])


def backtest(stock):
    cerebro = bt.Cerebro()
    cerebro.addstrategy(Strategy)
    cerebro.broker.setcash(100000.0)

    # Fetch historical data
    price_data = yf.Ticker(stock).history(period="1mo", interval="30m")
    price_data.index = pd.to_datetime(price_data.index)
    price_data = price_data[price_data.index.to_series().dt.dayofweek < 5]
    # Filter for trading hours only
    price_data.between_time(datetime.time(9,0,0), datetime.time(16,30,0))

    # Feed filtered data to Backtrader
    data = bt.feeds.PandasData(dataname=price_data)
    cerebro.adddata(data)

    print(f'Starting Portfolio Value: {cerebro.broker.getvalue()}')
    cerebro.run()
    print(f'Final Portfolio Value: {cerebro.broker.getvalue()}')

# Run the backtest
backtest("XLK")
