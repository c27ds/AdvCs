import backtrader as bt
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt

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
        self.adx = bt.indicators.AverageDirectionalMovementIndex(self.data)
        self.sma14 = bt.indicators.SimpleMovingAverage(self.data.close, period=14)
        self.sma28 = bt.indicators.SimpleMovingAverage(self.data.close, period=28)

        # Record storage
        self.data_records = []

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

        # Print values to console
        print(
            f"{current_record['datetime']} | Price: {current_record['price']:.2f} | RSI: {current_record['rsi']:.2f} | "
            f"MACD: {current_record['macd']:.2f} | MACD Signal: {current_record['macd_signal']:.2f} | "
            f"ADX: {current_record['adx']:.2f}"
        )

        # Calculate Bollinger Band width as a percentage
        bb_width_pct = (current_record['bb_upper'] - current_record['bb_lower']) / current_record['price'] * 100

        # Determine dynamic trade size based on ADX
        dynamic_size = int(self.params.base_size * (self.adx[0] / 25))

        # Long entry condition
        if self.macd_signal[0] > 0.1 and self.rsi[0] > 70 and self.adx[0] > 27.5 and 2 < bb_width_pct < 5:
            if not self.position:  # No position
                self.buy(size=dynamic_size)
                print(f"LONG: Price={self.data.close[0]:.2f}, Size={dynamic_size}")
            elif self.position.size < 0:  # If short, close and go long
                self.close()
                self.buy(size=dynamic_size)
                print(f"CLOSE SHORT, GO LONG: Price={self.data.close[0]:.2f}, Size={dynamic_size}")

        # Short entry condition
        elif self.macd_signal[0] < -0.1 and self.rsi[0] < 30 and self.adx[0] > 27.5 and 2 < bb_width_pct < 5:
            if not self.position:  # No position
                self.sell(size=dynamic_size)
                print(f"SHORT: Price={self.data.close[0]:.2f}, Size={dynamic_size}")
            elif self.position.size > 0:  # If long, close and go short
                self.close()
                self.sell(size=dynamic_size)
                print(f"CLOSE LONG, GO SHORT: Price={self.data.close[0]:.2f}, Size={dynamic_size}")

        # Exit conditions
        if self.position.size > 0 and self.sma14[0] < self.sma28[0]:  # Exit long
            self.close()
            print("EXIT LONG: Price below SMA crossover")

        elif self.position.size < 0 and self.sma14[0] > self.sma28[0]:  # Exit short
            self.close()
            print("EXIT SHORT: Price above SMA crossover")

    def stop(self):
        # Save data to CSV
        df = pd.DataFrame(self.data_records)
        df.to_csv('recorded_data.csv', index=False)
        print("Data has been recorded to 'recorded_data.csv'.")

        # Plot the data
        self.plot_data(df)

    def plot_data(self, df):
        plt.figure(figsize=(12, 8))
        
        # Price and Bollinger Bands
        plt.subplot(3, 1, 1)
        plt.plot(df['datetime'], df['price'], label='Price', color='blue')
        plt.fill_between(df['datetime'], df['bb_upper'], df['bb_lower'], color='gray', alpha=0.3, label='Bollinger Bands')
        plt.legend()
        plt.title('Price and Bollinger Bands')

        # RSI
        plt.subplot(3, 1, 2)
        plt.plot(df['datetime'], df['rsi'], label='RSI', color='purple')
        plt.axhline(70, linestyle='--', color='red', alpha=0.7, label='Overbought (70)')
        plt.axhline(30, linestyle='--', color='green', alpha=0.7, label='Oversold (30)')
        plt.legend()
        plt.title('RSI')

        # MACD and Signal
        plt.subplot(3, 1, 3)
        plt.plot(df['datetime'], df['macd'], label='MACD', color='orange')
        plt.plot(df['datetime'], df['macd_signal'], label='MACD Signal', color='red', linestyle='--')
        plt.legend()
        plt.title('MACD and Signal')

        plt.tight_layout()
        plt.show()

def backtest(stock):
    cerebro = bt.Cerebro()
    cerebro.addstrategy(Strategy)
    cerebro.broker.setcash(100000.0)
    
    # Fetch historical data
    price_data = yf.Ticker(stock).history(period="1mo", interval="30m")
    price_data.index = pd.to_datetime(price_data.index)
    data = bt.feeds.PandasData(dataname=price_data)
    cerebro.adddata(data)
    
    print(f'Starting Portfolio Value: {cerebro.broker.getvalue()}')
    cerebro.run()
    print(f'Final Portfolio Value: {cerebro.broker.getvalue()}')

# Run the backtest
backtest("TSLA")
