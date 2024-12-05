import backtrader as bt
import datetime 
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import mplcursors

class EnhancedStrategy(bt.Strategy):
    params = (
        ('base_size', 180),          # Base quantity for trades
        ('stop_loss', 0.02),         # 2% stop loss
        ('take_profit', 0.03),       # 3% take profit
        ('max_risk_pct', 0.02),      # Maximum risk per trade
        ('max_drawdown', 0.15),      # Maximum drawdown allowed (15%)
        ('volume_factor', 1.5),      # Minimum volume requirement vs average
        ('max_position_size', 0.2),  # Maximum position size as % of portfolio
        ('time_exit', 20),           # Bars to hold position before time-based exit
    )

    def __init__(self):
        # Core Indicators
        self.rsi = bt.indicators.RSI(self.data.close, period=14)
        self.ema12 = bt.indicators.EMA(self.data.close, period=12)
        self.ema26 = bt.indicators.EMA(self.data.close, period=26)
        self.macd = self.ema12 - self.ema26
        self.macd_signal = bt.indicators.EMA(self.macd, period=9)
        self.bbands = bt.indicators.BollingerBands(self.data.close, period=56, devfactor=2.0)
        self.adx = bt.indicators.AverageDirectionalMovementIndex(self.data, period=14)
        self.sma14 = bt.indicators.SimpleMovingAverage(self.data.close, period=14)
        self.sma28 = bt.indicators.SimpleMovingAverage(self.data.close, period=28)

        # Additional Risk Management Indicators
        self.trailing_stop = bt.indicators.Highest(self.data.close, period=20)
        self.atr = bt.indicators.ATR(self.data, period=14)
        self.volume_ma = bt.indicators.SMA(self.data.volume, period=20)
        
        # Position Management
        self.entry_price = None
        self.entry_time = None
        self.highest_price = None
        self.lowest_price = None
        self.initial_portfolio_value = self.broker.getvalue()
        
        # Record storage
        self.data_records = []
        self.buy_events = []
        self.sell_events = []
        self.close_events = []
        
        # Drawdown tracking
        self.max_portfolio_value = self.broker.getvalue()
        self.current_drawdown = 0

    def update_drawdown(self):
        """Calculate and update current drawdown"""
        current_value = self.broker.getvalue()
        self.max_portfolio_value = max(self.max_portfolio_value, current_value)
        self.current_drawdown = (self.max_portfolio_value - current_value) / self.max_portfolio_value
        return self.current_drawdown

    def check_volume(self):
        """Check if volume is sufficient for trading"""
        return self.data.volume[0] > self.volume_ma[0] * self.params.volume_factor

    def calculate_position_size(self):
        """Calculate position size based on risk management rules"""
        risk_amount = self.broker.getvalue() * self.params.max_risk_pct
        price_diff = self.data.close[0] * self.params.stop_loss
        
        # ATR-based position sizing
        atr_size = risk_amount / (self.atr[0] * 2) if self.atr[0] > 0 else self.params.base_size
        
        # ADX-based adjustment
        adx_size = int(self.params.base_size * (self.adx[0] / 25))
        
        # Combine different sizing methods
        dynamic_size = min(atr_size, adx_size)
        
        # Ensure position size doesn't exceed maximum portfolio allocation
        max_size = (self.broker.getvalue() * self.params.max_position_size) / self.data.close[0]
        
        return int(min(dynamic_size, max_size))

    def next(self):
        # Record current data
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
            'sma28': self.sma28[0],
            'volume': self.data.volume[0],
            'atr': self.atr[0]
        }
        self.data_records.append(current_record)

        # Update drawdown
        current_drawdown = self.update_drawdown()
        
        # Check if max drawdown is exceeded
        if current_drawdown > self.params.max_drawdown:
            if self.position:
                self.close()
                self.close_events.append((self.data.datetime.datetime(0), self.data.close[0]))
                print(f"MAX DRAWDOWN EXCEEDED ({current_drawdown:.2%}): Closing all positions")
            return

        # Calculate Bollinger Band width
        bb_width_pct = (current_record['bb_upper'] - current_record['bb_lower']) / current_record['price'] * 100

        # Time-based exit check
        if self.position and self.entry_time:
            bars_held = len(self.data) - self.entry_time
            if bars_held >= self.params.time_exit:
                self.close()
                self.close_events.append((self.data.datetime.datetime(0), self.data.close[0]))
                print(f"TIME EXIT: Position held for {bars_held} bars")
                return

        # Volume check
        if not self.check_volume():
            return

        # Calculate position size
        dynamic_size = self.calculate_position_size()

        # Long entry condition
        if (self.macd_signal[0] > 0.1 and self.rsi[0] > 70 and 
            self.adx[0] > 27.5 and 2 < bb_width_pct < 5):
            
            if not self.position:
                self.buy(size=dynamic_size)
                entry_price = self.data.close[0]
                self.entry_time = len(self.data)
                self.highest_price = entry_price
                
                # Set stop loss and take profit orders
                stop_price = entry_price * (1 - self.params.stop_loss)
                target_price = entry_price * (1 + self.params.take_profit)
                
                self.sell(size=dynamic_size, exectype=bt.Order.Stop, price=stop_price)
                self.sell(size=dynamic_size, exectype=bt.Order.Limit, price=target_price)
                
                self.buy_events.append((self.data.datetime.datetime(0), entry_price))
                print(f"LONG: Price={entry_price:.2f}, Size={dynamic_size}")

        # Short entry condition
        elif (self.macd_signal[0] < -0.1 and self.rsi[0] < 30 and 
              self.adx[0] > 27.5 and 2 < bb_width_pct < 5):
            
            if not self.position:
                self.sell(size=dynamic_size)
                entry_price = self.data.close[0]
                self.entry_time = len(self.data)
                self.lowest_price = entry_price
                
                # Set stop loss and take profit orders
                stop_price = entry_price * (1 + self.params.stop_loss)
                target_price = entry_price * (1 - self.params.take_profit)
                
                self.buy(size=dynamic_size, exectype=bt.Order.Stop, price=stop_price)
                self.buy(size=dynamic_size, exectype=bt.Order.Limit, price=target_price)
                
                self.sell_events.append((self.data.datetime.datetime(0), entry_price))
                print(f"SHORT: Price={entry_price:.2f}, Size={dynamic_size}")

        # Update trailing stops
        if self.position:
            if self.position.size > 0:  # Long position
                self.highest_price = max(self.highest_price, self.data.close[0])
                trail_stop = self.highest_price * (1 - self.params.stop_loss)
                if self.data.close[0] < trail_stop:
                    self.close()
                    self.close_events.append((self.data.datetime.datetime(0), self.data.close[0]))
                    print(f"TRAILING STOP: Closing long position at {self.data.close[0]:.2f}")
            
            else:  # Short position
                self.lowest_price = min(self.lowest_price, self.data.close[0])
                trail_stop = self.lowest_price * (1 + self.params.stop_loss)
                if self.data.close[0] > trail_stop:
                    self.close()
                    self.close_events.append((self.data.datetime.datetime(0), self.data.close[0]))
                    print(f"TRAILING STOP: Closing short position at {self.data.close[0]:.2f}")

    def stop(self):
        # Calculate and print strategy metrics
        final_value = self.broker.getvalue()
        returns = (final_value - self.initial_portfolio_value) / self.initial_portfolio_value
        print(f"\nStrategy Results:")
        print(f"Initial Portfolio Value: ${self.initial_portfolio_value:,.2f}")
        print(f"Final Portfolio Value: ${final_value:,.2f}")
        print(f"Total Return: {returns:.2%}")
        print(f"Maximum Drawdown: {self.current_drawdown:.2%}")

        # Save data to CSV
        df = pd.DataFrame(self.data_records)
        df['datetime'] = pd.to_datetime(df['datetime'])
        df.to_csv('enhanced_strategy_data.csv', index=False)
        
        # Plot the data
        self.plot_data(df)

    def plot_data(self, df):
        # Create figure with subplots
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(15, 12), sharex=True)
        
        # Plot 1: Price and Bollinger Bands
        price_line = ax1.plot(df['datetime'], df['price'], label='Price', color='blue', alpha=0.7)[0]
        ax1.plot(df['datetime'], df['bb_upper'], label='BB Upper', color='gray', linestyle='--', alpha=0.5)
        ax1.plot(df['datetime'], df['bb_lower'], label='BB Lower', color='gray', linestyle='--', alpha=0.5)
        
        # Add buy/sell markers
        for buy_time, buy_price in self.buy_events:
            ax1.scatter(buy_time, buy_price, color='green', marker='^', s=100)
        for sell_time, sell_price in self.sell_events:
            ax1.scatter(sell_time, sell_price, color='red', marker='v', s=100)
        for close_time, close_price in self.close_events:
            ax1.scatter(close_time, close_price, color='black', marker='x', s=100)
        
        ax1.set_title('Price Action and Signals')
        ax1.legend()
        ax1.grid(True)
        
        # Plot 2: RSI and ADX
        ax2.plot(df['datetime'], df['rsi'], label='RSI', color='purple')
        ax2.plot(df['datetime'], df['adx'], label='ADX', color='brown')
        ax2.axhline(y=70, color='red', linestyle='--', alpha=0.5)
        ax2.axhline(y=30, color='green', linestyle='--', alpha=0.5)
        ax2.set_title('RSI and ADX')
        ax2.legend()
        ax2.grid(True)
        
        # Plot 3: MACD
        ax3.plot(df['datetime'], df['macd'], label='MACD', color='blue')
        ax3.plot(df['datetime'], df['macd_signal'], label='Signal', color='orange')
        ax3.fill_between(df['datetime'], df['macd'] - df['macd_signal'], color='gray', alpha=0.2)
        ax3.set_title('MACD')
        ax3.legend()
        ax3.grid(True)
        
        # Add vertical lines that sync across all subplots
        vlines = [ax1.axvline(x=df['datetime'].iloc[0], color='gray', linestyle='--', alpha=0.5),
                 ax2.axvline(x=df['datetime'].iloc[0], color='gray', linestyle='--', alpha=0.5),
                 ax3.axvline(x=df['datetime'].iloc[0], color='gray', linestyle='--', alpha=0.5)]
        
        # Add text annotation
        text_annotation = ax1.text(0.02, 0.98, '', transform=ax1.transAxes, 
                                 verticalalignment='top', bbox=dict(facecolor='white', alpha=0.8))
        
        def format_indicator_text(x_pos):
            try:
                # Convert x_pos to datetime for comparison
                x_datetime = pd.to_datetime(x_pos, unit='D')
                # Find the nearest datetime in our dataframe
                nearest_idx = (df['datetime'] - x_datetime).abs().idxmin()
                
                price = df['price'].iloc[nearest_idx]
                rsi = df['rsi'].iloc[nearest_idx]
                adx = df['adx'].iloc[nearest_idx]
                macd = df['macd'].iloc[nearest_idx]
                macd_signal = df['macd_signal'].iloc[nearest_idx]
                
                # Determine trading condition
                if macd_signal > 0.1 and rsi > 70 and adx > 27.5:
                    cond_color = 'green'
                    cond_text = 'BUY'
                elif macd_signal < -0.1 and rsi < 30 and adx > 27.5:
                    cond_color = 'red'
                    cond_text = 'SELL'
                else:
                    cond_color = 'black'
                    cond_text = 'NEUTRAL'
                
                # Color coding for RSI
                rsi_color = 'green' if rsi > 70 else 'red' if rsi < 30 else 'black'
                rsi_text = f'RSI: {rsi:.2f}'
                
                text = (
                    f'Price: ${price:.2f}\n'
                    f'{rsi_text}\n'
                    f'ADX: {adx:.2f}\n'
                    f'MACD Signal: {macd_signal:.2f}\n'
                    f'Condition: {cond_text}'
                )
                return text, rsi_color, cond_color
            except Exception as e:
                print(f"Error in format_indicator_text: {e}")
                return "", "black", "black"
        
        def hover(event):
            try:
                if event.inaxes:
                    x_pos = event.xdata
                    # Update all vertical lines
                    for vline in vlines:
                        vline.set_data([x_pos, x_pos], [0, 1])
                    text, rsi_color, cond_color = format_indicator_text(x_pos)
                    text_annotation.set_text(text)
                    text_annotation.set_color(cond_color)
                    fig.canvas.draw_idle()
            except Exception as e:
                print(f"Error in hover: {e}")
        
        # Connect the hover event to the figure
        fig.canvas.mpl_connect('motion_notify_event', hover)
        
        # Make sure the date format is correct
        fig.autofmt_xdate()
        
        # Adjust layout and show plot
        plt.tight_layout()
        plt.show()

def backtest(stock):
    cerebro = bt.Cerebro()
    cerebro.addstrategy(EnhancedStrategy)
    cerebro.broker.setcash(100000.0)
    
    # Add realistic commission
    cerebro.broker.setcommission(commission=0.001)  # 0.1% commission per trade
    
    # Fetch historical data
    price_data = yf.Ticker(stock).history(period="1mo", interval="30m")
    price_data.index = pd.to_datetime(price_data.index)
    
    # Filter for US market hours (9:30 AM - 4:00 PM ET)
    price_data = price_data.between_time('09:30', '16:00')
    
    # Remove weekends
    price_data = price_data[price_data.index.to_series().dt.dayofweek < 5]
    
    # Feed filtered data to Backtrader
    data = bt.feeds.PandasData(dataname=price_data)
    cerebro.adddata(data)
    
    # Add analyzers
    cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='sharpe')
    cerebro.addanalyzer(bt.analyzers.DrawDown, _name='drawdown')
    cerebro.addanalyzer(bt.analyzers.Returns, _name='returns')
    
    print(f'Starting Portfolio Value: ${cerebro.broker.getvalue():,.2f}')
    results = cerebro.run()
    print(f'Final Portfolio Value: ${cerebro.broker.getvalue():,.2f}')
    
    # Print analysis
    strat = results[0]
    
    # Handle Sharpe Ratio safely
    sharpe_ratio = strat.analyzers.sharpe.get_analysis().get('sharperatio', None)
    print(f"\nSharpe Ratio: {sharpe_ratio if sharpe_ratio is not None else 'N/A'}")
    print(f"Max Drawdown: {strat.analyzers.drawdown.get_analysis()['max']['drawdown']:.2%}")
    print(f"Total Return: {strat.analyzers.returns.get_analysis()['rtot']:.2%}")

if __name__ == "__main__":
    backtest("XLK") 
    