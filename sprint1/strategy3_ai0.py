import backtrader as bt
import datetime 
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import mplcursors
import matplotlib.dates as mdates
from pandas.tseries.holiday import USFederalHolidayCalendar

class EnhancedStrategy(bt.Strategy):
    params = (
        ('base_size', 180),          # Base quantity for trades
        ('max_risk_pct', 0.02),      # Maximum risk per trade
        ('max_drawdown', 0.15),      # Maximum drawdown allowed (15%)
        ('volume_factor', 1.5),      # Minimum volume requirement vs average
        ('max_position_size', 0.8),  # Maximum position size as % of portfolio
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
        self.atr = bt.indicators.ATR(self.data, period=14)
        self.volume_ma = bt.indicators.SMA(self.data.volume, period=20)
        
        # Position Management
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
        
        signal = ""
        # Check if max drawdown is exceeded
        if current_drawdown > self.params.max_drawdown and self.position:
            position_size = abs(self.position.size)
            self.close()
            self.close_events.append((self.data.datetime.datetime(0), self.data.close[0]))
            signal = f"CLOSE (MAX DRAWDOWN) {position_size}"
        else:
            # Calculate Bollinger Band width
            bb_width_pct = (current_record['bb_upper'] - current_record['bb_lower']) / current_record['price'] * 100
            
            # Volume check
            if self.check_volume():
                # Calculate position size
                dynamic_size = self.calculate_position_size()
                
                # Long entry condition
                if (self.macd_signal[0] > 0.1 and self.rsi[0] > 70 and 
                    self.adx[0] > 27.5 and 2 < bb_width_pct < 5 and not self.position):
                    self.buy(size=dynamic_size)
                    self.buy_events.append((self.data.datetime.datetime(0), self.data.close[0]))
                    signal = f"BUY {dynamic_size}"
                
                # Short entry condition
                elif (self.macd_signal[0] < -0.1 and self.rsi[0] < 30 and 
                      self.adx[0] > 27.5 and 2 < bb_width_pct < 5 and not self.position):
                    self.sell(size=dynamic_size)
                    self.sell_events.append((self.data.datetime.datetime(0), self.data.close[0]))
                    signal = f"SELL {dynamic_size}"
                
                # Close long position
                elif self.position.size > 0 and self.sma14[0] < self.sma28[0]:
                    position_size = self.position.size
                    self.close()
                    self.close_events.append((self.data.datetime.datetime(0), self.data.close[0]))
                    signal = f"CLOSE LONG {position_size}"
                
                # Close short position
                elif self.position.size < 0 and self.sma14[0] > self.sma28[0]:
                    position_size = abs(self.position.size)
                    self.close()
                    self.close_events.append((self.data.datetime.datetime(0), self.data.close[0]))
                    signal = f"CLOSE SHORT {position_size}"

        # Print current state with signal if present
        print(f"{current_record['datetime'].strftime('%Y-%m-%d %H:%M')} | ${current_record['price']:.2f} | RSI: {current_record['rsi']:.1f} | ADX: {current_record['adx']:.1f} | MACD: {current_record['macd_signal']:.3f} {' | ' + signal if signal else ''}")

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
        # Convert datetime to pandas datetime if not already
        df['datetime'] = pd.to_datetime(df['datetime'])
        
        # Remove bank holidays and weekends
        us_holidays = USFederalHolidayCalendar().holidays(
            start=df['datetime'].min(),
            end=df['datetime'].max()
        )
        
        # Adjust market hours filter for UTC time
        market_open = pd.Timestamp('14:00').time()
        market_close = pd.Timestamp('21:00').time()
        
        # Create mask for market hours
        mask = (
            (df['datetime'].dt.time >= market_open) & 
            (df['datetime'].dt.time <= market_close) & 
            (df['datetime'].dt.dayofweek < 5) & 
            (~df['datetime'].dt.normalize().isin(us_holidays))
        )
        
        filtered_df = df[mask].reset_index(drop=True)  # Reset index to remove gaps
        
        # Create figure with subplots
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(15, 12), sharex=True)
        
        def plot_data_on_axes(data_df):
            ax1.clear()
            ax2.clear()
            ax3.clear()
            
            # Plot price and indicators
            ax1.plot(data_df['datetime'], data_df['price'], 'b-', label='Price')
            ax1.plot(data_df['datetime'], data_df['bb_upper'], 'gray', linestyle='--', alpha=0.5, label='BB Upper')
            ax1.plot(data_df['datetime'], data_df['bb_lower'], 'gray', linestyle='--', alpha=0.5, label='BB Lower')
            
            # Plot trade markers - ensure they're within the current view
            buy_markers = [(time, price) for time, price in self.buy_events 
                          if time >= data_df['datetime'].min() and time <= data_df['datetime'].max()]
            sell_markers = [(time, price) for time, price in self.sell_events 
                           if time >= data_df['datetime'].min() and time <= data_df['datetime'].max()]
            close_markers = [(time, price) for time, price in self.close_events 
                            if time >= data_df['datetime'].min() and time <= data_df['datetime'].max()]
            
            if buy_markers:
                times, prices = zip(*buy_markers)
                ax1.scatter(times, prices, color='green', marker='^', s=100, label='Buy')
            if sell_markers:
                times, prices = zip(*sell_markers)
                ax1.scatter(times, prices, color='red', marker='v', s=100, label='Sell')
            if close_markers:
                times, prices = zip(*close_markers)
                ax1.scatter(times, prices, color='black', marker='x', s=100, label='Close')
            
            # Plot 2: RSI and ADX
            ax2.plot(data_df['datetime'], data_df['rsi'], 'purple', label='RSI')
            ax2.plot(data_df['datetime'], data_df['adx'], 'brown', label='ADX')
            ax2.axhline(y=70, color='red', linestyle='--', alpha=0.5)
            ax2.axhline(y=30, color='green', linestyle='--', alpha=0.5)
            
            # Plot 3: MACD
            ax3.plot(data_df['datetime'], data_df['macd'], 'blue', label='MACD')
            ax3.plot(data_df['datetime'], data_df['macd_signal'], 'orange', label='Signal')
            
            # Set titles and legends
            ax1.set_title('Price Action and Signals')
            ax2.set_title('RSI and ADX')
            ax3.set_title('MACD')
            
            # Add legends and grids
            ax1.legend()
            ax2.legend()
            ax3.legend()
            ax1.grid(True)
            ax2.grid(True)
            ax3.grid(True)
            
            # Update x-axis formatting to maintain detail level
            for ax in [ax1, ax2, ax3]:
                ax.set_xlim(data_df['datetime'].min(), data_df['datetime'].max())
                ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M'))
                ax.xaxis.set_major_locator(mdates.AutoDateLocator())
                ax.xaxis.set_tick_params(rotation=45)
            
            # Adjust layout to prevent label cutoff
            plt.gcf().autofmt_xdate()
            plt.tight_layout()
        
        plot_data_on_axes(filtered_df)
        
        # Add hover functionality
        vlines = [ax1.axvline(x=filtered_df['datetime'].iloc[0], color='gray', linestyle='--', alpha=0.5),
                 ax2.axvline(x=filtered_df['datetime'].iloc[0], color='gray', linestyle='--', alpha=0.5),
                 ax3.axvline(x=filtered_df['datetime'].iloc[0], color='gray', linestyle='--', alpha=0.5)]
        
        text_annotation = ax1.text(0.02, 0.98, '', transform=ax1.transAxes, 
                                 verticalalignment='top', 
                                 bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'),
                                 zorder=1000)  # Ensure it's always on top
        
        def format_indicator_text(x_pos):
            try:
                # Convert matplotlib date to datetime
                x_datetime = mdates.num2date(x_pos).replace(tzinfo=None)
                
                # Find the nearest data point
                nearest_idx = (filtered_df['datetime'] - x_datetime).abs().idxmin()
                
                if nearest_idx is not None and nearest_idx < len(filtered_df):
                    price = filtered_df['price'].iloc[nearest_idx]
                    rsi = filtered_df['rsi'].iloc[nearest_idx]
                    adx = filtered_df['adx'].iloc[nearest_idx]
                    macd = filtered_df['macd'].iloc[nearest_idx]
                    macd_signal = filtered_df['macd_signal'].iloc[nearest_idx]
                    
                    if macd_signal > 0.1 and rsi > 70 and adx > 27.5:
                        cond_color = 'green'
                        cond_text = 'BUY'
                    elif macd_signal < -0.1 and rsi < 30 and adx > 27.5:
                        cond_color = 'red'
                        cond_text = 'SELL'
                    else:
                        cond_color = 'black'
                        cond_text = 'NEUTRAL'
                    
                    text = (
                        f'Date: {x_datetime.strftime("%Y-%m-%d %H:%M")}\n'
                        f'Price: ${price:.2f}\n'
                        f'RSI: {rsi:.2f}\n'
                        f'ADX: {adx:.2f}\n'
                        f'MACD: {macd:.2f}\n'
                        f'MACD Signal: {macd_signal:.2f}\n'
                        f'Condition: {cond_text}'
                    )
                    return text, cond_color
            except Exception as e:
                print(f"Error in format_indicator_text: {e}")
            return "", "black"
        
        def hover(event):
            if event.inaxes:
                try:
                    x_pos = event.xdata
                    for vline in vlines:
                        vline.set_data([x_pos, x_pos], [0, 1])
                    text, color = format_indicator_text(x_pos)
                    text_annotation.set_text(text)
                    text_annotation.set_color(color)
                    fig.canvas.draw_idle()
                except Exception as e:
                    print(f"Error in hover: {e}")
        
        fig.canvas.mpl_connect('motion_notify_event', hover)
        
        # Add button with fixed position
        plt.subplots_adjust(top=0.9)  # Make room for button
        button_ax = plt.axes([0.85, 0.95, 0.1, 0.04])
        filter_button = plt.Button(button_ax, 'Market Hours')
        
        def toggle_filter(event):
            nonlocal filtered_df, vlines, text_annotation
            
            # Store current view limits
            current_xlim = ax1.get_xlim()
            current_ylims = [ax.get_ylim() for ax in [ax1, ax2, ax3]]
            
            # Switch data
            if filter_button.label.get_text() == 'Market Hours':
                filtered_df = df.copy()
                filter_button.label.set_text('All Hours')
            else:
                filtered_df = df[mask].reset_index(drop=True)
                filter_button.label.set_text('Market Hours')
            
            plot_data_on_axes(filtered_df)
            
            # Remove old vertical lines and text annotation
            for vline in vlines:
                vline.remove()
            text_annotation.remove()
            
            # Recreate text annotation
            text_annotation = ax1.text(0.02, 0.98, '', transform=ax1.transAxes, 
                                     verticalalignment='top', 
                                     bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'),
                                     zorder=1000)
            
            # Create new vertical lines
            vlines = [ax1.axvline(x=filtered_df['datetime'].iloc[0], color='gray', 
                                 linestyle='--', alpha=0.5),
                     ax2.axvline(x=filtered_df['datetime'].iloc[0], color='gray', 
                                 linestyle='--', alpha=0.5),
                     ax3.axvline(x=filtered_df['datetime'].iloc[0], color='gray', 
                                 linestyle='--', alpha=0.5)]
            
            # Try to maintain similar view
            try:
                for ax, ylim in zip([ax1, ax2, ax3], current_ylims):
                    ax.set_ylim(ylim)
            except Exception:
                pass
            
            fig.canvas.draw_idle()
        
        # Initialize vertical lines
        vlines = [ax1.axvline(x=filtered_df['datetime'].iloc[0], color='gray', linestyle='--', alpha=0.5),
                 ax2.axvline(x=filtered_df['datetime'].iloc[0], color='gray', linestyle='--', alpha=0.5),
                 ax3.axvline(x=filtered_df['datetime'].iloc[0], color='gray', linestyle='--', alpha=0.5)]
        
        filter_button.on_clicked(toggle_filter)
        
        plt.show()

def backtest(stock):
    cerebro = bt.Cerebro()
    cerebro.addstrategy(EnhancedStrategy)
    cerebro.broker.setcash(100000.0)
    
    # Add realistic commission
    cerebro.broker.setcommission(commission=0.001)  # 0.1% commission per trade
    
    # Fetch historical data with explicit start and end dates
    end_date = datetime.datetime.now()
    start_date = end_date - datetime.timedelta(days=30)  # Exactly 1 month
    
    price_data = yf.Ticker(stock).history(start=start_date, end=end_date, interval="30m")
    price_data.index = pd.to_datetime(price_data.index)
    
    # Debug: Print the first and last few rows of the data
    print("Data fetched from Yahoo Finance:")
    print(price_data.head())
    print(price_data.tail())
    
    # Filter for US market hours (9:30 AM - 4:00 PM ET)
    price_data = price_data.between_time('09:30', '16:00')
    
    # Remove weekends
    price_data = price_data[price_data.index.to_series().dt.dayofweek < 5]
    
    # Debug: Print the first and last few rows of the filtered data
    print("Filtered data:")
    print(price_data.head())
    print(price_data.tail())
    
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
    