import backtrader as bt
import datetime 
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import mplcursors
import matplotlib.dates as mdates
from pandas.tseries.holiday import USFederalHolidayCalendar
import sklearn
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

class AiStrategy(bt.Strategy):
    params = (
        ('base_size', 180),
        ('max_risk_pct', 0.02),
        ('max_drawdown', 0.15),
        ('volume_factor', 1.5),
        ('max_position_size', 0.8),
        ('training_period', 500),  # Number of bars to use for initial training
        ('n_neighbors', 5),  # Add KNN parameter
    )

    def __init__(self):
        self.rsi = bt.indicators.RSI(self.data.close, period=14)
        self.ema12 = bt.indicators.EMA(self.data.close, period=12)
        self.ema26 = bt.indicators.EMA(self.data.close, period=26)
        self.macd = self.ema12 - self.ema26
        self.macd_signal = bt.indicators.EMA(self.macd, period=9)
        self.bbands = bt.indicators.BollingerBands(self.data.close, period=56, devfactor=2.0)
        self.adx = bt.indicators.AverageDirectionalMovementIndex(self.data, period=14)
        self.sma14 = bt.indicators.SimpleMovingAverage(self.data.close, period=14)
        self.sma28 = bt.indicators.SimpleMovingAverage(self.data.close, period=28)
        self.atr = bt.indicators.ATR(self.data, period=14)
        self.volume_ma = bt.indicators.SMA(self.data.volume, period=20)

        # Initialize ML components
        self.model = KNeighborsClassifier(n_neighbors=self.params.n_neighbors)
        self.scaler = StandardScaler()
        self.is_trained = False
        
        # Store historical data for training
        self.feature_data = []
        self.labels = []
        
        # Keep existing initialization
        self.initial_portfolio_value = self.broker.getvalue()
        self.data_records = []
        self.buy_events = []
        self.sell_events = []
        self.close_events = []
        self.max_portfolio_value = self.broker.getvalue()
        self.current_drawdown = 0

    def get_features(self):
        return np.array([
            self.rsi[0],
            self.macd[0],
            self.macd_signal[0],
            self.adx[0],
            (self.data.close[0] - self.bbands.lines.bot[0]) / self.data.close[0],
            (self.bbands.lines.top[0] - self.data.close[0]) / self.data.close[0],
            (self.data.volume[0] / self.volume_ma[0]) - 1,
            (self.sma14[0] / self.sma28[0]) - 1,
            self.atr[0] / self.data.close[0],
        ])

    def generate_label(self, lookforward=5):
        if len(self.data) - self.line.idx <= lookforward:
            return 0
        
        future_return = (self.data.close[lookforward] - self.data.close[0]) / self.data.close[0]
        if future_return > 0.01:  # 1% threshold for buy
            return 1
        elif future_return < -0.01:  # -1% threshold for sell
            return -1
        return 0

    def next(self):
        # Calculate next movement (1 if price went up, 0 if down)
        if len(self.data) > 1:
            next_movement = 1 if self.data.close[0] > self.data.close[-1] else 0
        else:
            next_movement = 0

        # Make prediction based on current indicators
        prediction = 0
        if self.rsi[0] > 70 and self.macd_signal[0] > 0.1 and self.adx[0] > 27.5:
            prediction = 1  # Predict up
        elif self.rsi[0] < 30 and self.macd_signal[0] < -0.1 and self.adx[0] > 27.5:
            prediction = -1  # Predict down

        # Trading logic
        if not self.position:  # If we're not in a position
            if prediction == 1:
                size = self.calculate_position_size()
                self.buy(size=size)
                self.buy_events.append((self.data.datetime.datetime(0), self.data.close[0]))
            elif prediction == -1:
                size = self.calculate_position_size()
                self.sell(size=size)
                self.sell_events.append((self.data.datetime.datetime(0), self.data.close[0]))
        else:  # If we're in a position
            # Exit logic
            current_drawdown = self.update_drawdown()
            if current_drawdown > self.params.max_drawdown:
                self.close()
                self.close_events.append((self.data.datetime.datetime(0), self.data.close[0]))
            elif (self.position.size > 0 and self.sma14[0] < self.sma28[0]) or \
                 (self.position.size < 0 and self.sma14[0] > self.sma28[0]):
                self.close()
                self.close_events.append((self.data.datetime.datetime(0), self.data.close[0]))

        # Record keeping
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
            'atr': self.atr[0],
            'next_movement': next_movement,
            'prediction': prediction  # Add prediction to records
        }
        self.data_records.append(current_record)

        # Print current state
        print(f"{current_record['datetime'].strftime('%Y-%m-%d %H:%M')} | "
              f"${current_record['price']:.2f} | RSI: {current_record['rsi']:.1f} | "
              f"ADX: {current_record['adx']:.1f} | MACD: {current_record['macd_signal']:.3f}")

    def update_drawdown(self):
        current_value = self.broker.getvalue()
        self.max_portfolio_value = max(self.max_portfolio_value, current_value)
        self.current_drawdown = (self.max_portfolio_value - current_value) / self.max_portfolio_value
        return self.current_drawdown

    def check_volume(self):
        return self.data.volume[0] > self.volume_ma[0] * self.params.volume_factor

    def calculate_position_size(self):
        risk_amount = self.broker.getvalue() * self.params.max_risk_pct
        atr_size = risk_amount / (self.atr[0] * 2) if self.atr[0] > 0 else self.params.base_size
        adx_size = int(self.params.base_size * (self.adx[0] / 25))
        dynamic_size = min(atr_size, adx_size)
        max_size = (self.broker.getvalue() * self.params.max_position_size) / self.data.close[0]
        return int(min(dynamic_size, max_size))

    def stop(self):
        final_value = self.broker.getvalue()
        returns = (final_value - self.initial_portfolio_value) / self.initial_portfolio_value
        print(f"\nStrategy Results:")
        print(f"Initial Portfolio Value: ${self.initial_portfolio_value:,.2f}")
        print(f"Final Portfolio Value: ${final_value:,.2f}")
        print(f"Total Return: {returns:.2%}")
        print(f"Maximum Drawdown: {self.current_drawdown:.2%}")

        df = pd.DataFrame(self.data_records)
        
        if len(df) > 0:
            df.to_csv('enhanced_strategy_data.csv', index=False)
            
            # Calculate prediction accuracy
            correct_predictions = sum((df['prediction'] == 1) & (df['next_movement'] == 1)) + \
                                sum((df['prediction'] == -1) & (df['next_movement'] == 0))
            total_predictions = sum(df['prediction'] != 0)
            accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0
            print(f"\nPrediction Accuracy: {accuracy:.2%}")
            
            # Create plot
            fig = plt.figure(figsize=(15, 10))
            
            # Plot price and indicators
            ax1 = plt.subplot(3, 1, 1)
            price_line = ax1.plot(df['datetime'], df['price'], label='Price', zorder=1)
            ax1.plot(df['datetime'], df['bb_upper'], 'gray', linestyle='--', alpha=0.5, label='BB Upper')
            ax1.plot(df['datetime'], df['bb_lower'], 'gray', linestyle='--', alpha=0.5, label='BB Lower')
            
            # Add buy/sell markers
            if self.buy_events:
                buy_times, buy_prices = zip(*self.buy_events)
                ax1.scatter(buy_times, buy_prices, color='g', marker='^', s=100, label='Buy', zorder=2)
            if self.sell_events:
                sell_times, sell_prices = zip(*self.sell_events)
                ax1.scatter(sell_times, sell_prices, color='r', marker='v', s=100, label='Sell', zorder=2)
            if self.close_events:
                close_times, close_prices = zip(*self.close_events)
                ax1.scatter(close_times, close_prices, color='k', marker='x', s=100, label='Close', zorder=2)
            
            # Add prediction arrows with much smaller sizes
            for idx, row in df.iterrows():
                time_delta = (df['datetime'].max() - df['datetime'].min()).total_seconds() / 86400  # Convert to days
                arrow_width = time_delta * 0.01  # Adjust width based on time range
                
                if row['prediction'] == 1:  # Upward prediction
                    ax1.arrow(row['datetime'], row['price']*0.998, 0, row['price']*0.001, 
                             head_width=arrow_width, head_length=row['price']*0.0005, 
                             fc='g', ec='g', alpha=0.5, width=arrow_width*0.1)
                elif row['prediction'] == -1:  # Downward prediction
                    ax1.arrow(row['datetime'], row['price']*1.002, 0, -row['price']*0.001, 
                             head_width=arrow_width, head_length=row['price']*0.0005, 
                             fc='r', ec='r', alpha=0.5, width=arrow_width*0.1)
            
            ax1.set_title('Price Action with Predictions')
            ax1.legend()
            ax1.grid(True)
            
            # Plot RSI
            ax2 = plt.subplot(3, 1, 2)
            ax2.plot(df['datetime'], df['rsi'], 'purple', label='RSI')
            ax2.plot(df['datetime'], df['adx'], 'brown', label='ADX')
            ax2.axhline(y=70, color='r', linestyle='--', alpha=0.5)
            ax2.axhline(y=30, color='g', linestyle='--', alpha=0.5)
            ax2.set_title('RSI and ADX')
            ax2.legend()
            ax2.grid(True)
            
            # Plot MACD
            ax3 = plt.subplot(3, 1, 3)
            ax3.plot(df['datetime'], df['macd'], 'blue', label='MACD')
            ax3.plot(df['datetime'], df['macd_signal'], 'orange', label='Signal')
            ax3.set_title('MACD')
            ax3.legend()
            ax3.grid(True)

            # Format x-axis dates
            for ax in [ax1, ax2, ax3]:
                ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M'))
                ax.xaxis.set_major_locator(mdates.AutoDateLocator())
                plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)

            plt.tight_layout()

            # Create text box for cursor info
            cursor_text = ax1.text(1.02, 0.5, '', transform=ax1.transAxes, 
                                 bbox=dict(facecolor='white', alpha=0.8),
                                 verticalalignment='center')

            # Create vertical lines for cursor tracking
            vlines = [ax1.axvline(x=df['datetime'].iloc[0], color='gray', linestyle='--', alpha=0.5, visible=False),
                     ax2.axvline(x=df['datetime'].iloc[0], color='gray', linestyle='--', alpha=0.5, visible=False),
                     ax3.axvline(x=df['datetime'].iloc[0], color='gray', linestyle='--', alpha=0.5, visible=False)]

            def on_mouse_move(event):
                if event.inaxes:
                    try:
                        # Convert x from float to datetime
                        x_datetime = mdates.num2date(event.xdata).replace(tzinfo=None)
                        
                        # Find the closest datetime in our DataFrame
                        closest_idx = (df['datetime'] - x_datetime).abs().argmin()
                        closest_row = df.iloc[closest_idx]
                        
                        # Update vertical lines
                        for vline in vlines:
                            vline.set_xdata([closest_row['datetime'], closest_row['datetime']])
                            vline.set_visible(True)
                        
                        # Determine signal
                        if closest_row['prediction'] == 1:
                            signal = "BUY SIGNAL"
                            signal_color = 'green'
                        elif closest_row['prediction'] == -1:
                            signal = "SELL SIGNAL"
                            signal_color = 'red'
                        else:
                            signal = "NEUTRAL"
                            signal_color = 'black'
                        
                        # Update text box
                        info_text = (f'Time: {closest_row["datetime"].strftime("%Y-%m-%d %H:%M")}\n'
                                   f'Price: ${closest_row["price"]:.2f}\n'
                                   f'RSI: {closest_row["rsi"]:.1f}\n'
                                   f'MACD: {closest_row["macd"]:.3f}\n'
                                   f'Signal: {closest_row["macd_signal"]:.3f}\n'
                                   f'ADX: {closest_row["adx"]:.1f}\n'
                                   f'Status: {signal}')
                        cursor_text.set_text(info_text)
                        cursor_text.set_color(signal_color)
                        fig.canvas.draw_idle()  # Force update of the figure
                    except Exception as e:
                        print(f"Debug - Error in on_mouse_move: {e}")

            # Connect the mouse move event
            fig.canvas.mpl_connect('motion_notify_event', on_mouse_move)

            # Adjust layout to make room for text box
            plt.subplots_adjust(right=0.85)

            plt.show()

class EnhancedStrategy(bt.Strategy):
    params = (
        ('base_size', 180),
        ('max_risk_pct', 0.02),
        ('max_drawdown', 0.15),
        ('volume_factor', 1.5),
        ('max_position_size', 0.8),
    )

    def __init__(self):
        self.rsi = bt.indicators.RSI(self.data.close, period=14)
        self.ema12 = bt.indicators.EMA(self.data.close, period=12)
        self.ema26 = bt.indicators.EMA(self.data.close, period=26)
        self.macd = self.ema12 - self.ema26
        self.macd_signal = bt.indicators.EMA(self.macd, period=9)
        self.bbands = bt.indicators.BollingerBands(self.data.close, period=56, devfactor=2.0)
        self.adx = bt.indicators.AverageDirectionalMovementIndex(self.data, period=14)
        self.sma14 = bt.indicators.SimpleMovingAverage(self.data.close, period=14)
        self.sma28 = bt.indicators.SimpleMovingAverage(self.data.close, period=28)
        self.atr = bt.indicators.ATR(self.data, period=14)
        self.volume_ma = bt.indicators.SMA(self.data.volume, period=20)
        self.initial_portfolio_value = self.broker.getvalue()
        self.data_records = []
        self.buy_events = []
        self.sell_events = []
        self.close_events = []
        self.max_portfolio_value = self.broker.getvalue()
        self.current_drawdown = 0

    def update_drawdown(self):
        current_value = self.broker.getvalue()
        self.max_portfolio_value = max(self.max_portfolio_value, current_value)
        self.current_drawdown = (self.max_portfolio_value - current_value) / self.max_portfolio_value
        return self.current_drawdown

    def check_volume(self):
        return self.data.volume[0] > self.volume_ma[0] * self.params.volume_factor

    def calculate_position_size(self):
        risk_amount = self.broker.getvalue() * self.params.max_risk_pct
        atr_size = risk_amount / (self.atr[0] * 2) if self.atr[0] > 0 else self.params.base_size
        adx_size = int(self.params.base_size * (self.adx[0] / 25))
        dynamic_size = min(atr_size, adx_size)
        max_size = (self.broker.getvalue() * self.params.max_position_size) / self.data.close[0]
        return int(min(dynamic_size, max_size))

    def next(self):
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

        current_drawdown = self.update_drawdown()
        
        signal = ""
        if current_drawdown > self.params.max_drawdown and self.position:
            position_size = abs(self.position.size)
            self.close()
            self.close_events.append((self.data.datetime.datetime(0), self.data.close[0]))
            signal = f"CLOSE (MAX DRAWDOWN) {position_size}"
        else:
            bb_width_pct = (current_record['bb_upper'] - current_record['bb_lower']) / current_record['price'] * 100
            
            if self.check_volume():
                dynamic_size = self.calculate_position_size()
                
                if (self.macd_signal[0] > 0.1 and self.rsi[0] > 70 and 
                    self.adx[0] > 27.5 and 2 < bb_width_pct < 5 and not self.position):
                    self.buy(size=dynamic_size)
                    self.buy_events.append((self.data.datetime.datetime(0), self.data.close[0]))
                    signal = f"BUY {dynamic_size}"
                elif (self.macd_signal[0] < -0.1 and self.rsi[0] < 30 and 
                      self.adx[0] > 27.5 and 2 < bb_width_pct < 5 and not self.position):
                    self.sell(size=dynamic_size)
                    self.sell_events.append((self.data.datetime.datetime(0), self.data.close[0]))
                    signal = f"SELL {dynamic_size}"
                elif self.position.size > 0 and self.sma14[0] < self.sma28[0]:
                    position_size = self.position.size
                    self.close()
                    self.close_events.append((self.data.datetime.datetime(0), self.data.close[0]))
                    signal = f"CLOSE LONG {position_size}"
                elif self.position.size < 0 and self.sma14[0] > self.sma28[0]:
                    position_size = abs(self.position.size)
                    self.close()
                    self.close_events.append((self.data.datetime.datetime(0), self.data.close[0]))
                    signal = f"CLOSE SHORT {position_size}"

        print(f"{current_record['datetime'].strftime('%Y-%m-%d %H:%M')} | ${current_record['price']:.2f} | RSI: {current_record['rsi']:.1f} | ADX: {current_record['adx']:.1f} | MACD: {current_record['macd_signal']:.3f} {' | ' + signal if signal else ''}")

    def stop(self):
        final_value = self.broker.getvalue()
        returns = (final_value - self.initial_portfolio_value) / self.initial_portfolio_value
        print(f"\nstrategy results:")
        print(f"initial portfolio value: ${self.initial_portfolio_value:,.2f}")
        print(f"final portfolio value: ${final_value:,.2f}")
        print(f"total return: {returns:.2%}")
        print(f"maximum drawdown: {self.current_drawdown:.2%}")

        df = pd.DataFrame(self.data_records)
        df['datetime'] = pd.to_datetime(df['datetime'])
        df.to_csv('enhanced_strategy_data.csv', index=False)
        
        self.plot_data(df)

    def plot_data(self, df):
        df['datetime'] = pd.to_datetime(df['datetime'])
        
        us_holidays = USFederalHolidayCalendar().holidays(
            start=df['datetime'].min(),
            end=df['datetime'].max()
        )
        
        market_open = pd.Timestamp('14:00').time()
        market_close = pd.Timestamp('21:00').time()
        
        mask = (
            (df['datetime'].dt.time >= market_open) & 
            (df['datetime'].dt.time <= market_close) & 
            (df['datetime'].dt.dayofweek < 5) & 
            (~df['datetime'].dt.normalize().isin(us_holidays))
        )
        
        filtered_df = df[mask].reset_index(drop=True)
        
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(15, 12), sharex=True)
        
        def plot_data_on_axes(data_df):
            ax1.clear()
            ax2.clear()
            ax3.clear()
            
            ax1.plot(data_df['datetime'], data_df['price'], 'b-', label='Price')
            ax1.plot(data_df['datetime'], data_df['bb_upper'], 'gray', linestyle='--', alpha=0.5, label='BB Upper')
            ax1.plot(data_df['datetime'], data_df['bb_lower'], 'gray', linestyle='--', alpha=0.5, label='BB Lower')
            
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
            
            ax2.plot(data_df['datetime'], data_df['rsi'], 'purple', label='RSI')
            ax2.plot(data_df['datetime'], data_df['adx'], 'brown', label='ADX')
            ax2.axhline(y=70, color='red', linestyle='--', alpha=0.5)
            ax2.axhline(y=30, color='green', linestyle='--', alpha=0.5)
            
            ax3.plot(data_df['datetime'], data_df['macd'], 'blue', label='MACD')
            ax3.plot(data_df['datetime'], data_df['macd_signal'], 'orange', label='Signal')
            
            ax1.set_title('Price Action and Signals')
            ax2.set_title('RSI and ADX')
            ax3.set_title('MACD')
            
            ax1.legend()
            ax2.legend()
            ax3.legend()
            ax1.grid(True)
            ax2.grid(True)
            ax3.grid(True)
            
            for ax in [ax1, ax2, ax3]:
                ax.set_xlim(data_df['datetime'].min(), data_df['datetime'].max())
                ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M'))
                ax.xaxis.set_major_locator(mdates.AutoDateLocator())
                ax.xaxis.set_tick_params(rotation=45)
            
            plt.gcf().autofmt_xdate()
            plt.tight_layout()
        
        plot_data_on_axes(filtered_df)
        
        vlines = [ax1.axvline(x=filtered_df['datetime'].iloc[0], color='gray', linestyle='--', alpha=0.5),
                 ax2.axvline(x=filtered_df['datetime'].iloc[0], color='gray', linestyle='--', alpha=0.5),
                 ax3.axvline(x=filtered_df['datetime'].iloc[0], color='gray', linestyle='--', alpha=0.5)]
        
        text_annotation = ax1.text(0.02, 0.98, '', transform=ax1.transAxes, 
                                 verticalalignment='top', 
                                 bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'),
                                 zorder=1000)
        
        def format_indicator_text(x_pos):
            try:
                x_datetime = mdates.num2date(x_pos).replace(tzinfo=None)
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
        
        plt.subplots_adjust(top=0.9)
        button_ax = plt.axes([0.85, 0.95, 0.1, 0.04])
        filter_button = plt.Button(button_ax, 'Market Hours')
        
        def toggle_filter(event):
            nonlocal filtered_df, vlines, text_annotation
            
            current_xlim = ax1.get_xlim()
            current_ylims = [ax.get_ylim() for ax in [ax1, ax2, ax3]]
            
            if filter_button.label.get_text() == 'Market Hours':
                filtered_df = df.copy()
                filter_button.label.set_text('All Hours')
            else:
                filtered_df = df[mask].reset_index(drop=True)
                filter_button.label.set_text('Market Hours')
            
            plot_data_on_axes(filtered_df)
            
            for vline in vlines:
                vline.remove()
            text_annotation.remove()
            
            text_annotation = ax1.text(0.02, 0.98, '', transform=ax1.transAxes, 
                                     verticalalignment='top', 
                                     bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'),
                                     zorder=1000)
            
            vlines = [ax1.axvline(x=filtered_df['datetime'].iloc[0], color='gray', 
                                 linestyle='--', alpha=0.5),
                     ax2.axvline(x=filtered_df['datetime'].iloc[0], color='gray', 
                                 linestyle='--', alpha=0.5),
                     ax3.axvline(x=filtered_df['datetime'].iloc[0], color='gray', 
                                 linestyle='--', alpha=0.5)]
            
            try:
                for ax, ylim in zip([ax1, ax2, ax3], current_ylims):
                    ax.set_ylim(ylim)
            except Exception:
                pass
            
            fig.canvas.draw_idle()
        
        filter_button.on_clicked(toggle_filter)
        
        plt.show()

def backtest(stock):
    cerebro = bt.Cerebro()
    strategy_choice = input("Enter the strategy number (1 or 2): ").strip()
    
    # Ensure valid strategy choice
    if strategy_choice == "1":
        cerebro.addstrategy(EnhancedStrategy)
    elif strategy_choice == "2":
        cerebro.addstrategy(AiStrategy)
    else:
        print("Invalid strategy choice. Please enter 1 or 2.")
        return
    
    cerebro.broker.setcash(1000000.0)
    cerebro.broker.setcommission(commission=0.001)
    
    end_date = datetime.datetime.now()
    start_date = end_date - datetime.timedelta(days=60)  # Ensure enough data points
    
    price_data = yf.Ticker(stock).history(start=start_date, end=end_date, interval="30m")
    price_data.index = pd.to_datetime(price_data.index)
    
    if price_data.empty:
        print("No data fetched. Please check the stock symbol or date range.")
        return
    
    print("Data fetched from Yahoo Finance:")
    print(price_data.head())
    print(price_data.tail())
    
    price_data = price_data.between_time('09:30', '16:00')
    price_data = price_data[price_data.index.to_series().dt.dayofweek < 5]
    
    if price_data.empty:
        print("Filtered data is empty. Please check the date range or market hours.")
        return
    
    print("Filtered data:")
    print(price_data.head())
    print(price_data.tail())
    
    data = bt.feeds.PandasData(dataname=price_data)
    cerebro.adddata(data)
    
    cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='sharpe')
    cerebro.addanalyzer(bt.analyzers.DrawDown, _name='drawdown')
    cerebro.addanalyzer(bt.analyzers.Returns, _name='returns')
    
    print(f'Starting Portfolio Value: ${cerebro.broker.getvalue():,.2f}')
    results = cerebro.run()
    print(f'Final Portfolio Value: ${cerebro.broker.getvalue():,.2f}')
    
    strat = results[0]
    
    sharpe_ratio = strat.analyzers.sharpe.get_analysis().get('sharperatio', None)
    print(f"\nSharpe Ratio: {sharpe_ratio if sharpe_ratio is not None else 'N/A'}")
    print(f"Max Drawdown: {strat.analyzers.drawdown.get_analysis()['max']['drawdown']:.2%}")
    print(f"Total Return: {strat.analyzers.returns.get_analysis()['rtot']:.2%}")

if __name__ == "__main__":
    backtest_choice = input("Enter the stock symbol: ").upper()
    backtest(backtest_choice)