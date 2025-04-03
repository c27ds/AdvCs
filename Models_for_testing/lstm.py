import backtrader as bt
import pandas as pd
import numpy as np
import yfinance as yf
from matplotlib import pyplot as plt
import datetime 
import matplotlib.dates as mdates
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import to_categorical
from keras.layers import LSTM, Dropout


def create_model(stock):
    end_date = datetime.datetime.now() - datetime.timedelta(days=10)
    print(end_date)
    start_date = end_date - datetime.timedelta(days=49)
    print(start_date)
    price_data = yf.Ticker(stock).history(start=start_date, end=end_date, interval="30m")
    price_data.index = pd.to_datetime(price_data.index)
    price_data = price_data.between_time('09:30', '16:00')
    price_data = price_data[price_data.index.to_series().dt.dayofweek < 5]
    data = price_data[['Open', 'High', 'Low', 'Close', 'Volume']]

    scaler = MinMaxScaler()
    data_scaled = scaler.fit_transform(data)
    sequence_length = 250
    x = []
    y = []


#data is the scaled data
#idx is the index of datapoint 
#how many datapoints ahead we are looking for training the model
    def generate_label(data, idx, lookforward=14, threshold=0.05):
        if idx + lookforward >= len(data):
            return 0
        #close of a given datapoint bcs 3 = close in data array
        current_price = data[idx, 3]
        
        for step in range(1, lookforward + 1):
            future_price = data[idx + step, 3]
            future_return = (future_price - current_price) / current_price
            if future_return > threshold:
                print("Up")
                print(future_return)
                return 1
            elif future_return < -threshold:
                print("Down")
                print(future_return)
                return -1
        print("Flat")
        return 0

    for i in range(sequence_length, len(data_scaled)):
        x.append(data_scaled[i-sequence_length:i])
        label = generate_label(data_scaled, i)
        y.append(label)

    # Create training data for the model (with y as the labels predicted)

    x = np.array(x)
    y = np.array(y)
    print(y)
    y_categorical = to_categorical(y, num_classes=3)
    # Creates 3 categories for probabilities (down, flat, up) in that order
    # Adds layers to the model
    model = Sequential()
    model.add(LSTM(50, activation='relu', return_sequences=True))
    # LSTM model (basically enhanced RNN)
    model.add(Dropout(0.2))
    # Randomly removes 20% of data to prevent overfitting
    model.add(LSTM(50, activation='relu'))
    # Another LSTM layer to pick up more complex patterns and refine previous one
    model.add(Dense(3, activation='softmax'))
    # Activates the model with 3 possible outputs (softmax creates probabailities)

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    # Compiles using the probably most widely used optimizer and loss function
    model.fit(x, y_categorical, epochs=10, batch_size=32)
    # Fits model to the data (values are commonly used ones)
    return model, scaler

class LSTMAlgo(bt.Strategy):
    params = (
        ('base_size', 180),
        ('max_risk_pct', 0.02),
        ('max_drawdown', 0.15),
        ('volume_factor', 1.5),
        ('max_position_size', 0.8)
    )

    def __init__(self, scaler, model):
        self.rsi = bt.indicators.RSI(self.data.close, period=14)
        self.model = model
        self.scaler = scaler
        self.initial_portfolio_value = self.broker.getvalue()
        self.data_records = []
        self.buy_events = []
        self.sell_events = []
        self.close_events = []
        self.max_portfolio_value = self.broker.getvalue()
        self.current_drawdown = 0

    def preprocess_data(self, data):
        data_scaled = self.scaler.transform(data)
        sequence_length = 250
        x = np.array([data_scaled[-sequence_length:]])
        return x

    def next(self):
        if len(self.data) > 1:
            next_movement = 1 if self.data.close[0] > self.data.close[-1] else 0
        else:
            next_movement = 0

        latest_data = np.array([[self.data.open[i], self.data.high[i], self.data.low[i], self.data.close[i], self.data.volume[i]] for i in range(0,30)])

        x_rnn = self.preprocess_data(latest_data)
        rnn_prediction = self.model.predict(x_rnn)
        index_dict = {0: -1, 1: 0, 2: 1}
        highest_probability = np.argmax(rnn_prediction)
        rnn_prediction = index_dict[highest_probability]

        print(rnn_prediction)
        print(f"rnn_prediction is {rnn_prediction}")
        prediction = 0
        if rnn_prediction == 1:
            prediction = 1
            print(f"Buy: Rnn says {rnn_prediction}")
        elif rnn_prediction == -1:
            prediction = -1
            print(f"Sell: Rnn says {rnn_prediction}")

        if not self.position:
            if prediction == 1:
                size = self.calculate_position_size()
                self.buy(size=size)
                self.buy_events.append((self.data.datetime.datetime(0), self.data.close[0]))
            elif prediction == -1:
                size = self.calculate_position_size()
                self.sell(size=size)
                self.sell_events.append((self.data.datetime.datetime(0), self.data.close[0]))
        else:
            current_drawdown = self.update_drawdown()
            if current_drawdown > self.params.max_drawdown:
                self.close()
                self.close_events.append((self.data.datetime.datetime(0), self.data.close[0]))
            elif (self.position.size > 0 and self.sma14[0] < self.sma28[0]) or (self.position.size < 0 and self.sma14[0] > self.sma28[0]):
                self.close()
                self.close_events.append((self.data.datetime.datetime(0), self.data.close[0]))

    def update_drawdown(self):
        current_value = self.broker.getvalue()
        self.maxn_portfolio_value = max(self.max_portfolio_value, current_value)
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
            
            correct_predictions = sum((df['prediction'] == 1) & (df['next_movement'] == 1)) + \
                                sum((df['prediction'] == -1) & (df['next_movement'] == 0))
            total_predictions = sum(df['prediction'] != 0)
            accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0
            print(f"\nPrediction Accuracy: {accuracy:.2%}")
            
            fig = plt.figure(figsize=(15, 10))
            
            ax1 = plt.subplot(3, 1, 1)
            price_line = ax1.plot(df['datetime'], df['price'], label='Price', zorder=1)
            ax1.plot(df['datetime'], df['bb_upper'], 'gray', linestyle='--', alpha=0.5, label='BB Upper')
            ax1.plot(df['datetime'], df['bb_lower'], 'gray', linestyle='--', alpha=0.5, label='BB Lower')
            
            if self.buy_events:
                buy_times, buy_prices = zip(*self.buy_events)
                ax1.scatter(buy_times, buy_prices, color='g', marker='^', s=100, label='Buy', zorder=2)
            if self.sell_events:
                sell_times, sell_prices = zip(*self.sell_events)
                ax1.scatter(sell_times, sell_prices, color='r', marker='v', s=100, label='Sell', zorder=2)
            if self.close_events:
                close_times, close_prices = zip(*self.close_events)
                ax1.scatter(close_times, close_prices, color='k', marker='x', s=100, label='Close', zorder=2)
            for idx, row in df.iterrows():
                time_delta = (df['datetime'].max() - df['datetime'].min()).total_seconds() / 86400
                arrow_width = time_delta * 0.01
                
                if row['prediction'] == 1:
                    ax1.arrow(row['datetime'], row['price']*0.998, 0, row['price']*0.001, 
                             head_width=arrow_width, head_length=row['price']*0.0005, 
                             fc='g', ec='g', alpha=0.5, width=arrow_width*0.1)
                elif row['prediction'] == -1:
                    ax1.arrow(row['datetime'], row['price']*1.002, 0, -row['price']*0.001, 
                             head_width=arrow_width, head_length=row['price']*0.0005, 
                             fc='r', ec='r', alpha=0.5, width=arrow_width*0.1)
            
            ax1.set_title('Price Action with Predictions')
            ax1.legend()
            ax1.grid(True)
            
            ax2 = plt.subplot(3, 1, 2)
            ax2.plot(df['datetime'], df['rsi'], 'purple', label='RSI')
            ax2.plot(df['datetime'], df['adx'], 'brown', label='ADX')
            ax2.axhline(y=70, color='r', linestyle='--', alpha=0.5)
            ax2.axhline(y=30, color='g', linestyle='--', alpha=0.5)
            ax2.set_title('RSI and ADX')
            ax2.legend()
            ax2.grid(True)
            
            ax3 = plt.subplot(3, 1, 3)
            ax3.plot(df['datetime'], df['macd'], 'blue', label='MACD')
            ax3.plot(df['datetime'], df['macd_signal'], 'orange', label='Signal')
            ax3.set_title('MACD')
            ax3.legend()
            ax3.grid(True)

            for ax in [ax1, ax2, ax3]:
                ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M'))
                ax.xaxis.set_major_locator(mdates.AutoDateLocator())
                plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)

            plt.tight_layout()

            cursor_text = ax1.text(1.02, 0.5, '', transform=ax1.transAxes, 
                                 bbox=dict(facecolor='white', alpha=0.8),
                                 verticalalignment='center')

            vlines = [ax1.axvline(x=df['datetime'].iloc[0], color='gray', linestyle='--', alpha=0.5, visible=False),
                     ax2.axvline(x=df['datetime'].iloc[0], color='gray', linestyle='--', alpha=0.5, visible=False),
                     ax3.axvline(x=df['datetime'].iloc[0], color='gray', linestyle='--', alpha=0.5, visible=False)]

            def on_mouse_move(event):
                if event.inaxes:
                    try:
                        x_datetime = mdates.num2date(event.xdata).replace(tzinfo=None)
                        
                        closest_idx = (df['datetime'] - x_datetime).abs().argmin()
                        closest_row = df.iloc[closest_idx]
                        
                        for vline in vlines:
                            vline.set_xdata([closest_row['datetime'], closest_row['datetime']])
                            vline.set_visible(True)
                        
                        if closest_row['prediction'] == 1:
                            signal = "BUY SIGNAL"
                            signal_color = 'green'
                        elif closest_row['prediction'] == -1:
                            signal = "SELL SIGNAL"
                            signal_color = 'red'
                        else:
                            signal = "NEUTRAL"
                            signal_color = 'black'
                        
                        info_text = (f'Time: {closest_row["datetime"].strftime("%Y-%m-%d %H:%M")}\n'
                                   f'Price: ${closest_row["price"]:.2f}\n'
                                   f'RSI: {closest_row["rsi"]:.1f}\n'
                                   f'MACD: {closest_row["macd"]:.3f}\n'
                                   f'Signal: {closest_row["macd_signal"]:.3f}\n'
                                   f'ADX: {closest_row["adx"]:.1f}\n'
                                   f'Status: {signal}')
                        cursor_text.set_text(info_text)
                        cursor_text.set_color(signal_color)
                        fig.canvas.draw_idle()
                    except Exception as e:
                        print(f"Debug - Error in on_mouse_move: {e}")

            fig.canvas.mpl_connect('motion_notify_event', on_mouse_move)

            plt.subplots_adjust(right=0.85)

            plt.show()


def backtest_part_two(stock, scaler, model):
    cerebro = bt.Cerebro()
    cerebro.addstrategy(LSTMAlgo, scaler, model)
    
    cerebro.broker.setcash(100000.0)
    cerebro.broker.setcommission(commission=0.001)
    
    end_date = datetime.datetime.now()
    start_date = end_date - datetime.timedelta(days=30)
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
    return_total = strat.analyzers.returns.get_analysis()['rtot']
    return {"Total Return":return_total, "Max Drawdown":strat.analyzers.drawdown.get_analysis()['max']['drawdown'], "Sharpe Ratio":sharpe_ratio if sharpe_ratio is not None else 'N/A'}
def backtest(stock):
    model, scaler = create_model(stock)
    results = backtest_part_two(stock, scaler, model)
    return results

if __name__ == "__main__":
    model, scaler = create_model("AAPL")
    print(backtest_part_two("AAPL", scaler, model))