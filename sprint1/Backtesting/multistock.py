#for backtesting multiple stocks
import knn
import rnn
import lstm
stocks = ["XLK","QQQ","TSLA","MSFT","AAPL"]
def backtest_all(stocks):
    for i in stocks:
        lstm_results = lstm.backtest(i)
        rnn_results = rnn.backtest(i)
        knn_results = knn.backtest(i)

