#for backtesting multiple stocks
import knn
import rnn
import lstm
stocks = ["XLK","QQQ","TSLA","MSFT","AAPL"]
def backtest_all(stocks):
    lstm_results = {"Total Return":return_total, "Max Drawdown":strat.analyzers.drawdown.get_analysis()['max']['drawdown'], "Sharpe Ratio":sharpe_ratio if sharpe_ratio is not None else 'N/A'}
    rnn_results
    knn_results
    for i in stocks:
        lstm_results = lstm.backtest(i)
        rnn_results = rnn.backtest(i)
        knn_results = knn.backtest(i)

