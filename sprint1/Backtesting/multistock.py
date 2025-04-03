#for backtesting multiple stocks
import Models_for_testing.knn as knn
import sprint1.Models.rnn as rnn
import Models_for_testing.lstm as lstm
stocks = ["XLK","QQQ","TSLA","MSFT","AAPL"]
def backtest_all(stocks):
    lstm_results = {"Total Return":[], "Max Drawdown":[], "Sharpe Ratio":[]}
    rnn_results = {"Total Return":[], "Max Drawdown":[], "Sharpe Ratio":[]}
    knn_results = {"Total Return":[], "Max Drawdown":[], "Sharpe Ratio":[]}
    for i in stocks:
        lstm_res = lstm.backtest(i)
        rnn_res = rnn.backtest(i)
        knn_res = knn.backtest(i)
        lstm_results["Max Drawdown"].append(lstm_res["Max Drawdown"])
        lstm_results["Total Return"].append(lstm_res["Total Return"])
        lstm_results["Sharpe Ratio"].append(lstm_res["Sharpe Ratio"])
        rnn_results["Max Drawdown"].append(rnn_res["Max Drawdown"])
        rnn_results["Total Return"].append(rnn_res["Total Return"])
        rnn_results["Sharpe Ratio"].append(rnn_res["Sharpe Ratio"])
        knn_results["Max Drawdown"].append(knn_res["Max Drawdown"])
        knn_results["Total Return"].append(knn_res["Total Return"])
        knn_results["Sharpe Ratio"].append(knn_res["Sharpe Ratio"])
        

