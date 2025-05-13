#for backtesting multiple stocks
import lstm_regression as lstm
stocks = ["XLK","QQQ","TSLA","MSFT","AAPL"]
def backtest_all(stocks):
    lstm_results = {"RMSE":[], "R2":[]}
#     rnn_results = {"Total Return":[], "Max Drawdown":[], "Sharpe Ratio":[]}
#     knn_results = {"Total Return":[], "Max Drawdown":[], "Sharpe Ratio":[]}
    for i in stocks:
        li = []
        for x in range(5):
            li.append(lstm.run_lstm(i))
        chosen = max(li, key=lambda x: x[1])
        lstm_results["R2"].append(chosen[1])
        lstm_results["RMSE"].append(chosen[0])
#         rnn_res = rnn.backtest(i)
#         knn_res = knn.backtest(i)
        
#         rnn_results["Max Drawdown"].append(rnn_res["Max Drawdown"])
#         rnn_results["Total Return"].append(rnn_res["Total Return"])
#         rnn_results["Sharpe Ratio"].append(rnn_res["Sharpe Ratio"])
#         knn_results["Max Drawdown"].append(knn_res["Max Drawdown"])
#         knn_results["Total Return"].append(knn_res["Total Return"])
#         knn_results["Sharpe Ratio"].append(knn_res["Sharpe Ratio"])
    print(lstm_results)
        

