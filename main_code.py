#for backtesting multiple stocks
# import Models_for_testing.knn as knn
import Models_for_testing.lstm as lstm
#Test across the S&P 500
stocks = ["XLK"]
def backtest_all(stocks):
    # models = {
    #     "LSTM": lstm,
    #     "KNN": knn,
    #     "Logistic Regression": logistic_regression,
    #     "Random Forest": random_forest,
    #     "XGBoost": xgboost,
    #     "DRNN": drnn
    # }
    models = {"LSTM":lstm}
    results = {}
    for i in models:
        results[i] = {"Total Return": [], "Max Drawdown": [], "Sharpe Ratio": []}
    
    for stock in stocks:
        for model_name, model in models.items():
            res = model.backtest(stock)
            results[model_name]["Max Drawdown"].append(res["Max Drawdown"])
            results[model_name]["Total Return"].append(res["Total Return"])
            results[model_name]["Sharpe Ratio"].append(res["Sharpe Ratio"])

    return results

print(backtest_all(stocks))

