from YfinanceDownloader import DownloadTradingData
from MarketRegineHmm import MarketRegineHmm


def Main() -> 'pd.DataFrame':
    Data = DownloadTradingData("AAPL", "2021-01-01", "2021-03-01", "1d")
    CloseColumn = [Column for Column in Data.columns if Column.startswith("Close")][0]
    Data["Return"] = Data[CloseColumn].pct_change()
    FeatureColumns = ["Return", "MA20", "MA50", "RSI"]
    Observations = Data[FeatureColumns].dropna().values
    HmmModel = MarketRegineHmm()
    HmmModel.Fit(Observations)
    Regimes = HmmModel.PredictRegimes(Observations)
    Result = Data.loc[Data[FeatureColumns].dropna().index].copy()
    Result["Regime"] = Regimes
    return Result


if __name__ == "__main__":
    ResultData = Main()
    print(ResultData.head())
