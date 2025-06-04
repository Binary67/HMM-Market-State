from YfinanceDownloader import DownloadTradingData
from MarketRegimeHmm import MarketRegimeHmm


def Main() -> 'pd.DataFrame':
    Data = DownloadTradingData("AAPL", "2021-01-01", "2021-03-01", "1d")
    CloseColumn = [Column for Column in Data.columns if Column.startswith("Close")][0]
    Observations = Data[CloseColumn].pct_change().dropna().values.reshape(-1, 1)
    HmmModel = MarketRegimeHmm()
    HmmModel.Fit(Observations)
    Regimes = HmmModel.PredictRegimes(Observations)
    Result = Data.iloc[1:].copy()
    Result["Regime"] = Regimes
    return Result


if __name__ == "__main__":
    ResultData = Main()
    print(ResultData.head())
