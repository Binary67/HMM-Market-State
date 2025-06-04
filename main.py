from YfinanceDownloader import DownloadTradingData, EnsurePandasTaCompatibility
from MarketRegimeHmm import MarketRegimeHmm
from RegimeAccuracyChecker import RegimeAccuracyChecker


def Main() -> 'pd.DataFrame':
    EnsurePandasTaCompatibility()
    Data = DownloadTradingData("AAPL", "2020-01-01", "2021-03-01", "1d")
    CloseColumn = [Column for Column in Data.columns if Column.startswith("Close")][0]
    Data["Return"] = Data[CloseColumn].pct_change()
    FeatureColumns = [
        "Return",
        "MA20",
        "MA50",
        "BBL",
        "BBM",
        "BBU",
        "MACD",
        "EMA20",
        "RSI",
        "ATR14",
        "STOCH",
        "OBV",
    ]
    Observations = Data[FeatureColumns].dropna().values
    HmmModel = MarketRegimeHmm()
    HmmModel.Fit(Observations)
    Regimes = HmmModel.PredictRegimes(Observations)
    Result = Data.loc[Data[FeatureColumns].dropna().index].copy()
    Result["Regime"] = Regimes
    AccuracyChecker = RegimeAccuracyChecker()
    BestLookahead, BestThreshold = AccuracyChecker.GridSearchParameters(
        Result,
        "Regime",
        "Return",
        LookaheadOptions=[5, 10, 15],
        ThresholdOptions=[0.01, 0.02, 0.03],
    )
    print(
        f"Best params - LookaheadDays: {BestLookahead}, SidewayThreshold: {BestThreshold}"
    )
    Accuracy = AccuracyChecker.CalculateAccuracy(Result, "Regime", "Return")
    print(f"Prediction accuracy: {Accuracy:.2%}")
    return Result


if __name__ == "__main__":
    ResultData = Main()
