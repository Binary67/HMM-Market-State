from DataDownloader import DownloadTradingData
from FeatureEngineering import FeatureEngineering
from HiddenMarkovModel import HiddenMarkovModel
from BacktestingModule import RunBacktest
from ModelEvaluation import EvaluateRegimePrediction
import numpy as np


def Main() -> None:
    Data = DownloadTradingData(
        "AAPL",
        "2020-01-01",
        "2024-12-31",
        "1d",
    )

    FeatureEngineer = FeatureEngineering()

    IndicatorsToApply = [
        "SMA",
        "EMA",
        "RSI",
        "MACD",
        "BollingerBands",
        "OnBalanceVolume",
        "WilliamsPercentR",
        "CommodityChannelIndex",
    ]
    IndicatorParameters = {
        "SMA": {"Window": 20},
        "EMA": {"Window": 12},
        "RSI": {"Window": 14},
        "MACD": {"FastPeriod": 12, "SlowPeriod": 26, "SignalPeriod": 9},
        "BollingerBands": {"Window": 20, "StandardDeviations": 2.0},
        "OnBalanceVolume": {},
        "WilliamsPercentR": {"Window": 14},
        "CommodityChannelIndex": {"Window": 20},
    }

    Data = FeatureEngineer.ApplyTechnicalAnalysis(
        Data, IndicatorsToApply, IndicatorParameters
    )

    Data["LogReturn"] = np.log(Data["Close"]).diff()

    OriginalColumns = ["Open", "High", "Low", "Close", "Volume"]
    TechnicalFeatureColumns = [
        Column
        for Column in Data.columns
        if Column not in OriginalColumns
    ]

    MarketModel = HiddenMarkovModel()
    MarketModel.Fit(Data, TechnicalFeatureColumns)
    Metrics = EvaluateRegimePrediction(Data, TechnicalFeatureColumns)
    print("Validation metrics:", Metrics)
    TransitionMatrix = MarketModel.GetTransitionProbabilities()
    print("Transition probabilities:\n", TransitionMatrix)
    print("State mapping:", MarketModel.StateMapping)
    Data = MarketModel.PredictRegime(Data, TechnicalFeatureColumns)
    Data["MostLikelyState"] = Data["MostLikelyState"].shift(-1)
    Data["StateProbability"] = Data["StateProbability"].shift(-1)

    Stats = RunBacktest(
        Data,
        TrailingTakeProfit=0.03,
        RiskPercent=0.01,
        StateProbabilityThreshold=0.6,
    )
    print("Backtest statistics:\n", Stats)


if __name__ == "__main__":
    Main()
