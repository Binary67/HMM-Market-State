from DataDownloader import DownloadTradingData
from FeatureEngineering import FeatureEngineering
from HiddenMarkovModel import HiddenMarkovModel
from BacktestingModule import RunBacktest
import numpy as np


def Main() -> None:
    Data = DownloadTradingData(
        "AAPL",
        "2023-10-01",
        "2024-01-15",
        "1d",
    )

    FeatureEngineer = FeatureEngineering()

    IndicatorsToApply = ["SMA", "EMA", "RSI", "MACD", "BollingerBands"]
    IndicatorParameters = {
        "SMA": {"Window": 20},
        "EMA": {"Window": 12},
        "RSI": {"Window": 14},
        "MACD": {"FastPeriod": 12, "SlowPeriod": 26, "SignalPeriod": 9},
        "BollingerBands": {"Window": 20, "StandardDeviations": 2.0},
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

    Hmm = HiddenMarkovModel()
    Hmm.Fit(Data, TechnicalFeatureColumns)
    Data = Hmm.PredictRegime(Data, TechnicalFeatureColumns)
    print(Data.tail())

    Stats = RunBacktest(Data)
    print("Backtest statistics:\n", Stats)


if __name__ == "__main__":
    Main()
