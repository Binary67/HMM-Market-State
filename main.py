from DataDownloader import DownloadTradingData
from FeatureEngineering import FeatureEngineering


def Main() -> None:
    Data = DownloadTradingData(
        "AAPL",
        "2024-01-01",
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
        "BollingerBands": {"Window": 20, "StandardDeviations": 2.0}
    }
    
    Data = FeatureEngineer.ApplyTechnicalAnalysis(
        Data, IndicatorsToApply, IndicatorParameters
    )


if __name__ == "__main__":
    Main()
