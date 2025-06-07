import streamlit as st
import numpy as np
import pandas as pd
from typing import Tuple, Dict

from DataDownloader import DownloadTradingData
from FeatureEngineering import FeatureEngineering
from HiddenMarkovModel import HiddenMarkovModel
from ModelEvaluation import EvaluateRegimePrediction
from BacktestingModule import RunBacktest


def FormatBacktestStats(Stats: pd.Series) -> pd.DataFrame:
    """Convert stats to a DataFrame friendly for Streamlit display."""
    StatsFrame = Stats.to_frame(name="Value")
    StatsFrame["Value"] = StatsFrame["Value"].apply(
        lambda X: str(X) if isinstance(X, pd.Timedelta) else X
    )
    return StatsFrame


def RunAnalysis(
    Ticker: str,
    StartDate: str,
    EndDate: str,
    Interval: str,
    TrailingTakeProfit: float,
    RiskPercent: float,
    StateProbabilityThreshold: float,
    Data: pd.DataFrame | None = None,
) -> Tuple[pd.DataFrame, Dict[str, float], pd.Series]:
    """Execute the full analysis pipeline and return processed data, evaluation metrics and backtest statistics."""
    if Data is None:
        Data = DownloadTradingData(Ticker, StartDate, EndDate, Interval)

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
    Data = MarketModel.PredictRegime(Data, TechnicalFeatureColumns)
    Data["MostLikelyState"] = Data["MostLikelyState"].shift(-1)
    Data["StateProbability"] = Data["StateProbability"].shift(-1)
    Stats = RunBacktest(
        Data,
        TrailingTakeProfit=TrailingTakeProfit,
        RiskPercent=RiskPercent,
        StateProbabilityThreshold=StateProbabilityThreshold,
    )
    return Data, Metrics, Stats


def DisplayInterface() -> None:
    """Launch a Streamlit interface for running the analysis pipeline."""
    st.title("HMM Market State Backtester")
    TickerInput = st.text_input("Ticker Symbol", "AAPL")
    StartDateInput = st.date_input("Start Date")
    EndDateInput = st.date_input("End Date")
    IntervalInput = st.selectbox("Interval", ["1d", "1h", "30m", "15m"])
    TrailingTakeProfitInput = st.slider("Trailing Take Profit", 0.01, 0.2, 0.03)
    RiskPercentInput = st.slider("Risk Percent", 0.01, 0.2, 0.01)
    StateProbabilityThresholdInput = st.slider("State Probability Threshold", 0.5, 1.0, 0.6)

    if st.button("Run Backtest"):
        Data, Metrics, Stats = RunAnalysis(
            TickerInput,
            StartDateInput.strftime("%Y-%m-%d"),
            EndDateInput.strftime("%Y-%m-%d"),
            IntervalInput,
            TrailingTakeProfitInput,
            RiskPercentInput,
            StateProbabilityThresholdInput,
        )
        st.subheader("Validation Metrics")
        st.json(Metrics)
        st.subheader("Backtest Performance")
        FormattedStats = FormatBacktestStats(Stats)
        st.write(FormattedStats)
        EquityCurve = Stats._equity_curve
        st.subheader("Equity Curve")
        st.line_chart(EquityCurve["Equity"])
        st.subheader("Closing Price")
        st.line_chart(Data["Close"])  # Price chart


if __name__ == "__main__":
    DisplayInterface()
