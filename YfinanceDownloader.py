import pandas as pd
import yfinance as yf


def DownloadTradingData(TickerSymbol: str, StartDate: str, EndDate: str, Interval: str) -> pd.DataFrame:
    """Download trading data from Yahoo Finance and return a flat DataFrame.

    The DataFrame includes Moving Averages, RSI, ATR, Stochastic oscillator and
    On Balance Volume for richer HMM observations.
    """
    Data = yf.download(tickers=TickerSymbol, start=StartDate, end=EndDate, interval=Interval)
    if isinstance(Data.columns, pd.MultiIndex):
        Data.columns = ["_".join(Column).strip() for Column in Data.columns.to_flat_index()]
    if isinstance(Data.index, pd.MultiIndex):
        Data.reset_index(inplace=True)
    else:
        Data = Data.reset_index()

    CloseColumn = [Column for Column in Data.columns if Column.startswith("Close")][0]
    Data["MA20"] = Data[CloseColumn].rolling(window=20, min_periods=1).mean()
    Data["MA50"] = Data[CloseColumn].rolling(window=50, min_periods=1).mean()
    Delta = Data[CloseColumn].diff()
    Up = Delta.clip(lower=0)
    Down = -1 * Delta.clip(upper=0)
    RollUp = Up.rolling(window=14, min_periods=1).mean()
    RollDown = Down.rolling(window=14, min_periods=1).mean()
    Rs = RollUp / RollDown
    Data["RSI"] = 100.0 - (100.0 / (1.0 + Rs))

    HighColumn = [Column for Column in Data.columns if Column.startswith("High")][0]
    LowColumn = [Column for Column in Data.columns if Column.startswith("Low")][0]
    VolumeColumn = [Column for Column in Data.columns if Column.startswith("Volume")][0]
    PrevClose = Data[CloseColumn].shift(1)
    HighLow = Data[HighColumn] - Data[LowColumn]
    HighPrevClose = (Data[HighColumn] - PrevClose).abs()
    LowPrevClose = (Data[LowColumn] - PrevClose).abs()
    TrueRange = pd.concat([HighLow, HighPrevClose, LowPrevClose], axis=1).max(axis=1)
    Data["ATR14"] = TrueRange.rolling(window=14, min_periods=1).mean()

    LowestLow = Data[LowColumn].rolling(window=14, min_periods=1).min()
    HighestHigh = Data[HighColumn].rolling(window=14, min_periods=1).max()
    Range = HighestHigh - LowestLow
    Range[Range == 0] = 1
    Data["STOCH"] = 100 * (Data[CloseColumn] - LowestLow) / Range

    PriceDiff = Data[CloseColumn].diff()
    Direction = PriceDiff.apply(lambda Value: 1 if Value > 0 else -1 if Value < 0 else 0)
    Data["OBV"] = (Direction * Data[VolumeColumn]).cumsum()
    return Data
