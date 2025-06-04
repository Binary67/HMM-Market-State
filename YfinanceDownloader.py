import numpy as np
import pandas as pd
import yfinance as yf


def EnsurePandasTaCompatibility() -> None:
    """Ensure pandas-ta works with newer NumPy versions."""
    if not hasattr(np, "NaN"):
        np.NaN = np.nan


EnsurePandasTaCompatibility()
import pandas_ta as ta



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
    HighColumn = [Column for Column in Data.columns if Column.startswith("High")][0]
    LowColumn = [Column for Column in Data.columns if Column.startswith("Low")][0]
    VolumeColumn = [Column for Column in Data.columns if Column.startswith("Volume")][0]

    Data["MA20"] = ta.sma(Data[CloseColumn], length=20)
    Data["MA50"] = ta.sma(Data[CloseColumn], length=50)
    Data["RSI"] = ta.rsi(Data[CloseColumn], length=14)
    Data["ATR14"] = ta.atr(Data[HighColumn], Data[LowColumn], Data[CloseColumn], length=14)
    Stoch = ta.stoch(Data[HighColumn], Data[LowColumn], Data[CloseColumn], k=14)
    Data["STOCH"] = Stoch.iloc[:, 0] if isinstance(Stoch, pd.DataFrame) else Stoch
    Data["OBV"] = ta.obv(Data[CloseColumn], Data[VolumeColumn])
    return Data
