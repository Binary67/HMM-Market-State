import pandas as pd
import yfinance as yf


def DownloadTradingData(TickerSymbol: str, StartDate: str, EndDate: str, Interval: str) -> pd.DataFrame:
    """Download trading data from Yahoo Finance and return a flat DataFrame."""
    Data = yf.download(tickers=TickerSymbol, start=StartDate, end=EndDate, interval=Interval)
    if isinstance(Data.columns, pd.MultiIndex):
        Data.columns = ['_'.join(Column).strip() for Column in Data.columns.to_flat_index()]
    if isinstance(Data.index, pd.MultiIndex):
        Data.reset_index(inplace=True)
    else:
        Data = Data.reset_index()
    return Data
