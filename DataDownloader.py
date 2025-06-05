import pandas as pd
import yfinance as yf


def DownloadTradingData(TickerSymbol: str, StartDate: str, EndDate: str, Interval: str) -> pd.DataFrame:
    """Download OHLCV trading data from yfinance and return a DataFrame with flat columns."""
    Data = yf.download(
        tickers=TickerSymbol,
        start=StartDate,
        end=EndDate,
        interval=Interval,
        group_by="ticker",
        auto_adjust=False,
        threads=True,
        progress=False,
    )

    OhlcvColumns = ["Open", "High", "Low", "Close", "Volume"]
    if isinstance(Data.columns, pd.MultiIndex):
        Data = Data.loc[:, Data.columns.get_level_values(1).isin(OhlcvColumns)]
        Data.columns = Data.columns.get_level_values(1)
    else:
        Data = Data.loc[:, Data.columns.isin(OhlcvColumns)]
    return Data
