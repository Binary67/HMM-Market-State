import pandas as pd
import yfinance as yf


def DownloadMarketData(TickerSymbol: str, StartDate: str, EndDate: str, Interval: str) -> pd.DataFrame:
    """Download OHLCV trading data from Yahoo Finance as a multi-index DataFrame."""
    Data = yf.download(
        tickers=[TickerSymbol],
        start=StartDate,
        end=EndDate,
        interval=Interval,
        group_by="ticker",
        auto_adjust=False,
        progress=False,
    )

    # Ensure multi-index even for a single ticker
    if not isinstance(Data.columns, pd.MultiIndex):
        Data.columns = pd.MultiIndex.from_product([[TickerSymbol], Data.columns])

    OhclvColumns = [
        (TickerSymbol, "Open"),
        (TickerSymbol, "High"),
        (TickerSymbol, "Low"),
        (TickerSymbol, "Close"),
        (TickerSymbol, "Volume"),
    ]

    # Filter to OHLCV columns and drop others
    Data = Data.loc[:, OhclvColumns]
    return Data
