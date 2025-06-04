import pandas as pd
from YfinanceDownloader import DownloadTradingData


def test_download_trading_data_returns_dataframe():
    DataFrame = DownloadTradingData("AAPL", "2021-01-01", "2021-01-10", "1d")
    assert isinstance(DataFrame, pd.DataFrame)
    assert not isinstance(DataFrame.index, pd.MultiIndex)
    assert not isinstance(DataFrame.columns, pd.MultiIndex)
