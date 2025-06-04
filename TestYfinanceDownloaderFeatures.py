import pandas as pd
from YfinanceDownloader import DownloadTradingData


def test_download_trading_data_has_technical_features():
    data = DownloadTradingData("AAPL", "2021-01-01", "2021-01-31", "1d")
    assert "MA20" in data.columns
    assert "MA50" in data.columns
    assert "RSI" in data.columns
