import pandas as pd
from YfinanceDownloader import DownloadTradingData
from MarketRegimeHmm import MarketRegimeHmm


def test_market_regime_hmm_predicts_labels():
    Data = DownloadTradingData("AAPL", "2021-01-01", "2021-01-15", "1d")
    CloseColumn = [Column for Column in Data.columns if Column.startswith("Close")][0]
    Observations = Data[CloseColumn].pct_change().dropna().values.reshape(-1, 1)
    HmmModel = MarketRegimeHmm()
    HmmModel.Fit(Observations)
    Labels = HmmModel.PredictRegimes(Observations)
    assert len(Labels) == len(Observations)
    assert set(Labels).issubset({"Uptrend", "Downtrend", "Sideway"})
