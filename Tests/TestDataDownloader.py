import unittest
import pandas as pd
from HmmTrading.DataDownloader import YahooFinanceDownloader

class TestYahooFinanceDownloader(unittest.TestCase):

    def test_ValidTickerSymbol(self):
        print("Running test_ValidTickerSymbol...")
        Downloader = YahooFinanceDownloader(TickerSymbol="MSFT")
        self.assertEqual(Downloader.TickerSymbol, "MSFT")
        print("test_ValidTickerSymbol PASSED")

    def test_InvalidTickerSymbol_Empty(self):
        print("Running test_InvalidTickerSymbol_Empty...")
        with self.assertRaises(ValueError):
            YahooFinanceDownloader(TickerSymbol="")
        print("test_InvalidTickerSymbol_Empty PASSED")

    def test_InvalidTickerSymbol_NonString(self):
        print("Running test_InvalidTickerSymbol_NonString...")
        with self.assertRaises(ValueError):
            YahooFinanceDownloader(TickerSymbol=12345) # type: ignore
        print("test_InvalidTickerSymbol_NonString PASSED")

    def test_DownloadData_Valid(self):
        print("Running test_DownloadData_Valid...")
        # Using a known ticker that is unlikely to be delisted soon for a short period
        Downloader = YahooFinanceDownloader(TickerSymbol="GOOG")
        Data = Downloader.DownloadData(StartDate="2023-01-01", EndDate="2023-01-10")
        self.assertIsInstance(Data, pd.DataFrame)
        self.assertFalse(Data.empty)
        self.assertIn("Close", Data.columns)
        print(f"test_DownloadData_Valid PASSED (Downloaded {len(Data)} rows for GOOG)")

    def test_DownloadData_InvalidTicker(self):
        print("Running test_DownloadData_InvalidTicker...")
        Downloader = YahooFinanceDownloader(TickerSymbol="THISISNOTAVALIDTICKERXYZ")
        Data = Downloader.DownloadData(StartDate="2023-01-01", EndDate="2023-01-10")
        self.assertIsInstance(Data, pd.DataFrame)
        self.assertTrue(Data.empty)
        print("test_DownloadData_InvalidTicker PASSED")

    def test_DownloadData_NoDataPeriod(self):
        print("Running test_DownloadData_NoDataPeriod...")
        # Using a valid ticker but a future period where no data should exist
        Downloader = YahooFinanceDownloader(TickerSymbol="AAPL")
        Data = Downloader.DownloadData(StartDate="2099-01-01", EndDate="2099-01-10")
        self.assertIsInstance(Data, pd.DataFrame)
        self.assertTrue(Data.empty)
        print("test_DownloadData_NoDataPeriod PASSED")

if __name__ == '__main__':
    print("Starting tests for DataDownloader...")
    unittest.main(argv=['first-arg-is-ignored'], exit=False)
    print("Finished tests for DataDownloader.")
