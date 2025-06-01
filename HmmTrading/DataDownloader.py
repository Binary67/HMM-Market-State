import yfinance as yf
import pandas as pd

class YahooFinanceDownloader:
    def __init__(self, TickerSymbol: str):
        if not isinstance(TickerSymbol, str) or not TickerSymbol:
            raise ValueError("TickerSymbol must be a non-empty string.")
        self.TickerSymbol = TickerSymbol
        self.Ticker = yf.Ticker(self.TickerSymbol)

    def DownloadData(self, StartDate: str, EndDate: str) -> pd.DataFrame:
        try:
            print(f"Downloading data for {self.TickerSymbol} from {StartDate} to {EndDate}...")
            HistoricalData = self.Ticker.history(start=StartDate, end=EndDate)
            if HistoricalData.empty:
                print(f"No data found for {self.TickerSymbol} between {StartDate} and {EndDate}.")
                return pd.DataFrame()
            print("Data downloaded successfully.")
            return HistoricalData
        except Exception as E:
            print(f"An error occurred while downloading data for {self.TickerSymbol}: {E}")
            # Return an empty DataFrame in case of any error
            return pd.DataFrame()

if __name__ == '__main__':
    # Example Usage
    Downloader = YahooFinanceDownloader(TickerSymbol="AAPL")
    AppleData = Downloader.DownloadData(StartDate="2020-01-01", EndDate="2023-01-01")
    if not AppleData.empty:
        print("Apple Inc. (AAPL) historical data:")
        print(AppleData.head())
    else:
        print("Failed to download Apple Inc. (AAPL) data.")

    print("\n--------------------------------------------------\n")

    DownloaderInvalid = YahooFinanceDownloader(TickerSymbol="INVALIDTICKER")
    InvalidData = DownloaderInvalid.DownloadData(StartDate="2020-01-01", EndDate="2023-01-01")
    if InvalidData.empty:
        print("Attempt to download data for INVALIDTICKER correctly resulted in no data or an error.")
    else:
        print("Downloaded data for INVALIDTICKER, which was not expected.")

    try:
        YahooFinanceDownloader(TickerSymbol="")
    except ValueError as VE:
        print(f"Correctly caught error for empty TickerSymbol: {VE}")

    try:
        YahooFinanceDownloader(TickerSymbol=123) # type: ignore
    except ValueError as VE:
        print(f"Correctly caught error for invalid TickerSymbol type: {VE}")
