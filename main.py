from DataDownloader import DownloadTradingData


def Main() -> None:
    Data = DownloadTradingData(
        "AAPL",
        "2024-01-01",
        "2024-01-15",
        "1d",
    )
    print(Data.head())


if __name__ == "__main__":
    Main()
