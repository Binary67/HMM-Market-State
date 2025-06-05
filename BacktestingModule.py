from backtesting import Backtest, Strategy
import pandas as pd


class RegimeStrategy(Strategy):
    TrailingTakeProfit = 0.05

    def init(self):
        self.HighPrice = None
        self.LowPrice = None

    def next(self):
        close = self.data.Close[-1]
        regime = self.data.Regime[-1]

        if self.position:
            if self.position.is_long:
                self.HighPrice = (
                    max(self.HighPrice, close) if self.HighPrice is not None else close
                )
                TrailingPrice = self.HighPrice * (1 - self.TrailingTakeProfit)
                if close <= TrailingPrice:
                    self.position.close()
                    self.HighPrice = None
                else:
                    self.position.sl = TrailingPrice
            elif self.position.is_short:
                self.LowPrice = (
                    min(self.LowPrice, close) if self.LowPrice is not None else close
                )
                TrailingPrice = self.LowPrice * (1 + self.TrailingTakeProfit)
                if close >= TrailingPrice:
                    self.position.close()
                    self.LowPrice = None
                else:
                    self.position.sl = TrailingPrice

        if regime == "Uptrend":
            if not self.position.is_long:
                if self.position:
                    self.position.close()
                self.buy()
                self.HighPrice = close
                self.LowPrice = None
        elif regime == "Downtrend":
            if not self.position.is_short:
                if self.position:
                    self.position.close()
                self.sell()
                self.LowPrice = close
                self.HighPrice = None
        else:  # Sideway
            if self.position:
                self.position.close()
                self.HighPrice = None
                self.LowPrice = None


def RunBacktest(Data: pd.DataFrame, TrailingTakeProfit: float = 0.05) -> pd.DataFrame:
    RequiredColumns = {"Open", "High", "Low", "Close", "Regime"}
    MissingCols = RequiredColumns - set(Data.columns)
    if MissingCols:
        raise ValueError(f"Dataframe is missing required columns: {MissingCols}")

    CleanData = Data.dropna(subset=["Close", "Regime"])
    bt = Backtest(CleanData, RegimeStrategy, cash=10000, commission=0.0)
    stats = bt.run(TrailingTakeProfit=TrailingTakeProfit)
    return stats
