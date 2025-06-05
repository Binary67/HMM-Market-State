from backtesting import Backtest, Strategy
import pandas as pd


class RegimeStrategy(Strategy):
    def init(self):
        pass

    def next(self):
        regime = self.data.Regime[-1]
        if regime == "Uptrend":
            if not self.position.is_long:
                if self.position:
                    self.position.close()
                self.buy()
        elif regime == "Downtrend":
            if not self.position.is_short:
                if self.position:
                    self.position.close()
                self.sell()
        else:  # Sideway
            if self.position:
                self.position.close()


def RunBacktest(Data: pd.DataFrame) -> pd.DataFrame:
    RequiredColumns = {"Open", "High", "Low", "Close", "Regime"}
    MissingCols = RequiredColumns - set(Data.columns)
    if MissingCols:
        raise ValueError(f"Dataframe is missing required columns: {MissingCols}")

    CleanData = Data.dropna(subset=["Close", "Regime"])
    bt = Backtest(CleanData, RegimeStrategy, cash=10000, commission=0.0)
    stats = bt.run()
    return stats
