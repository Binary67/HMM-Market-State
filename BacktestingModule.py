from backtesting import Backtest, Strategy
import pandas as pd


class RegimeStrategy(Strategy):
    TrailingTakeProfit = 0.05
    RiskPercent = 0.05
    StateProbabilityThreshold = 0.6

    def init(self):
        self.HighPrice = None
        self.LowPrice = None
        self.InitialCapital = self.equity

    def next(self):
        close = self.data.Close[-1]
        regime = self.data.Regime[-1]
        predicted_state = self.data.MostLikelyState[-1]
        state_probability = self.data.StateProbability[-1]

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
            if (
                not self.position.is_long
                and predicted_state == "Uptrend"
                and state_probability >= self.StateProbabilityThreshold
            ):
                if self.position:
                    self.position.close()
                RiskCapital = self.InitialCapital * self.RiskPercent
                RiskPerUnit = close * self.TrailingTakeProfit
                Size = int(RiskCapital / RiskPerUnit) if RiskPerUnit else 0
                Size = min(Size, int(self.equity / close))
                Size = max(Size, 1)
                self.buy(size=Size, sl=close * (1 - self.TrailingTakeProfit))
                self.HighPrice = close
                self.LowPrice = None
        elif regime == "Downtrend":
            if (
                not self.position.is_short
                and predicted_state == "Downtrend"
                and state_probability >= self.StateProbabilityThreshold
            ):
                if self.position:
                    self.position.close()
                RiskCapital = self.InitialCapital * self.RiskPercent
                RiskPerUnit = close * self.TrailingTakeProfit
                Size = int(RiskCapital / RiskPerUnit) if RiskPerUnit else 0
                Size = min(Size, int(self.equity / close))
                Size = max(Size, 1)
                self.sell(size=Size, sl=close * (1 + self.TrailingTakeProfit))
                self.LowPrice = close
                self.HighPrice = None
        else:  # Sideway
            if self.position:
                self.position.close()
                self.HighPrice = None
                self.LowPrice = None


def RunBacktest(
    Data: pd.DataFrame,
    TrailingTakeProfit: float = 0.05,
    RiskPercent: float = 0.05,
    StateProbabilityThreshold: float = 0.6,
) -> pd.DataFrame:
    RequiredColumns = {
        "Open",
        "High",
        "Low",
        "Close",
        "Regime",
        "MostLikelyState",
        "StateProbability",
    }
    MissingCols = RequiredColumns - set(Data.columns)
    if MissingCols:
        raise ValueError(f"Dataframe is missing required columns: {MissingCols}")

    CleanData = Data.dropna(subset=["Close", "Regime", "MostLikelyState", "StateProbability"])
    bt = Backtest(CleanData, RegimeStrategy, cash=10000, commission=0.0)
    stats = bt.run(
        TrailingTakeProfit=TrailingTakeProfit,
        RiskPercent=RiskPercent,
        StateProbabilityThreshold=StateProbabilityThreshold,
    )
    return stats
