import pandas as pd

class RegimeAccuracyChecker:
    """Calculate accuracy of HMM regime predictions."""

    def __init__(self, LookaheadDays: int = 10, SidewayThreshold: float = 0.02) -> None:
        self.LookaheadDays = LookaheadDays
        self.SidewayThreshold = SidewayThreshold

    def CalculateAccuracy(self, Data: pd.DataFrame, RegimeColumn: str, ReturnColumn: str) -> float:
        CorrectCount = 0
        TotalCount = len(Data) - self.LookaheadDays
        for Index in range(TotalCount):
            Regime = Data[RegimeColumn].iloc[Index]
            FutureReturn = Data[ReturnColumn].iloc[Index + 1: Index + 1 + self.LookaheadDays].sum()
            if Regime == "Uptrend" and FutureReturn > 0:
                CorrectCount += 1
            elif Regime == "Downtrend" and FutureReturn < 0:
                CorrectCount += 1
            elif Regime == "Sideway" and abs(FutureReturn) <= self.SidewayThreshold:
                CorrectCount += 1
        if TotalCount <= 0:
            return 0.0
        return CorrectCount / TotalCount

    def GridSearchParameters(
        self,
        Data: pd.DataFrame,
        RegimeColumn: str,
        ReturnColumn: str,
        LookaheadOptions: list[int],
        ThresholdOptions: list[float],
    ) -> tuple[int, float]:
        """Return the LookaheadDays and SidewayThreshold yielding the best accuracy."""
        BestLookahead = self.LookaheadDays
        BestThreshold = self.SidewayThreshold
        BestAccuracy = -1.0
        for Lookahead in LookaheadOptions:
            for Threshold in ThresholdOptions:
                Checker = RegimeAccuracyChecker(Lookahead, Threshold)
                Accuracy = Checker.CalculateAccuracy(Data, RegimeColumn, ReturnColumn)
                if Accuracy > BestAccuracy:
                    BestAccuracy = Accuracy
                    BestLookahead = Lookahead
                    BestThreshold = Threshold
        self.LookaheadDays = BestLookahead
        self.SidewayThreshold = BestThreshold
        return BestLookahead, BestThreshold
