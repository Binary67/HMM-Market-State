import pandas as pd
from typing import List, Dict
from sklearn.metrics import accuracy_score, f1_score

from HiddenMarkovModel import HiddenMarkovModel
import numpy as np


def TrackEquityCurve(Data: pd.DataFrame, InitialCapital: float = 10000.0) -> pd.Series:
    """Return equity over time based on predicted market regime."""
    RequiredColumns = {"LogReturn", "MostLikelyState"}
    if not RequiredColumns.issubset(Data.columns):
        raise ValueError("Data must contain LogReturn and MostLikelyState columns")

    CleanData = Data.dropna(subset=["LogReturn", "MostLikelyState"])
    EquityValues = []
    CurrentEquity = InitialCapital
    for _, Row in CleanData.iterrows():
        Regime = Row["MostLikelyState"]
        LogReturn = Row["LogReturn"]
        if Regime == "Uptrend":
            GrowthFactor = float(np.exp(LogReturn))
        elif Regime == "Downtrend":
            GrowthFactor = float(np.exp(-LogReturn))
        else:
            GrowthFactor = 1.0
        CurrentEquity *= GrowthFactor
        EquityValues.append(CurrentEquity)

    return pd.Series(EquityValues, index=CleanData.index)


def EvaluateRegimePrediction(Data: pd.DataFrame, FeatureColumns: List[str], TrainFraction: float = 0.8) -> Dict[str, float]:
    """Train on a portion of the data and evaluate predictions on the remainder."""
    if "LogReturn" not in Data.columns:
        raise ValueError("Data must contain LogReturn column")

    SplitIndex = int(len(Data) * TrainFraction)
    TrainData = Data.iloc[:SplitIndex].copy()
    ValidationData = Data.iloc[SplitIndex:].copy()

    Model = HiddenMarkovModel()
    Model.Fit(TrainData, FeatureColumns)

    ValidationData = Model.PredictRegime(ValidationData, FeatureColumns)
    ValidationData = ValidationData.dropna(subset=["LogReturn", "MostLikelyState"])

    CleanValidation = ValidationData.dropna(subset=FeatureColumns)
    ValMatrix = Model.Scaler.transform(CleanValidation[FeatureColumns].values)
    LogLikelihood = Model.Model.score(ValMatrix)

    def DetermineActualRegime(Return):
        if Return > 0:
            return "Uptrend"
        elif Return < 0:
            return "Downtrend"
        return "Sideway"

    ValidationData["ActualRegime"] = ValidationData["LogReturn"].apply(DetermineActualRegime)

    Accuracy = accuracy_score(ValidationData["ActualRegime"], ValidationData["MostLikelyState"])
    F1 = f1_score(ValidationData["ActualRegime"], ValidationData["MostLikelyState"], average="macro")

    return {"LogLikelihood": LogLikelihood, "Accuracy": Accuracy, "F1Score": F1}
