import pandas as pd
import numpy as np
from hmmlearn.hmm import GaussianHMM
from sklearn.preprocessing import StandardScaler
from typing import List


class HiddenMarkovModel:
    """Simple wrapper around GaussianHMM for market regime detection."""

    def __init__(self, NumberOfStates: int = 3, Iterations: int = 100) -> None:
        self.NumberOfStates = NumberOfStates
        self.Iterations = Iterations
        self.Model = GaussianHMM(n_components=NumberOfStates, covariance_type="full", n_iter=Iterations)
        self.Scaler = StandardScaler()
        self.StateMapping = {}

    def Fit(self, Data: pd.DataFrame, FeatureColumns: List[str]) -> None:
        if "LogReturn" not in Data.columns:
            raise ValueError("LogReturn column is required for fitting the HMM")
        CleanData = Data.dropna(subset=FeatureColumns + ["LogReturn"])
        if CleanData.empty:
            raise ValueError(
                "No data available to train the HMM. "
                "Please ensure your dataset has sufficient rows without missing values for all features."
            )
        TrainingMatrix = CleanData[FeatureColumns].values
        ScaledMatrix = self.Scaler.fit_transform(TrainingMatrix)
        self.Model.fit(ScaledMatrix)
        self.TrainingIndex = CleanData.index
        PredictedStates = self.Model.predict(ScaledMatrix)
        LogReturnMeans = {}
        for State in range(self.NumberOfStates):
            StateReturns = CleanData.loc[PredictedStates == State, "LogReturn"]
            LogReturnMeans[State] = StateReturns.mean() if not StateReturns.empty else np.nan
        SortedStates = sorted(LogReturnMeans.items(), key=lambda x: x[1])
        self.StateMapping = {SortedStates[-1][0]: "Uptrend", SortedStates[0][0]: "Downtrend"}
        for State in range(self.NumberOfStates):
            if State not in self.StateMapping:
                self.StateMapping[State] = "Sideway"

    def GetTransitionProbabilities(self) -> pd.DataFrame:
        """Return transition probability matrix as a DataFrame."""
        Matrix = self.Model.transmat_
        Labels = [self.StateMapping.get(i, f"State{i}") for i in range(self.NumberOfStates)]
        return pd.DataFrame(Matrix, index=Labels, columns=Labels)

    def PredictRegime(self, Data: pd.DataFrame, FeatureColumns: List[str]) -> pd.DataFrame:
        ResultData = Data.copy()
        CleanData = ResultData.dropna(subset=FeatureColumns)
        ObservationMatrix = CleanData[FeatureColumns].values
        ScaledMatrix = self.Scaler.transform(ObservationMatrix)
        Predictions = self.Model.predict(ScaledMatrix)
        Probabilities = self.Model.predict_proba(ScaledMatrix)
        Mapping = self.StateMapping
        RegimeSeries = pd.Series(index=CleanData.index, data=[Mapping[p] for p in Predictions])
        ResultData["Regime"] = RegimeSeries
        MostLikelyStates = Probabilities.argmax(axis=1)
        MaxProbabilities = Probabilities.max(axis=1)
        StateSeries = pd.Series(index=CleanData.index, data=[Mapping[p] for p in MostLikelyStates])
        ProbabilitySeries = pd.Series(index=CleanData.index, data=MaxProbabilities)
        ResultData["MostLikelyState"] = StateSeries
        ResultData["StateProbability"] = ProbabilitySeries
        return ResultData
