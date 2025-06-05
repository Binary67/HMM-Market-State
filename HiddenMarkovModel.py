import pandas as pd
import numpy as np
from hmmlearn.hmm import GaussianHMM
from typing import List


class HiddenMarkovModel:
    """Simple wrapper around GaussianHMM for market regime detection."""

    def __init__(self, NumberOfStates: int = 3, Iterations: int = 100) -> None:
        self.NumberOfStates = NumberOfStates
        self.Iterations = Iterations
        self.Model = GaussianHMM(n_components=NumberOfStates, covariance_type="full", n_iter=Iterations)

    def Fit(self, Data: pd.DataFrame, FeatureColumns: List[str]) -> None:
        CleanData = Data.dropna(subset=FeatureColumns)
        if CleanData.empty:
            raise ValueError(
                "No data available to train the HMM. "
                "Please ensure your dataset has sufficient rows without missing values for all features."
            )
        TrainingMatrix = CleanData[FeatureColumns].values
        self.Model.fit(TrainingMatrix)
        self.TrainingIndex = CleanData.index
        self.StateOrder = np.argsort(self.Model.means_[:, 0])

    def GetTransitionProbabilities(self) -> pd.DataFrame:
        """Return transition probability matrix as a DataFrame."""
        Matrix = self.Model.transmat_
        Mapping = {
            self.StateOrder[-1]: "Uptrend",
            self.StateOrder[0]: "Downtrend",
            self.StateOrder[1]: "Sideway",
        }
        Labels = [Mapping.get(i, f"State{i}") for i in range(self.NumberOfStates)]
        return pd.DataFrame(Matrix, index=Labels, columns=Labels)

    def PredictRegime(self, Data: pd.DataFrame, FeatureColumns: List[str]) -> pd.DataFrame:
        ResultData = Data.copy()
        CleanData = ResultData.dropna(subset=FeatureColumns)
        ObservationMatrix = CleanData[FeatureColumns].values
        Predictions = self.Model.predict(ObservationMatrix)
        Probabilities = self.Model.predict_proba(ObservationMatrix)
        Mapping = {
            self.StateOrder[-1]: "Uptrend",
            self.StateOrder[0]: "Downtrend",
            self.StateOrder[1]: "Sideway",
        }
        RegimeSeries = pd.Series(index=CleanData.index, data=[Mapping[p] for p in Predictions])
        ResultData["Regime"] = RegimeSeries
        MostLikelyStates = Probabilities.argmax(axis=1)
        MaxProbabilities = Probabilities.max(axis=1)
        StateSeries = pd.Series(index=CleanData.index, data=[Mapping[p] for p in MostLikelyStates])
        ProbabilitySeries = pd.Series(index=CleanData.index, data=MaxProbabilities)
        ResultData["MostLikelyState"] = StateSeries
        ResultData["StateProbability"] = ProbabilitySeries
        return ResultData
