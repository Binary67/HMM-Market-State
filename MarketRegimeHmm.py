import numpy as np
from hmmlearn.hmm import GaussianHMM


class MarketRegimeHmm:
    """Hidden Markov Model for market regime prediction."""

    def __init__(self, StateCount: int = 3) -> None:
        self.Model = GaussianHMM(n_components=StateCount, covariance_type="diag", n_iter=1000)

    def Fit(self, Observations: np.ndarray) -> None:
        self.Model.fit(Observations)

    def PredictStates(self, Observations: np.ndarray) -> np.ndarray:
        return self.Model.predict(Observations)

    def PredictRegimes(self, Observations: np.ndarray) -> list[str]:
        HiddenStates = self.Model.predict(Observations)
        Means = self.Model.means_.flatten()
        Order = np.argsort(Means)
        LabelMapping = {Order[2]: "Uptrend", Order[0]: "Downtrend", Order[1]: "Sideway"}
        return [LabelMapping[State] for State in HiddenStates]
