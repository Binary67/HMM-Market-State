import numpy as np


def StandardizeObservations(Observations: np.ndarray) -> np.ndarray:
    """Return standardized observations with zero mean and unit variance."""
    Mean = Observations.mean(axis=0)
    Std = Observations.std(axis=0)
    Std[Std == 0] = 1
    return (Observations - Mean) / Std
