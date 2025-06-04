import numpy as np
from ObservationsHelper import StandardizeObservations


def test_standardize_observations_zero_mean_unit_variance():
    Data = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])
    Standardized = StandardizeObservations(Data)
    assert np.allclose(Standardized.mean(axis=0), 0.0)
    assert np.allclose(Standardized.std(axis=0), 1.0)
