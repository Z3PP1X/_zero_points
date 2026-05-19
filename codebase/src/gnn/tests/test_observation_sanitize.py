import numpy as np
import torch

from observation_sanitize import finite_float, sanitize_numpy_features, sanitize_torch_features


def test_finite_float_replaces_non_finite_values():
    assert finite_float(float("nan")) == 0.0
    assert finite_float(float("inf")) == 0.0
    assert finite_float("not-a-number") == 0.0
    assert finite_float(1.25) == 1.25


def test_sanitize_numpy_and_torch_features():
    array = np.array([1.0, np.nan, np.inf], dtype=np.float32)
    assert np.allclose(sanitize_numpy_features(array), [1.0, 0.0, 0.0])

    tensor = torch.tensor([1.0, float("nan"), float("-inf")])
    cleaned = sanitize_torch_features(tensor)
    assert torch.allclose(cleaned, torch.tensor([1.0, 0.0, 0.0]))
