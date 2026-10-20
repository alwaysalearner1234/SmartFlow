"""Tests for the directional GRU architecture and baselines."""

from __future__ import annotations

import numpy as np
import pytest

from models.directional_model import (
    DirectionalGRU,
    PersistenceForecast,
    RidgeFlattenedForecast,
    ZeroForecast,
    select_device,
)


def test_gru_forward_shape():
    model = DirectionalGRU(num_features=14, hidden_size=32, num_layers=1, dropout=0.1)
    import torch

    x = torch.zeros(4, 20, 14)
    y = model(x)
    assert tuple(y.shape) == (4,)


def test_cpu_inference_and_invalid_shapes():
    import torch

    model = DirectionalGRU(num_features=14, hidden_size=16)
    model.eval()
    with torch.no_grad():
        out = model(torch.randn(1, 20, 14))
    assert out.shape == (1,)

    with pytest.raises(ValueError):
        model(torch.randn(1, 20, 13))
    with pytest.raises(ValueError):
        model(torch.randn(20, 14))
    with pytest.raises(ValueError):
        model(torch.tensor([[[float("nan")] * 14] * 20]))


def test_select_device_without_requiring_cuda():
    device = select_device(prefer_cuda=False)
    assert str(device) == "cpu"
    device_auto = select_device(prefer_cuda=True)
    assert str(device_auto) in {"cpu", "cuda"}


def test_baselines():
    sequences = np.zeros((5, 20, 14), dtype=np.float64)
    sequences[:, -1, 0] = [0.1, -0.2, 0.0, 0.3, -0.1]
    targets = np.array([0.05, -0.04, 0.0, 0.02, -0.01])
    zero = ZeroForecast().predict(sequences)
    persist = PersistenceForecast().predict(sequences)
    ridge = RidgeFlattenedForecast().fit(sequences, targets).predict(sequences)
    assert np.allclose(zero, 0.0)
    assert np.allclose(persist, sequences[:, -1, 0])
    assert ridge.shape == (5,)
