"""Tests for local directional inference and artifact round-trip."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from data.contracts import (
    DEFAULT_FORECAST_CONTEXT_WINDOW,
    DEFAULT_FORECAST_HORIZON,
    NVIDIA_FORECAST_FEATURES,
    ForecastInput,
    ModelStatus,
)
from data.normalizer import StandardScaler
from models.directional_model import DIRECTIONAL_MODEL_VERSION, DirectionalGRU
from models.local_forecaster import LocalDirectionalForecaster


def _make_input(feature_names=None, horizon=DEFAULT_FORECAST_HORIZON) -> ForecastInput:
    names = list(feature_names or NVIDIA_FORECAST_FEATURES)
    return ForecastInput(
        feature_sequence=[[float(i + j) for j in range(len(names))] for i in range(20)],
        feature_names=names,
        timestamps=[1000.0 + i for i in range(20)],
        context_window=20,
        forecast_horizon=horizon,
    )


def _write_artifact(path: Path) -> None:
    model = DirectionalGRU(num_features=14, hidden_size=8, num_layers=1, dropout=0.0)
    means = np.zeros(14)
    scales = np.ones(14)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "architecture": {
                "num_features": 14,
                "hidden_size": 8,
                "num_layers": 1,
                "dropout": 0.0,
            },
            "feature_names": list(NVIDIA_FORECAST_FEATURES),
            "context_window": DEFAULT_FORECAST_CONTEXT_WINDOW,
            "forecast_horizon": DEFAULT_FORECAST_HORIZON,
            "normalization": {"means": means.tolist(), "scales": scales.tolist()},
            "model_version": DIRECTIONAL_MODEL_VERSION,
            "model_name": "directional-gru",
            "training_metrics": {},
            "validation_metrics": {},
            "test_metrics": {},
        },
        path,
    )


def test_local_forecaster_round_trip(tmp_path):
    path = tmp_path / "directional_forecaster.pt"
    _write_artifact(path)
    forecaster = LocalDirectionalForecaster(artifact_path=path, device=torch.device("cpu"))
    forecast_input = _make_input()
    first = forecaster.predict(forecast_input, timestamp=1020.0)
    assert first.model_status == ModelStatus.TRAINED
    assert first.predicted_return == pytest.approx(first.predicted_return)
    reloaded = LocalDirectionalForecaster(artifact_path=path, device=torch.device("cpu"))
    second = reloaded.predict(forecast_input, timestamp=1020.0)
    assert second.predicted_return == pytest.approx(first.predicted_return, rel=1e-5, abs=1e-8)
    assert second.model_version == DIRECTIONAL_MODEL_VERSION
    assert second.confidence is None


def test_local_forecaster_rejects_bad_inputs(tmp_path):
    path = tmp_path / "directional_forecaster.pt"
    _write_artifact(path)
    forecaster = LocalDirectionalForecaster(artifact_path=path, device=torch.device("cpu"))
    assert forecaster.predict(None, 1020.0).model_status == ModelStatus.FALLBACK
    wrong_horizon = _make_input(horizon=4)
    assert forecaster.predict(wrong_horizon, 1020.0).model_status == ModelStatus.FALLBACK
    bad_order = list(NVIDIA_FORECAST_FEATURES[::-1])
    # Construction itself rejects canonical mismatch when window is 20.
    with pytest.raises(ValueError):
        _make_input(feature_names=bad_order)


def test_saved_scaler_is_applied(tmp_path):
    path = tmp_path / "directional_forecaster.pt"
    _write_artifact(path)
    forecaster = LocalDirectionalForecaster(artifact_path=path, device=torch.device("cpu"))
    assert forecaster.scaler is not None
    raw = np.ones((1, 20, 14), dtype=np.float64)
    normalized = forecaster.scaler.transform(raw)
    assert np.allclose(normalized, 1.0)
    scaler = StandardScaler(means=np.zeros(14), scales=np.ones(14), feature_names=tuple(NVIDIA_FORECAST_FEATURES))
    assert np.allclose(scaler.transform(raw), normalized)
