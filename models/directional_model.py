"""Compact sequence models for short-horizon mid-price return forecasting.

The primary model is a GRU that maps ``(batch, 20, 14)`` feature windows to a
single predicted future mid-price return. Baselines exist only for evaluation.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

try:
    import torch
    import torch.nn as nn
except ImportError:  # pragma: no cover - exercised only when torch is absent
    torch = None
    nn = None


NUM_FEATURES = 14
CONTEXT_WINDOW = 20
DEFAULT_HIDDEN_SIZE = 64
DEFAULT_NUM_LAYERS = 1
DEFAULT_DROPOUT = 0.1
DIRECTIONAL_MODEL_VERSION = "directional-gru-v1"


def select_device(prefer_cuda: bool = True):
    """Return ``cuda`` when available, otherwise ``cpu``. Tests never require CUDA."""
    if torch is None:
        return "cpu"
    if prefer_cuda and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def require_torch():
    if torch is None or nn is None:
        raise ImportError(
            "PyTorch is required for directional forecasting. "
            "Install a CPU wheel with: pip install torch"
        )
    return torch, nn


class DirectionalGRU(nn.Module if nn is not None else object):
    """GRU encoder with a small fully-connected regression head.

    Input:
        x with shape ``(batch, context_window, num_features)``
    Output:
        predicted future return with shape ``(batch,)``
    """

    def __init__(
        self,
        num_features: int = NUM_FEATURES,
        hidden_size: int = DEFAULT_HIDDEN_SIZE,
        num_layers: int = DEFAULT_NUM_LAYERS,
        dropout: float = DEFAULT_DROPOUT,
    ) -> None:
        require_torch()
        super().__init__()
        if num_features <= 0 or hidden_size <= 0 or num_layers <= 0:
            raise ValueError("num_features, hidden_size, and num_layers must be positive.")
        if dropout < 0.0 or dropout >= 1.0:
            raise ValueError("dropout must be in [0, 1).")

        self.num_features = int(num_features)
        self.hidden_size = int(hidden_size)
        self.num_layers = int(num_layers)
        self.dropout = float(dropout)

        gru_dropout = self.dropout if self.num_layers > 1 else 0.0
        self.gru = nn.GRU(
            input_size=self.num_features,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            batch_first=True,
            dropout=gru_dropout,
        )
        self.head = nn.Sequential(
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_size, max(8, self.hidden_size // 2)),
            nn.ReLU(),
            nn.Linear(max(8, self.hidden_size // 2), 1),
        )

    def forward(self, x):
        if x.ndim != 3:
            raise ValueError(f"Expected input shape (batch, time, features); got {tuple(x.shape)}.")
        if x.shape[-1] != self.num_features:
            raise ValueError(
                f"Expected {self.num_features} features; got {x.shape[-1]}."
            )
        if not torch.isfinite(x).all():
            raise ValueError("Input tensor contains NaN or infinite values.")
        _output, hidden = self.gru(x)
        last_hidden = hidden[-1]
        predicted_return = self.head(last_hidden).squeeze(-1)
        return predicted_return

    def count_parameters(self) -> int:
        return int(sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad))


class ZeroForecast:
    """Always predicts a zero return."""

    name = "zero-return"

    def predict(self, sequences: np.ndarray) -> np.ndarray:
        values = np.asarray(sequences)
        if values.ndim != 3:
            raise ValueError("sequences must have shape (N, T, F).")
        return np.zeros(values.shape[0], dtype=np.float64)


class PersistenceForecast:
    """Predicts the last observed mid-price return (feature index 0)."""

    name = "last-return-persistence"

    def predict(self, sequences: np.ndarray) -> np.ndarray:
        values = np.asarray(sequences, dtype=np.float64)
        if values.ndim != 3:
            raise ValueError("sequences must have shape (N, T, F).")
        return values[:, -1, 0].copy()


class RidgeFlattenedForecast:
    """Ridge regression on the flattened (T * F) window. Fit on train only."""

    name = "ridge-flattened"

    def __init__(self, alpha: float = 1.0) -> None:
        from sklearn.linear_model import Ridge

        self.alpha = float(alpha)
        self._model = Ridge(alpha=self.alpha)
        self._fitted = False

    def fit(self, sequences: np.ndarray, targets: np.ndarray) -> "RidgeFlattenedForecast":
        values = np.asarray(sequences, dtype=np.float64)
        y = np.asarray(targets, dtype=np.float64)
        if values.ndim != 3:
            raise ValueError("sequences must have shape (N, T, F).")
        self._model.fit(values.reshape(values.shape[0], -1), y)
        self._fitted = True
        return self

    def predict(self, sequences: np.ndarray) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("RidgeFlattenedForecast must be fit before predict().")
        values = np.asarray(sequences, dtype=np.float64)
        return np.asarray(self._model.predict(values.reshape(values.shape[0], -1)), dtype=np.float64)


__all__ = [
    "NUM_FEATURES",
    "CONTEXT_WINDOW",
    "DEFAULT_HIDDEN_SIZE",
    "DEFAULT_NUM_LAYERS",
    "DEFAULT_DROPOUT",
    "DIRECTIONAL_MODEL_VERSION",
    "select_device",
    "require_torch",
    "DirectionalGRU",
    "ZeroForecast",
    "PersistenceForecast",
    "RidgeFlattenedForecast",
]
