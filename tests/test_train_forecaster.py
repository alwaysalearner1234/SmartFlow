"""Forecast trainer metrics and compact early-stopping behaviour."""

from __future__ import annotations

import numpy as np
import torch

from models.directional_model import DirectionalGRU
from models.train_forecaster import compute_regression_metrics


def test_compute_regression_metrics_detects_zero_model():
    y_true = np.array([0.1, -0.1, 0.05, -0.02])
    y_zero = np.zeros_like(y_true)
    metrics = compute_regression_metrics(y_true, y_zero)
    assert metrics["sample_count"] == 4
    assert metrics["mae"] > 0
    assert metrics["prediction_std"] == 0.0
    assert metrics["actual_return_std"] > 0


def test_early_stopping_restores_best_validation_checkpoint():
    torch.manual_seed(0)
    np.random.seed(0)
    x = torch.randn(32, 20, 14)
    y = x[:, -1, 0] * 0.01
    model = DirectionalGRU(num_features=14, hidden_size=8, dropout=0.0)
    criterion = torch.nn.SmoothL1Loss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.05)

    best_state = None
    best_val = float("inf")
    patience = 2
    stale = 0
    for _epoch in range(8):
        model.train()
        optimizer.zero_grad()
        loss = criterion(model(x), y)
        loss.backward()
        optimizer.step()
        model.eval()
        with torch.no_grad():
            val_loss = float(criterion(model(x), y).item())
        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
            if stale >= patience:
                break
    assert best_state is not None
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        restored = float(criterion(model(x), y).item())
    assert restored <= best_val + 1e-6
