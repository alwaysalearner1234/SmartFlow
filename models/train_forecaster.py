"""Train and evaluate the directional GRU forecasting model.

The trainer consumes the existing chronological forecasting pipeline:

    snapshots -> feature history -> sequences -> targets
        -> ForecastingDataset -> chronological split
        -> fit scaler on TRAIN only -> transform val/test
        -> train DirectionalGRU with SmoothL1Loss + AdamW
        -> keep the best validation checkpoint

SmoothL1Loss (Huber) is used because the synthetic generator injects occasional
price jumps; squared error would overweight those shocks.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from config.config import DEFAULT_CONFIG, ForecastingConfig, SAVED_MODELS_DIR
from data.contracts import NVIDIA_FORECAST_FEATURES
from data.feature_history import FeatureHistoryConfig, build_feature_history
from data.forecasting_dataset import build_forecasting_dataset
from data.generator import MarketDataGenerator
from data.normalizer import ForecastingDatasetNormalizer, StandardScaler
from data.sequence_builder import build_sequences
from data.splitter import split_from_forecasting_config
from data.target_builder import build_targets_from_forecasting_config
from models.directional_model import (
    DIRECTIONAL_MODEL_VERSION,
    DirectionalGRU,
    PersistenceForecast,
    RidgeFlattenedForecast,
    ZeroForecast,
    require_torch,
    select_device,
)


NEUTRAL_SIGN_THRESHOLD = 1e-8
DEFAULT_TRAIN_SCENARIOS = (
    "normal_market",
    "high_volatility",
    "poor_liquidity",
    "high_adverse_selection",
    "low_adverse_selection",
    "short_execution_horizon",
)


def set_deterministic_seeds(seed: int) -> None:
    torch, _ = require_torch()
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def compute_regression_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    sign_threshold: float = NEUTRAL_SIGN_THRESHOLD,
) -> Dict[str, Any]:
    """Honest regression / direction metrics. Tiny values are treated as flat."""
    actual = np.asarray(y_true, dtype=np.float64).reshape(-1)
    predicted = np.asarray(y_pred, dtype=np.float64).reshape(-1)
    if actual.shape != predicted.shape:
        raise ValueError("y_true and y_pred must have the same shape.")
    if actual.size == 0:
        raise ValueError("Cannot score an empty prediction array.")

    residual = predicted - actual
    mae = float(np.mean(np.abs(residual)))
    mse = float(np.mean(residual ** 2))
    rmse = float(np.sqrt(mse))

    pred_sign = np.where(predicted > sign_threshold, 1, np.where(predicted < -sign_threshold, -1, 0))
    actual_sign = np.where(actual > sign_threshold, 1, np.where(actual < -sign_threshold, -1, 0))
    directional_mask = actual_sign != 0
    if int(np.sum(directional_mask)) == 0:
        directional_accuracy = None
    else:
        directional_accuracy = float(np.mean(pred_sign[directional_mask] == actual_sign[directional_mask]))

    if actual.size > 1 and float(np.std(predicted)) > 0 and float(np.std(actual)) > 0:
        correlation = float(np.corrcoef(predicted, actual)[0, 1])
    else:
        correlation = None

    return {
        "sample_count": int(actual.size),
        "mae": mae,
        "mse": mse,
        "rmse": rmse,
        "directional_accuracy": directional_accuracy,
        "correlation": correlation,
        "mean_predicted_return": float(np.mean(predicted)),
        "mean_actual_return": float(np.mean(actual)),
        "prediction_std": float(np.std(predicted)),
        "actual_return_std": float(np.std(actual)),
        "sign_threshold": sign_threshold,
    }


def _offset_history_timestamps(history, offset: float):
    frame = history.copy()
    frame["timestamp"] = frame["timestamp"].astype(float) + float(offset)
    return frame


def build_forecasting_dataset_from_scenarios(
    scenarios: Sequence[str] = DEFAULT_TRAIN_SCENARIOS,
    *,
    num_ticks: int = 400,
    dt: float = 0.5,
    seed: int = 42,
    forecasting_config: Optional[ForecastingConfig] = None,
):
    """Build one chronological ForecastingDataset from several synthetic regimes."""
    forecasting_config = forecasting_config or DEFAULT_CONFIG.forecasting
    generator = MarketDataGenerator(seed=seed)
    history_config = FeatureHistoryConfig(drop_invalid_rows=True, allow_missing_values=False)
    frames = []
    for index, scenario in enumerate(scenarios):
        snapshots = generator.generate_scenario_stream(
            scenario,
            num_ticks=num_ticks,
            dt=dt,
            start_time=1_700_000_000.0 + index * 100_000.0,
        )
        history = build_feature_history(snapshots, config=history_config)
        frames.append(_offset_history_timestamps(history, index * 1_000_000.0))

    import pandas as pd

    combined = pd.concat(frames, axis=0, ignore_index=True)
    combined = combined.sort_values("timestamp", kind="stable").reset_index(drop=True)
    sequences = build_sequences(combined)
    targets = build_targets_from_forecasting_config(combined, forecasting_config=forecasting_config)
    return build_forecasting_dataset(sequences, targets)


def _numpy_batches(sequences: np.ndarray, targets: np.ndarray, batch_size: int) -> Iterable[Tuple[np.ndarray, np.ndarray]]:
    for start in range(0, len(sequences), batch_size):
        yield sequences[start:start + batch_size], targets[start:start + batch_size]


def train_directional_forecaster(
    *,
    forecasting_config: Optional[ForecastingConfig] = None,
    scenarios: Sequence[str] = DEFAULT_TRAIN_SCENARIOS,
    num_ticks: int = 400,
    save: bool = True,
    device=None,
) -> Dict[str, Any]:
    """Train the GRU on a leakage-safe chronological split and persist the best model."""
    torch, nn = require_torch()
    config = forecasting_config or DEFAULT_CONFIG.forecasting
    set_deterministic_seeds(config.random_seed)
    resolved_device = device or select_device()

    dataset = build_forecasting_dataset_from_scenarios(
        scenarios,
        num_ticks=num_ticks,
        seed=config.random_seed,
        forecasting_config=config,
    )
    split = split_from_forecasting_config(dataset, config)
    normalizer = ForecastingDatasetNormalizer.from_forecasting_config(config)
    train_norm, val_norm, test_norm = normalizer.transform_splits(
        split.train,
        split.validation,
        split.test,
    )

    train_x = np.asarray(train_norm.sequences, dtype=np.float32)
    train_y = np.asarray(train_norm.targets, dtype=np.float32)
    val_x = np.asarray(val_norm.sequences, dtype=np.float32)
    val_y = np.asarray(val_norm.targets, dtype=np.float32)
    test_x = np.asarray(test_norm.sequences, dtype=np.float32)
    test_y = np.asarray(test_norm.targets, dtype=np.float32)

    model = DirectionalGRU(
        num_features=len(NVIDIA_FORECAST_FEATURES),
        hidden_size=config.hidden_size,
        num_layers=config.num_layers,
        dropout=config.dropout,
    ).to(resolved_device)

    # SmoothL1Loss is robust to occasional synthetic jumps.
    criterion = nn.SmoothL1Loss(beta=1.0)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )

    best_state = None
    best_val_loss = float("inf")
    epochs_without_improvement = 0
    history: List[Dict[str, float]] = []
    epochs_trained = 0

    for epoch in range(1, config.num_epochs + 1):
        model.train()
        epoch_losses = []
        for batch_x, batch_y in _numpy_batches(train_x, train_y, config.batch_size):
            tensor_x = torch.from_numpy(np.ascontiguousarray(batch_x)).to(resolved_device)
            tensor_y = torch.from_numpy(np.ascontiguousarray(batch_y)).to(resolved_device)
            optimizer.zero_grad()
            predicted = model(tensor_x)
            loss = criterion(predicted, tensor_y)
            loss.backward()
            optimizer.step()
            epoch_losses.append(float(loss.item()))

        val_metrics, val_loss = _eval_model(model, val_x, val_y, criterion, resolved_device)
        train_loss = float(np.mean(epoch_losses)) if epoch_losses else float("nan")
        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})
        epochs_trained = epoch

        if val_loss + 1e-12 < best_val_loss:
            best_val_loss = val_loss
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= config.patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    test_metrics, _test_loss = _eval_model(model, test_x, test_y, criterion, resolved_device)
    val_metrics, _ = _eval_model(model, val_x, val_y, criterion, resolved_device)
    train_metrics, _ = _eval_model(model, train_x, train_y, criterion, resolved_device)

    zero_metrics = compute_regression_metrics(test_y, ZeroForecast().predict(test_x))
    persistence_metrics = compute_regression_metrics(test_y, PersistenceForecast().predict(test_x))
    ridge = RidgeFlattenedForecast()
    ridge.fit(train_x, train_y)
    ridge_metrics = compute_regression_metrics(test_y, ridge.predict(test_x))

    scaler: Optional[StandardScaler] = normalizer.scaler
    architecture = {
        "num_features": len(NVIDIA_FORECAST_FEATURES),
        "hidden_size": config.hidden_size,
        "num_layers": config.num_layers,
        "dropout": config.dropout,
    }
    artifact = {
        "state_dict": {key: value.cpu() for key, value in model.state_dict().items()},
        "architecture": architecture,
        "feature_names": list(NVIDIA_FORECAST_FEATURES),
        "context_window": config.context_window,
        "forecast_horizon": config.forecast_horizon,
        "normalization": {
            "method": "standard",
            "means": scaler.means.tolist() if scaler is not None else None,
            "scales": scaler.scales.tolist() if scaler is not None else None,
        },
        "model_version": DIRECTIONAL_MODEL_VERSION,
        "model_name": "directional-gru",
        "training_metrics": train_metrics,
        "validation_metrics": val_metrics,
        "test_metrics": test_metrics,
        "baseline_metrics": {
            "zero": zero_metrics,
            "persistence": persistence_metrics,
            "ridge_flattened": ridge_metrics,
        },
        "training_history": history,
        "epochs_trained": epochs_trained,
        "best_validation_loss": best_val_loss,
        "device": str(resolved_device),
        "loss": "SmoothL1Loss",
        "optimizer": "AdamW",
        "parameter_count": model.count_parameters(),
    }

    saved_path = Path(config.saved_model_path)
    if save:
        saved_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(artifact, saved_path)
        metrics_path = SAVED_MODELS_DIR / "forecasting_metrics.json"
        serialisable = {
            key: value
            for key, value in artifact.items()
            if key != "state_dict"
        }
        metrics_path.write_text(json.dumps(serialisable, indent=2, default=_json_default), encoding="utf-8")

    artifact["saved_model_path"] = str(saved_path)
    artifact["model"] = model
    artifact["scaler"] = scaler
    return artifact


def _json_default(value: Any):
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Object of type {type(value)} is not JSON serializable")


def _eval_model(model, sequences: np.ndarray, targets: np.ndarray, criterion, device) -> Tuple[Dict[str, Any], float]:
    torch, _ = require_torch()
    model.eval()
    with torch.no_grad():
        tensor_x = torch.from_numpy(np.ascontiguousarray(sequences)).to(device)
        tensor_y = torch.from_numpy(np.ascontiguousarray(targets)).to(device)
        predicted = model(tensor_x)
        loss = float(criterion(predicted, tensor_y).item())
        pred_np = predicted.detach().cpu().numpy()
    metrics = compute_regression_metrics(targets, pred_np)
    metrics["smooth_l1_loss"] = loss
    return metrics, loss


def load_directional_artifact(path: Optional[Path] = None, device=None) -> Dict[str, Any]:
    torch, _ = require_torch()
    artifact_path = Path(path or DEFAULT_CONFIG.forecasting.saved_model_path)
    resolved_device = device or select_device()
    payload = torch.load(artifact_path, map_location=resolved_device, weights_only=False)
    architecture = payload.get("architecture", {})
    model = DirectionalGRU(
        num_features=architecture.get("num_features", len(NVIDIA_FORECAST_FEATURES)),
        hidden_size=architecture.get("hidden_size", 64),
        num_layers=architecture.get("num_layers", 1),
        dropout=architecture.get("dropout", 0.1),
    )
    model.load_state_dict(payload["state_dict"])
    model.to(resolved_device)
    model.eval()
    payload["model"] = model
    payload["device"] = resolved_device
    return payload


if __name__ == "__main__":
    result = train_directional_forecaster()
    print("saved", result["saved_model_path"])
    print("test_metrics", json.dumps(result["test_metrics"], indent=2, default=_json_default))
    print("baselines", json.dumps(result["baseline_metrics"], indent=2, default=_json_default))
