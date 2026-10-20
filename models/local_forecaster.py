"""Local inference for the trained directional GRU artifact."""

from __future__ import annotations

from math import isfinite
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

from config.config import DEFAULT_CONFIG
from data.contracts import (
    DEFAULT_FORECAST_CONTEXT_WINDOW,
    DEFAULT_FORECAST_HORIZON,
    NVIDIA_FORECAST_FEATURES,
    ForecastInput,
    ForecastResult,
    ModelStatus,
    validate_nvidia_forecast_input_contract,
)
from data.normalizer import StandardScaler
from models.directional_model import (
    DIRECTIONAL_MODEL_VERSION,
    DirectionalGRU,
    require_torch,
    select_device,
)


class LocalDirectionalForecaster:
    """Run the saved DirectionalGRU locally with stored train-only normalization."""

    def __init__(
        self,
        artifact_path: Optional[Path] = None,
        device=None,
    ) -> None:
        self.artifact_path = Path(
            artifact_path or DEFAULT_CONFIG.forecasting.saved_model_path
        )
        self.device = device or select_device()
        self.model: Optional[DirectionalGRU] = None
        self.scaler: Optional[StandardScaler] = None
        self.model_version = DIRECTIONAL_MODEL_VERSION
        self.model_name = "directional-gru"
        self.feature_names = list(NVIDIA_FORECAST_FEATURES)
        self.context_window = DEFAULT_FORECAST_CONTEXT_WINDOW
        self.forecast_horizon = DEFAULT_FORECAST_HORIZON
        self._load_error: Optional[str] = None
        self._load_artifact()

    def _load_artifact(self) -> None:
        if not self.artifact_path.exists():
            self._load_error = "artifact_not_found"
            return
        try:
            torch, _ = require_torch()
            payload = torch.load(self.artifact_path, map_location=self.device, weights_only=False)
            architecture = payload.get("architecture", {})
            model = DirectionalGRU(
                num_features=int(architecture.get("num_features", len(NVIDIA_FORECAST_FEATURES))),
                hidden_size=int(architecture.get("hidden_size", 64)),
                num_layers=int(architecture.get("num_layers", 1)),
                dropout=float(architecture.get("dropout", 0.1)),
            )
            model.load_state_dict(payload["state_dict"])
            model.to(self.device)
            model.eval()
            self.model = model
            self.model_version = str(payload.get("model_version", DIRECTIONAL_MODEL_VERSION))
            self.model_name = str(payload.get("model_name", "directional-gru"))
            self.feature_names = list(payload.get("feature_names", NVIDIA_FORECAST_FEATURES))
            self.context_window = int(payload.get("context_window", DEFAULT_FORECAST_CONTEXT_WINDOW))
            self.forecast_horizon = int(payload.get("forecast_horizon", DEFAULT_FORECAST_HORIZON))
            normalization = payload.get("normalization") or {}
            means = normalization.get("means")
            scales = normalization.get("scales")
            if means is not None and scales is not None:
                self.scaler = StandardScaler(
                    means=np.asarray(means, dtype=np.float64),
                    scales=np.asarray(scales, dtype=np.float64),
                    feature_names=tuple(self.feature_names),
                )
            self._load_error = None
        except Exception as exc:
            self.model = None
            self.scaler = None
            self._load_error = f"artifact_load_failed: {type(exc).__name__}: {exc}"

    @property
    def is_ready(self) -> bool:
        return self.model is not None

    def _fallback(self, timestamp: float, reason: str) -> ForecastResult:
        return ForecastResult(
            timestamp=float(timestamp) if isfinite(float(timestamp or 0.0)) else 0.0,
            forecast_horizon=DEFAULT_FORECAST_HORIZON,
            predicted_return=0.0,
            model_name=self.model_name,
            model_version=self.model_version,
            model_status=ModelStatus.FALLBACK,
            confidence=None,
            feature_names=list(NVIDIA_FORECAST_FEATURES),
            metadata={
                "reason": reason,
                "provider": "local",
                "artifact_path": str(self.artifact_path),
            },
        )

    def _normalize(self, forecast_input: ForecastInput) -> np.ndarray:
        raw = np.asarray(forecast_input.feature_sequence, dtype=np.float64).reshape(
            1, self.context_window, len(self.feature_names)
        )
        if self.scaler is None:
            return raw
        return self.scaler.transform(raw)

    def predict(
        self,
        forecast_input: Optional[ForecastInput],
        timestamp: float,
    ) -> ForecastResult:
        safe_timestamp = 0.0
        try:
            safe_timestamp = float(timestamp)
            if not isfinite(safe_timestamp):
                return self._fallback(0.0, "invalid_execution_timestamp")
        except (TypeError, ValueError):
            return self._fallback(0.0, "invalid_execution_timestamp")

        if self.model is None:
            return self._fallback(safe_timestamp, self._load_error or "model_unavailable")

        if forecast_input is None:
            return self._fallback(safe_timestamp, "missing_forecast_input")

        try:
            if not isinstance(forecast_input, ForecastInput):
                return self._fallback(safe_timestamp, "invalid_forecast_input: TypeError")
            validate_nvidia_forecast_input_contract(forecast_input)
            if tuple(forecast_input.feature_names) != tuple(self.feature_names):
                return self._fallback(safe_timestamp, "invalid_forecast_input: feature order mismatch")
            if forecast_input.latest_timestamp > safe_timestamp:
                return self._fallback(
                    safe_timestamp,
                    "future_context_timestamp: input window ends after execution timestamp",
                )
            array = self._normalize(forecast_input)
            if not np.isfinite(array).all():
                return self._fallback(safe_timestamp, "non_finite_normalized_input")

            torch, _ = require_torch()
            tensor = torch.from_numpy(np.ascontiguousarray(array, dtype=np.float32)).to(self.device)
            self.model.eval()
            with torch.no_grad():
                predicted = self.model(tensor)
                value = float(predicted.detach().cpu().reshape(-1)[0].item())
            if not isfinite(value):
                return self._fallback(safe_timestamp, "non_finite_model_output")

            return ForecastResult(
                timestamp=safe_timestamp,
                forecast_horizon=self.forecast_horizon,
                predicted_return=value,
                model_name=self.model_name,
                model_version=self.model_version,
                model_status=ModelStatus.TRAINED,
                confidence=None,
                feature_names=list(self.feature_names),
                metadata={
                    "reason": "local_inference",
                    "provider": "local",
                    "device": str(self.device),
                    "normalized": self.scaler is not None,
                },
            )
        except Exception as exc:
            return self._fallback(
                safe_timestamp,
                f"unexpected_error: {type(exc).__name__}: {exc}",
            )


__all__ = ["LocalDirectionalForecaster"]
