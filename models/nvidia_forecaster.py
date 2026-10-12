"""NVIDIA forecasting model interface for SmartFlow.

This module defines the model-facing interface for a Nebius-hosted NVIDIA
forecasting model. The remote request/response schema is not yet specified in
this repository, so this implementation deliberately does not make HTTP calls.
Until that schema and endpoint are wired in, ``predict`` returns a safe
FALLBACK result and records why inference was not performed.

Forecast contract:
    context window: 20 observations
    feature count: 14, in NVIDIA_FORECAST_FEATURES order
    forecast horizon: 5 observations
    predicted_return: cumulative mid-price return as a decimal
    expected_move_bps: predicted_return * 10_000
"""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from numbers import Real
from typing import Any, Dict, Optional
from urllib.parse import urlparse

from data.contracts import (
    DEFAULT_FORECAST_CONTEXT_WINDOW,
    DEFAULT_FORECAST_HORIZON,
    NVIDIA_FORECAST_FEATURES,
    ForecastInput,
    ForecastResult,
    ModelStatus,
    validate_nvidia_feature_contract,
)


NVIDIA_FORECASTER_DEFAULT_HORIZON = DEFAULT_FORECAST_HORIZON
NVIDIA_FORECASTER_DEFAULT_CONTEXT_WINDOW = DEFAULT_FORECAST_CONTEXT_WINDOW
NVIDIA_FORECASTER_MODEL_NAME = "nvidia-forecaster"
NVIDIA_FORECASTER_MODEL_VERSION = "stub-1"


@dataclass(frozen=True)
class _NvidiaForecasterConfig:
    """Validated configuration for the NVIDIA forecaster."""

    endpoint_url: Optional[str] = None
    timeout_sec: float = 2.0
    enabled: bool = False

    def __post_init__(self) -> None:
        if self.endpoint_url is not None:
            if not isinstance(self.endpoint_url, str) or not self.endpoint_url.strip():
                raise ValueError("endpoint_url must be None or a non-empty URL string.")
            parsed = urlparse(self.endpoint_url.strip())
            if parsed.scheme not in {"http", "https"} or not parsed.netloc:
                raise ValueError("endpoint_url must be an absolute HTTP or HTTPS URL.")

        if isinstance(self.timeout_sec, bool) or not isinstance(self.timeout_sec, Real):
            raise TypeError("timeout_sec must be a finite positive number.")
        if not isfinite(float(self.timeout_sec)) or float(self.timeout_sec) <= 0:
            raise ValueError("timeout_sec must be a finite positive number.")

        if not isinstance(self.enabled, bool):
            raise TypeError("enabled must be a bool.")


class NvidiaForecaster:
    """Short-horizon market forecaster with a fail-safe fallback interface.

    ``predict`` intentionally never propagates input/processing errors. A
    malformed input, a future-dated context window, or an unconnected remote
    endpoint produces a ``ForecastResult`` with ``ModelStatus.FALLBACK``.

    Important: setting ``enabled=True`` does not activate remote inference yet.
    An HTTP call is intentionally omitted until the deployed Nebius endpoint's
    request and response schema is known. This prevents sending incorrectly
    shaped requests or interpreting an unknown response as a valid forecast.
    """

    def __init__(
        self,
        endpoint_url: Optional[str] = None,
        timeout_sec: float = 2.0,
        enabled: bool = False,
    ) -> None:
        self.config = _NvidiaForecasterConfig(
            endpoint_url=endpoint_url,
            timeout_sec=timeout_sec,
            enabled=enabled,
        )

    @staticmethod
    def _safe_result_timestamp(timestamp: Any) -> float:
        """Return a finite numeric timestamp suitable for ForecastResult."""
        if isinstance(timestamp, bool) or not isinstance(timestamp, Real):
            return 0.0
        value = float(timestamp)
        return value if isfinite(value) else 0.0

    @staticmethod
    def _validate_input(forecast_input: ForecastInput) -> None:
        """Validate the strict NVIDIA input contract without mutating input."""
        if not isinstance(forecast_input, ForecastInput):
            raise TypeError("forecast_input must be a ForecastInput instance.")

        if forecast_input.context_window != NVIDIA_FORECASTER_DEFAULT_CONTEXT_WINDOW:
            raise ValueError(
                "NVIDIA context window must be "
                f"{NVIDIA_FORECASTER_DEFAULT_CONTEXT_WINDOW}; received "
                f"{forecast_input.context_window}."
            )
        if forecast_input.forecast_horizon != NVIDIA_FORECASTER_DEFAULT_HORIZON:
            raise ValueError(
                "NVIDIA forecast horizon must be "
                f"{NVIDIA_FORECASTER_DEFAULT_HORIZON}; received "
                f"{forecast_input.forecast_horizon}."
            )

        validate_nvidia_feature_contract(forecast_input.feature_names)

        expected_shape = (
            NVIDIA_FORECASTER_DEFAULT_CONTEXT_WINDOW,
            len(NVIDIA_FORECAST_FEATURES),
        )
        actual_shape = (
            len(forecast_input.feature_sequence),
            len(forecast_input.feature_sequence[0])
            if forecast_input.feature_sequence
            else 0,
        )
        if actual_shape != expected_shape:
            raise ValueError(
                f"NVIDIA feature_sequence must have shape {expected_shape}; "
                f"received {actual_shape}."
            )

        # The ForecastInput contract validates chronology and finite numeric
        # values on construction. Recheck the dimensions here because lists can
        # be mutated after the dataclass was constructed.
        if len(forecast_input.timestamps) != NVIDIA_FORECASTER_DEFAULT_CONTEXT_WINDOW:
            raise ValueError(
                "NVIDIA timestamps must contain exactly "
                f"{NVIDIA_FORECASTER_DEFAULT_CONTEXT_WINDOW} values."
            )
        if any(
            isinstance(value, bool)
            or not isinstance(value, Real)
            or not isfinite(float(value))
            for row in forecast_input.feature_sequence
            for value in row
        ):
            raise ValueError("feature_sequence must contain only finite numeric values.")
        if any(
            isinstance(value, bool)
            or not isinstance(value, Real)
            or not isfinite(float(value))
            for value in forecast_input.timestamps
        ):
            raise ValueError("timestamps must contain only finite numeric values.")
        if any(
            float(forecast_input.timestamps[index])
            >= float(forecast_input.timestamps[index + 1])
            for index in range(len(forecast_input.timestamps) - 1)
        ):
            raise ValueError("timestamps must be strictly increasing.")

    def _fallback_result(self, timestamp: float, reason: str) -> ForecastResult:
        """Build a structurally valid fallback result with diagnostic metadata."""
        metadata: Dict[str, Any] = {
            "reason": reason,
            "remote_inference_connected": False,
            "configured_enabled": self.config.enabled,
            "endpoint_configured": self.config.endpoint_url is not None,
        }
        if self.config.enabled and self.config.endpoint_url:
            metadata["remote_inference_note"] = (
                "Remote inference is not implemented because the Nebius "
                "request/response schema has not been supplied."
            )
        return ForecastResult(
            timestamp=timestamp,
            forecast_horizon=NVIDIA_FORECASTER_DEFAULT_HORIZON,
            predicted_return=0.0,
            model_name=NVIDIA_FORECASTER_MODEL_NAME,
            model_version=NVIDIA_FORECASTER_MODEL_VERSION,
            model_status=ModelStatus.FALLBACK,
            feature_names=list(NVIDIA_FORECAST_FEATURES),
            metadata=metadata,
        )

    def predict(
        self,
        forecast_input: Optional[ForecastInput],
        timestamp: float,
    ) -> ForecastResult:
        """Return a forecast or a fail-safe FALLBACK result.

        Args:
            forecast_input: Chronological model input with shape ``(20, 14)``.
            timestamp: Current execution timestamp. Must not precede the latest
                timestamp in ``forecast_input``.

        Returns:
            A ``ForecastResult``. Until remote inference is implemented, every
            valid call returns FALLBACK with ``predicted_return=0.0``. Errors
            are converted to FALLBACK results rather than propagated.
        """
        safe_timestamp = self._safe_result_timestamp(timestamp)

        try:
            if isinstance(timestamp, bool) or not isinstance(timestamp, Real):
                return self._fallback_result(safe_timestamp, "invalid_execution_timestamp")
            if not isfinite(float(timestamp)):
                return self._fallback_result(safe_timestamp, "invalid_execution_timestamp")

            if forecast_input is None:
                return self._fallback_result(safe_timestamp, "missing_forecast_input")

            try:
                self._validate_input(forecast_input)
            except (TypeError, ValueError, IndexError, AttributeError) as exc:
                return self._fallback_result(
                    safe_timestamp,
                    f"invalid_forecast_input: {type(exc).__name__}: {exc}",
                )

            input_latest = float(forecast_input.latest_timestamp)
            if input_latest > float(timestamp):
                return self._fallback_result(
                    safe_timestamp,
                    "future_context_timestamp: input window ends after execution timestamp",
                )

            # Deliberately do not make a network call until the actual Nebius
            # request/response contract is available and can be tested.
            if not self.config.enabled:
                return self._fallback_result(safe_timestamp, "forecaster_disabled")
            if not self.config.endpoint_url:
                return self._fallback_result(
                    safe_timestamp,
                    "endpoint_not_configured",
                )
            return self._fallback_result(
                safe_timestamp,
                "remote_inference_not_implemented",
            )
        except Exception as exc:
            # Preserve the interface's fail-safe behavior for unexpected errors.
            try:
                return self._fallback_result(
                    safe_timestamp,
                    f"unexpected_error: {type(exc).__name__}: {exc}",
                )
            except Exception:
                # The contract's normal constructor should not fail for the
                # sanitized timestamp and fixed, canonical fields. This last
                # guard prevents internal errors from escaping predict().
                return ForecastResult(
                    timestamp=0.0,
                    forecast_horizon=NVIDIA_FORECASTER_DEFAULT_HORIZON,
                    predicted_return=0.0,
                    model_name=NVIDIA_FORECASTER_MODEL_NAME,
                    model_version=NVIDIA_FORECASTER_MODEL_VERSION,
                    model_status=ModelStatus.FALLBACK,
                    feature_names=list(NVIDIA_FORECAST_FEATURES),
                    metadata={"reason": "fallback_result_construction_error"},
                )


__all__ = [
    "NVIDIA_FORECASTER_DEFAULT_HORIZON",
    "NVIDIA_FORECASTER_DEFAULT_CONTEXT_WINDOW",
    "NVIDIA_FORECASTER_MODEL_NAME",
    "NVIDIA_FORECASTER_MODEL_VERSION",
    "NvidiaForecaster",
]

