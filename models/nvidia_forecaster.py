"""NVIDIA/Nebius remote forecasting client for SmartFlow.

The client validates a 20 x 14 ``ForecastInput``, serializes it to JSON, and
posts it to a configured inference endpoint. Any transport or schema failure
returns a transparent FALLBACK ``ForecastResult``. Remote inference is never
required for local SmartFlow execution.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from math import isfinite
from numbers import Real
from typing import Any, Callable, Dict, Optional
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import Request, urlopen

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
NVIDIA_FORECASTER_MODEL_VERSION = "nvidia-remote-v1"

HttpTransport = Callable[[str, bytes, Dict[str, str], float], Dict[str, Any]]


def default_http_transport(
    url: str,
    payload: bytes,
    headers: Dict[str, str],
    timeout_sec: float,
) -> Dict[str, Any]:
    request = Request(url, data=payload, headers=headers, method="POST")
    with urlopen(request, timeout=timeout_sec) as response:
        raw = response.read()
        status = getattr(response, "status", 200)
        if status < 200 or status >= 300:
            raise RuntimeError(f"http_status_{status}")
    return json.loads(raw.decode("utf-8"))


@dataclass(frozen=True)
class _NvidiaForecasterConfig:
    """Validated configuration for the NVIDIA forecaster."""

    endpoint_url: Optional[str] = None
    timeout_sec: float = 2.0
    enabled: bool = False
    api_key: Optional[str] = None

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
        if self.api_key is not None and (not isinstance(self.api_key, str) or not self.api_key.strip()):
            raise ValueError("api_key must be None or a non-empty string.")


class NvidiaForecaster:
    """Short-horizon market forecaster with a fail-safe remote client."""

    def __init__(
        self,
        endpoint_url: Optional[str] = None,
        timeout_sec: float = 2.0,
        enabled: bool = False,
        api_key: Optional[str] = None,
        transport: Optional[HttpTransport] = None,
    ) -> None:
        self.config = _NvidiaForecasterConfig(
            endpoint_url=endpoint_url.strip() if isinstance(endpoint_url, str) else endpoint_url,
            timeout_sec=timeout_sec,
            enabled=enabled,
            api_key=api_key.strip() if isinstance(api_key, str) else api_key,
        )
        self._transport = transport or default_http_transport

    @staticmethod
    def _safe_result_timestamp(timestamp: Any) -> float:
        if isinstance(timestamp, bool) or not isinstance(timestamp, Real):
            return 0.0
        value = float(timestamp)
        return value if isfinite(value) else 0.0

    @staticmethod
    def _validate_input(forecast_input: ForecastInput) -> None:
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
        metadata: Dict[str, Any] = {
            "reason": reason,
            "remote_inference_connected": False,
            "configured_enabled": self.config.enabled,
            "endpoint_configured": self.config.endpoint_url is not None,
            "provider": "nvidia-remote",
        }
        return ForecastResult(
            timestamp=timestamp,
            forecast_horizon=NVIDIA_FORECASTER_DEFAULT_HORIZON,
            predicted_return=0.0,
            model_name=NVIDIA_FORECASTER_MODEL_NAME,
            model_version=NVIDIA_FORECASTER_MODEL_VERSION,
            model_status=ModelStatus.FALLBACK,
            confidence=None,
            feature_names=list(NVIDIA_FORECAST_FEATURES),
            metadata=metadata,
        )

    def _serialize_request(self, forecast_input: ForecastInput) -> Dict[str, Any]:
        return {
            "feature_sequence": forecast_input.feature_sequence,
            "feature_names": list(forecast_input.feature_names),
            "timestamps": list(forecast_input.timestamps),
            "context_window": forecast_input.context_window,
            "forecast_horizon": forecast_input.forecast_horizon,
        }

    def _parse_response(self, payload: Any, timestamp: float) -> ForecastResult:
        if not isinstance(payload, dict):
            raise ValueError("response is not a JSON object")
        if "predicted_return" not in payload:
            raise ValueError("missing predicted_return")
        predicted = payload["predicted_return"]
        if isinstance(predicted, bool) or not isinstance(predicted, Real):
            raise ValueError("predicted_return has the wrong data type")
        predicted_return = float(predicted)
        if not isfinite(predicted_return):
            raise ValueError("predicted_return is not finite")

        horizon = payload.get("forecast_horizon", NVIDIA_FORECASTER_DEFAULT_HORIZON)
        if int(horizon) != NVIDIA_FORECASTER_DEFAULT_HORIZON:
            raise ValueError("wrong horizon")

        confidence = payload.get("confidence")
        if confidence is not None:
            if isinstance(confidence, bool) or not isinstance(confidence, Real):
                raise ValueError("confidence has the wrong data type")
            confidence = float(confidence)
            if not isfinite(confidence):
                raise ValueError("confidence is not finite")

        model_name = str(payload.get("model_name") or "directional-gru")
        model_version = str(payload.get("model_version") or "v1")
        return ForecastResult(
            timestamp=timestamp,
            forecast_horizon=NVIDIA_FORECASTER_DEFAULT_HORIZON,
            predicted_return=predicted_return,
            model_name=model_name,
            model_version=model_version,
            model_status=ModelStatus.TRAINED,
            confidence=confidence,
            feature_names=list(NVIDIA_FORECAST_FEATURES),
            metadata={
                "reason": "remote_inference",
                "remote_inference_connected": True,
                "configured_enabled": True,
                "endpoint_configured": True,
                "provider": "nvidia-remote",
            },
        )

    def predict(
        self,
        forecast_input: Optional[ForecastInput],
        timestamp: float,
    ) -> ForecastResult:
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

            if not self.config.enabled:
                return self._fallback_result(safe_timestamp, "forecaster_disabled")
            if not self.config.endpoint_url:
                return self._fallback_result(safe_timestamp, "endpoint_not_configured")

            headers = {
                "Content-Type": "application/json",
                "Accept": "application/json",
            }
            if self.config.api_key:
                headers["Authorization"] = f"Bearer {self.config.api_key}"

            try:
                payload = json.dumps(self._serialize_request(forecast_input)).encode("utf-8")
                response = self._transport(
                    self.config.endpoint_url,
                    payload,
                    headers,
                    float(self.config.timeout_sec),
                )
            except TimeoutError:
                return self._fallback_result(safe_timestamp, "timeout")
            except HTTPError as exc:
                return self._fallback_result(safe_timestamp, f"http_error_{exc.code}")
            except URLError as exc:
                reason = getattr(exc, "reason", exc)
                if "timed out" in str(reason).lower():
                    return self._fallback_result(safe_timestamp, "timeout")
                return self._fallback_result(safe_timestamp, f"connection_failure: {reason}")
            except json.JSONDecodeError:
                return self._fallback_result(safe_timestamp, "invalid_json")
            except Exception as exc:
                message = str(exc).lower()
                if "timed out" in message or "timeout" in message:
                    return self._fallback_result(safe_timestamp, "timeout")
                if "http_status_" in message:
                    return self._fallback_result(safe_timestamp, message)
                return self._fallback_result(
                    safe_timestamp,
                    f"connection_failure: {type(exc).__name__}: {exc}",
                )

            try:
                return self._parse_response(response, safe_timestamp)
            except Exception as exc:
                return self._fallback_result(
                    safe_timestamp,
                    f"invalid_response_schema: {exc}",
                )
        except Exception as exc:
            try:
                return self._fallback_result(
                    safe_timestamp,
                    f"unexpected_error: {type(exc).__name__}: {exc}",
                )
            except Exception:
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
