"""Unit tests for the NVIDIA forecasting interface.

These tests intentionally exercise the fail-safe local interface only. They do
not call Nebius or any remote service: remote inference is not implemented until
its request/response schema is defined.
"""

from __future__ import annotations

from math import inf, nan

import pytest

from data.contracts import (
    DEFAULT_FORECAST_CONTEXT_WINDOW,
    DEFAULT_FORECAST_HORIZON,
    NVIDIA_FORECAST_FEATURES,
    ForecastInput,
    ForecastResult,
    ModelStatus,
)
from models.nvidia_forecaster import (
    NVIDIA_FORECASTER_DEFAULT_CONTEXT_WINDOW,
    NVIDIA_FORECASTER_DEFAULT_HORIZON,
    NVIDIA_FORECASTER_MODEL_NAME,
    NVIDIA_FORECASTER_MODEL_VERSION,
    NvidiaForecaster,
)


CONTEXT_WINDOW = DEFAULT_FORECAST_CONTEXT_WINDOW
FORECAST_HORIZON = DEFAULT_FORECAST_HORIZON
NUM_FEATURES = len(NVIDIA_FORECAST_FEATURES)


def make_forecast_input(
    *,
    context_window: int = CONTEXT_WINDOW,
    forecast_horizon: int = FORECAST_HORIZON,
) -> ForecastInput:
    """Build a valid chronological input for the canonical NVIDIA contract."""
    return ForecastInput(
        feature_sequence=[
            [float(row * NUM_FEATURES + column + 1) for column in range(NUM_FEATURES)]
            for row in range(context_window)
        ],
        feature_names=list(NVIDIA_FORECAST_FEATURES),
        timestamps=[float(1_000 + index) for index in range(context_window)],
        context_window=context_window,
        forecast_horizon=forecast_horizon,
    )


def assert_safe_fallback(
    result: ForecastResult,
    *,
    timestamp: float,
    reason: str,
    enabled: bool = False,
    endpoint_configured: bool = False,
) -> None:
    """Assert the stable, public fallback-result contract."""
    assert isinstance(result, ForecastResult)
    assert result.timestamp == timestamp
    assert result.forecast_horizon == FORECAST_HORIZON
    assert result.predicted_return == 0.0
    assert result.model_name == NVIDIA_FORECASTER_MODEL_NAME
    assert result.model_version == NVIDIA_FORECASTER_MODEL_VERSION
    assert result.model_status == ModelStatus.FALLBACK
    assert result.feature_names == list(NVIDIA_FORECAST_FEATURES)
    assert result.metadata["reason"].startswith(reason)
    assert result.metadata["remote_inference_connected"] is False
    assert result.metadata["configured_enabled"] is enabled
    assert result.metadata["endpoint_configured"] is endpoint_configured


class TestNvidiaForecasterConfiguration:
    def test_default_configuration_is_disabled_and_unconfigured(self) -> None:
        forecaster = NvidiaForecaster()

        assert forecaster.config.endpoint_url is None
        assert forecaster.config.timeout_sec == 2.0
        assert forecaster.config.enabled is False
        assert NVIDIA_FORECASTER_DEFAULT_CONTEXT_WINDOW == CONTEXT_WINDOW
        assert NVIDIA_FORECASTER_DEFAULT_HORIZON == FORECAST_HORIZON

    @pytest.mark.parametrize(
        "endpoint_url",
        ["", "   ", "nebius.example.com", "ftp://example.com/model", "https:///missing-host"],
    )
    def test_rejects_invalid_endpoint_urls(self, endpoint_url: str) -> None:
        with pytest.raises(ValueError):
            NvidiaForecaster(endpoint_url=endpoint_url)

    @pytest.mark.parametrize(
        "endpoint_url",
        ["http://localhost:8080/predict", "https://example.com/v1/predict"],
    )
    def test_accepts_absolute_http_and_https_urls(self, endpoint_url: str) -> None:
        forecaster = NvidiaForecaster(endpoint_url=endpoint_url)
        assert forecaster.config.endpoint_url == endpoint_url

    def test_none_endpoint_url_is_allowed(self) -> None:
        assert NvidiaForecaster(endpoint_url=None).config.endpoint_url is None

    @pytest.mark.parametrize("timeout_sec", [0, -1, inf, -inf, nan])
    def test_rejects_non_positive_or_non_finite_timeout(self, timeout_sec: float) -> None:
        with pytest.raises(ValueError):
            NvidiaForecaster(timeout_sec=timeout_sec)

    @pytest.mark.parametrize("timeout_sec", [True, False, "2", None])
    def test_rejects_timeout_with_wrong_type(self, timeout_sec: object) -> None:
        with pytest.raises(TypeError):
            NvidiaForecaster(timeout_sec=timeout_sec)  # type: ignore[arg-type]

    @pytest.mark.parametrize("enabled", [1, 0, "true", None])
    def test_enabled_must_be_a_boolean(self, enabled: object) -> None:
        with pytest.raises(TypeError):
            NvidiaForecaster(enabled=enabled)  # type: ignore[arg-type]


class TestNvidiaForecasterPredict:
    def test_missing_input_returns_fallback(self) -> None:
        result = NvidiaForecaster().predict(None, timestamp=2_000.0)

        assert_safe_fallback(
            result,
            timestamp=2_000.0,
            reason="missing_forecast_input",
        )

    def test_valid_input_returns_fallback_when_disabled(self) -> None:
        result = NvidiaForecaster().predict(make_forecast_input(), timestamp=2_000.0)

        assert_safe_fallback(
            result,
            timestamp=2_000.0,
            reason="forecaster_disabled",
        )

    def test_enabled_without_endpoint_returns_fallback(self) -> None:
        result = NvidiaForecaster(enabled=True).predict(
            make_forecast_input(), timestamp=2_000.0
        )

        assert_safe_fallback(
            result,
            timestamp=2_000.0,
            reason="endpoint_not_configured",
            enabled=True,
        )

    def test_enabled_with_endpoint_uses_transport_for_real_inference(self) -> None:
        def fake_transport(url, payload, headers, timeout_sec):
            assert url == "https://example.com/predict"
            return {
                "predicted_return": 0.00042,
                "model_name": "directional-gru",
                "model_version": "v1",
                "confidence": None,
                "forecast_horizon": FORECAST_HORIZON,
            }

        result = NvidiaForecaster(
            endpoint_url="https://example.com/predict",
            enabled=True,
            transport=fake_transport,
        ).predict(make_forecast_input(), timestamp=2_000.0)

        assert result.model_status == ModelStatus.TRAINED
        assert result.predicted_return == 0.00042
        assert result.metadata["remote_inference_connected"] is True

    def test_context_ending_after_execution_time_is_rejected(self) -> None:
        result = NvidiaForecaster().predict(make_forecast_input(), timestamp=1_018.0)

        assert_safe_fallback(
            result,
            timestamp=1_018.0,
            reason="future_context_timestamp",
        )

    @pytest.mark.parametrize("timestamp", [True, "2000", None])
    def test_invalid_execution_timestamp_returns_fallback(self, timestamp: object) -> None:
        result = NvidiaForecaster().predict(  # type: ignore[arg-type]
            make_forecast_input(), timestamp=timestamp  # type: ignore[arg-type]
        )

        assert_safe_fallback(
            result,
            timestamp=0.0,
            reason="invalid_execution_timestamp",
        )

    @pytest.mark.parametrize("timestamp", [inf, -inf, nan])
    def test_non_finite_execution_timestamp_returns_fallback(self, timestamp: float) -> None:
        result = NvidiaForecaster().predict(make_forecast_input(), timestamp=timestamp)

        assert_safe_fallback(
            result,
            timestamp=0.0,
            reason="invalid_execution_timestamp",
        )

    def test_non_forecast_input_object_returns_fallback(self) -> None:
        result = NvidiaForecaster().predict(object(), timestamp=2_000.0)  # type: ignore[arg-type]

        assert_safe_fallback(
            result,
            timestamp=2_000.0,
            reason="invalid_forecast_input",
        )

    def test_wrong_context_window_returns_fallback(self) -> None:
        invalid_input = make_forecast_input(context_window=19)
        result = NvidiaForecaster().predict(invalid_input, timestamp=2_000.0)

        assert_safe_fallback(
            result,
            timestamp=2_000.0,
            reason="invalid_forecast_input",
        )

    def test_wrong_forecast_horizon_returns_fallback(self) -> None:
        invalid_input = make_forecast_input(forecast_horizon=10)
        result = NvidiaForecaster().predict(invalid_input, timestamp=2_000.0)

        assert_safe_fallback(
            result,
            timestamp=2_000.0,
            reason="invalid_forecast_input",
        )

    def test_mutated_feature_row_shape_returns_fallback(self) -> None:
        forecast_input = make_forecast_input()
        forecast_input.feature_sequence[0].pop()

        result = NvidiaForecaster().predict(forecast_input, timestamp=2_000.0)

        assert_safe_fallback(
            result,
            timestamp=2_000.0,
            reason="invalid_forecast_input",
        )

    def test_mutated_feature_values_with_nan_return_fallback(self) -> None:
        forecast_input = make_forecast_input()
        forecast_input.feature_sequence[0][0] = nan

        result = NvidiaForecaster().predict(forecast_input, timestamp=2_000.0)

        assert_safe_fallback(
            result,
            timestamp=2_000.0,
            reason="invalid_forecast_input",
        )

    def test_mutated_timestamps_with_non_increasing_order_return_fallback(self) -> None:
        forecast_input = make_forecast_input()
        forecast_input.timestamps[-1] = forecast_input.timestamps[-2]

        result = NvidiaForecaster().predict(forecast_input, timestamp=2_000.0)

        assert_safe_fallback(
            result,
            timestamp=2_000.0,
            reason="invalid_forecast_input",
        )

    def test_execution_timestamp_equal_to_latest_context_timestamp_is_allowed(self) -> None:
        forecast_input = make_forecast_input()
        result = NvidiaForecaster().predict(
            forecast_input,
            timestamp=forecast_input.latest_timestamp,
        )

        assert_safe_fallback(
            result,
            timestamp=forecast_input.latest_timestamp,
            reason="forecaster_disabled",
        )

    def test_execution_timestamp_after_context_is_allowed(self) -> None:
        forecast_input = make_forecast_input()
        result = NvidiaForecaster().predict(
            forecast_input,
            timestamp=forecast_input.latest_timestamp + 1.0,
        )

        assert_safe_fallback(
            result,
            timestamp=forecast_input.latest_timestamp + 1.0,
            reason="forecaster_disabled",
        )

    def test_prediction_does_not_mutate_valid_input(self) -> None:
        forecast_input = make_forecast_input()
        original_features = [row.copy() for row in forecast_input.feature_sequence]
        original_names = forecast_input.feature_names.copy()
        original_timestamps = forecast_input.timestamps.copy()

        NvidiaForecaster().predict(forecast_input, timestamp=2_000.0)

        assert forecast_input.feature_sequence == original_features
        assert forecast_input.feature_names == original_names
        assert forecast_input.timestamps == original_timestamps

    def test_result_metadata_identifies_fallback_and_connection_state(self) -> None:
        result = NvidiaForecaster().predict(make_forecast_input(), timestamp=2_000.0)

        assert result.model_status is ModelStatus.FALLBACK
        assert result.metadata["remote_inference_connected"] is False
        assert result.metadata["configured_enabled"] is False
        assert result.metadata["endpoint_configured"] is False
        assert result.metadata["reason"] == "forecaster_disabled"


class TestNvidiaForecasterRemoteFailures:
    def _forecaster(self, transport):
        return NvidiaForecaster(
            endpoint_url="https://example.com/predict",
            enabled=True,
            transport=transport,
        )

    def test_timeout_returns_fallback(self) -> None:
        def transport(*args, **kwargs):
            raise TimeoutError("timed out")

        result = self._forecaster(transport).predict(make_forecast_input(), timestamp=2_000.0)
        assert_safe_fallback(result, timestamp=2_000.0, reason="timeout", enabled=True, endpoint_configured=True)

    def test_connection_exception_returns_fallback(self) -> None:
        def transport(*args, **kwargs):
            raise ConnectionError("connection refused")

        result = self._forecaster(transport).predict(make_forecast_input(), timestamp=2_000.0)
        assert result.model_status == ModelStatus.FALLBACK
        assert result.predicted_return == 0.0
        assert "connection_failure" in result.metadata["reason"]

    def test_http_500_returns_fallback(self) -> None:
        def transport(*args, **kwargs):
            raise RuntimeError("http_status_500")

        result = self._forecaster(transport).predict(make_forecast_input(), timestamp=2_000.0)
        assert result.model_status == ModelStatus.FALLBACK
        assert "http_status_500" in result.metadata["reason"]

    def test_malformed_json_object_returns_fallback(self) -> None:
        result = self._forecaster(lambda *a, **k: ["not-an-object"]).predict(
            make_forecast_input(), timestamp=2_000.0
        )
        assert result.model_status == ModelStatus.FALLBACK
        assert "invalid_response_schema" in result.metadata["reason"]

    def test_missing_predicted_return_returns_fallback(self) -> None:
        result = self._forecaster(lambda *a, **k: {"model_name": "x"}).predict(
            make_forecast_input(), timestamp=2_000.0
        )
        assert "missing predicted_return" in result.metadata["reason"]

    def test_nan_predicted_return_returns_fallback(self) -> None:
        result = self._forecaster(lambda *a, **k: {"predicted_return": nan}).predict(
            make_forecast_input(), timestamp=2_000.0
        )
        assert result.model_status == ModelStatus.FALLBACK

    def test_wrong_predicted_return_type_returns_fallback(self) -> None:
        result = self._forecaster(lambda *a, **k: {"predicted_return": "0.1"}).predict(
            make_forecast_input(), timestamp=2_000.0
        )
        assert result.model_status == ModelStatus.FALLBACK

    def test_wrong_horizon_returns_fallback(self) -> None:
        result = self._forecaster(
            lambda *a, **k: {"predicted_return": 0.1, "forecast_horizon": 9}
        ).predict(make_forecast_input(), timestamp=2_000.0)
        assert "wrong horizon" in result.metadata["reason"]


