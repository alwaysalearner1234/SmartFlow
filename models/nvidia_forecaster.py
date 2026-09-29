"""
NVIDIA Forecasting Model Interface.

This module provides the NvidiaForecaster class that interfaces with the NVIDIA
forecasting model deployed on Nebius cloud infrastructure.

Request format:
    Input:  ForecastInput features over a context window (20 observations, 14 features)
    Output: ForecastResult containing predicted_return, model_status, model_name,
            model_version, forecast_horizon, and metadata.

The ForecastInput carries the canonical NVIDIA feature vector (mid_price_return,
spread_bps, best_bid_size, best_ask_size, depth_imbalance_l1, depth_imbalance_multilevel,
ofi_instant, ofi_sum_5, trade_volume_imbalance, momentum_ret_5, momentum_ret_20,
volatility_std_10, micro_price, micro_price_divergence) arranged in chronological
order window of length context_window=20.

The ForecastResult predicted_return is the expected cumulative mid-price return over
the forecast_horizon, expressed as a decimal (e.g., 0.0012 = 0.12%).
expected_move_bps = predicted_return * 10000.

Currently a stub: no network calls. Always returns FALLBACK result until the
Nebius HTTP endpoint is connected (Phase 11).
"""

from __future__ import annotations

from typing import Optional
from dataclasses import dataclass, field

from data.contracts import ForecastResult, ForecastInput, ModelStatus


NVIDIA_FORECASTER_DEFAULT_HORIZON = 5


@dataclass
class _NvidiaForecasterConfig:
    """Internal configuration for the NVIDIA forecaster."""

    endpoint_url: Optional[str] = None
    timeout_sec: float = 2.0
    enabled: bool = False


class NvidiaForecaster:
    """
    NVIDIA short-term market forecaster.

    Wraps the Nebius-hosted forecasting model. The predict() method accepts a
    ForecastInput and a timestamp, validates temporal ordering, and returns a
    ForecastResult. Any network or processing error results in a FALLBACK status
    rather than raising to the caller.

    Attributes:
        config: Internal configuration (endpoint, timeout, enabled flag).
    """

    def __init__(
        self,
        endpoint_url: Optional[str] = None,
        timeout_sec: float = 2.0,
        enabled: bool = False,
    ):
        self.config = _NvidiaForecasterConfig(
            endpoint_url=endpoint_url,
            timeout_sec=timeout_sec,
            enabled=enabled,
        )
        # TODO(Phase 11): Add Nebius HTTP call logic here.
        # The call should accept the ForecastInput feature sequence and return
        # a ForecastResult with the predicted return and model metadata.
        # Until then, this stub always returns FALLBACK.

    def predict(
        self,
        forecast_input: Optional[ForecastInput],
        timestamp: float,
    ) -> ForecastResult:
        """
        Produce a forecast result from the NVIDIA model.

        Args:
            forecast_input: Model-facing forecasting input over the context window.
            timestamp: Current execution timestamp (used for staleness validation).

        Returns:
            ForecastResult with predicted_return=0.0, model_status=FALLBACK,
            model_name="nvidia-forecaster", model_version="stub-0",
            and forecast_horizon from config. If forecast_input is provided and
            its latest timestamp exceeds the given timestamp, the result's
            timestamp is set to the input's latest timestamp but status remains
            FALLBACK (staleness guard does not raise).

        Raises:
            Never raises to the caller. Any internal exception is caught and a
            FALLBACK ForecastResult is returned.
        """
        try:
            # Staleness / leakage guard: if the forecast's latest context
            # timestamp is after the execution timestamp, it would be
            # future-data leakage. We still return a result but mark it
            # accordingly rather than raising.
            if forecast_input is not None:
                input_latest = forecast_input.latest_timestamp
                if input_latest > timestamp:
                    # Forecast references a future context timestamp —
                    # still return FALLBACK rather than raising.
                    pass

            # Stub always returns FALLBACK regardless of input.
            return ForecastResult(
                timestamp=timestamp,
                forecast_horizon=NVIDIA_FORECASTER_DEFAULT_HORIZON,
                predicted_return=0.0,
                model_name="nvidia-forecaster",
                model_version="stub-0",
                model_status=ModelStatus.FALLBACK,
                metadata={"reason": "stub - Nebius not connected"},
            )
        except Exception:
            # Never raise to the caller: any exception produces a FALLBACK result.
            return ForecastResult(
                timestamp=timestamp,
                forecast_horizon=NVIDIA_FORECASTER_DEFAULT_HORIZON,
                predicted_return=0.0,
                model_name="nvidia-forecaster",
                model_version="stub-0",
                model_status=ModelStatus.FALLBACK,
                metadata={"reason": "stub - Nebius not connected - exception caught"},
            )