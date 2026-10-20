"""Runtime forecast buffer, engine plumbing, staleness, and end-to-end path."""

from __future__ import annotations

from data.contracts import (
    NVIDIA_FORECAST_FEATURES,
    ExecutionContext,
    ForecastInput,
    ForecastResult,
    MarketSnapshot,
    ModelStatus,
    OrderSide,
    PredictionResult,
)
from data.forecast_buffer import RuntimeForecastHistory
from execution.strategy_engine import DynamicStrategyEngine
from models.nvidia_forecaster import NvidiaForecaster
from simulator.execution_simulator import ExecutionSimulator
from simulator.fill_model import FillModel
from strategies.proposed import ProposedStrategy


def _snapshot(timestamp: float, sequence_id: int = 1) -> MarketSnapshot:
    return MarketSnapshot(
        timestamp=timestamp,
        sequence_id=sequence_id,
        bids=[(99.95, 200.0), (99.90, 300.0)],
        asks=[(100.05, 200.0), (100.10, 300.0)],
    )


def _features(index: float) -> dict:
    return {name: float(index) for name in NVIDIA_FORECAST_FEATURES}


def _prediction(probability: float = 0.2, timestamp: float = 100.0) -> PredictionResult:
    return PredictionResult(
        probability=probability,
        timestamp=timestamp,
        prediction_horizon=10,
        model_name="stub",
        model_status=ModelStatus.FALLBACK_HEURISTIC,
    )


def _forecast(
    predicted_return: float,
    timestamp: float,
    status=ModelStatus.TRAINED,
) -> ForecastResult:
    return ForecastResult(
        timestamp=timestamp,
        forecast_horizon=5,
        predicted_return=predicted_return,
        model_name="directional-gru",
        model_version="directional-gru-v1",
        model_status=status,
        feature_names=list(NVIDIA_FORECAST_FEATURES),
    )


def _context(forecast: ForecastResult, timestamp: float, max_staleness_sec=None) -> ExecutionContext:
    return ExecutionContext(
        timestamp=timestamp,
        remaining_quantity=500.0,
        initial_quantity=500.0,
        elapsed_time=10.0,
        total_horizon=60.0,
        urgency=0.5,
        ac_slice_quantity=80.0,
        ac_expected_cost=1.0,
        ac_schedule_index=0,
        adverse_risk=_prediction(0.2, timestamp),
        forecast=forecast,
        max_staleness_sec=max_staleness_sec,
    )


class StrictForecastInputForecaster:
    def __init__(self):
        self.calls = []

    def predict(self, forecast_input, timestamp: float):
        self.calls.append((forecast_input, timestamp))
        assert forecast_input is None or isinstance(forecast_input, ForecastInput)
        if forecast_input is None:
            return ForecastResult(
                timestamp=timestamp,
                forecast_horizon=5,
                predicted_return=0.0,
                model_name="strict",
                model_version="test",
                model_status=ModelStatus.FALLBACK,
                feature_names=list(NVIDIA_FORECAST_FEATURES),
                metadata={"reason": "insufficient_forecast_history"},
            )
        return ForecastResult(
            timestamp=timestamp,
            forecast_horizon=5,
            predicted_return=0.0001,
            model_name="strict",
            model_version="test",
            model_status=ModelStatus.TRAINED,
            feature_names=list(NVIDIA_FORECAST_FEATURES),
            metadata={"reason": "ok"},
        )


def test_runtime_buffer_warmup_window_and_reset():
    buffer = RuntimeForecastHistory()
    for i in range(19):
        buffer.append(100.0 + i, _features(i))
        assert buffer.to_forecast_input() is None
    buffer.append(119.0, _features(19))
    first = buffer.to_forecast_input()
    assert first is not None
    assert first.shape == (20, 14)
    assert first.feature_names == list(NVIDIA_FORECAST_FEATURES)
    assert first.timestamps[0] == 100.0
    buffer.append(120.0, _features(20))
    second = buffer.to_forecast_input()
    assert second.timestamps[0] == 101.0
    assert second.timestamps[-1] == 120.0
    buffer.reset()
    assert len(buffer) == 0
    assert buffer.to_forecast_input() is None


def test_engine_passes_forecast_input_not_prediction_result():
    strict = StrictForecastInputForecaster()
    engine = DynamicStrategyEngine(forecaster=strict)
    snap = _snapshot(100.0)
    features = _features(1)
    engine.evaluate_and_decide(
        strategy_name="SmartFlow (ML + NVIDIA + AC)",
        snapshot=snap,
        features_dict=features,
        remaining_quantity=500.0,
        initial_quantity=500.0,
        elapsed_time=1.0,
        total_horizon=60.0,
        side=OrderSide.BUY,
    )
    assert strict.calls
    assert not isinstance(strict.calls[0][0], PredictionResult)


def test_engine_builds_forecast_input_after_20_ticks():
    strict = StrictForecastInputForecaster()
    engine = DynamicStrategyEngine(forecaster=strict)
    for i in range(20):
        engine.evaluate_and_decide(
            strategy_name="SmartFlow (ML + NVIDIA + AC)",
            snapshot=_snapshot(100.0 + i, i),
            features_dict=_features(i),
            remaining_quantity=500.0,
            initial_quantity=500.0,
            elapsed_time=float(i),
            total_horizon=60.0,
            side=OrderSide.BUY,
        )
    last_input, _ = strict.calls[-1]
    assert isinstance(last_input, ForecastInput)
    assert last_input.shape == (20, 14)
    engine.reset()
    assert len(engine.forecast_history) == 0


def test_staleness_and_validity():
    fresh = _forecast(0.0005, timestamp=100.0)
    assert _context(fresh, 100.0, max_staleness_sec=5.0).has_valid_forecast is True
    future = _forecast(0.0005, timestamp=110.0)
    assert _context(future, 100.0, max_staleness_sec=5.0).has_valid_forecast is False
    old = _forecast(0.0005, timestamp=90.0)
    assert _context(old, 100.0, max_staleness_sec=5.0).has_valid_forecast is False
    fallback = _forecast(0.0005, timestamp=100.0, status=ModelStatus.FALLBACK)
    assert _context(fallback, 100.0, max_staleness_sec=5.0).has_valid_forecast is False


def test_end_to_end_local_forecast_path(tmp_path):
    from models.local_forecaster import LocalDirectionalForecaster
    from models.directional_model import DirectionalGRU, DIRECTIONAL_MODEL_VERSION
    from data.contracts import DEFAULT_FORECAST_CONTEXT_WINDOW, DEFAULT_FORECAST_HORIZON
    import numpy as np
    import torch

    model = DirectionalGRU(num_features=14, hidden_size=8, num_layers=1, dropout=0.0)
    artifact = tmp_path / "directional_forecaster.pt"
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
            "normalization": {
                "means": np.zeros(14).tolist(),
                "scales": np.ones(14).tolist(),
            },
            "model_version": DIRECTIONAL_MODEL_VERSION,
            "model_name": "directional-gru",
        },
        artifact,
    )

    class _LowRiskPredictor:
        def predict(self, features, side=OrderSide.BUY, timestamp=0.0):
            return _prediction(0.20, timestamp)

    class _NeverFillLimits(FillModel):
        def _simulate_limit_fill(self, order, snapshot, adverse_risk):
            return None

    forecaster = LocalDirectionalForecaster(artifact_path=artifact, device=torch.device("cpu"))
    engine = DynamicStrategyEngine(forecaster=forecaster, predictor=_LowRiskPredictor())
    sim = ExecutionSimulator(strategy_engine=engine, fill_model=_NeverFillLimits(execution_delay_ms=0.0))
    snapshots = [_snapshot(100.0 + i, i) for i in range(30)]
    result = sim.run_execution(
        strategy_name="SmartFlow (ML + NVIDIA + AC)",
        snapshots=snapshots,
        total_quantity=200.0,
        horizon_sec=40.0,
        side=OrderSide.BUY,
    )
    assert result.trajectory
    later = [pt for pt in result.trajectory if pt.get("has_valid_forecast")]
    assert later, "expected a valid forecast after 20 observations"
    sample = later[0]
    assert "forecast_predicted_return" in sample
    assert "forecast_direction" in sample
    assert "forecast_expected_move_bps" in sample
    assert "forecast_model_status" in sample
    assert "execution_mode" in sample
    assert sample["reason"]
