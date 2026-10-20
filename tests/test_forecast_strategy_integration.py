"""BUY/SELL directional symmetry and ML+AC fallback preservation."""

from __future__ import annotations

from data.contracts import (
    ExecutionContext,
    ExecutionMode,
    ForecastResult,
    MarketSnapshot,
    ModelStatus,
    NVIDIA_FORECAST_FEATURES,
    OrderSide,
    PredictionResult,
)
from strategies.proposed import ProposedStrategy


def snapshot() -> MarketSnapshot:
    return MarketSnapshot(
        timestamp=100.0,
        sequence_id=1,
        bids=[(99.95, 200.0), (99.90, 300.0)],
        asks=[(100.05, 200.0), (100.10, 300.0)],
    )


def prediction(probability: float) -> PredictionResult:
    return PredictionResult(
        probability=probability,
        timestamp=100.0,
        prediction_horizon=10,
        model_name="stub",
        model_status=ModelStatus.FALLBACK_HEURISTIC,
    )


def forecast(predicted_return: float, status=ModelStatus.TRAINED) -> ForecastResult:
    return ForecastResult(
        timestamp=100.0,
        forecast_horizon=5,
        predicted_return=predicted_return,
        model_name="directional-gru",
        model_version="directional-gru-v1",
        model_status=status,
        feature_names=list(NVIDIA_FORECAST_FEATURES),
    )


def context(probability: float, predicted_return: float, status=ModelStatus.TRAINED, elapsed=10.0, remaining=500.0) -> ExecutionContext:
    rem_ratio = remaining / 500.0
    urgency = round(min(1.0, 0.5 * (elapsed / 60.0) + 0.5 * rem_ratio), 4)
    return ExecutionContext(
        timestamp=100.0,
        remaining_quantity=remaining,
        initial_quantity=500.0,
        elapsed_time=elapsed,
        total_horizon=60.0,
        urgency=urgency,
        ac_slice_quantity=80.0,
        ac_expected_cost=1.0,
        ac_schedule_index=0,
        adverse_risk=prediction(probability),
        forecast=forecast(predicted_return, status=status),
        max_staleness_sec=5.0,
    )


def decide(side, probability, predicted_return, status=ModelStatus.TRAINED, elapsed=10.0):
    strat = ProposedStrategy(forecast_min_move_bps=2.0, forecast_strong_move_bps=8.0)
    return strat.compute_decision(
        snapshot=snapshot(),
        remaining_quantity=500.0,
        elapsed_time=elapsed,
        total_horizon=60.0,
        side=side,
        initial_quantity=500.0,
        prediction_result=prediction(probability),
        execution_context=context(probability, predicted_return, status=status, elapsed=elapsed),
    )


def test_buy_high_toxicity_strong_up_is_aggressive():
    decision = decide(OrderSide.BUY, 0.85, 0.0012)  # +12 bps
    assert decision.execution_mode == ExecutionMode.AGGRESSIVE
    assert decision.quantity > 0
    assert "unfavorable" in decision.reason.lower() or "costly" in decision.reason.lower()


def test_buy_high_toxicity_strong_down_is_hold():
    decision = decide(OrderSide.BUY, 0.85, -0.0012)  # -12 bps
    assert decision.execution_mode == ExecutionMode.HOLD
    assert decision.quantity == 0.0
    assert "withdraw" in decision.reason.lower() or "unsafe" in decision.reason.lower() or "toxicity" in decision.reason.lower()


def test_buy_low_toxicity_neutral_is_passive():
    decision = decide(OrderSide.BUY, 0.20, 0.00005)  # 0.5 bps < min threshold
    assert decision.execution_mode == ExecutionMode.PASSIVE
    assert decision.limit_price == snapshot().best_bid


def test_buy_high_urgency_down_forecast_still_aggressive():
    decision = decide(OrderSide.BUY, 0.85, -0.0012, elapsed=50.0)
    assert decision.execution_mode == ExecutionMode.AGGRESSIVE
    assert "completion" in decision.reason.lower() or "deadline" in decision.reason.lower()


def test_sell_high_toxicity_strong_down_is_aggressive():
    decision = decide(OrderSide.SELL, 0.85, -0.0012)
    assert decision.execution_mode == ExecutionMode.AGGRESSIVE
    assert decision.quantity > 0


def test_sell_high_toxicity_strong_up_is_hold():
    decision = decide(OrderSide.SELL, 0.85, 0.0012)
    assert decision.execution_mode == ExecutionMode.HOLD
    assert decision.quantity == 0.0


def test_sell_low_toxicity_neutral_is_passive():
    decision = decide(OrderSide.SELL, 0.20, 0.00005)
    assert decision.execution_mode == ExecutionMode.PASSIVE
    assert decision.limit_price == snapshot().best_ask


def test_sell_high_urgency_up_forecast_still_aggressive():
    decision = decide(OrderSide.SELL, 0.85, 0.0012, elapsed=50.0)
    assert decision.execution_mode == ExecutionMode.AGGRESSIVE
    assert "completion" in decision.reason.lower() or "deadline" in decision.reason.lower()


def test_fallback_preserves_ml_ac_matrix():
    high_risk = decide(OrderSide.BUY, 0.85, 0.0012, status=ModelStatus.FALLBACK)
    assert high_risk.execution_mode == ExecutionMode.HOLD
    assert high_risk.quantity == 0.0

    low_risk = decide(OrderSide.BUY, 0.20, 0.0012, status=ModelStatus.FALLBACK)
    assert low_risk.execution_mode == ExecutionMode.PASSIVE

    moderate = decide(OrderSide.BUY, 0.50, 0.0012, status=ModelStatus.FALLBACK)
    assert moderate.execution_mode == ExecutionMode.AGGRESSIVE
    assert "almgren" in moderate.reason.lower()
