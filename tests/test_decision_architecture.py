"""
Unit tests for the separated execution-decision architecture.

Adverse-selection risk controls passive-exposure safety only.
Completion urgency / Almgren-Chriss control whether and how much to execute.
These tests use deterministic PredictionResult objects rather than the on-disk model.
"""

from data.contracts import (
    ExecutionDecision,
    ExecutionMode,
    MarketSnapshot,
    ModelStatus,
    OrderSide,
    OrderStatus,
    OrderType,
    PredictionResult,
)
from execution.order_manager import OrderManager
from execution.strategy_engine import DynamicStrategyEngine
from simulator.execution_simulator import ExecutionSimulator
from simulator.fill_model import FillModel
from strategies.aggressive import AggressiveStrategy
from strategies.almgren_chriss_strat import AlmgrenChrissStrategy
from strategies.market import MarketStrategy
from strategies.passive import PassiveStrategy
from strategies.proposed import ProposedStrategy
from strategies.twap import TWAPStrategy
from strategies.vwap import VWAPStrategy


DIRECTIONAL_FORECAST_PHRASES = (
    "before price moves",
    "before the market falls",
    "before the market rises",
    "before the adverse move",
    "before adverse move",
    "price is about to drop",
    "price is about to rally",
    "taking liquidity before",
    "taking available",
    "price plunges",
    "price rallies",
    "risk defense",
    "unfavorable depth",
    "market is going down",
    "market is going up",
    "price deteriorates",
    "price will fall",
    "price will rise",
)


class StubPredictor:
    """Deterministic adverse-selection predictor for isolated decision tests."""

    def __init__(self, probability: float = 0.5):
        self.probability = probability

    def predict(self, features, side=OrderSide.BUY, timestamp: float = 0.0) -> PredictionResult:
        return make_prediction(self.probability, timestamp)


class TimestampPredictor:
    """Returns a toxicity probability based on elapsed time from a start timestamp."""

    def __init__(self, start_time: float, schedule):
        self.start_time = start_time
        self.schedule = schedule  # list of (max_elapsed, probability)

    def predict(self, features, side=OrderSide.BUY, timestamp: float = 0.0) -> PredictionResult:
        elapsed = timestamp - self.start_time
        probability = self.schedule[-1][1]
        for max_elapsed, prob in self.schedule:
            if elapsed <= max_elapsed:
                probability = prob
                break
        return make_prediction(probability, timestamp)


class NeverFillLimits(FillModel):
    """Keeps resting maker orders alive so HOLD-cancel behavior can be observed."""

    def _simulate_limit_fill(self, order, snapshot, adverse_risk):
        return None


def make_prediction(probability: float, timestamp: float = 100.0) -> PredictionResult:
    return PredictionResult(
        probability=probability,
        timestamp=timestamp,
        prediction_horizon=10,
        model_name="deterministic-test-stub",
        model_status=ModelStatus.FALLBACK_HEURISTIC,
    )


def make_snapshot(timestamp: float = 100.0, sequence_id: int = 1) -> MarketSnapshot:
    return MarketSnapshot(
        timestamp=timestamp,
        sequence_id=sequence_id,
        bids=[(99.95, 200.0), (99.90, 300.0)],
        asks=[(100.05, 200.0), (100.10, 300.0)],
    )


def make_snapshot_stream(num_ticks: int = 8, dt: float = 1.0, t0: float = 100.0):
    return [make_snapshot(timestamp=t0 + i * dt, sequence_id=i) for i in range(num_ticks)]


def make_decision(**overrides) -> ExecutionDecision:
    values = {
        "timestamp": 100.0,
        "strategy": "custom-label-without-mode-words",
        "side": OrderSide.BUY,
        "quantity": 50.0,
        "urgency": 0.40,
        "risk_score": 0.20,
        "reason": "test decision",
        "remaining_quantity": 500.0,
        "limit_price": 99.95,
        "execution_mode": ExecutionMode.AUTO,
        "cancel_active_orders": False,
    }
    values.update(overrides)
    return ExecutionDecision(**values)


def decide(side: OrderSide, risk: float, elapsed: float, horizon: float = 60.0, remaining: float = 500.0, initial: float = 500.0, **kwargs):
    strat = ProposedStrategy(predictor=StubPredictor(risk))
    return strat.compute_decision(
        snapshot=make_snapshot(),
        remaining_quantity=remaining,
        elapsed_time=elapsed,
        total_horizon=horizon,
        side=side,
        initial_quantity=initial,
        prediction_result=make_prediction(risk),
        depth_imbalance=kwargs.get("depth_imbalance", -0.9),
        momentum=kwargs.get("momentum", -0.05),
        trade_flow=kwargs.get("trade_flow", -0.8),
    )


def assert_no_directional_forecast(reason: str) -> None:
    lowered = reason.lower()
    for phrase in DIRECTIONAL_FORECAST_PHRASES:
        assert phrase not in lowered, f"Directional forecast language found in reason: {phrase!r}\n{reason}"


def test_high_risk_buy_does_not_trigger_aggressive_execution():
    decision = decide(OrderSide.BUY, risk=0.85, elapsed=10.0)

    assert decision.execution_mode == ExecutionMode.HOLD
    assert decision.quantity == 0.0
    assert decision.cancel_active_orders is True
    lowered = decision.reason.lower()
    assert "withdraw" in lowered or "avoid" in lowered or "unsafe" in lowered
    assert "passive" in lowered
    assert "urgency" in lowered
    assert "HOLD" in decision.reason
    assert_no_directional_forecast(decision.reason)


def test_high_risk_sell_does_not_trigger_aggressive_execution():
    decision = decide(OrderSide.SELL, risk=0.85, elapsed=10.0)

    assert decision.execution_mode == ExecutionMode.HOLD
    assert decision.quantity == 0.0
    assert decision.cancel_active_orders is True
    assert "withdraw" in decision.reason.lower() or "unsafe" in decision.reason.lower()
    assert_no_directional_forecast(decision.reason)


def test_high_risk_buy_and_sell_are_symmetric():
    buy = decide(OrderSide.BUY, risk=0.85, elapsed=10.0)
    sell = decide(OrderSide.SELL, risk=0.85, elapsed=10.0)
    assert buy.execution_mode == sell.execution_mode == ExecutionMode.HOLD
    assert buy.quantity == sell.quantity == 0.0
    assert buy.cancel_active_orders is True
    assert sell.cancel_active_orders is True


def test_high_urgency_overrides_passive_risk_avoidance():
    decision = decide(OrderSide.BUY, risk=0.85, elapsed=45.0)

    assert decision.urgency >= 0.75
    assert decision.execution_mode == ExecutionMode.AGGRESSIVE
    assert decision.quantity > 0
    assert decision.cancel_active_orders is True
    lowered = decision.reason.lower()
    assert "urgency" in lowered
    assert "completion" in lowered or "deadline" in lowered
    assert "aggressive" in lowered
    assert_no_directional_forecast(decision.reason)


def test_low_adverse_selection_risk_chooses_maker_execution_buy():
    decision = decide(OrderSide.BUY, risk=0.20, elapsed=10.0)
    snap = make_snapshot()

    assert decision.execution_mode == ExecutionMode.PASSIVE
    assert decision.limit_price == snap.best_bid
    assert decision.quantity > 0
    assert decision.cancel_active_orders is False
    lowered = decision.reason.lower()
    assert "passive" in lowered
    assert "risk" in lowered
    assert "urgency" in lowered
    assert_no_directional_forecast(decision.reason)


def test_low_adverse_selection_risk_chooses_maker_execution_sell():
    decision = decide(OrderSide.SELL, risk=0.20, elapsed=10.0)
    snap = make_snapshot()

    assert decision.execution_mode == ExecutionMode.PASSIVE
    assert decision.limit_price == snap.best_ask
    assert decision.quantity > 0
    assert decision.cancel_active_orders is False
    assert_no_directional_forecast(decision.reason)


def test_moderate_risk_follows_almgren_chriss_schedule():
    decision = decide(OrderSide.BUY, risk=0.50, elapsed=10.0)

    assert decision.quantity > 0
    assert decision.execution_mode == ExecutionMode.AGGRESSIVE
    assert decision.cancel_active_orders is True
    lowered = decision.reason.lower()
    assert "almgren" in lowered
    assert "trajectory" in lowered or "scheduled" in lowered or "inventory" in lowered
    assert_no_directional_forecast(decision.reason)


def test_completed_order_returns_hold():
    strat = ProposedStrategy(predictor=StubPredictor(0.5))
    decision = strat.compute_decision(
        snapshot=make_snapshot(),
        remaining_quantity=0.0,
        elapsed_time=10.0,
        total_horizon=60.0,
        side=OrderSide.BUY,
        prediction_result=make_prediction(0.5),
    )
    assert decision.execution_mode == ExecutionMode.HOLD
    assert decision.quantity == 0.0
    assert "complete" in decision.reason.lower()


def test_hold_cancels_existing_resting_order():
    sim = ExecutionSimulator(fill_model=NeverFillLimits(execution_delay_ms=0.0))
    om = OrderManager()
    snap = make_snapshot()

    resting = om.create_order(
        timestamp=snap.timestamp,
        side=OrderSide.BUY,
        order_type=OrderType.LIMIT,
        quantity=80.0,
        price=snap.best_bid,
    )
    om.submit_order(resting.order_id, snap.timestamp)
    assert resting.is_active
    assert len(om.get_active_orders()) == 1

    hold = make_decision(
        strategy="Proposed (ML + AC)",
        quantity=0.0,
        execution_mode=ExecutionMode.HOLD,
        cancel_active_orders=True,
        limit_price=None,
        risk_score=0.85,
        reason="withdraw passive exposure",
    )
    remaining, _ = sim.apply_execution_decision(
        decision=hold,
        order_manager=om,
        snap=snap,
        side=OrderSide.BUY,
        remaining_qty=500.0,
        total_quantity=500.0,
    )

    assert remaining == 500.0
    assert resting.status == OrderStatus.CANCELLED
    assert om.get_active_orders() == []
    assert len(om.get_order_history()) == 1


def test_simulator_honors_execution_mode_not_strategy_name():
    sim = ExecutionSimulator(fill_model=NeverFillLimits(execution_delay_ms=0.0))
    snap = make_snapshot()

    om_passive = OrderManager()
    passive = make_decision(
        strategy="Aggressive Market Sweep",
        execution_mode=ExecutionMode.PASSIVE,
        cancel_active_orders=False,
        quantity=40.0,
        limit_price=snap.best_bid,
    )
    sim.apply_execution_decision(passive, om_passive, snap, OrderSide.BUY, 500.0, 500.0)
    created = om_passive.get_order_history()
    assert len(created) == 1
    assert created[0].order_type == OrderType.LIMIT

    om_agg = OrderManager()
    aggressive = make_decision(
        strategy="Passive Maker Quote",
        execution_mode=ExecutionMode.AGGRESSIVE,
        cancel_active_orders=True,
        quantity=40.0,
        limit_price=snap.best_ask,
    )
    sim.apply_execution_decision(aggressive, om_agg, snap, OrderSide.BUY, 500.0, 500.0)
    created = om_agg.get_order_history()
    assert len(created) == 1
    assert created[0].order_type == OrderType.MARKET

    om_hold = OrderManager()
    hold = make_decision(
        strategy="Aggressive",
        execution_mode=ExecutionMode.HOLD,
        cancel_active_orders=True,
        quantity=0.0,
        limit_price=None,
    )
    sim.apply_execution_decision(hold, om_hold, snap, OrderSide.BUY, 500.0, 500.0)
    assert om_hold.get_order_history() == []


def test_resolve_order_type_uses_enum_before_strategy_text():
    aggressive_named_passive = make_decision(
        strategy="aggressive market",
        execution_mode=ExecutionMode.PASSIVE,
    )
    assert ExecutionSimulator.resolve_order_type(aggressive_named_passive) == OrderType.LIMIT

    passive_named_aggressive = make_decision(
        strategy="passive withdrawn",
        execution_mode=ExecutionMode.AGGRESSIVE,
    )
    assert ExecutionSimulator.resolve_order_type(passive_named_aggressive) == OrderType.MARKET

    hold = make_decision(strategy="Market", execution_mode=ExecutionMode.HOLD, quantity=10.0)
    assert ExecutionSimulator.resolve_order_type(hold) is None


def test_auto_mode_preserves_legacy_strategy_name_inference():
    market = make_decision(strategy="Market", execution_mode=ExecutionMode.AUTO)
    assert ExecutionSimulator.resolve_order_type(market) == OrderType.MARKET

    aggressive = make_decision(strategy="Aggressive", execution_mode=ExecutionMode.AUTO)
    assert ExecutionSimulator.resolve_order_type(aggressive) == OrderType.MARKET

    twap = make_decision(strategy="TWAP", execution_mode=ExecutionMode.AUTO)
    assert ExecutionSimulator.resolve_order_type(twap) == OrderType.LIMIT

    vwap = make_decision(strategy="VWAP", execution_mode=ExecutionMode.AUTO)
    assert ExecutionSimulator.resolve_order_type(vwap) == OrderType.LIMIT

    ac = make_decision(strategy="Almgren-Chriss", execution_mode=ExecutionMode.AUTO)
    assert ExecutionSimulator.resolve_order_type(ac) == OrderType.LIMIT

    passive = make_decision(strategy="Passive", execution_mode=ExecutionMode.AUTO)
    assert ExecutionSimulator.resolve_order_type(passive) == OrderType.LIMIT


def test_existing_baseline_strategies_still_function():
    snap = make_snapshot()
    kwargs = {"initial_quantity": 500.0, "risk_score": 0.30, "prediction_result": make_prediction(0.30)}
    common = dict(
        snapshot=snap,
        remaining_quantity=500.0,
        elapsed_time=5.0,
        total_horizon=60.0,
        side=OrderSide.BUY,
    )

    market = MarketStrategy().compute_decision(**common, **kwargs)
    twap = TWAPStrategy().compute_decision(**common, **kwargs)
    vwap = VWAPStrategy().compute_decision(**common, **kwargs)
    ac = AlmgrenChrissStrategy().compute_decision(**common, **kwargs)
    passive = PassiveStrategy().compute_decision(**common, **kwargs)
    aggressive = AggressiveStrategy().compute_decision(**common, **kwargs)

    for decision in (market, twap, vwap, ac, passive, aggressive):
        assert decision.execution_mode == ExecutionMode.AUTO
        assert decision.quantity > 0
        assert decision.cancel_active_orders is False

    sim = ExecutionSimulator(fill_model=FillModel(execution_delay_ms=0.0))
    snapshots = make_snapshot_stream(num_ticks=12, dt=1.0)
    for name in ("Market", "TWAP", "VWAP", "Passive", "Aggressive", "Almgren-Chriss"):
        result = sim.run_execution(
            strategy_name=name,
            snapshots=snapshots,
            total_quantity=200.0,
            horizon_sec=20.0,
            side=OrderSide.BUY,
        )
        assert result.strategy_name == name
        if name != "Passive":
            assert result.num_orders > 0
            assert result.executed_quantity > 0


def test_proposed_trajectory_records_passive_hold_and_aggressive():
    snapshots = make_snapshot_stream(num_ticks=8, dt=1.0, t0=100.0)
    predictor = TimestampPredictor(
        start_time=100.0,
        schedule=[(1.0, 0.20), (3.0, 0.85), (999.0, 0.50)],
    )
    engine = DynamicStrategyEngine(predictor=predictor)
    sim = ExecutionSimulator(
        strategy_engine=engine,
        fill_model=NeverFillLimits(execution_delay_ms=0.0),
    )
    result = sim.run_execution(
        strategy_name="Proposed (ML + AC)",
        snapshots=snapshots,
        total_quantity=500.0,
        horizon_sec=60.0,
        side=OrderSide.BUY,
    )

    modes = [pt["execution_mode"] for pt in result.trajectory]
    assert "passive" in modes
    assert "hold" in modes
    assert "aggressive" in modes

    for pt in result.trajectory:
        assert "execution_mode" in pt
        assert "cancel_active_orders" in pt
        assert_no_directional_forecast(pt["reason"])

    cancelled = [o for o in result.orders if o.status == OrderStatus.CANCELLED]
    assert cancelled, "HOLD should cancel the previously resting maker order"


def test_legacy_execution_decision_constructor_still_works():
    decision = ExecutionDecision(
        timestamp=100.0,
        strategy="TWAP",
        side=OrderSide.BUY,
        quantity=10.0,
        urgency=0.2,
        risk_score=0.5,
        reason="legacy constructor",
        remaining_quantity=100.0,
    )
    assert decision.execution_mode == ExecutionMode.AUTO
    assert decision.cancel_active_orders is False
    assert decision.limit_price is None
