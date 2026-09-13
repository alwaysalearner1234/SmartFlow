"""Unit tests for execution strategies, Almgren-Chriss, and Order Manager."""

import pytest
import numpy as np

from data.contracts import (
    OrderSide,
    OrderType,
    OrderStatus,
    LiquidityType,
    MarketSnapshot,
    ExecutionDecision,
)
from execution.almgren_chriss import AlmgrenChrissModel
from execution.order_manager import OrderManager
from execution.passive import PassiveStrategy
from execution.aggressive import AggressiveStrategy
from strategies.twap import TWAPStrategy
from strategies.vwap import VWAPStrategy
from strategies.proposed import ProposedStrategy
from execution.strategy_engine import DynamicStrategyEngine


@pytest.fixture
def mock_snapshot():
    return MarketSnapshot(
        timestamp=100.0,
        sequence_id=1,
        bids=[(99.95, 200.0), (99.90, 300.0)],
        asks=[(100.05, 200.0), (100.10, 300.0)],
    )


def test_almgren_chriss_schedule():
    ac = AlmgrenChrissModel(risk_aversion=1e-4, temporary_impact=2.5e-4, permanent_impact=2.5e-5)
    total_qty = 1000.0
    sched = ac.generate_schedule(total_quantity=total_qty, horizon_sec=60.0, num_slices=10)

    # Initial holdings match total quantity
    assert abs(sched.holdings[0] - total_qty) < 1e-4
    # Final holdings approach zero
    assert abs(sched.holdings[-1]) < 1.0
    # Sum of trade sizes equals total quantity
    assert abs(np.sum(sched.trade_sizes) - total_qty) < 1e-4
    # Holdings monotonically decrease
    assert np.all(np.diff(sched.holdings) <= 1e-5)
    # Expected cost is positive
    assert sched.expected_cost > 0.0


def test_order_manager_lifecycle():
    om = OrderManager()
    order = om.create_order(
        timestamp=100.0,
        side=OrderSide.BUY,
        order_type=OrderType.LIMIT,
        quantity=500.0,
        price=99.95,
    )
    assert order.status == OrderStatus.PENDING
    assert order.remaining_quantity == 500.0

    # Submit
    om.submit_order(order.order_id, timestamp=100.1)
    assert order.status == OrderStatus.SUBMITTED

    # Partial Fill
    fill1 = om.record_fill(
        order_id=order.order_id,
        timestamp=100.5,
        price=99.95,
        quantity=200.0,
        liquidity_type=LiquidityType.MAKER,
    )
    assert order.status == OrderStatus.PARTIALLY_FILLED
    assert order.filled_quantity == 200.0
    assert order.remaining_quantity == 300.0

    # Final Fill
    fill2 = om.record_fill(
        order_id=order.order_id,
        timestamp=101.0,
        price=99.95,
        quantity=300.0,
        liquidity_type=LiquidityType.MAKER,
    )
    assert order.status == OrderStatus.FILLED
    assert order.remaining_quantity == 0.0
    assert om.total_filled_quantity == 500.0


def test_passive_strategy_adverse_backoff(mock_snapshot):
    strat = PassiveStrategy(risk_threshold=0.65)

    # Benign market -> places passive order
    dec_low = strat.compute_decision(
        snapshot=mock_snapshot,
        remaining_quantity=500.0,
        elapsed_time=10.0,
        total_horizon=60.0,
        side=OrderSide.BUY,
        risk_score=0.20,
    )
    assert dec_low.quantity > 0
    assert dec_low.limit_price == mock_snapshot.best_bid

    # Toxic adverse market -> backs off (quantity = 0)
    dec_high = strat.compute_decision(
        snapshot=mock_snapshot,
        remaining_quantity=500.0,
        elapsed_time=10.0,
        total_horizon=60.0,
        side=OrderSide.BUY,
        risk_score=0.85,
    )
    assert dec_high.quantity == 0.0
    assert "adverse risk" in dec_high.reason.lower()


def test_dynamic_strategy_engine_decisions(mock_snapshot):
    engine = DynamicStrategyEngine()
    features = {"depth_imbalance_l1": 0.5, "spread_expansion_ratio": 1.0, "volatility_std_10": 0.15}

    decision = engine.evaluate_and_decide(
        strategy_name="Proposed (ML + AC)",
        snapshot=mock_snapshot,
        features_dict=features,
        remaining_quantity=500.0,
        initial_quantity=500.0,
        elapsed_time=5.0,
        total_horizon=60.0,
        side=OrderSide.BUY,
    )
    assert isinstance(decision, ExecutionDecision)
    assert decision.quantity > 0
    assert len(decision.reason) > 10
