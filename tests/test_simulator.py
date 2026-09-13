"""Unit tests for the market and fill simulators."""

import pytest
import numpy as np

from data.contracts import (
    OrderSide,
    OrderType,
    LiquidityType,
    MarketSnapshot,
    Order,
)
from simulator.market_simulator import MarketSimulator
from simulator.fill_model import FillModel
from simulator.order_book_simulator import OrderBookSimulator
from simulator.execution_simulator import ExecutionSimulator
from data.generator import MarketDataGenerator


@pytest.fixture
def mock_snapshots():
    gen = MarketDataGenerator(seed=777)
    return gen.generate_scenario_stream("normal_market", num_ticks=20, dt=1.0)


def test_market_simulator_replay(mock_snapshots):
    sim = MarketSimulator(mock_snapshots)
    assert sim.total_snapshots == 20

    stream = list(sim.get_stream())
    assert len(stream) == 20
    assert stream[0].timestamp < stream[-1].timestamp


def test_fill_model_market_order():
    fill_model = FillModel(execution_delay_ms=0.0)
    snap = MarketSnapshot(
        timestamp=100.0,
        sequence_id=1,
        bids=[(99.0, 100.0)],
        asks=[(101.0, 50.0), (102.0, 100.0)],
    )

    # Buy market order for 80 units: 50 @ 101, 30 @ 102 -> avg 101.375
    order = Order(
        order_id="TEST-1",
        timestamp=100.0,
        side=OrderSide.BUY,
        order_type=OrderType.MARKET,
        quantity=80.0,
    )

    fill = fill_model.simulate_fill(order, snap)
    assert fill is not None
    assert fill.quantity == 80.0
    assert abs(fill.price - 101.375) < 1e-3
    assert fill.liquidity_type == LiquidityType.TAKER
    assert fill.slippage > 0.0


def test_fill_model_latency_delay():
    # 50ms delay
    fill_model = FillModel(execution_delay_ms=50.0)
    snap = MarketSnapshot(
        timestamp=100.020,  # Only 20ms later
        sequence_id=2,
        bids=[(99.0, 100.0)],
        asks=[(101.0, 100.0)],
    )
    order = Order(
        order_id="TEST-DELAY",
        timestamp=100.000,
        side=OrderSide.BUY,
        order_type=OrderType.MARKET,
        quantity=50.0,
    )

    # Has not arrived yet -> fill should be None
    fill = fill_model.simulate_fill(order, snap)
    assert fill is None


def test_order_book_simulator_impact():
    ob_sim = OrderBookSimulator(permanent_impact_factor=1e-3, temporary_impact_factor=1e-3)
    snap = MarketSnapshot(
        timestamp=100.0,
        sequence_id=1,
        bids=[(100.0, 500.0)],
        asks=[(100.1, 500.0)],
    )
    from data.contracts import Fill
    fill = Fill(
        fill_id="F-1",
        order_id="O-1",
        timestamp=100.0,
        price=100.1,
        quantity=100.0,
        liquidity_type=LiquidityType.TAKER,
    )

    # A BUY fill pushes prices UP
    shifted = ob_sim.apply_fill_impact(snap, fill, side=OrderSide.BUY)
    assert shifted.best_bid > snap.best_bid
    assert shifted.best_ask > snap.best_ask
