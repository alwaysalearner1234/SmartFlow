"""Phase 12 integration tests for SmartFlow synthetic market scenarios.

These tests verify scenario registration, deterministic generation, expected
market characteristics, temporal consistency, and data-quality invariants.
No network calls or live market data are used.
"""

import math

import numpy as np
import pytest

from config.config import MarketConfig, SCENARIOS
from data.contracts import MarketSnapshot, OrderSide
from data.generator import MarketDataGenerator


@pytest.fixture
def market_config():
    return MarketConfig(
        symbol="SCENARIO-TEST",
        tick_size=0.01,
        lot_size=0.001,
        num_levels=5,
        initial_price=100.0,
        base_spread=0.05,
        base_volatility=0.25,
        base_depth_per_level=500.0,
        random_seed=202612,
    )


def make_generator(market_config, seed=202612):
    return MarketDataGenerator(config=market_config, seed=seed)


def _signature(snapshots):
    return [
        (
            s.timestamp,
            s.sequence_id,
            tuple(s.bids),
            tuple(s.asks),
            s.last_trade_price,
            s.last_trade_size,
            s.last_trade_side,
            tuple((t.timestamp, t.trade_id, t.price, t.size, t.side) for t in s.recent_trades),
        )
        for s in snapshots
    ]


def _mean_spread(snapshots):
    return float(np.mean([s.spread for s in snapshots]))


def _mean_depth(snapshots):
    return float(np.mean([sum(size for _, size in s.bids + s.asks) for s in snapshots]))


def _return_volatility(snapshots):
    mids = np.asarray([s.mid_price for s in snapshots], dtype=float)
    returns = np.diff(mids) / mids[:-1]
    return float(np.std(returns))


class TestScenarioRegistry:
    def test_scenario_registry_is_nonempty(self):
        assert isinstance(SCENARIOS, dict)
        assert SCENARIOS

    def test_every_registered_scenario_generates_snapshots(self, market_config):
        generator = make_generator(market_config)
        for key in SCENARIOS:
            snapshots = generator.generate_scenario_stream(
                scenario_key=key, num_ticks=4
            )
            assert len(snapshots) == 4, key
            assert all(isinstance(snapshot, MarketSnapshot) for snapshot in snapshots)

    @pytest.mark.parametrize(
        "scenario_key",
        [
            "normal_market",
            "high_volatility",
            "poor_liquidity",
            "high_adverse_selection",
            "low_adverse_selection",
            "small_order",
            "medium_order",
            "large_order",
        ],
    )
    def test_core_scenarios_are_registered(self, scenario_key):
        assert scenario_key in SCENARIOS

    @pytest.mark.parametrize("scenario_key", ["", " ", None, "unknown", 42])
    def test_unknown_scenario_is_rejected(self, market_config, scenario_key):
        with pytest.raises(ValueError):
            make_generator(market_config).generate_scenario_stream(
                scenario_key=scenario_key, num_ticks=2
            )


class TestScenarioReproducibility:
    @pytest.mark.parametrize(
        "scenario_key",
        ["normal_market", "high_volatility", "poor_liquidity", "high_adverse_selection"],
    )
    def test_same_seed_and_scenario_produce_identical_streams(
        self, market_config, scenario_key
    ):
        first = make_generator(market_config, seed=71).generate_scenario_stream(
            scenario_key=scenario_key, num_ticks=30
        )
        second = make_generator(market_config, seed=71).generate_scenario_stream(
            scenario_key=scenario_key, num_ticks=30
        )
        assert _signature(first) == _signature(second)

    def test_different_seeds_produce_different_streams(self, market_config):
        first = make_generator(market_config, seed=71).generate_scenario_stream(num_ticks=20)
        second = make_generator(market_config, seed=72).generate_scenario_stream(num_ticks=20)
        assert _signature(first) != _signature(second)

    def test_scenario_stream_generation_does_not_change_requested_length(self, market_config):
        generator = make_generator(market_config)
        assert len(generator.generate_scenario_stream(num_ticks=7)) == 7
        assert len(generator.generate_scenario_stream(num_ticks=13)) == 13


class TestScenarioMarketCharacteristics:
    def test_high_volatility_has_wider_spreads_than_normal_market(self, market_config):
        normal = make_generator(market_config).generate_scenario_stream(
            scenario_key="normal_market", num_ticks=250
        )
        volatile = make_generator(market_config).generate_scenario_stream(
            scenario_key="high_volatility", num_ticks=250
        )
        assert _mean_spread(volatile) > _mean_spread(normal)

    def test_poor_liquidity_has_lower_visible_depth_than_normal_market(self, market_config):
        normal = make_generator(market_config).generate_scenario_stream(
            scenario_key="normal_market", num_ticks=100
        )
        thin = make_generator(market_config).generate_scenario_stream(
            scenario_key="poor_liquidity", num_ticks=100
        )
        assert _mean_depth(thin) < _mean_depth(normal)

    def test_low_adverse_selection_scenario_is_configured_less_toxic_than_high(self):
        low = SCENARIOS["low_adverse_selection"].get("adverse_intensity", 0.0)
        high = SCENARIOS["high_adverse_selection"].get("adverse_intensity", 0.0)
        assert low < high

    def test_high_volatility_scenario_has_higher_configured_volatility(self):
        assert SCENARIOS["high_volatility"]["volatility"] > SCENARIOS["normal_market"]["volatility"]

    def test_poor_liquidity_scenario_has_lower_configured_depth(self):
        assert SCENARIOS["poor_liquidity"]["depth_mult"] < SCENARIOS["normal_market"]["depth_mult"]

    def test_low_adverse_selection_has_less_adverse_intensity_than_normal(self):
        assert SCENARIOS["low_adverse_selection"]["adverse_intensity"] < SCENARIOS["normal_market"]["adverse_intensity"]


class TestScenarioTemporalAndBookIntegrity:
    @pytest.mark.parametrize(
        "scenario_key",
        ["normal_market", "high_volatility", "poor_liquidity", "high_adverse_selection"],
    )
    def test_snapshots_are_chronological_with_unique_sequence_ids(
        self, market_config, scenario_key
    ):
        start = 1_700_000_000.0
        dt = 0.25
        snapshots = make_generator(market_config).generate_scenario_stream(
            scenario_key=scenario_key, num_ticks=20, dt=dt, start_time=start
        )
        assert [s.timestamp for s in snapshots] == sorted(s.timestamp for s in snapshots)
        assert [s.timestamp for s in snapshots] == pytest.approx(
            [start + dt * (i + 1) for i in range(20)]
        )
        assert [s.sequence_id for s in snapshots] == list(range(20))

    def test_generated_books_are_ordered_nonempty_and_uncrossed(self, market_config):
        snapshots = make_generator(market_config).generate_scenario_stream(
            scenario_key="poor_liquidity", num_ticks=100
        )
        for snapshot in snapshots:
            assert snapshot.bids and snapshot.asks
            bid_prices = [price for price, _ in snapshot.bids]
            ask_prices = [price for price, _ in snapshot.asks]
            assert bid_prices == sorted(bid_prices, reverse=True)
            assert ask_prices == sorted(ask_prices)
            assert snapshot.best_bid < snapshot.best_ask
            assert all(math.isfinite(p) and p > 0 and math.isfinite(q) and q > 0
                       for p, q in snapshot.bids + snapshot.asks)

    def test_trade_timestamps_never_exceed_snapshot_timestamp(self, market_config):
        start = 1_700_000_000.0
        dt = 0.5
        snapshots = make_generator(market_config).generate_scenario_stream(
            scenario_key="high_adverse_selection", num_ticks=50, dt=dt, start_time=start
        )
        for i, snapshot in enumerate(snapshots):
            lower = start + i * dt
            upper = start + (i + 1) * dt
            trade_times = [trade.timestamp for trade in snapshot.recent_trades]
            assert all(lower <= timestamp <= upper for timestamp in trade_times)
            assert all(timestamp <= snapshot.timestamp for timestamp in trade_times)
            assert trade_times == sorted(trade_times)

    def test_trade_ids_are_unique_across_a_stream(self, market_config):
        snapshots = make_generator(market_config).generate_scenario_stream(num_ticks=100)
        trade_ids = [trade.trade_id for s in snapshots for trade in s.recent_trades]
        assert len(trade_ids) == len(set(trade_ids))

    def test_trade_sides_use_order_side_enum(self, market_config):
        snapshots = make_generator(market_config).generate_scenario_stream(num_ticks=60)
        trades = [trade for s in snapshots for trade in s.recent_trades]
        assert trades
        assert all(trade.side in (OrderSide.BUY, OrderSide.SELL) for trade in trades)

    def test_all_scenario_snapshots_have_finite_mid_prices(self, market_config):
        generator = make_generator(market_config)
        for key in SCENARIOS:
            snapshots = generator.generate_scenario_stream(scenario_key=key, num_ticks=10)
            assert all(math.isfinite(s.mid_price) and s.mid_price > 0 for s in snapshots), key

