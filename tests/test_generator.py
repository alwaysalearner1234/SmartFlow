"""Phase 12 tests for reproducible synthetic market-data generation.

These tests exercise public generator behavior and its data-quality invariants.
They intentionally avoid network access and real market-data dependencies.
"""

from dataclasses import replace
import math

import numpy as np
import pandas as pd
import pytest

import data.generator as generator_module
from config.config import MarketConfig, SCENARIOS
from data.contracts import MarketSnapshot, OrderSide, Trade
from data.generator import MarketDataGenerator


@pytest.fixture
def market_config():
    """Small deterministic market configuration for unit tests."""
    return MarketConfig(
        symbol="TEST-USD",
        tick_size=0.01,
        lot_size=0.001,
        num_levels=5,
        initial_price=100.0,
        base_spread=0.05,
        base_volatility=0.25,
        base_depth_per_level=500.0,
        random_seed=12345,
    )


@pytest.fixture
def generator(market_config):
    return MarketDataGenerator(config=market_config, seed=12345)


def _snapshot_signature(snapshots):
    """Return stable, comparable fields from generated snapshots."""
    return [
        (
            snapshot.timestamp,
            snapshot.sequence_id,
            tuple(snapshot.bids),
            tuple(snapshot.asks),
            snapshot.last_trade_price,
            snapshot.last_trade_size,
            snapshot.last_trade_side,
            tuple(
                (trade.timestamp, trade.trade_id, trade.price, trade.size, trade.side)
                for trade in snapshot.recent_trades
            ),
        )
        for snapshot in snapshots
    ]


class TestGeneratorConfiguration:
    def test_default_constructor_is_reproducible_from_config_seed(self, market_config):
        first = MarketDataGenerator(config=market_config)
        second = MarketDataGenerator(config=market_config)

        assert _snapshot_signature(first.generate_scenario_stream(num_ticks=8)) == (
            _snapshot_signature(second.generate_scenario_stream(num_ticks=8))
        )

    def test_explicit_seed_overrides_config_seed(self, market_config):
        first = MarketDataGenerator(config=market_config, seed=10)
        second = MarketDataGenerator(config=market_config, seed=10)
        different = MarketDataGenerator(config=market_config, seed=11)

        first_data = first.generate_scenario_stream(num_ticks=8)
        second_data = second.generate_scenario_stream(num_ticks=8)
        different_data = different.generate_scenario_stream(num_ticks=8)

        assert _snapshot_signature(first_data) == _snapshot_signature(second_data)
        assert _snapshot_signature(first_data) != _snapshot_signature(different_data)

    @pytest.mark.parametrize(
        ("field", "value"),
        [
            ("tick_size", 0),
            ("tick_size", float("nan")),
            ("lot_size", -1),
            ("initial_price", float("inf")),
            ("base_spread", 0),
            ("base_volatility", float("nan")),
            ("base_depth_per_level", -10),
            ("num_levels", 0),
            ("num_levels", 1.5),
        ],
    )
    def test_invalid_market_configuration_is_rejected(self, market_config, field, value):
        with pytest.raises(ValueError):
            MarketDataGenerator(config=replace(market_config, **{field: value}))

    @pytest.mark.parametrize("seed", [True, 1.5, "123", object()])
    def test_invalid_explicit_seed_is_rejected(self, market_config, seed):
        with pytest.raises(ValueError, match="seed"):
            MarketDataGenerator(config=market_config, seed=seed)


class TestScenarioStreamParameters:
    @pytest.mark.parametrize("num_ticks", [0, -1, 1.2, True, "10", None])
    def test_invalid_tick_count_is_rejected(self, generator, num_ticks):
        with pytest.raises(ValueError, match="num_ticks"):
            generator.generate_scenario_stream(num_ticks=num_ticks)

    @pytest.mark.parametrize("dt", [0, -0.1, float("nan"), float("inf"), True, "0.5"])
    def test_invalid_sampling_interval_is_rejected(self, generator, dt):
        with pytest.raises(ValueError, match="dt"):
            generator.generate_scenario_stream(num_ticks=2, dt=dt)

    @pytest.mark.parametrize(
        "start_time", [0, -1, float("nan"), float("inf"), True, "1700000000"]
    )
    def test_invalid_start_time_is_rejected(self, generator, start_time):
        with pytest.raises(ValueError, match="start_time"):
            generator.generate_scenario_stream(num_ticks=2, start_time=start_time)

    @pytest.mark.parametrize("scenario_key", ["", " ", None, 1, "does_not_exist"])
    def test_unknown_or_malformed_scenario_is_rejected(self, generator, scenario_key):
        with pytest.raises(ValueError, match="scenario_key|Unknown scenario"):
            generator.generate_scenario_stream(scenario_key=scenario_key, num_ticks=2)

    def test_all_configured_scenarios_generate_requested_number_of_snapshots(self, generator):
        assert SCENARIOS
        for scenario_key in SCENARIOS:
            snapshots = generator.generate_scenario_stream(
                scenario_key=scenario_key,
                num_ticks=3,
            )
            assert len(snapshots) == 3

    def test_empty_stream_is_not_allowed(self, generator):
        with pytest.raises(ValueError):
            generator.generate_scenario_stream(num_ticks=0)


class TestGeneratedMarketDataQuality:
    def test_stream_has_exact_count_and_chronological_timestamps(self, generator):
        start_time = 1_700_000_000.0
        dt = 0.25
        snapshots = generator.generate_scenario_stream(
            num_ticks=12, dt=dt, start_time=start_time
        )

        assert len(snapshots) == 12
        timestamps = [snapshot.timestamp for snapshot in snapshots]
        assert timestamps == sorted(timestamps)
        assert timestamps == pytest.approx(
            [start_time + dt * (index + 1) for index in range(12)]
        )
        assert [snapshot.sequence_id for snapshot in snapshots] == list(range(12))

    def test_snapshot_timestamps_and_sequence_ids_are_unique(self, generator):
        snapshots = generator.generate_scenario_stream(num_ticks=25)
        assert len({snapshot.timestamp for snapshot in snapshots}) == len(snapshots)
        assert len({snapshot.sequence_id for snapshot in snapshots}) == len(snapshots)

    def test_book_levels_have_valid_prices_sizes_and_ordering(self, generator, market_config):
        snapshots = generator.generate_scenario_stream(num_ticks=40)

        for snapshot in snapshots:
            assert isinstance(snapshot, MarketSnapshot)
            assert len(snapshot.bids) == market_config.num_levels
            assert len(snapshot.asks) == market_config.num_levels
            assert all(
                math.isfinite(price) and price > 0 and math.isfinite(size) and size > 0
                for price, size in snapshot.bids + snapshot.asks
            )
            bid_prices = [price for price, _ in snapshot.bids]
            ask_prices = [price for price, _ in snapshot.asks]
            assert bid_prices == sorted(bid_prices, reverse=True)
            assert ask_prices == sorted(ask_prices)
            assert snapshot.best_bid < snapshot.best_ask
            assert snapshot.mid_price > 0
            assert snapshot.spread > 0

    def test_generated_prices_align_to_tick_size(self, generator, market_config):
        snapshots = generator.generate_scenario_stream(num_ticks=30)
        for snapshot in snapshots:
            for price, _ in snapshot.bids + snapshot.asks:
                ticks = price / market_config.tick_size
                assert ticks == pytest.approx(round(ticks), abs=1e-6)

    def test_trade_timestamps_are_inside_their_sampling_interval(self, generator):
        start_time = 1_700_000_000.0
        dt = 0.5
        snapshots = generator.generate_scenario_stream(
            num_ticks=20, dt=dt, start_time=start_time
        )

        for index, snapshot in enumerate(snapshots):
            interval_start = start_time + index * dt
            interval_end = start_time + (index + 1) * dt
            trade_times = [trade.timestamp for trade in snapshot.recent_trades]
            assert all(interval_start <= timestamp <= interval_end for timestamp in trade_times)
            assert all(timestamp <= snapshot.timestamp for timestamp in trade_times)
            assert trade_times == sorted(trade_times)

    def test_trade_fields_are_valid_and_sides_are_enums(self, generator):
        snapshots = generator.generate_scenario_stream(num_ticks=60)
        trades = [trade for snapshot in snapshots for trade in snapshot.recent_trades]

        assert trades, "The deterministic test stream should contain trades."
        assert all(isinstance(trade, Trade) for trade in trades)
        assert all(math.isfinite(trade.timestamp) for trade in trades)
        assert all(math.isfinite(trade.price) and trade.price > 0 for trade in trades)
        assert all(math.isfinite(trade.size) and trade.size > 0 for trade in trades)
        assert all(trade.side in (OrderSide.BUY, OrderSide.SELL) for trade in trades)
        assert len({trade.trade_id for trade in trades}) == len(trades)

    def test_last_trade_summary_fields_are_valid_when_present(self, generator):
        snapshots = generator.generate_scenario_stream(num_ticks=40)
        for snapshot in snapshots:
            if snapshot.last_trade_price is not None:
                assert math.isfinite(snapshot.last_trade_price)
                assert snapshot.last_trade_price > 0
            if snapshot.last_trade_size is not None:
                assert math.isfinite(snapshot.last_trade_size)
                assert snapshot.last_trade_size > 0
            if snapshot.last_trade_side is not None:
                assert snapshot.last_trade_side in (OrderSide.BUY, OrderSide.SELL)

    def test_generator_does_not_mutate_market_config(self, market_config):
        before = repr(market_config)
        generator = MarketDataGenerator(config=market_config, seed=99)
        generator.generate_scenario_stream(num_ticks=5)
        assert repr(market_config) == before


class TestScenarioCharacteristics:
    def test_high_volatility_scenario_uses_wider_spreads_than_normal(self, market_config):
        normal = MarketDataGenerator(config=market_config, seed=987).generate_scenario_stream(
            "normal_market", num_ticks=100
        )
        volatile = MarketDataGenerator(config=market_config, seed=987).generate_scenario_stream(
            "high_volatility", num_ticks=100
        )
        normal_spread = np.mean([snapshot.spread for snapshot in normal])
        volatile_spread = np.mean([snapshot.spread for snapshot in volatile])
        assert volatile_spread > normal_spread

    def test_poor_liquidity_scenario_has_less_depth_than_normal(self, market_config):
        normal = MarketDataGenerator(config=market_config, seed=987).generate_scenario_stream(
            "normal_market", num_ticks=100
        )
        thin = MarketDataGenerator(config=market_config, seed=987).generate_scenario_stream(
            "poor_liquidity", num_ticks=100
        )
        normal_depth = np.mean([snapshot.total_bid_depth + snapshot.total_ask_depth for snapshot in normal])
        thin_depth = np.mean([snapshot.total_bid_depth + snapshot.total_ask_depth for snapshot in thin])
        assert thin_depth < normal_depth

    def test_scenario_parameters_must_be_finite_and_valid(self, monkeypatch, generator):
        original = SCENARIOS["normal_market"]
        bad_scenario = dict(original)
        bad_scenario["volatility"] = float("nan")
        monkeypatch.setitem(SCENARIOS, "test_bad_volatility", bad_scenario)
        with pytest.raises(ValueError, match="volatility"):
            generator.generate_scenario_stream("test_bad_volatility", num_ticks=2)

        bad_scenario = dict(original)
        bad_scenario["spread_mult"] = 0
        monkeypatch.setitem(SCENARIOS, "test_bad_spread", bad_scenario)
        with pytest.raises(ValueError, match="spread_mult"):
            generator.generate_scenario_stream("test_bad_spread", num_ticks=2)

        bad_scenario = dict(original)
        bad_scenario["depth_mult"] = float("inf")
        monkeypatch.setitem(SCENARIOS, "test_bad_depth", bad_scenario)
        with pytest.raises(ValueError, match="depth_mult"):
            generator.generate_scenario_stream("test_bad_depth", num_ticks=2)

        bad_scenario = dict(original)
        bad_scenario["adverse_intensity"] = float("nan")
        monkeypatch.setitem(SCENARIOS, "test_bad_adverse", bad_scenario)
        with pytest.raises(ValueError, match="adverse_intensity"):
            generator.generate_scenario_stream("test_bad_adverse", num_ticks=2)


class TestDatasetPersistence:
    @pytest.mark.parametrize(
        "output_name", ["", " ", "../escape.parquet", "nested/data.parquet", "data.csv", "data.parquet.tmp"]
    )
    def test_invalid_output_name_is_rejected(self, generator, output_name):
        with pytest.raises(ValueError, match="output_name"):
            generator.generate_and_save_dataset(num_ticks=2, output_name=output_name)

    def test_save_writes_chronological_csv_and_parquet(self, tmp_path, monkeypatch, generator):
        raw_dir = tmp_path / "raw"
        processed_dir = tmp_path / "processed"
        monkeypatch.setattr(generator_module, "RAW_DATA_DIR", raw_dir)
        monkeypatch.setattr(generator_module, "PROCESSED_DATA_DIR", processed_dir)

        parquet_calls = []

        def fake_to_parquet(frame, path, index=False, **kwargs):
            parquet_calls.append((frame.copy(), path, index))
            # Create a marker file so persistence path behavior can be asserted.
            with open(path, "wb") as file_handle:
                file_handle.write(b"test parquet placeholder")

        monkeypatch.setattr(pd.DataFrame, "to_parquet", fake_to_parquet)

        frame = generator.generate_and_save_dataset(
            scenario_key="normal_market",
            num_ticks=6,
            output_name="phase12_test.parquet",
        )

        csv_path = raw_dir / "normal_market_raw.csv"
        parquet_path = processed_dir / "phase12_test.parquet"
        assert csv_path.is_file()
        assert parquet_path.is_file()
        assert len(parquet_calls) == 1
        assert parquet_calls[0][1] == parquet_path
        assert parquet_calls[0][2] is False
        assert len(frame) == 6
        assert frame["timestamp"].is_monotonic_increasing
        assert list(pd.read_csv(csv_path)["timestamp"]) == list(frame["timestamp"])

    def test_saved_dataframe_contains_core_market_columns(self, tmp_path, monkeypatch, generator):
        monkeypatch.setattr(generator_module, "RAW_DATA_DIR", tmp_path / "raw")
        monkeypatch.setattr(generator_module, "PROCESSED_DATA_DIR", tmp_path / "processed")
        monkeypatch.setattr(pd.DataFrame, "to_parquet", lambda self, path, index=False, **kwargs: None)

        frame = generator.generate_and_save_dataset(num_ticks=3)
        expected = {
            "timestamp", "sequence_id", "mid_price", "spread", "best_bid",
            "best_bid_size", "best_ask", "best_ask_size", "total_bid_depth",
            "total_ask_depth", "last_trade_price", "last_trade_size",
            "last_trade_side", "num_recent_trades", "bid_price_0", "bid_size_0",
            "ask_price_0", "ask_size_0",
        }
        assert expected.issubset(frame.columns)

