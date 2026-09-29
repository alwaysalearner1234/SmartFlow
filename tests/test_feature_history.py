"""
Feature History Builder Tests.

This test module validates the Phase 2 chronological feature-history layer
that feeds the forecasting pipeline.

Responsibilities
----------------
- Validate the NVIDIA feature-history schema and feature ordering.
- Validate chronological and unique timestamp behavior.
- Validate causal trade-timestamp protection.
- Validate extractor output and merge behavior.
- Validate missing and non-finite feature handling.
- Validate deterministic construction and input immutability.
- Validate convenience APIs and configuration behavior.
- Validate compatibility with the synthetic market-data generator.

The tests do not validate forecasting sequence construction, target creation,
normalization, model architecture, or execution behavior. Those concerns are
covered by their respective modules.
"""

# ---------------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------------

from dataclasses import replace
from typing import List

import numpy as np
import pandas as pd
import pytest

import data.feature_history as feature_history_module
from data.contracts import MarketSnapshot, OrderSide, Trade
from data.feature_history import (
    FeatureHistoryBuilder,
    FeatureHistoryConfig,
    NVIDIA_FEATURES,
    REQUIRED_HISTORY_COLUMNS,
    build_feature_history,
    get_feature_names,
    get_required_columns,
)
from data.generator import MarketDataGenerator


# ---------------------------------------------------------------------------
# Test Constants
# ---------------------------------------------------------------------------

START_TIME = 1_700_000_000.0


# ---------------------------------------------------------------------------
# Test Helpers
# ---------------------------------------------------------------------------


def make_snapshot(
    timestamp: float,
    sequence_id: int,
    *,
    mid_price: float = 100.0,
    recent_trades: List[Trade] | None = None,
) -> MarketSnapshot:
    """Create a compact valid MarketSnapshot for isolated tests."""

    half_spread = 0.05

    return MarketSnapshot(
        timestamp=timestamp,
        sequence_id=sequence_id,
        bids=[
            (mid_price - half_spread, 100.0),
            (mid_price - 0.10, 80.0),
            (mid_price - 0.15, 60.0),
        ],
        asks=[
            (mid_price + half_spread, 100.0),
            (mid_price + 0.10, 80.0),
            (mid_price + 0.15, 60.0),
        ],
        last_trade_price=mid_price,
        last_trade_size=10.0,
        last_trade_side=OrderSide.BUY,
        recent_trades=(
            list(recent_trades)
            if recent_trades is not None
            else []
        ),
    )


def make_snapshots(count: int = 30) -> List[MarketSnapshot]:
    """Create a deterministic chronological snapshot stream."""

    return [
        make_snapshot(
            timestamp=START_TIME + (index + 1) * 0.5,
            sequence_id=index,
            mid_price=100.0 + index * 0.02,
        )
        for index in range(count)
    ]


def make_feature_frame(
    snapshots: List[MarketSnapshot],
    *,
    value: float = 1.0,
) -> pd.DataFrame:
    """Create a complete synthetic extractor output frame."""

    return pd.DataFrame(
        {
            "timestamp": [snapshot.timestamp for snapshot in snapshots],
            **{
                feature: np.full(
                    len(snapshots),
                    value,
                    dtype=float,
                )
                for feature in NVIDIA_FEATURES
                if feature != "mid_price_return"
            },
            "mid_price_return": np.full(
                len(snapshots),
                value,
                dtype=float,
            ),
            "mid_price": np.full(
                len(snapshots),
                100.0,
                dtype=float,
            ),
        }
    )


def make_extractor_frames(
    snapshots: List[MarketSnapshot],
) -> dict[str, pd.DataFrame]:
    """Create one valid frame for each feature extractor family."""

    timestamps = [snapshot.timestamp for snapshot in snapshots]

    order_book = pd.DataFrame(
        {
            "timestamp": timestamps,
            "mid_price": [100.0 + i for i in range(len(snapshots))],
            "best_bid_size": 100.0,
            "best_ask_size": 100.0,
            "micro_price": 100.0,
        }
    )

    imbalance = pd.DataFrame(
        {
            "timestamp": timestamps,
            "depth_imbalance_l1": 0.0,
            "depth_imbalance_multilevel": 0.0,
        }
    )

    spread = pd.DataFrame(
        {
            "timestamp": timestamps,
            "spread_bps": 10.0,
        }
    )

    momentum = pd.DataFrame(
        {
            "timestamp": timestamps,
            "mid_price_return": 0.001,
            "momentum_ret_5": 0.002,
            "momentum_ret_20": 0.003,
            "micro_price_divergence": 0.0001,
        }
    )

    volatility = pd.DataFrame(
        {
            "timestamp": timestamps,
            "volatility_std_10": 0.01,
        }
    )

    trade_flow = pd.DataFrame(
        {
            "timestamp": timestamps,
            "ofi_instant": 1.0,
            "ofi_sum_5": 5.0,
            "trade_volume_imbalance": 0.1,
        }
    )

    return {
        "order_book": order_book,
        "imbalance": imbalance,
        "spread": spread,
        "momentum": momentum,
        "volatility": volatility,
        "trade_flow": trade_flow,
    }


def patch_extractors(
    monkeypatch: pytest.MonkeyPatch,
    snapshots: List[MarketSnapshot],
) -> dict[str, pd.DataFrame]:
    """Patch all extractor entry points with deterministic test frames."""

    frames = make_extractor_frames(snapshots)

    monkeypatch.setattr(
        feature_history_module,
        "extract_order_book_features",
        lambda _: frames["order_book"].copy(),
    )
    monkeypatch.setattr(
        feature_history_module,
        "extract_imbalance_features",
        lambda _: frames["imbalance"].copy(),
    )
    monkeypatch.setattr(
        feature_history_module,
        "extract_spread_features",
        lambda _: frames["spread"].copy(),
    )
    monkeypatch.setattr(
        feature_history_module,
        "extract_momentum_features",
        lambda _: frames["momentum"].copy(),
    )
    monkeypatch.setattr(
        feature_history_module,
        "extract_volatility_features",
        lambda _: frames["volatility"].copy(),
    )
    monkeypatch.setattr(
        feature_history_module,
        "extract_trade_flow_features",
        lambda _: frames["trade_flow"].copy(),
    )

    return frames


# ---------------------------------------------------------------------------
# Contract Constants
# ---------------------------------------------------------------------------


class TestFeatureHistoryConstants:
    """Validate the stable feature-history schema."""

    def test_nvidia_features_are_nonempty(self) -> None:
        """The NVIDIA forecasting feature contract must contain features."""

        assert NVIDIA_FEATURES
        assert len(NVIDIA_FEATURES) == 14

    def test_nvidia_features_are_unique(self) -> None:
        """Feature names must not be duplicated."""

        assert len(NVIDIA_FEATURES) == len(set(NVIDIA_FEATURES))

    def test_required_columns_begin_with_timestamp_and_mid_price(self) -> None:
        """Required history columns must begin with the temporal price fields."""

        assert REQUIRED_HISTORY_COLUMNS[:2] == [
            "timestamp",
            "mid_price",
        ]

    def test_required_columns_contain_exact_nvidia_features(self) -> None:
        """Required columns must expose the complete NVIDIA feature contract."""

        assert REQUIRED_HISTORY_COLUMNS[2:] == NVIDIA_FEATURES

    def test_feature_names_helper_returns_copy(self) -> None:
        """Feature-name helper output must not mutate the module constant."""

        names = get_feature_names()
        names.append("unexpected_feature")

        assert "unexpected_feature" not in NVIDIA_FEATURES

    def test_required_columns_helper_returns_copy(self) -> None:
        """Required-column helper output must not mutate the module constant."""

        columns = get_required_columns()
        columns.append("unexpected_column")

        assert "unexpected_column" not in REQUIRED_HISTORY_COLUMNS


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


class TestFeatureHistoryConfig:
    """Validate feature-history configuration defaults and customization."""

    def test_default_configuration(self) -> None:
        """Default configuration should enforce safe chronological behavior."""

        config = FeatureHistoryConfig()

        assert config.require_chronological_order is True
        assert config.allow_missing_values is False
        assert config.drop_invalid_rows is False
        assert config.require_unique_timestamps is True
        assert config.sort_output is True

    def test_configuration_is_immutable(self) -> None:
        """FeatureHistoryConfig should remain frozen after construction."""

        config = FeatureHistoryConfig()

        with pytest.raises((AttributeError, TypeError)):
            config.sort_output = False  # type: ignore[misc]

    def test_custom_configuration_is_preserved(self) -> None:
        """Explicit configuration values should be retained."""

        config = FeatureHistoryConfig(
            require_chronological_order=False,
            allow_missing_values=True,
            drop_invalid_rows=True,
            require_unique_timestamps=False,
            sort_output=False,
        )

        assert config.require_chronological_order is False
        assert config.allow_missing_values is True
        assert config.drop_invalid_rows is True
        assert config.require_unique_timestamps is False
        assert config.sort_output is False


# ---------------------------------------------------------------------------
# Builder Construction
# ---------------------------------------------------------------------------


class TestFeatureHistoryBuilderConstruction:
    """Validate builder construction and configuration handling."""

    def test_default_builder_uses_default_configuration(self) -> None:
        """Builder should create the documented default configuration."""

        builder = FeatureHistoryBuilder()

        assert isinstance(builder.config, FeatureHistoryConfig)
        assert builder.config == FeatureHistoryConfig()

    def test_custom_configuration_is_used(self) -> None:
        """Builder should retain a supplied configuration object."""

        config = FeatureHistoryConfig(sort_output=False)
        builder = FeatureHistoryBuilder(config=config)

        assert builder.config is config


# ---------------------------------------------------------------------------
# Empty and Invalid Inputs
# ---------------------------------------------------------------------------


class TestInputValidation:
    """Validate snapshot-input rejection and edge cases."""

    def test_none_snapshots_raises(self) -> None:
        """None is not a valid snapshot stream."""

        with pytest.raises(ValueError, match="must not be None"):
            FeatureHistoryBuilder().build(None)  # type: ignore[arg-type]

    def test_empty_snapshot_sequence_returns_schema_only_dataframe(self) -> None:
        """Empty input should return the declared empty history schema."""

        result = FeatureHistoryBuilder().build([])

        assert isinstance(result, pd.DataFrame)
        assert result.empty
        assert list(result.columns) == REQUIRED_HISTORY_COLUMNS

    def test_nonfinite_snapshot_timestamp_raises(self) -> None:
        """NaN and infinite snapshot timestamps must be rejected."""

        for timestamp in [np.nan, np.inf, -np.inf]:
            snapshots = [
                make_snapshot(timestamp, 0),
            ]

            with pytest.raises(ValueError, match="not finite"):
                FeatureHistoryBuilder().build(snapshots)

    def test_out_of_order_snapshots_raise_by_default(self) -> None:
        """Chronological order is mandatory under the default configuration."""

        snapshots = make_snapshots(3)
        snapshots[1], snapshots[2] = snapshots[2], snapshots[1]

        with pytest.raises(ValueError, match="chronologically ordered"):
            FeatureHistoryBuilder().build(snapshots)

    def test_duplicate_snapshot_timestamps_raise(self) -> None:
        """Duplicate snapshot timestamps must be rejected by default."""

        snapshots = make_snapshots(3)
        snapshots[1].timestamp = snapshots[0].timestamp

        with pytest.raises(ValueError, match="Duplicate snapshot timestamps"):
            FeatureHistoryBuilder().build(snapshots)

    def test_non_snapshot_input_is_rejected(self) -> None:
        """The builder should fail clearly when required snapshot attributes are absent."""

        with pytest.raises(AttributeError):
            FeatureHistoryBuilder().build([object()])  # type: ignore[list-item]


# ---------------------------------------------------------------------------
# Temporal Safety
# ---------------------------------------------------------------------------


class TestTemporalSafety:
    """Validate snapshot/trade temporal leakage protection."""

    def test_trade_at_snapshot_timestamp_is_allowed(self) -> None:
        """A trade occurring exactly at snapshot time is causally valid."""

        timestamp = START_TIME + 1.0
        trade = Trade(
            timestamp=timestamp,
            trade_id="T-1",
            price=100.0,
            size=10.0,
            side=OrderSide.BUY,
        )

        snapshots = [
            make_snapshot(
                timestamp,
                0,
                recent_trades=[trade],
            )
        ]

        result = FeatureHistoryBuilder().build(snapshots)

        assert result["timestamp"].tolist() == [timestamp]

    def test_trade_before_snapshot_timestamp_is_allowed(self) -> None:
        """Trades occurring before the snapshot timestamp are causal."""

        timestamp = START_TIME + 1.0
        trade = Trade(
            timestamp=timestamp - 0.25,
            trade_id="T-1",
            price=100.0,
            size=10.0,
            side=OrderSide.BUY,
        )

        snapshots = [
            make_snapshot(
                timestamp,
                0,
                recent_trades=[trade],
            )
        ]

        result = FeatureHistoryBuilder().build(snapshots)

        assert len(result) == 1

    def test_trade_after_snapshot_timestamp_raises(self) -> None:
        """A future trade attached to a snapshot must be rejected."""

        timestamp = START_TIME + 1.0
        trade = Trade(
            timestamp=timestamp + 0.01,
            trade_id="T-1",
            price=100.0,
            size=10.0,
            side=OrderSide.BUY,
        )

        snapshots = [
            make_snapshot(
                timestamp,
                0,
                recent_trades=[trade],
            )
        ]

        with pytest.raises(ValueError, match="Temporal leakage detected"):
            FeatureHistoryBuilder().build(snapshots)

    def test_future_trade_in_later_snapshot_is_checked_independently(self) -> None:
        """Causal validation must inspect every snapshot, not just the first."""

        snapshots = make_snapshots(3)
        snapshots[2].recent_trades.append(
            Trade(
                timestamp=snapshots[2].timestamp + 1.0,
                trade_id="T-future",
                price=100.0,
                size=5.0,
                side=OrderSide.SELL,
            )
        )

        with pytest.raises(ValueError, match="Temporal leakage detected"):
            FeatureHistoryBuilder().build(snapshots)


# ---------------------------------------------------------------------------
# Real Feature-History Construction
# ---------------------------------------------------------------------------


class TestFeatureHistoryConstruction:
    """Validate complete construction using the real feature extractors."""

    def test_build_returns_dataframe(self) -> None:
        """The public builder should return a pandas DataFrame."""

        snapshots = make_snapshots(30)
        result = FeatureHistoryBuilder().build(snapshots)

        assert isinstance(result, pd.DataFrame)

    def test_build_has_exact_required_schema(self) -> None:
        """The public output must expose the stable Phase 2 schema."""

        result = FeatureHistoryBuilder().build(make_snapshots(30))

        assert list(result.columns) == REQUIRED_HISTORY_COLUMNS

    def test_build_has_expected_feature_count(self) -> None:
        """The history should contain timestamp, price, and 14 NVIDIA features."""

        result = FeatureHistoryBuilder().build(make_snapshots(30))

        assert result.shape[1] == 16
        assert len(NVIDIA_FEATURES) == 14

    def test_build_preserves_snapshot_count_when_valid(self) -> None:
        """A valid extractor pipeline should produce one row per snapshot."""

        snapshots = make_snapshots(30)
        result = FeatureHistoryBuilder().build(snapshots)

        assert len(result) == len(snapshots)

    def test_build_is_chronologically_ordered(self) -> None:
        """Output timestamps must be monotonically increasing."""

        result = FeatureHistoryBuilder().build(make_snapshots(30))

        assert result["timestamp"].is_monotonic_increasing

    def test_build_has_unique_timestamps(self) -> None:
        """Default output should contain one row per unique timestamp."""

        result = FeatureHistoryBuilder().build(make_snapshots(30))

        assert not result["timestamp"].duplicated().any()

    def test_build_contains_no_missing_required_values(self) -> None:
        """Default validation should produce finite required values."""

        result = FeatureHistoryBuilder().build(make_snapshots(30))

        assert not result[REQUIRED_HISTORY_COLUMNS].isna().any().any()

    def test_build_contains_only_finite_numeric_features(self) -> None:
        """Required numeric columns should contain finite values."""

        result = FeatureHistoryBuilder().build(make_snapshots(30))
        numeric = result.drop(columns=["timestamp"]).to_numpy(dtype=float)

        assert np.isfinite(numeric).all()

    def test_build_does_not_expose_uncontracted_columns(self) -> None:
        """The returned schema should remain limited to the stable contract."""

        result = FeatureHistoryBuilder().build(make_snapshots(30))

        assert set(result.columns) == set(REQUIRED_HISTORY_COLUMNS)

    def test_build_preserves_input_snapshots(self) -> None:
        """Feature construction must not mutate the supplied snapshot stream."""

        snapshots = make_snapshots(30)
        before = [
            (
                snapshot.timestamp,
                snapshot.sequence_id,
                list(snapshot.bids),
                list(snapshot.asks),
                list(snapshot.recent_trades),
            )
            for snapshot in snapshots
        ]

        FeatureHistoryBuilder().build(snapshots)

        after = [
            (
                snapshot.timestamp,
                snapshot.sequence_id,
                list(snapshot.bids),
                list(snapshot.asks),
                list(snapshot.recent_trades),
            )
            for snapshot in snapshots
        ]

        assert after == before


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------


class TestDeterminism:
    """Validate deterministic feature-history construction."""

    def test_same_snapshots_produce_same_history(self) -> None:
        """Repeated builds from identical snapshots should match exactly."""

        snapshots = make_snapshots(30)

        first = FeatureHistoryBuilder().build(snapshots)
        second = FeatureHistoryBuilder().build(snapshots)

        pd.testing.assert_frame_equal(first, second)

    def test_generator_seed_produces_reproducible_history(self) -> None:
        """A fixed generator seed should reproduce the complete history."""

        generator_one = MarketDataGenerator(seed=42)
        generator_two = MarketDataGenerator(seed=42)

        snapshots_one = generator_one.generate_scenario_stream(
            num_ticks=40,
        )
        snapshots_two = generator_two.generate_scenario_stream(
            num_ticks=40,
        )

        history_one = FeatureHistoryBuilder().build(snapshots_one)
        history_two = FeatureHistoryBuilder().build(snapshots_two)

        pd.testing.assert_frame_equal(history_one, history_two)


# ---------------------------------------------------------------------------
# Sorting and Chronology Configuration
# ---------------------------------------------------------------------------


class TestChronologyConfiguration:
    """Validate chronology-related configuration behavior."""

    def test_sort_output_false_preserves_already_valid_input_order(self) -> None:
        """Disabling final sorting must not disturb valid chronological input."""

        snapshots = make_snapshots(10)
        config = FeatureHistoryConfig(sort_output=False)

        result = FeatureHistoryBuilder(config=config).build(snapshots)

        assert result["timestamp"].is_monotonic_increasing

    def test_require_chronological_order_false_allows_unsorted_input_when_safe(self) -> None:
        """Relaxing input chronology allows extractors to receive unsorted snapshots."""

        snapshots = make_snapshots(10)
        snapshots[2], snapshots[7] = snapshots[7], snapshots[2]

        config = FeatureHistoryConfig(
            require_chronological_order=False,
            sort_output=True,
        )

        result = FeatureHistoryBuilder(config=config).build(snapshots)

        assert result["timestamp"].is_monotonic_increasing
        assert len(result) == len(snapshots)

    def test_unique_timestamp_requirement_can_be_disabled(self) -> None:
        """Duplicate timestamps can be permitted explicitly by configuration."""

        snapshots = make_snapshots(4)
        duplicate_timestamp = snapshots[1].timestamp
        snapshots[2].timestamp = duplicate_timestamp

        config = FeatureHistoryConfig(
            require_unique_timestamps=False,
        )

        with pytest.raises(ValueError, match="duplicate timestamps"):
            FeatureHistoryBuilder(config=config).build(snapshots)


# ---------------------------------------------------------------------------
# Extractor Output Validation
# ---------------------------------------------------------------------------


class TestExtractorValidation:
    """Validate defensive checks around individual extractor outputs."""

    def test_extractor_returning_none_raises(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """None extractor output must be rejected clearly."""

        snapshots = make_snapshots(5)
        monkeypatch.setattr(
            feature_history_module,
            "extract_order_book_features",
            lambda _: None,
        )

        with pytest.raises(ValueError, match="returned None"):
            FeatureHistoryBuilder().build(snapshots)

    def test_extractor_returning_non_dataframe_raises(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Non-DataFrame extractor output must be rejected."""

        snapshots = make_snapshots(5)
        monkeypatch.setattr(
            feature_history_module,
            "extract_order_book_features",
            lambda _: [],
        )

        with pytest.raises(TypeError, match="must return a pandas DataFrame"):
            FeatureHistoryBuilder().build(snapshots)

    def test_extractor_missing_timestamp_raises(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Extractor frames must contain a timestamp column."""

        snapshots = make_snapshots(5)
        frame = pd.DataFrame({"mid_price": [100.0] * 5})

        monkeypatch.setattr(
            feature_history_module,
            "extract_order_book_features",
            lambda _: frame.copy(),
        )

        with pytest.raises(ValueError, match="missing the 'timestamp' column"):
            FeatureHistoryBuilder().build(snapshots)

    def test_extractor_duplicate_timestamps_raise(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Individual extractor frames must not contain duplicate timestamps."""

        snapshots = make_snapshots(5)
        frame = make_feature_frame(snapshots)
        frame.loc[1, "timestamp"] = frame.loc[0, "timestamp"]

        monkeypatch.setattr(
            feature_history_module,
            "extract_order_book_features",
            lambda _: frame.copy(),
        )

        with pytest.raises(ValueError, match="duplicate timestamps"):
            FeatureHistoryBuilder().build(snapshots)


# ---------------------------------------------------------------------------
# Missing and Non-Finite Feature Handling
# ---------------------------------------------------------------------------


class TestFeatureValueValidation:
    """Validate missing and non-finite feature handling policies."""

    def test_missing_required_value_raises_by_default(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Default configuration must reject missing required features."""

        snapshots = make_snapshots(5)
        frames = patch_extractors(monkeypatch, snapshots)
        frames["spread"].loc[2, "spread_bps"] = np.nan

        monkeypatch.setattr(
            feature_history_module,
            "extract_spread_features",
            lambda _: frames["spread"].copy(),
        )

        with pytest.raises(ValueError, match="missing required values"):
            FeatureHistoryBuilder().build(snapshots)

    def test_missing_values_can_be_allowed(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Explicit missing-value allowance should retain NaN values."""

        snapshots = make_snapshots(5)
        frames = patch_extractors(monkeypatch, snapshots)
        frames["spread"].loc[2, "spread_bps"] = np.nan

        monkeypatch.setattr(
            feature_history_module,
            "extract_spread_features",
            lambda _: frames["spread"].copy(),
        )

        config = FeatureHistoryConfig(
            allow_missing_values=True,
        )

        result = FeatureHistoryBuilder(config=config).build(snapshots)

        assert pd.isna(result.loc[2, "spread_bps"])

    def test_missing_values_can_be_dropped(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Explicit row-dropping should remove rows with invalid required features."""

        snapshots = make_snapshots(5)
        frames = patch_extractors(monkeypatch, snapshots)
        frames["spread"].loc[2, "spread_bps"] = np.nan

        monkeypatch.setattr(
            feature_history_module,
            "extract_spread_features",
            lambda _: frames["spread"].copy(),
        )

        config = FeatureHistoryConfig(
            drop_invalid_rows=True,
        )

        result = FeatureHistoryBuilder(config=config).build(snapshots)

        assert len(result) == 4
        assert not result[REQUIRED_HISTORY_COLUMNS].isna().any().any()

    @pytest.mark.parametrize("value", [np.inf, -np.inf])
    def test_infinite_required_value_raises_by_default(
        self,
        monkeypatch: pytest.MonkeyPatch,
        value: float,
    ) -> None:
        """Infinite required feature values must be rejected by default."""

        snapshots = make_snapshots(5)
        frames = patch_extractors(monkeypatch, snapshots)
        frames["spread"].loc[2, "spread_bps"] = value

        monkeypatch.setattr(
            feature_history_module,
            "extract_spread_features",
            lambda _: frames["spread"].copy(),
        )

        with pytest.raises(ValueError, match="infinite or non-finite"):
            FeatureHistoryBuilder().build(snapshots)

    def test_drop_invalid_rows_does_not_leave_empty_history(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Dropping all invalid rows must still fail rather than return an empty history."""

        snapshots = make_snapshots(4)
        frames = patch_extractors(monkeypatch, snapshots)
        frames["spread"]["spread_bps"] = np.nan

        monkeypatch.setattr(
            feature_history_module,
            "extract_spread_features",
            lambda _: frames["spread"].copy(),
        )

        config = FeatureHistoryConfig(
            drop_invalid_rows=True,
        )

        with pytest.raises(ValueError, match="no valid rows"):
            FeatureHistoryBuilder(config=config).build(snapshots)


# ---------------------------------------------------------------------------
# Merge and Schema Integrity
# ---------------------------------------------------------------------------


class TestMergeIntegrity:
    """Validate timestamp-aligned merging and stable output schema."""

    def test_all_extractor_families_are_called(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Every required extractor family must participate in construction."""

        snapshots = make_snapshots(5)
        frames = make_extractor_frames(snapshots)
        calls = {name: 0 for name in frames}

        for name, frame in frames.items():
            extractor_name = f"extract_{name}_features"

            def factory(
                frame: pd.DataFrame,
                name: str,
            ):
                def extractor(
                    _: List[MarketSnapshot],
                ) -> pd.DataFrame:
                    calls[name] += 1
                    return frame.copy()

                return extractor

            monkeypatch.setattr(
                feature_history_module,
                extractor_name,
                factory(frame, name),
            )

        result = FeatureHistoryBuilder().build(snapshots)

        assert len(result) == 5
        assert calls == {name: 1 for name in frames}

    def test_missing_timestamp_in_one_merge_frame_is_rejected(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Missing timestamps in an extractor should be rejected before merging."""

        snapshots = make_snapshots(5)
        frames = patch_extractors(monkeypatch, snapshots)
        frames["spread"] = frames["spread"].drop(columns=["timestamp"])

        monkeypatch.setattr(
            feature_history_module,
            "extract_spread_features",
            lambda _: frames["spread"].copy(),
        )

        with pytest.raises(ValueError, match="missing the 'timestamp' column"):
            FeatureHistoryBuilder().build(snapshots)

    def test_disjoint_extractor_timestamps_produce_empty_merge_error(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Extractor timestamp misalignment must not silently create invalid history."""

        snapshots = make_snapshots(5)
        frames = patch_extractors(monkeypatch, snapshots)
        frames["spread"]["timestamp"] += 10_000.0

        monkeypatch.setattr(
            feature_history_module,
            "extract_spread_features",
            lambda _: frames["spread"].copy(),
        )

        with pytest.raises(ValueError, match="empty merged history"):
            FeatureHistoryBuilder().build(snapshots)

    def test_output_column_order_is_stable(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Internal extractor column order must not change the public schema."""

        snapshots = make_snapshots(5)
        frames = make_extractor_frames(snapshots)

        for name in frames:
            frames[name] = frames[name].sample(
                frac=1.0,
                axis=1,
                random_state=42,
            )

        for name, frame in frames.items():
            monkeypatch.setattr(
                feature_history_module,
                f"extract_{name}_features",
                lambda _, frame=frame: frame.copy(),
            )

        result = FeatureHistoryBuilder().build(snapshots)

        assert list(result.columns) == REQUIRED_HISTORY_COLUMNS


# ---------------------------------------------------------------------------
# Convenience APIs
# ---------------------------------------------------------------------------


class TestConvenienceAPIs:
    """Validate module-level convenience functions."""

    def test_build_feature_history_matches_builder(self) -> None:
        """Convenience function should match direct builder construction."""

        snapshots = make_snapshots(20)

        direct = FeatureHistoryBuilder().build(snapshots)
        convenience = build_feature_history(snapshots)

        pd.testing.assert_frame_equal(direct, convenience)

    def test_build_feature_history_accepts_configuration(self) -> None:
        """Convenience function should forward configuration to the builder."""

        snapshots = make_snapshots(10)
        config = FeatureHistoryConfig(
            require_chronological_order=True,
            sort_output=True,
        )

        result = build_feature_history(
            snapshots,
            config=config,
        )

        assert list(result.columns) == REQUIRED_HISTORY_COLUMNS
        assert result["timestamp"].is_monotonic_increasing

    def test_builder_static_feature_names_match_module_helper(self) -> None:
        """Builder helper and module helper should expose the same feature contract."""

        builder_names = FeatureHistoryBuilder.get_feature_names()
        module_names = get_feature_names()

        assert builder_names == module_names

    def test_builder_static_required_columns_match_module_helper(self) -> None:
        """Builder schema helper and module helper should match."""

        builder_columns = FeatureHistoryBuilder.get_required_columns()
        module_columns = get_required_columns()

        assert builder_columns == module_columns


# ---------------------------------------------------------------------------
# Synthetic Generator Integration
# ---------------------------------------------------------------------------


class TestGeneratorIntegration:
    """Validate compatibility with the project's synthetic market stream."""

    def test_generated_stream_is_accepted(self) -> None:
        """A normal synthetic market stream should build successfully."""

        generator = MarketDataGenerator(seed=42)
        snapshots = generator.generate_scenario_stream(
            num_ticks=50,
        )

        result = FeatureHistoryBuilder().build(snapshots)

        assert len(result) == 50
        assert list(result.columns) == REQUIRED_HISTORY_COLUMNS

    def test_generated_stream_timestamps_match_history(self) -> None:
        """History timestamps should correspond exactly to snapshot timestamps."""

        generator = MarketDataGenerator(seed=42)
        snapshots = generator.generate_scenario_stream(
            num_ticks=25,
        )

        result = FeatureHistoryBuilder().build(snapshots)
        expected = np.asarray(
            [snapshot.timestamp for snapshot in snapshots],
            dtype=float,
        )

        np.testing.assert_array_equal(
            result["timestamp"].to_numpy(dtype=float),
            expected,
        )

    def test_generated_stream_contains_causal_trades(self) -> None:
        """Generated trades must satisfy the feature-history causal invariant."""

        generator = MarketDataGenerator(seed=42)
        snapshots = generator.generate_scenario_stream(
            num_ticks=25,
        )

        for snapshot in snapshots:
            assert all(
                trade.timestamp <= snapshot.timestamp
                for trade in snapshot.recent_trades
            )

        result = FeatureHistoryBuilder().build(snapshots)

        assert not result.empty


# ---------------------------------------------------------------------------
# Configuration Isolation
# ---------------------------------------------------------------------------


class TestConfigurationIsolation:
    """Validate that builder configuration does not leak across instances."""

    def test_two_builders_have_independent_default_configuration(self) -> None:
        """Default builders should not share mutable configuration state."""

        first = FeatureHistoryBuilder()
        second = FeatureHistoryBuilder()

        assert first.config is not second.config
        assert first.config == second.config

    def test_custom_configuration_does_not_modify_default_builder(self) -> None:
        """Custom configuration must remain isolated from default instances."""

        custom = FeatureHistoryConfig(
            allow_missing_values=True,
        )

        custom_builder = FeatureHistoryBuilder(config=custom)
        default_builder = FeatureHistoryBuilder()

        assert custom_builder.config.allow_missing_values is True
        assert default_builder.config.allow_missing_values is False

    def test_configuration_replace_can_create_variant(self) -> None:
        """Dataclass configuration should support safe derived configurations."""

        base = FeatureHistoryConfig()
        variant = replace(
            base,
            sort_output=False,
        )

        assert base.sort_output is True
        assert variant.sort_output is False

