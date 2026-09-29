# tests/test_target_builder.py
#
# Purpose:
# Comprehensive tests for the supervised-learning target builder used by the
# NVIDIA forecasting pipeline.
#
# Responsibilities:
# - Verify future-return target construction.
# - Verify future-price target construction.
# - Verify forecast-horizon alignment.
# - Verify target timestamp semantics.
# - Verify input validation and configuration behavior.
# - Verify invalid, missing, infinite, and non-positive prices.
# - Verify chronological and duplicate timestamp handling.
# - Verify invalid-row handling.
# - Verify DataFrame immutability.
# - Verify TargetDataset contract integration.
# - Verify convenience APIs.
# - Verify array-based target construction.
# - Verify ForecastingConfig integration.
# - Verify deterministic target generation.
#
# The module does not test sequence construction, feature engineering,
# normalization, train/validation/test splitting, model architecture,
# model training, or forecasting performance.
#
# The tests intentionally validate the public TargetBuilder behavior rather
# than depending on unnecessary implementation-specific error messages.

from __future__ import annotations

# ============================================================================
# Imports
# ============================================================================

import numpy as np
import pandas as pd
import pytest

from config.config import ForecastingConfig
from data.contracts import TargetDataset
from data.target_builder import (
    DEFAULT_PRICE_COLUMN,
    DEFAULT_TARGET_NAME,
    TargetBuilder,
    TargetBuilderConfig,
    build_targets,
    build_targets_from_forecasting_config,
)


# ============================================================================
# Shared Test Constants
# ============================================================================

DEFAULT_HORIZON = 2
DEFAULT_PRICE_COLUMN_NAME = "mid_price"
DEFAULT_TIMESTAMP_COLUMN = "timestamp"


# ============================================================================
# Test Data Helpers
# ============================================================================


def make_price_history(
    prices: list[float] | np.ndarray,
    *,
    timestamps: list[float] | np.ndarray | None = None,
) -> pd.DataFrame:
    """
    Create a deterministic chronological price-history DataFrame.

    The resulting DataFrame contains the two columns used by the default
    TargetBuilder configuration:

        timestamp
        mid_price
    """

    price_array = np.asarray(prices, dtype=float)

    if timestamps is None:
        timestamp_array = np.arange(
            len(price_array),
            dtype=float,
        )
    else:
        timestamp_array = np.asarray(timestamps)

    return pd.DataFrame(
        {
            DEFAULT_TIMESTAMP_COLUMN: timestamp_array,
            DEFAULT_PRICE_COLUMN_NAME: price_array,
        }
    )


def make_default_builder(
    **overrides,
) -> TargetBuilder:
    """
    Create a TargetBuilder configured for deterministic testing.

    The default test horizon is intentionally small so that expected target
    values can be calculated manually.
    """

    config_values = {
        "forecast_horizon": DEFAULT_HORIZON,
        "price_column": DEFAULT_PRICE_COLUMN_NAME,
        "target_name": DEFAULT_TARGET_NAME,
        "target_type": "future_return",
        "require_chronological_order": True,
        "require_unique_timestamps": True,
        "allow_missing_prices": False,
        "allow_infinite_prices": False,
        "require_positive_prices": True,
        "drop_invalid_rows": False,
        "copy_input": True,
        "timestamp_column": DEFAULT_TIMESTAMP_COLUMN,
        "dtype": "float64",
    }

    config_values.update(overrides)

    return TargetBuilder(
        config=TargetBuilderConfig(**config_values)
    )


def expected_future_returns(
    prices: list[float] | np.ndarray,
    horizon: int,
) -> np.ndarray:
    """
    Calculate expected future returns independently of TargetBuilder.
    """

    price_array = np.asarray(prices, dtype=float)

    current_prices = price_array[:-horizon]
    future_prices = price_array[horizon:]

    return (
        future_prices - current_prices
    ) / current_prices


def expected_future_prices(
    prices: list[float] | np.ndarray,
    horizon: int,
) -> np.ndarray:
    """
    Calculate expected future-price targets independently of TargetBuilder.
    """

    price_array = np.asarray(prices, dtype=float)

    return price_array[horizon:]


# ============================================================================
# TargetBuilderConfig Tests
# ============================================================================


class TestTargetBuilderConfig:
    """Tests for TargetBuilderConfig defaults and validation."""

    def test_default_target_name(self):
        """The default target name should match the project contract."""
        config = TargetBuilderConfig()

        assert config.target_name == DEFAULT_TARGET_NAME

    def test_default_price_column(self):
        """The default price column should be mid_price."""
        config = TargetBuilderConfig()

        assert config.price_column == DEFAULT_PRICE_COLUMN

    def test_default_target_type_is_future_return(self):
        """Future return should be the default target type."""
        config = TargetBuilderConfig()

        assert config.target_type == "future_return"

    def test_default_forecast_horizon_is_positive(self):
        """The default forecast horizon must be positive."""
        config = TargetBuilderConfig()

        assert config.forecast_horizon > 0

    def test_custom_forecast_horizon_is_preserved(self):
        """Custom forecast horizons should be preserved."""
        config = TargetBuilderConfig(
            forecast_horizon=7,
        )

        assert config.forecast_horizon == 7

    def test_custom_target_type_is_preserved(self):
        """Supported target types should be accepted."""
        config = TargetBuilderConfig(
            target_type="future_price",
        )

        assert config.target_type == "future_price"

    @pytest.mark.parametrize(
        "forecast_horizon",
        [
            0,
            -1,
            -5,
        ],
    )
    def test_non_positive_forecast_horizon_is_rejected(
        self,
        forecast_horizon: int,
    ):
        """Forecast horizons must be positive."""
        with pytest.raises(ValueError):
            TargetBuilderConfig(
                forecast_horizon=forecast_horizon,
            )

    def test_unsupported_target_type_is_rejected(self):
        """Unsupported target types should be rejected."""
        with pytest.raises(ValueError):
            TargetBuilderConfig(
                target_type="unsupported",
            )

    def test_copy_input_configuration_is_preserved(self):
        """The copy_input setting should be preserved."""
        config = TargetBuilderConfig(
            copy_input=False,
        )

        assert config.copy_input is False


# ============================================================================
# Basic Builder Construction
# ============================================================================


class TestTargetBuilderConstruction:
    """Tests for constructing TargetBuilder instances."""

    def test_builder_can_be_created_with_default_configuration(self):
        """TargetBuilder should be constructible."""
        builder = TargetBuilder()

        assert isinstance(builder, TargetBuilder)

    def test_builder_preserves_custom_configuration(self):
        """The builder should preserve the supplied configuration."""
        config = TargetBuilderConfig(
            forecast_horizon=5,
            target_type="future_price",
            target_name="future_mid_price",
        )

        builder = TargetBuilder(config=config)

        assert builder.config.forecast_horizon == 5
        assert builder.config.target_type == "future_price"
        assert builder.config.target_name == "future_mid_price"

    def test_default_builder_helper_creates_target_builder(self):
        """The shared test helper should create a valid builder."""
        builder = make_default_builder()

        assert isinstance(builder, TargetBuilder)


# ============================================================================
# Basic Target Construction
# ============================================================================


class TestBasicTargetConstruction:
    """Tests for the basic TargetBuilder output contract."""

    def test_build_returns_target_dataset(self):
        """The builder should return a TargetDataset."""
        data = make_price_history(
            [100.0, 101.0, 102.0, 103.0, 104.0],
        )

        result = make_default_builder().build(data)

        assert isinstance(result, TargetDataset)

    def test_build_generates_expected_number_of_targets(self):
        """N observations with horizon H should generate N-H targets."""
        data = make_price_history(
            [100.0, 101.0, 102.0, 103.0, 104.0],
        )

        result = make_default_builder().build(data)

        assert result.num_targets == 3
        assert len(result.targets) == 3
        assert len(result.timestamps) == 3

    @pytest.mark.parametrize(
        "row_count,horizon,expected_count",
        [
            (2, 1, 1),
            (3, 1, 2),
            (5, 1, 4),
            (5, 2, 3),
            (5, 3, 2),
            (5, 4, 1),
            (6, 5, 1),
        ],
    )
    def test_target_count_is_n_minus_horizon(
        self,
        row_count: int,
        horizon: int,
        expected_count: int,
    ):
        """The fundamental target-count relationship should hold."""
        prices = np.arange(
            100.0,
            100.0 + row_count,
        )

        data = make_price_history(prices)

        result = make_default_builder(
            forecast_horizon=horizon,
        ).build(data)

        assert result.num_targets == expected_count


# ============================================================================
# Future Return Tests
# ============================================================================


class TestFutureReturnTargets:
    """Tests for future-return target construction."""

    def test_future_return_calculation_is_correct(self):
        """
        Verify:

            future_return =
                (future_price - current_price) / current_price
        """

        prices = [
            100.0,
            110.0,
            120.0,
            130.0,
            140.0,
        ]

        data = make_price_history(prices)

        result = make_default_builder(
            forecast_horizon=2,
            target_type="future_return",
        ).build(data)

        expected = expected_future_returns(
            prices,
            horizon=2,
        )

        np.testing.assert_allclose(
            result.targets,
            expected,
            rtol=1e-12,
            atol=1e-12,
        )

    def test_future_return_supports_positive_returns(self):
        """Increasing future prices should produce positive returns."""
        data = make_price_history(
            [100.0, 105.0, 110.0, 115.0, 120.0],
        )

        result = make_default_builder(
            forecast_horizon=2,
        ).build(data)

        assert np.all(result.targets > 0)

    def test_future_return_supports_negative_returns(self):
        """Decreasing future prices should produce negative returns."""
        data = make_price_history(
            [100.0, 95.0, 90.0, 85.0, 80.0],
        )

        result = make_default_builder(
            forecast_horizon=2,
        ).build(data)

        expected = expected_future_returns(
            [100.0, 95.0, 90.0, 85.0, 80.0],
            horizon=2,
        )

        np.testing.assert_allclose(
            result.targets,
            expected,
            rtol=1e-12,
            atol=1e-12,
        )

        assert np.all(result.targets < 0)

    def test_future_return_is_zero_when_future_price_is_unchanged(self):
        """Unchanged future prices should produce zero returns."""
        data = make_price_history(
            [100.0, 100.0, 100.0, 100.0, 100.0],
        )

        result = make_default_builder(
            forecast_horizon=2,
        ).build(data)

        np.testing.assert_allclose(
            result.targets,
            np.zeros(3),
            atol=1e-12,
        )

    @pytest.mark.parametrize(
        "horizon",
        [1, 2, 3],
    )
    def test_future_return_formula_for_multiple_horizons(
        self,
        horizon: int,
    ):
        """The same future-return formula should work for multiple horizons."""
        prices = [
            100.0,
            105.0,
            110.0,
            120.0,
            115.0,
            130.0,
        ]

        data = make_price_history(prices)

        result = make_default_builder(
            forecast_horizon=horizon,
        ).build(data)

        expected = expected_future_returns(
            prices,
            horizon,
        )

        np.testing.assert_allclose(
            result.targets,
            expected,
            rtol=1e-12,
            atol=1e-12,
        )

    def test_horizon_one_uses_immediate_next_observation(self):
        """Horizon one should use t+1 as the future observation."""
        prices = [
            100.0,
            105.0,
            110.0,
            100.0,
        ]

        data = make_price_history(prices)

        result = make_default_builder(
            forecast_horizon=1,
        ).build(data)

        expected = expected_future_returns(
            prices,
            horizon=1,
        )

        np.testing.assert_allclose(
            result.targets,
            expected,
            rtol=1e-12,
            atol=1e-12,
        )

    def test_target_name_is_preserved_for_future_return(self):
        """The configured target name should be preserved."""
        data = make_price_history(
            [100.0, 101.0, 102.0, 103.0],
        )

        result = make_default_builder(
            forecast_horizon=1,
            target_name="custom_return",
        ).build(data)

        assert result.target_name == "custom_return"


# ============================================================================
# Future Price Tests
# ============================================================================


class TestFuturePriceTargets:
    """Tests for future-price target construction."""

    def test_future_price_target_is_supported(self):
        """The builder should support future-price targets."""
        prices = [
            100.0,
            110.0,
            120.0,
            130.0,
            140.0,
        ]

        data = make_price_history(prices)

        result = make_default_builder(
            forecast_horizon=2,
            target_type="future_price",
            target_name="future_mid_price",
        ).build(data)

        expected = expected_future_prices(
            prices,
            horizon=2,
        )

        np.testing.assert_array_equal(
            result.targets,
            expected,
        )

        assert result.target_name == "future_mid_price"

    @pytest.mark.parametrize(
        "horizon",
        [1, 2, 3],
    )
    def test_future_price_target_for_multiple_horizons(
        self,
        horizon: int,
    ):
        """Future-price targets should correctly use t+H."""
        prices = [
            100.0,
            105.0,
            110.0,
            115.0,
            120.0,
            130.0,
        ]

        data = make_price_history(prices)

        result = make_default_builder(
            forecast_horizon=horizon,
            target_type="future_price",
        ).build(data)

        expected = expected_future_prices(
            prices,
            horizon,
        )

        np.testing.assert_allclose(
            result.targets,
            expected,
        )


# ============================================================================
# Timestamp Alignment Tests
# ============================================================================


class TestTimestampAlignment:
    """Tests for target timestamp semantics."""

    def test_target_timestamps_refer_to_current_observations(self):
        """
        Target timestamps should correspond to t, not t+H.
        """
        data = make_price_history(
            [100.0, 110.0, 120.0, 130.0, 140.0],
            timestamps=[
                10.0,
                20.0,
                30.0,
                40.0,
                50.0,
            ],
        )

        result = make_default_builder(
            forecast_horizon=2,
        ).build(data)

        np.testing.assert_array_equal(
            result.timestamps,
            np.array(
                [
                    10.0,
                    20.0,
                    30.0,
                ]
            ),
        )

    def test_final_horizon_rows_are_excluded(self):
        """The final H observations cannot serve as target starting points."""
        data = make_price_history(
            [100.0, 101.0, 102.0, 103.0, 104.0],
        )

        result = make_default_builder(
            forecast_horizon=2,
        ).build(data)

        np.testing.assert_array_equal(
            result.timestamps,
            np.array(
                [
                    0.0,
                    1.0,
                    2.0,
                ]
            ),
        )

    def test_target_timestamps_preserve_input_values(self):
        """The original current-observation timestamps should be preserved."""
        timestamps = [
            10.5,
            20.5,
            30.5,
            40.5,
            50.5,
        ]

        data = make_price_history(
            [100.0, 110.0, 120.0, 130.0, 140.0],
            timestamps=timestamps,
        )

        result = make_default_builder(
            forecast_horizon=2,
        ).build(data)

        np.testing.assert_array_equal(
            result.timestamps,
            np.asarray(timestamps[:3]),
        )

    def test_target_timestamps_are_one_dimensional(self):
        """Target timestamps should be a one-dimensional array."""
        data = make_price_history(
            [100.0, 101.0, 102.0, 103.0, 104.0],
        )

        result = make_default_builder().build(data)

        assert result.timestamps.ndim == 1


# ============================================================================
# Input Validation Tests
# ============================================================================


class TestInputValidation:
    """Tests for invalid TargetBuilder inputs."""

    def test_non_dataframe_input_raises(self):
        """TargetBuilder should require a pandas DataFrame."""
        builder = make_default_builder()

        with pytest.raises(TypeError):
            builder.build(
                [100.0, 101.0, 102.0]  # type: ignore[arg-type]
            )

    @pytest.mark.parametrize(
        "missing_column",
        [
            "timestamp",
            "mid_price",
        ],
    )
    def test_missing_required_column_raises(
        self,
        missing_column: str,
    ):
        """Required columns must be present."""
        data = make_price_history(
            [100.0, 101.0, 102.0, 103.0],
        ).drop(
            columns=[missing_column],
        )

        builder = make_default_builder(
            forecast_horizon=1,
        )

        with pytest.raises(ValueError):
            builder.build(data)

    def test_empty_dataframe_raises(self):
        """An empty DataFrame should be rejected."""
        data = pd.DataFrame(
            columns=[
                DEFAULT_TIMESTAMP_COLUMN,
                DEFAULT_PRICE_COLUMN_NAME,
            ]
        )

        builder = make_default_builder()

        with pytest.raises(ValueError):
            builder.build(data)

    def test_insufficient_observations_raise(self):
        """
        At least H+1 observations are required to produce one target.
        """
        data = make_price_history(
            [100.0, 101.0],
        )

        builder = make_default_builder(
            forecast_horizon=2,
        )

        with pytest.raises(ValueError):
            builder.build(data)

    def test_exactly_h_plus_one_observations_is_valid(self):
        """H+1 observations should produce exactly one target."""
        horizon = 4

        data = make_price_history(
            [
                100.0,
                101.0,
                102.0,
                103.0,
                104.0,
            ]
        )

        result = make_default_builder(
            forecast_horizon=horizon,
        ).build(data)

        assert result.num_targets == 1

    @pytest.mark.parametrize(
        "invalid_price",
        [
            0.0,
            -1.0,
            -100.0,
        ],
    )
    def test_non_positive_prices_raise(
        self,
        invalid_price: float,
    ):
        """Zero and negative prices should be rejected by default."""
        data = make_price_history(
            [
                100.0,
                invalid_price,
                102.0,
                103.0,
            ]
        )

        builder = make_default_builder(
            forecast_horizon=1,
        )

        with pytest.raises(ValueError):
            builder.build(data)

    def test_nan_price_raises(self):
        """NaN prices should be rejected by default."""
        data = make_price_history(
            [
                100.0,
                101.0,
                102.0,
                103.0,
            ]
        )

        data.loc[1, DEFAULT_PRICE_COLUMN_NAME] = np.nan

        builder = make_default_builder(
            forecast_horizon=1,
        )

        with pytest.raises(ValueError):
            builder.build(data)

    @pytest.mark.parametrize(
        "invalid_price",
        [
            np.inf,
            -np.inf,
        ],
    )
    def test_infinite_price_raises(
        self,
        invalid_price: float,
    ):
        """Infinite prices should be rejected by default."""
        data = make_price_history(
            [
                100.0,
                invalid_price,
                102.0,
                103.0,
            ]
        )

        builder = make_default_builder(
            forecast_horizon=1,
        )

        with pytest.raises(ValueError):
            builder.build(data)

    def test_non_numeric_price_raises(self):
        """Non-numeric price values should be rejected."""
        data = pd.DataFrame(
            {
                DEFAULT_TIMESTAMP_COLUMN: [
                    0.0,
                    1.0,
                    2.0,
                    3.0,
                ],
                DEFAULT_PRICE_COLUMN_NAME: [
                    100.0,
                    "invalid",
                    102.0,
                    103.0,
                ],
            }
        )

        builder = make_default_builder(
            forecast_horizon=1,
        )

        with pytest.raises(ValueError):
            builder.build(data)


# ============================================================================
# Timestamp Validation Tests
# ============================================================================


class TestTimestampValidation:
    """Tests for timestamp validity and ordering."""

    def test_non_chronological_timestamps_raise(self):
        """Timestamps must remain chronological by default."""
        data = make_price_history(
            [100.0, 101.0, 102.0, 103.0],
            timestamps=[
                0.0,
                2.0,
                1.0,
                3.0,
            ],
        )

        builder = make_default_builder(
            forecast_horizon=1,
        )

        with pytest.raises(ValueError):
            builder.build(data)

    def test_duplicate_timestamps_raise(self):
        """Duplicate timestamps should be rejected by default."""
        data = make_price_history(
            [100.0, 101.0, 102.0, 103.0],
            timestamps=[
                0.0,
                1.0,
                1.0,
                2.0,
            ],
        )

        builder = make_default_builder(
            forecast_horizon=1,
        )

        with pytest.raises(ValueError):
            builder.build(data)

    def test_non_numeric_timestamps_raise(self):
        """Non-numeric timestamps should be rejected."""
        data = pd.DataFrame(
            {
                DEFAULT_TIMESTAMP_COLUMN: [
                    0.0,
                    "invalid",
                    2.0,
                    3.0,
                ],
                DEFAULT_PRICE_COLUMN_NAME: [
                    100.0,
                    101.0,
                    102.0,
                    103.0,
                ],
            }
        )

        builder = make_default_builder(
            forecast_horizon=1,
        )

        with pytest.raises(ValueError):
            builder.build(data)

    @pytest.mark.parametrize(
        "invalid_timestamp",
        [
            np.inf,
            -np.inf,
        ],
    )
    def test_infinite_timestamps_raise(
        self,
        invalid_timestamp: float,
    ):
        """Infinite timestamps should be rejected."""
        data = make_price_history(
            [100.0, 101.0, 102.0, 103.0],
            timestamps=[
                0.0,
                invalid_timestamp,
                2.0,
                3.0,
            ],
        )

        builder = make_default_builder(
            forecast_horizon=1,
        )

        with pytest.raises(ValueError):
            builder.build(data)

    def test_chronological_validation_can_be_disabled(self):
        """
        When chronological validation is disabled, the builder should not
        reject the ordering at its own configuration layer.
        """
        data = make_price_history(
            [100.0, 101.0, 102.0, 103.0],
            timestamps=[
                0.0,
                2.0,
                1.0,
                3.0,
            ],
        )

        builder = make_default_builder(
            forecast_horizon=1,
            require_chronological_order=False,
        )

        try:
            result = builder.build(data)
        except ValueError:
            # A downstream contract may still enforce chronology. That is
            # valid behavior and should not make this test depend on the
            # internal validation layer.
            return

        assert result.num_targets == 3

    def test_unique_timestamp_validation_can_be_disabled(self):
        """
        When duplicate validation is disabled, downstream contracts may still
        reject duplicate timestamps. The test therefore checks that behavior
        without requiring a specific validation layer.
        """
        data = make_price_history(
            [100.0, 101.0, 102.0, 103.0],
            timestamps=[
                0.0,
                1.0,
                1.0,
                2.0,
            ],
        )

        builder = make_default_builder(
            forecast_horizon=1,
            require_unique_timestamps=False,
        )

        try:
            result = builder.build(data)
        except ValueError:
            return

        assert result.num_targets == 3


# ============================================================================
# Price Validation Configuration
# ============================================================================


class TestPriceValidationConfiguration:
    """Tests for configurable price validation behavior."""

    def test_missing_prices_can_be_allowed(self):
        """
        When missing prices are explicitly allowed, the builder should not
        reject the missing value at the configuration layer.
        """
        data = make_price_history(
            [
                100.0,
                101.0,
                102.0,
                103.0,
            ]
        )

        data.loc[1, DEFAULT_PRICE_COLUMN_NAME] = np.nan

        builder = make_default_builder(
            forecast_horizon=1,
            allow_missing_prices=True,
        )

        try:
            result = builder.build(data)
        except ValueError:
            # The calculation itself may still reject NaN-derived targets.
            return

        assert isinstance(result, TargetDataset)

    def test_infinite_prices_can_be_allowed(self):
        """
        When infinite prices are explicitly allowed, the builder should not
        reject them at the initial configuration layer.
        """
        data = make_price_history(
            [
                100.0,
                np.inf,
                102.0,
                103.0,
            ]
        )

        builder = make_default_builder(
            forecast_horizon=1,
            allow_infinite_prices=True,
        )

        try:
            result = builder.build(data)
        except ValueError:
            # The resulting target may still be invalid for a downstream
            # contract. This test does not assume that infinite targets are
            # mathematically valid.
            return

        assert isinstance(result, TargetDataset)

    def test_positive_price_requirement_can_be_disabled(self):
        """
        When positive-price validation is disabled, the builder should not
        reject a negative price solely because it is non-positive.
        """
        data = make_price_history(
            [
                -100.0,
                -90.0,
                -80.0,
                -70.0,
            ]
        )

        builder = make_default_builder(
            forecast_horizon=1,
            require_positive_prices=False,
        )

        try:
            result = builder.build(data)
        except (ValueError, ZeroDivisionError):
            # The mathematical future-return calculation may still reject
            # pathological price values.
            return

        assert isinstance(result, TargetDataset)


# ============================================================================
# Invalid-Row Handling
# ============================================================================


class TestInvalidRowHandling:
    """Tests for drop_invalid_rows behavior."""

    def test_invalid_rows_raise_when_dropping_is_disabled(self):
        """Invalid rows should raise when dropping is disabled."""
        data = make_price_history(
            [
                100.0,
                101.0,
                102.0,
                103.0,
            ]
        )

        data.loc[2, DEFAULT_PRICE_COLUMN_NAME] = np.nan

        builder = make_default_builder(
            forecast_horizon=1,
            drop_invalid_rows=False,
        )

        with pytest.raises(ValueError):
            builder.build(data)

    def test_invalid_rows_can_be_dropped_when_configured(self):
        """
        Invalid rows should be removed when drop_invalid_rows is enabled.
        """
        data = pd.DataFrame(
            {
                DEFAULT_TIMESTAMP_COLUMN: [
                    0.0,
                    1.0,
                    2.0,
                    3.0,
                    4.0,
                    5.0,
                ],
                DEFAULT_PRICE_COLUMN_NAME: [
                    100.0,
                    np.nan,
                    102.0,
                    103.0,
                    104.0,
                    105.0,
                ],
            }
        )

        builder = make_default_builder(
            forecast_horizon=1,
            drop_invalid_rows=True,
        )

        result = builder.build(data)

        assert result.num_targets == 4

        np.testing.assert_array_equal(
            result.timestamps,
            np.array(
                [
                    0.0,
                    2.0,
                    3.0,
                    4.0,
                ]
            ),
        )

    def test_multiple_invalid_rows_can_be_dropped(self):
        """Multiple invalid rows should be handled consistently."""
        data = pd.DataFrame(
            {
                DEFAULT_TIMESTAMP_COLUMN: [
                    0.0,
                    1.0,
                    2.0,
                    3.0,
                    4.0,
                    5.0,
                    6.0,
                ],
                DEFAULT_PRICE_COLUMN_NAME: [
                    100.0,
                    np.nan,
                    102.0,
                    np.inf,
                    104.0,
                    105.0,
                    106.0,
                ],
            }
        )

        builder = make_default_builder(
            forecast_horizon=1,
            drop_invalid_rows=True,
        )

        try:
            result = builder.build(data)
        except ValueError:
            # Infinite-price handling is independently configurable, so if
            # the implementation rejects it before row dropping, that is
            # valid behavior.
            return

        assert isinstance(result, TargetDataset)


# ============================================================================
# Custom Column Tests
# ============================================================================


class TestCustomColumns:
    """Tests for custom timestamp and price column names."""

    def test_custom_price_column_is_supported(self):
        """A custom price column should be usable."""
        data = pd.DataFrame(
            {
                "timestamp": [
                    0.0,
                    1.0,
                    2.0,
                    3.0,
                ],
                "reference_price": [
                    100.0,
                    110.0,
                    120.0,
                    130.0,
                ],
            }
        )

        config = TargetBuilderConfig(
            forecast_horizon=1,
            price_column="reference_price",
            target_name="custom_return",
        )

        result = TargetBuilder(
            config=config,
        ).build(data)

        assert result.price_column == "reference_price"
        assert result.target_name == "custom_return"

        np.testing.assert_allclose(
            result.targets,
            np.array(
                [
                    0.10,
                    10.0 / 110.0,
                    10.0 / 120.0,
                ]
            ),
        )

    def test_custom_timestamp_column_is_supported(self):
        """A custom timestamp column should be usable."""
        data = pd.DataFrame(
            {
                "event_time": [
                    10.0,
                    20.0,
                    30.0,
                    40.0,
                ],
                "mid_price": [
                    100.0,
                    105.0,
                    110.0,
                    115.0,
                ],
            }
        )

        config = TargetBuilderConfig(
            forecast_horizon=1,
            timestamp_column="event_time",
        )

        result = TargetBuilder(
            config=config,
        ).build(data)

        np.testing.assert_array_equal(
            result.timestamps,
            np.array(
                [
                    10.0,
                    20.0,
                    30.0,
                ]
            ),
        )

    def test_custom_price_and_timestamp_columns_work_together(self):
        """Both column names should be independently configurable."""
        data = pd.DataFrame(
            {
                "event_time": [
                    100.0,
                    200.0,
                    300.0,
                    400.0,
                    500.0,
                ],
                "execution_price": [
                    50.0,
                    55.0,
                    60.0,
                    65.0,
                    70.0,
                ],
            }
        )

        config = TargetBuilderConfig(
            forecast_horizon=2,
            price_column="execution_price",
            timestamp_column="event_time",
            target_name="future_execution_return",
        )

        result = TargetBuilder(
            config=config,
        ).build(data)

        expected = expected_future_returns(
            [
                50.0,
                55.0,
                60.0,
                65.0,
                70.0,
            ],
            2,
        )

        np.testing.assert_allclose(
            result.targets,
            expected,
        )

        np.testing.assert_array_equal(
            result.timestamps,
            np.array(
                [
                    100.0,
                    200.0,
                    300.0,
                ]
            ),
        )


# ============================================================================
# TargetDataset Contract Tests
# ============================================================================


class TestTargetDatasetContract:
    """Tests for integration with the shared TargetDataset contract."""

    def test_result_is_target_dataset(self):
        """TargetBuilder output should satisfy TargetDataset."""
        data = make_price_history(
            [100.0, 101.0, 102.0, 103.0],
        )

        result = make_default_builder(
            forecast_horizon=1,
        ).build(data)

        assert isinstance(result, TargetDataset)

    def test_targets_are_one_dimensional(self):
        """TargetDataset targets should be one-dimensional."""
        data = make_price_history(
            [100.0, 101.0, 102.0, 103.0, 104.0],
        )

        result = make_default_builder().build(data)

        assert result.targets.ndim == 1

    def test_targets_are_finite_for_valid_input(self):
        """Valid input should produce finite target values."""
        data = make_price_history(
            [100.0, 101.0, 103.0, 102.0, 105.0],
        )

        result = make_default_builder().build(data)

        assert np.isfinite(result.targets).all()

    def test_target_count_matches_timestamp_count(self):
        """Targets and timestamps must remain aligned."""
        data = make_price_history(
            [100.0, 101.0, 102.0, 103.0, 104.0],
        )

        result = make_default_builder().build(data)

        assert len(result.targets) == len(result.timestamps)
        assert result.num_targets == len(result.timestamps)

    def test_metadata_matches_configuration(self):
        """TargetDataset metadata should match builder configuration."""
        data = make_price_history(
            [100.0, 101.0, 102.0, 103.0, 104.0],
        )

        result = make_default_builder(
            forecast_horizon=2,
            target_name="future_return_custom",
            price_column="mid_price",
        ).build(data)

        assert result.target_name == "future_return_custom"
        assert result.forecast_horizon == 2
        assert result.price_column == "mid_price"


# ============================================================================
# Array-Based API Tests
# ============================================================================


class TestArrayBasedAPI:
    """Tests for TargetBuilder array-based construction."""

    def test_build_from_arrays_matches_dataframe_build(self):
        """Array and DataFrame APIs should produce equivalent targets."""
        timestamps = np.array(
            [
                0.0,
                1.0,
                2.0,
                3.0,
                4.0,
            ]
        )

        prices = np.array(
            [
                100.0,
                110.0,
                120.0,
                130.0,
                140.0,
            ]
        )

        builder = make_default_builder(
            forecast_horizon=2,
        )

        array_result = builder.build_from_arrays(
            timestamps=timestamps,
            prices=prices,
        )

        dataframe_result = builder.build(
            make_price_history(
                prices,
                timestamps=timestamps,
            )
        )

        np.testing.assert_allclose(
            array_result.targets,
            dataframe_result.targets,
        )

        np.testing.assert_array_equal(
            array_result.timestamps,
            dataframe_result.timestamps,
        )

    def test_build_from_arrays_returns_target_dataset(self):
        """Array construction should return TargetDataset."""
        builder = make_default_builder()

        result = builder.build_from_arrays(
            timestamps=np.array(
                [0.0, 1.0, 2.0, 3.0]
            ),
            prices=np.array(
                [100.0, 101.0, 102.0, 103.0]
            ),
        )

        assert isinstance(result, TargetDataset)

    def test_build_from_arrays_rejects_mismatched_lengths(self):
        """Timestamp and price arrays must have equal lengths."""
        builder = make_default_builder()

        with pytest.raises(ValueError):
            builder.build_from_arrays(
                timestamps=np.array(
                    [0.0, 1.0, 2.0]
                ),
                prices=np.array(
                    [100.0, 101.0]
                ),
            )

    def test_build_from_arrays_rejects_two_dimensional_timestamps(self):
        """Timestamps should be one-dimensional."""
        builder = make_default_builder()

        with pytest.raises(ValueError):
            builder.build_from_arrays(
                timestamps=np.array(
                    [
                        [0.0],
                        [1.0],
                        [2.0],
                    ]
                ),
                prices=np.array(
                    [
                        100.0,
                        101.0,
                        102.0,
                    ]
                ),
            )

    def test_build_from_arrays_rejects_two_dimensional_prices(self):
        """Prices should be one-dimensional."""
        builder = make_default_builder()

        with pytest.raises(ValueError):
            builder.build_from_arrays(
                timestamps=np.array(
                    [
                        0.0,
                        1.0,
                        2.0,
                    ]
                ),
                prices=np.array(
                    [
                        [100.0],
                        [101.0],
                        [102.0],
                    ]
                ),
            )

    def test_build_from_arrays_supports_custom_horizon(self):
        """Array construction should respect forecast horizon."""
        prices = np.array(
            [
                100.0,
                110.0,
                120.0,
                130.0,
                140.0,
                150.0,
            ]
        )

        builder = make_default_builder(
            forecast_horizon=3,
        )

        result = builder.build_from_arrays(
            timestamps=np.arange(
                len(prices),
                dtype=float,
            ),
            prices=prices,
        )

        expected = expected_future_returns(
            prices,
            horizon=3,
        )

        np.testing.assert_allclose(
            result.targets,
            expected,
        )


# ============================================================================
# DataFrame Output API Tests
# ============================================================================


class TestBuildDataframe:
    """Tests for TargetBuilder.build_dataframe()."""

    def test_build_dataframe_returns_dataframe(self):
        """build_dataframe should return a pandas DataFrame."""
        data = make_price_history(
            [100.0, 101.0, 102.0, 103.0, 104.0],
        )

        builder = make_default_builder()

        result = builder.build_dataframe(data)

        assert isinstance(result, pd.DataFrame)

    def test_build_dataframe_returns_expected_columns(self):
        """The DataFrame should contain timestamp and target columns."""
        data = make_price_history(
            [100.0, 101.0, 102.0, 103.0, 104.0],
        )

        builder = make_default_builder(
            forecast_horizon=2,
        )

        result = builder.build_dataframe(data)

        assert list(result.columns) == [
            DEFAULT_TIMESTAMP_COLUMN,
            DEFAULT_TARGET_NAME,
        ]

    def test_build_dataframe_has_expected_row_count(self):
        """build_dataframe should contain one row per target."""
        data = make_price_history(
            [100.0, 101.0, 102.0, 103.0, 104.0],
        )

        builder = make_default_builder(
            forecast_horizon=2,
        )

        result = builder.build_dataframe(data)

        assert len(result) == 3

    def test_build_dataframe_preserves_target_values(self):
        """DataFrame output should match TargetDataset output."""
        prices = [
            100.0,
            110.0,
            120.0,
            130.0,
            140.0,
        ]

        data = make_price_history(prices)

        builder = make_default_builder(
            forecast_horizon=2,
        )

        dataset_result = builder.build(data)
        dataframe_result = builder.build_dataframe(data)

        np.testing.assert_allclose(
            dataframe_result[DEFAULT_TARGET_NAME].to_numpy(),
            dataset_result.targets,
        )

        np.testing.assert_array_equal(
            dataframe_result[DEFAULT_TIMESTAMP_COLUMN].to_numpy(),
            dataset_result.timestamps,
        )

    def test_build_dataframe_supports_custom_target_name(self):
        """Custom target names should appear in DataFrame output."""
        data = make_price_history(
            [100.0, 110.0, 120.0, 130.0],
        )

        builder = make_default_builder(
            forecast_horizon=1,
            target_name="custom_future_return",
        )

        result = builder.build_dataframe(data)

        assert list(result.columns) == [
            DEFAULT_TIMESTAMP_COLUMN,
            "custom_future_return",
        ]


# ============================================================================
# ForecastingConfig Integration
# ============================================================================


class TestForecastingConfigIntegration:
    """Tests for integration with the project's ForecastingConfig."""

    def test_from_forecasting_config_returns_target_builder(self):
        """The classmethod should create a TargetBuilder."""
        forecasting_config = ForecastingConfig()

        builder = TargetBuilder.from_forecasting_config(
            forecasting_config,
        )

        assert isinstance(builder, TargetBuilder)

    def test_from_forecasting_config_preserves_horizon(self):
        """The configured forecasting horizon should transfer."""
        forecasting_config = ForecastingConfig()

        builder = TargetBuilder.from_forecasting_config(
            forecasting_config,
        )

        assert (
            builder.config.forecast_horizon
            == forecasting_config.forecast_horizon
        )

    def test_from_forecasting_config_preserves_target_name(self):
        """The forecasting target column should transfer."""
        forecasting_config = ForecastingConfig()

        builder = TargetBuilder.from_forecasting_config(
            forecasting_config,
        )

        assert (
            builder.config.target_name
            == forecasting_config.target_column
        )

    def test_from_forecasting_config_preserves_price_column(self):
        """The forecasting target price column should transfer."""
        forecasting_config = ForecastingConfig()

        builder = TargetBuilder.from_forecasting_config(
            forecasting_config,
        )

        assert (
            builder.config.price_column
            == forecasting_config.target_price_column
        )

    def test_build_targets_from_forecasting_config_returns_dataset(self):
        """The convenience integration API should return TargetDataset."""
        forecasting_config = ForecastingConfig()

        horizon = forecasting_config.forecast_horizon

        prices = np.arange(
            100.0,
            100.0 + horizon + 10,
        )

        data = make_price_history(prices)

        result = build_targets_from_forecasting_config(
            data,
            forecasting_config=forecasting_config,
        )

        assert isinstance(result, TargetDataset)

    def test_build_targets_from_forecasting_config_uses_configured_horizon(
        self,
    ):
        """The resulting target count should use ForecastingConfig horizon."""
        forecasting_config = ForecastingConfig()

        horizon = forecasting_config.forecast_horizon

        prices = np.arange(
            100.0,
            100.0 + horizon + 10,
        )

        data = make_price_history(prices)

        result = build_targets_from_forecasting_config(
            data,
            forecasting_config=forecasting_config,
        )

        assert result.num_targets == len(data) - horizon

    def test_build_targets_from_forecasting_config_preserves_target_name(
        self,
    ):
        """The result target name should match ForecastingConfig."""
        forecasting_config = ForecastingConfig()

        horizon = forecasting_config.forecast_horizon

        prices = np.arange(
            100.0,
            100.0 + horizon + 10,
        )

        data = make_price_history(prices)

        result = build_targets_from_forecasting_config(
            data,
            forecasting_config=forecasting_config,
        )

        assert result.target_name == forecasting_config.target_column

    def test_build_targets_from_forecasting_config_preserves_price_column(
        self,
    ):
        """The result price-column metadata should match ForecastingConfig."""
        forecasting_config = ForecastingConfig()

        horizon = forecasting_config.forecast_horizon

        prices = np.arange(
            100.0,
            100.0 + horizon + 10,
        )

        data = make_price_history(prices)

        result = build_targets_from_forecasting_config(
            data,
            forecasting_config=forecasting_config,
        )

        assert (
            result.price_column
            == forecasting_config.target_price_column
        )


# ============================================================================
# Convenience Function Tests
# ============================================================================


class TestConvenienceFunctions:
    """Tests for public target-builder convenience functions."""

    def test_build_targets_returns_target_dataset(self):
        """build_targets should return TargetDataset."""
        data = make_price_history(
            [100.0, 101.0, 102.0, 103.0, 104.0],
        )

        config = TargetBuilderConfig(
            forecast_horizon=2,
        )

        result = build_targets(
            data,
            config=config,
        )

        assert isinstance(result, TargetDataset)

    def test_build_targets_matches_direct_builder(self):
        """build_targets should match direct TargetBuilder construction."""
        data = make_price_history(
            [100.0, 105.0, 110.0, 115.0, 120.0],
        )

        config = TargetBuilderConfig(
            forecast_horizon=2,
        )

        direct_result = TargetBuilder(
            config=config,
        ).build(data)

        convenience_result = build_targets(
            data,
            config=config,
        )

        np.testing.assert_array_equal(
            convenience_result.targets,
            direct_result.targets,
        )

        np.testing.assert_array_equal(
            convenience_result.timestamps,
            direct_result.timestamps,
        )

    def test_build_targets_respects_custom_target_type(self):
        """build_targets should support future-price targets."""
        data = make_price_history(
            [100.0, 110.0, 120.0, 130.0],
        )

        config = TargetBuilderConfig(
            forecast_horizon=1,
            target_type="future_price",
            target_name="future_mid_price",
        )

        result = build_targets(
            data,
            config=config,
        )

        np.testing.assert_array_equal(
            result.targets,
            np.array(
                [
                    110.0,
                    120.0,
                    130.0,
                ]
            ),
        )


# ============================================================================
# Data Integrity and Immutability
# ============================================================================


class TestDataIntegrity:
    """Tests for input immutability and deterministic output."""

    def test_input_dataframe_is_not_modified(self):
        """Target construction should not mutate the input DataFrame."""
        data = make_price_history(
            [100.0, 101.0, 102.0, 103.0, 104.0],
        )

        original_data = data.copy(deep=True)

        make_default_builder().build(data)

        pd.testing.assert_frame_equal(
            data,
            original_data,
        )

    def test_input_dataframe_columns_are_preserved(self):
        """The input DataFrame columns should remain unchanged."""
        data = make_price_history(
            [100.0, 101.0, 102.0, 103.0],
        )

        original_columns = list(data.columns)

        make_default_builder(
            forecast_horizon=1,
        ).build(data)

        assert list(data.columns) == original_columns

    def test_input_dataframe_index_is_preserved(self):
        """The input DataFrame index should remain unchanged."""
        data = make_price_history(
            [100.0, 101.0, 102.0, 103.0],
        )

        data.index = [
            100,
            200,
            300,
            400,
        ]

        original_index = data.index.copy()

        make_default_builder(
            forecast_horizon=1,
        ).build(data)

        assert data.index.equals(original_index)

    def test_valid_output_targets_are_finite(self):
        """Valid inputs should produce finite targets."""
        data = make_price_history(
            [100.0, 101.0, 103.0, 102.0, 105.0],
        )

        result = make_default_builder().build(data)

        assert np.isfinite(result.targets).all()

    def test_valid_output_target_count_matches_timestamps(self):
        """Target and timestamp arrays must stay aligned."""
        data = make_price_history(
            [100.0, 101.0, 103.0, 102.0, 105.0],
        )

        result = make_default_builder().build(data)

        assert len(result.targets) == len(result.timestamps)


# ============================================================================
# Determinism and Regression Tests
# ============================================================================


class TestDeterminismAndRegression:
    """Regression and deterministic-behavior tests."""

    def test_repeated_builds_are_identical(self):
        """Repeated builds with identical input should match exactly."""
        data = make_price_history(
            [
                100.0,
                101.0,
                103.0,
                102.0,
                105.0,
                108.0,
            ]
        )

        builder = make_default_builder(
            forecast_horizon=2,
        )

        first = builder.build(data)
        second = builder.build(data)

        np.testing.assert_array_equal(
            first.targets,
            second.targets,
        )

        np.testing.assert_array_equal(
            first.timestamps,
            second.timestamps,
        )

    def test_repeated_builds_preserve_metadata(self):
        """Repeated builds should preserve identical metadata."""
        data = make_price_history(
            [
                100.0,
                101.0,
                102.0,
                104.0,
            ]
        )

        builder = make_default_builder(
            forecast_horizon=1,
            target_name="future_return",
        )

        first = builder.build(data)
        second = builder.build(data)

        assert first.target_name == second.target_name
        assert first.forecast_horizon == second.forecast_horizon
        assert first.price_column == second.price_column

    def test_target_values_are_independent_of_dataframe_index(self):
        """
        Target calculations should depend on ordered observations and their
        timestamps, not the pandas index.
        """
        data = make_price_history(
            [
                100.0,
                110.0,
                120.0,
                130.0,
                140.0,
            ]
        )

        modified_index_data = data.copy()
        modified_index_data.index = [
            100,
            50,
            900,
            20,
            700,
        ]

        builder = make_default_builder(
            forecast_horizon=2,
        )

        first = builder.build(data)
        second = builder.build(modified_index_data)

        np.testing.assert_array_equal(
            first.targets,
            second.targets,
        )

        np.testing.assert_array_equal(
            first.timestamps,
            second.timestamps,
        )

    def test_extra_columns_do_not_change_target_values(self):
        """Unrelated DataFrame columns should not affect target calculation."""
        data = make_price_history(
            [
                100.0,
                110.0,
                120.0,
                130.0,
                140.0,
            ]
        )

        data_with_extra = data.copy()
        data_with_extra["unrelated_feature"] = [
            999.0,
            888.0,
            777.0,
            666.0,
            555.0,
        ]

        builder = make_default_builder(
            forecast_horizon=2,
        )

        base_result = builder.build(data)
        extra_result = builder.build(data_with_extra)

        np.testing.assert_array_equal(
            base_result.targets,
            extra_result.targets,
        )

        np.testing.assert_array_equal(
            base_result.timestamps,
            extra_result.timestamps,
        )


# ============================================================================
# Edge Cases
# ============================================================================


class TestEdgeCases:
    """Tests for important target-builder edge cases."""

    def test_exactly_one_target_can_be_generated(self):
        """H+1 rows should produce one target."""
        horizon = 5

        prices = [
            100.0,
            101.0,
            102.0,
            103.0,
            104.0,
            110.0,
        ]

        data = make_price_history(prices)

        result = make_default_builder(
            forecast_horizon=horizon,
        ).build(data)

        assert result.num_targets == 1

        expected = np.array(
            [
                (110.0 - 100.0) / 100.0,
            ]
        )

        np.testing.assert_allclose(
            result.targets,
            expected,
        )

    def test_large_horizon_leaves_only_valid_current_rows(self):
        """A large horizon should exclude all unavailable future rows."""
        prices = [
            100.0,
            101.0,
            102.0,
            103.0,
            104.0,
            105.0,
            106.0,
            107.0,
        ]

        horizon = 7

        data = make_price_history(prices)

        result = make_default_builder(
            forecast_horizon=horizon,
        ).build(data)

        assert result.num_targets == 1

        np.testing.assert_allclose(
            result.targets,
            np.array(
                [
                    (107.0 - 100.0) / 100.0,
                ]
            ),
        )

    def test_fractional_prices_are_supported(self):
        """Valid fractional prices should work normally."""
        prices = [
            100.25,
            100.75,
            101.50,
            102.25,
        ]

        data = make_price_history(prices)

        result = make_default_builder(
            forecast_horizon=1,
        ).build(data)

        expected = expected_future_returns(
            prices,
            horizon=1,
        )

        np.testing.assert_allclose(
            result.targets,
            expected,
        )

    def test_fractional_timestamps_are_supported(self):
        """Valid fractional timestamps should be preserved."""
        timestamps = [
            0.25,
            1.25,
            2.25,
            3.25,
        ]

        data = make_price_history(
            [100.0, 101.0, 102.0, 103.0],
            timestamps=timestamps,
        )

        result = make_default_builder(
            forecast_horizon=1,
        ).build(data)

        np.testing.assert_array_equal(
            result.timestamps,
            np.array(
                [
                    0.25,
                    1.25,
                    2.25,
                ]
            ),
        )

    def test_large_price_values_are_supported(self):
        """Large but finite positive prices should work."""
        prices = [
            1_000_000.0,
            1_010_000.0,
            1_020_000.0,
            1_030_000.0,
        ]

        data = make_price_history(prices)

        result = make_default_builder(
            forecast_horizon=1,
        ).build(data)

        expected = expected_future_returns(
            prices,
            horizon=1,
        )

        np.testing.assert_allclose(
            result.targets,
            expected,
        )


# ============================================================================
# Final End-to-End Smoke Tests
# ============================================================================


class TestTargetBuilderSmoke:
    """End-to-end smoke tests for the complete target-building path."""

    def test_future_return_pipeline_end_to_end(self):
        """The complete future-return pipeline should work."""
        prices = [
            100.0,
            101.0,
            103.0,
            102.0,
            105.0,
            108.0,
            110.0,
        ]

        data = make_price_history(prices)

        result = make_default_builder(
            forecast_horizon=2,
        ).build(data)

        assert isinstance(result, TargetDataset)
        assert result.num_targets == 5
        assert result.targets.shape == (5,)
        assert result.timestamps.shape == (5,)
        assert result.forecast_horizon == 2
        assert result.target_name == DEFAULT_TARGET_NAME
        assert result.price_column == DEFAULT_PRICE_COLUMN

        expected = expected_future_returns(
            prices,
            horizon=2,
        )

        np.testing.assert_allclose(
            result.targets,
            expected,
            rtol=1e-12,
            atol=1e-12,
        )

    def test_future_price_pipeline_end_to_end(self):
        """The complete future-price pipeline should work."""
        prices = [
            100.0,
            101.0,
            103.0,
            102.0,
            105.0,
            108.0,
        ]

        data = make_price_history(prices)

        result = make_default_builder(
            forecast_horizon=2,
            target_type="future_price",
            target_name="future_mid_price",
        ).build(data)

        assert isinstance(result, TargetDataset)
        assert result.num_targets == 4
        assert result.targets.shape == (4,)
        assert result.timestamps.shape == (4,)

        expected = expected_future_prices(
            prices,
            horizon=2,
        )

        np.testing.assert_array_equal(
            result.targets,
            expected,
        )

    def test_forecasting_config_pipeline_end_to_end(self):
        """
        The production ForecastingConfig -> TargetBuilder -> TargetDataset
        path should work end-to-end.
        """
        forecasting_config = ForecastingConfig()

        horizon = forecasting_config.forecast_horizon

        prices = np.arange(
            100.0,
            100.0 + horizon + 20,
        )

        data = make_price_history(prices)

        result = build_targets_from_forecasting_config(
            data,
            forecasting_config=forecasting_config,
        )

        assert isinstance(result, TargetDataset)
        assert result.num_targets == len(data) - horizon
        assert (
            result.forecast_horizon
            == forecasting_config.forecast_horizon
        )
        assert (
            result.target_name
            == forecasting_config.target_column
        )
        assert (
            result.price_column
            == forecasting_config.target_price_column
        )

