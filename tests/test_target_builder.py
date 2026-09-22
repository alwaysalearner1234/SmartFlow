"""
Tests for the Target Builder
=============================

This test module validates the construction of supervised-learning targets
for the NVIDIA forecasting pipeline.

The tests cover:

- Future-return calculations.
- Future-price calculations.
- Timestamp alignment.
- Forecast-horizon behavior.
- Exclusion of final rows without future observations.
- Input validation.
- Chronological ordering.
- Duplicate timestamps.
- Missing prices.
- Infinite prices.
- Non-positive prices.
- Insufficient observations.
- DataFrame input immutability.
- TargetDataset contract integration.
- Convenience functions.
- Array-based construction.
- Configuration validation.
"""

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
# Test Data Helpers
# ============================================================================


def make_price_history(
    prices: list[float] | np.ndarray,
    *,
    timestamps: list[float] | np.ndarray | None = None,
) -> pd.DataFrame:
    """
    Create a simple chronological price-history DataFrame.
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
            "timestamp": timestamp_array,
            "mid_price": price_array,
        }
    )


def make_default_builder(
    **overrides,
) -> TargetBuilder:
    """
    Create a TargetBuilder with a small forecast horizon for testing.
    """

    config_values = {
        "forecast_horizon": 2,
        "price_column": "mid_price",
        "target_name": DEFAULT_TARGET_NAME,
        "target_type": "future_return",
        "require_chronological_order": True,
        "require_unique_timestamps": True,
        "allow_missing_prices": False,
        "allow_infinite_prices": False,
        "require_positive_prices": True,
        "drop_invalid_rows": False,
        "copy_input": True,
        "timestamp_column": "timestamp",
        "dtype": "float64",
    }

    config_values.update(overrides)

    return TargetBuilder(
        config=TargetBuilderConfig(**config_values)
    )


# ============================================================================
# Basic Construction Tests
# ============================================================================


def test_build_returns_target_dataset() -> None:
    """
    The builder should return a validated TargetDataset instance.
    """

    data = make_price_history(
        [100.0, 101.0, 102.0, 103.0, 104.0]
    )

    builder = make_default_builder()

    result = builder.build(data)

    assert isinstance(result, TargetDataset)


def test_build_generates_expected_number_of_targets() -> None:
    """
    With N observations and horizon H, the builder should produce N - H
    targets.
    """

    data = make_price_history(
        [100.0, 101.0, 102.0, 103.0, 104.0]
    )

    builder = make_default_builder()

    result = builder.build(data)

    assert result.num_targets == 3
    assert len(result.targets) == 3
    assert len(result.timestamps) == 3


def test_final_forecast_horizon_rows_are_excluded() -> None:
    """
    The final H rows must be excluded because they do not have enough future
    observations to calculate a target.
    """

    data = make_price_history(
        [100.0, 101.0, 102.0, 103.0, 104.0]
    )

    builder = make_default_builder(
        forecast_horizon=2
    )

    result = builder.build(data)

    np.testing.assert_array_equal(
        result.timestamps,
        np.array([0.0, 1.0, 2.0]),
    )


def test_target_timestamps_refer_to_current_observations() -> None:
    """
    Target timestamps must correspond to the current observation used as the
    beginning of the forecast interval, not the future observation.
    """

    data = make_price_history(
        [100.0, 110.0, 120.0, 130.0, 140.0],
        timestamps=[10.0, 20.0, 30.0, 40.0, 50.0],
    )

    builder = make_default_builder(
        forecast_horizon=2
    )

    result = builder.build(data)

    np.testing.assert_array_equal(
        result.timestamps,
        np.array([10.0, 20.0, 30.0]),
    )


# ============================================================================
# Future Return Calculation Tests
# ============================================================================


def test_future_return_calculation_is_correct() -> None:
    """
    Verify:

        future_return = (future_price - current_price) / current_price
    """

    data = make_price_history(
        [100.0, 110.0, 120.0, 130.0, 140.0]
    )

    builder = make_default_builder(
        forecast_horizon=2,
        target_type="future_return",
    )

    result = builder.build(data)

    expected_targets = np.array(
        [
            (120.0 - 100.0) / 100.0,
            (130.0 - 110.0) / 110.0,
            (140.0 - 120.0) / 120.0,
        ]
    )

    np.testing.assert_allclose(
        result.targets,
        expected_targets,
        rtol=1e-12,
        atol=1e-12,
    )


def test_future_return_supports_negative_returns() -> None:
    """
    Falling future prices should produce negative returns.
    """

    data = make_price_history(
        [100.0, 95.0, 90.0, 85.0, 80.0]
    )

    builder = make_default_builder(
        forecast_horizon=2,
        target_type="future_return",
    )

    result = builder.build(data)

    expected_targets = np.array(
        [
            (90.0 - 100.0) / 100.0,
            (85.0 - 95.0) / 95.0,
            (80.0 - 90.0) / 90.0,
        ]
    )

    np.testing.assert_allclose(
        result.targets,
        expected_targets,
        rtol=1e-12,
        atol=1e-12,
    )

    assert np.all(result.targets < 0)


def test_future_return_is_zero_when_future_price_is_unchanged() -> None:
    """
    Unchanged future prices should produce zero returns.
    """

    data = make_price_history(
        [100.0, 100.0, 100.0, 100.0, 100.0]
    )

    builder = make_default_builder(
        forecast_horizon=2,
        target_type="future_return",
    )

    result = builder.build(data)

    np.testing.assert_allclose(
        result.targets,
        np.zeros(3),
        atol=1e-12,
    )


def test_forecast_horizon_one_uses_next_observation() -> None:
    """
    A horizon of one should compare each current price with the immediately
    following price.
    """

    data = make_price_history(
        [100.0, 105.0, 110.0, 100.0]
    )

    builder = make_default_builder(
        forecast_horizon=1,
        target_type="future_return",
    )

    result = builder.build(data)

    expected_targets = np.array(
        [
            (105.0 - 100.0) / 100.0,
            (110.0 - 105.0) / 105.0,
            (100.0 - 110.0) / 110.0,
        ]
    )

    np.testing.assert_allclose(
        result.targets,
        expected_targets,
        rtol=1e-12,
        atol=1e-12,
    )


# ============================================================================
# Future Price Calculation Tests
# ============================================================================


def test_future_price_target_type_is_supported() -> None:
    """
    The builder should support future-price targets in addition to returns.
    """

    data = make_price_history(
        [100.0, 110.0, 120.0, 130.0, 140.0]
    )

    builder = make_default_builder(
        forecast_horizon=2,
        target_type="future_price",
        target_name="future_mid_price",
    )

    result = builder.build(data)

    np.testing.assert_array_equal(
        result.targets,
        np.array([120.0, 130.0, 140.0]),
    )

    assert result.target_name == "future_mid_price"


# ============================================================================
# Horizon Tests
# ============================================================================


@pytest.mark.parametrize(
    "forecast_horizon, expected_count",
    [
        (1, 5),
        (2, 4),
        (3, 3),
        (4, 2),
        (5, 1),
    ],
)
def test_target_count_matches_forecast_horizon(
    forecast_horizon: int,
    expected_count: int,
) -> None:
    """
    For N observations and horizon H, the expected number of targets is:

        N - H
    """

    data = make_price_history(
        [100.0, 101.0, 102.0, 103.0, 104.0, 105.0]
    )

    builder = make_default_builder(
        forecast_horizon=forecast_horizon
    )

    result = builder.build(data)

    assert result.num_targets == expected_count


def test_large_horizon_excludes_all_but_one_possible_current_observation() -> None:
    """
    With N = H + 1 observations, exactly one target should be generated.
    """

    data = make_price_history(
        [100.0, 101.0, 102.0, 103.0, 104.0]
    )

    builder = make_default_builder(
        forecast_horizon=4
    )

    result = builder.build(data)

    assert result.num_targets == 1

    np.testing.assert_allclose(
        result.targets,
        np.array([(104.0 - 100.0) / 100.0]),
    )


# ============================================================================
# Input Validation Tests
# ============================================================================


def test_non_dataframe_input_raises_type_error() -> None:
    """
    The builder should require a pandas DataFrame.
    """

    builder = make_default_builder()

    with pytest.raises(
        TypeError,
        match="data must be a pandas DataFrame",
    ):
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
def test_missing_required_column_raises_value_error(
    missing_column: str,
) -> None:
    """
    Missing required columns should be rejected.
    """

    data = make_price_history(
        [100.0, 101.0, 102.0, 103.0]
    )

    data = data.drop(columns=[missing_column])

    builder = make_default_builder(
        forecast_horizon=1
    )

    with pytest.raises(
        ValueError,
        match="missing required column",
    ):
        builder.build(data)


def test_empty_dataframe_raises_value_error() -> None:
    """
    Empty input data should be rejected.
    """

    data = pd.DataFrame(
        columns=["timestamp", "mid_price"]
    )

    builder = make_default_builder()

    with pytest.raises(
        ValueError,
        match="empty DataFrame",
    ):
        builder.build(data)


def test_insufficient_observations_raise_value_error() -> None:
    """
    At least H + 1 observations are required to generate one target.
    """

    data = make_price_history(
        [100.0, 101.0]
    )

    builder = make_default_builder(
        forecast_horizon=2
    )

    with pytest.raises(
        ValueError,
        match="Insufficient observations",
    ):
        builder.build(data)


@pytest.mark.parametrize(
    "invalid_price",
    [
        0.0,
        -1.0,
        -100.0,
    ],
)
def test_non_positive_prices_raise_value_error(
    invalid_price: float,
) -> None:
    """
    Zero and negative prices should be rejected.
    """

    data = make_price_history(
        [100.0, invalid_price, 102.0, 103.0]
    )

    builder = make_default_builder(
        forecast_horizon=1
    )

    with pytest.raises(
        ValueError,
        match="invalid values",
    ):
        builder.build(data)


def test_missing_price_raises_value_error() -> None:
    """
    Missing prices should be rejected by default.
    """

    data = make_price_history(
        [100.0, 101.0, 102.0, 103.0]
    )

    data.loc[1, "mid_price"] = np.nan

    builder = make_default_builder(
        forecast_horizon=1
    )

    with pytest.raises(
        ValueError,
        match="invalid values",
    ):
        builder.build(data)


@pytest.mark.parametrize(
    "invalid_price",
    [
        np.inf,
        -np.inf,
    ],
)
def test_infinite_price_raises_value_error(
    invalid_price: float,
) -> None:
    """
    Infinite prices should be rejected.
    """

    data = make_price_history(
        [100.0, invalid_price, 102.0, 103.0]
    )

    builder = make_default_builder(
        forecast_horizon=1
    )

    with pytest.raises(
        ValueError,
        match="invalid values",
    ):
        builder.build(data)


def test_non_numeric_price_raises_value_error() -> None:
    """
    Non-numeric price values should be rejected.
    """

    data = pd.DataFrame(
        {
            "timestamp": [0.0, 1.0, 2.0, 3.0],
            "mid_price": [100.0, "invalid", 102.0, 103.0],
        }
    )

    builder = make_default_builder(
        forecast_horizon=1
    )

    with pytest.raises(
        ValueError,
        match="invalid values",
    ):
        builder.build(data)


# ============================================================================
# Timestamp Validation Tests
# ============================================================================


def test_non_chronological_timestamps_raise_value_error() -> None:
    """
    Timestamps must be strictly increasing by default.
    """

    data = make_price_history(
        [100.0, 101.0, 102.0, 103.0],
        timestamps=[0.0, 2.0, 1.0, 3.0],
    )

    builder = make_default_builder(
        forecast_horizon=1
    )

    with pytest.raises(
        ValueError,
        match="strictly increasing",
    ):
        builder.build(data)


def test_duplicate_timestamps_raise_value_error() -> None:
    """
    Duplicate timestamps should be rejected by default.
    """

    data = make_price_history(
        [100.0, 101.0, 102.0, 103.0],
        timestamps=[0.0, 1.0, 1.0, 2.0],
    )

    builder = make_default_builder(
        forecast_horizon=1
    )

    with pytest.raises(
        ValueError,
        match="unique",
    ):
        builder.build(data)


def test_non_numeric_timestamps_raise_value_error() -> None:
    """
    Timestamp values must be numeric.
    """

    data = pd.DataFrame(
        {
            "timestamp": [0.0, "invalid", 2.0, 3.0],
            "mid_price": [100.0, 101.0, 102.0, 103.0],
        }
    )

    builder = make_default_builder(
        forecast_horizon=1
    )

    with pytest.raises(
        ValueError,
        match="timestamps must be numeric",
    ):
        builder.build(data)


def test_infinite_timestamps_raise_value_error() -> None:
    """
    Infinite timestamps should be rejected.
    """

    data = make_price_history(
        [100.0, 101.0, 102.0, 103.0],
        timestamps=[0.0, np.inf, 2.0, 3.0],
    )

    builder = make_default_builder(
        forecast_horizon=1
    )

    with pytest.raises(
        ValueError,
        match="timestamps must be finite",
    ):
        builder.build(data)


# ============================================================================
# Configuration Behavior Tests
# ============================================================================


def test_custom_price_column_is_supported() -> None:
    """
    The builder should support a custom price-column name.
    """

    data = pd.DataFrame(
        {
            "timestamp": [0.0, 1.0, 2.0, 3.0],
            "reference_price": [100.0, 110.0, 120.0, 130.0],
        }
    )

    config = TargetBuilderConfig(
        forecast_horizon=1,
        price_column="reference_price",
        target_name="custom_return",
    )

    builder = TargetBuilder(config=config)

    result = builder.build(data)

    assert result.price_column == "reference_price"
    assert result.target_name == "custom_return"

    np.testing.assert_allclose(
        result.targets,
        np.array(
            [
                (110.0 - 100.0) / 100.0,
                (120.0 - 110.0) / 110.0,
                (130.0 - 120.0) / 120.0,
            ]
        ),
    )


def test_custom_timestamp_column_is_supported() -> None:
    """
    The builder should support a custom timestamp-column name.
    """

    data = pd.DataFrame(
        {
            "event_time": [10.0, 20.0, 30.0, 40.0],
            "mid_price": [100.0, 105.0, 110.0, 115.0],
        }
    )

    config = TargetBuilderConfig(
        forecast_horizon=1,
        timestamp_column="event_time",
    )

    builder = TargetBuilder(config=config)

    result = builder.build(data)

    np.testing.assert_array_equal(
        result.timestamps,
        np.array([10.0, 20.0, 30.0]),
    )


def test_invalid_target_type_raises_value_error() -> None:
    """
    Unsupported target types should be rejected during configuration
    construction.
    """

    with pytest.raises(
        ValueError,
        match="Unsupported target_type",
    ):
        TargetBuilderConfig(
            target_type="unsupported"  # type: ignore[arg-type]
        )


def test_zero_forecast_horizon_raises_value_error() -> None:
    """
    The forecast horizon must be positive.
    """

    with pytest.raises(
        ValueError,
        match="greater than zero",
    ):
        TargetBuilderConfig(
            forecast_horizon=0
        )


def test_negative_forecast_horizon_raises_value_error() -> None:
    """
    Negative forecast horizons should be rejected.
    """

    with pytest.raises(
        ValueError,
        match="greater than zero",
    ):
        TargetBuilderConfig(
            forecast_horizon=-1
        )


# ============================================================================
# Invalid-Row Handling Tests
# ============================================================================


def test_invalid_rows_can_be_dropped_when_configured() -> None:
    """
    Invalid price rows should be removable when drop_invalid_rows=True.
    """

    data = pd.DataFrame(
        {
            "timestamp": [0.0, 1.0, 2.0, 3.0, 4.0, 5.0],
            "mid_price": [
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
        np.array([0.0, 2.0, 3.0, 4.0]),
    )


def test_invalid_rows_raise_when_drop_is_disabled() -> None:
    """
    Invalid rows should raise an error when dropping is disabled.
    """

    data = make_price_history(
        [100.0, 101.0, 102.0, 103.0]
    )

    data.loc[2, "mid_price"] = np.nan

    builder = make_default_builder(
        forecast_horizon=1,
        drop_invalid_rows=False,
    )

    with pytest.raises(
        ValueError,
        match="invalid values",
    ):
        builder.build(data)


# ============================================================================
# Data Integrity Tests
# ============================================================================


def test_input_dataframe_is_not_modified() -> None:
    """
    Building targets should not mutate the caller's DataFrame.
    """

    data = make_price_history(
        [100.0, 101.0, 102.0, 103.0, 104.0]
    )

    original_data = data.copy(deep=True)

    builder = make_default_builder()

    builder.build(data)

    pd.testing.assert_frame_equal(
        data,
        original_data,
    )


def test_output_targets_are_finite() -> None:
    """
    Generated targets must contain only finite values.
    """

    data = make_price_history(
        [100.0, 101.0, 102.0, 103.0, 104.0]
    )

    builder = make_default_builder()

    result = builder.build(data)

    assert np.isfinite(result.targets).all()


def test_output_targets_are_one_dimensional() -> None:
    """
    Target values should be represented as a one-dimensional array.
    """

    data = make_price_history(
        [100.0, 101.0, 102.0, 103.0, 104.0]
    )

    builder = make_default_builder()

    result = builder.build(data)

    assert result.targets.ndim == 1


def test_output_timestamps_are_one_dimensional() -> None:
    """
    Target timestamps should be represented as a one-dimensional array.
    """

    data = make_price_history(
        [100.0, 101.0, 102.0, 103.0, 104.0]
    )

    builder = make_default_builder()

    result = builder.build(data)

    assert result.timestamps.ndim == 1


# ============================================================================
# Convenience API Tests
# ============================================================================


def test_build_targets_convenience_function() -> None:
    """
    The build_targets convenience function should produce the same type of
    result as TargetBuilder.build.
    """

    data = make_price_history(
        [100.0, 101.0, 102.0, 103.0, 104.0]
    )

    config = TargetBuilderConfig(
        forecast_horizon=2
    )

    result = build_targets(
        data,
        config=config,
    )

    assert isinstance(result, TargetDataset)
    assert result.num_targets == 3


def test_build_targets_from_forecasting_config() -> None:
    """
    The forecasting-config convenience function should use the project's
    ForecastingConfig values.
    """

    data = make_price_history(
        [100.0, 101.0, 102.0, 103.0, 104.0, 105.0]
    )

    forecasting_config = ForecastingConfig()

    result = build_targets_from_forecasting_config(
        data,
        forecasting_config=forecasting_config,
    )

    expected_count = len(data) - forecasting_config.forecast_horizon

    assert result.num_targets == expected_count
    assert result.target_name == forecasting_config.target_column
    assert result.forecast_horizon == forecasting_config.forecast_horizon
    assert result.price_column == forecasting_config.target_price_column


def test_from_forecasting_config_creates_matching_builder() -> None:
    """
    TargetBuilder.from_forecasting_config should transfer the relevant
    forecasting settings.
    """

    forecasting_config = ForecastingConfig()

    builder = TargetBuilder.from_forecasting_config(
        forecasting_config
    )

    assert (
        builder.config.forecast_horizon
        == forecasting_config.forecast_horizon
    )

    assert (
        builder.config.price_column
        == forecasting_config.target_price_column
    )

    assert (
        builder.config.target_name
        == forecasting_config.target_column
    )


# ============================================================================
# Array-Based API Tests
# ============================================================================


def test_build_from_arrays_is_supported() -> None:
    """
    The array-based helper should produce the same targets as DataFrame input.
    """

    timestamps = np.array(
        [0.0, 1.0, 2.0, 3.0, 4.0]
    )

    prices = np.array(
        [100.0, 110.0, 120.0, 130.0, 140.0]
    )

    builder = make_default_builder(
        forecast_horizon=2
    )

    result = builder.build_from_arrays(
        timestamps=timestamps,
        prices=prices,
    )

    expected_targets = np.array(
        [
            (120.0 - 100.0) / 100.0,
            (130.0 - 110.0) / 110.0,
            (140.0 - 120.0) / 120.0,
        ]
    )

    np.testing.assert_allclose(
        result.targets,
        expected_targets,
    )


def test_build_from_arrays_rejects_mismatched_lengths() -> None:
    """
    Timestamp and price arrays must have matching lengths.
    """

    builder = make_default_builder()

    with pytest.raises(
        ValueError,
        match="same length",
    ):
        builder.build_from_arrays(
            timestamps=np.array([0.0, 1.0, 2.0]),
            prices=np.array([100.0, 101.0]),
        )


def test_build_from_arrays_rejects_two_dimensional_timestamps() -> None:
    """
    Timestamps must be one-dimensional.
    """

    builder = make_default_builder()

    with pytest.raises(
        ValueError,
        match="one-dimensional",
    ):
        builder.build_from_arrays(
            timestamps=np.array([[0.0], [1.0], [2.0]]),
            prices=np.array([100.0, 101.0, 102.0]),
        )


def test_build_from_arrays_rejects_two_dimensional_prices() -> None:
    """
    Prices must be one-dimensional.
    """

    builder = make_default_builder()

    with pytest.raises(
        ValueError,
        match="one-dimensional",
    ):
        builder.build_from_arrays(
            timestamps=np.array([0.0, 1.0, 2.0]),
            prices=np.array([[100.0], [101.0], [102.0]]),
        )


# ============================================================================
# DataFrame Convenience Output Tests
# ============================================================================


def test_build_dataframe_returns_expected_columns() -> None:
    """
    build_dataframe should return timestamp and target columns.
    """

    data = make_price_history(
        [100.0, 101.0, 102.0, 103.0, 104.0]
    )

    builder = make_default_builder(
        forecast_horizon=2
    )

    result = builder.build_dataframe(data)

    assert list(result.columns) == [
        "timestamp",
        DEFAULT_TARGET_NAME,
    ]

    assert len(result) == 3


def test_build_dataframe_preserves_target_values() -> None:
    """
    build_dataframe should preserve the exact target values generated by
    build.
    """

    data = make_price_history(
        [100.0, 110.0, 120.0, 130.0, 140.0]
    )

    builder = make_default_builder(
        forecast_horizon=2
    )

    dataset_result = builder.build(data)
    dataframe_result = builder.build_dataframe(data)

    np.testing.assert_allclose(
        dataframe_result[DEFAULT_TARGET_NAME].to_numpy(),
        dataset_result.targets,
    )

    np.testing.assert_array_equal(
        dataframe_result["timestamp"].to_numpy(),
        dataset_result.timestamps,
    )


# ============================================================================
# Contract Metadata Tests
# ============================================================================


def test_target_dataset_contains_correct_metadata() -> None:
    """
    The returned TargetDataset should contain the correct metadata.
    """

    data = make_price_history(
        [100.0, 101.0, 102.0, 103.0, 104.0]
    )

    builder = make_default_builder(
        forecast_horizon=2,
        target_name="custom_future_return",
        price_column="mid_price",
    )

    result = builder.build(data)

    assert result.target_name == "custom_future_return"
    assert result.forecast_horizon == 2
    assert result.price_column == "mid_price"


def test_target_values_are_deterministic() -> None:
    """
    Repeated builds with identical input should produce identical output.
    """

    data = make_price_history(
        [100.0, 101.0, 103.0, 102.0, 105.0]
    )

    builder = make_default_builder(
        forecast_horizon=2
    )

    first_result = builder.build(data)
    second_result = builder.build(data)

    np.testing.assert_array_equal(
        first_result.targets,
        second_result.targets,
    )

    np.testing.assert_array_equal(
        first_result.timestamps,
        second_result.timestamps,
    ) 

