"""
Tests for the SmartFlow rolling forecasting sequence builder.

This test module validates Phase 3 of the forecasting data pipeline.

Phase 3 transforms chronological feature history into overlapping fixed-length
input sequences for the NVIDIA forecasting model.

The tests focus on:

- correct sequence shape
- correct sequence count
- correct feature ordering
- correct rolling-window behavior
- chronological timestamps
- sequence timestamp alignment
- insufficient-history handling
- missing-column handling
- duplicate-timestamp handling
- non-chronological input handling
- missing-value handling
- infinite-value handling
- deterministic output
- input immutability
- custom context-window support
- public convenience API

The tests intentionally do not test future-return targets because target
construction belongs to Phase 4.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from data.contracts import SequenceDataset
from data.feature_history import NVIDIA_FEATURES
from data.sequence_builder import (
    DEFAULT_CONTEXT_WINDOW,
    SequenceBuilder,
    SequenceBuilderConfig,
    build_sequences,
)


# ============================================================================
# Test Constants
# ============================================================================

DEFAULT_FEATURE_COUNT = 14


# ============================================================================
# Test Data Helpers
# ============================================================================


def make_feature_history(
    row_count: int = 25,
) -> pd.DataFrame:
    """
    Create deterministic synthetic Phase 2 feature history.

    Each feature receives a simple deterministic sequence so that individual
    rolling-window values can be checked precisely.

    For example:

        mid_price_return:
            0.001
            0.002
            0.003
            ...

        spread_bps:
            0.002
            0.004
            0.006
            ...

    The exact values are less important than their deterministic relationship
    to the row index.
    """

    timestamps = np.arange(
        1,
        row_count + 1,
        dtype=float,
    )

    data = {
        "timestamp": timestamps,
    }

    for feature_index, feature_name in enumerate(NVIDIA_FEATURES):
        data[feature_name] = (
            np.arange(
                1,
                row_count + 1,
                dtype=float,
            )
            * float(feature_index + 1)
        )

    return pd.DataFrame(data)


def make_feature_history_with_extra_columns(
    row_count: int = 25,
) -> pd.DataFrame:
    """Create valid feature history containing unrelated extra columns."""

    frame = make_feature_history(row_count)

    frame["mid_price"] = np.arange(
        100.0,
        100.0 + row_count,
    )

    frame["unused_column"] = np.arange(
        0,
        row_count,
    )

    return frame


# ============================================================================
# Configuration Tests
# ============================================================================


def test_default_context_window_is_twenty() -> None:
    """The Phase 3 default context window should be 20 observations."""

    assert DEFAULT_CONTEXT_WINDOW == 20

    config = SequenceBuilderConfig()

    assert config.context_window == 20


def test_invalid_context_window_is_rejected() -> None:
    """Context windows must be positive."""

    with pytest.raises(
        ValueError,
        match="context_window must be greater than zero",
    ):
        SequenceBuilderConfig(context_window=0)

    with pytest.raises(
        ValueError,
        match="context_window must be greater than zero",
    ):
        SequenceBuilderConfig(context_window=-5)


def test_invalid_dtype_is_rejected() -> None:
    """An invalid NumPy dtype should fail during configuration."""

    with pytest.raises(
        ValueError,
        match="Invalid NumPy dtype",
    ):
        SequenceBuilderConfig(dtype="not_a_real_dtype")


# ============================================================================
# Basic Construction Tests
# ============================================================================


def test_build_returns_sequence_dataset() -> None:
    """The builder should return the shared SequenceDataset contract."""

    frame = make_feature_history(25)

    dataset = SequenceBuilder().build(frame)

    assert isinstance(dataset, SequenceDataset)


def test_default_sequence_shape_is_correct() -> None:
    """
    25 rows with a context window of 20 should produce 6 sequences.

    Formula:

        N - W + 1
        25 - 20 + 1
        = 6
    """

    frame = make_feature_history(25)

    dataset = SequenceBuilder().build(frame)

    assert dataset.shape == (6, 20, DEFAULT_FEATURE_COUNT)


def test_sequence_count_is_correct() -> None:
    """Verify the rolling-window sequence-count formula."""

    for row_count in [20, 21, 25, 40, 100]:
        frame = make_feature_history(row_count)

        dataset = SequenceBuilder().build(frame)

        expected_count = row_count - 20 + 1

        assert dataset.num_sequences == expected_count


def test_sequence_feature_count_is_fourteen() -> None:
    """Every sequence must contain exactly the 14 agreed forecasting features."""

    frame = make_feature_history(25)

    dataset = SequenceBuilder().build(frame)

    assert dataset.num_features == DEFAULT_FEATURE_COUNT
    assert len(dataset.feature_names) == DEFAULT_FEATURE_COUNT


def test_feature_names_match_nvidia_contract() -> None:
    """Feature names and ordering must exactly match NVIDIA_FEATURES."""

    frame = make_feature_history(25)

    dataset = SequenceBuilder().build(frame)

    assert dataset.feature_names == list(NVIDIA_FEATURES)


# ============================================================================
# Rolling-Window Correctness Tests
# ============================================================================


def test_first_sequence_contains_first_twenty_rows() -> None:
    """The first sequence must contain rows 1 through 20."""

    frame = make_feature_history(25)

    dataset = SequenceBuilder().build(frame)

    expected = frame[
        list(NVIDIA_FEATURES)
    ].iloc[:20].to_numpy()

    np.testing.assert_array_equal(
        dataset.sequences[0],
        expected,
    )


def test_second_sequence_shifts_forward_by_one_row() -> None:
    """The second sequence must begin at the second input observation."""

    frame = make_feature_history(25)

    dataset = SequenceBuilder().build(frame)

    expected = frame[
        list(NVIDIA_FEATURES)
    ].iloc[1:21].to_numpy()

    np.testing.assert_array_equal(
        dataset.sequences[1],
        expected,
    )


def test_last_sequence_contains_last_twenty_rows() -> None:
    """The final sequence must end at the final available observation."""

    frame = make_feature_history(25)

    dataset = SequenceBuilder().build(frame)

    expected = frame[
        list(NVIDIA_FEATURES)
    ].iloc[-20:].to_numpy()

    np.testing.assert_array_equal(
        dataset.sequences[-1],
        expected,
    )


def test_sequences_overlap_by_nineteen_rows() -> None:
    """
    Adjacent 20-row sequences should overlap by 19 observations.

    This confirms that the builder creates a rolling window rather than
    non-overlapping chunks.
    """

    frame = make_feature_history(25)

    dataset = SequenceBuilder().build(frame)

    np.testing.assert_array_equal(
        dataset.sequences[0, 1:],
        dataset.sequences[1, :-1],
    )


def test_feature_order_is_preserved() -> None:
    """Feature columns must remain in the exact NVIDIA contract order."""

    frame = make_feature_history(25)

    # Reverse the DataFrame's column order to ensure the builder does not
    # depend on the caller's column ordering.
    frame = frame[
        list(reversed(frame.columns))
    ]

    dataset = SequenceBuilder().build(frame)

    assert dataset.feature_names == list(NVIDIA_FEATURES)

    expected_first_value = 1.0

    for feature_index in range(DEFAULT_FEATURE_COUNT):
        assert (
            dataset.sequences[0, 0, feature_index]
            == expected_first_value * (feature_index + 1)
        )


# ============================================================================
# Timestamp Tests
# ============================================================================


def test_sequence_timestamps_match_feature_rows() -> None:
    """Every sequence must retain the timestamps of its source observations."""

    frame = make_feature_history(25)

    dataset = SequenceBuilder().build(frame)

    expected_first = list(
        frame["timestamp"].iloc[:20]
    )

    expected_second = list(
        frame["timestamp"].iloc[1:21]
    )

    assert dataset.timestamps[0] == expected_first
    assert dataset.timestamps[1] == expected_second


def test_sequence_timestamps_are_chronological() -> None:
    """Every generated timestamp window must be chronological."""

    frame = make_feature_history(25)

    dataset = SequenceBuilder().build(frame)

    for timestamp_window in dataset.timestamps:
        assert timestamp_window == sorted(timestamp_window)


def test_sequence_timestamps_are_unique() -> None:
    """Every timestamp within a sequence must be unique."""

    frame = make_feature_history(25)

    dataset = SequenceBuilder().build(frame)

    for timestamp_window in dataset.timestamps:
        assert len(timestamp_window) == len(
            set(timestamp_window)
        )


# ============================================================================
# Boundary Tests
# ============================================================================


def test_exact_context_window_produces_one_sequence() -> None:
    """Exactly 20 observations should produce exactly one sequence."""

    frame = make_feature_history(20)

    dataset = SequenceBuilder().build(frame)

    assert dataset.shape == (1, 20, 14)


def test_insufficient_history_is_rejected() -> None:
    """Fewer observations than the context window cannot form a sequence."""

    frame = make_feature_history(19)

    with pytest.raises(
        ValueError,
        match="Insufficient feature history",
    ):
        SequenceBuilder().build(frame)


# ============================================================================
# Schema Validation Tests
# ============================================================================


def test_non_dataframe_input_is_rejected() -> None:
    """The builder requires a pandas DataFrame."""

    with pytest.raises(
        TypeError,
        match="feature_history must be a pandas DataFrame",
    ):
        SequenceBuilder().build([])  # type: ignore[arg-type]


def test_missing_required_feature_is_rejected() -> None:
    """Every NVIDIA forecasting feature must be present."""

    frame = make_feature_history(25)

    frame = frame.drop(
        columns=[NVIDIA_FEATURES[0]]
    )

    with pytest.raises(
        ValueError,
        match="missing required columns",
    ):
        SequenceBuilder().build(frame)


def test_missing_timestamp_is_rejected() -> None:
    """Timestamp is required for sequence alignment."""

    frame = make_feature_history(25)

    frame = frame.drop(
        columns=["timestamp"]
    )

    with pytest.raises(
        ValueError,
        match="missing required columns",
    ):
        SequenceBuilder().build(frame)


def test_extra_columns_do_not_change_sequence_output() -> None:
    """
    Additional DataFrame columns should be ignored.

    Only the agreed NVIDIA feature columns should enter the sequence tensor.
    """

    base = make_feature_history(25)

    extended = make_feature_history_with_extra_columns(25)

    base_dataset = SequenceBuilder().build(base)
    extended_dataset = SequenceBuilder().build(extended)

    np.testing.assert_array_equal(
        base_dataset.sequences,
        extended_dataset.sequences,
    )


# ============================================================================
# Chronology Tests
# ============================================================================


def test_non_chronological_input_is_rejected() -> None:
    """Chronological ordering must be enforced by default."""

    frame = make_feature_history(25)

    frame.loc[10, "timestamp"] = 5.5

    with pytest.raises(
        ValueError,
        match="must be in chronological order",
    ):
        SequenceBuilder().build(frame)


def test_duplicate_timestamps_are_rejected() -> None:
    """Duplicate timestamps can create ambiguous sequence boundaries."""

    frame = make_feature_history(25)

    frame.loc[10, "timestamp"] = frame.loc[9, "timestamp"]

    with pytest.raises(
        ValueError,
        match="duplicate timestamps",
    ):
        SequenceBuilder().build(frame)


def test_non_numeric_timestamp_is_rejected() -> None:
    """Timestamps must be numeric."""

    frame = make_feature_history(25)

    frame["timestamp"] = frame["timestamp"].astype(object)
    frame.loc[5, "timestamp"] = "invalid"

    with pytest.raises(
        ValueError,
        match="timestamps must be numeric and finite",
    ):
        SequenceBuilder().build(frame)



def test_infinite_timestamp_is_rejected() -> None:
    """Infinite timestamps are invalid."""

    frame = make_feature_history(25)

    frame.loc[5, "timestamp"] = np.inf

    with pytest.raises(
        ValueError,
        match="timestamps must be finite",
    ):
        SequenceBuilder().build(frame)


# ============================================================================
# Feature-Value Validation Tests
# ============================================================================


def test_missing_feature_value_is_rejected() -> None:
    """NaN feature values are rejected by default."""

    frame = make_feature_history(25)

    frame.loc[5, NVIDIA_FEATURES[0]] = np.nan

    with pytest.raises(
        ValueError,
        match="missing or non-numeric values",
    ):
        SequenceBuilder().build(frame)


def test_non_numeric_feature_value_is_rejected() -> None:
    """Non-numeric feature values must not enter the sequence tensor."""

    frame = make_feature_history(25)

    feature_name = NVIDIA_FEATURES[0]

    frame[feature_name] = frame[feature_name].astype(object)
    frame.loc[5, feature_name] = "invalid"

    with pytest.raises(ValueError, match="missing or non-numeric"):
        SequenceBuilder().build(frame)



def test_infinite_feature_value_is_rejected() -> None:
    """Infinite feature values are invalid by default."""

    frame = make_feature_history(25)

    frame.loc[5, NVIDIA_FEATURES[0]] = np.inf

    with pytest.raises(
        ValueError,
        match="infinite feature values",
    ):
        SequenceBuilder().build(frame)


def test_negative_infinite_feature_value_is_rejected() -> None:
    """Negative infinity must also be rejected."""

    frame = make_feature_history(25)

    frame.loc[5, NVIDIA_FEATURES[0]] = -np.inf

    with pytest.raises(
        ValueError,
        match="infinite feature values",
    ):
        SequenceBuilder().build(frame)


# ============================================================================
# Custom Configuration Tests
# ============================================================================


def test_custom_context_window_is_supported() -> None:
    """The builder should support context windows other than 20."""

    frame = make_feature_history(10)

    dataset = SequenceBuilder(
        SequenceBuilderConfig(
            context_window=5,
        )
    ).build(frame)

    assert dataset.shape == (6, 5, 14)
    assert dataset.context_window == 5


def test_convenience_function_supports_custom_context_window() -> None:
    """build_sequences should expose context-window customization."""

    frame = make_feature_history(10)

    dataset = build_sequences(
        frame,
        context_window=5,
    )

    assert dataset.shape == (6, 5, 14)


def test_allow_missing_values_configuration_can_be_enabled() -> None:
    """
    Missing values may be explicitly allowed by configuration.

    This is not the default because the NVIDIA training pipeline should
    normally receive complete feature data.
    """

    frame = make_feature_history(25)

    frame.loc[5, NVIDIA_FEATURES[0]] = np.nan

    builder = SequenceBuilder(
        SequenceBuilderConfig(
            allow_missing_values=True,
        )
    )

    dataset = builder.build(frame)

    assert np.isnan(
        dataset.sequences
    ).any()


def test_allow_infinite_values_configuration_can_be_enabled() -> None:
    """
    Infinite values may be explicitly allowed by configuration.

    This option exists for controlled experimentation, not as the default
    production data policy.
    """

    frame = make_feature_history(25)

    frame.loc[5, NVIDIA_FEATURES[0]] = np.inf

    builder = SequenceBuilder(
        SequenceBuilderConfig(
            allow_infinite_values=True,
        )
    )

    dataset = builder.build(frame)

    assert np.isinf(
        dataset.sequences
    ).any()


# ============================================================================
# Determinism Tests
# ============================================================================


def test_sequence_construction_is_deterministic() -> None:
    """Identical input should always produce identical sequences."""

    frame = make_feature_history(100)

    dataset_one = SequenceBuilder().build(frame)
    dataset_two = SequenceBuilder().build(frame)

    np.testing.assert_array_equal(
        dataset_one.sequences,
        dataset_two.sequences,
    )

    assert dataset_one.timestamps == dataset_two.timestamps
    assert dataset_one.feature_names == dataset_two.feature_names


# ============================================================================
# Input Immutability Tests
# ============================================================================


def test_builder_does_not_mutate_input_by_default() -> None:
    """The default builder configuration should preserve the input DataFrame."""

    frame = make_feature_history(25)

    original = frame.copy(deep=True)

    SequenceBuilder().build(frame)

    pd.testing.assert_frame_equal(
        frame,
        original,
    )


# ============================================================================
# Data-Type Tests
# ============================================================================


def test_default_sequence_dtype_is_float64() -> None:
    """The default sequence tensor should use float64."""

    frame = make_feature_history(25)

    dataset = SequenceBuilder().build(frame)

    assert dataset.sequences.dtype == np.float64


def test_custom_sequence_dtype_is_supported() -> None:
    """The sequence tensor dtype should be configurable."""

    frame = make_feature_history(25)

    builder = SequenceBuilder(
        SequenceBuilderConfig(
            dtype="float32",
        )
    )

    dataset = builder.build(frame)

    assert dataset.sequences.dtype == np.float32


# ============================================================================
# Public API Tests
# ============================================================================


def test_convenience_function_uses_default_configuration() -> None:
    """build_sequences should match SequenceBuilder's default behavior."""

    frame = make_feature_history(25)

    direct_dataset = SequenceBuilder().build(frame)
    convenience_dataset = build_sequences(frame)

    np.testing.assert_array_equal(
        direct_dataset.sequences,
        convenience_dataset.sequences,
    )

    assert direct_dataset.timestamps == convenience_dataset.timestamps
    assert (
        direct_dataset.feature_names
        == convenience_dataset.feature_names
    )


# ============================================================================
# Temporal Safety Tests
# ============================================================================


def test_sequence_does_not_contain_future_rows() -> None:
    """
    Verify that a sequence contains only its own historical window.

    For the first sequence, the final timestamp must be t20 rather than t21
    or later.
    """

    frame = make_feature_history(25)

    dataset = SequenceBuilder().build(frame)

    first_timestamp_window = dataset.timestamps[0]

    assert first_timestamp_window[0] == 1.0
    assert first_timestamp_window[-1] == 20.0

    assert 21.0 not in first_timestamp_window
    assert 22.0 not in first_timestamp_window
    assert 23.0 not in first_timestamp_window


def test_each_sequence_ends_at_its_corresponding_history_row() -> None:
    """
    Every sequence's final timestamp should correspond to the final source
    observation included in that sequence.
    """

    frame = make_feature_history(25)

    dataset = SequenceBuilder().build(frame)

    for sequence_index, timestamp_window in enumerate(
        dataset.timestamps
    ):
        expected_final_timestamp = float(
            sequence_index + DEFAULT_CONTEXT_WINDOW
        )

        assert timestamp_window[-1] == expected_final_timestamp


def test_sequence_timestamp_and_feature_data_remain_aligned() -> None:
    """
    Verify that timestamp windows and feature windows reference the same
    source rows.
    """

    frame = make_feature_history(25)

    dataset = SequenceBuilder().build(frame)

    for sequence_index in range(
        dataset.num_sequences
    ):
        expected_timestamp = frame.iloc[
            sequence_index + DEFAULT_CONTEXT_WINDOW - 1
        ]["timestamp"]

        actual_timestamp = dataset.timestamps[
            sequence_index
        ][-1]

        assert actual_timestamp == expected_timestamp

