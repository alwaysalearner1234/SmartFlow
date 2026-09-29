# tests/test_sequence_builder.py
#
# Purpose:
# Comprehensive tests for the rolling sequence dataset builder used by the
# NVIDIA forecasting data pipeline.
#
# Responsibilities:
# - Verify SequenceBuilder configuration and defaults.
# - Verify required feature-history columns.
# - Verify timestamp validation and chronological ordering.
# - Verify NVIDIA feature validation.
# - Verify rolling-window sequence construction.
# - Verify sequence/timestamp alignment.
# - Verify SequenceDataset integration.
# - Verify convenience API behavior.
# - Verify NVIDIA forecasting input-contract compatibility.
# - Verify invalid and edge-case inputs.
# - Verify deterministic and input-independent behavior.
#
# The module does not test model architecture, model training, normalization,
# target construction, chronological splitting, or forecasting performance.
#
# The tests intentionally follow the behavior of the production
# SequenceBuilder and SequenceDataset implementations.

from dataclasses import replace

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


# ---------------------------------------------------------------------------
# Shared test constants
# ---------------------------------------------------------------------------

CONTEXT_WINDOW = DEFAULT_CONTEXT_WINDOW
FEATURE_NAMES = list(NVIDIA_FEATURES)
FEATURE_COUNT = len(FEATURE_NAMES)

DEFAULT_ROW_COUNT = 40


# ---------------------------------------------------------------------------
# Test-data helpers
# ---------------------------------------------------------------------------


def make_feature_history(
    row_count: int = DEFAULT_ROW_COUNT,
    *,
    start_timestamp: float = 1_000.0,
    timestamp_step: float = 1.0,
) -> pd.DataFrame:
    """
    Build a deterministic feature-history DataFrame for sequence-builder tests.

    The generated data follows the same column order expected by the
    production SequenceBuilder.
    """
    if row_count < 1:
        raise ValueError("row_count must be positive.")

    timestamps = [
        start_timestamp + index * timestamp_step
        for index in range(row_count)
    ]

    data = {
        "timestamp": timestamps,
    }

    for feature_index, feature_name in enumerate(FEATURE_NAMES):
        data[feature_name] = [
            float(feature_index + row_index / 100.0)
            for row_index in range(row_count)
        ]

    return pd.DataFrame(data)


def make_builder(
    *,
    context_window: int = CONTEXT_WINDOW,
    **overrides,
) -> SequenceBuilder:
    """Create a SequenceBuilder with the requested configuration overrides."""
    config = SequenceBuilderConfig(
        context_window=context_window,
        **overrides,
    )

    return SequenceBuilder(config=config)


# ---------------------------------------------------------------------------
# SequenceBuilderConfig tests
# ---------------------------------------------------------------------------


class TestSequenceBuilderConfig:
    """Tests for SequenceBuilderConfig defaults and configuration behavior."""

    def test_default_context_window(self):
        """The default context window should match the forecasting contract."""
        config = SequenceBuilderConfig()

        assert config.context_window == CONTEXT_WINDOW

    def test_default_chronological_validation(self):
        """Chronological validation should be enabled by default."""
        config = SequenceBuilderConfig()

        assert config.require_chronological_order is True

    def test_default_unique_timestamp_validation(self):
        """Unique timestamp validation should be enabled by default."""
        config = SequenceBuilderConfig()

        assert config.require_unique_timestamps is True

    def test_default_missing_value_policy(self):
        """Missing values should be rejected by default."""
        config = SequenceBuilderConfig()

        assert config.allow_missing_values is False

    def test_default_infinite_value_policy(self):
        """Infinite feature values should be rejected by default."""
        config = SequenceBuilderConfig()

        assert config.allow_infinite_values is False

    def test_default_copy_policy(self):
        """Input data should be copied by default."""
        config = SequenceBuilderConfig()

        assert config.copy_input is True

    def test_default_dtype(self):
        """Sequences should use float64 by default."""
        config = SequenceBuilderConfig()

        assert config.dtype == "float64"

    def test_custom_context_window(self):
        """A custom positive context window should be accepted."""
        config = SequenceBuilderConfig(context_window=5)

        assert config.context_window == 5

    @pytest.mark.parametrize(
        "context_window",
        [0, -1, -5],
    )
    def test_invalid_context_window_rejected(self, context_window):
        """Non-positive context windows should be rejected."""
        with pytest.raises(ValueError):
            SequenceBuilderConfig(context_window=context_window)


# ---------------------------------------------------------------------------
# Basic builder construction tests
# ---------------------------------------------------------------------------


class TestSequenceBuilderConstruction:
    """Tests for constructing SequenceBuilder instances."""

    def test_builder_accepts_default_config(self):
        """A builder should be constructible with the default configuration."""
        builder = SequenceBuilder()

        assert builder.config.context_window == CONTEXT_WINDOW

    def test_builder_accepts_custom_config(self):
        """A builder should preserve custom configuration values."""
        config = SequenceBuilderConfig(context_window=10)
        builder = SequenceBuilder(config=config)

        assert builder.config.context_window == 10

    def test_builder_preserves_config(self):
        """The builder should expose the supplied configuration."""
        config = SequenceBuilderConfig(
            context_window=10,
            require_chronological_order=False,
            require_unique_timestamps=False,
        )

        builder = SequenceBuilder(config=config)

        assert builder.config == config


# ---------------------------------------------------------------------------
# Required-column validation
# ---------------------------------------------------------------------------


class TestRequiredColumns:
    """Tests for required feature-history columns."""

    def test_timestamp_column_is_required(self):
        """A feature history without timestamp should be rejected."""
        frame = make_feature_history().drop(columns=["timestamp"])

        builder = make_builder()

        with pytest.raises(ValueError, match="timestamp"):
            builder.build(frame)

    @pytest.mark.parametrize("feature_name", FEATURE_NAMES)
    def test_each_nvidia_feature_is_required(self, feature_name):
        """Every NVIDIA forecasting feature must be present."""
        frame = make_feature_history().drop(columns=[feature_name])

        builder = make_builder()

        with pytest.raises(ValueError, match=feature_name):
            builder.build(frame)

    def test_missing_multiple_columns_is_rejected(self):
        """Multiple missing required columns should still be rejected."""
        frame = make_feature_history().drop(
            columns=["timestamp", FEATURE_NAMES[0], FEATURE_NAMES[1]]
        )

        builder = make_builder()

        with pytest.raises(ValueError):
            builder.build(frame)

    def test_extra_columns_are_allowed(self):
        """Additional columns should not prevent sequence construction."""
        frame = make_feature_history()
        frame["extra_column"] = np.arange(len(frame), dtype=float)

        builder = make_builder()

        result = builder.build(frame)

        assert result.sequences.shape[1:] == (
            CONTEXT_WINDOW,
            FEATURE_COUNT,
        )


# ---------------------------------------------------------------------------
# Input type validation
# ---------------------------------------------------------------------------


class TestInputTypeValidation:
    """Tests for invalid input objects."""

    @pytest.mark.parametrize(
        "invalid_input",
        [
            None,
            [],
            {},
            np.array([1, 2, 3]),
            "not a dataframe",
        ],
    )
    def test_non_dataframe_input_raises(self, invalid_input):
        """SequenceBuilder should require a pandas DataFrame."""
        builder = make_builder()

        with pytest.raises((TypeError, ValueError)):
            builder.build(invalid_input)


# ---------------------------------------------------------------------------
# Timestamp validation
# ---------------------------------------------------------------------------


class TestTimestampValidation:
    """Tests for timestamp validity and ordering."""

    def test_timestamps_must_be_numeric(self):
        """Non-numeric timestamps should be rejected."""
        frame = make_feature_history()

        frame["timestamp"] = frame["timestamp"].astype(object)
        frame.loc[5, "timestamp"] = "invalid"

        builder = make_builder()

        with pytest.raises(
            ValueError,
            match="timestamps must be numeric and finite",
        ):
            builder.build(frame)

    def test_nan_timestamp_raises(self):
        """NaN timestamps should be rejected."""
        frame = make_feature_history()
        frame.loc[5, "timestamp"] = np.nan

        builder = make_builder()

        with pytest.raises(ValueError):
            builder.build(frame)

    def test_infinite_timestamp_raises(self):
        """Infinite timestamps should be rejected."""
        frame = make_feature_history()
        frame.loc[5, "timestamp"] = np.inf

        builder = make_builder()

        with pytest.raises(ValueError):
            builder.build(frame)

    def test_negative_infinite_timestamp_raises(self):
        """Negative infinite timestamps should be rejected."""
        frame = make_feature_history()
        frame.loc[5, "timestamp"] = -np.inf

        builder = make_builder()

        with pytest.raises(ValueError):
            builder.build(frame)

    def test_duplicate_timestamps_raise_by_default(self):
        """Duplicate timestamps should be rejected by default."""
        frame = make_feature_history()
        frame.loc[10, "timestamp"] = frame.loc[9, "timestamp"]

        builder = make_builder()

        with pytest.raises(ValueError):
            builder.build(frame)

    def test_duplicate_timestamp_validation_can_be_disabled(self):
        """
        Disabling builder-level duplicate validation does not bypass the
        downstream SequenceDataset timestamp contract.
        """
        frame = make_feature_history()
        frame.loc[10, "timestamp"] = frame.loc[9, "timestamp"]

        builder = make_builder(
            require_unique_timestamps=False,
        )

        with pytest.raises(ValueError):
            builder.build(frame)

    def test_non_chronological_timestamps_raise_by_default(self):
        """Non-chronological timestamps should be rejected."""
        frame = make_feature_history()

        frame.loc[10, "timestamp"], frame.loc[11, "timestamp"] = (
            frame.loc[11, "timestamp"],
            frame.loc[10, "timestamp"],
        )

        builder = make_builder()

        with pytest.raises(ValueError):
            builder.build(frame)

    def test_chronological_validation_can_be_disabled(self):
        """
        Disabling builder-level chronological validation does not bypass the
        downstream SequenceDataset timestamp contract.
        """
        frame = make_feature_history()

        frame.loc[10, "timestamp"], frame.loc[11, "timestamp"] = (
            frame.loc[11, "timestamp"],
            frame.loc[10, "timestamp"],
        )

        builder = make_builder(
            require_chronological_order=False,
        )

        with pytest.raises(ValueError):
            builder.build(frame)

    def test_equal_adjacent_timestamps_are_rejected_by_dataset_contract(self):
        """
        Even when builder-level uniqueness validation is disabled, the
        SequenceDataset contract requires unique timestamps.
        """
        frame = make_feature_history()
        frame.loc[10, "timestamp"] = frame.loc[9, "timestamp"]

        builder = make_builder(
            require_unique_timestamps=False,
        )

        with pytest.raises(ValueError):
            builder.build(frame)

    def test_timestamp_order_is_preserved(self):
        """Valid timestamps should remain chronological in every sequence."""
        frame = make_feature_history()
        builder = make_builder()

        result = builder.build(frame)

        for timestamp_window in result.timestamps:
            assert timestamp_window == sorted(timestamp_window)

    def test_timestamp_windows_have_context_window_length(self):
        """Every timestamp window should contain exactly W timestamps."""
        frame = make_feature_history()
        builder = make_builder()

        result = builder.build(frame)

        assert all(
            len(timestamp_window) == CONTEXT_WINDOW
            for timestamp_window in result.timestamps
        )


# ---------------------------------------------------------------------------
# Feature-value validation
# ---------------------------------------------------------------------------


class TestFeatureValidation:
    """Tests for numeric, missing, and infinite feature values."""

    @pytest.mark.parametrize("feature_name", FEATURE_NAMES)
    def test_non_numeric_feature_value_raises(self, feature_name):
        """Non-numeric feature values should be rejected."""
        frame = make_feature_history()

        frame[feature_name] = frame[feature_name].astype(object)
        frame.loc[5, feature_name] = "invalid"

        builder = make_builder()

        with pytest.raises(
            ValueError,
            match="missing or non-numeric values",
        ):
            builder.build(frame)

    @pytest.mark.parametrize("feature_name", FEATURE_NAMES)
    def test_nan_feature_value_raises(self, feature_name):
        """NaN feature values should be rejected by default."""
        frame = make_feature_history()
        frame.loc[5, feature_name] = np.nan

        builder = make_builder()

        with pytest.raises(
            ValueError,
            match="missing or non-numeric values",
        ):
            builder.build(frame)

    @pytest.mark.parametrize("feature_name", FEATURE_NAMES)
    def test_infinite_feature_value_raises(self, feature_name):
        """Infinite feature values should be rejected by default."""
        frame = make_feature_history()
        frame.loc[5, feature_name] = np.inf

        builder = make_builder()

        with pytest.raises(
            ValueError,
            match="infinite",
        ):
            builder.build(frame)

    @pytest.mark.parametrize("feature_name", FEATURE_NAMES)
    def test_negative_infinite_feature_value_raises(self, feature_name):
        """Negative infinite feature values should be rejected by default."""
        frame = make_feature_history()
        frame.loc[5, feature_name] = -np.inf

        builder = make_builder()

        with pytest.raises(
            ValueError,
            match="infinite",
        ):
            builder.build(frame)

    def test_missing_values_can_be_allowed(self):
        """
        When missing values are explicitly allowed, the builder should not
        reject them at the feature-validation stage.
        """
        frame = make_feature_history()
        frame.loc[5, FEATURE_NAMES[0]] = np.nan

        builder = make_builder(
            allow_missing_values=True,
        )

        result = builder.build(frame)

        assert result.sequences.shape[1:] == (
            CONTEXT_WINDOW,
            FEATURE_COUNT,
        )

    def test_infinite_values_can_be_allowed(self):
        """
        When infinite values are explicitly allowed, the builder should not
        reject them at the feature-validation stage.
        """
        frame = make_feature_history()
        frame.loc[5, FEATURE_NAMES[0]] = np.inf

        builder = make_builder(
            allow_infinite_values=True,
        )

        result = builder.build(frame)

        assert result.sequences.shape[1:] == (
            CONTEXT_WINDOW,
            FEATURE_COUNT,
        )


# ---------------------------------------------------------------------------
# History-length validation
# ---------------------------------------------------------------------------


class TestHistoryLength:
    """Tests for minimum feature-history length."""

    def test_exact_context_window_is_valid(self):
        """Exactly W rows should produce one sequence."""
        frame = make_feature_history(
            row_count=CONTEXT_WINDOW,
        )

        builder = make_builder()

        result = builder.build(frame)

        assert result.num_sequences == 1
        assert result.sequences.shape == (
            1,
            CONTEXT_WINDOW,
            FEATURE_COUNT,
        )

    def test_less_than_context_window_raises(self):
        """Fewer than W rows should be rejected."""
        frame = make_feature_history(
            row_count=CONTEXT_WINDOW - 1,
        )

        builder = make_builder()

        with pytest.raises(ValueError):
            builder.build(frame)

    def test_one_more_than_context_window_produces_two_sequences(self):
        """W+1 rows should produce two rolling windows."""
        frame = make_feature_history(
            row_count=CONTEXT_WINDOW + 1,
        )

        builder = make_builder()

        result = builder.build(frame)

        assert result.num_sequences == 2

    @pytest.mark.parametrize(
        "row_count",
        [
            CONTEXT_WINDOW + 5,
            CONTEXT_WINDOW + 10,
            CONTEXT_WINDOW + 20,
        ],
    )
    def test_expected_sequence_count(self, row_count):
        """The number of sequences should be N-W+1."""
        frame = make_feature_history(row_count=row_count)

        builder = make_builder()

        result = builder.build(frame)

        expected = row_count - CONTEXT_WINDOW + 1

        assert result.num_sequences == expected


# ---------------------------------------------------------------------------
# Rolling sequence construction
# ---------------------------------------------------------------------------


class TestSequenceConstruction:
    """Tests for rolling sequence generation."""

    def test_sequence_batch_shape(self):
        """The sequence batch should have shape (N-W+1, W, F)."""
        row_count = 40
        frame = make_feature_history(row_count=row_count)

        builder = make_builder()

        result = builder.build(frame)

        expected_sequence_count = row_count - CONTEXT_WINDOW + 1

        assert result.sequences.shape == (
            expected_sequence_count,
            CONTEXT_WINDOW,
            FEATURE_COUNT,
        )

    def test_custom_context_window_changes_sequence_shape(self):
        """Changing W should change the sequence dimensions correctly."""
        custom_context = 5
        row_count = 20

        frame = make_feature_history(row_count=row_count)

        builder = make_builder(
            context_window=custom_context,
        )

        result = builder.build(frame)

        assert result.sequences.shape == (
            row_count - custom_context + 1,
            custom_context,
            FEATURE_COUNT,
        )

    def test_first_sequence_contains_first_context_rows(self):
        """The first sequence should contain the first W feature rows."""
        frame = make_feature_history()

        builder = make_builder()

        result = builder.build(frame)

        expected = frame[FEATURE_NAMES].iloc[
            :CONTEXT_WINDOW
        ].to_numpy(dtype=np.float64)

        np.testing.assert_allclose(
            result.sequences[0],
            expected,
        )

    def test_last_sequence_contains_last_context_rows(self):
        """The final sequence should contain the final W feature rows."""
        frame = make_feature_history()

        builder = make_builder()

        result = builder.build(frame)

        expected = frame[FEATURE_NAMES].iloc[
            -CONTEXT_WINDOW:
        ].to_numpy(dtype=np.float64)

        np.testing.assert_allclose(
            result.sequences[-1],
            expected,
        )

    def test_rolling_windows_shift_by_one_row(self):
        """Adjacent sequences should shift forward by exactly one row."""
        frame = make_feature_history()

        builder = make_builder()

        result = builder.build(frame)

        np.testing.assert_allclose(
            result.sequences[0, 1:],
            result.sequences[1, :-1],
        )

    def test_feature_order_matches_nvidia_contract(self):
        """The final feature dimension must follow NVIDIA_FEATURES order."""
        frame = make_feature_history()

        builder = make_builder()

        result = builder.build(frame)

        assert result.feature_names == FEATURE_NAMES

    def test_sequence_dtype_matches_configuration(self):
        """The resulting sequence dtype should match configuration."""
        frame = make_feature_history()

        builder = make_builder(
            dtype="float32",
        )

        result = builder.build(frame)

        assert result.sequences.dtype == np.float32

    def test_default_sequence_dtype_is_float64(self):
        """The default sequence dtype should be float64."""
        frame = make_feature_history()

        builder = make_builder()

        result = builder.build(frame)

        assert result.sequences.dtype == np.float64


# ---------------------------------------------------------------------------
# Timestamp-window construction
# ---------------------------------------------------------------------------


class TestTimestampWindows:
    """Tests for sequence timestamp windows."""

    def test_timestamps_are_nested_windows(self):
        """Each sequence should have its own complete timestamp window."""
        frame = make_feature_history()

        builder = make_builder()

        result = builder.build(frame)

        assert isinstance(result.timestamps, list)
        assert all(
            isinstance(timestamp_window, list)
            for timestamp_window in result.timestamps
        )

    def test_timestamp_window_count_matches_sequence_count(self):
        """There should be one timestamp window per sequence."""
        frame = make_feature_history()

        builder = make_builder()

        result = builder.build(frame)

        assert len(result.timestamps) == result.num_sequences

    def test_first_timestamp_window_matches_first_rows(self):
        """The first timestamp window should contain the first W timestamps."""
        frame = make_feature_history()

        builder = make_builder()

        result = builder.build(frame)

        expected = frame["timestamp"].iloc[
            :CONTEXT_WINDOW
        ].astype(float).tolist()

        assert result.timestamps[0] == expected

    def test_last_timestamp_window_matches_last_rows(self):
        """The final timestamp window should contain the final W timestamps."""
        frame = make_feature_history()

        builder = make_builder()

        result = builder.build(frame)

        expected = frame["timestamp"].iloc[
            -CONTEXT_WINDOW:
        ].astype(float).tolist()

        assert result.timestamps[-1] == expected

    def test_adjacent_timestamp_windows_shift_by_one(self):
        """Adjacent timestamp windows should shift by exactly one row."""
        frame = make_feature_history()

        builder = make_builder()

        result = builder.build(frame)

        assert result.timestamps[0][1:] == result.timestamps[1][:-1]


# ---------------------------------------------------------------------------
# Input-copy and output-independence behavior
# ---------------------------------------------------------------------------


class TestInputHandling:
    """Tests for input-copy behavior and output independence."""

    def test_builder_does_not_modify_input_structure(self):
        """Building sequences should not alter the input DataFrame structure."""
        frame = make_feature_history()

        original_columns = list(frame.columns)
        original_index = frame.index.copy()

        builder = make_builder()

        builder.build(frame)

        assert list(frame.columns) == original_columns
        assert frame.index.equals(original_index)

    def test_output_sequences_do_not_depend_on_later_input_mutation(self):
        """
        Mutating the original feature history after construction should not
        alter already-built sequences.
        """
        frame = make_feature_history()

        builder = make_builder()

        result = builder.build(frame)

        original_value = result.sequences[0, 0, 0]

        frame.loc[0, FEATURE_NAMES[0]] = -999999.0

        assert result.sequences[0, 0, 0] == original_value

    def test_output_sequences_are_independent_of_future_input_changes(self):
        """Later changes to the input should not alter stored sequence values."""
        frame = make_feature_history()

        builder = make_builder()

        result = builder.build(frame)

        original = result.sequences.copy()

        frame.loc[:, FEATURE_NAMES] = -123456.0

        np.testing.assert_array_equal(
            result.sequences,
            original,
        )

    def test_output_sequences_can_be_mutated_independently(self):
        """
        The returned NumPy sequence array is not required to be read-only.
        Mutating it should not mutate the original feature history.
        """
        frame = make_feature_history()

        result = build_sequences(frame)

        original_input_value = frame.loc[0, FEATURE_NAMES[0]]
        original_output_value = result.sequences[0, 0, 0]

        result.sequences[0, 0, 0] = -777777.0

        assert result.sequences[0, 0, 0] == -777777.0
        assert frame.loc[0, FEATURE_NAMES[0]] == original_input_value
        assert original_output_value != -777777.0


# ---------------------------------------------------------------------------
# SequenceDataset integration
# ---------------------------------------------------------------------------


class TestSequenceDatasetIntegration:
    """Tests for integration with the shared SequenceDataset contract."""

    def test_result_is_sequence_dataset(self):
        """SequenceBuilder should return SequenceDataset."""
        frame = make_feature_history()

        builder = make_builder()

        result = builder.build(frame)

        assert isinstance(result, SequenceDataset)

    def test_sequence_dataset_feature_names(self):
        """SequenceDataset feature names should match NVIDIA_FEATURES."""
        frame = make_feature_history()

        builder = make_builder()

        result = builder.build(frame)

        assert result.feature_names == FEATURE_NAMES

    def test_sequence_dataset_context_window(self):
        """SequenceDataset should preserve the configured context window."""
        frame = make_feature_history()

        builder = make_builder()

        result = builder.build(frame)

        assert result.context_window == CONTEXT_WINDOW

    def test_sequence_dataset_num_sequences(self):
        """SequenceDataset should report the correct number of sequences."""
        frame = make_feature_history()

        builder = make_builder()

        result = builder.build(frame)

        assert result.num_sequences == (
            len(frame) - CONTEXT_WINDOW + 1
        )

    def test_sequence_dataset_shape_is_full_batch_shape(self):
        """
        SequenceDataset.shape should represent the complete batch:
        (N-W+1, W, F).
        """
        frame = make_feature_history()

        builder = make_builder()

        result = builder.build(frame)

        assert result.shape == (
            result.num_sequences,
            CONTEXT_WINDOW,
            FEATURE_COUNT,
        )

    def test_sequence_dataset_input_shape_matches_actual_contract(self):
        """
        SequenceDataset.input_shape follows the current shared contract and
        represents the complete stored sequence tensor shape.
        """
        frame = make_feature_history()

        builder = make_builder()

        result = builder.build(frame)

        assert result.input_shape == (
            result.num_sequences,
            CONTEXT_WINDOW,
            FEATURE_COUNT,
        )

    def test_sequence_dataset_timestamps_match_sequence_count(self):
        """Timestamp-window count should match sequence count."""
        frame = make_feature_history()

        builder = make_builder()

        result = builder.build(frame)

        assert len(result.timestamps) == result.num_sequences


# ---------------------------------------------------------------------------
# Convenience API
# ---------------------------------------------------------------------------


class TestConvenienceAPI:
    """Tests for build_sequences()."""

    def test_build_sequences_returns_sequence_dataset(self):
        """The convenience API should return SequenceDataset."""
        frame = make_feature_history()

        result = build_sequences(
            frame,
            context_window=CONTEXT_WINDOW,
        )

        assert isinstance(result, SequenceDataset)

    def test_build_sequences_uses_requested_context_window(self):
        """The convenience API should respect its context_window argument."""
        frame = make_feature_history()

        custom_context = 5

        result = build_sequences(
            frame,
            context_window=custom_context,
        )

        assert result.context_window == custom_context
        assert result.sequences.shape[1] == custom_context

    def test_build_sequences_matches_builder_output(self):
        """The convenience API should match direct SequenceBuilder output."""
        frame = make_feature_history()

        direct_builder = make_builder(
            context_window=10,
        )

        direct_result = direct_builder.build(frame)

        convenience_result = build_sequences(
            frame,
            context_window=10,
        )

        np.testing.assert_array_equal(
            direct_result.sequences,
            convenience_result.sequences,
        )

        assert direct_result.timestamps == convenience_result.timestamps
        assert direct_result.feature_names == convenience_result.feature_names
        assert (
            direct_result.context_window
            == convenience_result.context_window
        )


# ---------------------------------------------------------------------------
# NVIDIA forecasting contract
# ---------------------------------------------------------------------------


class TestNVIDIAForecastingContract:
    """Tests that the sequence builder satisfies the NVIDIA input contract."""

    def test_nvidia_feature_count(self):
        """The forecasting contract should contain the expected feature count."""
        assert FEATURE_COUNT == 14

    def test_nvidia_feature_names_are_preserved(self):
        """The exact NVIDIA feature ordering should be preserved."""
        frame = make_feature_history()

        result = build_sequences(frame)

        assert result.feature_names == FEATURE_NAMES

    def test_default_context_window_matches_nvidia_contract(self):
        """The default context window should be 20."""
        frame = make_feature_history()

        result = build_sequences(frame)

        assert result.context_window == 20
        assert result.sequences.shape[1] == 20

    def test_default_batch_shape_matches_nvidia_contract(self):
        """
        The batch shape should be (N, 20, 14), where N is the number of
        generated sequences.
        """
        frame = make_feature_history()

        result = build_sequences(frame)

        assert result.sequences.shape == (
            result.num_sequences,
            20,
            14,
        )

    def test_nvidia_feature_order_is_deterministic(self):
        """Feature ordering must not depend on DataFrame column order."""
        frame = make_feature_history()

        shuffled_columns = [
            "timestamp",
            *reversed(FEATURE_NAMES),
        ]

        shuffled = frame[shuffled_columns]

        result = build_sequences(shuffled)

        assert result.feature_names == FEATURE_NAMES

        expected = frame[FEATURE_NAMES].iloc[
            :CONTEXT_WINDOW
        ].to_numpy(dtype=np.float64)

        np.testing.assert_allclose(
            result.sequences[0],
            expected,
        )

    def test_model_facing_batch_dimension_is_present(self):
        """
        The stored tensor must include the batch dimension before the context
        and feature dimensions.
        """
        frame = make_feature_history()

        result = build_sequences(frame)

        assert result.sequences.ndim == 3
        assert result.sequences.shape[1] == CONTEXT_WINDOW
        assert result.sequences.shape[2] == FEATURE_COUNT


# ---------------------------------------------------------------------------
# Complete pipeline contract
# ---------------------------------------------------------------------------


class TestCompleteSequenceBuilderPipelineContract:
    """
    End-to-end tests for the sequence-builder portion of the forecasting
    pipeline.
    """

    def test_complete_default_pipeline(self):
        """A valid feature history should build a complete dataset."""
        frame = make_feature_history()

        result = build_sequences(frame)

        assert isinstance(result, SequenceDataset)
        assert result.num_sequences == (
            len(frame) - CONTEXT_WINDOW + 1
        )
        assert result.sequences.shape == (
            result.num_sequences,
            CONTEXT_WINDOW,
            FEATURE_COUNT,
        )
        assert result.feature_names == FEATURE_NAMES
        assert result.context_window == CONTEXT_WINDOW

    def test_complete_custom_window_pipeline(self):
        """The pipeline should work with a custom context window."""
        custom_context = 8
        frame = make_feature_history(row_count=30)

        result = build_sequences(
            frame,
            context_window=custom_context,
        )

        assert result.num_sequences == (
            len(frame) - custom_context + 1
        )

        assert result.sequences.shape == (
            result.num_sequences,
            custom_context,
            FEATURE_COUNT,
        )

    def test_sequence_and_timestamp_counts_remain_aligned(self):
        """Every generated sequence must have one timestamp window."""
        frame = make_feature_history()

        result = build_sequences(frame)

        assert result.num_sequences == len(result.timestamps)

    def test_each_sequence_has_matching_timestamp_window(self):
        """Every sequence and timestamp window must have length W."""
        frame = make_feature_history()

        result = build_sequences(frame)

        for sequence, timestamp_window in zip(
            result.sequences,
            result.timestamps,
        ):
            assert sequence.shape == (
                CONTEXT_WINDOW,
                FEATURE_COUNT,
            )

            assert len(timestamp_window) == CONTEXT_WINDOW

    def test_model_facing_tensor_is_three_dimensional(self):
        """
        The sequence tensor should be directly suitable as a batched
        sequence-model input: (batch, context, features).
        """
        frame = make_feature_history()

        result = build_sequences(frame)

        assert result.sequences.ndim == 3

    def test_batch_dimension_is_not_part_of_single_sequence_contents(self):
        """
        Individual sequence contents should have shape (W, F), while the
        stored dataset contains the batch dimension.
        """
        frame = make_feature_history()

        result = build_sequences(frame)

        first_sequence = result.sequences[0]

        assert first_sequence.shape == (
            CONTEXT_WINDOW,
            FEATURE_COUNT,
        )

        assert result.sequences.shape[0] == result.num_sequences


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------


class TestReproducibility:
    """Tests for deterministic sequence construction."""

    def test_same_input_produces_same_sequences(self):
        """Repeated builds from the same input should be identical."""
        frame = make_feature_history()

        builder = make_builder()

        first = builder.build(frame)
        second = builder.build(frame)

        np.testing.assert_array_equal(
            first.sequences,
            second.sequences,
        )

    def test_same_input_produces_same_timestamps(self):
        """Repeated builds should produce identical timestamp windows."""
        frame = make_feature_history()

        builder = make_builder()

        first = builder.build(frame)
        second = builder.build(frame)

        assert first.timestamps == second.timestamps

    def test_feature_order_is_reproducible(self):
        """Repeated builds should preserve feature ordering."""
        frame = make_feature_history()

        first = build_sequences(frame)
        second = build_sequences(frame)

        assert first.feature_names == second.feature_names


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    """Tests for important sequence-builder edge cases."""

    def test_single_sequence_history(self):
        """A history exactly W rows long should produce one sequence."""
        frame = make_feature_history(
            row_count=CONTEXT_WINDOW,
        )

        result = build_sequences(frame)

        assert result.num_sequences == 1
        assert len(result.timestamps) == 1

    def test_large_history_produces_expected_sequence_count(self):
        """Large valid histories should follow N-W+1 exactly."""
        row_count = 500
        frame = make_feature_history(row_count=row_count)

        result = build_sequences(frame)

        assert result.num_sequences == (
            row_count - CONTEXT_WINDOW + 1
        )

    def test_fractional_timestamps_are_supported(self):
        """Numeric fractional timestamps should be accepted."""
        frame = make_feature_history(
            start_timestamp=1_000.5,
            timestamp_step=0.25,
        )

        result = build_sequences(frame)

        assert result.num_sequences > 0

    def test_negative_numeric_feature_values_are_supported(self):
        """Negative numeric features are valid numeric inputs."""
        frame = make_feature_history()

        frame.loc[:, FEATURE_NAMES[0]] = np.linspace(
            -10.0,
            10.0,
            len(frame),
        )

        result = build_sequences(frame)

        assert result.num_sequences > 0

    def test_zero_feature_values_are_supported(self):
        """Zero-valued features should be accepted."""
        frame = make_feature_history()

        frame.loc[:, FEATURE_NAMES[0]] = 0.0

        result = build_sequences(frame)

        assert result.num_sequences > 0

    def test_duplicate_index_values_do_not_define_sequence_order(self):
        """
        Sequence ordering should be based on timestamps rather than requiring
        a special pandas index.
        """
        frame = make_feature_history()

        frame.index = [0] * len(frame)

        result = build_sequences(frame)

        assert result.num_sequences == (
            len(frame) - CONTEXT_WINDOW + 1
        )

    def test_non_default_index_is_supported(self):
        """A non-default pandas index should not affect sequence generation."""
        frame = make_feature_history()

        frame.index = range(100, 100 + len(frame))

        result = build_sequences(frame)

        assert result.num_sequences == (
            len(frame) - CONTEXT_WINDOW + 1
        )


# ---------------------------------------------------------------------------
# Regression tests for previously identified failures
# ---------------------------------------------------------------------------


class TestRegressionCases:
    """
    Regression coverage for issues discovered while validating the original
    sequence-builder test suite.
    """

    def test_config_does_not_require_sort_output(self):
        """
        The test suite should rely only on the actual SequenceBuilderConfig
        API and should not require a nonexistent sort_output field.
        """
        config = SequenceBuilderConfig()

        assert hasattr(config, "context_window")
        assert hasattr(config, "require_chronological_order")
        assert hasattr(config, "require_unique_timestamps")
        assert hasattr(config, "allow_missing_values")
        assert hasattr(config, "allow_infinite_values")
        assert hasattr(config, "copy_input")
        assert hasattr(config, "dtype")

    def test_config_does_not_require_drop_invalid_rows(self):
        """
        The test suite should rely only on the actual SequenceBuilderConfig
        API and should not require a nonexistent drop_invalid_rows field.
        """
        config = SequenceBuilderConfig()

        assert not hasattr(config, "drop_invalid_rows")

    def test_timestamps_are_not_assumed_to_be_one_dimensional(self):
        """
        SequenceDataset timestamps are represented as one timestamp window
        per sequence.
        """
        frame = make_feature_history()

        result = build_sequences(frame)

        assert len(result.timestamps) == result.num_sequences
        assert all(
            len(window) == CONTEXT_WINDOW
            for window in result.timestamps
        )

    def test_convenience_api_accepts_context_window_directly(self):
        """
        build_sequences() accepts context_window directly rather than a
        config keyword argument.
        """
        frame = make_feature_history()

        result = build_sequences(
            frame,
            context_window=10,
        )

        assert result.context_window == 10

    def test_sequence_output_is_independent_from_input(self):
        """
        Mutating the original feature history after construction should not
        modify the already-built sequence dataset.
        """
        frame = make_feature_history()

        result = build_sequences(frame)

        original_value = result.sequences[0, 0, 0]

        frame.loc[0, FEATURE_NAMES[0]] = -999999.0

        assert result.sequences[0, 0, 0] == original_value

    def test_single_sequence_contents_have_expected_shape(self):
        """
        A single sequence extracted from the batch should have shape (W, F).
        """
        frame = make_feature_history()

        result = build_sequences(frame)

        assert result.sequences[0].shape == (
            CONTEXT_WINDOW,
            FEATURE_COUNT,
        )

    def test_full_dataset_tensor_has_expected_batch_shape(self):
        """
        The complete tensor should have shape (N, W, F).
        """
        frame = make_feature_history()

        result = build_sequences(frame)

        assert result.sequences.shape == (
            result.num_sequences,
            CONTEXT_WINDOW,
            FEATURE_COUNT,
        )

    def test_output_array_is_mutable_without_affecting_input(self):
        """
        The current SequenceDataset contract does not require returned
        sequence arrays to be read-only.
        """
        frame = make_feature_history()

        result = build_sequences(frame)

        original_input_value = frame.loc[0, FEATURE_NAMES[0]]

        result.sequences[0, 0, 0] = -888888.0

        assert result.sequences[0, 0, 0] == -888888.0
        assert frame.loc[0, FEATURE_NAMES[0]] == original_input_value


# ---------------------------------------------------------------------------
# Final integration smoke test
# ---------------------------------------------------------------------------


class TestSequenceBuilderSmoke:
    """Small end-to-end smoke test for the complete builder."""

    def test_sequence_builder_end_to_end(self):
        """The production sequence-building path should work end-to-end."""
        frame = make_feature_history(
            row_count=50,
        )

        result = build_sequences(frame)

        assert isinstance(result, SequenceDataset)
        assert result.num_sequences == 31
        assert result.context_window == 20
        assert result.feature_names == FEATURE_NAMES
        assert result.sequences.shape == (31, 20, 14)
        assert len(result.timestamps) == 31

        for timestamp_window in result.timestamps:
            assert len(timestamp_window) == 20
            assert timestamp_window == sorted(timestamp_window)

