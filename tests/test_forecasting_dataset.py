"""
tests/test_forecasting_dataset.py

Comprehensive Phase 9 tests for the forecasting-dataset alignment layer.

Responsibilities
----------------
- Validate exact timestamp-based sequence/target alignment.
- Validate chronological and leakage-safe dataset construction.
- Validate the ForecastingDataset model-facing API.
- Validate the canonical NVIDIA forecasting data contract.
- Validate configuration, dtype, copying, determinism, and edge cases.
- Validate invalid-input rejection and output integrity.

This module does not test:

- feature extraction
- target calculation
- chronological train/validation/test splitting
- normalization
- forecasting model architecture
- model training
- model inference
- execution logic
"""

from __future__ import annotations

import numpy as np
import pytest

from data.contracts import SequenceDataset, TargetDataset
from data.forecasting_dataset import (
    ForecastingDataset,
    ForecastingDatasetBuilder,
    ForecastingDatasetConfig,
    align_sequence_targets,
    build_forecasting_dataset,
)


# =============================================================================
# Constants
# =============================================================================

FEATURE_NAMES = (
    "mid_price_return",
    "spread_bps",
    "best_bid_size",
    "best_ask_size",
    "depth_imbalance_l1",
    "depth_imbalance_multilevel",
    "ofi_instant",
    "ofi_sum_5",
    "trade_volume_imbalance",
    "momentum_ret_5",
    "momentum_ret_20",
    "volatility_std_10",
    "micro_price",
    "micro_price_divergence",
)

CONTEXT_WINDOW = 20
FORECAST_HORIZON = 5
TARGET_NAME = "future_mid_price_return"
PRICE_COLUMN = "mid_price"
START_TIMESTAMP = 1000.0


# =============================================================================
# Factories
# =============================================================================


def make_sequence_dataset(
    *,
    num_rows: int = 40,
    context_window: int = CONTEXT_WINDOW,
    feature_names: tuple[str, ...] = FEATURE_NAMES,
    start_timestamp: float = START_TIMESTAMP,
    timestamp_step: float = 1.0,
) -> SequenceDataset:
    """Create deterministic rolling sequences with traceable values."""

    if num_rows < context_window:
        raise ValueError(
            "num_rows must be at least context_window."
        )

    num_sequences = num_rows - context_window + 1
    num_features = len(feature_names)

    sequences = np.empty(
        (
            num_sequences,
            context_window,
            num_features,
        ),
        dtype=np.float64,
    )

    for sample in range(num_sequences):
        for timestep in range(context_window):
            for feature in range(num_features):
                sequences[
                    sample,
                    timestep,
                    feature,
                ] = (
                    sample * 1000
                    + timestep * 10
                    + feature
                )

    timestamps = (
        start_timestamp
        + np.arange(num_rows, dtype=np.float64)
        * timestamp_step
    )

    windows = [
        timestamps[
            index : index + context_window
        ].tolist()
        for index in range(num_sequences)
    ]

    return SequenceDataset(
        sequences=sequences,
        timestamps=windows,
        feature_names=list(feature_names),
        context_window=context_window,
    )


def make_target_dataset(
    *,
    num_rows: int = 40,
    forecast_horizon: int = FORECAST_HORIZON,
    target_name: str = TARGET_NAME,
    price_column: str = PRICE_COLUMN,
    start_timestamp: float = START_TIMESTAMP,
    timestamp_step: float = 1.0,
    target_offset: float = 0.0,
) -> TargetDataset:
    """Create deterministic timestamp-keyed targets."""

    timestamps = (
        start_timestamp
        + np.arange(num_rows, dtype=np.float64)
        * timestamp_step
    )

    targets = (
        np.arange(num_rows, dtype=np.float64)
        * 0.001
        + target_offset
    )

    return TargetDataset(
        targets=targets,
        timestamps=timestamps.tolist(),
        target_name=target_name,
        forecast_horizon=forecast_horizon,
        price_column=price_column,
    )


def make_forecasting_dataset(
    *,
    num_rows: int = 40,
    context_window: int = CONTEXT_WINDOW,
    forecast_horizon: int = FORECAST_HORIZON,
) -> ForecastingDataset:
    """Create a complete aligned forecasting dataset."""

    return build_forecasting_dataset(
        make_sequence_dataset(
            num_rows=num_rows,
            context_window=context_window,
        ),
        make_target_dataset(
            num_rows=num_rows,
            forecast_horizon=forecast_horizon,
        ),
    )


# =============================================================================
# Construction and Shape
# =============================================================================


class TestConstructionAndShape:
    """Validate construction and fundamental array shapes."""

    def test_valid_inputs_build_dataset(self):
        """Valid sequence and target datasets produce a forecasting dataset."""

        dataset = make_forecasting_dataset()

        assert isinstance(
            dataset,
            ForecastingDataset,
        )

    def test_expected_sample_count(self):
        """Forty observations and a twenty-step context produce 21 sequences."""

        dataset = make_forecasting_dataset(
            num_rows=40,
            context_window=20,
        )

        assert dataset.num_samples == 21

    def test_expected_feature_count(self):
        """The forecasting dataset contains all 14 NVIDIA features."""

        dataset = make_forecasting_dataset()

        assert dataset.num_features == 14

    def test_complete_shape(self):
        """The complete feature tensor has shape (N, 20, 14)."""

        dataset = make_forecasting_dataset()

        assert dataset.shape == (
            21,
            20,
            14,
        )

    def test_single_input_shape(self):
        """One model input has shape (20, 14)."""

        dataset = make_forecasting_dataset()

        assert dataset.input_shape == (
            20,
            14,
        )

    def test_model_input_shape_alias_matches_input_shape(self):
        """model_input_shape must remain an explicit alias of input_shape."""

        dataset = make_forecasting_dataset()

        assert dataset.model_input_shape == dataset.input_shape

    def test_target_shape(self):
        """Targets have one value per aligned forecasting sample."""

        dataset = make_forecasting_dataset()

        assert dataset.target_shape == (
            dataset.num_samples,
        )

    def test_timestamp_shape(self):
        """Timestamps have one value per aligned forecasting sample."""

        dataset = make_forecasting_dataset()

        assert dataset.timestamp_shape == (
            dataset.num_samples,
        )

    def test_sequence_dtype_is_float(self):
        """Feature sequences are represented numerically as floating-point data."""

        dataset = make_forecasting_dataset()

        assert np.issubdtype(
            dataset.sequences.dtype,
            np.floating,
        )

    def test_target_dtype_is_float(self):
        """Targets are represented as floating-point data."""

        dataset = make_forecasting_dataset()

        assert np.issubdtype(
            dataset.targets.dtype,
            np.floating,
        )

    def test_timestamp_dtype_is_float(self):
        """Timestamps are represented as floating-point data."""

        dataset = make_forecasting_dataset()

        assert np.issubdtype(
            dataset.timestamps.dtype,
            np.floating,
        )

    @pytest.mark.parametrize(
        "context_window",
        [1, 2, 5, 10, 20, 25],
    )
    def test_valid_context_windows(
        self,
        context_window: int,
    ):
        """Different valid context windows preserve the expected tensor shape."""

        num_rows = context_window + 10

        dataset = make_forecasting_dataset(
            num_rows=num_rows,
            context_window=context_window,
        )

        expected_sequences = (
            num_rows - context_window + 1
        )

        assert dataset.shape == (
            expected_sequences,
            context_window,
            14,
        )

    def test_context_equal_history_length_produces_one_sample(self):
        """A context equal to the complete history produces one sequence."""

        dataset = make_forecasting_dataset(
            num_rows=20,
            context_window=20,
        )

        assert dataset.shape == (
            1,
            20,
            14,
        )

    def test_context_window_one_is_valid(self):
        """A one-step context remains a valid forecasting input."""

        dataset = make_forecasting_dataset(
            num_rows=10,
            context_window=1,
        )

        assert dataset.shape == (
            10,
            1,
            14,
        )


# =============================================================================
# Timestamp Alignment
# =============================================================================


class TestTimestampAlignment:
    """Validate exact timestamp-based sequence/target alignment."""

    def test_first_sample_uses_sequence_end_timestamp(self):
        """The first sequence must align to its ending timestamp."""

        dataset = make_forecasting_dataset()

        assert dataset.timestamps[0] == pytest.approx(
            1019.0
        )

    def test_first_target_matches_sequence_end_timestamp(self):
        """The first target must correspond to timestamp 1019."""

        dataset = make_forecasting_dataset()

        assert dataset.targets[0] == pytest.approx(
            0.019
        )

    def test_second_sample_uses_second_sequence_end(self):
        """The second sequence must align to timestamp 1020."""

        dataset = make_forecasting_dataset()

        assert dataset.timestamps[1] == pytest.approx(
            1020.0
        )

        assert dataset.targets[1] == pytest.approx(
            0.020
        )

    def test_alignment_is_not_index_based(self):
        """
        A shifted target value makes positional alignment visibly incorrect.

        The builder must use timestamp identity rather than equal array
        positions.
        """

        dataset = build_forecasting_dataset(
            make_sequence_dataset(),
            make_target_dataset(
                target_offset=10.0,
            ),
        )

        assert dataset.targets[0] == pytest.approx(
            10.019
        )

    def test_every_output_timestamp_is_a_sequence_end(self):
        """Every output timestamp equals the corresponding sequence end."""

        sequences = make_sequence_dataset()
        targets = make_target_dataset()

        dataset = build_forecasting_dataset(
            sequences,
            targets,
        )

        expected = np.asarray(
            [
                window[-1]
                for window in sequences.timestamps
            ],
            dtype=np.float64,
        )

        np.testing.assert_array_equal(
            dataset.timestamps,
            expected,
        )

    def test_every_target_is_mapped_by_timestamp(self):
        """Every aligned target follows its timestamp mapping."""

        dataset = make_forecasting_dataset()

        expected = (
            dataset.timestamps
            - START_TIMESTAMP
        ) * 0.001

        np.testing.assert_allclose(
            dataset.targets,
            expected,
        )

    def test_early_targets_are_ignored(self):
        """Targets occurring before the first sequence end are ignored."""

        dataset = build_forecasting_dataset(
            make_sequence_dataset(),
            make_target_dataset(
                start_timestamp=990.0,
            ),
        )

        assert dataset.timestamps[0] == pytest.approx(
            1019.0
        )

        assert dataset.targets[0] == pytest.approx(
            0.029
        )

    def test_extra_future_targets_do_not_create_samples(self):
        """Extra target timestamps cannot create feature sequences."""

        sequences = make_sequence_dataset()

        dataset = build_forecasting_dataset(
            sequences,
            make_target_dataset(
                num_rows=100,
            ),
        )

        assert dataset.num_samples == (
            sequences.num_sequences
        )

    def test_non_unit_timestamp_spacing_is_supported(self):
        """Timestamp alignment works with non-unit intervals."""

        dataset = build_forecasting_dataset(
            make_sequence_dataset(
                num_rows=30,
                context_window=10,
                timestamp_step=0.5,
            ),
            make_target_dataset(
                num_rows=30,
                timestamp_step=0.5,
            ),
        )

        assert dataset.timestamps[0] == pytest.approx(
            1004.5
        )

        assert dataset.timestamps[1] == pytest.approx(
            1005.0
        )


# =============================================================================
# Temporal Safety
# =============================================================================


class TestTemporalSafety:
    """Validate chronological and leakage-safe output."""

    def test_output_is_strictly_chronological(self):
        """Aligned timestamps must increase strictly."""

        dataset = make_forecasting_dataset()

        assert np.all(
            np.diff(dataset.timestamps) > 0
        )

    def test_output_timestamps_are_unique(self):
        """Aligned timestamps must be unique."""

        dataset = make_forecasting_dataset()

        assert len(
            np.unique(dataset.timestamps)
        ) == dataset.num_samples

    def test_target_timestamp_is_current_observation_timestamp(self):
        """
        Targets remain attached to the current observation timestamp t.

        The future horizon is represented by the target value, not by shifting
        the dataset timestamp to t+h.
        """

        dataset = make_forecasting_dataset()

        assert dataset.timestamps[0] == pytest.approx(
            1019.0
        )

        assert dataset.timestamps[-1] == pytest.approx(
            1039.0
        )

    def test_alignment_never_fabricates_target_values(self):
        """Every output timestamp must exist in the target dataset."""

        target_dataset = make_target_dataset(
            num_rows=25,
        )

        dataset = build_forecasting_dataset(
            make_sequence_dataset(),
            target_dataset,
        )

        valid_timestamps = set(
            target_dataset.timestamps
        )

        assert all(
            float(timestamp) in valid_timestamps
            for timestamp in dataset.timestamps
        )

    def test_disjoint_timestamps_are_rejected(self):
        """Completely disjoint datasets cannot produce aligned samples."""

        with pytest.raises(ValueError):
            build_forecasting_dataset(
                make_sequence_dataset(
                    start_timestamp=1000.0,
                ),
                make_target_dataset(
                    start_timestamp=2000.0,
                ),
            )

    def test_strict_alignment_rejects_unmatched_sequences(self):
        """Strict mode must reject any unmatched sequence timestamp."""

        with pytest.raises(ValueError):
            build_forecasting_dataset(
                make_sequence_dataset(),
                make_target_dataset(
                    num_rows=25,
                ),
                config=ForecastingDatasetConfig(
                    drop_unaligned=False,
                ),
            )

    def test_default_alignment_drops_unmatched_sequences(self):
        """Default alignment drops sequence windows without targets."""

        sequences = make_sequence_dataset()

        dataset = build_forecasting_dataset(
            sequences,
            make_target_dataset(
                num_rows=25,
            ),
        )

        assert dataset.num_samples < (
            sequences.num_sequences
        )


# =============================================================================
# Metadata and NVIDIA Contract
# =============================================================================


class TestMetadataAndNvidiaContract:
    """Validate metadata preservation and canonical NVIDIA structure."""

    def test_feature_names_are_preserved(self):
        """The exact ordered feature schema is preserved."""

        dataset = make_forecasting_dataset()

        assert dataset.feature_names == FEATURE_NAMES

    def test_context_window_is_preserved(self):
        """The sequence context remains part of the dataset contract."""

        dataset = make_forecasting_dataset()

        assert dataset.context_window == 20

    def test_forecast_horizon_is_preserved(self):
        """The target forecast horizon is preserved."""

        dataset = make_forecasting_dataset()

        assert dataset.forecast_horizon == 5

    def test_target_name_is_preserved(self):
        """The target name is preserved."""

        dataset = make_forecasting_dataset()

        assert dataset.target_name == TARGET_NAME

    def test_price_column_is_preserved(self):
        """The source price-column metadata is preserved."""

        dataset = make_forecasting_dataset()

        assert dataset.price_column == PRICE_COLUMN

    def test_custom_metadata_is_preserved(self):
        """Custom metadata survives the alignment stage."""

        custom_features = (
            "feature_a",
            "feature_b",
        )

        dataset = build_forecasting_dataset(
            make_sequence_dataset(
                num_rows=10,
                context_window=4,
                feature_names=custom_features,
            ),
            make_target_dataset(
                num_rows=10,
                forecast_horizon=3,
                target_name="custom_target",
                price_column="custom_price",
            ),
        )

        assert dataset.feature_names == custom_features
        assert dataset.context_window == 4
        assert dataset.forecast_horizon == 3
        assert dataset.target_name == "custom_target"
        assert dataset.price_column == "custom_price"

    def test_nvidia_input_shape_is_20_by_14(self):
        """The canonical NVIDIA single-sample input is (20, 14)."""

        dataset = make_forecasting_dataset()

        assert dataset.input_shape == (
            20,
            14,
        )

    def test_nvidia_batch_shape_is_n_by_20_by_14(self):
        """The canonical NVIDIA batch shape is (N, 20, 14)."""

        dataset = make_forecasting_dataset(
            num_rows=100,
        )

        assert dataset.shape[1:] == (
            20,
            14,
        )

    def test_nvidia_target_is_one_dimensional(self):
        """The forecasting target remains one-dimensional."""

        dataset = make_forecasting_dataset()

        assert dataset.targets.ndim == 1

    def test_nvidia_feature_order_is_exact(self):
        """The canonical 14-feature order remains unchanged."""

        dataset = make_forecasting_dataset()

        assert dataset.feature_names == FEATURE_NAMES


# =============================================================================
# Model-Facing API
# =============================================================================


class TestModelFacingAPI:
    """Validate APIs consumed by downstream model integrations."""

    def test_get_sample_returns_sequence_target_timestamp(self):
        """get_sample returns all three components of one aligned sample."""

        dataset = make_forecasting_dataset()

        sequence, target, timestamp = (
            dataset.get_sample(0)
        )

        assert sequence.shape == (
            20,
            14,
        )

        assert target == pytest.approx(
            0.019
        )

        assert timestamp == pytest.approx(
            1019.0
        )

    def test_get_sample_middle_index_is_correct(self):
        """get_sample works for an interior sample."""

        dataset = make_forecasting_dataset()

        sequence, target, timestamp = (
            dataset.get_sample(5)
        )

        assert sequence.shape == (
            20,
            14,
        )

        assert target == pytest.approx(
            0.024
        )

        assert timestamp == pytest.approx(
            1024.0
        )

    def test_get_sample_rejects_negative_index(self):
        """Negative indices are rejected by the explicit dataset API."""

        dataset = make_forecasting_dataset()

        with pytest.raises(IndexError):
            dataset.get_sample(-1)

    def test_get_sample_rejects_upper_bound(self):
        """Indices equal to num_samples are rejected."""

        dataset = make_forecasting_dataset()

        with pytest.raises(IndexError):
            dataset.get_sample(
                dataset.num_samples
            )

    @pytest.mark.parametrize(
        "index",
        [
            0.0,
            "0",
            True,
            None,
        ],
    )
    def test_get_sample_rejects_non_integer_index(
        self,
        index,
    ):
        """get_sample requires a real integer index."""

        dataset = make_forecasting_dataset()

        with pytest.raises(TypeError):
            dataset.get_sample(index)

    def test_to_model_input_returns_single_sequence(self):
        """to_model_input returns one (20, 14) model input."""

        dataset = make_forecasting_dataset()

        model_input = dataset.to_model_input(0)

        assert model_input.shape == (
            20,
            14,
        )

        np.testing.assert_array_equal(
            model_input,
            dataset.sequences[0],
        )

    def test_to_model_input_excludes_target_and_timestamp(self):
        """Single model input contains only the feature sequence."""

        dataset = make_forecasting_dataset()

        assert dataset.to_model_input(0).ndim == 2

    def test_to_model_batch_returns_complete_tensor(self):
        """to_model_batch returns the complete feature tensor."""

        dataset = make_forecasting_dataset()

        batch = dataset.to_model_batch()

        assert batch.shape == dataset.shape

        np.testing.assert_array_equal(
            batch,
            dataset.sequences,
        )

    def test_to_model_input_copy_is_independent(self):
        """The default model-input copy does not mutate the dataset."""

        dataset = make_forecasting_dataset()

        result = dataset.to_model_input(
            0,
            copy=True,
        )

        original = dataset.sequences[
            0,
            0,
            0,
        ]

        result[0, 0] = 999999.0

        assert dataset.sequences[
            0,
            0,
            0,
        ] == original

    def test_to_model_batch_copy_is_independent(self):
        """The default model-batch copy does not mutate the dataset."""

        dataset = make_forecasting_dataset()

        result = dataset.to_model_batch(
            copy=True,
        )

        original = dataset.sequences[
            0,
            0,
            0,
        ]

        result[0, 0, 0] = 999999.0

        assert dataset.sequences[
            0,
            0,
            0,
        ] == original

    def test_to_model_input_copy_false_exposes_underlying_array(self):
        """copy=False intentionally returns the underlying sequence."""

        dataset = make_forecasting_dataset()

        result = dataset.to_model_input(
            0,
            copy=False,
        )

        result[0, 0] = 999999.0

        assert dataset.sequences[
            0,
            0,
            0,
        ] == 999999.0

    def test_to_model_batch_copy_false_exposes_underlying_array(self):
        """copy=False intentionally returns the underlying tensor."""

        dataset = make_forecasting_dataset()

        result = dataset.to_model_batch(
            copy=False,
        )

        result[0, 0, 0] = 999999.0

        assert dataset.sequences[
            0,
            0,
            0,
        ] == 999999.0


# =============================================================================
# Dataset Properties and Copying
# =============================================================================


class TestDatasetProperties:
    """Validate compact properties and independent dataset copies."""

    def test_num_samples_matches_first_sequence_dimension(self):
        """num_samples reflects the batch dimension."""

        dataset = make_forecasting_dataset()

        assert dataset.num_samples == (
            dataset.sequences.shape[0]
        )

    def test_num_features_matches_last_sequence_dimension(self):
        """num_features reflects the feature dimension."""

        dataset = make_forecasting_dataset()

        assert dataset.num_features == (
            dataset.sequences.shape[2]
        )

    def test_shape_matches_sequence_shape(self):
        """shape exactly describes the feature tensor."""

        dataset = make_forecasting_dataset()

        assert dataset.shape == dataset.sequences.shape

    def test_target_shape_matches_targets(self):
        """target_shape exactly describes the target array."""

        dataset = make_forecasting_dataset()

        assert dataset.target_shape == (
            dataset.targets.shape
        )

    def test_timestamp_shape_matches_timestamps(self):
        """timestamp_shape exactly describes the timestamp array."""

        dataset = make_forecasting_dataset()

        assert dataset.timestamp_shape == (
            dataset.timestamps.shape
        )

    def test_target_values_alias_matches_targets(self):
        """target_values exposes the same target values."""

        dataset = make_forecasting_dataset()

        np.testing.assert_array_equal(
            dataset.target_values,
            dataset.targets,
        )

    def test_feature_tensor_alias_matches_sequences(self):
        """feature_tensor exposes the same feature tensor."""

        dataset = make_forecasting_dataset()

        np.testing.assert_array_equal(
            dataset.feature_tensor,
            dataset.sequences,
        )

    def test_earliest_timestamp_is_first(self):
        """earliest_timestamp returns the first aligned timestamp."""

        dataset = make_forecasting_dataset()

        assert dataset.earliest_timestamp == pytest.approx(
            dataset.timestamps[0]
        )

    def test_latest_timestamp_is_last(self):
        """latest_timestamp returns the final aligned timestamp."""

        dataset = make_forecasting_dataset()

        assert dataset.latest_timestamp == pytest.approx(
            dataset.timestamps[-1]
        )

    def test_copy_returns_independent_dataset(self):
        """copy() returns a separate ForecastingDataset."""

        dataset = make_forecasting_dataset()

        copied = dataset.copy()

        assert copied is not dataset

        np.testing.assert_array_equal(
            copied.sequences,
            dataset.sequences,
        )

        np.testing.assert_array_equal(
            copied.targets,
            dataset.targets,
        )

        np.testing.assert_array_equal(
            copied.timestamps,
            dataset.timestamps,
        )

    def test_copy_is_independent_for_sequences(self):
        """Changing copied sequences does not affect the original."""

        dataset = make_forecasting_dataset()
        copied = dataset.copy()

        copied.sequences[
            0,
            0,
            0,
        ] = 123456.0

        assert dataset.sequences[
            0,
            0,
            0,
        ] != 123456.0

    def test_copy_is_independent_for_targets(self):
        """Changing copied targets does not affect the original."""

        dataset = make_forecasting_dataset()
        copied = dataset.copy()

        copied.targets[0] = 123456.0

        assert dataset.targets[0] != 123456.0

    def test_copy_is_independent_for_timestamps(self):
        """Changing copied timestamps does not affect the original."""

        dataset = make_forecasting_dataset()
        copied = dataset.copy()

        copied.timestamps[0] = 123456.0

        assert dataset.timestamps[0] != 123456.0


# =============================================================================
# Configuration and Dtype
# =============================================================================


class TestConfiguration:
    """Validate ForecastingDatasetBuilder configuration behavior."""

    def test_default_configuration(self):
        """Default configuration matches the pipeline contract."""

        config = ForecastingDatasetConfig()

        assert config.drop_unaligned is True
        assert config.require_chronological_order is True
        assert config.require_unique_timestamps is True
        assert config.copy_arrays is True
        assert config.dtype == "float64"

    def test_custom_flags_are_preserved(self):
        """Custom configuration flags remain unchanged."""

        config = ForecastingDatasetConfig(
            drop_unaligned=False,
            require_chronological_order=False,
            require_unique_timestamps=False,
            copy_arrays=False,
        )

        assert config.drop_unaligned is False
        assert config.require_chronological_order is False
        assert config.require_unique_timestamps is False
        assert config.copy_arrays is False

    def test_float32_output_is_supported(self):
        """The builder can produce float32 arrays."""

        dataset = build_forecasting_dataset(
            make_sequence_dataset(),
            make_target_dataset(),
            config=ForecastingDatasetConfig(
                dtype="float32",
            ),
        )

        assert dataset.sequences.dtype == np.float32
        assert dataset.targets.dtype == np.float32
        assert dataset.timestamps.dtype == np.float32

    def test_invalid_dtype_is_rejected(self):
        """Invalid NumPy dtypes are rejected immediately."""

        with pytest.raises(ValueError):
            ForecastingDatasetConfig(
                dtype="not-a-real-dtype",
            )

    def test_copy_arrays_true_is_default(self):
        """Array copying is enabled by default."""

        assert ForecastingDatasetConfig().copy_arrays is True

    def test_drop_unaligned_false_requires_complete_matching(self):
        """Strict alignment rejects unmatched sequence endpoints."""

        with pytest.raises(ValueError):
            build_forecasting_dataset(
                make_sequence_dataset(),
                make_target_dataset(
                    num_rows=25,
                ),
                config=ForecastingDatasetConfig(
                    drop_unaligned=False,
                ),
            )


# =============================================================================
# Input Validation
# =============================================================================


class TestInputValidation:
    """Validate rejection of invalid builder and source-contract inputs."""

    def test_invalid_sequence_dataset_type(self):
        """The builder rejects non-SequenceDataset input."""

        with pytest.raises(TypeError):
            build_forecasting_dataset(
                "invalid",
                make_target_dataset(),
            )

    def test_invalid_target_dataset_type(self):
        """The builder rejects non-TargetDataset input."""

        with pytest.raises(TypeError):
            build_forecasting_dataset(
                make_sequence_dataset(),
                "invalid",
            )

    def test_empty_sequence_dataset_is_rejected_by_source_contract(self):
        """The source SequenceDataset contract rejects zero samples."""

        with pytest.raises(ValueError):
            SequenceDataset(
                sequences=np.empty(
                    (
                        0,
                        20,
                        14,
                    ),
                    dtype=np.float64,
                ),
                timestamps=[],
                feature_names=list(FEATURE_NAMES),
                context_window=20,
            )

    def test_nonpositive_target_horizon_is_rejected(self):
        """A target forecast horizon must be positive."""

        with pytest.raises(ValueError):
            TargetDataset(
                targets=np.array(
                    [0.1]
                ),
                timestamps=[
                    1.0
                ],
                target_name=TARGET_NAME,
                forecast_horizon=0,
            )

    def test_empty_target_name_is_rejected(self):
        """TargetDataset requires a non-empty target name."""

        with pytest.raises(ValueError):
            TargetDataset(
                targets=np.array(
                    [0.1]
                ),
                timestamps=[
                    1.0
                ],
                target_name="",
                forecast_horizon=5,
            )

    def test_empty_price_column_is_rejected(self):
        """TargetDataset requires a non-empty price column."""

        with pytest.raises(ValueError):
            TargetDataset(
                targets=np.array(
                    [0.1]
                ),
                timestamps=[
                    1.0
                ],
                target_name=TARGET_NAME,
                forecast_horizon=5,
                price_column="",
            )

    def test_duplicate_feature_names_are_rejected(self):
        """Feature schemas cannot contain duplicate names."""

        with pytest.raises(ValueError):
            SequenceDataset(
                sequences=np.ones(
                    (
                        2,
                        3,
                        2,
                    ),
                    dtype=np.float64,
                ),
                timestamps=[
                    [
                        1.0,
                        2.0,
                        3.0,
                    ],
                    [
                        2.0,
                        3.0,
                        4.0,
                    ],
                ],
                feature_names=[
                    "duplicate",
                    "duplicate",
                ],
                context_window=3,
            )

    def test_empty_feature_names_are_rejected(self):
        """A SequenceDataset requires at least one feature."""

        with pytest.raises(ValueError):
            SequenceDataset(
                sequences=np.ones(
                    (
                        2,
                        3,
                        0,
                    ),
                    dtype=np.float64,
                ),
                timestamps=[
                    [
                        1.0,
                        2.0,
                        3.0,
                    ],
                    [
                        2.0,
                        3.0,
                        4.0,
                    ],
                ],
                feature_names=[],
                context_window=3,
            )


# =============================================================================
# Source Immutability and Determinism
# =============================================================================


class TestImmutabilityAndDeterminism:
    """Validate deterministic and non-destructive alignment."""

    def test_repeated_builds_are_identical(self):
        """Repeated builds produce exactly the same arrays."""

        sequences = make_sequence_dataset()
        targets = make_target_dataset()

        first = build_forecasting_dataset(
            sequences,
            targets,
        )

        second = build_forecasting_dataset(
            sequences,
            targets,
        )

        np.testing.assert_array_equal(
            first.sequences,
            second.sequences,
        )

        np.testing.assert_array_equal(
            first.targets,
            second.targets,
        )

        np.testing.assert_array_equal(
            first.timestamps,
            second.timestamps,
        )

    def test_source_sequence_values_are_not_modified(self):
        """Building the dataset does not mutate source sequences."""

        sequences = make_sequence_dataset()

        before = sequences.sequences.copy()

        build_forecasting_dataset(
            sequences,
            make_target_dataset(),
        )

        np.testing.assert_array_equal(
            sequences.sequences,
            before,
        )

    def test_source_target_values_are_not_modified(self):
        """Building the dataset does not mutate source targets."""

        targets = make_target_dataset()

        before = targets.targets.copy()

        build_forecasting_dataset(
            make_sequence_dataset(),
            targets,
        )

        np.testing.assert_array_equal(
            targets.targets,
            before,
        )

    def test_source_target_timestamps_are_not_modified(self):
        """Building the dataset does not mutate target timestamps."""

        targets = make_target_dataset()

        before = np.asarray(
            targets.timestamps
        ).copy()

        build_forecasting_dataset(
            make_sequence_dataset(),
            targets,
        )

        np.testing.assert_array_equal(
            targets.timestamps,
            before,
        )

    def test_copy_arrays_true_separates_sequence_storage(self):
        """Default copied arrays are independent of source storage."""

        sequences = make_sequence_dataset()

        dataset = build_forecasting_dataset(
            sequences,
            make_target_dataset(),
            config=ForecastingDatasetConfig(
                copy_arrays=True,
            ),
        )

        original = dataset.sequences[
            0,
            0,
            0,
        ]

        sequences.sequences[
            0,
            0,
            0,
        ] = 999999.0

        assert dataset.sequences[
            0,
            0,
            0,
        ] == original

    def test_copy_arrays_true_separates_target_storage(self):
        """Default copied target arrays are independent of source storage."""

        targets = make_target_dataset()

        dataset = build_forecasting_dataset(
            make_sequence_dataset(),
            targets,
            config=ForecastingDatasetConfig(
                copy_arrays=True,
            ),
        )

        original = dataset.targets[0]

        targets.targets[19] = 999999.0

        assert dataset.targets[0] == original


# =============================================================================
# Output Integrity
# =============================================================================


class TestOutputIntegrity:
    """Validate final aligned-array invariants."""

    def test_features_are_finite(self):
        """All feature values are finite."""

        dataset = make_forecasting_dataset()

        assert np.isfinite(
            dataset.sequences
        ).all()

    def test_targets_are_finite(self):
        """All target values are finite."""

        dataset = make_forecasting_dataset()

        assert np.isfinite(
            dataset.targets
        ).all()

    def test_timestamps_are_finite(self):
        """All timestamps are finite."""

        dataset = make_forecasting_dataset()

        assert np.isfinite(
            dataset.timestamps
        ).all()

    def test_sample_counts_match(self):
        """Every sample has one target and one timestamp."""

        dataset = make_forecasting_dataset()

        assert dataset.num_samples == len(
            dataset.targets
        ) == len(
            dataset.timestamps
        )

    def test_feature_dimension_matches_metadata(self):
        """The tensor feature dimension matches feature_names."""

        dataset = make_forecasting_dataset()

        assert dataset.sequences.shape[2] == (
            len(dataset.feature_names)
        )

    def test_context_dimension_matches_metadata(self):
        """The tensor context dimension matches context_window."""

        dataset = make_forecasting_dataset()

        assert dataset.sequences.shape[1] == (
            dataset.context_window
        )

    def test_no_nan_values_anywhere(self):
        """The final dataset contains no NaN values."""

        dataset = make_forecasting_dataset()

        assert not np.isnan(
            dataset.sequences
        ).any()

        assert not np.isnan(
            dataset.targets
        ).any()

        assert not np.isnan(
            dataset.timestamps
        ).any()

    def test_no_infinite_values_anywhere(self):
        """The final dataset contains no infinite values."""

        dataset = make_forecasting_dataset()

        assert not np.isinf(
            dataset.sequences
        ).any()

        assert not np.isinf(
            dataset.targets
        ).any()

        assert not np.isinf(
            dataset.timestamps
        ).any()


# =============================================================================
# Forecasting Contract Regression
# =============================================================================


class TestForecastingContractRegression:
    """
    Lock the canonical Phase 8/9 NVIDIA-facing data contract.

    Current contract:

        context_window = 20
        num_features = 14
        forecast_horizon = 5
        target_name = future_mid_price_return
        price_column = mid_price
    """

    def test_context_window_is_20(self):
        """The canonical context window remains 20."""

        assert make_forecasting_dataset().context_window == 20

    def test_feature_count_is_14(self):
        """The canonical feature count remains 14."""

        assert make_forecasting_dataset().num_features == 14

    def test_forecast_horizon_is_5(self):
        """The canonical forecast horizon remains 5."""

        assert make_forecasting_dataset().forecast_horizon == 5

    def test_target_name_is_future_mid_price_return(self):
        """The canonical target name remains unchanged."""

        assert (
            make_forecasting_dataset().target_name
            == "future_mid_price_return"
        )

    def test_price_column_is_mid_price(self):
        """The canonical target source column remains mid_price."""

        assert (
            make_forecasting_dataset().price_column
            == "mid_price"
        )

    def test_single_model_input_is_20_by_14(self):
        """A single NVIDIA model input is exactly (20, 14)."""

        assert (
            make_forecasting_dataset().model_input_shape
            == (20, 14)
        )

    def test_model_batch_is_n_by_20_by_14(self):
        """A complete NVIDIA batch is (N, 20, 14)."""

        dataset = make_forecasting_dataset(
            num_rows=100,
        )

        assert dataset.shape[1:] == (
            20,
            14,
        )


# =============================================================================
# Complete Pipeline Regression
# =============================================================================


def test_complete_forecasting_dataset_pipeline_contract():
    """
    Validate the complete Phase 8/9 forecasting dataset contract.

    The resulting object must provide:

        - 3D model inputs
        - 1D targets
        - 1D timestamps
        - 20-step context
        - 14 ordered features
        - 5-step forecast horizon
        - matching sample counts
        - chronological timestamps
        - finite values
    """

    dataset = make_forecasting_dataset(
        num_rows=100,
        context_window=20,
        forecast_horizon=5,
    )

    assert isinstance(
        dataset,
        ForecastingDataset,
    )

    assert dataset.sequences.ndim == 3
    assert dataset.targets.ndim == 1
    assert dataset.timestamps.ndim == 1

    assert dataset.shape[1:] == (
        20,
        14,
    )

    assert dataset.input_shape == (
        20,
        14,
    )

    assert dataset.targets.shape == (
        dataset.num_samples,
    )

    assert dataset.timestamps.shape == (
        dataset.num_samples,
    )

    assert dataset.feature_names == FEATURE_NAMES
    assert dataset.context_window == 20
    assert dataset.forecast_horizon == 5
    assert dataset.target_name == TARGET_NAME
    assert dataset.price_column == PRICE_COLUMN

    assert np.all(
        np.diff(dataset.timestamps) > 0
    )

    assert np.isfinite(
        dataset.sequences
    ).all()

    assert np.isfinite(
        dataset.targets
    ).all()

    assert np.isfinite(
        dataset.timestamps
    ).all()


# =============================================================================
# Convenience API Regression
# =============================================================================


class TestConvenienceAPIs:
    """Validate public convenience APIs against the builder."""

    def test_build_function_returns_forecasting_dataset(self):
        """build_forecasting_dataset returns the expected contract."""

        dataset = build_forecasting_dataset(
            make_sequence_dataset(),
            make_target_dataset(),
        )

        assert isinstance(
            dataset,
            ForecastingDataset,
        )

    def test_align_sequence_targets_matches_build_function(self):
        """align_sequence_targets matches the main builder API."""

        sequences = make_sequence_dataset()
        targets = make_target_dataset()

        first = build_forecasting_dataset(
            sequences,
            targets,
        )

        second = align_sequence_targets(
            sequences,
            targets,
        )

        np.testing.assert_array_equal(
            first.sequences,
            second.sequences,
        )

        np.testing.assert_array_equal(
            first.targets,
            second.targets,
        )

        np.testing.assert_array_equal(
            first.timestamps,
            second.timestamps,
        )

    def test_builder_matches_convenience_function(self):
        """Builder and convenience function produce identical results."""

        sequences = make_sequence_dataset()
        targets = make_target_dataset()

        builder = ForecastingDatasetBuilder()

        first = builder.build(
            sequences,
            targets,
        )

        second = build_forecasting_dataset(
            sequences,
            targets,
        )

        np.testing.assert_array_equal(
            first.sequences,
            second.sequences,
        )

        np.testing.assert_array_equal(
            first.targets,
            second.targets,
        )

        np.testing.assert_array_equal(
            first.timestamps,
            second.timestamps,
        )


# =============================================================================
# Phase 10 — Final Forecasting Pipeline / NVIDIA I/O Contract
# =============================================================================


class TestTimestampWindowIntegration:
    """Validate preservation of the complete context timestamp windows."""

    def test_timestamp_windows_are_present_on_built_dataset(self):
        """The builder must preserve one complete timestamp window per sample."""

        dataset = make_forecasting_dataset()

        assert dataset.has_timestamp_windows is True
        assert dataset.timestamp_windows is not None
        assert len(dataset.timestamp_windows) == dataset.num_samples

    def test_timestamp_window_has_context_length(self):
        """Every stored timestamp window must match context_window."""

        dataset = make_forecasting_dataset()

        assert all(
            len(window) == dataset.context_window
            for window in dataset.timestamp_windows
        )

    def test_first_timestamp_window_is_exact(self):
        """The first stored window must be the original sequence window."""

        dataset = make_forecasting_dataset()

        expected = tuple(
            float(value)
            for value in range(1000, 1020)
        )

        assert dataset.timestamp_windows[0] == expected
        assert dataset.get_timestamp_window(0) == expected

    def test_last_timestamp_of_each_window_matches_aligned_timestamp(self):
        """Window endpoints must equal the aligned sample timestamps."""

        dataset = make_forecasting_dataset()

        for index, window in enumerate(dataset.timestamp_windows):
            assert window[-1] == pytest.approx(
                dataset.timestamps[index]
            )

    def test_timestamp_windows_are_strictly_chronological(self):
        """Every context timestamp window must be strictly increasing."""

        dataset = make_forecasting_dataset()

        for window in dataset.timestamp_windows:
            assert np.all(np.diff(window) > 0)

    def test_timestamp_windows_are_unique_within_each_sample(self):
        """A context window cannot contain duplicate timestamps."""

        dataset = make_forecasting_dataset()

        for window in dataset.timestamp_windows:
            assert len(set(window)) == len(window)

    def test_timestamp_windows_shift_one_step_between_adjacent_samples(self):
        """Rolling windows must advance by exactly one timestamp."""

        dataset = make_forecasting_dataset()

        first = np.asarray(dataset.timestamp_windows[0])
        second = np.asarray(dataset.timestamp_windows[1])

        np.testing.assert_array_equal(
            second[:-1],
            first[1:],
        )

        assert second[-1] == pytest.approx(first[-1] + 1.0)

    def test_non_unit_spacing_is_preserved_in_timestamp_windows(self):
        """The original timestamp spacing must survive alignment."""

        dataset = build_forecasting_dataset(
            make_sequence_dataset(
                num_rows=30,
                context_window=10,
                timestamp_step=0.5,
            ),
            make_target_dataset(
                num_rows=30,
                timestamp_step=0.5,
            ),
        )

        window = np.asarray(dataset.timestamp_windows[0])

        np.testing.assert_allclose(
            np.diff(window),
            0.5,
        )

        assert window[0] == pytest.approx(1000.0)
        assert window[-1] == pytest.approx(1004.5)

    def test_timestamp_window_count_matches_sequence_count_after_alignment(self):
        """Dropped unmatched sequences must also be removed from timestamp windows."""

        dataset = build_forecasting_dataset(
            make_sequence_dataset(),
            make_target_dataset(num_rows=25),
        )

        assert len(dataset.timestamp_windows) == dataset.num_samples
        assert dataset.num_samples < 21

        for index, window in enumerate(dataset.timestamp_windows):
            assert window[-1] == pytest.approx(dataset.timestamps[index])

    def test_timestamp_windows_are_immutable_tuples(self):
        """Stored timestamp windows use tuple-based immutable containers."""

        dataset = make_forecasting_dataset()

        assert isinstance(dataset.timestamp_windows, tuple)
        assert all(
            isinstance(window, tuple)
            for window in dataset.timestamp_windows
        )

    def test_get_timestamp_window_returns_tuple(self):
        """The public timestamp-window accessor returns an immutable tuple."""

        dataset = make_forecasting_dataset()

        result = dataset.get_timestamp_window(0)

        assert isinstance(result, tuple)
        assert len(result) == 20

    def test_get_timestamp_window_rejects_negative_index(self):
        """Negative timestamp-window indices are rejected."""

        dataset = make_forecasting_dataset()

        with pytest.raises(IndexError):
            dataset.get_timestamp_window(-1)

    def test_get_timestamp_window_rejects_upper_bound(self):
        """The upper sample boundary is rejected."""

        dataset = make_forecasting_dataset()

        with pytest.raises(IndexError):
            dataset.get_timestamp_window(dataset.num_samples)

    @pytest.mark.parametrize(
        "index",
        [0.0, "0", True, None],
    )
    def test_get_timestamp_window_rejects_non_integer_index(self, index):
        """The accessor requires a real integer index."""

        dataset = make_forecasting_dataset()

        with pytest.raises(TypeError):
            dataset.get_timestamp_window(index)


class TestTimestampWindowValidation:
    """Validate constructor-level timestamp-window integrity checks."""

    def _valid_dataset_kwargs(self):
        """Return valid constructor arguments for targeted validation tests."""

        dataset = make_forecasting_dataset()

        return {
            "sequences": dataset.sequences.copy(),
            "targets": dataset.targets.copy(),
            "timestamps": dataset.timestamps.copy(),
            "feature_names": tuple(dataset.feature_names),
            "context_window": dataset.context_window,
            "forecast_horizon": dataset.forecast_horizon,
            "target_name": dataset.target_name,
            "price_column": dataset.price_column,
            "timestamp_windows": tuple(
                tuple(window)
                for window in dataset.timestamp_windows
            ),
        }

    def test_wrong_number_of_timestamp_windows_is_rejected(self):
        """There must be exactly one timestamp window per sample."""

        kwargs = self._valid_dataset_kwargs()
        kwargs["timestamp_windows"] = kwargs["timestamp_windows"][:-1]

        with pytest.raises(ValueError, match="timestamp_windows"):
            ForecastingDataset(**kwargs)

    def test_wrong_window_length_is_rejected(self):
        """Each timestamp window must contain exactly context_window values."""

        kwargs = self._valid_dataset_kwargs()
        windows = list(kwargs["timestamp_windows"])
        windows[0] = windows[0][1:]
        kwargs["timestamp_windows"] = tuple(windows)

        with pytest.raises(ValueError, match="context_window"):
            ForecastingDataset(**kwargs)

    def test_nan_timestamp_window_value_is_rejected(self):
        """Timestamp windows cannot contain NaN values."""

        kwargs = self._valid_dataset_kwargs()
        windows = [list(window) for window in kwargs["timestamp_windows"]]
        windows[0][5] = np.nan
        kwargs["timestamp_windows"] = tuple(tuple(window) for window in windows)

        with pytest.raises(ValueError, match="finite"):
            ForecastingDataset(**kwargs)

    def test_infinite_timestamp_window_value_is_rejected(self):
        """Timestamp windows cannot contain infinite values."""

        kwargs = self._valid_dataset_kwargs()
        windows = [list(window) for window in kwargs["timestamp_windows"]]
        windows[0][5] = np.inf
        kwargs["timestamp_windows"] = tuple(tuple(window) for window in windows)

        with pytest.raises(ValueError, match="finite"):
            ForecastingDataset(**kwargs)

    def test_duplicate_timestamp_inside_window_is_rejected(self):
        """Timestamp windows must have unique timestamps."""

        kwargs = self._valid_dataset_kwargs()
        windows = [list(window) for window in kwargs["timestamp_windows"]]
        windows[0][5] = windows[0][4]
        kwargs["timestamp_windows"] = tuple(tuple(window) for window in windows)

        with pytest.raises(ValueError):
            ForecastingDataset(**kwargs)

    def test_non_chronological_timestamp_window_is_rejected(self):
        """Timestamp windows must be strictly chronological."""

        kwargs = self._valid_dataset_kwargs()
        windows = [list(window) for window in kwargs["timestamp_windows"]]
        windows[0][5], windows[0][6] = windows[0][6], windows[0][5]
        kwargs["timestamp_windows"] = tuple(tuple(window) for window in windows)

        with pytest.raises(ValueError):
            ForecastingDataset(**kwargs)

    def test_window_endpoint_must_match_aligned_timestamp(self):
        """The final timestamp in each window must equal sample timestamp."""

        kwargs = self._valid_dataset_kwargs()
        windows = [list(window) for window in kwargs["timestamp_windows"]]
        windows[0][-1] += 0.25
        kwargs["timestamp_windows"] = tuple(tuple(window) for window in windows)

        with pytest.raises(ValueError, match="aligned sample timestamp"):
            ForecastingDataset(**kwargs)

    def test_timestamp_windows_are_optional_for_backward_compatibility(self):
        """Manually constructed legacy datasets may omit timestamp windows."""

        kwargs = self._valid_dataset_kwargs()
        kwargs["timestamp_windows"] = None

        dataset = ForecastingDataset(**kwargs)

        assert dataset.has_timestamp_windows is False

        with pytest.raises(ValueError, match="Complete timestamp windows"):
            dataset.get_timestamp_window(0)


class TestNvidiaIntegrationContract:
    """Validate the completed dataset against the canonical NVIDIA contract."""

    def test_built_nvidia_dataset_reports_compatible(self):
        """The canonical generated dataset satisfies the NVIDIA contract."""

        dataset = make_forecasting_dataset()

        assert dataset.is_nvidia_compatible is True
        dataset.validate_nvidia_compatibility()
        ForecastingDatasetBuilder().validate_nvidia_contract(dataset)

    def test_nvidia_contract_does_not_modify_dataset(self):
        """Compatibility validation must be observational only."""

        dataset = make_forecasting_dataset()
        before_sequences = dataset.sequences.copy()
        before_targets = dataset.targets.copy()
        before_timestamps = dataset.timestamps.copy()
        before_windows = dataset.timestamp_windows

        dataset.validate_nvidia_compatibility()

        np.testing.assert_array_equal(dataset.sequences, before_sequences)
        np.testing.assert_array_equal(dataset.targets, before_targets)
        np.testing.assert_array_equal(dataset.timestamps, before_timestamps)
        assert dataset.timestamp_windows == before_windows

    def test_wrong_feature_order_is_not_nvidia_compatible(self):
        """Changing feature order must invalidate the NVIDIA contract."""

        dataset = make_forecasting_dataset()
        kwargs = {
            "sequences": dataset.sequences.copy(),
            "targets": dataset.targets.copy(),
            "timestamps": dataset.timestamps.copy(),
            "feature_names": tuple(reversed(dataset.feature_names)),
            "context_window": dataset.context_window,
            "forecast_horizon": dataset.forecast_horizon,
            "target_name": dataset.target_name,
            "price_column": dataset.price_column,
            "timestamp_windows": dataset.timestamp_windows,
        }

        modified = ForecastingDataset(**kwargs)

        assert modified.is_nvidia_compatible is False

        with pytest.raises(ValueError, match="feature order"):
            modified.validate_nvidia_compatibility()

    def test_wrong_horizon_is_not_nvidia_compatible(self):
        """Changing the forecast horizon must invalidate the NVIDIA contract."""

        dataset = make_forecasting_dataset(forecast_horizon=4)

        assert dataset.is_nvidia_compatible is False

        with pytest.raises(ValueError, match="forecast horizon"):
            dataset.validate_nvidia_compatibility()

    def test_wrong_target_name_is_not_nvidia_compatible(self):
        """Changing the target name must invalidate the NVIDIA contract."""

        dataset = build_forecasting_dataset(
            make_sequence_dataset(),
            make_target_dataset(target_name="wrong_target"),
        )

        assert dataset.is_nvidia_compatible is False

        with pytest.raises(ValueError, match="target"):
            dataset.validate_nvidia_compatibility()

    def test_wrong_context_window_is_not_nvidia_compatible(self):
        """A noncanonical context window must not pass NVIDIA validation."""

        dataset = make_forecasting_dataset(
            num_rows=25,
            context_window=5,
        )

        assert dataset.is_nvidia_compatible is False

        with pytest.raises(ValueError, match="context_window"):
            dataset.validate_nvidia_compatibility()

    def test_wrong_price_column_is_not_nvidia_compatible(self):
        """Changing the source price-column metadata invalidates the contract."""

        dataset = build_forecasting_dataset(
            make_sequence_dataset(),
            make_target_dataset(price_column="other_price"),
        )

        assert dataset.is_nvidia_compatible is False

        with pytest.raises(ValueError, match="price column"):
            dataset.validate_nvidia_compatibility()


class TestTimestampWindowCopyAndDeterminism:
    """Validate timestamp-window preservation under copying and repeated builds."""

    def test_copy_preserves_timestamp_windows(self):
        """copy() preserves the complete timestamp-window contract."""

        dataset = make_forecasting_dataset()
        copied = dataset.copy()

        assert copied.timestamp_windows == dataset.timestamp_windows
        assert copied.has_timestamp_windows is True

    def test_copy_creates_independent_timestamp_window_container(self):
        """The copied timestamp-window container is independently allocated."""

        dataset = make_forecasting_dataset()
        copied = dataset.copy()

        assert copied.timestamp_windows is not dataset.timestamp_windows
        assert copied.timestamp_windows == dataset.timestamp_windows

    def test_repeated_builds_preserve_identical_timestamp_windows(self):
        """Repeated builds must produce identical timestamp-window metadata."""

        sequences = make_sequence_dataset()
        targets = make_target_dataset()

        first = build_forecasting_dataset(sequences, targets)
        second = build_forecasting_dataset(sequences, targets)

        assert first.timestamp_windows == second.timestamp_windows

    def test_source_sequence_timestamp_windows_are_not_modified(self):
        """Building a dataset must not modify source sequence timestamps."""

        sequences = make_sequence_dataset()
        before = tuple(tuple(window) for window in sequences.timestamps)

        build_forecasting_dataset(sequences, make_target_dataset())

        after = tuple(tuple(window) for window in sequences.timestamps)
        assert after == before

    def test_timestamp_windows_preserve_source_values_exactly(self):
        """Aligned windows must match the source SequenceDataset windows exactly."""

        sequences = make_sequence_dataset()
        dataset = build_forecasting_dataset(
            sequences,
            make_target_dataset(),
        )

        expected = tuple(
            tuple(window)
            for window in sequences.timestamps
        )

        assert dataset.timestamp_windows == expected


class TestFinalPhaseTenPipelineContract:
    """Lock the complete Person 1 Phase 10 data-side contract."""

    def test_final_nvidia_ready_dataset_contract(self):
        """The final dataset exposes every required NVIDIA-facing invariant."""

        dataset = make_forecasting_dataset(num_rows=100)

        assert dataset.is_nvidia_compatible is True
        assert dataset.has_timestamp_windows is True
        assert dataset.shape == (81, 20, 14)
        assert dataset.input_shape == (20, 14)
        assert dataset.targets.shape == (81,)
        assert dataset.timestamps.shape == (81,)
        assert len(dataset.timestamp_windows) == 81
        assert all(len(window) == 20 for window in dataset.timestamp_windows)
        assert dataset.feature_names == FEATURE_NAMES
        assert dataset.context_window == 20
        assert dataset.forecast_horizon == 5
        assert dataset.target_name == TARGET_NAME
        assert dataset.price_column == PRICE_COLUMN

    def test_target_is_not_embedded_in_feature_tensor(self):
        """The future target remains separate from model feature inputs."""

        dataset = make_forecasting_dataset()

        assert dataset.sequences.shape[-1] == len(FEATURE_NAMES)
        assert TARGET_NAME not in dataset.feature_names

    def test_aligned_timestamp_is_window_endpoint_for_every_sample(self):
        """Every model sample's timestamp corresponds to its context endpoint."""

        dataset = make_forecasting_dataset(num_rows=100)

        for index in range(dataset.num_samples):
            assert dataset.get_timestamp_window(index)[-1] == pytest.approx(
                dataset.timestamps[index]
            )

    def test_final_pipeline_is_deterministic(self):
        """The same inputs must produce the same final NVIDIA-ready dataset."""

        first = make_forecasting_dataset(num_rows=100)
        second = make_forecasting_dataset(num_rows=100)

        np.testing.assert_array_equal(first.sequences, second.sequences)
        np.testing.assert_array_equal(first.targets, second.targets)
        np.testing.assert_array_equal(first.timestamps, second.timestamps)
        assert first.timestamp_windows == second.timestamp_windows
        assert first.feature_names == second.feature_names

