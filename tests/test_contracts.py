# tests/test_contracts.py
#
# Purpose:
#     Validate the shared data contracts used throughout the Smart Order Routing
#     forecasting pipeline.
#
# Responsibilities:
#     - Test ForecastInput validation and NVIDIA-facing input requirements.
#     - Test ForecastResult validation and derived prediction fields.
#     - Test SequenceDataset validation and rolling-sequence metadata.
#     - Test TargetDataset validation and target metadata.
#     - Test ForecastingDataset validation and model-facing dataset structure.
#     - Verify chronological ordering and timestamp alignment requirements.
#     - Verify expected feature dimensions and metadata preservation.
#     - Verify invalid contract data is rejected early.
#
# The module does not:
#     - Train forecasting models.
#     - Run NVIDIA model inference.
#     - Perform normalization.
#     - Build rolling sequences.
#     - Build targets.
#     - Split datasets.
#     - Execute trading strategies.
#     - Simulate market behavior.
#
# These tests validate the contracts consumed by those components rather than
# testing the implementation of those components themselves.

from __future__ import annotations

import numpy as np
import pytest

from data.contracts import (
    ForecastInput,
    ForecastResult,
    ModelStatus,
    SequenceDataset,
    TargetDataset,
)

from data.forecasting_dataset import ForecastingDataset


# ---------------------------------------------------------------------------
# Shared test configuration
# ---------------------------------------------------------------------------

FEATURE_NAMES = [
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
]

CONTEXT_WINDOW = 20
FORECAST_HORIZON = 5
NUM_FEATURES = 14
NUM_SAMPLES = 8


# ---------------------------------------------------------------------------
# Test data factories
# ---------------------------------------------------------------------------

def make_feature_sequence(
    context_window: int = CONTEXT_WINDOW,
    num_features: int = NUM_FEATURES,
) -> list[list[float]]:
    """Create one valid NVIDIA forecasting feature sequence."""
    return [
        [
            float(row * num_features + column + 1)
            for column in range(num_features)
        ]
        for row in range(context_window)
    ]


def make_timestamps(
    count: int = CONTEXT_WINDOW,
    start: float = 1_000.0,
) -> list[float]:
    """Create strictly increasing timestamps."""
    return [start + float(index) for index in range(count)]


def make_sequence_array(
    num_samples: int = NUM_SAMPLES,
    context_window: int = CONTEXT_WINDOW,
    num_features: int = NUM_FEATURES,
) -> np.ndarray:
    """Create a valid sequence dataset array."""
    values = np.arange(
        num_samples * context_window * num_features,
        dtype=np.float64,
    )

    return values.reshape(
        num_samples,
        context_window,
        num_features,
    )


def make_sequence_timestamps(
    num_samples: int = NUM_SAMPLES,
    context_window: int = CONTEXT_WINDOW,
) -> list[list[float]]:
    """Create valid timestamps for a sequence dataset."""
    return [
        [
            float(sample * 100 + offset)
            for offset in range(context_window)
        ]
        for sample in range(num_samples)
    ]


def make_target_values(
    num_samples: int = NUM_SAMPLES,
) -> np.ndarray:
    """Create valid future-return targets."""
    return np.linspace(
        0.0001,
        0.0008,
        num_samples,
        dtype=np.float64,
    )


def make_target_timestamps(
    num_samples: int = NUM_SAMPLES,
) -> list[float]:
    """Create valid target timestamps."""
    return [
        float(1_019 + index)
        for index in range(num_samples)
    ]


def make_forecasting_dataset(
    num_samples: int = NUM_SAMPLES,
    context_window: int = CONTEXT_WINDOW,
    num_features: int = NUM_FEATURES,
) -> ForecastingDataset:
    """Create a valid model-facing forecasting dataset."""
    sequences = make_sequence_array(
        num_samples=num_samples,
        context_window=context_window,
        num_features=num_features,
    )

    targets = make_target_values(num_samples)

    timestamps = np.arange(
        2_000.0,
        2_000.0 + num_samples,
        dtype=np.float64,
    )

    return ForecastingDataset(
        sequences=sequences,
        targets=targets,
        timestamps=timestamps,
        feature_names=tuple(FEATURE_NAMES),
        context_window=context_window,
        forecast_horizon=FORECAST_HORIZON,
        target_name="future_mid_price_return",
        price_column="mid_price",
    )


# ===========================================================================
# ForecastInput tests
# ===========================================================================

class TestForecastInput:
    """Tests for the NVIDIA forecasting input contract."""

    def test_valid_forecast_input_is_accepted(self):
        """A correctly shaped forecasting input should be accepted."""
        feature_sequence = make_feature_sequence()
        timestamps = make_timestamps()

        contract = ForecastInput(
            feature_sequence=feature_sequence,
            feature_names=FEATURE_NAMES,
            timestamps=timestamps,
            context_window=CONTEXT_WINDOW,
            forecast_horizon=FORECAST_HORIZON,
        )

        assert contract.context_window == CONTEXT_WINDOW
        assert contract.forecast_horizon == FORECAST_HORIZON
        assert len(contract.feature_sequence) == CONTEXT_WINDOW
        assert len(contract.feature_names) == NUM_FEATURES
        assert len(contract.timestamps) == CONTEXT_WINDOW

    def test_feature_sequence_has_expected_nvidia_shape(self):
        """The NVIDIA input should contain 20 rows and 14 features."""
        contract = ForecastInput(
            feature_sequence=make_feature_sequence(),
            feature_names=FEATURE_NAMES,
            timestamps=make_timestamps(),
            context_window=20,
            forecast_horizon=5,
        )

        assert len(contract.feature_sequence) == 20
        assert all(
            len(row) == 14
            for row in contract.feature_sequence
        )

    def test_context_window_must_be_positive(self):
        """A non-positive context window should be rejected."""
        with pytest.raises((ValueError, TypeError)):
            ForecastInput(
                feature_sequence=make_feature_sequence(),
                feature_names=FEATURE_NAMES,
                timestamps=make_timestamps(),
                context_window=0,
                forecast_horizon=FORECAST_HORIZON,
            )

    def test_forecast_horizon_must_be_positive(self):
        """A non-positive forecast horizon should be rejected."""
        with pytest.raises((ValueError, TypeError)):
            ForecastInput(
                feature_sequence=make_feature_sequence(),
                feature_names=FEATURE_NAMES,
                timestamps=make_timestamps(),
                context_window=CONTEXT_WINDOW,
                forecast_horizon=0,
            )

    def test_empty_feature_sequence_is_rejected(self):
        """An empty feature sequence should be rejected."""
        with pytest.raises((ValueError, TypeError)):
            ForecastInput(
                feature_sequence=[],
                feature_names=FEATURE_NAMES,
                timestamps=[],
                context_window=CONTEXT_WINDOW,
                forecast_horizon=FORECAST_HORIZON,
            )

    def test_feature_name_count_must_match_columns(self):
        """Feature metadata must match the sequence width."""
        feature_names = FEATURE_NAMES[:-1]

        with pytest.raises((ValueError, TypeError)):
            ForecastInput(
                feature_sequence=make_feature_sequence(),
                feature_names=feature_names,
                timestamps=make_timestamps(),
                context_window=CONTEXT_WINDOW,
                forecast_horizon=FORECAST_HORIZON,
            )

    def test_duplicate_feature_names_are_rejected(self):
        """Feature names must uniquely identify model inputs."""
        duplicate_names = FEATURE_NAMES.copy()
        duplicate_names[-1] = duplicate_names[0]

        with pytest.raises((ValueError, TypeError)):
            ForecastInput(
                feature_sequence=make_feature_sequence(),
                feature_names=duplicate_names,
                timestamps=make_timestamps(),
                context_window=CONTEXT_WINDOW,
                forecast_horizon=FORECAST_HORIZON,
            )

    def test_timestamp_count_must_match_context_window(self):
        """There must be one timestamp per context row."""
        with pytest.raises((ValueError, TypeError)):
            ForecastInput(
                feature_sequence=make_feature_sequence(),
                feature_names=FEATURE_NAMES,
                timestamps=make_timestamps()[:-1],
                context_window=CONTEXT_WINDOW,
                forecast_horizon=FORECAST_HORIZON,
            )

    def test_timestamps_must_be_chronological(self):
        """Forecast input timestamps must be strictly increasing."""
        timestamps = make_timestamps()
        timestamps[10], timestamps[11] = timestamps[11], timestamps[10]

        with pytest.raises((ValueError, TypeError)):
            ForecastInput(
                feature_sequence=make_feature_sequence(),
                feature_names=FEATURE_NAMES,
                timestamps=timestamps,
                context_window=CONTEXT_WINDOW,
                forecast_horizon=FORECAST_HORIZON,
            )

    def test_duplicate_timestamps_are_rejected(self):
        """Duplicate timestamps would break temporal ordering."""
        timestamps = make_timestamps()
        timestamps[10] = timestamps[9]

        with pytest.raises((ValueError, TypeError)):
            ForecastInput(
                feature_sequence=make_feature_sequence(),
                feature_names=FEATURE_NAMES,
                timestamps=timestamps,
                context_window=CONTEXT_WINDOW,
                forecast_horizon=FORECAST_HORIZON,
            )

    def test_non_numeric_feature_value_is_rejected(self):
        """Feature values must be numeric."""
        sequence = make_feature_sequence()
        sequence[5][3] = "invalid"

        with pytest.raises((ValueError, TypeError)):
            ForecastInput(
                feature_sequence=sequence,
                feature_names=FEATURE_NAMES,
                timestamps=make_timestamps(),
                context_window=CONTEXT_WINDOW,
                forecast_horizon=FORECAST_HORIZON,
            )

    def test_non_numeric_timestamp_is_rejected(self):
        """Timestamps must be numeric."""
        timestamps = make_timestamps()
        timestamps[5] = "invalid"

        with pytest.raises((ValueError, TypeError)):
            ForecastInput(
                feature_sequence=make_feature_sequence(),
                feature_names=FEATURE_NAMES,
                timestamps=timestamps,
                context_window=CONTEXT_WINDOW,
                forecast_horizon=FORECAST_HORIZON,
            )


# ===========================================================================
# ForecastResult tests
# ===========================================================================

class TestForecastResult:
    """Tests for the forecasting output contract."""

    def test_valid_forecast_result_is_accepted(self):
        """A valid forecast result should be accepted."""
        result = ForecastResult(
            timestamp=2_000.0,
            forecast_horizon=FORECAST_HORIZON,
            predicted_return=0.0015,
            model_name="NVIDIA Forecasting Model",
            model_version="1.0.0",
            model_status=ModelStatus.SUCCESS,
        )

        assert result.timestamp == 2_000.0
        assert result.forecast_horizon == FORECAST_HORIZON
        assert result.predicted_return == 0.0015
        assert result.model_name == "NVIDIA Forecasting Model"

    def test_positive_return_produces_up_direction(self):
        """A positive predicted return should map to an upward direction."""
        result = ForecastResult(
            timestamp=2_000.0,
            forecast_horizon=FORECAST_HORIZON,
            predicted_return=0.001,
            model_name="test",
            model_version="1.0",
            model_status=ModelStatus.SUCCESS,
        )

        assert result.predicted_direction == "UP"

    def test_negative_return_produces_down_direction(self):
        """A negative predicted return should map to a downward direction."""
        result = ForecastResult(
            timestamp=2_000.0,
            forecast_horizon=FORECAST_HORIZON,
            predicted_return=-0.001,
            model_name="test",
            model_version="1.0",
            model_status=ModelStatus.SUCCESS,
        )

        assert result.predicted_direction == "DOWN"

    def test_zero_return_produces_neutral_direction(self):
        """A zero predicted return should map to a neutral direction."""
        result = ForecastResult(
            timestamp=2_000.0,
            forecast_horizon=FORECAST_HORIZON,
            predicted_return=0.0,
            model_name="test",
            model_version="1.0",
            model_status=ModelStatus.SUCCESS,
        )

        assert result.predicted_direction == "FLAT"

    def test_expected_move_bps_is_derived(self):
        """Expected move should be converted from return to basis points."""
        result = ForecastResult(
            timestamp=2_000.0,
            forecast_horizon=FORECAST_HORIZON,
            predicted_return=0.001,
            model_name="test",
            model_version="1.0",
            model_status=ModelStatus.SUCCESS,
        )

        assert result.expected_move_bps == pytest.approx(10.0)

    def test_confidence_must_be_between_zero_and_one(self):
        """Confidence values outside [0, 1] should be rejected."""
        with pytest.raises((ValueError, TypeError)):
            ForecastResult(
                timestamp=2_000.0,
                forecast_horizon=FORECAST_HORIZON,
                predicted_return=0.001,
                model_name="test",
                model_version="1.0",
                model_status=ModelStatus.SUCCESS,
                confidence=1.1,
            )

    @pytest.mark.parametrize("confidence", [0.0, 0.5, 1.0])
    def test_valid_confidence_values_are_accepted(self, confidence):
        """Boundary confidence values should remain valid."""
        result = ForecastResult(
            timestamp=2_000.0,
            forecast_horizon=FORECAST_HORIZON,
            predicted_return=0.001,
            model_name="test",
            model_version="1.0",
            model_status=ModelStatus.SUCCESS,
            confidence=confidence,
        )

        assert result.confidence == confidence

    def test_forecast_horizon_must_be_positive(self):
        """A non-positive horizon should be rejected."""
        with pytest.raises((ValueError, TypeError)):
            ForecastResult(
                timestamp=2_000.0,
                forecast_horizon=0,
                predicted_return=0.001,
                model_name="test",
                model_version="1.0",
                model_status=ModelStatus.SUCCESS,
            )

    def test_model_name_cannot_be_empty(self):
        """Model name should identify the forecasting model."""
        with pytest.raises((ValueError, TypeError)):
            ForecastResult(
                timestamp=2_000.0,
                forecast_horizon=FORECAST_HORIZON,
                predicted_return=0.001,
                model_name="",
                model_version="1.0",
                model_status=ModelStatus.SUCCESS,
            )


# ===========================================================================
# SequenceDataset tests
# ===========================================================================

class TestSequenceDataset:
    """Tests for rolling sequence dataset contracts."""

    def test_valid_sequence_dataset_is_accepted(self):
        """A correctly shaped sequence dataset should be accepted."""
        dataset = SequenceDataset(
            sequences=make_sequence_array(),
            timestamps=make_sequence_timestamps(),
            feature_names=FEATURE_NAMES,
            context_window=CONTEXT_WINDOW,
        )

        assert dataset.num_sequences == NUM_SAMPLES
        assert dataset.num_features == NUM_FEATURES
        assert dataset.shape == (
            NUM_SAMPLES,
            CONTEXT_WINDOW,
            NUM_FEATURES,
        )

    def test_sequence_dataset_is_three_dimensional(self):
        """Sequence data must have sample, context, and feature dimensions."""
        with pytest.raises((ValueError, TypeError)):
            SequenceDataset(
                sequences=np.zeros((10, NUM_FEATURES)),
                timestamps=make_sequence_timestamps(10),
                feature_names=FEATURE_NAMES,
                context_window=CONTEXT_WINDOW,
            )

    def test_sequence_context_dimension_must_match(self):
        """Sequence context length must match the declared context window."""
        with pytest.raises((ValueError, TypeError)):
            SequenceDataset(
                sequences=make_sequence_array(
                    context_window=10,
                ),
                timestamps=make_sequence_timestamps(
                    context_window=10,
                ),
                feature_names=FEATURE_NAMES,
                context_window=CONTEXT_WINDOW,
            )

    def test_sequence_feature_dimension_must_match_metadata(self):
        """Feature width must match the number of feature names."""
        with pytest.raises((ValueError, TypeError)):
            SequenceDataset(
                sequences=make_sequence_array(
                    num_features=13,
                ),
                timestamps=make_sequence_timestamps(),
                feature_names=FEATURE_NAMES,
                context_window=CONTEXT_WINDOW,
            )

    def test_sequence_timestamp_count_must_match_samples(self):
        """There must be one timestamp window per sequence."""
        with pytest.raises((ValueError, TypeError)):
            SequenceDataset(
                sequences=make_sequence_array(),
                timestamps=make_sequence_timestamps()[:-1],
                feature_names=FEATURE_NAMES,
                context_window=CONTEXT_WINDOW,
            )

    def test_each_sequence_timestamp_window_must_match_context(self):
        """Every sequence must contain one timestamp per context row."""
        timestamps = make_sequence_timestamps()
        timestamps[0] = timestamps[0][:-1]

        with pytest.raises((ValueError, TypeError)):
            SequenceDataset(
                sequences=make_sequence_array(),
                timestamps=timestamps,
                feature_names=FEATURE_NAMES,
                context_window=CONTEXT_WINDOW,
            )

    def test_sequence_timestamps_must_be_chronological(self):
        """Each rolling sequence must preserve temporal ordering."""
        timestamps = make_sequence_timestamps()
        timestamps[0][5], timestamps[0][6] = (
            timestamps[0][6],
            timestamps[0][5],
        )

        with pytest.raises((ValueError, TypeError)):
            SequenceDataset(
                sequences=make_sequence_array(),
                timestamps=timestamps,
                feature_names=FEATURE_NAMES,
                context_window=CONTEXT_WINDOW,
            )

    def test_duplicate_sequence_timestamps_are_rejected(self):
        """Duplicate timestamps inside a sequence should be rejected."""
        timestamps = make_sequence_timestamps()
        timestamps[0][5] = timestamps[0][4]

        with pytest.raises((ValueError, TypeError)):
            SequenceDataset(
                sequences=make_sequence_array(),
                timestamps=timestamps,
                feature_names=FEATURE_NAMES,
                context_window=CONTEXT_WINDOW,
            )

    def test_feature_names_must_be_unique(self):
        """Sequence feature names must uniquely identify columns."""
        duplicate_names = FEATURE_NAMES.copy()
        duplicate_names[-1] = duplicate_names[0]

        with pytest.raises((ValueError, TypeError)):
            SequenceDataset(
                sequences=make_sequence_array(),
                timestamps=make_sequence_timestamps(),
                feature_names=duplicate_names,
                context_window=CONTEXT_WINDOW,
            )

    def test_sequence_shape_property_matches_data(self):
        """The shape property should expose the actual sequence shape."""
        dataset = SequenceDataset(
            sequences=make_sequence_array(),
            timestamps=make_sequence_timestamps(),
            feature_names=FEATURE_NAMES,
            context_window=CONTEXT_WINDOW,
        )

        assert dataset.shape == dataset.sequences.shape

    def test_sequence_feature_count_property_is_correct(self):
        """The feature count property should reflect the sequence width."""
        dataset = SequenceDataset(
            sequences=make_sequence_array(),
            timestamps=make_sequence_timestamps(),
            feature_names=FEATURE_NAMES,
            context_window=CONTEXT_WINDOW,
        )

        assert dataset.num_features == 14


# ===========================================================================
# TargetDataset tests
# ===========================================================================

class TestTargetDataset:
    """Tests for future-target dataset contracts."""

    def test_valid_target_dataset_is_accepted(self):
        """A valid target dataset should be accepted."""
        dataset = TargetDataset(
            targets=make_target_values(),
            timestamps=make_target_timestamps(),
            target_name="future_mid_price_return",
            forecast_horizon=FORECAST_HORIZON,
            price_column="mid_price",
        )

        assert dataset.num_targets == NUM_SAMPLES
        assert dataset.target_name == "future_mid_price_return"
        assert dataset.forecast_horizon == FORECAST_HORIZON

    def test_target_count_must_match_timestamps(self):
        """Every target must have exactly one timestamp."""
        with pytest.raises((ValueError, TypeError)):
            TargetDataset(
                targets=make_target_values(),
                timestamps=make_target_timestamps()[:-1],
                target_name="future_mid_price_return",
                forecast_horizon=FORECAST_HORIZON,
            )

    def test_target_timestamps_must_be_chronological(self):
        """Target timestamps must be strictly increasing."""
        timestamps = make_target_timestamps()
        timestamps[3], timestamps[4] = timestamps[4], timestamps[3]

        with pytest.raises((ValueError, TypeError)):
            TargetDataset(
                targets=make_target_values(),
                timestamps=timestamps,
                target_name="future_mid_price_return",
                forecast_horizon=FORECAST_HORIZON,
            )

    def test_duplicate_target_timestamps_are_rejected(self):
        """Duplicate target timestamps should be rejected."""
        timestamps = make_target_timestamps()
        timestamps[3] = timestamps[2]

        with pytest.raises((ValueError, TypeError)):
            TargetDataset(
                targets=make_target_values(),
                timestamps=timestamps,
                target_name="future_mid_price_return",
                forecast_horizon=FORECAST_HORIZON,
            )

    def test_forecast_horizon_must_be_positive(self):
        """Target horizon must be positive."""
        with pytest.raises((ValueError, TypeError)):
            TargetDataset(
                targets=make_target_values(),
                timestamps=make_target_timestamps(),
                target_name="future_mid_price_return",
                forecast_horizon=0,
            )

    def test_target_name_cannot_be_empty(self):
        """Target name must identify the prediction target."""
        with pytest.raises((ValueError, TypeError)):
            TargetDataset(
                targets=make_target_values(),
                timestamps=make_target_timestamps(),
                target_name="",
                forecast_horizon=FORECAST_HORIZON,
            )

    def test_price_column_defaults_to_mid_price(self):
        """The default source price column should be mid_price."""
        dataset = TargetDataset(
            targets=make_target_values(),
            timestamps=make_target_timestamps(),
            target_name="future_mid_price_return",
            forecast_horizon=FORECAST_HORIZON,
        )

        assert dataset.price_column == "mid_price"

    def test_nonfinite_target_values_are_rejected(self):
        """NaN and infinite targets should not reach the model."""
        targets = make_target_values()
        targets[2] = np.nan

        with pytest.raises((ValueError, TypeError)):
            TargetDataset(
                targets=targets,
                timestamps=make_target_timestamps(),
                target_name="future_mid_price_return",
                forecast_horizon=FORECAST_HORIZON,
            )

    def test_target_values_property_returns_target_data(self):
        """The target_values property should expose the target array."""
        targets = make_target_values()

        dataset = TargetDataset(
            targets=targets,
            timestamps=make_target_timestamps(),
            target_name="future_mid_price_return",
            forecast_horizon=FORECAST_HORIZON,
        )

        np.testing.assert_array_equal(
            dataset.target_values,
            targets,
        )


# ===========================================================================
# ForecastingDataset tests
# ===========================================================================

class TestForecastingDataset:
    """Tests for the final model-facing forecasting dataset."""

    def test_valid_forecasting_dataset_is_accepted(self):
        """A correctly aligned forecasting dataset should be accepted."""
        dataset = make_forecasting_dataset()

        assert dataset.num_samples == NUM_SAMPLES
        assert dataset.num_features == NUM_FEATURES
        assert dataset.shape == (
            NUM_SAMPLES,
            CONTEXT_WINDOW,
            NUM_FEATURES,
        )

    def test_input_shape_matches_nvidia_contract(self):
        """
        The input shape should describe one model-ready forecasting sequence.

        The complete dataset shape includes the sample dimension:
            (num_samples, context_window, num_features)

        The model-facing input shape describes one sequence:
            (context_window, num_features)
        """
        dataset = make_forecasting_dataset()

        assert dataset.input_shape == (
            CONTEXT_WINDOW,
            NUM_FEATURES,
        )

        assert dataset.shape == (
            NUM_SAMPLES,
            CONTEXT_WINDOW,
            NUM_FEATURES,
        )

    def test_forecasting_dataset_has_one_target_per_sequence(self):
        """Every model input sequence must have one aligned target."""
        dataset = make_forecasting_dataset()

        assert len(dataset.sequences) == len(dataset.targets)

    def test_forecasting_dataset_has_one_timestamp_per_sequence(self):
        """Every sequence must have one ending timestamp."""
        dataset = make_forecasting_dataset()

        assert len(dataset.sequences) == len(dataset.timestamps)

    def test_forecasting_dataset_is_three_dimensional(self):
        """Model input data must remain three-dimensional."""
        with pytest.raises((ValueError, TypeError)):
            ForecastingDataset(
                sequences=np.zeros(
                    (NUM_SAMPLES, NUM_FEATURES),
                    dtype=np.float64,
                ),
                targets=make_target_values(),
                timestamps=np.arange(
                    2_000.0,
                    2_000.0 + NUM_SAMPLES,
                ),
                feature_names=tuple(FEATURE_NAMES),
                context_window=CONTEXT_WINDOW,
                forecast_horizon=FORECAST_HORIZON,
                target_name="future_mid_price_return",
            )

    def test_sequence_count_must_match_target_count(self):
        """There must be exactly one target for every sequence."""
        with pytest.raises((ValueError, TypeError)):
            ForecastingDataset(
                sequences=make_sequence_array(),
                targets=make_target_values()[:-1],
                timestamps=np.arange(
                    2_000.0,
                    2_000.0 + NUM_SAMPLES,
                ),
                feature_names=tuple(FEATURE_NAMES),
                context_window=CONTEXT_WINDOW,
                forecast_horizon=FORECAST_HORIZON,
                target_name="future_mid_price_return",
            )

    def test_sequence_count_must_match_timestamp_count(self):
        """There must be one ending timestamp for every sequence."""
        with pytest.raises((ValueError, TypeError)):
            ForecastingDataset(
                sequences=make_sequence_array(),
                targets=make_target_values(),
                timestamps=np.arange(
                    2_000.0,
                    2_000.0 + NUM_SAMPLES - 1,
                ),
                feature_names=tuple(FEATURE_NAMES),
                context_window=CONTEXT_WINDOW,
                forecast_horizon=FORECAST_HORIZON,
                target_name="future_mid_price_return",
            )

    def test_forecasting_timestamps_must_be_chronological(self):
        """Final dataset timestamps must remain chronological."""
        timestamps = np.arange(
            2_000.0,
            2_000.0 + NUM_SAMPLES,
            dtype=np.float64,
        )
        timestamps[3], timestamps[4] = timestamps[4], timestamps[3]

        with pytest.raises((ValueError, TypeError)):
            ForecastingDataset(
                sequences=make_sequence_array(),
                targets=make_target_values(),
                timestamps=timestamps,
                feature_names=tuple(FEATURE_NAMES),
                context_window=CONTEXT_WINDOW,
                forecast_horizon=FORECAST_HORIZON,
                target_name="future_mid_price_return",
            )

    def test_duplicate_forecasting_timestamps_are_rejected(self):
        """Duplicate final timestamps would violate temporal uniqueness."""
        timestamps = np.arange(
            2_000.0,
            2_000.0 + NUM_SAMPLES,
            dtype=np.float64,
        )
        timestamps[4] = timestamps[3]

        with pytest.raises((ValueError, TypeError)):
            ForecastingDataset(
                sequences=make_sequence_array(),
                targets=make_target_values(),
                timestamps=timestamps,
                feature_names=tuple(FEATURE_NAMES),
                context_window=CONTEXT_WINDOW,
                forecast_horizon=FORECAST_HORIZON,
                target_name="future_mid_price_return",
            )

    def test_forecasting_feature_metadata_is_preserved(self):
        """Feature names should remain attached to the final dataset."""
        dataset = make_forecasting_dataset()

        assert tuple(dataset.feature_names) == tuple(FEATURE_NAMES)

    def test_forecasting_metadata_matches_phase_eight_contract(self):
        """Final metadata should match the NVIDIA forecasting contract."""
        dataset = make_forecasting_dataset()

        assert dataset.context_window == 20
        assert dataset.forecast_horizon == 5
        assert dataset.num_features == 14
        assert dataset.target_name == "future_mid_price_return"
        assert dataset.price_column == "mid_price"

    def test_target_values_property_returns_targets(self):
        """The target_values property should expose the final target array."""
        dataset = make_forecasting_dataset()

        np.testing.assert_array_equal(
            dataset.target_values,
            dataset.targets,
        )

    def test_nonfinite_sequence_values_are_rejected(self):
        """Non-finite model inputs should not reach the forecasting model."""
        sequences = make_sequence_array()
        sequences[0, 5, 3] = np.nan

        with pytest.raises((ValueError, TypeError)):
            ForecastingDataset(
                sequences=sequences,
                targets=make_target_values(),
                timestamps=np.arange(
                    2_000.0,
                    2_000.0 + NUM_SAMPLES,
                    dtype=np.float64,
                ),
                feature_names=tuple(FEATURE_NAMES),
                context_window=CONTEXT_WINDOW,
                forecast_horizon=FORECAST_HORIZON,
                target_name="future_mid_price_return",
            )

    def test_nonfinite_targets_are_rejected(self):
        """Non-finite target values should not reach model training."""
        targets = make_target_values()
        targets[2] = np.inf

        with pytest.raises((ValueError, TypeError)):
            ForecastingDataset(
                sequences=make_sequence_array(),
                targets=targets,
                timestamps=np.arange(
                    2_000.0,
                    2_000.0 + NUM_SAMPLES,
                    dtype=np.float64,
                ),
                feature_names=tuple(FEATURE_NAMES),
                context_window=CONTEXT_WINDOW,
                forecast_horizon=FORECAST_HORIZON,
                target_name="future_mid_price_return",
            )


# ===========================================================================
# Cross-contract integration tests
# ===========================================================================

class TestContractIntegration:
    """Tests covering compatibility between forecasting contracts."""

    def test_sequence_dataset_feature_metadata_matches_forecast_input(self):
        """Sequence metadata should be directly usable by ForecastInput."""
        sequence_dataset = SequenceDataset(
            sequences=make_sequence_array(),
            timestamps=make_sequence_timestamps(),
            feature_names=FEATURE_NAMES,
            context_window=CONTEXT_WINDOW,
        )

        forecast_input = ForecastInput(
            feature_sequence=sequence_dataset.sequences[0].tolist(),
            feature_names=list(sequence_dataset.feature_names),
            timestamps=sequence_dataset.timestamps[0],
            context_window=sequence_dataset.context_window,
            forecast_horizon=FORECAST_HORIZON,
        )

        assert forecast_input.feature_names == FEATURE_NAMES
        assert forecast_input.context_window == CONTEXT_WINDOW
        assert len(forecast_input.feature_sequence) == CONTEXT_WINDOW

    def test_forecasting_dataset_can_produce_single_model_input(self):
        """One final dataset sample should satisfy ForecastInput."""
        dataset = make_forecasting_dataset()

        forecast_input = ForecastInput(
            feature_sequence=dataset.sequences[0].tolist(),
            feature_names=list(dataset.feature_names),
            timestamps=[
                float(index)
                for index in range(dataset.context_window)
            ],
            context_window=dataset.context_window,
            forecast_horizon=dataset.forecast_horizon,
        )

        assert len(forecast_input.feature_sequence) == 20
        assert len(forecast_input.feature_sequence[0]) == 14
        assert len(forecast_input.feature_names) == 14

    def test_target_horizon_matches_forecasting_horizon(self):
        """Target and forecasting contracts must use the same horizon."""
        target_dataset = TargetDataset(
            targets=make_target_values(),
            timestamps=make_target_timestamps(),
            target_name="future_mid_price_return",
            forecast_horizon=FORECAST_HORIZON,
        )

        forecasting_dataset = make_forecasting_dataset()

        assert (
            target_dataset.forecast_horizon
            == forecasting_dataset.forecast_horizon
        )

    def test_final_dataset_has_expected_nvidia_contract(self):
        """
        Verify the complete Phase 8 model-facing contract.

        Expected:
            samples x 20 context rows x 14 features
            one scalar target per sample
            five-step future-return target
        """
        dataset = make_forecasting_dataset()

        assert dataset.sequences.ndim == 3
        assert dataset.sequences.shape[1] == 20
        assert dataset.sequences.shape[2] == 14
        assert dataset.targets.ndim == 1
        assert dataset.targets.shape[0] == dataset.sequences.shape[0]
        assert dataset.timestamps.shape[0] == dataset.sequences.shape[0]
        assert dataset.context_window == 20
        assert dataset.forecast_horizon == 5
        assert dataset.target_name == "future_mid_price_return"

    def test_final_dataset_preserves_temporal_order(self):
        """The model-facing dataset must remain chronologically ordered."""
        dataset = make_forecasting_dataset()

        assert np.all(
            np.diff(dataset.timestamps) > 0
        )

    def test_final_dataset_contains_only_finite_values(self):
        """Inputs and targets should contain finite numeric values."""
        dataset = make_forecasting_dataset()

        assert np.all(np.isfinite(dataset.sequences))
        assert np.all(np.isfinite(dataset.targets))
        assert np.all(np.isfinite(dataset.timestamps))

    def test_forecast_input_and_result_use_same_horizon(self):
        """Input and output contracts should agree on forecast horizon."""
        forecast_input = ForecastInput(
            feature_sequence=make_feature_sequence(),
            feature_names=FEATURE_NAMES,
            timestamps=make_timestamps(),
            context_window=CONTEXT_WINDOW,
            forecast_horizon=FORECAST_HORIZON,
        )

        forecast_result = ForecastResult(
            timestamp=2_000.0,
            forecast_horizon=FORECAST_HORIZON,
            predicted_return=0.001,
            model_name="NVIDIA Forecasting Model",
            model_version="1.0.0",
            model_status=ModelStatus.SUCCESS,
        )

        assert (
            forecast_input.forecast_horizon
            == forecast_result.forecast_horizon
        )


# ===========================================================================
# Enhanced Phase 9 Contract Hardening Tests
# ===========================================================================

class TestForecastingConstantsAndFeatureContract:
    """Tests for the canonical NVIDIA forecasting contract definition."""

    def test_canonical_feature_order_is_exact(self):
        """The model feature order is part of the public contract."""
        from data.contracts import NVIDIA_FORECAST_FEATURES

        assert tuple(FEATURE_NAMES) == NVIDIA_FORECAST_FEATURES
        assert len(NVIDIA_FORECAST_FEATURES) == NUM_FEATURES

    def test_canonical_defaults_match_phase_contract(self):
        """Context, horizon, target, and price defaults remain synchronized."""
        from data.contracts import (
            DEFAULT_FORECAST_CONTEXT_WINDOW,
            DEFAULT_FORECAST_HORIZON,
            DEFAULT_FORECAST_TARGET_NAME,
            DEFAULT_FORECAST_PRICE_COLUMN,
        )

        assert DEFAULT_FORECAST_CONTEXT_WINDOW == CONTEXT_WINDOW
        assert DEFAULT_FORECAST_HORIZON == FORECAST_HORIZON
        assert DEFAULT_FORECAST_TARGET_NAME == "future_mid_price_return"
        assert DEFAULT_FORECAST_PRICE_COLUMN == "mid_price"

    def test_feature_contract_accepts_canonical_order(self):
        """The canonical feature list should validate without error."""
        from data.contracts import validate_nvidia_feature_contract

        validate_nvidia_feature_contract(FEATURE_NAMES)

    def test_feature_contract_rejects_reordered_features(self):
        """Reordering model features must be rejected."""
        from data.contracts import validate_nvidia_feature_contract

        reordered = FEATURE_NAMES.copy()
        reordered[0], reordered[1] = reordered[1], reordered[0]

        with pytest.raises(ValueError, match="contract mismatch"):
            validate_nvidia_feature_contract(reordered)

    def test_feature_contract_rejects_missing_feature(self):
        """Removing one canonical feature must be rejected."""
        from data.contracts import validate_nvidia_feature_contract

        with pytest.raises(ValueError, match="contract mismatch"):
            validate_nvidia_feature_contract(FEATURE_NAMES[:-1])

    def test_feature_contract_rejects_extra_feature(self):
        """Adding an undeclared feature must be rejected."""
        from data.contracts import validate_nvidia_feature_contract

        with pytest.raises(ValueError, match="contract mismatch"):
            validate_nvidia_feature_contract(
                FEATURE_NAMES + ["unexpected_feature"]
            )


class TestForecastInputEnhanced:
    """Additional edge-case tests for the model input contract."""

    def test_shape_and_input_shape_are_per_sample_shape(self):
        """ForecastInput exposes the individual model sample shape."""
        contract = ForecastInput(
            feature_sequence=make_feature_sequence(),
            feature_names=FEATURE_NAMES,
            timestamps=make_timestamps(),
            context_window=CONTEXT_WINDOW,
            forecast_horizon=FORECAST_HORIZON,
        )

        assert contract.shape == (20, 14)
        assert contract.input_shape == (20, 14)
        assert contract.num_features == 14

    def test_earliest_and_latest_timestamps_are_exposed(self):
        """Timestamp convenience properties should identify window boundaries."""
        timestamps = make_timestamps(start=5_000.0)
        contract = ForecastInput(
            feature_sequence=make_feature_sequence(),
            feature_names=FEATURE_NAMES,
            timestamps=timestamps,
            context_window=CONTEXT_WINDOW,
            forecast_horizon=FORECAST_HORIZON,
        )

        assert contract.earliest_timestamp == 5_000.0
        assert contract.latest_timestamp == 5_019.0

    @pytest.mark.parametrize("value", [False, True])
    def test_boolean_context_window_is_rejected(self, value):
        """Booleans must not be accepted as integer configuration values."""
        with pytest.raises((ValueError, TypeError)):
            ForecastInput(
                feature_sequence=make_feature_sequence(),
                feature_names=FEATURE_NAMES,
                timestamps=make_timestamps(),
                context_window=value,
                forecast_horizon=FORECAST_HORIZON,
            )

    @pytest.mark.parametrize("value", [False, True])
    def test_boolean_forecast_horizon_is_rejected(self, value):
        """Booleans must not be accepted as forecast horizons."""
        with pytest.raises((ValueError, TypeError)):
            ForecastInput(
                feature_sequence=make_feature_sequence(),
                feature_names=FEATURE_NAMES,
                timestamps=make_timestamps(),
                context_window=CONTEXT_WINDOW,
                forecast_horizon=value,
            )

    @pytest.mark.parametrize("bad_value", [np.nan, np.inf, -np.inf])
    def test_nonfinite_feature_values_are_rejected(self, bad_value):
        """NaN and infinite features cannot enter model inference."""
        sequence = make_feature_sequence()
        sequence[3][7] = bad_value

        with pytest.raises((ValueError, TypeError)):
            ForecastInput(
                feature_sequence=sequence,
                feature_names=FEATURE_NAMES,
                timestamps=make_timestamps(),
                context_window=CONTEXT_WINDOW,
                forecast_horizon=FORECAST_HORIZON,
            )

    @pytest.mark.parametrize("bad_value", [np.nan, np.inf, -np.inf])
    def test_nonfinite_timestamps_are_rejected(self, bad_value):
        """Timestamps must remain finite and chronological."""
        timestamps = make_timestamps()
        timestamps[4] = bad_value

        with pytest.raises((ValueError, TypeError)):
            ForecastInput(
                feature_sequence=make_feature_sequence(),
                feature_names=FEATURE_NAMES,
                timestamps=timestamps,
                context_window=CONTEXT_WINDOW,
                forecast_horizon=FORECAST_HORIZON,
            )

    def test_noncanonical_feature_order_is_allowed_for_non_nvidia_context(self):
        """Generic non-20-step inputs remain structurally supported."""
        feature_names = ["feature_a", "feature_b"]
        sequence = make_feature_sequence(context_window=10, num_features=2)
        timestamps = make_timestamps(count=10)

        contract = ForecastInput(
            feature_sequence=sequence,
            feature_names=feature_names,
            timestamps=timestamps,
            context_window=10,
            forecast_horizon=3,
        )

        assert contract.shape == (10, 2)


class TestForecastResultEnhanced:
    """Additional validation and derived-field tests for forecast outputs."""

    def make_result(self, **overrides):
        """Build a valid forecast result with optional field overrides."""
        values = {
            "timestamp": 2_000.0,
            "forecast_horizon": FORECAST_HORIZON,
            "predicted_return": 0.001,
            "model_name": "NVIDIA Forecasting Model",
            "model_version": "1.0.0",
            "model_status": ModelStatus.SUCCESS,
        }
        values.update(overrides)
        return ForecastResult(**values)

    @pytest.mark.parametrize("bad_value", [np.nan, np.inf, -np.inf])
    def test_nonfinite_timestamp_is_rejected(self, bad_value):
        """Forecast result timestamps must be finite."""
        with pytest.raises((ValueError, TypeError)):
            self.make_result(timestamp=bad_value)

    @pytest.mark.parametrize("bad_value", [np.nan, np.inf, -np.inf])
    def test_nonfinite_predicted_return_is_rejected(self, bad_value):
        """Forecast returns must be finite."""
        with pytest.raises((ValueError, TypeError)):
            self.make_result(predicted_return=bad_value)

    @pytest.mark.parametrize("bad_value", [np.nan, np.inf, -np.inf])
    def test_nonfinite_confidence_is_rejected(self, bad_value):
        """Confidence must be a finite probability."""
        with pytest.raises((ValueError, TypeError)):
            self.make_result(confidence=bad_value)

    @pytest.mark.parametrize("direction", ["up", "Up", "DOWN", "flat"])
    def test_explicit_direction_is_normalized(self, direction):
        """Explicit valid directions should be normalized to uppercase."""
        result = self.make_result(predicted_direction=direction)
        assert result.predicted_direction == direction.upper()

    def test_invalid_direction_is_rejected(self):
        """Unknown directional labels must not reach downstream consumers."""
        with pytest.raises(ValueError, match="predicted_direction"):
            self.make_result(predicted_direction="SIDEWAYS")

    def test_explicit_expected_move_bps_is_preserved(self):
        """A supplied expected move should not be silently recalculated."""
        result = self.make_result(expected_move_bps=42.5)
        assert result.expected_move_bps == pytest.approx(42.5)

    def test_default_feature_names_are_canonical(self):
        """Forecast results should carry the canonical feature metadata by default."""
        result = self.make_result()
        assert tuple(result.feature_names) == tuple(FEATURE_NAMES)

    def test_metadata_defaults_to_empty_mapping(self):
        """Optional forecast metadata should default to an empty mapping."""
        result = self.make_result()
        assert result.metadata == {}

    def test_custom_metadata_is_preserved(self):
        """Caller-provided metadata should remain attached to the result."""
        metadata = {"source": "test", "request_id": "abc"}
        result = self.make_result(metadata=metadata)
        assert result.metadata == metadata


class TestSequenceDatasetEnhanced:
    """Additional validation for rolling sequence contracts."""

    def make_dataset(self, **overrides):
        """Build a valid sequence dataset with optional overrides."""
        values = {
            "sequences": make_sequence_array(),
            "timestamps": make_sequence_timestamps(),
            "feature_names": FEATURE_NAMES.copy(),
            "context_window": CONTEXT_WINDOW,
        }
        values.update(overrides)
        return SequenceDataset(**values)

    def test_input_shape_matches_full_sequence_shape(self):
        """SequenceDataset retains its full batch shape for compatibility."""
        dataset = self.make_dataset()
        assert dataset.input_shape == dataset.shape
        assert dataset.input_shape == (NUM_SAMPLES, 20, 14)

    def test_timestamp_windows_are_independently_chronological(self):
        """Every sequence must have its own strictly increasing timestamp window."""
        timestamps = make_sequence_timestamps()
        timestamps[3][8], timestamps[3][9] = timestamps[3][9], timestamps[3][8]

        with pytest.raises(ValueError, match="chronological"):
            self.make_dataset(timestamps=timestamps)

    def test_timestamp_window_count_must_match_sequence_count(self):
        """Every rolling sequence needs exactly one timestamp window."""
        with pytest.raises(ValueError, match="one timestamp window"):
            self.make_dataset(timestamps=make_sequence_timestamps()[:-1])

    def test_timestamp_window_length_must_match_context(self):
        """Every timestamp window must contain exactly the context rows."""
        timestamps = make_sequence_timestamps()
        timestamps[0] = timestamps[0][:-1]

        with pytest.raises(ValueError, match="exactly"):
            self.make_dataset(timestamps=timestamps)

    def test_zero_sequences_are_rejected(self):
        """A sequence dataset must contain at least one rolling window."""
        with pytest.raises(ValueError, match="at least one sequence"):
            self.make_dataset(
                sequences=np.empty(
                    (0, CONTEXT_WINDOW, NUM_FEATURES),
                    dtype=np.float64,
                ),
                timestamps=[],
            )

    def test_negative_context_window_is_rejected(self):
        """Context windows must be positive integers."""
        with pytest.raises((ValueError, TypeError)):
            self.make_dataset(context_window=-1)


class TestTargetDatasetEnhanced:
    """Additional target contract and temporal-alignment tests."""

    def make_dataset(self, **overrides):
        """Build a valid target dataset with optional overrides."""
        values = {
            "targets": make_target_values(),
            "timestamps": make_target_timestamps(),
            "target_name": "future_mid_price_return",
            "forecast_horizon": FORECAST_HORIZON,
            "price_column": "mid_price",
        }
        values.update(overrides)
        return TargetDataset(**values)

    def test_target_count_property_matches_values(self):
        """num_targets should reflect the target collection length."""
        dataset = self.make_dataset()
        assert dataset.num_targets == NUM_SAMPLES

    def test_target_values_property_returns_original_values(self):
        """target_values should expose the stored targets."""
        dataset = self.make_dataset()
        np.testing.assert_array_equal(dataset.target_values, dataset.targets)

    def test_empty_targets_are_rejected(self):
        """A target dataset cannot contain zero targets."""
        with pytest.raises(ValueError, match="at least one"):
            self.make_dataset(targets=np.array([], dtype=np.float64), timestamps=[])

    @pytest.mark.parametrize("bad_value", [np.nan, np.inf, -np.inf])
    def test_nonfinite_target_is_rejected(self, bad_value):
        """Targets must remain finite."""
        targets = make_target_values()
        targets[0] = bad_value

        with pytest.raises((ValueError, TypeError)):
            self.make_dataset(targets=targets)

    @pytest.mark.parametrize("bad_horizon", [0, -1])
    def test_nonpositive_horizon_is_rejected(self, bad_horizon):
        """Target horizons must be positive."""
        with pytest.raises((ValueError, TypeError)):
            self.make_dataset(forecast_horizon=bad_horizon)

    def test_target_timestamps_represent_prediction_observations(self):
        """Target timestamps are aligned to the prediction-time observations."""
        dataset = self.make_dataset()
        assert dataset.timestamps[0] == 1_019.0
        assert dataset.timestamps[-1] == 1_026.0


class TestForecastingDatasetEnhanced:
    """Comprehensive validation of the final model-facing dataset."""

    def make_dataset(self, **overrides):
        """Build a valid forecasting dataset with optional overrides."""
        values = {
            "sequences": make_sequence_array(),
            "targets": make_target_values(),
            "timestamps": np.arange(
                2_000.0,
                2_000.0 + NUM_SAMPLES,
                dtype=np.float64,
            ),
            "feature_names": tuple(FEATURE_NAMES),
            "context_window": CONTEXT_WINDOW,
            "forecast_horizon": FORECAST_HORIZON,
            "target_name": "future_mid_price_return",
            "price_column": "mid_price",
        }
        values.update(overrides)
        return ForecastingDataset(**values)

    def test_shape_and_input_shape_have_distinct_meanings(self):
        """Batch shape and single-sample input shape must remain distinct."""
        dataset = self.make_dataset()
        assert dataset.shape == (NUM_SAMPLES, 20, 14)
        assert dataset.input_shape == (20, 14)
        assert dataset.model_input_shape == (20, 14)

    def test_num_samples_and_num_features_are_correct(self):
        """Dataset dimension properties should reflect the tensor."""
        dataset = self.make_dataset()
        assert dataset.num_samples == NUM_SAMPLES
        assert dataset.num_features == NUM_FEATURES

    def test_earliest_and_latest_timestamps_are_correct(self):
        """Timestamp properties should expose dataset boundaries."""
        dataset = self.make_dataset()
        assert dataset.earliest_timestamp == 2_000.0
        assert dataset.latest_timestamp == 2_007.0

    def test_empty_dataset_is_rejected_at_contract_boundary(self):
        """The final forecasting dataset requires at least one sample."""
        with pytest.raises(ValueError, match="at least one sample"):
            self.make_dataset(
                sequences=np.empty(
                    (0, CONTEXT_WINDOW, NUM_FEATURES),
                    dtype=np.float64,
                ),
                targets=np.empty(0, dtype=np.float64),
                timestamps=np.empty(0, dtype=np.float64),
            )

    @pytest.mark.parametrize("bad_value", [np.nan, np.inf, -np.inf])
    def test_nonfinite_timestamp_is_rejected(self, bad_value):
        """Final dataset timestamps must be finite."""
        timestamps = np.arange(
            2_000.0,
            2_000.0 + NUM_SAMPLES,
            dtype=np.float64,
        )
        timestamps[2] = bad_value

        with pytest.raises((ValueError, TypeError)):
            self.make_dataset(timestamps=timestamps)

    def test_boolean_feature_values_are_rejected(self):
        """Boolean model features are not valid numeric market features."""
        sequences = make_sequence_array().astype(object)
        sequences[0, 0, 0] = True

        with pytest.raises((ValueError, TypeError)):
            self.make_dataset(sequences=sequences)

    def test_boolean_targets_are_rejected(self):
        """Boolean targets are not valid numerical forecasts."""
        targets = make_target_values().astype(object)
        targets[0] = True

        with pytest.raises((ValueError, TypeError)):
            self.make_dataset(targets=targets)

    def test_feature_names_are_stored_as_tuple(self):
        """The frozen dataset should carry immutable feature metadata."""
        dataset = self.make_dataset()
        assert isinstance(dataset.feature_names, tuple)
        assert dataset.feature_names == tuple(FEATURE_NAMES)

    def test_dataset_is_frozen(self):
        """ForecastingDataset metadata cannot be reassigned after construction."""
        dataset = self.make_dataset()

        with pytest.raises((AttributeError, TypeError)):
            dataset.context_window = 99

    def test_target_values_property_returns_targets(self):
        """The target_values property should remain a direct dataset view."""
        dataset = self.make_dataset()
        np.testing.assert_array_equal(dataset.target_values, dataset.targets)

    def test_target_name_and_price_column_are_preserved(self):
        """Target metadata must survive contract construction."""
        dataset = self.make_dataset()
        assert dataset.target_name == "future_mid_price_return"
        assert dataset.price_column == "mid_price"

    def test_feature_dimension_must_match_feature_metadata(self):
        """The tensor width must equal the declared feature count."""
        with pytest.raises(ValueError, match="feature dimension"):
            self.make_dataset(
                sequences=np.zeros(
                    (NUM_SAMPLES, CONTEXT_WINDOW, NUM_FEATURES - 1),
                    dtype=np.float64,
                )
            )

    def test_context_dimension_must_match_context_window(self):
        """The tensor context dimension must equal context_window."""
        with pytest.raises(ValueError, match="context dimension"):
            self.make_dataset(
                sequences=np.zeros(
                    (NUM_SAMPLES, CONTEXT_WINDOW - 1, NUM_FEATURES),
                    dtype=np.float64,
                )
            )

    def test_target_count_must_match_sequence_count(self):
        """Each model sequence must have exactly one target."""
        with pytest.raises(ValueError, match="same number"):
            self.make_dataset(
                targets=make_target_values()[:-1]
            )

    def test_timestamp_count_must_match_sequence_count(self):
        """Each model sequence must have exactly one ending timestamp."""
        with pytest.raises(ValueError, match="same number"):
            self.make_dataset(
                timestamps=np.arange(
                    2_000.0,
                    2_000.0 + NUM_SAMPLES - 1,
                    dtype=np.float64,
                )
            )

    def test_temporal_order_is_strictly_increasing(self):
        """Final sample timestamps must remain strictly chronological."""
        dataset = self.make_dataset()
        assert np.all(np.diff(dataset.timestamps) > 0)

    def test_duplicate_timestamps_are_rejected(self):
        """Duplicate sample timestamps create ambiguous alignment."""
        timestamps = np.arange(
            2_000.0,
            2_000.0 + NUM_SAMPLES,
            dtype=np.float64,
        )
        timestamps[4] = timestamps[3]

        with pytest.raises(ValueError, match="unique"):
            self.make_dataset(timestamps=timestamps)

    def test_canonical_metadata_matches_nvidia_contract(self):
        """The final dataset metadata must match the canonical NVIDIA contract."""
        from data.contracts import (
            DEFAULT_FORECAST_CONTEXT_WINDOW,
            DEFAULT_FORECAST_HORIZON,
            DEFAULT_FORECAST_TARGET_NAME,
            DEFAULT_FORECAST_PRICE_COLUMN,
            NVIDIA_FORECAST_FEATURES,
        )

        dataset = self.make_dataset()

        assert dataset.context_window == DEFAULT_FORECAST_CONTEXT_WINDOW
        assert dataset.forecast_horizon == DEFAULT_FORECAST_HORIZON
        assert tuple(dataset.feature_names) == NVIDIA_FORECAST_FEATURES
        assert dataset.target_name == DEFAULT_FORECAST_TARGET_NAME
        assert dataset.price_column == DEFAULT_FORECAST_PRICE_COLUMN

    def test_get_sample_returns_sequence_target_and_timestamp(self):
        """One aligned sample should expose all three aligned components."""
        dataset = self.make_dataset()
        sequence, target, timestamp = dataset.get_sample(2)

        np.testing.assert_array_equal(sequence, dataset.sequences[2])
        assert target == pytest.approx(float(dataset.targets[2]))
        assert timestamp == pytest.approx(float(dataset.timestamps[2]))

    def test_get_sample_rejects_boolean_index(self):
        """Boolean values must not be accepted as sample indices."""
        dataset = self.make_dataset()

        with pytest.raises(TypeError, match="index"):
            dataset.get_sample(True)

    def test_get_sample_rejects_non_integer_index(self):
        """Sample indexing requires an integer."""
        dataset = self.make_dataset()

        with pytest.raises(TypeError, match="index"):
            dataset.get_sample("0")

    @pytest.mark.parametrize("index", [-1, NUM_SAMPLES])
    def test_get_sample_rejects_out_of_range_index(self, index):
        """Out-of-range samples must be rejected."""
        dataset = self.make_dataset()

        with pytest.raises(IndexError, match="out of range"):
            dataset.get_sample(index)

    def test_to_model_input_returns_single_sample_shape(self):
        """One model input must have shape (20, 14)."""
        dataset = self.make_dataset()
        model_input = dataset.to_model_input(0)

        assert model_input.shape == (20, 14)
        np.testing.assert_array_equal(model_input, dataset.sequences[0])

    def test_to_model_input_copy_is_independent(self):
        """The default model-input conversion should return an independent array."""
        dataset = self.make_dataset()
        model_input = dataset.to_model_input(0)
        original_value = dataset.sequences[0, 0, 0]

        model_input[0, 0] = original_value + 999.0

        assert dataset.sequences[0, 0, 0] == original_value

    def test_to_model_input_can_return_view_when_copy_disabled(self):
        """copy=False should expose the underlying sequence for advanced callers."""
        dataset = self.make_dataset()
        model_input = dataset.to_model_input(0, copy=False)

        assert np.shares_memory(model_input, dataset.sequences)

    def test_to_model_batch_returns_complete_model_shape(self):
        """The model batch should preserve the full (N, 20, 14) tensor."""
        dataset = self.make_dataset()
        model_batch = dataset.to_model_batch()

        assert model_batch.shape == (NUM_SAMPLES, 20, 14)
        np.testing.assert_array_equal(model_batch, dataset.sequences)

    def test_to_model_batch_copy_is_independent(self):
        """The default model-batch conversion should return an independent array."""
        dataset = self.make_dataset()
        model_batch = dataset.to_model_batch()
        original_value = dataset.sequences[0, 0, 0]

        model_batch[0, 0, 0] = original_value + 999.0

        assert dataset.sequences[0, 0, 0] == original_value

    def test_to_model_batch_can_return_underlying_array(self):
        """copy=False should return the stored feature tensor directly."""
        dataset = self.make_dataset()
        model_batch = dataset.to_model_batch(copy=False)

        assert model_batch is dataset.sequences

    def test_copy_produces_independent_dataset(self):
        """Dataset copy should duplicate arrays while preserving metadata."""
        dataset = self.make_dataset()
        copied = dataset.copy()

        assert copied is not dataset
        assert copied.feature_names == dataset.feature_names
        assert copied.context_window == dataset.context_window
        assert copied.forecast_horizon == dataset.forecast_horizon
        assert copied.target_name == dataset.target_name
        assert copied.price_column == dataset.price_column

        copied.sequences[0, 0, 0] += 100.0
        copied.targets[0] += 100.0
        copied.timestamps[0] += 100.0

        assert not np.array_equal(copied.sequences, dataset.sequences)
        assert not np.array_equal(copied.targets, dataset.targets)
        assert not np.array_equal(copied.timestamps, dataset.timestamps)


class TestContractIntegrationEnhanced:
    """End-to-end compatibility checks across the forecasting contracts."""

    def test_sequence_to_forecast_input_preserves_feature_order(self):
        """A rolling sequence can be converted without changing feature order."""
        sequence_dataset = SequenceDataset(
            sequences=make_sequence_array(),
            timestamps=make_sequence_timestamps(),
            feature_names=FEATURE_NAMES,
            context_window=CONTEXT_WINDOW,
        )

        forecast_input = ForecastInput(
            feature_sequence=sequence_dataset.sequences[0].tolist(),
            feature_names=list(sequence_dataset.feature_names),
            timestamps=sequence_dataset.timestamps[0],
            context_window=sequence_dataset.context_window,
            forecast_horizon=FORECAST_HORIZON,
        )

        assert forecast_input.feature_names == FEATURE_NAMES
        assert forecast_input.shape == (20, 14)

    def test_target_and_final_dataset_horizons_match(self):
        """Target and aligned forecasting datasets must use the same horizon."""
        target_dataset = TargetDataset(
            targets=make_target_values(),
            timestamps=make_target_timestamps(),
            target_name="future_mid_price_return",
            forecast_horizon=FORECAST_HORIZON,
        )
        forecasting_dataset = ForecastingDataset(
            sequences=make_sequence_array(),
            targets=make_target_values(),
            timestamps=np.arange(
                2_000.0,
                2_000.0 + NUM_SAMPLES,
                dtype=np.float64,
            ),
            feature_names=tuple(FEATURE_NAMES),
            context_window=CONTEXT_WINDOW,
            forecast_horizon=FORECAST_HORIZON,
            target_name="future_mid_price_return",
            price_column="mid_price",
        )

        assert target_dataset.forecast_horizon == forecasting_dataset.forecast_horizon

    def test_final_dataset_contains_only_finite_values(self):
        """Every model-facing numeric array must be finite."""
        dataset = ForecastingDataset(
            sequences=make_sequence_array(),
            targets=make_target_values(),
            timestamps=np.arange(
                2_000.0,
                2_000.0 + NUM_SAMPLES,
                dtype=np.float64,
            ),
            feature_names=tuple(FEATURE_NAMES),
            context_window=CONTEXT_WINDOW,
            forecast_horizon=FORECAST_HORIZON,
            target_name="future_mid_price_return",
            price_column="mid_price",
        )

        assert np.isfinite(dataset.sequences).all()
        assert np.isfinite(dataset.targets).all()
        assert np.isfinite(dataset.timestamps).all()

    def test_forecast_result_horizon_matches_forecast_input_horizon(self):
        """Model input and output contracts must agree on the forecast horizon."""
        forecast_input = ForecastInput(
            feature_sequence=make_feature_sequence(),
            feature_names=FEATURE_NAMES,
            timestamps=make_timestamps(),
            context_window=CONTEXT_WINDOW,
            forecast_horizon=FORECAST_HORIZON,
        )
        forecast_result = ForecastResult(
            timestamp=2_000.0,
            forecast_horizon=FORECAST_HORIZON,
            predicted_return=0.001,
            model_name="NVIDIA Forecasting Model",
            model_version="1.0.0",
            model_status=ModelStatus.SUCCESS,
        )

        assert forecast_input.forecast_horizon == forecast_result.forecast_horizon

    def test_final_dataset_feature_order_matches_forecast_input(self):
        """Dataset metadata must be directly usable by the model input contract."""
        dataset = ForecastingDataset(
            sequences=make_sequence_array(),
            targets=make_target_values(),
            timestamps=np.arange(
                2_000.0,
                2_000.0 + NUM_SAMPLES,
                dtype=np.float64,
            ),
            feature_names=tuple(FEATURE_NAMES),
            context_window=CONTEXT_WINDOW,
            forecast_horizon=FORECAST_HORIZON,
            target_name="future_mid_price_return",
            price_column="mid_price",
        )

        forecast_input = ForecastInput(
            feature_sequence=dataset.sequences[0].tolist(),
            feature_names=list(dataset.feature_names),
            timestamps=make_timestamps(),
            context_window=dataset.context_window,
            forecast_horizon=dataset.forecast_horizon,
        )

        assert tuple(forecast_input.feature_names) == dataset.feature_names
        assert forecast_input.shape == dataset.input_shape

    def test_forecast_result_expected_move_bps_matches_target_units(self):
        """A return of 0.001 corresponds to a 10 basis-point expected move."""
        result = ForecastResult(
            timestamp=2_000.0,
            forecast_horizon=FORECAST_HORIZON,
            predicted_return=0.001,
            model_name="NVIDIA Forecasting Model",
            model_version="1.0.0",
            model_status=ModelStatus.SUCCESS,
        )

        assert result.expected_move_bps == pytest.approx(10.0)


