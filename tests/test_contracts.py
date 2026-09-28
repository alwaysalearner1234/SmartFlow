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

