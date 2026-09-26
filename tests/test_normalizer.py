# test_normalizer.py

# Tests for SmartFlow Forecasting Feature Normalization

# Verifies that forecasting feature normalization:
#
#     - fits only on training data
#     - preserves dataset structure and metadata
#     - produces deterministic results
#     - handles constant features safely
#     - rejects invalid input
#     - respects ForecastingConfig
#     - prevents validation/test data from influencing the scaler
#
# These tests intentionally focus only on Phase 7 normalization behavior.


from __future__ import annotations

from typing import Tuple

import numpy as np
import pytest

from config.config import ForecastingConfig
from data.forecasting_dataset import ForecastingDataset
from data.normalizer import (
    DEFAULT_NORMALIZATION_METHOD,
    SUPPORTED_NORMALIZATION_METHODS,
    ForecastingDatasetNormalizer,
    NormalizerConfig,
    StandardScaler,
    fit_normalizer,
    normalize_forecasting_splits,
    normalize_from_forecasting_config,
)


# ============================================================================
# Test Constants
# ============================================================================

FEATURE_NAMES: Tuple[str, ...] = (
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

CONTEXT_WINDOW = 4
NUM_FEATURES = len(FEATURE_NAMES)
FORECAST_HORIZON = 5


# ============================================================================
# Dataset Helpers
# ============================================================================

def create_dataset(
    num_samples: int = 10,
    start_value: float = 1.0,
    value_step: float = 1.0,
    target_start: float = 0.01,
    timestamp_start: float = 100.0,
) -> ForecastingDataset:
    """
    Create a deterministic ForecastingDataset for testing.

    Each feature receives a deterministic offset so that feature-wise
    normalization can be verified independently.
    """

    total_values = (
        num_samples
        * CONTEXT_WINDOW
        * NUM_FEATURES
    )

    base = np.arange(
        total_values,
        dtype=np.float64,
    ).reshape(
        num_samples,
        CONTEXT_WINDOW,
        NUM_FEATURES,
    )

    sequences = (
        start_value
        + (base * value_step)
    )

    targets = (
        target_start
        + np.arange(
            num_samples,
            dtype=np.float64,
        )
        * 0.01
    )

    timestamps = (
        timestamp_start
        + np.arange(
            num_samples,
            dtype=np.float64,
        )
    )

    return ForecastingDataset(
        sequences=sequences,
        targets=targets,
        timestamps=timestamps,
        feature_names=FEATURE_NAMES,
        context_window=CONTEXT_WINDOW,
        forecast_horizon=FORECAST_HORIZON,
        target_name="future_mid_price_return",
        price_column="mid_price",
    )


def create_constant_feature_dataset(
    num_samples: int = 10,
    constant_value: float = 7.0,
) -> ForecastingDataset:
    """
    Create a dataset where the first feature is constant.

    The remaining features contain deterministic variation.
    """

    dataset = create_dataset(
        num_samples=num_samples,
    )

    sequences = dataset.sequences.copy()

    sequences[:, :, 0] = constant_value

    return ForecastingDataset(
        sequences=sequences,
        targets=dataset.targets.copy(),
        timestamps=dataset.timestamps.copy(),
        feature_names=dataset.feature_names,
        context_window=dataset.context_window,
        forecast_horizon=dataset.forecast_horizon,
        target_name=dataset.target_name,
        price_column=dataset.price_column,
    )


def create_single_sample_dataset() -> ForecastingDataset:
    """Create the smallest valid forecasting dataset."""

    sequences = np.ones(
        (1, CONTEXT_WINDOW, NUM_FEATURES),
        dtype=np.float64,
    )

    targets = np.array(
        [0.01],
        dtype=np.float64,
    )

    timestamps = np.array(
        [100.0],
        dtype=np.float64,
    )

    return ForecastingDataset(
        sequences=sequences,
        targets=targets,
        timestamps=timestamps,
        feature_names=FEATURE_NAMES,
        context_window=CONTEXT_WINDOW,
        forecast_horizon=FORECAST_HORIZON,
        target_name="future_mid_price_return",
        price_column="mid_price",
    )


# ============================================================================
# Basic Configuration Tests
# ============================================================================

class TestNormalizationConstants:
    """Tests for normalization constants."""

    def test_default_normalization_method_is_standard(self) -> None:
        """The default normalization method should be standardization."""

        assert DEFAULT_NORMALIZATION_METHOD == "standard"

    def test_standard_normalization_is_supported(self) -> None:
        """Standard normalization should be listed as supported."""

        assert "standard" in SUPPORTED_NORMALIZATION_METHODS


class TestNormalizerConfiguration:
    """Tests for NormalizerConfig."""

    def test_default_configuration_is_training_only(self) -> None:
        """The default configuration must require training-only fitting."""

        config = NormalizerConfig()

        assert config.method == "standard"
        assert config.normalize_features is True
        assert config.fit_on_training_only is True
        assert config.dtype == "float64"
        assert config.copy_arrays is True

    def test_unsupported_method_is_rejected(self) -> None:
        """Unsupported normalization methods should be rejected."""

        with pytest.raises(
            ValueError,
            match="Unsupported normalization method",
        ):
            ForecastingDatasetNormalizer(
                NormalizerConfig(
                    method="minmax",
                )
            )

    def test_empty_method_is_rejected(self) -> None:
        """An empty normalization method should be rejected."""

        with pytest.raises(
            ValueError,
            match="must not be empty",
        ):
            ForecastingDatasetNormalizer(
                NormalizerConfig(
                    method="",
                )
            )

    def test_non_string_method_is_rejected(self) -> None:
        """A non-string normalization method should be rejected."""

        with pytest.raises(
            TypeError,
            match="must be a string",
        ):
            ForecastingDatasetNormalizer(
                NormalizerConfig(
                    method=123,  # type: ignore[arg-type]
                )
            )

    def test_training_only_setting_cannot_be_disabled(self) -> None:
        """The forecasting normalizer must require training-only fitting."""

        with pytest.raises(
            ValueError,
            match="fit_on_training_only must be True",
        ):
            ForecastingDatasetNormalizer(
                NormalizerConfig(
                    fit_on_training_only=False,
                )
            )

    def test_invalid_dtype_is_rejected(self) -> None:
        """Invalid NumPy dtypes should be rejected."""

        with pytest.raises(
            TypeError,
            match="Invalid normalization dtype",
        ):
            ForecastingDatasetNormalizer(
                NormalizerConfig(
                    dtype="not-a-real-dtype",
                )
            )


# ============================================================================
# Basic Normalization Tests
# ============================================================================

class TestBasicNormalization:
    """Tests for basic standard normalization."""

    def test_fit_creates_scaler(self) -> None:
        """Fitting should create a reusable scaler."""

        dataset = create_dataset()

        normalizer = ForecastingDatasetNormalizer()

        assert normalizer.fitted is False
        assert normalizer.scaler is None

        normalizer.fit(dataset)

        assert normalizer.fitted is True
        assert normalizer.scaler is not None

    def test_fitted_scaler_contains_correct_feature_count(self) -> None:
        """The fitted scaler should contain one statistic per feature."""

        dataset = create_dataset()

        normalizer = ForecastingDatasetNormalizer()

        normalizer.fit(dataset)

        assert normalizer.scaler is not None
        assert normalizer.scaler.num_features == NUM_FEATURES

    def test_feature_names_are_preserved(self) -> None:
        """The fitted scaler should preserve the training feature order."""

        dataset = create_dataset()

        normalizer = ForecastingDatasetNormalizer()

        normalizer.fit(dataset)

        assert normalizer.feature_names == FEATURE_NAMES

    def test_transform_returns_forecasting_dataset(self) -> None:
        """Transformation should return the existing dataset contract."""

        dataset = create_dataset()

        normalizer = ForecastingDatasetNormalizer()

        normalizer.fit(dataset)

        transformed = normalizer.transform(dataset)

        assert isinstance(
            transformed,
            ForecastingDataset,
        )

    def test_normalized_shape_is_preserved(self) -> None:
        """Normalization must preserve the sequence tensor shape."""

        dataset = create_dataset(
            num_samples=12,
        )

        normalizer = ForecastingDatasetNormalizer()

        normalizer.fit(dataset)

        transformed = normalizer.transform(dataset)

        assert transformed.sequences.shape == dataset.sequences.shape

    def test_standardized_training_features_have_zero_mean(self) -> None:
        """
        Training features should have approximately zero mean after
        standardization.
        """

        dataset = create_dataset(
            num_samples=20,
        )

        normalizer = ForecastingDatasetNormalizer()

        transformed = normalizer.fit_transform(
            dataset
        )

        flattened = transformed.sequences.reshape(
            -1,
            NUM_FEATURES,
        )

        means = np.mean(
            flattened,
            axis=0,
        )

        assert np.allclose(
            means,
            0.0,
            atol=1e-12,
        )

    def test_standardized_training_features_have_unit_variance(self) -> None:
        """
        Training features should have approximately unit variance after
        standardization.
        """

        dataset = create_dataset(
            num_samples=20,
        )

        normalizer = ForecastingDatasetNormalizer()

        transformed = normalizer.fit_transform(
            dataset
        )

        flattened = transformed.sequences.reshape(
            -1,
            NUM_FEATURES,
        )

        standard_deviations = np.std(
            flattened,
            axis=0,
        )

        assert np.allclose(
            standard_deviations,
            1.0,
            atol=1e-12,
        )


# ============================================================================
# Training-Only Fitting Tests
# ============================================================================

class TestTrainingOnlyFitting:
    """Tests ensuring scaler statistics come exclusively from training data."""

    def test_scaler_statistics_match_training_data(self) -> None:
        """Scaler means and scales must be calculated from training data."""

        training = create_dataset(
            num_samples=10,
            start_value=1.0,
        )

        normalizer = ForecastingDatasetNormalizer()

        normalizer.fit(training)

        flattened = training.sequences.reshape(
            -1,
            NUM_FEATURES,
        )

        expected_means = np.mean(
            flattened,
            axis=0,
        )

        expected_scales = np.std(
            flattened,
            axis=0,
        )

        expected_scales = np.where(
            expected_scales == 0.0,
            1.0,
            expected_scales,
        )

        assert normalizer.scaler is not None

        assert np.allclose(
            normalizer.scaler.means,
            expected_means,
        )

        assert np.allclose(
            normalizer.scaler.scales,
            expected_scales,
        )

    def test_validation_data_does_not_change_scaler(self) -> None:
        """
        Transforming validation data must not modify training-fitted
        statistics.
        """

        training = create_dataset(
            num_samples=10,
            start_value=1.0,
        )

        validation = create_dataset(
            num_samples=10,
            start_value=100000.0,
        )

        normalizer = ForecastingDatasetNormalizer()

        normalizer.fit(training)

        assert normalizer.scaler is not None

        means_before = normalizer.scaler.means.copy()
        scales_before = normalizer.scaler.scales.copy()

        normalizer.transform(validation)

        assert np.array_equal(
            normalizer.scaler.means,
            means_before,
        )

        assert np.array_equal(
            normalizer.scaler.scales,
            scales_before,
        )

    def test_test_data_does_not_change_scaler(self) -> None:
        """
        Transforming test data must not modify training-fitted statistics.
        """

        training = create_dataset(
            num_samples=10,
            start_value=1.0,
        )

        test = create_dataset(
            num_samples=10,
            start_value=500000.0,
        )

        normalizer = ForecastingDatasetNormalizer()

        normalizer.fit(training)

        assert normalizer.scaler is not None

        means_before = normalizer.scaler.means.copy()
        scales_before = normalizer.scaler.scales.copy()

        normalizer.transform(test)

        assert np.array_equal(
            normalizer.scaler.means,
            means_before,
        )

        assert np.array_equal(
            normalizer.scaler.scales,
            scales_before,
        )

    def test_validation_values_are_transformed_using_training_statistics(
        self,
    ) -> None:
        """Validation values must use training means and scales."""

        training = create_dataset(
            num_samples=10,
            start_value=1.0,
        )

        validation = create_dataset(
            num_samples=5,
            start_value=1000.0,
        )

        normalizer = ForecastingDatasetNormalizer()

        normalizer.fit(training)

        transformed = normalizer.transform(
            validation
        )

        assert normalizer.scaler is not None

        expected = (
            validation.sequences
            - normalizer.scaler.means.reshape(
                1,
                1,
                -1,
            )
        ) / normalizer.scaler.scales.reshape(
            1,
            1,
            -1,
        )

        assert np.allclose(
            transformed.sequences,
            expected,
        )

    def test_large_validation_shift_does_not_recenter_validation(
        self,
    ) -> None:
        """
        Validation data shifted far from training data should remain shifted
        after applying the training scaler.
        """

        training = create_dataset(
            num_samples=10,
            start_value=0.0,
        )

        validation = create_dataset(
            num_samples=5,
            start_value=1000000.0,
        )

        normalizer = ForecastingDatasetNormalizer()

        normalizer.fit(training)

        transformed = normalizer.transform(
            validation
        )

        flattened = transformed.sequences.reshape(
            -1,
            NUM_FEATURES,
        )

        # If validation data had incorrectly been used to fit its own scaler,
        # its mean would be approximately zero. It should not be zero here.
        assert not np.allclose(
            np.mean(flattened, axis=0),
            0.0,
        )


# ============================================================================
# Split Transformation Tests
# ============================================================================

class TestSplitTransformation:
    """Tests for simultaneous train/validation/test normalization."""

    def test_transform_splits_returns_three_datasets(self) -> None:
        """All three forecasting partitions should be returned."""

        training = create_dataset(
            num_samples=10,
        )

        validation = create_dataset(
            num_samples=5,
            start_value=100.0,
        )

        test = create_dataset(
            num_samples=5,
            start_value=200.0,
        )

        normalizer = ForecastingDatasetNormalizer()

        result = normalizer.transform_splits(
            training_dataset=training,
            validation_dataset=validation,
            test_dataset=test,
        )

        assert len(result) == 3

        normalized_training = result[0]
        normalized_validation = result[1]
        normalized_test = result[2]

        assert isinstance(
            normalized_training,
            ForecastingDataset,
        )

        assert isinstance(
            normalized_validation,
            ForecastingDataset,
        )

        assert isinstance(
            normalized_test,
            ForecastingDataset,
        )

    def test_training_validation_and_test_shapes_are_preserved(
        self,
    ) -> None:
        """All partition shapes should remain unchanged."""

        training = create_dataset(
            num_samples=10,
        )

        validation = create_dataset(
            num_samples=5,
            start_value=100.0,
        )

        test = create_dataset(
            num_samples=7,
            start_value=200.0,
        )

        normalizer = ForecastingDatasetNormalizer()

        (
            normalized_training,
            normalized_validation,
            normalized_test,
        ) = normalizer.transform_splits(
            training,
            validation,
            test,
        )

        assert (
            normalized_training.sequences.shape
            == training.sequences.shape
        )

        assert (
            normalized_validation.sequences.shape
            == validation.sequences.shape
        )

        assert (
            normalized_test.sequences.shape
            == test.sequences.shape
        )

    def test_transform_splits_fits_only_once_on_training_data(
        self,
    ) -> None:
        """The resulting scaler must contain training-only statistics."""

        training = create_dataset(
            num_samples=10,
            start_value=1.0,
        )

        validation = create_dataset(
            num_samples=10,
            start_value=10000.0,
        )

        test = create_dataset(
            num_samples=10,
            start_value=20000.0,
        )

        normalizer = ForecastingDatasetNormalizer()

        normalizer.transform_splits(
            training,
            validation,
            test,
        )

        assert normalizer.scaler is not None

        training_flattened = training.sequences.reshape(
            -1,
            NUM_FEATURES,
        )

        expected_means = np.mean(
            training_flattened,
            axis=0,
        )

        assert np.allclose(
            normalizer.scaler.means,
            expected_means,
        )

    def test_convenience_split_function_matches_normalizer(
        self,
    ) -> None:
        """The convenience function should perform the same operation."""

        training = create_dataset(
            num_samples=10,
        )

        validation = create_dataset(
            num_samples=5,
            start_value=100.0,
        )

        test = create_dataset(
            num_samples=5,
            start_value=200.0,
        )

        expected_normalizer = ForecastingDatasetNormalizer()

        expected = expected_normalizer.transform_splits(
            training,
            validation,
            test,
        )

        actual = normalize_forecasting_splits(
            training,
            validation,
            test,
        )

        for expected_dataset, actual_dataset in zip(
            expected,
            actual,
        ):
            assert np.allclose(
                expected_dataset.sequences,
                actual_dataset.sequences,
            )


# ============================================================================
# Dataset Preservation Tests
# ============================================================================

class TestDatasetPreservation:
    """Tests ensuring normalization does not alter non-feature data."""

    def test_targets_are_unchanged(self) -> None:
        """Normalization must never modify forecasting targets."""

        dataset = create_dataset()

        normalizer = ForecastingDatasetNormalizer()

        transformed = normalizer.fit_transform(
            dataset
        )

        assert np.array_equal(
            transformed.targets,
            dataset.targets,
        )

    def test_timestamps_are_unchanged(self) -> None:
        """Normalization must preserve timestamps exactly."""

        dataset = create_dataset()

        normalizer = ForecastingDatasetNormalizer()

        transformed = normalizer.fit_transform(
            dataset
        )

        assert np.array_equal(
            transformed.timestamps,
            dataset.timestamps,
        )

    def test_feature_names_are_unchanged(self) -> None:
        """Normalization must preserve feature ordering."""

        dataset = create_dataset()

        normalizer = ForecastingDatasetNormalizer()

        transformed = normalizer.fit_transform(
            dataset
        )

        assert transformed.feature_names == dataset.feature_names

    def test_context_window_is_unchanged(self) -> None:
        """Normalization must preserve the context window."""

        dataset = create_dataset()

        normalizer = ForecastingDatasetNormalizer()

        transformed = normalizer.fit_transform(
            dataset
        )

        assert (
            transformed.context_window
            == dataset.context_window
        )

    def test_forecast_horizon_is_unchanged(self) -> None:
        """Normalization must preserve the forecast horizon."""

        dataset = create_dataset()

        normalizer = ForecastingDatasetNormalizer()

        transformed = normalizer.fit_transform(
            dataset
        )

        assert (
            transformed.forecast_horizon
            == dataset.forecast_horizon
        )

    def test_target_name_is_unchanged(self) -> None:
        """Normalization must preserve the target name."""

        dataset = create_dataset()

        normalizer = ForecastingDatasetNormalizer()

        transformed = normalizer.fit_transform(
            dataset
        )

        assert (
            transformed.target_name
            == dataset.target_name
        )

    def test_price_column_is_unchanged(self) -> None:
        """Normalization must preserve the source price column."""

        dataset = create_dataset()

        normalizer = ForecastingDatasetNormalizer()

        transformed = normalizer.fit_transform(
            dataset
        )

        assert (
            transformed.price_column
            == dataset.price_column
        )

    def test_original_dataset_is_not_modified(self) -> None:
        """Normalization should not mutate the input dataset."""

        dataset = create_dataset()

        original_sequences = dataset.sequences.copy()
        original_targets = dataset.targets.copy()
        original_timestamps = dataset.timestamps.copy()

        normalizer = ForecastingDatasetNormalizer()

        normalizer.fit_transform(
            dataset
        )

        assert np.array_equal(
            dataset.sequences,
            original_sequences,
        )

        assert np.array_equal(
            dataset.targets,
            original_targets,
        )

        assert np.array_equal(
            dataset.timestamps,
            original_timestamps,
        )


# ============================================================================
# Constant Feature Tests
# ============================================================================

class TestConstantFeatures:
    """Tests for zero-variance feature handling."""

    def test_constant_feature_does_not_produce_nan(self) -> None:
        """Constant features must not produce NaN values."""

        dataset = create_constant_feature_dataset()

        normalizer = ForecastingDatasetNormalizer()

        transformed = normalizer.fit_transform(
            dataset
        )

        assert not np.any(
            np.isnan(
                transformed.sequences
            )
        )

    def test_constant_feature_does_not_produce_infinity(self) -> None:
        """Constant features must not produce infinite values."""

        dataset = create_constant_feature_dataset()

        normalizer = ForecastingDatasetNormalizer()

        transformed = normalizer.fit_transform(
            dataset
        )

        assert np.all(
            np.isfinite(
                transformed.sequences
            )
        )

    def test_constant_feature_normalizes_to_zero(self) -> None:
        """A constant feature should become zero after standardization."""

        dataset = create_constant_feature_dataset(
            constant_value=7.0,
        )

        normalizer = ForecastingDatasetNormalizer()

        transformed = normalizer.fit_transform(
            dataset
        )

        assert np.allclose(
            transformed.sequences[:, :, 0],
            0.0,
        )

    def test_constant_feature_uses_unit_scale(self) -> None:
        """Zero-variance features should use a unit scale internally."""

        dataset = create_constant_feature_dataset()

        normalizer = ForecastingDatasetNormalizer()

        normalizer.fit(dataset)

        assert normalizer.scaler is not None

        assert normalizer.scaler.scales[0] == 1.0


# ============================================================================
# Determinism Tests
# ============================================================================

class TestDeterminism:
    """Tests for deterministic normalization behavior."""

    def test_repeated_fit_produces_same_statistics(self) -> None:
        """Repeated fitting on identical data must be deterministic."""

        dataset = create_dataset()

        first = ForecastingDatasetNormalizer()
        second = ForecastingDatasetNormalizer()

        first.fit(dataset)
        second.fit(dataset)

        assert first.scaler is not None
        assert second.scaler is not None

        assert np.array_equal(
            first.scaler.means,
            second.scaler.means,
        )

        assert np.array_equal(
            first.scaler.scales,
            second.scaler.scales,
        )

    def test_repeated_transformation_is_identical(self) -> None:
        """Repeated transformation with one scaler must be deterministic."""

        dataset = create_dataset()

        normalizer = ForecastingDatasetNormalizer()

        normalizer.fit(dataset)

        first = normalizer.transform(dataset)
        second = normalizer.transform(dataset)

        assert np.array_equal(
            first.sequences,
            second.sequences,
        )

    def test_fit_transform_matches_fit_then_transform(self) -> None:
        """fit_transform should match explicit fit followed by transform."""

        dataset = create_dataset()

        first = ForecastingDatasetNormalizer()

        first_result = first.fit_transform(
            dataset
        )

        second = ForecastingDatasetNormalizer()

        second.fit(dataset)

        second_result = second.transform(
            dataset
        )

        assert np.array_equal(
            first_result.sequences,
            second_result.sequences,
        )


# ============================================================================
# Scaler Reuse Tests
# ============================================================================

class TestScalerReuse:
    """Tests for reusing a fitted scaler."""

    def test_fitted_scaler_can_transform_new_dataset(self) -> None:
        """A fitted scaler should work on compatible new data."""

        training = create_dataset(
            num_samples=10,
        )

        new_dataset = create_dataset(
            num_samples=5,
            start_value=50.0,
        )

        normalizer = ForecastingDatasetNormalizer()

        normalizer.fit(training)

        transformed = normalizer.transform(
            new_dataset
        )

        assert transformed.num_samples == 5

        assert transformed.num_features == NUM_FEATURES

    def test_unfitted_normalizer_cannot_transform(self) -> None:
        """Transformation before fitting should be rejected."""

        dataset = create_dataset()

        normalizer = ForecastingDatasetNormalizer()

        with pytest.raises(
            RuntimeError,
            match="has not been fitted",
        ):
            normalizer.transform(dataset)

    def test_scaler_feature_names_are_checked(self) -> None:
        """A mismatched feature contract should be rejected."""

        training = create_dataset()

        different_feature_names = tuple(
            f"different_feature_{index}"
            for index in range(NUM_FEATURES)
        )

        incompatible = ForecastingDataset(
            sequences=training.sequences.copy(),
            targets=training.targets.copy(),
            timestamps=training.timestamps.copy(),
            feature_names=different_feature_names,
            context_window=training.context_window,
            forecast_horizon=training.forecast_horizon,
            target_name=training.target_name,
            price_column=training.price_column,
        )

        normalizer = ForecastingDatasetNormalizer()

        normalizer.fit(training)

        with pytest.raises(
            ValueError,
            match="feature names do not match",
        ):
            normalizer.transform(incompatible)

    def test_scaler_transform_checks_feature_names(self) -> None:
        """StandardScaler should independently validate feature ordering."""

        dataset = create_dataset()

        normalizer = ForecastingDatasetNormalizer()

        normalizer.fit(dataset)

        assert normalizer.scaler is not None

        with pytest.raises(
            ValueError,
            match="feature names do not match",
        ):
            normalizer.scaler.transform(
                dataset.sequences,
                tuple(
                    reversed(dataset.feature_names)
                ),
            )


# ============================================================================
# Disabled Normalization Tests
# ============================================================================

class TestDisabledNormalization:
    """Tests for normalize_features=False."""

    def test_disabled_normalization_preserves_values(self) -> None:
        """Disabling normalization should preserve feature values."""

        dataset = create_dataset()

        normalizer = ForecastingDatasetNormalizer(
            NormalizerConfig(
                normalize_features=False,
            )
        )

        normalizer.fit(dataset)

        transformed = normalizer.transform(
            dataset
        )

        assert np.array_equal(
            transformed.sequences,
            dataset.sequences,
        )

    def test_disabled_normalization_does_not_create_scaler(self) -> None:
        """No scaler should be created when normalization is disabled."""

        dataset = create_dataset()

        normalizer = ForecastingDatasetNormalizer(
            NormalizerConfig(
                normalize_features=False,
            )
        )

        normalizer.fit(dataset)

        assert normalizer.scaler is None
        assert normalizer.fitted is True

    def test_disabled_normalization_still_preserves_metadata(self) -> None:
        """Disabling normalization must still preserve the dataset contract."""

        dataset = create_dataset()

        normalizer = ForecastingDatasetNormalizer(
            NormalizerConfig(
                normalize_features=False,
            )
        )

        transformed = normalizer.fit_transform(
            dataset
        )

        assert transformed.feature_names == dataset.feature_names
        assert np.array_equal(
            transformed.targets,
            dataset.targets,
        )
        assert np.array_equal(
            transformed.timestamps,
            dataset.timestamps,
        )


# ============================================================================
# Configuration Integration Tests
# ============================================================================

class TestForecastingConfigIntegration:
    """Tests for integration with ForecastingConfig."""

    def test_from_forecasting_config_uses_standard_method(self) -> None:
        """ForecastingConfig should create a standard normalizer."""

        forecasting_config = ForecastingConfig()

        normalizer = (
            ForecastingDatasetNormalizer
            .from_forecasting_config(
                forecasting_config
            )
        )

        assert normalizer.method == "standard"

    def test_from_forecasting_config_preserves_training_only_setting(
        self,
    ) -> None:
        """ForecastingConfig training-only setting should be enforced."""

        forecasting_config = ForecastingConfig()

        normalizer = (
            ForecastingDatasetNormalizer
            .from_forecasting_config(
                forecasting_config
            )
        )

        assert normalizer.config.fit_on_training_only is True

    def test_from_forecasting_config_uses_feature_normalization_setting(
        self,
    ) -> None:
        """ForecastingConfig normalization setting should be respected."""

        forecasting_config = ForecastingConfig(
            normalize_features=False,
        )

        normalizer = (
            ForecastingDatasetNormalizer
            .from_forecasting_config(
                forecasting_config
            )
        )

        assert normalizer.config.normalize_features is False

    def test_invalid_forecasting_config_type_is_rejected(self) -> None:
        """Invalid configuration objects should be rejected."""

        with pytest.raises(
            TypeError,
            match="must be an instance of ForecastingConfig",
        ):
            ForecastingDatasetNormalizer.from_forecasting_config(
                object(),  # type: ignore[arg-type]
            )

    def test_normalize_from_forecasting_config_works(self) -> None:
        """Configuration-based convenience normalization should work."""

        training = create_dataset(
            num_samples=10,
        )

        validation = create_dataset(
            num_samples=5,
            start_value=100.0,
        )

        test = create_dataset(
            num_samples=5,
            start_value=200.0,
        )

        config = ForecastingConfig()

        (
            normalized_training,
            normalized_validation,
            normalized_test,
        ) = normalize_from_forecasting_config(
            training,
            validation,
            test,
            config,
        )

        assert (
            normalized_training.sequences.shape
            == training.sequences.shape
        )

        assert (
            normalized_validation.sequences.shape
            == validation.sequences.shape
        )

        assert (
            normalized_test.sequences.shape
            == test.sequences.shape
        )


# ============================================================================
# Invalid Input Tests
# ============================================================================

class TestInvalidInput:
    """Tests for invalid normalization inputs."""

    def test_invalid_training_dataset_type_is_rejected(self) -> None:
        """fit() should reject non-ForecastingDataset objects."""

        normalizer = ForecastingDatasetNormalizer()

        with pytest.raises(
            TypeError,
            match="training_dataset must be an instance",
        ):
            normalizer.fit(
                object(),  # type: ignore[arg-type]
            )

    def test_invalid_transform_dataset_type_is_rejected(self) -> None:
        """transform() should reject non-ForecastingDataset objects."""

        normalizer = ForecastingDatasetNormalizer()

        with pytest.raises(
            TypeError,
            match="dataset must be an instance",
        ):
            normalizer.transform(
                object(),  # type: ignore[arg-type]
            )

    def test_nonfinite_training_values_are_rejected(self) -> None:
        """Training data containing NaN should be rejected."""

        dataset = create_dataset()

        sequences = dataset.sequences.copy()
        sequences[0, 0, 0] = np.nan

        invalid = ForecastingDataset.__new__(
            ForecastingDataset
        )

        object.__setattr__(
            invalid,
            "sequences",
            sequences,
        )

        object.__setattr__(
            invalid,
            "targets",
            dataset.targets,
        )

        object.__setattr__(
            invalid,
            "timestamps",
            dataset.timestamps,
        )

        object.__setattr__(
            invalid,
            "feature_names",
            dataset.feature_names,
        )

        object.__setattr__(
            invalid,
            "context_window",
            dataset.context_window,
        )

        object.__setattr__(
            invalid,
            "forecast_horizon",
            dataset.forecast_horizon,
        )

        object.__setattr__(
            invalid,
            "target_name",
            dataset.target_name,
        )

        object.__setattr__(
            invalid,
            "price_column",
            dataset.price_column,
        )

        normalizer = ForecastingDatasetNormalizer()

        with pytest.raises(
            ValueError,
            match="must contain only finite values",
        ):
            normalizer.fit(invalid)

    def test_incompatible_validation_context_window_is_rejected(
        self,
    ) -> None:
        """Validation context-window mismatch should be rejected."""

        training = create_dataset()

        validation_sequences = np.ones(
            (
                5,
                CONTEXT_WINDOW + 1,
                NUM_FEATURES,
            ),
            dtype=np.float64,
        )

        validation = ForecastingDataset(
            sequences=validation_sequences,
            targets=np.arange(
                5,
                dtype=np.float64,
            ),
            timestamps=np.arange(
                200,
                205,
                dtype=np.float64,
            ),
            feature_names=FEATURE_NAMES,
            context_window=CONTEXT_WINDOW + 1,
            forecast_horizon=FORECAST_HORIZON,
            target_name="future_mid_price_return",
            price_column="mid_price",
        )

        normalizer = ForecastingDatasetNormalizer()

        with pytest.raises(
            ValueError,
            match="context window does not match",
        ):
            normalizer.transform_splits(
                training,
                validation,
                create_dataset(
                    num_samples=5,
                    start_value=200.0,
                ),
            )

    def test_incompatible_test_feature_count_is_rejected(
        self,
    ) -> None:
        """A mismatched test feature dimension should be rejected."""

        training = create_dataset()

        test_sequences = np.ones(
            (
                5,
                CONTEXT_WINDOW,
                NUM_FEATURES + 1,
            ),
            dtype=np.float64,
        )

        test_feature_names = FEATURE_NAMES + (
            "extra_feature",
        )

        test = ForecastingDataset(
            sequences=test_sequences,
            targets=np.arange(
                5,
                dtype=np.float64,
            ),
            timestamps=np.arange(
                200,
                205,
                dtype=np.float64,
            ),
            feature_names=test_feature_names,
            context_window=CONTEXT_WINDOW,
            forecast_horizon=FORECAST_HORIZON,
            target_name="future_mid_price_return",
            price_column="mid_price",
        )

        normalizer = ForecastingDatasetNormalizer()

        with pytest.raises(
            ValueError,
            match="feature names do not match",
        ):
            normalizer.transform_splits(
                training,
                create_dataset(
                    num_samples=5,
                    start_value=100.0,
                ),
                test,
            )


# ============================================================================
# Convenience Function Tests
# ============================================================================

class TestConvenienceFunctions:
    """Tests for public normalization convenience functions."""

    def test_fit_normalizer_returns_fitted_normalizer(self) -> None:
        """fit_normalizer should return a fitted normalizer."""

        dataset = create_dataset()

        normalizer = fit_normalizer(
            dataset
        )

        assert isinstance(
            normalizer,
            ForecastingDatasetNormalizer,
        )

        assert normalizer.fitted is True

    def test_normalize_forecasting_splits_returns_expected_shapes(
        self,
    ) -> None:
        """Convenience split normalization should preserve all shapes."""

        training = create_dataset(
            num_samples=8,
        )

        validation = create_dataset(
            num_samples=4,
            start_value=100.0,
        )

        test = create_dataset(
            num_samples=3,
            start_value=200.0,
        )

        (
            normalized_training,
            normalized_validation,
            normalized_test,
        ) = normalize_forecasting_splits(
            training,
            validation,
            test,
        )

        assert (
            normalized_training.sequences.shape
            == training.sequences.shape
        )

        assert (
            normalized_validation.sequences.shape
            == validation.sequences.shape
        )

        assert (
            normalized_test.sequences.shape
            == test.sequences.shape
        )


# ============================================================================
# Minimal Dataset Tests
# ============================================================================

class TestMinimalDataset:
    """Tests for the smallest valid forecasting dataset."""

    def test_single_sample_dataset_can_be_normalized(self) -> None:
        """A single valid sample should be normalizable."""

        dataset = create_single_sample_dataset()

        normalizer = ForecastingDatasetNormalizer()

        transformed = normalizer.fit_transform(
            dataset
        )

        assert transformed.sequences.shape == (
            1,
            CONTEXT_WINDOW,
            NUM_FEATURES,
        )

        assert np.all(
            np.isfinite(
                transformed.sequences
            )
        )

    def test_single_sample_dataset_normalizes_to_zero(self) -> None:
        """
        A completely constant single-sample dataset should normalize to zero.
        """

        dataset = create_single_sample_dataset()

        normalizer = ForecastingDatasetNormalizer()

        transformed = normalizer.fit_transform(
            dataset
        )

        assert np.allclose(
            transformed.sequences,
            0.0,
        )


# ============================================================================
# Public Export Tests
# ============================================================================

class TestStandardScaler:
    """Tests for the standalone StandardScaler contract."""

    def test_scaler_reports_feature_count(self) -> None:
        """StandardScaler should report its number of features."""

        dataset = create_dataset()

        normalizer = ForecastingDatasetNormalizer()

        normalizer.fit(dataset)

        assert normalizer.scaler is not None

        assert (
            normalizer.scaler.num_features
            == NUM_FEATURES
        )

    def test_scaler_reports_fitted_state(self) -> None:
        """A fitted StandardScaler should report fitted=True."""

        dataset = create_dataset()

        normalizer = ForecastingDatasetNormalizer()

        normalizer.fit(dataset)

        assert normalizer.scaler is not None

        assert normalizer.scaler.fitted is True

