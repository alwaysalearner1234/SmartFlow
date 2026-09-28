"""
tests/test_normalizer.py

Comprehensive tests for forecasting feature normalization.

This test module verifies:

    - NormalizerConfig validation
    - StandardScaler construction
    - StandardScaler transformation
    - training-only scaler fitting
    - feature-wise mean/std calculation
    - constant-feature handling
    - feature schema protection
    - dataset transformation
    - train/validation/test transformation
    - target preservation
    - timestamp preservation
    - metadata preservation
    - model input shape preservation
    - dtype behavior
    - copy behavior
    - disabled normalization
    - ForecastingConfig integration
    - convenience functions
    - leakage protection
    - deterministic behavior
    - invalid input handling
    - NVIDIA forecasting input compatibility

Normalization contract:

    Training sequences
        |
        v
    Fit StandardScaler
        |
        +----> training statistics ONLY
        |
        v
    Transform train
    Transform validation
    Transform test

The scaler must never be fitted using validation or test data.

Current NVIDIA forecasting contract:

    context_window = 20
    num_features = 14
    forecast_horizon = 5

Therefore model inputs have shape:

    (N, 20, 14)

and targets have shape:

    (N,)

This module does not test:

    - feature extraction
    - sequence construction
    - target construction
    - sequence/target alignment
    - chronological splitting
    - model architecture
    - model training
    - NVIDIA inference
"""

from __future__ import annotations

import numpy as np
import pytest

from config.config import ForecastingConfig
from data.forecasting_dataset import ForecastingDataset
from data.normalizer import (
    DEFAULT_DTYPE,
    DEFAULT_NORMALIZATION_METHOD,
    SUPPORTED_NORMALIZATION_METHODS,
    ForecastingDatasetNormalizer,
    NormalizedForecastingDataset,
    NormalizerConfig,
    StandardScaler,
    fit_normalizer,
    normalize_forecasting_splits,
    normalize_from_forecasting_config,
)


# ---------------------------------------------------------------------------
# Test constants
# ---------------------------------------------------------------------------

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
NUM_FEATURES = 14

TARGET_NAME = "future_mid_price_return"
PRICE_COLUMN = "mid_price"


# ---------------------------------------------------------------------------
# Dataset factories
# ---------------------------------------------------------------------------


def make_dataset(
    *,
    num_samples: int = 10,
    context_window: int = CONTEXT_WINDOW,
    num_features: int = NUM_FEATURES,
    feature_names: tuple[str, ...] = FEATURE_NAMES,
    sequence_offset: float = 0.0,
    sequence_scale: float = 1.0,
    target_offset: float = 0.0,
    timestamp_start: float = 1000.0,
) -> ForecastingDataset:
    """
    Create a deterministic ForecastingDataset for testing.

    Feature values are generated deterministically so expected means,
    standard deviations, and transformed values can be calculated exactly.
    """

    sequences = np.arange(
        num_samples
        * context_window
        * num_features,
        dtype=np.float64,
    ).reshape(
        num_samples,
        context_window,
        num_features,
    )

    sequences = (
        sequences * sequence_scale
        + sequence_offset
    )

    targets = (
        np.arange(
            num_samples,
            dtype=np.float64,
        )
        + target_offset
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
        feature_names=feature_names,
        context_window=context_window,
        forecast_horizon=FORECAST_HORIZON,
        target_name=TARGET_NAME,
        price_column=PRICE_COLUMN,
    )


def make_constant_feature_dataset(
    *,
    num_samples: int = 10,
    context_window: int = CONTEXT_WINDOW,
    constant_value: float = 7.0,
) -> ForecastingDataset:
    """
    Create a dataset where every feature has constant values.
    """

    sequences = np.full(
        (
            num_samples,
            context_window,
            NUM_FEATURES,
        ),
        constant_value,
        dtype=np.float64,
    )

    targets = np.arange(
        num_samples,
        dtype=np.float64,
    )

    timestamps = (
        1000.0
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
        context_window=context_window,
        forecast_horizon=FORECAST_HORIZON,
        target_name=TARGET_NAME,
        price_column=PRICE_COLUMN,
    )


def make_custom_feature_dataset(
    *,
    num_samples: int = 5,
    context_window: int = 3,
    feature_names: tuple[str, ...] = (
        "feature_a",
        "feature_b",
    ),
    offset: float = 0.0,
) -> ForecastingDataset:
    """
    Create a small dataset with a custom feature schema.
    """

    num_features = len(feature_names)

    sequences = np.arange(
        num_samples
        * context_window
        * num_features,
        dtype=np.float64,
    ).reshape(
        num_samples,
        context_window,
        num_features,
    )

    sequences = sequences + offset

    targets = np.arange(
        num_samples,
        dtype=np.float64,
    )

    timestamps = (
        500.0
        + np.arange(
            num_samples,
            dtype=np.float64,
        )
    )

    return ForecastingDataset(
        sequences=sequences,
        targets=targets,
        timestamps=timestamps,
        feature_names=feature_names,
        context_window=context_window,
        forecast_horizon=FORECAST_HORIZON,
        target_name=TARGET_NAME,
        price_column=PRICE_COLUMN,
    )


# ---------------------------------------------------------------------------
# Configuration tests
# ---------------------------------------------------------------------------


class TestNormalizerConfig:
    """Tests for NormalizerConfig."""

    def test_default_method(self):
        """Default normalization method is standard."""

        config = NormalizerConfig()

        assert config.method == DEFAULT_NORMALIZATION_METHOD

    def test_default_normalization_enabled(self):
        """Feature normalization is enabled by default."""

        config = NormalizerConfig()

        assert config.normalize_features is True

    def test_default_training_only_behavior(self):
        """Training-only fitting is enabled by default."""

        config = NormalizerConfig()

        assert config.fit_on_training_only is True

    def test_default_dtype(self):
        """Default output dtype is float64."""

        config = NormalizerConfig()

        assert config.dtype == DEFAULT_DTYPE

    def test_default_copy_behavior(self):
        """Arrays are copied by default."""

        config = NormalizerConfig()

        assert config.copy_arrays is True

    def test_supported_methods(self):
        """The supported normalization methods are exposed."""

        assert "standard" in SUPPORTED_NORMALIZATION_METHODS

    def test_method_is_case_normalized_by_normalizer(self):
        """Uppercase method names can be accepted."""

        config = NormalizerConfig(
            method="STANDARD",
        )

        normalizer = ForecastingDatasetNormalizer(
            config=config,
        )

        assert normalizer.method == "standard"

    def test_invalid_method_raises(self):
        """Unsupported normalization methods are rejected."""

        with pytest.raises(ValueError):
            NormalizerConfig(
                method="minmax",
            )

    def test_non_string_method_raises(self):
        """Normalization method must be a string."""

        with pytest.raises(TypeError):
            NormalizerConfig(
                method=123,
            )

    def test_invalid_dtype_raises(self):
        """Invalid NumPy dtypes are rejected."""

        with pytest.raises(ValueError):
            NormalizerConfig(
                dtype="not-a-real-dtype",
            )

    def test_empty_dtype_raises(self):
        """Empty dtype strings are rejected."""

        with pytest.raises(ValueError):
            NormalizerConfig(
                dtype="",
            )

    def test_normalize_features_must_be_boolean(self):
        """normalize_features must be a boolean."""

        with pytest.raises(TypeError):
            NormalizerConfig(
                normalize_features=1,
            )

    def test_fit_on_training_only_must_be_boolean(self):
        """fit_on_training_only must be a boolean."""

        with pytest.raises(TypeError):
            NormalizerConfig(
                fit_on_training_only=1,
            )

    def test_copy_arrays_must_be_boolean(self):
        """copy_arrays must be a boolean."""

        with pytest.raises(TypeError):
            NormalizerConfig(
                copy_arrays=1,
            )


# ---------------------------------------------------------------------------
# StandardScaler construction
# ---------------------------------------------------------------------------


class TestStandardScaler:
    """Tests for the fitted StandardScaler contract."""

    def test_valid_scaler_constructs(self):
        """A valid scaler can be constructed."""

        scaler = StandardScaler(
            means=np.array([1.0, 2.0]),
            scales=np.array([0.5, 2.0]),
            feature_names=("a", "b"),
        )

        assert scaler.fitted is True

    def test_num_features(self):
        """num_features reports the fitted feature count."""

        scaler = StandardScaler(
            means=np.array([1.0, 2.0, 3.0]),
            scales=np.array([1.0, 1.0, 1.0]),
            feature_names=("a", "b", "c"),
        )

        assert scaler.num_features == 3

    def test_means_are_stored(self):
        """Means are preserved."""

        means = np.array([1.0, 2.0])

        scaler = StandardScaler(
            means=means,
            scales=np.array([1.0, 2.0]),
            feature_names=("a", "b"),
        )

        np.testing.assert_array_equal(
            scaler.means,
            means,
        )

    def test_scales_are_stored(self):
        """Scales are preserved."""

        scales = np.array([1.0, 2.0])

        scaler = StandardScaler(
            means=np.array([1.0, 2.0]),
            scales=scales,
            feature_names=("a", "b"),
        )

        np.testing.assert_array_equal(
            scaler.scales,
            scales,
        )

    def test_feature_names_are_stored_as_tuple(self):
        """Feature names are normalized to an immutable tuple."""

        scaler = StandardScaler(
            means=np.array([1.0, 2.0]),
            scales=np.array([1.0, 2.0]),
            feature_names=["a", "b"],
        )

        assert isinstance(
            scaler.feature_names,
            tuple,
        )

    def test_mismatched_means_and_scales_raise(self):
        """Means and scales must have matching dimensions."""

        with pytest.raises(ValueError):
            StandardScaler(
                means=np.array([1.0, 2.0]),
                scales=np.array([1.0]),
                feature_names=("a", "b"),
            )

    def test_empty_scaler_raises(self):
        """A scaler must contain at least one feature."""

        with pytest.raises(ValueError):
            StandardScaler(
                means=np.array([]),
                scales=np.array([]),
                feature_names=(),
            )

    def test_duplicate_feature_names_raise(self):
        """Scaler feature names must be unique."""

        with pytest.raises(ValueError):
            StandardScaler(
                means=np.array([1.0, 2.0]),
                scales=np.array([1.0, 1.0]),
                feature_names=("a", "a"),
            )

    def test_empty_feature_name_raises(self):
        """Scaler feature names must be non-empty."""

        with pytest.raises(ValueError):
            StandardScaler(
                means=np.array([1.0]),
                scales=np.array([1.0]),
                feature_names=("",),
            )

    def test_nonfinite_mean_raises(self):
        """Scaler means must be finite."""

        with pytest.raises(ValueError):
            StandardScaler(
                means=np.array([np.nan]),
                scales=np.array([1.0]),
                feature_names=("a",),
            )

    def test_infinite_mean_raises(self):
        """Infinite means are rejected."""

        with pytest.raises(ValueError):
            StandardScaler(
                means=np.array([np.inf]),
                scales=np.array([1.0]),
                feature_names=("a",),
            )

    def test_nonfinite_scale_raises(self):
        """Scaler scales must be finite."""

        with pytest.raises(ValueError):
            StandardScaler(
                means=np.array([1.0]),
                scales=np.array([np.nan]),
                feature_names=("a",),
            )

    def test_zero_scale_raises(self):
        """Directly constructed zero scales are rejected."""

        with pytest.raises(ValueError):
            StandardScaler(
                means=np.array([1.0]),
                scales=np.array([0.0]),
                feature_names=("a",),
            )

    def test_negative_scale_raises(self):
        """Negative scales are rejected."""

        with pytest.raises(ValueError):
            StandardScaler(
                means=np.array([1.0]),
                scales=np.array([-1.0]),
                feature_names=("a",),
            )


# ---------------------------------------------------------------------------
# StandardScaler transformation
# ---------------------------------------------------------------------------


class TestStandardScalerTransform:
    """Tests for direct StandardScaler transformation."""

    def test_standard_transformation(self):
        """Values are transformed using (x - mean) / scale."""

        scaler = StandardScaler(
            means=np.array([10.0, 20.0]),
            scales=np.array([2.0, 5.0]),
            feature_names=("a", "b"),
        )

        sequences = np.array(
            [
                [
                    [12.0, 25.0],
                    [8.0, 15.0],
                ]
            ],
            dtype=np.float64,
        )

        transformed = scaler.transform(
            sequences,
        )

        expected = np.array(
            [
                [
                    [1.0, 1.0],
                    [-1.0, -1.0],
                ]
            ],
            dtype=np.float64,
        )

        np.testing.assert_allclose(
            transformed,
            expected,
        )

    def test_shape_is_preserved(self):
        """Transformation does not change tensor dimensions."""

        scaler = StandardScaler(
            means=np.zeros(2),
            scales=np.ones(2),
            feature_names=("a", "b"),
        )

        sequences = np.ones(
            (4, 3, 2),
            dtype=np.float64,
        )

        transformed = scaler.transform(
            sequences,
        )

        assert transformed.shape == sequences.shape

    def test_zero_mean_unit_scale_is_identity(self):
        """Zero means and unit scales leave values unchanged."""

        scaler = StandardScaler(
            means=np.zeros(2),
            scales=np.ones(2),
            feature_names=("a", "b"),
        )

        sequences = np.array(
            [
                [
                    [1.0, 2.0],
                    [3.0, 4.0],
                ]
            ],
            dtype=np.float64,
        )

        transformed = scaler.transform(
            sequences,
        )

        np.testing.assert_array_equal(
            transformed,
            sequences,
        )

    def test_nan_values_are_rejected(self):
        """NaN feature values cannot be transformed."""

        scaler = StandardScaler(
            means=np.zeros(1),
            scales=np.ones(1),
            feature_names=("a",),
        )

        sequences = np.array(
            [[[np.nan]]],
            dtype=np.float64,
        )

        with pytest.raises(ValueError):
            scaler.transform(sequences)

    def test_infinite_values_are_rejected(self):
        """Infinite feature values cannot be transformed."""

        scaler = StandardScaler(
            means=np.zeros(1),
            scales=np.ones(1),
            feature_names=("a",),
        )

        sequences = np.array(
            [[[np.inf]]],
            dtype=np.float64,
        )

        with pytest.raises(ValueError):
            scaler.transform(sequences)

    def test_wrong_feature_count_raises(self):
        """Input feature count must match fitted scaler."""

        scaler = StandardScaler(
            means=np.zeros(2),
            scales=np.ones(2),
            feature_names=("a", "b"),
        )

        sequences = np.ones(
            (2, 3, 3),
            dtype=np.float64,
        )

        with pytest.raises(ValueError):
            scaler.transform(sequences)

    def test_two_dimensional_input_raises(self):
        """Scaler expects 3D sequence arrays."""

        scaler = StandardScaler(
            means=np.zeros(2),
            scales=np.ones(2),
            feature_names=("a", "b"),
        )

        sequences = np.ones(
            (3, 2),
            dtype=np.float64,
        )

        with pytest.raises(ValueError):
            scaler.transform(sequences)

    def test_float32_output(self):
        """Output dtype can be configured as float32."""

        scaler = StandardScaler(
            means=np.zeros(2),
            scales=np.ones(2),
            feature_names=("a", "b"),
        )

        sequences = np.ones(
            (2, 3, 2),
            dtype=np.float64,
        )

        transformed = scaler.transform(
            sequences,
            dtype="float32",
        )

        assert transformed.dtype == np.float32

    def test_copy_true_produces_independent_array(self):
        """copy=True returns independent storage."""

        scaler = StandardScaler(
            means=np.zeros(1),
            scales=np.ones(1),
            feature_names=("a",),
        )

        sequences = np.ones(
            (2, 2, 1),
            dtype=np.float64,
        )

        transformed = scaler.transform(
            sequences,
            copy=True,
        )

        sequences[0, 0, 0] = 999.0

        assert transformed[0, 0, 0] != 999.0


# ---------------------------------------------------------------------------
# Normalizer initialization
# ---------------------------------------------------------------------------


class TestForecastingDatasetNormalizerInitialization:
    """Tests for ForecastingDatasetNormalizer initialization."""

    def test_default_initialization(self):
        """Normalizer can be initialized with defaults."""

        normalizer = ForecastingDatasetNormalizer()

        assert normalizer.fitted is False
        assert normalizer.scaler is None
        assert normalizer.feature_names is None

    def test_custom_config_is_stored(self):
        """Custom configuration is retained."""

        config = NormalizerConfig(
            dtype="float32",
        )

        normalizer = ForecastingDatasetNormalizer(
            config=config,
        )

        assert normalizer.config is config

    def test_method_property(self):
        """method exposes normalized method name."""

        normalizer = ForecastingDatasetNormalizer()

        assert normalizer.method == "standard"

    def test_normalization_enabled_property(self):
        """normalization_enabled reflects configuration."""

        enabled = ForecastingDatasetNormalizer()

        disabled = ForecastingDatasetNormalizer(
            NormalizerConfig(
                normalize_features=False,
            )
        )

        assert enabled.normalization_enabled is True
        assert disabled.normalization_enabled is False


# ---------------------------------------------------------------------------
# Fitting
# ---------------------------------------------------------------------------


class TestNormalizerFit:
    """Tests for training-only scaler fitting."""

    def test_fit_marks_normalizer_as_fitted(self):
        """Successful fit sets fitted=True."""

        dataset = make_dataset()

        normalizer = ForecastingDatasetNormalizer()

        normalizer.fit(dataset)

        assert normalizer.fitted is True

    def test_fit_creates_scaler(self):
        """Standard normalization creates a fitted scaler."""

        dataset = make_dataset()

        normalizer = ForecastingDatasetNormalizer()

        normalizer.fit(dataset)

        assert normalizer.scaler is not None
        assert normalizer.scaler.fitted is True

    def test_fit_stores_feature_names(self):
        """Training feature schema is stored."""

        dataset = make_dataset()

        normalizer = ForecastingDatasetNormalizer()

        normalizer.fit(dataset)

        assert normalizer.feature_names == FEATURE_NAMES

    def test_fit_calculates_per_feature_means(self):
        """Means are calculated independently for each feature."""

        dataset = make_dataset(
            num_samples=2,
            context_window=2,
            num_features=2,
            feature_names=(
                "feature_a",
                "feature_b",
            ),
        )

        normalizer = ForecastingDatasetNormalizer()

        normalizer.fit(dataset)

        flattened = dataset.sequences.reshape(
            -1,
            2,
        )

        expected_means = np.mean(
            flattened,
            axis=0,
        )

        np.testing.assert_allclose(
            normalizer.scaler.means,
            expected_means,
        )

    def test_fit_calculates_per_feature_scales(self):
        """Standard deviations are calculated independently."""

        dataset = make_dataset(
            num_samples=2,
            context_window=2,
            num_features=2,
            feature_names=(
                "feature_a",
                "feature_b",
            ),
        )

        normalizer = ForecastingDatasetNormalizer()

        normalizer.fit(dataset)

        flattened = dataset.sequences.reshape(
            -1,
            2,
        )

        expected_scales = np.std(
            flattened,
            axis=0,
        )

        np.testing.assert_allclose(
            normalizer.scaler.scales,
            expected_scales,
        )

    def test_scaler_uses_all_context_timesteps(self):
        """
        Statistics are calculated over all samples and all context
        timesteps, not just the final timestep.
        """

        dataset = make_dataset(
            num_samples=2,
            context_window=3,
            num_features=1,
            feature_names=("feature_a",),
        )

        normalizer = ForecastingDatasetNormalizer()

        normalizer.fit(dataset)

        flattened = dataset.sequences.reshape(
            -1,
            1,
        )

        expected_mean = np.mean(
            flattened[:, 0]
        )

        assert normalizer.scaler.means[0] == pytest.approx(
            expected_mean
        )

    def test_fit_is_deterministic(self):
        """Repeated fitting on identical data produces identical statistics."""

        dataset = make_dataset()

        first = ForecastingDatasetNormalizer()
        second = ForecastingDatasetNormalizer()

        first.fit(dataset)
        second.fit(dataset)

        np.testing.assert_array_equal(
            first.scaler.means,
            second.scaler.means,
        )

        np.testing.assert_array_equal(
            first.scaler.scales,
            second.scaler.scales,
        )

    def test_constant_features_receive_unit_scale(self):
        """
        Constant features receive scale=1 instead of zero so transformation
        remains numerically valid.
        """

        dataset = make_constant_feature_dataset()

        normalizer = ForecastingDatasetNormalizer()

        normalizer.fit(dataset)

        np.testing.assert_array_equal(
            normalizer.scaler.scales,
            np.ones(NUM_FEATURES),
        )

    def test_constant_features_transform_to_zero(self):
        """Constant training features become zero after centering."""

        dataset = make_constant_feature_dataset(
            constant_value=7.0,
        )

        normalizer = ForecastingDatasetNormalizer()

        normalizer.fit(dataset)

        transformed = normalizer.transform(
            dataset,
        )

        np.testing.assert_allclose(
            transformed.sequences,
            0.0,
        )

    def test_fit_rejects_nan_training_values(self):
        """NaN values cannot be used to fit normalization."""

        dataset = make_dataset()

        dataset.sequences[0, 0, 0] = np.nan

        normalizer = ForecastingDatasetNormalizer()

        with pytest.raises(ValueError):
            normalizer.fit(dataset)

    def test_fit_rejects_infinite_training_values(self):
        """Infinite values cannot be used to fit normalization."""

        dataset = make_dataset()

        dataset.sequences[0, 0, 0] = np.inf

        normalizer = ForecastingDatasetNormalizer()

        with pytest.raises(ValueError):
            normalizer.fit(dataset)

    def test_fit_rejects_non_dataset_input(self):
        """fit() requires a ForecastingDataset."""

        normalizer = ForecastingDatasetNormalizer()

        with pytest.raises(TypeError):
            normalizer.fit("not-a-dataset")


# ---------------------------------------------------------------------------
# Training-only leakage protection
# ---------------------------------------------------------------------------


class TestTrainingOnlyLeakageProtection:
    """Tests that normalization statistics come exclusively from training."""

    def test_validation_data_does_not_affect_fitted_statistics(self):
        """
        Changing validation data must not change scaler statistics when
        fitting directly on the training dataset.
        """

        training = make_dataset(
            sequence_offset=0.0,
        )

        validation_a = make_dataset(
            sequence_offset=1000.0,
        )

        validation_b = make_dataset(
            sequence_offset=999999.0,
        )

        normalizer_a = ForecastingDatasetNormalizer()
        normalizer_b = ForecastingDatasetNormalizer()

        normalizer_a.fit(training)
        normalizer_b.fit(training)

        np.testing.assert_array_equal(
            normalizer_a.scaler.means,
            normalizer_b.scaler.means,
        )

        np.testing.assert_array_equal(
            normalizer_a.scaler.scales,
            normalizer_b.scaler.scales,
        )

        assert validation_a.sequences.mean() != (
            validation_b.sequences.mean()
        )

    def test_test_data_does_not_affect_fitted_statistics(self):
        """
        Test distribution changes cannot affect training-only scaler
        statistics.
        """

        training = make_dataset(
            sequence_offset=0.0,
        )

        test_a = make_dataset(
            sequence_offset=100.0,
        )

        test_b = make_dataset(
            sequence_offset=-10000.0,
        )

        normalizer = ForecastingDatasetNormalizer()

        normalizer.fit(training)

        original_means = normalizer.scaler.means.copy()
        original_scales = normalizer.scaler.scales.copy()

        normalizer.transform(test_a)
        normalizer.transform(test_b)

        np.testing.assert_array_equal(
            normalizer.scaler.means,
            original_means,
        )

        np.testing.assert_array_equal(
            normalizer.scaler.scales,
            original_scales,
        )

    def test_transform_does_not_refit_scaler(self):
        """Calling transform never changes fitted statistics."""

        training = make_dataset()

        validation = make_dataset(
            sequence_offset=100000.0,
        )

        normalizer = ForecastingDatasetNormalizer()

        normalizer.fit(training)

        means_before = normalizer.scaler.means.copy()
        scales_before = normalizer.scaler.scales.copy()

        normalizer.transform(validation)

        np.testing.assert_array_equal(
            normalizer.scaler.means,
            means_before,
        )

        np.testing.assert_array_equal(
            normalizer.scaler.scales,
            scales_before,
        )

    def test_disabling_training_only_fit_is_rejected(self):
        """
        Leakage-safe forecasting normalization requires
        fit_on_training_only=True.
        """

        config = NormalizerConfig(
            fit_on_training_only=False,
        )

        normalizer = ForecastingDatasetNormalizer(
            config=config,
        )

        training = make_dataset()

        with pytest.raises(ValueError):
            normalizer.fit(training)


# ---------------------------------------------------------------------------
# Transformation
# ---------------------------------------------------------------------------


class TestNormalizerTransform:
    """Tests for ForecastingDataset transformation."""

    def test_transform_requires_fit(self):
        """Transforming before fit is rejected."""

        dataset = make_dataset()

        normalizer = ForecastingDatasetNormalizer()

        with pytest.raises(RuntimeError):
            normalizer.transform(dataset)

    def test_transform_preserves_shape(self):
        """Normalization does not alter input tensor shape."""

        dataset = make_dataset()

        normalizer = ForecastingDatasetNormalizer()
        normalizer.fit(dataset)

        transformed = normalizer.transform(
            dataset,
        )

        assert transformed.sequences.shape == (
            dataset.sequences.shape
        )

    def test_transform_preserves_sample_count(self):
        """Normalization does not add or remove samples."""

        dataset = make_dataset()

        normalizer = ForecastingDatasetNormalizer()
        normalizer.fit(dataset)

        transformed = normalizer.transform(
            dataset,
        )

        assert transformed.num_samples == dataset.num_samples

    def test_transform_produces_zero_training_means(self):
        """
        Training features should have approximately zero mean after
        standard normalization.
        """

        dataset = make_dataset(
            num_samples=20,
        )

        normalizer = ForecastingDatasetNormalizer()
        normalizer.fit(dataset)

        transformed = normalizer.transform(
            dataset,
        )

        flattened = transformed.sequences.reshape(
            -1,
            transformed.num_features,
        )

        means = flattened.mean(
            axis=0,
        )

        np.testing.assert_allclose(
            means,
            np.zeros(transformed.num_features),
            atol=1e-12,
        )

    def test_transform_produces_unit_training_scales(self):
        """
        Non-constant training features should have unit standard deviation.
        """

        dataset = make_dataset(
            num_samples=20,
        )

        normalizer = ForecastingDatasetNormalizer()
        normalizer.fit(dataset)

        transformed = normalizer.transform(
            dataset,
        )

        flattened = transformed.sequences.reshape(
            -1,
            transformed.num_features,
        )

        scales = flattened.std(
            axis=0,
        )

        np.testing.assert_allclose(
            scales,
            np.ones(transformed.num_features),
            atol=1e-12,
        )

    def test_transform_preserves_targets(self):
        """Normalization does not modify targets."""

        dataset = make_dataset()

        normalizer = ForecastingDatasetNormalizer()
        normalizer.fit(dataset)

        transformed = normalizer.transform(
            dataset,
        )

        np.testing.assert_array_equal(
            transformed.targets,
            dataset.targets,
        )

    def test_transform_preserves_timestamps(self):
        """Normalization does not modify timestamps."""

        dataset = make_dataset()

        normalizer = ForecastingDatasetNormalizer()
        normalizer.fit(dataset)

        transformed = normalizer.transform(
            dataset,
        )

        np.testing.assert_array_equal(
            transformed.timestamps,
            dataset.timestamps,
        )

    def test_transform_preserves_feature_names(self):
        """Normalization does not reorder features."""

        dataset = make_dataset()

        normalizer = ForecastingDatasetNormalizer()
        normalizer.fit(dataset)

        transformed = normalizer.transform(
            dataset,
        )

        assert transformed.feature_names == (
            dataset.feature_names
        )

    def test_transform_preserves_context_window(self):
        """Normalization preserves context window metadata."""

        dataset = make_dataset()

        normalizer = ForecastingDatasetNormalizer()
        normalizer.fit(dataset)

        transformed = normalizer.transform(
            dataset,
        )

        assert transformed.context_window == (
            dataset.context_window
        )

    def test_transform_preserves_forecast_horizon(self):
        """Normalization preserves forecast horizon metadata."""

        dataset = make_dataset()

        normalizer = ForecastingDatasetNormalizer()
        normalizer.fit(dataset)

        transformed = normalizer.transform(
            dataset,
        )

        assert transformed.forecast_horizon == (
            dataset.forecast_horizon
        )

    def test_transform_preserves_target_name(self):
        """Normalization preserves target metadata."""

        dataset = make_dataset()

        normalizer = ForecastingDatasetNormalizer()
        normalizer.fit(dataset)

        transformed = normalizer.transform(
            dataset,
        )

        assert transformed.target_name == (
            dataset.target_name
        )

    def test_transform_preserves_price_column(self):
        """Normalization preserves price column metadata."""

        dataset = make_dataset()

        normalizer = ForecastingDatasetNormalizer()
        normalizer.fit(dataset)

        transformed = normalizer.transform(
            dataset,
        )

        assert transformed.price_column == (
            dataset.price_column
        )


# ---------------------------------------------------------------------------
# Feature schema protection
# ---------------------------------------------------------------------------


class TestFeatureSchemaProtection:
    """Tests preventing feature-order/schema mismatches."""

    def test_wrong_feature_names_raise(self):
        """A dataset with different feature names is rejected."""

        training = make_dataset()

        validation = make_custom_feature_dataset(
            feature_names=(
                "wrong_a",
                "wrong_b",
            ),
        )

        normalizer = ForecastingDatasetNormalizer()
        normalizer.fit(training)

        with pytest.raises(ValueError):
            normalizer.transform(validation)

    def test_same_feature_count_wrong_order_raises(self):
        """Identical feature counts with different ordering are rejected."""

        training = make_custom_feature_dataset(
            feature_names=(
                "feature_a",
                "feature_b",
            ),
        )

        validation = make_custom_feature_dataset(
            feature_names=(
                "feature_b",
                "feature_a",
            ),
        )

        normalizer = ForecastingDatasetNormalizer()
        normalizer.fit(training)

        with pytest.raises(ValueError):
            normalizer.transform(validation)

    def test_feature_count_mismatch_raises(self):
        """Different feature counts are rejected."""

        training = make_custom_feature_dataset(
            feature_names=(
                "a",
                "b",
            ),
        )

        validation = make_custom_feature_dataset(
            feature_names=(
                "a",
                "b",
                "c",
            ),
        )

        normalizer = ForecastingDatasetNormalizer()
        normalizer.fit(training)

        with pytest.raises(ValueError):
            normalizer.transform(validation)

    def test_training_schema_is_immutable_reference(self):
        """
        The normalizer stores the training feature schema as a tuple so
        external list mutation cannot silently change it.
        """

        feature_names = [
            "feature_a",
            "feature_b",
        ]

        dataset = make_custom_feature_dataset(
            feature_names=tuple(feature_names),
        )

        normalizer = ForecastingDatasetNormalizer()
        normalizer.fit(dataset)

        feature_names[0] = "changed"

        assert normalizer.feature_names == (
            "feature_a",
            "feature_b",
        )


# ---------------------------------------------------------------------------
# Split transformation
# ---------------------------------------------------------------------------


class TestTransformSplits:
    """Tests for train/validation/test normalization."""

    def test_transform_splits_returns_three_datasets(self):
        """transform_splits returns train, validation, and test."""

        training = make_dataset(
            num_samples=20,
        )

        validation = make_dataset(
            num_samples=10,
            sequence_offset=100.0,
            timestamp_start=2000.0,
        )

        test = make_dataset(
            num_samples=10,
            sequence_offset=200.0,
            timestamp_start=3000.0,
        )

        normalizer = ForecastingDatasetNormalizer()

        result = normalizer.transform_splits(
            training,
            validation,
            test,
        )

        assert len(result) == 3

        normalized_train, normalized_validation, normalized_test = result

        assert isinstance(
            normalized_train,
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

    def test_transform_splits_fits_on_training_only(self):
        """Validation and test distributions do not alter scaler statistics."""

        training = make_dataset(
            sequence_offset=0.0,
        )

        validation = make_dataset(
            sequence_offset=100000.0,
            timestamp_start=2000.0,
        )

        test = make_dataset(
            sequence_offset=-100000.0,
            timestamp_start=3000.0,
        )

        normalizer = ForecastingDatasetNormalizer()

        normalizer.transform_splits(
            training,
            validation,
            test,
        )

        flattened_training = training.sequences.reshape(
            -1,
            training.num_features,
        )

        expected_means = flattened_training.mean(
            axis=0,
        )

        expected_scales = flattened_training.std(
            axis=0,
        )

        np.testing.assert_allclose(
            normalizer.scaler.means,
            expected_means,
        )

        np.testing.assert_allclose(
            normalizer.scaler.scales,
            expected_scales,
        )

    def test_transform_splits_preserves_sample_counts(self):
        """Each split keeps its original number of samples."""

        training = make_dataset(
            num_samples=20,
        )

        validation = make_dataset(
            num_samples=10,
            timestamp_start=2000.0,
        )

        test = make_dataset(
            num_samples=8,
            timestamp_start=3000.0,
        )

        normalizer = ForecastingDatasetNormalizer()

        normalized = normalizer.transform_splits(
            training,
            validation,
            test,
        )

        assert normalized[0].num_samples == 20
        assert normalized[1].num_samples == 10
        assert normalized[2].num_samples == 8

    def test_transform_splits_preserves_targets(self):
        """Targets remain unchanged across all splits."""

        training = make_dataset(
            num_samples=20,
            target_offset=10.0,
        )

        validation = make_dataset(
            num_samples=10,
            target_offset=20.0,
            timestamp_start=2000.0,
        )

        test = make_dataset(
            num_samples=8,
            target_offset=30.0,
            timestamp_start=3000.0,
        )

        original_targets = (
            training.targets.copy(),
            validation.targets.copy(),
            test.targets.copy(),
        )

        normalizer = ForecastingDatasetNormalizer()

        normalized = normalizer.transform_splits(
            training,
            validation,
            test,
        )

        for result, expected in zip(
            normalized,
            original_targets,
        ):
            np.testing.assert_array_equal(
                result.targets,
                expected,
            )

    def test_transform_splits_preserves_timestamps(self):
        """Timestamps remain unchanged across all splits."""

        training = make_dataset(
            num_samples=20,
        )

        validation = make_dataset(
            num_samples=10,
            timestamp_start=2000.0,
        )

        test = make_dataset(
            num_samples=8,
            timestamp_start=3000.0,
        )

        original_timestamps = (
            training.timestamps.copy(),
            validation.timestamps.copy(),
            test.timestamps.copy(),
        )

        normalizer = ForecastingDatasetNormalizer()

        normalized = normalizer.transform_splits(
            training,
            validation,
            test,
        )

        for result, expected in zip(
            normalized,
            original_timestamps,
        ):
            np.testing.assert_array_equal(
                result.timestamps,
                expected,
            )

    def test_transform_splits_preserves_model_shape(self):
        """All normalized splits remain model-compatible."""

        training = make_dataset(
            num_samples=20,
        )

        validation = make_dataset(
            num_samples=10,
            timestamp_start=2000.0,
        )

        test = make_dataset(
            num_samples=8,
            timestamp_start=3000.0,
        )

        normalizer = ForecastingDatasetNormalizer()

        normalized = normalizer.transform_splits(
            training,
            validation,
            test,
        )

        for dataset in normalized:
            assert dataset.sequences.ndim == 3
            assert dataset.sequences.shape[1:] == (
                20,
                14,
            )
            assert dataset.targets.ndim == 1


# ---------------------------------------------------------------------------
# fit_transform
# ---------------------------------------------------------------------------


class TestFitTransform:
    """Tests for fit_transform."""

    def test_fit_transform_fits_normalizer(self):
        """fit_transform leaves the normalizer fitted."""

        dataset = make_dataset()

        normalizer = ForecastingDatasetNormalizer()

        result = normalizer.fit_transform(
            dataset,
        )

        assert normalizer.fitted is True
        assert isinstance(
            result,
            ForecastingDataset,
        )

    def test_fit_transform_matches_fit_then_transform(self):
        """fit_transform matches the explicit two-step workflow."""

        dataset = make_dataset()

        first = ForecastingDatasetNormalizer()

        result_one = first.fit_transform(
            dataset,
        )

        second = ForecastingDatasetNormalizer()

        second.fit(dataset)

        result_two = second.transform(
            dataset,
        )

        np.testing.assert_allclose(
            result_one.sequences,
            result_two.sequences,
        )

        np.testing.assert_array_equal(
            result_one.targets,
            result_two.targets,
        )

        np.testing.assert_array_equal(
            result_one.timestamps,
            result_two.timestamps,
        )


# ---------------------------------------------------------------------------
# Disabled normalization
# ---------------------------------------------------------------------------


class TestDisabledNormalization:
    """Tests for normalize_features=False."""

    def test_disabled_normalization_has_no_scaler(self):
        """No scaler is created when normalization is disabled."""

        config = NormalizerConfig(
            normalize_features=False,
        )

        normalizer = ForecastingDatasetNormalizer(
            config=config,
        )

        dataset = make_dataset()

        normalizer.fit(dataset)

        assert normalizer.fitted is True
        assert normalizer.scaler is None

    def test_disabled_normalization_preserves_features(self):
        """Feature values remain unchanged when normalization is disabled."""

        config = NormalizerConfig(
            normalize_features=False,
        )

        normalizer = ForecastingDatasetNormalizer(
            config=config,
        )

        dataset = make_dataset()

        normalizer.fit(dataset)

        transformed = normalizer.transform(
            dataset,
        )

        np.testing.assert_array_equal(
            transformed.sequences,
            dataset.sequences,
        )

    def test_disabled_normalization_preserves_targets(self):
        """Targets remain unchanged when normalization is disabled."""

        config = NormalizerConfig(
            normalize_features=False,
        )

        normalizer = ForecastingDatasetNormalizer(
            config=config,
        )

        dataset = make_dataset()

        normalizer.fit(dataset)

        transformed = normalizer.transform(
            dataset,
        )

        np.testing.assert_array_equal(
            transformed.targets,
            dataset.targets,
        )

    def test_disabled_normalization_still_requires_fit(self):
        """Transform still requires the normalizer to be fitted."""

        config = NormalizerConfig(
            normalize_features=False,
        )

        normalizer = ForecastingDatasetNormalizer(
            config=config,
        )

        dataset = make_dataset()

        with pytest.raises(RuntimeError):
            normalizer.transform(dataset)

    def test_disabled_normalization_fitted_property_is_true_after_fit(self):
        """
        Disabled normalization still counts as successfully fitted.

        This protects the regression fixed during Phase 7.
        """

        config = NormalizerConfig(
            normalize_features=False,
        )

        normalizer = ForecastingDatasetNormalizer(
            config=config,
        )

        dataset = make_dataset()

        normalizer.fit(dataset)

        assert normalizer.fitted is True


# ---------------------------------------------------------------------------
# Model input validation
# ---------------------------------------------------------------------------


class TestModelInputValidation:
    """Tests for model-facing validation."""

    def test_valid_dataset_passes(self):
        """A compatible dataset passes model input validation."""

        dataset = make_dataset()

        normalizer = ForecastingDatasetNormalizer()

        normalizer.fit(dataset)

        normalizer.validate_model_input(
            dataset,
        )

    def test_validation_requires_fit(self):
        """Model input validation requires a fitted normalizer."""

        dataset = make_dataset()

        normalizer = ForecastingDatasetNormalizer()

        with pytest.raises(RuntimeError):
            normalizer.validate_model_input(dataset)

    def test_wrong_feature_schema_fails(self):
        """Model input validation rejects incompatible feature schemas."""

        training = make_dataset()

        validation = make_custom_feature_dataset(
            feature_names=(
                "wrong_a",
                "wrong_b",
            ),
        )

        normalizer = ForecastingDatasetNormalizer()

        normalizer.fit(training)

        with pytest.raises(ValueError):
            normalizer.validate_model_input(
                validation,
            )


# ---------------------------------------------------------------------------
# ForecastingConfig integration
# ---------------------------------------------------------------------------


class TestForecastingConfigIntegration:
    """Tests for ForecastingConfig integration."""

    def test_from_forecasting_config(self):
        """Normalizer can be constructed from ForecastingConfig."""

        forecasting_config = ForecastingConfig()

        normalizer = (
            ForecastingDatasetNormalizer.from_forecasting_config(
                forecasting_config
            )
        )

        assert isinstance(
            normalizer,
            ForecastingDatasetNormalizer,
        )

    def test_forecasting_config_method_is_used(self):
        """ForecastingConfig normalization method is propagated."""

        forecasting_config = ForecastingConfig(
            normalization_method="standard",
        )

        normalizer = (
            ForecastingDatasetNormalizer.from_forecasting_config(
                forecasting_config
            )
        )

        assert normalizer.method == "standard"

    def test_forecasting_config_normalization_flag_is_used(self):
        """normalize_features is propagated from ForecastingConfig."""

        forecasting_config = ForecastingConfig(
            normalize_features=False,
        )

        normalizer = (
            ForecastingDatasetNormalizer.from_forecasting_config(
                forecasting_config
            )
        )

        assert normalizer.normalization_enabled is False

    def test_forecasting_config_training_only_flag_is_used(self):
        """Training-only setting is propagated."""

        forecasting_config = ForecastingConfig(
            scaler_fit_on_training_only=True,
        )

        normalizer = (
            ForecastingDatasetNormalizer.from_forecasting_config(
                forecasting_config
            )
        )

        assert normalizer.config.fit_on_training_only is True

    def test_invalid_forecasting_config_type_raises(self):
        """from_forecasting_config requires ForecastingConfig."""

        with pytest.raises(TypeError):
            ForecastingDatasetNormalizer.from_forecasting_config(
                "not-a-config",
            )


# ---------------------------------------------------------------------------
# Convenience function tests
# ---------------------------------------------------------------------------


class TestConvenienceFunctions:
    """Tests for public normalization convenience functions."""

    def test_fit_normalizer(self):
        """fit_normalizer returns a fitted normalizer."""

        dataset = make_dataset()

        normalizer = fit_normalizer(
            dataset,
        )

        assert isinstance(
            normalizer,
            ForecastingDatasetNormalizer,
        )

        assert normalizer.fitted is True

    def test_fit_normalizer_accepts_config(self):
        """fit_normalizer accepts custom configuration."""

        dataset = make_dataset()

        config = NormalizerConfig(
            dtype="float32",
        )

        normalizer = fit_normalizer(
            dataset,
            config=config,
        )

        assert normalizer.config.dtype == "float32"

    def test_normalize_forecasting_splits(self):
        """Convenience split normalization works."""

        training = make_dataset(
            num_samples=20,
        )

        validation = make_dataset(
            num_samples=10,
            timestamp_start=2000.0,
        )

        test = make_dataset(
            num_samples=8,
            timestamp_start=3000.0,
        )

        normalized = normalize_forecasting_splits(
            training,
            validation,
            test,
        )

        assert len(normalized) == 3

        for dataset in normalized:
            assert isinstance(
                dataset,
                ForecastingDataset,
            )

    def test_normalize_from_forecasting_config(self):
        """ForecastingConfig convenience API works."""

        training = make_dataset(
            num_samples=20,
        )

        validation = make_dataset(
            num_samples=10,
            timestamp_start=2000.0,
        )

        test = make_dataset(
            num_samples=8,
            timestamp_start=3000.0,
        )

        config = ForecastingConfig()

        normalized = normalize_from_forecasting_config(
            training,
            validation,
            test,
            config,
        )

        assert len(normalized) == 3

        for dataset in normalized:
            assert isinstance(
                dataset,
                ForecastingDataset,
            )


# ---------------------------------------------------------------------------
# NormalizedForecastingDataset wrapper
# ---------------------------------------------------------------------------


class TestNormalizedForecastingDataset:
    """Tests for the optional normalized dataset wrapper."""

    def test_wrapper_constructs_with_scaler(self):
        """Wrapper accepts a normalized dataset and scaler."""

        dataset = make_dataset()

        normalizer = ForecastingDatasetNormalizer()

        normalizer.fit(dataset)

        normalized = normalizer.transform(
            dataset,
        )

        wrapper = NormalizedForecastingDataset(
            dataset=normalized,
            scaler=normalizer.scaler,
            normalization_enabled=True,
        )

        assert wrapper.dataset is normalized
        assert wrapper.scaler is normalizer.scaler
        assert wrapper.normalization_enabled is True

    def test_wrapper_exposes_sequences(self):
        """Wrapper exposes normalized sequences."""

        dataset = make_dataset()

        normalizer = ForecastingDatasetNormalizer()

        normalizer.fit(dataset)

        normalized = normalizer.transform(
            dataset,
        )

        wrapper = NormalizedForecastingDataset(
            dataset=normalized,
            scaler=normalizer.scaler,
            normalization_enabled=True,
        )

        np.testing.assert_array_equal(
            wrapper.sequences,
            normalized.sequences,
        )

    def test_wrapper_exposes_targets(self):
        """Wrapper exposes unchanged targets."""

        dataset = make_dataset()

        normalizer = ForecastingDatasetNormalizer()

        normalizer.fit(dataset)

        normalized = normalizer.transform(
            dataset,
        )

        wrapper = NormalizedForecastingDataset(
            dataset=normalized,
            scaler=normalizer.scaler,
            normalization_enabled=True,
        )

        np.testing.assert_array_equal(
            wrapper.targets,
            normalized.targets,
        )

    def test_wrapper_exposes_timestamps(self):
        """Wrapper exposes unchanged timestamps."""

        dataset = make_dataset()

        normalizer = ForecastingDatasetNormalizer()

        normalizer.fit(dataset)

        normalized = normalizer.transform(
            dataset,
        )

        wrapper = NormalizedForecastingDataset(
            dataset=normalized,
            scaler=normalizer.scaler,
            normalization_enabled=True,
        )

        np.testing.assert_array_equal(
            wrapper.timestamps,
            normalized.timestamps,
        )

    def test_wrapper_exposes_shape(self):
        """Wrapper exposes model input shape."""

        dataset = make_dataset()

        normalizer = ForecastingDatasetNormalizer()

        normalizer.fit(dataset)

        normalized = normalizer.transform(
            dataset,
        )

        wrapper = NormalizedForecastingDataset(
            dataset=normalized,
            scaler=normalizer.scaler,
            normalization_enabled=True,
        )

        assert wrapper.shape == normalized.shape

    def test_enabled_wrapper_requires_scaler(self):
        """Enabled normalization requires a scaler."""

        dataset = make_dataset()

        with pytest.raises(ValueError):
            NormalizedForecastingDataset(
                dataset=dataset,
                scaler=None,
                normalization_enabled=True,
            )

    def test_disabled_wrapper_can_have_no_scaler(self):
        """Disabled normalization can omit the scaler."""

        dataset = make_dataset()

        wrapper = NormalizedForecastingDataset(
            dataset=dataset,
            scaler=None,
            normalization_enabled=False,
        )

        assert wrapper.scaler is None
        assert wrapper.normalization_enabled is False

    def test_wrapper_requires_forecasting_dataset(self):
        """Wrapper requires a ForecastingDataset."""

        with pytest.raises(TypeError):
            NormalizedForecastingDataset(
                dataset="not-a-dataset",
                scaler=None,
                normalization_enabled=False,
            )


# ---------------------------------------------------------------------------
# Dtype behavior
# ---------------------------------------------------------------------------


class TestDtypeBehavior:
    """Tests for normalization dtype handling."""

    def test_default_output_is_float64(self):
        """Default normalized features are float64."""

        dataset = make_dataset()

        normalizer = ForecastingDatasetNormalizer()

        normalizer.fit(dataset)

        transformed = normalizer.transform(
            dataset,
        )

        assert transformed.sequences.dtype == np.float64

    def test_float32_output(self):
        """Configured float32 normalization produces float32 features."""

        config = NormalizerConfig(
            dtype="float32",
        )

        dataset = make_dataset()

        normalizer = ForecastingDatasetNormalizer(
            config=config,
        )

        normalizer.fit(dataset)

        transformed = normalizer.transform(
            dataset,
        )

        assert transformed.sequences.dtype == np.float32

    def test_float32_preserves_shape(self):
        """Changing dtype does not change model dimensions."""

        config = NormalizerConfig(
            dtype="float32",
        )

        dataset = make_dataset()

        normalizer = ForecastingDatasetNormalizer(
            config=config,
        )

        normalizer.fit(dataset)

        transformed = normalizer.transform(
            dataset,
        )

        assert transformed.sequences.shape == (
            dataset.sequences.shape
        )


# ---------------------------------------------------------------------------
# Copy behavior
# ---------------------------------------------------------------------------


class TestNormalizerCopyBehavior:
    """Tests for normalizer copy semantics."""

    def test_copy_enabled_produces_independent_features(self):
        """copy_arrays=True isolates transformed feature arrays."""

        dataset = make_dataset()

        normalizer = ForecastingDatasetNormalizer(
            NormalizerConfig(
                copy_arrays=True,
            )
        )

        normalizer.fit(dataset)

        transformed = normalizer.transform(
            dataset,
        )

        original = transformed.sequences.copy()

        dataset.sequences[0, 0, 0] = (
            dataset.sequences[0, 0, 0] + 100000.0
        )

        np.testing.assert_array_equal(
            transformed.sequences,
            original,
        )

    def test_copy_enabled_produces_independent_targets(self):
        """copy_arrays=True isolates target arrays."""

        dataset = make_dataset()

        normalizer = ForecastingDatasetNormalizer(
            NormalizerConfig(
                copy_arrays=True,
            )
        )

        normalizer.fit(dataset)

        transformed = normalizer.transform(
            dataset,
        )

        original = transformed.targets.copy()

        dataset.targets[0] = 999999.0

        np.testing.assert_array_equal(
            transformed.targets,
            original,
        )

    def test_copy_enabled_produces_independent_timestamps(self):
        """copy_arrays=True isolates timestamp arrays."""

        dataset = make_dataset()

        normalizer = ForecastingDatasetNormalizer(
            NormalizerConfig(
                copy_arrays=True,
            )
        )

        normalizer.fit(dataset)

        transformed = normalizer.transform(
            dataset,
        )

        original = transformed.timestamps.copy()

        dataset.timestamps[0] = 999999.0

        np.testing.assert_array_equal(
            transformed.timestamps,
            original,
        )


# ---------------------------------------------------------------------------
# Repeated transformation
# ---------------------------------------------------------------------------


class TestRepeatedTransformation:
    """Tests for deterministic repeated transformations."""

    def test_repeated_transformations_are_identical(self):
        """Transforming the same dataset twice produces identical output."""

        dataset = make_dataset()

        normalizer = ForecastingDatasetNormalizer()

        normalizer.fit(dataset)

        first = normalizer.transform(
            dataset,
        )

        second = normalizer.transform(
            dataset,
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

    def test_repeated_split_transformations_are_identical(self):
        """Repeated split transformation is deterministic."""

        training = make_dataset(
            num_samples=20,
        )

        validation = make_dataset(
            num_samples=10,
            timestamp_start=2000.0,
        )

        test = make_dataset(
            num_samples=8,
            timestamp_start=3000.0,
        )

        normalizer = ForecastingDatasetNormalizer()

        first = normalizer.transform_splits(
            training,
            validation,
            test,
        )

        second = normalizer.transform_splits(
            training,
            validation,
            test,
        )

        for first_dataset, second_dataset in zip(
            first,
            second,
        ):
            np.testing.assert_array_equal(
                first_dataset.sequences,
                second_dataset.sequences,
            )

            np.testing.assert_array_equal(
                first_dataset.targets,
                second_dataset.targets,
            )

            np.testing.assert_array_equal(
                first_dataset.timestamps,
                second_dataset.timestamps,
            )


# ---------------------------------------------------------------------------
# Invalid dataset values
# ---------------------------------------------------------------------------


class TestInvalidDatasetValues:
    """Tests for invalid feature values."""

    def test_nan_validation_features_raise(self):
        """NaN validation features cannot be transformed."""

        training = make_dataset()
        validation = make_dataset(
            timestamp_start=2000.0,
        )

        validation.sequences[0, 0, 0] = np.nan

        normalizer = ForecastingDatasetNormalizer()

        normalizer.fit(training)

        with pytest.raises(ValueError):
            normalizer.transform(validation)

    def test_infinite_validation_features_raise(self):
        """Infinite validation features cannot be transformed."""

        training = make_dataset()
        validation = make_dataset(
            timestamp_start=2000.0,
        )

        validation.sequences[0, 0, 0] = np.inf

        normalizer = ForecastingDatasetNormalizer()

        normalizer.fit(training)

        with pytest.raises(ValueError):
            normalizer.transform(validation)

    def test_nan_test_features_raise(self):
        """NaN test features cannot be transformed."""

        training = make_dataset()
        test = make_dataset(
            timestamp_start=3000.0,
        )

        test.sequences[0, 0, 0] = np.nan

        normalizer = ForecastingDatasetNormalizer()

        normalizer.fit(training)

        with pytest.raises(ValueError):
            normalizer.transform(test)


# ---------------------------------------------------------------------------
# NVIDIA forecasting contract regression tests
# ---------------------------------------------------------------------------


class TestNVIDIAForecastingContract:
    """
    Regression tests for the current NVIDIA forecasting input contract.

    Expected:

        context_window = 20
        feature_count = 14
        target shape = (N,)
    """

    def test_normalized_input_is_3d(self):
        """Normalized NVIDIA inputs remain three-dimensional."""

        dataset = make_dataset()

        normalizer = ForecastingDatasetNormalizer()

        normalizer.fit(dataset)

        normalized = normalizer.transform(
            dataset,
        )

        assert normalized.sequences.ndim == 3

    def test_normalized_input_shape_is_n_20_14(self):
        """Normalized NVIDIA input has shape (N, 20, 14)."""

        dataset = make_dataset()

        normalizer = ForecastingDatasetNormalizer()

        normalizer.fit(dataset)

        normalized = normalizer.transform(
            dataset,
        )

        assert normalized.sequences.shape[1:] == (
            20,
            14,
        )

    def test_normalized_target_shape_is_n(self):
        """Normalized dataset retains one target per sequence."""

        dataset = make_dataset()

        normalizer = ForecastingDatasetNormalizer()

        normalizer.fit(dataset)

        normalized = normalizer.transform(
            dataset,
        )

        assert normalized.targets.shape == (
            normalized.num_samples,
        )

    def test_normalization_does_not_change_context_window(self):
        """Context window remains exactly 20."""

        dataset = make_dataset()

        normalizer = ForecastingDatasetNormalizer()

        normalizer.fit(dataset)

        normalized = normalizer.transform(
            dataset,
        )

        assert normalized.context_window == 20

    def test_normalization_does_not_change_feature_order(self):
        """Feature ordering remains the NVIDIA contract order."""

        dataset = make_dataset()

        normalizer = ForecastingDatasetNormalizer()

        normalizer.fit(dataset)

        normalized = normalizer.transform(
            dataset,
        )

        assert normalized.feature_names == FEATURE_NAMES

    def test_normalization_does_not_change_forecast_horizon(self):
        """Forecast horizon remains five observations."""

        dataset = make_dataset()

        normalizer = ForecastingDatasetNormalizer()

        normalizer.fit(dataset)

        normalized = normalizer.transform(
            dataset,
        )

        assert normalized.forecast_horizon == 5


# ---------------------------------------------------------------------------
# Complete integration regression test
# ---------------------------------------------------------------------------


def test_complete_phase_seven_normalization_contract():
    """
    Verify the complete leakage-safe normalization contract.

    The final normalized dataset must provide:

        - 3D feature inputs
        - 20 timestep context
        - 14 features
        - 1D targets
        - unchanged timestamps
        - unchanged targets
        - finite normalized features
        - training features approximately zero mean
        - training features approximately unit variance
        - fitted scaler metadata
    """

    training = make_dataset(
        num_samples=30,
        sequence_offset=0.0,
    )

    validation = make_dataset(
        num_samples=10,
        sequence_offset=100000.0,
        timestamp_start=2000.0,
    )

    test = make_dataset(
        num_samples=10,
        sequence_offset=-100000.0,
        timestamp_start=3000.0,
    )

    original_validation_targets = validation.targets.copy()
    original_validation_timestamps = (
        validation.timestamps.copy()
    )

    original_test_targets = test.targets.copy()
    original_test_timestamps = (
        test.timestamps.copy()
    )

    normalizer = ForecastingDatasetNormalizer()

    normalized_train, normalized_validation, normalized_test = (
        normalizer.transform_splits(
            training,
            validation,
            test,
        )
    )

    # Fitted state.
    assert normalizer.fitted is True
    assert normalizer.scaler is not None
    assert normalizer.feature_names == FEATURE_NAMES

    # Model input contract.
    assert normalized_train.sequences.ndim == 3
    assert normalized_train.sequences.shape[1:] == (
        20,
        14,
    )

    assert normalized_validation.sequences.shape[1:] == (
        20,
        14,
    )

    assert normalized_test.sequences.shape[1:] == (
        20,
        14,
    )

    # Target contract.
    assert normalized_train.targets.ndim == 1
    assert normalized_validation.targets.ndim == 1
    assert normalized_test.targets.ndim == 1

    # Training normalization.
    flattened_training = (
        normalized_train.sequences.reshape(
            -1,
            normalized_train.num_features,
        )
    )

    np.testing.assert_allclose(
        flattened_training.mean(axis=0),
        np.zeros(normalized_train.num_features),
        atol=1e-12,
    )

    np.testing.assert_allclose(
        flattened_training.std(axis=0),
        np.ones(normalized_train.num_features),
        atol=1e-12,
    )

    # Validation/test values must remain finite.
    assert np.all(
        np.isfinite(
            normalized_validation.sequences
        )
    )

    assert np.all(
        np.isfinite(
            normalized_test.sequences
        )
    )

    # Validation/test targets remain unchanged.
    np.testing.assert_array_equal(
        normalized_validation.targets,
        original_validation_targets,
    )

    np.testing.assert_array_equal(
        normalized_test.targets,
        original_test_targets,
    )

    # Validation/test timestamps remain unchanged.
    np.testing.assert_array_equal(
        normalized_validation.timestamps,
        original_validation_timestamps,
    )

    np.testing.assert_array_equal(
        normalized_test.timestamps,
        original_test_timestamps,
    )

    # Metadata remains intact.
    assert normalized_train.feature_names == FEATURE_NAMES
    assert normalized_train.context_window == 20
    assert normalized_train.forecast_horizon == 5
    assert normalized_train.target_name == (
        "future_mid_price_return"
    )
    assert normalized_train.price_column == "mid_price" 

