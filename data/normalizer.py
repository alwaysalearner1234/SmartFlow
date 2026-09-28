"""
data/normalizer.py

Training-only feature normalization for forecasting datasets.

This module provides the normalization layer between the chronological
forecasting dataset pipeline and downstream machine-learning models.

The normalization pipeline is intentionally strict:

    ForecastingDataset
        -> Training Split
        -> Fit Scaler on Training Features Only
        -> Transform Training Features
        -> Transform Validation Features
        -> Transform Test Features
        -> NVIDIA Forecasting Model

The most important rule in this module is leakage prevention:

    The scaler MUST be fitted using training data only.

Validation and test data are transformed using the statistics learned from
the training split. Their statistics must never influence the fitted scaler.

Current normalization support:
    - Standard normalization

Standard normalization:

    normalized_value = (value - mean) / scale

where mean and scale are calculated independently for each feature using
training samples and all timesteps inside each context window.

Expected model input shape:

    (num_samples, context_window, num_features)

For the current NVIDIA forecasting contract:

    context_window = 20
    num_features = 14

Therefore a typical model input has shape:

    (N, 20, 14)

Targets remain unchanged by normalization and retain shape:

    (N,)

This module is responsible for:
    - validating forecasting datasets
    - fitting feature normalization parameters
    - enforcing training-only scaler fitting
    - transforming datasets
    - transforming train/validation/test splits consistently
    - preserving forecasting metadata
    - exposing fitted scaler information
    - supporting the ForecastingConfig contract

This module does not:
    - build feature sequences
    - build forecast targets
    - align sequences and targets
    - split chronological datasets
    - train machine-learning models
    - perform model inference
    - normalize target values
    - modify timestamps
    - modify feature names
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import numpy as np

from config.config import ForecastingConfig
from data.forecasting_dataset import ForecastingDataset


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_NORMALIZATION_METHOD = "standard"
SUPPORTED_NORMALIZATION_METHODS = ("standard",)

DEFAULT_DTYPE = "float64"


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class NormalizerConfig:
    """
    Configuration for forecasting feature normalization.

    Attributes:
        method:
            Normalization method to use.

        normalize_features:
            Whether feature normalization is enabled.

        fit_on_training_only:
            Whether scaler statistics must be learned from training data
            only. This should remain True for leakage-safe forecasting.

        dtype:
            NumPy dtype used for transformed feature arrays.

        copy_arrays:
            Whether transformed arrays should be independent copies rather
            than views into the original dataset.
    """

    method: str = DEFAULT_NORMALIZATION_METHOD
    normalize_features: bool = True
    fit_on_training_only: bool = True
    dtype: str = DEFAULT_DTYPE
    copy_arrays: bool = True

    def __post_init__(self) -> None:
        """Validate normalization configuration."""

        if not isinstance(self.method, str):
            raise TypeError("method must be a string.")

        normalized_method = self.method.strip().lower()

        if normalized_method not in SUPPORTED_NORMALIZATION_METHODS:
            raise ValueError(
                f"Unsupported normalization method: {self.method!r}. "
                f"Supported methods: {SUPPORTED_NORMALIZATION_METHODS}."
            )

        if not isinstance(self.normalize_features, bool):
            raise TypeError("normalize_features must be a boolean.")

        if not isinstance(self.fit_on_training_only, bool):
            raise TypeError("fit_on_training_only must be a boolean.")

        if not isinstance(self.dtype, str) or not self.dtype.strip():
            raise ValueError("dtype must be a non-empty string.")

        try:
            np.dtype(self.dtype)
        except TypeError as exc:
            raise ValueError(f"Invalid NumPy dtype: {self.dtype!r}.") from exc

        if not isinstance(self.copy_arrays, bool):
            raise TypeError("copy_arrays must be a boolean.")


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------


def _validate_method(method: str) -> str:
    """
    Normalize and validate a normalization method name.
    """

    if not isinstance(method, str):
        raise TypeError("Normalization method must be a string.")

    normalized_method = method.strip().lower()

    if normalized_method not in SUPPORTED_NORMALIZATION_METHODS:
        raise ValueError(
            f"Unsupported normalization method: {method!r}. "
            f"Supported methods: {SUPPORTED_NORMALIZATION_METHODS}."
        )

    return normalized_method


def _validate_dtype(dtype: str) -> np.dtype:
    """
    Validate and return a NumPy dtype.
    """

    if not isinstance(dtype, str) or not dtype.strip():
        raise ValueError("dtype must be a non-empty string.")

    try:
        return np.dtype(dtype)
    except TypeError as exc:
        raise ValueError(f"Invalid NumPy dtype: {dtype!r}.") from exc


def _validate_dataset(
    dataset: ForecastingDataset,
    *,
    name: str = "dataset",
) -> None:
    """
    Validate that an object is a usable ForecastingDataset.
    """

    if not isinstance(dataset, ForecastingDataset):
        raise TypeError(
            f"{name} must be a ForecastingDataset, "
            f"got {type(dataset).__name__}."
        )

    if dataset.sequences.ndim != 3:
        raise ValueError(
            f"{name}.sequences must be 3-dimensional. "
            f"Got shape {dataset.sequences.shape}."
        )

    if dataset.targets.ndim != 1:
        raise ValueError(
            f"{name}.targets must be 1-dimensional. "
            f"Got shape {dataset.targets.shape}."
        )

    if dataset.timestamps.ndim != 1:
        raise ValueError(
            f"{name}.timestamps must be 1-dimensional. "
            f"Got shape {dataset.timestamps.shape}."
        )

    if dataset.sequences.shape[0] != dataset.targets.shape[0]:
        raise ValueError(
            f"{name} sequence/target sample counts do not match."
        )

    if dataset.sequences.shape[0] != dataset.timestamps.shape[0]:
        raise ValueError(
            f"{name} sequence/timestamp sample counts do not match."
        )

    if dataset.sequences.shape[0] == 0:
        raise ValueError(f"{name} must contain at least one sample.")


def _validate_sequence_values(
    dataset: ForecastingDataset,
    *,
    name: str = "dataset",
) -> None:
    """
    Validate feature values before fitting or transforming.
    """

    try:
        values = np.asarray(dataset.sequences, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise TypeError(
            f"{name}.sequences must contain numeric feature values."
        ) from exc

    if values.ndim != 3:
        raise ValueError(
            f"{name}.sequences must be 3-dimensional. "
            f"Got shape {values.shape}."
        )

    if np.any(np.isinf(values)):
        raise ValueError(
            f"{name}.sequences contains infinite feature values."
        )

    if np.any(np.isnan(values)):
        raise ValueError(
            f"{name}.sequences contains NaN feature values. "
            "Normalization requires complete numeric feature sequences."
        )


def _validate_feature_compatibility(
    expected_names: Sequence[str],
    dataset: ForecastingDataset,
    *,
    name: str = "dataset",
) -> None:
    """
    Ensure a dataset contains the exact feature schema used during fitting.
    """

    expected = tuple(expected_names)
    actual = tuple(dataset.feature_names)

    if actual != expected:
        raise ValueError(
            f"{name} feature schema does not match the fitted scaler. "
            f"Expected {expected}, got {actual}."
        )


def _validate_training_only_requirement(
    config: NormalizerConfig,
) -> None:
    """
    Ensure leakage-safe training-only fitting is enabled.
    """

    if not config.fit_on_training_only:
        raise ValueError(
            "fit_on_training_only must be True for the forecasting "
            "normalization pipeline. Fitting normalization statistics "
            "using validation or test data would introduce data leakage."
        )


# ---------------------------------------------------------------------------
# Standard scaler
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class StandardScaler:
    """
    Fitted per-feature standard normalization parameters.

    The scaler stores one mean and one scale for every feature.

    Statistics are learned across:
        - all training samples
        - all context-window timesteps

    They are NOT learned independently per timestep.

    Attributes:
        means:
            Per-feature training means.

        scales:
            Per-feature standard deviations.

        feature_names:
            Feature names in the exact order used during fitting.
    """

    means: np.ndarray
    scales: np.ndarray
    feature_names: Tuple[str, ...]

    def __post_init__(self) -> None:
        """Validate fitted scaler parameters."""

        means = np.asarray(self.means, dtype=np.float64)
        scales = np.asarray(self.scales, dtype=np.float64)

        if means.ndim != 1:
            raise ValueError("means must be a 1-dimensional array.")

        if scales.ndim != 1:
            raise ValueError("scales must be a 1-dimensional array.")

        if means.shape != scales.shape:
            raise ValueError(
                "means and scales must have identical shapes."
            )

        if means.size == 0:
            raise ValueError("Scaler must contain at least one feature.")

        if len(self.feature_names) != means.size:
            raise ValueError(
                "Number of feature names must match scaler statistics."
            )

        if any(
            not isinstance(name, str) or not name.strip()
            for name in self.feature_names
        ):
            raise ValueError(
                "feature_names must contain non-empty strings."
            )

        if len(set(self.feature_names)) != len(self.feature_names):
            raise ValueError(
                "feature_names must contain unique feature names."
            )

        if not np.all(np.isfinite(means)):
            raise ValueError("means must contain only finite values.")

        if not np.all(np.isfinite(scales)):
            raise ValueError("scales must contain only finite values.")

        if np.any(scales <= 0):
            raise ValueError("All scaler scales must be greater than zero.")

        object.__setattr__(self, "means", means)
        object.__setattr__(self, "scales", scales)
        object.__setattr__(
            self,
            "feature_names",
            tuple(self.feature_names),
        )

    @property
    def num_features(self) -> int:
        """Return the number of normalized features."""

        return int(self.means.shape[0])

    @property
    def fitted(self) -> bool:
        """Return whether the scaler contains fitted statistics."""

        return True

    def transform(
        self,
        sequences: np.ndarray,
        *,
        dtype: str = DEFAULT_DTYPE,
        copy: bool = True,
    ) -> np.ndarray:
        """
        Normalize a 3D sequence array using fitted statistics.

        Expected input shape:

            (samples, context_window, num_features)

        Returns:
            Normalized sequence array with identical shape.
        """

        values = np.asarray(sequences, dtype=np.float64)

        if values.ndim != 3:
            raise ValueError(
                "sequences must be 3-dimensional. "
                f"Got shape {values.shape}."
            )

        if values.shape[-1] != self.num_features:
            raise ValueError(
                "Sequence feature count does not match the fitted scaler. "
                f"Expected {self.num_features}, "
                f"got {values.shape[-1]}."
            )

        if np.any(np.isnan(values)):
            raise ValueError(
                "sequences contains NaN values and cannot be normalized."
            )

        if np.any(np.isinf(values)):
            raise ValueError(
                "sequences contains infinite values and cannot be normalized."
            )

        normalized = (
            values - self.means.reshape(1, 1, -1)
        ) / self.scales.reshape(1, 1, -1)

        result_dtype = _validate_dtype(dtype)

        if copy:
            return np.array(
                normalized,
                dtype=result_dtype,
                copy=True,
            )

        return normalized.astype(result_dtype, copy=False)


# ---------------------------------------------------------------------------
# Normalized dataset wrapper
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class NormalizedForecastingDataset:
    """
    Optional wrapper containing a normalized dataset and its fitted scaler.

    This wrapper is useful when callers need both the transformed dataset and
    the exact normalization parameters used to create it.

    The underlying ForecastingDataset remains the primary public dataset
    contract.
    """

    dataset: ForecastingDataset
    scaler: Optional[StandardScaler]
    normalization_enabled: bool

    def __post_init__(self) -> None:
        """Validate normalized dataset wrapper."""

        if not isinstance(self.dataset, ForecastingDataset):
            raise TypeError(
                "dataset must be a ForecastingDataset."
            )

        if not isinstance(self.normalization_enabled, bool):
            raise TypeError(
                "normalization_enabled must be a boolean."
            )

        if self.normalization_enabled and self.scaler is None:
            raise ValueError(
                "A scaler is required when normalization is enabled."
            )

    @property
    def shape(self) -> tuple:
        """Return the normalized feature sequence shape."""

        return self.dataset.shape

    @property
    def sequences(self) -> np.ndarray:
        """Return normalized feature sequences."""

        return self.dataset.sequences

    @property
    def targets(self) -> np.ndarray:
        """Return unchanged forecasting targets."""

        return self.dataset.targets

    @property
    def timestamps(self) -> np.ndarray:
        """Return unchanged sequence-end timestamps."""

        return self.dataset.timestamps


# ---------------------------------------------------------------------------
# Forecasting dataset normalizer
# ---------------------------------------------------------------------------


class ForecastingDatasetNormalizer:
    """
    Fit and apply leakage-safe feature normalization.

    Typical usage:

        normalizer = ForecastingDatasetNormalizer.from_forecasting_config(
            forecasting_config
        )

        normalizer.fit(train_dataset)

        train_normalized = normalizer.transform(train_dataset)
        validation_normalized = normalizer.transform(validation_dataset)
        test_normalized = normalizer.transform(test_dataset)

    Or:

        train_normalized, validation_normalized, test_normalized = (
            normalizer.transform_splits(
                train_dataset,
                validation_dataset,
                test_dataset,
            )
        )

    Important:
        Calling fit() on the training split is the intended workflow.

    The normalizer stores no target statistics because targets are not
    normalized in this forecasting pipeline.
    """

    def __init__(
        self,
        config: Optional[NormalizerConfig] = None,
    ) -> None:
        """
        Initialize the forecasting dataset normalizer.
        """

        self.config = config or NormalizerConfig()

        _validate_method(self.config.method)
        _validate_dtype(self.config.dtype)

        self._scaler: Optional[StandardScaler] = None
        self._training_feature_names: Optional[Tuple[str, ...]] = None
        self._fitted = False

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def fitted(self) -> bool:
        """
        Return whether the fit operation has completed successfully.
        """

        return self._fitted

    @property
    def scaler(self) -> Optional[StandardScaler]:
        """
        Return the fitted scaler.

        Returns:
            StandardScaler when normalization is enabled and fitted.

            None when normalization is disabled.
        """

        return self._scaler

    @property
    def feature_names(self) -> Optional[Tuple[str, ...]]:
        """
        Return the feature schema used during fitting.
        """

        return self._training_feature_names

    @property
    def method(self) -> str:
        """Return the active normalization method."""

        return _validate_method(self.config.method)

    @property
    def normalization_enabled(self) -> bool:
        """Return whether feature normalization is enabled."""

        return self.config.normalize_features

    # ------------------------------------------------------------------
    # Fitting
    # ------------------------------------------------------------------

    def fit(
        self,
        training_dataset: ForecastingDataset,
    ) -> "ForecastingDatasetNormalizer":
        """
        Fit normalization statistics using training data only.

        For standard normalization, each feature's mean and standard
        deviation are calculated over every training sample and every
        timestep in its context window.

        No validation or test data is accepted by this method.

        Args:
            training_dataset:
                Chronological training ForecastingDataset.

        Returns:
            This normalizer instance.

        Raises:
            ValueError:
                If the training dataset is invalid or contains invalid
                feature values.
        """

        _validate_training_only_requirement(self.config)

        _validate_dataset(
            training_dataset,
            name="training_dataset",
        )

        _validate_sequence_values(
            training_dataset,
            name="training_dataset",
        )

        feature_names = tuple(training_dataset.feature_names)

        if not feature_names:
            raise ValueError(
                "training_dataset must contain at least one feature."
            )

        self._training_feature_names = feature_names

        if not self.config.normalize_features:
            self._scaler = None
            self._fitted = True
            return self

        method = _validate_method(self.config.method)

        if method == "standard":
            values = np.asarray(
                training_dataset.sequences,
                dtype=np.float64,
            )

            # Collapse samples and context timesteps while preserving
            # the final feature dimension.
            flattened = values.reshape(-1, values.shape[-1])

            means = np.mean(flattened, axis=0)
            scales = np.std(
                flattened,
                axis=0,
                ddof=0,
            )

            if not np.all(np.isfinite(means)):
                raise ValueError(
                    "Training feature means contain non-finite values."
                )

            if not np.all(np.isfinite(scales)):
                raise ValueError(
                    "Training feature scales contain non-finite values."
                )

            # A constant feature contains no useful variance but should
            # remain usable. A scale of 1 preserves the centered value
            # without causing division by zero.
            scales = np.where(scales == 0.0, 1.0, scales)

            self._scaler = StandardScaler(
                means=means,
                scales=scales,
                feature_names=feature_names,
            )

        else:
            raise ValueError(
                f"Unsupported normalization method: {method!r}."
            )

        self._fitted = True

        return self

    # ------------------------------------------------------------------
    # Transformation
    # ------------------------------------------------------------------

    def transform(
        self,
        dataset: ForecastingDataset,
    ) -> ForecastingDataset:
        """
        Transform a ForecastingDataset using fitted training statistics.

        The feature schema must exactly match the schema used during fit.

        Targets and timestamps are copied unchanged.

        Args:
            dataset:
                Dataset to transform.

        Returns:
            A new ForecastingDataset containing normalized feature
            sequences and unchanged targets/timestamps.
        """

        if not self._fitted:
            raise RuntimeError(
                "Normalizer has not been fitted. Call fit() using the "
                "training dataset before transforming data."
            )

        _validate_dataset(dataset)
        _validate_sequence_values(dataset)

        if self._training_feature_names is None:
            raise RuntimeError(
                "Normalizer is marked as fitted but has no training "
                "feature schema."
            )

        _validate_feature_compatibility(
            self._training_feature_names,
            dataset,
        )

        if self.config.normalize_features:
            if self._scaler is None:
                raise RuntimeError(
                    "Normalizer is marked as fitted but has no scaler."
                )

            normalized_sequences = self._scaler.transform(
                dataset.sequences,
                dtype=self.config.dtype,
                copy=self.config.copy_arrays,
            )

        else:
            dtype = _validate_dtype(self.config.dtype)

            normalized_sequences = np.asarray(
                dataset.sequences,
                dtype=dtype,
            )

            if self.config.copy_arrays:
                normalized_sequences = normalized_sequences.copy()

        targets = np.asarray(
            dataset.targets,
            dtype=dtype if not self.config.normalize_features else np.float64,
        )

        timestamps = np.asarray(
            dataset.timestamps,
            dtype=np.float64,
        )

        if self.config.copy_arrays:
            targets = targets.copy()
            timestamps = timestamps.copy()

        return ForecastingDataset(
            sequences=normalized_sequences,
            targets=targets,
            timestamps=timestamps,
            feature_names=tuple(dataset.feature_names),
            context_window=dataset.context_window,
            forecast_horizon=dataset.forecast_horizon,
            target_name=dataset.target_name,
            price_column=dataset.price_column,
        )

    def fit_transform(
        self,
        training_dataset: ForecastingDataset,
    ) -> ForecastingDataset:
        """
        Fit using training data and immediately transform it.

        This is equivalent to:

            normalizer.fit(training_dataset)
            normalizer.transform(training_dataset)

        The scaler is fitted before transformation and therefore does not
        introduce leakage from validation or test data.
        """

        self.fit(training_dataset)
        return self.transform(training_dataset)

    # ------------------------------------------------------------------
    # Split transformation
    # ------------------------------------------------------------------

    def transform_splits(
        self,
        training_dataset: ForecastingDataset,
        validation_dataset: ForecastingDataset,
        test_dataset: ForecastingDataset,
    ) -> Tuple[
        ForecastingDataset,
        ForecastingDataset,
        ForecastingDataset,
    ]:
        """
        Fit on training data and transform all three forecasting splits.

        This is the preferred high-level API for the forecasting pipeline.

        The order of operations is strictly:

            1. Validate train/validation/test.
            2. Fit scaler using training only.
            3. Transform training.
            4. Transform validation using training statistics.
            5. Transform test using training statistics.

        Validation and test distributions never influence the fitted scaler.

        Returns:
            Tuple containing:

                normalized_training_dataset
                normalized_validation_dataset
                normalized_test_dataset
        """

        _validate_dataset(
            training_dataset,
            name="training_dataset",
        )
        _validate_dataset(
            validation_dataset,
            name="validation_dataset",
        )
        _validate_dataset(
            test_dataset,
            name="test_dataset",
        )

        _validate_sequence_values(
            training_dataset,
            name="training_dataset",
        )
        _validate_sequence_values(
            validation_dataset,
            name="validation_dataset",
        )
        _validate_sequence_values(
            test_dataset,
            name="test_dataset",
        )

        feature_names = tuple(training_dataset.feature_names)

        _validate_feature_compatibility(
            feature_names,
            validation_dataset,
            name="validation_dataset",
        )

        _validate_feature_compatibility(
            feature_names,
            test_dataset,
            name="test_dataset",
        )

        self.fit(training_dataset)

        normalized_training = self.transform(
            training_dataset
        )

        normalized_validation = self.transform(
            validation_dataset
        )

        normalized_test = self.transform(
            test_dataset
        )

        return (
            normalized_training,
            normalized_validation,
            normalized_test,
        )

    # ------------------------------------------------------------------
    # Model-facing validation
    # ------------------------------------------------------------------

    def validate_model_input(
        self,
        dataset: ForecastingDataset,
    ) -> None:
        """
        Validate that a dataset is compatible with the fitted normalizer.

        This method verifies the model-facing feature schema and shape
        without modifying the dataset.

        The expected shape is:

            (N, context_window, num_features)

        where N may differ between train, validation, and test.
        """

        _validate_dataset(dataset)

        if not self._fitted:
            raise RuntimeError(
                "Normalizer has not been fitted."
            )

        if self._training_feature_names is None:
            raise RuntimeError(
                "Fitted normalizer has no feature schema."
            )

        _validate_feature_compatibility(
            self._training_feature_names,
            dataset,
        )

        if dataset.context_window <= 0:
            raise ValueError(
                "Dataset context_window must be positive."
            )

        expected_features = len(self._training_feature_names)

        if dataset.num_features != expected_features:
            raise ValueError(
                "Dataset feature count does not match the fitted "
                f"normalization schema. Expected {expected_features}, "
                f"got {dataset.num_features}."
            )

        if dataset.targets.ndim != 1:
            raise ValueError(
                "Forecasting targets must remain 1-dimensional."
            )

    # ------------------------------------------------------------------
    # Configuration integration
    # ------------------------------------------------------------------

    @classmethod
    def from_forecasting_config(
        cls,
        forecasting_config: ForecastingConfig,
    ) -> "ForecastingDatasetNormalizer":
        """
        Construct a normalizer from the project's ForecastingConfig.

        The ForecastingConfig controls whether normalization is enabled,
        which method is used, and whether scaler fitting is restricted to
        training data.

        The project's leakage-safe contract requires
        scaler_fit_on_training_only=True.
        """

        if not isinstance(
            forecasting_config,
            ForecastingConfig,
        ):
            raise TypeError(
                "forecasting_config must be a ForecastingConfig."
            )

        config = NormalizerConfig(
            method=forecasting_config.normalization_method,
            normalize_features=forecasting_config.normalize_features,
            fit_on_training_only=(
                forecasting_config.scaler_fit_on_training_only
            ),
        )

        return cls(config=config)


# ---------------------------------------------------------------------------
# Convenience functions
# ---------------------------------------------------------------------------


def fit_normalizer(
    training_dataset: ForecastingDataset,
    config: Optional[NormalizerConfig] = None,
) -> ForecastingDatasetNormalizer:
    """
    Fit a leakage-safe normalizer on a training dataset.

    This is a convenience wrapper around ForecastingDatasetNormalizer.fit().
    """

    normalizer = ForecastingDatasetNormalizer(config=config)
    normalizer.fit(training_dataset)
    return normalizer


def normalize_forecasting_splits(
    training_dataset: ForecastingDataset,
    validation_dataset: ForecastingDataset,
    test_dataset: ForecastingDataset,
    config: Optional[NormalizerConfig] = None,
) -> Tuple[
    ForecastingDataset,
    ForecastingDataset,
    ForecastingDataset,
]:
    """
    Fit on training data and normalize train/validation/test splits.

    This is the recommended convenience function for the complete
    normalization stage of the forecasting pipeline.
    """

    normalizer = ForecastingDatasetNormalizer(config=config)

    return normalizer.transform_splits(
        training_dataset,
        validation_dataset,
        test_dataset,
    )


def normalize_from_forecasting_config(
    training_dataset: ForecastingDataset,
    validation_dataset: ForecastingDataset,
    test_dataset: ForecastingDataset,
    forecasting_config: ForecastingConfig,
) -> Tuple[
    ForecastingDataset,
    ForecastingDataset,
    ForecastingDataset,
]:
    """
    Normalize forecasting splits using the project's ForecastingConfig.
    """

    normalizer = ForecastingDatasetNormalizer.from_forecasting_config(
        forecasting_config
    )

    return normalizer.transform_splits(
        training_dataset,
        validation_dataset,
        test_dataset,
    )


# ---------------------------------------------------------------------------
# Public exports
# ---------------------------------------------------------------------------


__all__ = [
    "DEFAULT_NORMALIZATION_METHOD",
    "SUPPORTED_NORMALIZATION_METHODS",
    "DEFAULT_DTYPE",
    "NormalizerConfig",
    "StandardScaler",
    "NormalizedForecastingDataset",
    "ForecastingDatasetNormalizer",
    "fit_normalizer",
    "normalize_forecasting_splits",
    "normalize_from_forecasting_config",
]

