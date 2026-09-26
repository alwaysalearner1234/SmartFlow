# normalizer.py

# Forecasting Feature Normalization Module for SmartFlow

# Provides training-only normalization for the forecasting pipeline.
#
# This module intentionally operates on the existing ForecastingDataset
# contract and does not contain model-training or model-inference logic.
#
# The scaler is fitted exclusively on the training dataset and can then
# be reused to transform validation, test, and later inference data.
#
# The normalization process preserves:
#
#     - sequence shape
#     - target values
#     - timestamps
#     - feature ordering
#     - forecasting metadata
#
# The default normalization method is standardization:
#
#     normalized_value = (value - mean) / standard_deviation
#
# For sequence data with shape:
#
#     (num_samples, context_window, num_features)
#
# scaler statistics are learned independently for each feature across all
# training observations and all positions inside the context windows.
#
# This module is designed to prevent temporal leakage. Validation and test
# observations are never used when fitting the scaler.


from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

from config.config import ForecastingConfig
from data.forecasting_dataset import ForecastingDataset


# ============================================================================
# Constants
# ============================================================================

DEFAULT_NORMALIZATION_METHOD = "standard"


SUPPORTED_NORMALIZATION_METHODS = (
    "standard",
)


# ============================================================================
# Normalization Configuration
# ============================================================================

@dataclass(frozen=True)
class NormalizerConfig:
    """
    Configuration for forecasting feature normalization.

    Attributes:
        method:
            Normalization method to use.

            Currently supported:

                "standard"

        normalize_features:
            If True, feature values are normalized.

            If False, the input dataset is returned unchanged except for
            defensive copying.

        fit_on_training_only:
            If True, the normalizer is explicitly configured for training-only
            scaler fitting.

            This is the required setting for the forecasting pipeline.

        dtype:
            NumPy dtype used for normalized feature arrays.

        copy_arrays:
            If True, transformed arrays are copied instead of sharing memory
            with the source dataset.
    """

    method: str = DEFAULT_NORMALIZATION_METHOD
    normalize_features: bool = True
    fit_on_training_only: bool = True
    dtype: str = "float64"
    copy_arrays: bool = True


# ============================================================================
# Validation Helpers
# ============================================================================

def _validate_method(method: str) -> str:
    """
    Validate and normalize the configured normalization method.

    Args:
        method:
            Normalization method name.

    Returns:
        Lowercase normalization method.

    Raises:
        ValueError:
            If the method is empty or unsupported.
    """

    if not isinstance(method, str):
        raise TypeError(
            "Normalization method must be a string."
        )

    normalized_method = method.strip().lower()

    if not normalized_method:
        raise ValueError(
            "Normalization method must not be empty."
        )

    if normalized_method not in SUPPORTED_NORMALIZATION_METHODS:
        raise ValueError(
            "Unsupported normalization method: "
            f"{method!r}. Supported methods are: "
            f"{SUPPORTED_NORMALIZATION_METHODS}."
        )

    return normalized_method


def _validate_dataset(
    dataset: ForecastingDataset,
    dataset_name: str,
) -> None:
    """
    Validate that the supplied object is a forecasting dataset.

    ForecastingDataset performs its own structural validation. This helper
    provides an explicit type boundary for the normalizer.

    Args:
        dataset:
            Dataset to validate.

        dataset_name:
            Human-readable name used in error messages.

    Raises:
        TypeError:
            If the supplied object is not a ForecastingDataset.

        ValueError:
            If the dataset contains no samples.
    """

    if not isinstance(dataset, ForecastingDataset):
        raise TypeError(
            f"{dataset_name} must be an instance of ForecastingDataset."
        )

    if dataset.num_samples == 0:
        raise ValueError(
            f"{dataset_name} must contain at least one sample."
        )


def _validate_sequence_values(
    sequences: np.ndarray,
    dataset_name: str,
) -> None:
    """
    Validate the numerical values of a sequence tensor.

    Args:
        sequences:
            Three-dimensional feature tensor.

        dataset_name:
            Human-readable dataset name for error messages.

    Raises:
        ValueError:
            If the sequence tensor is not numeric, three-dimensional,
            or contains non-finite values.
    """

    sequences = np.asarray(sequences)

    if sequences.ndim != 3:
        raise ValueError(
            f"{dataset_name} sequences must be three-dimensional."
        )

    if not np.issubdtype(sequences.dtype, np.number):
        raise ValueError(
            f"{dataset_name} sequences must contain numeric values."
        )

    if not np.all(np.isfinite(sequences)):
        raise ValueError(
            f"{dataset_name} sequences must contain only finite values."
        )


def _validate_feature_compatibility(
    reference_dataset: ForecastingDataset,
    dataset: ForecastingDataset,
    dataset_name: str,
) -> None:
    """
    Validate that two forecasting datasets use the same feature contract.

    Args:
        reference_dataset:
            Dataset whose feature contract is considered authoritative.

        dataset:
            Dataset being compared against the reference.

        dataset_name:
            Human-readable name for the dataset being validated.

    Raises:
        ValueError:
            If feature names, context window, or feature dimensions differ.
    """

    if tuple(dataset.feature_names) != tuple(
        reference_dataset.feature_names
    ):
        raise ValueError(
            f"{dataset_name} feature names do not match the training "
            "feature contract."
        )

    if dataset.context_window != reference_dataset.context_window:
        raise ValueError(
            f"{dataset_name} context window does not match the training "
            "dataset."
        )

    if dataset.num_features != reference_dataset.num_features:
        raise ValueError(
            f"{dataset_name} feature count does not match the training "
            "dataset."
        )


def _validate_dtype(dtype: str) -> None:
    """
    Validate the requested NumPy dtype.

    Args:
        dtype:
            NumPy dtype string.

    Raises:
        TypeError:
            If the dtype cannot be interpreted by NumPy.
    """

    try:
        np.dtype(dtype)
    except (TypeError, ValueError) as exc:
        raise TypeError(
            f"Invalid normalization dtype: {dtype!r}."
        ) from exc


# ============================================================================
# Standard Scaler
# ============================================================================

@dataclass(frozen=True)
class StandardScaler:
    """
    Lightweight feature-wise standard scaler.

    The scaler stores one mean and one standard deviation for each feature.

    Statistics are fitted from the training dataset only.

    Attributes:
        means:
            Per-feature training means.

        scales:
            Per-feature standard deviations.

        feature_names:
            Ordered feature names used during fitting.

        fitted:
            Indicates whether valid training statistics are available.
    """

    means: np.ndarray
    scales: np.ndarray
    feature_names: Tuple[str, ...]

    def __post_init__(self) -> None:
        """Validate stored scaler statistics."""

        means = np.asarray(self.means)
        scales = np.asarray(self.scales)

        if means.ndim != 1:
            raise ValueError(
                "Scaler means must be one-dimensional."
            )

        if scales.ndim != 1:
            raise ValueError(
                "Scaler scales must be one-dimensional."
            )

        if len(means) != len(scales):
            raise ValueError(
                "Scaler means and scales must contain the same number "
                "of features."
            )

        if len(means) != len(self.feature_names):
            raise ValueError(
                "Scaler statistics must match the number of feature names."
            )

        if not self.feature_names:
            raise ValueError(
                "Scaler must contain at least one feature."
            )

        if len(set(self.feature_names)) != len(self.feature_names):
            raise ValueError(
                "Scaler feature names must be unique."
            )

        if not np.issubdtype(means.dtype, np.number):
            raise ValueError(
                "Scaler means must be numeric."
            )

        if not np.issubdtype(scales.dtype, np.number):
            raise ValueError(
                "Scaler scales must be numeric."
            )

        if not np.all(np.isfinite(means)):
            raise ValueError(
                "Scaler means must contain only finite values."
            )

        if not np.all(np.isfinite(scales)):
            raise ValueError(
                "Scaler scales must contain only finite values."
            )

        if np.any(scales <= 0):
            raise ValueError(
                "Scaler scales must be greater than zero."
            )

    @property
    def num_features(self) -> int:
        """Return the number of features represented by the scaler."""

        return int(len(self.feature_names))

    @property
    def fitted(self) -> bool:
        """Return True because this object represents fitted statistics."""

        return True

    def transform(
        self,
        sequences: np.ndarray,
        feature_names: Tuple[str, ...],
    ) -> np.ndarray:
        """
        Transform feature sequences using the stored training statistics.

        Args:
            sequences:
                Three-dimensional feature tensor with shape:

                    (num_samples, context_window, num_features)

            feature_names:
                Ordered feature names corresponding to the final axis.

        Returns:
            Standardized feature tensor.

        Raises:
            ValueError:
                If the input feature contract does not match the scaler.
        """

        sequences = np.asarray(sequences)

        _validate_sequence_values(
            sequences,
            "Input",
        )

        if sequences.shape[2] != self.num_features:
            raise ValueError(
                "Input feature dimension does not match the fitted scaler."
            )

        if tuple(feature_names) != tuple(self.feature_names):
            raise ValueError(
                "Input feature names do not match the fitted scaler."
            )

        means = np.asarray(
            self.means,
            dtype=sequences.dtype,
        )

        scales = np.asarray(
            self.scales,
            dtype=sequences.dtype,
        )

        transformed = (
            sequences - means.reshape(1, 1, -1)
        ) / scales.reshape(1, 1, -1)

        if not np.all(np.isfinite(transformed)):
            raise ValueError(
                "Normalization produced non-finite values."
            )

        return transformed


# ============================================================================
# Normalization Result
# ============================================================================

@dataclass(frozen=True)
class NormalizedForecastingDataset:
    """
    Result of normalizing a ForecastingDataset.

    This wrapper retains the normalized dataset and the scaler used to
    transform it.

    Attributes:
        dataset:
            ForecastingDataset containing normalized feature sequences.

        scaler:
            Training-fitted scaler used for the transformation.

        normalized:
            Indicates whether feature normalization was applied.
    """

    dataset: ForecastingDataset
    scaler: Optional[StandardScaler]
    normalized: bool


# ============================================================================
# Forecasting Dataset Normalizer
# ============================================================================

class ForecastingDatasetNormalizer:
    """
    Normalize forecasting features using training-only statistics.

    The normalizer has two distinct operations:

        1. fit()
            Learn normalization statistics from training data.

        2. transform()
            Apply already-fitted statistics to a forecasting dataset.

    This separation is intentional.

    Validation and test data must never be passed to fit().
    """

    def __init__(
        self,
        config: Optional[NormalizerConfig] = None,
    ) -> None:
        """
        Initialize the forecasting dataset normalizer.

        Args:
            config:
                Optional normalization configuration.
        """

        self.config = config or NormalizerConfig()

        self._method = _validate_method(
            self.config.method
        )

        _validate_dtype(
            self.config.dtype
        )

        if not self.config.fit_on_training_only:
            raise ValueError(
                "fit_on_training_only must be True for the forecasting "
                "pipeline."
            )

        self._scaler: Optional[StandardScaler] = None
        self._training_feature_names: Optional[Tuple[str, ...]] = None
        self._fitted = False

    # ------------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------------

    @property
    def scaler(self) -> Optional[StandardScaler]:
        """
        Return the fitted scaler.

        Returns:
            Fitted StandardScaler, or None if fit() has not been called.
        """

        return self._scaler

    @property
    def fitted(self) -> bool:
        """Return whether the fit operation has completed successfully."""

        return self._fitted 

    @property
    def method(self) -> str:
        """Return the configured normalization method."""

        return self._method

    @property
    def feature_names(self) -> Optional[Tuple[str, ...]]:
        """Return the feature contract learned during fitting."""

        return self._training_feature_names

    # ------------------------------------------------------------------------
    # Fit
    # ------------------------------------------------------------------------

    def fit(
        self,
        training_dataset: ForecastingDataset,
    ) -> "ForecastingDatasetNormalizer":
        """
        Fit normalization statistics using training data only.

        The context-window and sample dimensions are flattened temporarily
        so that one statistic is learned independently for each feature.

        For input shape:

            (samples, context_window, features)

        fitting operates conceptually on:

            (samples * context_window, features)

        Validation and test datasets are intentionally not accepted here.

        When feature normalization is disabled, no scaler is created, but
        the normalizer is still marked as fitted so that the fitted state
        consistently represents completion of the fit operation.

        Args:
            training_dataset:
                Training-only forecasting dataset.

        Returns:
            This normalizer instance.

        Raises:
            TypeError:
                If training_dataset is not a ForecastingDataset.

            ValueError:
                If the training data is invalid or contains non-finite
                values.
        """

        _validate_dataset(
            training_dataset,
            "training_dataset",
        )

        _validate_sequence_values(
            training_dataset.sequences,
            "Training",
        )

        feature_names = tuple(
            training_dataset.feature_names
        )

        if self.config.normalize_features:
            flattened = np.asarray(
                training_dataset.sequences,
                dtype=self.config.dtype,
            ).reshape(
                -1,
                training_dataset.num_features,
            )

            if self._method == "standard":
                means = np.mean(
                    flattened,
                    axis=0,
                    dtype=np.float64,
                )

                scales = np.std(
                    flattened,
                    axis=0,
                    dtype=np.float64,
                )

                # Constant features have zero variance. They are assigned
                # a unit scale so that their normalized values become zero
                # instead of producing division-by-zero values.
                scales = np.where(
                    scales == 0.0,
                    1.0,
                    scales,
                )

            else:
                raise ValueError(
                    f"Unsupported normalization method: {self._method!r}."
                )

            self._scaler = StandardScaler(
                means=np.asarray(
                    means,
                    dtype=self.config.dtype,
                ),
                scales=np.asarray(
                    scales,
                    dtype=self.config.dtype,
                ),
                feature_names=feature_names,
            )

        else:
            # Normalization is intentionally disabled. No scaler is
            # required, but fit() has still successfully completed.
            self._scaler = None

        self._training_feature_names = feature_names
        self._fitted = True

        return self

    # ------------------------------------------------------------------------
    # Transform
    # ------------------------------------------------------------------------

    def transform(
        self,
        dataset: ForecastingDataset,
    ) -> ForecastingDataset:
        """
        Transform a forecasting dataset using fitted training statistics.

        This method does not fit or modify the scaler.

        Therefore it can safely be called for:

            - training data
            - validation data
            - test data
            - later inference data

        provided that the feature contract is compatible.

        Args:
            dataset:
                Forecasting dataset to transform.

        Returns:
            ForecastingDataset with normalized feature sequences.

        Raises:
            RuntimeError:
                If normalization is enabled but fit() has not been called.

            TypeError:
                If dataset is not a ForecastingDataset.

            ValueError:
                If dataset's feature contract is incompatible with the
                training feature contract.
        """

        _validate_dataset(
            dataset,
            "dataset",
        )

        _validate_sequence_values(
            dataset.sequences,
            "Dataset",
        )

        if self._training_feature_names is None:
            raise RuntimeError(
                "The normalizer has not been fitted. "
                "Call fit() using the training dataset first."
            )

        if tuple(dataset.feature_names) != tuple(
            self._training_feature_names
        ):
            raise ValueError(
                "Dataset feature names do not match the fitted training "
                "feature contract."
            )

        if self.config.normalize_features:
            if self._scaler is None:
                raise RuntimeError(
                    "The normalizer has no fitted scaler. "
                    "Call fit() using the training dataset first."
                )

            normalized_sequences = self._scaler.transform(
                np.asarray(
                    dataset.sequences,
                    dtype=self.config.dtype,
                ),
                tuple(dataset.feature_names),
            )

        else:
            normalized_sequences = np.asarray(
                dataset.sequences,
                dtype=self.config.dtype,
            )

        if self.config.copy_arrays:
            normalized_sequences = normalized_sequences.copy()

        return ForecastingDataset(
            sequences=normalized_sequences,
            targets=np.asarray(
                dataset.targets,
                dtype=self.config.dtype,
            ).copy(),
            timestamps=np.asarray(
                dataset.timestamps,
                dtype=self.config.dtype,
            ).copy(),
            feature_names=tuple(dataset.feature_names),
            context_window=dataset.context_window,
            forecast_horizon=dataset.forecast_horizon,
            target_name=dataset.target_name,
            price_column=dataset.price_column,
        )

    # ------------------------------------------------------------------------
    # Fit and Transform Training Data
    # ------------------------------------------------------------------------

    def fit_transform(
        self,
        training_dataset: ForecastingDataset,
    ) -> ForecastingDataset:
        """
        Fit the scaler on training data and transform that same training data.

        This is the intended operation for the training partition.

        Args:
            training_dataset:
                Training-only forecasting dataset.

        Returns:
            Normalized training dataset.
        """

        self.fit(
            training_dataset
        )

        return self.transform(
            training_dataset
        )

    # ------------------------------------------------------------------------
    # Transform Split
    # ------------------------------------------------------------------------

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
        Fit on training data and transform all three dataset partitions.

        The scaler is fitted exactly once using the training dataset.

        Validation and test datasets are only transformed. Their statistics
        never influence the fitted scaler.

        Args:
            training_dataset:
                Chronological training partition.

            validation_dataset:
                Chronological validation partition.

            test_dataset:
                Chronological test partition.

        Returns:
            Tuple containing:

                normalized_training,
                normalized_validation,
                normalized_test
        """

        _validate_dataset(
            training_dataset,
            "training_dataset",
        )

        _validate_dataset(
            validation_dataset,
            "validation_dataset",
        )

        _validate_dataset(
            test_dataset,
            "test_dataset",
        )

        _validate_feature_compatibility(
            training_dataset,
            validation_dataset,
            "Validation dataset",
        )

        _validate_feature_compatibility(
            training_dataset,
            test_dataset,
            "Test dataset",
        )

        self.fit(
            training_dataset
        )

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

    # ------------------------------------------------------------------------
    # Configuration Integration
    # ------------------------------------------------------------------------

    @classmethod
    def from_forecasting_config(
        cls,
        forecasting_config: ForecastingConfig,
    ) -> "ForecastingDatasetNormalizer":
        """
        Create a normalizer from ForecastingConfig.

        Args:
            forecasting_config:
                Existing forecasting configuration.

        Returns:
            Configured ForecastingDatasetNormalizer.

        Raises:
            TypeError:
                If the supplied configuration is not a ForecastingConfig.
        """

        if not isinstance(
            forecasting_config,
            ForecastingConfig,
        ):
            raise TypeError(
                "forecasting_config must be an instance of "
                "ForecastingConfig."
            )

        return cls(
            config=NormalizerConfig(
                method=forecasting_config.normalization_method,
                normalize_features=(
                    forecasting_config.normalize_features
                ),
                fit_on_training_only=(
                    forecasting_config.scaler_fit_on_training_only
                ),
            )
        )


# ============================================================================
# Convenience Functions
# ============================================================================

def fit_normalizer(
    training_dataset: ForecastingDataset,
    config: Optional[NormalizerConfig] = None,
) -> ForecastingDatasetNormalizer:
    """
    Fit a forecasting normalizer using training data only.

    Args:
        training_dataset:
            Training forecasting dataset.

        config:
            Optional normalization configuration.

    Returns:
        Fitted ForecastingDatasetNormalizer.
    """

    normalizer = ForecastingDatasetNormalizer(
        config=config,
    )

    normalizer.fit(
        training_dataset
    )

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
    Fit normalization on training data and transform all partitions.

    Args:
        training_dataset:
            Training forecasting dataset.

        validation_dataset:
            Validation forecasting dataset.

        test_dataset:
            Test forecasting dataset.

        config:
            Optional normalization configuration.

    Returns:
        Tuple containing normalized training, validation, and test datasets.
    """

    normalizer = ForecastingDatasetNormalizer(
        config=config,
    )

    return normalizer.transform_splits(
        training_dataset=training_dataset,
        validation_dataset=validation_dataset,
        test_dataset=test_dataset,
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
    Normalize forecasting partitions using ForecastingConfig.

    The scaler is fitted exclusively on the training dataset.

    Args:
        training_dataset:
            Training forecasting dataset.

        validation_dataset:
            Validation forecasting dataset.

        test_dataset:
            Test forecasting dataset.

        forecasting_config:
            Existing forecasting configuration.

    Returns:
        Tuple containing normalized training, validation, and test datasets.
    """

    normalizer = ForecastingDatasetNormalizer.from_forecasting_config(
        forecasting_config
    )

    return normalizer.transform_splits(
        training_dataset=training_dataset,
        validation_dataset=validation_dataset,
        test_dataset=test_dataset,
    )


# ============================================================================
# Public Exports
# ============================================================================

__all__ = [
    "DEFAULT_NORMALIZATION_METHOD",
    "SUPPORTED_NORMALIZATION_METHODS",
    "NormalizerConfig",
    "NormalizedForecastingDataset",
    "StandardScaler",
    "ForecastingDatasetNormalizer",
    "fit_normalizer",
    "normalize_forecasting_splits",
    "normalize_from_forecasting_config",
]

