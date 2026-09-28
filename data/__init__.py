"""
SmartFlow Data Package.

This package provides the shared data contracts and data-preparation
interfaces used throughout SmartFlow.

The package is responsible for exposing stable interfaces between:

    Market Data
        ↓
    Feature Extraction
        ↓
    Chronological Feature History
        ↓
    Rolling Sequence Construction
        ↓
    Target Construction
        ↓
    Forecasting Dataset Alignment
        ↓
    Chronological Train / Validation / Test Splitting
        ↓
    Training-Only Normalization
        ↓
    Forecasting Model Integration

The package intentionally keeps implementation responsibilities separated.

Shared contracts live in:

    data.contracts

Feature-history construction lives in:

    data.feature_history

Rolling sequence construction lives in:

    data.sequence_builder

Target construction lives in:

    data.target_builder

Sequence/target alignment lives in:

    data.forecasting_dataset

Chronological dataset splitting lives in:

    data.splitter

Training-only normalization lives in:

    data.normalizer

This package does not:

    - train machine-learning models
    - perform model inference
    - make execution decisions
    - submit trading orders
    - simulate fills
    - perform backtesting
    - implement the NVIDIA forecasting model

Those responsibilities belong to their respective modules.
"""


# =============================================================================
# Core Data Contracts
# =============================================================================

from data.contracts import (
    # -------------------------------------------------------------------------
    # Enums
    # -------------------------------------------------------------------------
    ModelStatus,
    OrderSide,
    OrderType,
    ExecutionStatus,

    # -------------------------------------------------------------------------
    # Market contracts
    # -------------------------------------------------------------------------
    MarketSnapshot,

    # -------------------------------------------------------------------------
    # Order / execution contracts
    # -------------------------------------------------------------------------
    Order,
    ExecutionResult,

    # -------------------------------------------------------------------------
    # Generic machine-learning contracts
    # -------------------------------------------------------------------------
    PredictionResult,

    # -------------------------------------------------------------------------
    # NVIDIA forecasting constants
    # -------------------------------------------------------------------------
    NVIDIA_FORECAST_FEATURES,
    DEFAULT_FORECAST_CONTEXT_WINDOW,
    DEFAULT_FORECAST_HORIZON,
    DEFAULT_FORECAST_TARGET_NAME,
    DEFAULT_FORECAST_PRICE_COLUMN,

    # -------------------------------------------------------------------------
    # Forecasting contract validation
    # -------------------------------------------------------------------------
    validate_nvidia_feature_contract,
    validate_forecasting_dataset_contract,

    # -------------------------------------------------------------------------
    # Forecasting model contracts
    # -------------------------------------------------------------------------
    ForecastInput,
    ForecastResult,

    # -------------------------------------------------------------------------
    # Forecasting dataset contracts
    # -------------------------------------------------------------------------
    SequenceDataset,
    TargetDataset,
    ForecastingDataset,
)


# =============================================================================
# Feature History API
# =============================================================================

from data.feature_history import (
    FeatureHistoryBuilder,
    FeatureHistoryConfig,
    build_feature_history,
    get_feature_names,
    get_required_columns,
)


# =============================================================================
# Sequence Construction API
# =============================================================================

from data.sequence_builder import (
    SequenceBuilder,
    SequenceBuilderConfig,
    build_sequences,
)


# =============================================================================
# Target Construction API
# =============================================================================

from data.target_builder import (
    TargetBuilder,
    TargetBuilderConfig,
    build_targets,
    build_targets_from_forecasting_config,
)


# =============================================================================
# Forecasting Dataset Alignment API
# =============================================================================

from data.forecasting_dataset import (
    ForecastingDatasetBuilder,
    ForecastingDatasetConfig,
    align_sequence_targets,
    build_forecasting_dataset,
)


# =============================================================================
# Chronological Dataset Splitting API
# =============================================================================

from data.splitter import (
    ChronologicalDatasetSplit,
    ChronologicalSplitter,
    SplitterConfig,
    split_forecasting_dataset,
    split_from_forecasting_config,
)


# =============================================================================
# Training-Only Normalization API
# =============================================================================

from data.normalizer import (
    ForecastingDatasetNormalizer,
    NormalizedForecastingDataset,
    NormalizerConfig,
    StandardScaler,
    fit_normalizer,
    normalize_forecasting_splits,
    normalize_from_forecasting_config,
)


# =============================================================================
# Public Package API
# =============================================================================

__all__ = [
    # -------------------------------------------------------------------------
    # Core enums
    # -------------------------------------------------------------------------
    "OrderSide",
    "OrderType",
    "ExecutionStatus",
    "ModelStatus",

    # -------------------------------------------------------------------------
    # Market contracts
    # -------------------------------------------------------------------------
    "MarketSnapshot",

    # -------------------------------------------------------------------------
    # Order / execution contracts
    # -------------------------------------------------------------------------
    "Order",
    "ExecutionResult",

    # -------------------------------------------------------------------------
    # Generic machine-learning contracts
    # -------------------------------------------------------------------------
    "PredictionResult",

    # -------------------------------------------------------------------------
    # NVIDIA forecasting constants
    # -------------------------------------------------------------------------
    "NVIDIA_FORECAST_FEATURES",
    "DEFAULT_FORECAST_CONTEXT_WINDOW",
    "DEFAULT_FORECAST_HORIZON",
    "DEFAULT_FORECAST_TARGET_NAME",
    "DEFAULT_FORECAST_PRICE_COLUMN",

    # -------------------------------------------------------------------------
    # Forecasting contract validation
    # -------------------------------------------------------------------------
    "validate_nvidia_feature_contract",
    "validate_forecasting_dataset_contract",

    # -------------------------------------------------------------------------
    # Forecasting model contracts
    # -------------------------------------------------------------------------
    "ForecastInput",
    "ForecastResult",

    # -------------------------------------------------------------------------
    # Forecasting dataset contracts
    # -------------------------------------------------------------------------
    "SequenceDataset",
    "TargetDataset",
    "ForecastingDataset",

    # -------------------------------------------------------------------------
    # Feature history
    # -------------------------------------------------------------------------
    "FeatureHistoryConfig",
    "FeatureHistoryBuilder",
    "build_feature_history",
    "get_feature_names",
    "get_required_columns",

    # -------------------------------------------------------------------------
    # Sequence construction
    # -------------------------------------------------------------------------
    "SequenceBuilderConfig",
    "SequenceBuilder",
    "build_sequences",

    # -------------------------------------------------------------------------
    # Target construction
    # -------------------------------------------------------------------------
    "TargetBuilderConfig",
    "TargetBuilder",
    "build_targets",
    "build_targets_from_forecasting_config",

    # -------------------------------------------------------------------------
    # Forecasting dataset alignment
    # -------------------------------------------------------------------------
    "ForecastingDatasetConfig",
    "ForecastingDatasetBuilder",
    "build_forecasting_dataset",
    "align_sequence_targets",

    # -------------------------------------------------------------------------
    # Chronological splitting
    # -------------------------------------------------------------------------
    "SplitterConfig",
    "ChronologicalDatasetSplit",
    "ChronologicalSplitter",
    "split_forecasting_dataset",
    "split_from_forecasting_config",

    # -------------------------------------------------------------------------
    # Normalization
    # -------------------------------------------------------------------------
    "NormalizerConfig",
    "StandardScaler",
    "NormalizedForecastingDataset",
    "ForecastingDatasetNormalizer",
    "fit_normalizer",
    "normalize_forecasting_splits",
    "normalize_from_forecasting_config",
]

