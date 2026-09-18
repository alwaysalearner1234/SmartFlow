"""
Public API for the SmartFlow data package.

The ``data`` package provides the shared contracts and data-preparation
interfaces used throughout SmartFlow.

Public API Categories
---------------------

Core Contracts
    Market snapshots, trades, orders, fills, execution decisions, execution
    results, routing results, and backtest results.

Forecasting Contracts
    ForecastInput represents one model inference window.
    ForecastResult represents one model forecast.

Sequence Contracts
    SequenceDataset represents the complete collection of rolling forecasting
    windows produced from chronological feature history.

Feature History
    FeatureHistoryBuilder and build_feature_history convert MarketSnapshot
    objects into a unified chronological feature DataFrame.

Sequence Construction
    SequenceBuilder and build_sequences convert chronological feature history
    into fixed-length forecasting sequences.

Architecture
------------
The intended forecasting data flow is:

    MarketSnapshot
        |
        v
    Feature Extraction
        |
        v
    Feature History
        |
        v
    SequenceDataset
        |
        v
    ForecastInput
        |
        v
    NVIDIA Forecasting Model
        |
        v
    ForecastResult

The package exports shared interfaces only. Feature-extraction and sequence
construction implementations remain in their dedicated modules.
"""

# ============================================================================
# Core Data Contracts
# ============================================================================

from data.contracts import (
    # Enums
    ExecutionMode,
    LiquidityType,
    MarketRegime,
    ModelStatus,
    OrderSide,
    OrderStatus,
    OrderType,
    RiskCategory,

    # ML / model contracts
    ForecastInput,
    ForecastResult,
    ModelSystemStatus,
    PredictionResult,
    SequenceDataset,

    # Market contracts
    MarketSnapshot,
    Trade,

    # Routing contracts
    VenueQuote,
    VenueRoutingResult,

    # Order / execution contracts
    Fill,
    Order,
    ExecutionDecision,
    ExecutionResult,

    # Backtesting contracts
    BacktestResult,
)


# ============================================================================
# Feature History API
# ============================================================================

from data.feature_history import (
    FeatureHistoryBuilder,
    FeatureHistoryConfig,
    build_feature_history,
    get_feature_names,
    get_required_columns,
)


# ============================================================================
# Sequence Construction API
# ============================================================================

from data.sequence_builder import (
    SequenceBuilder,
    SequenceBuilderConfig,
    build_sequences,
)


# ============================================================================
# Public Package API
# ============================================================================

__all__ = [
    # ------------------------------------------------------------------------
    # Core enums
    # ------------------------------------------------------------------------
    "OrderSide",
    "OrderType",
    "OrderStatus",
    "LiquidityType",
    "RiskCategory",
    "ExecutionMode",
    "ModelStatus",
    "MarketRegime",

    # ------------------------------------------------------------------------
    # Model / ML contracts
    # ------------------------------------------------------------------------
    "ModelSystemStatus",
    "PredictionResult",
    "ForecastInput",
    "ForecastResult",
    "SequenceDataset",

    # ------------------------------------------------------------------------
    # Market contracts
    # ------------------------------------------------------------------------
    "Trade",
    "MarketSnapshot",

    # ------------------------------------------------------------------------
    # Order / execution contracts
    # ------------------------------------------------------------------------
    "Order",
    "Fill",
    "ExecutionDecision",
    "ExecutionResult",

    # ------------------------------------------------------------------------
    # Routing contracts
    # ------------------------------------------------------------------------
    "VenueQuote",
    "VenueRoutingResult",

    # ------------------------------------------------------------------------
    # Backtesting contracts
    # ------------------------------------------------------------------------
    "BacktestResult",

    # ------------------------------------------------------------------------
    # Phase 2 feature-history API
    # ------------------------------------------------------------------------
    "FeatureHistoryConfig",
    "FeatureHistoryBuilder",
    "build_feature_history",
    "get_feature_names",
    "get_required_columns",

    # ------------------------------------------------------------------------
    # Phase 3 sequence-construction API
    # ------------------------------------------------------------------------
    "SequenceBuilderConfig",
    "SequenceBuilder",
    "build_sequences",
]

