"""
Data Package Public API.

Provides the public data contracts, forecasting contracts, and feature-history
interfaces used throughout the SmartFlow system.

This package acts as the central import surface for shared data-layer objects.
Keeping these exports here allows other modules to import commonly used data
structures from ``data`` without depending on the internal organization of
individual implementation modules.

Public Components
-----------------
Contracts:
    Core market, order, trade, execution, prediction, and backtest contracts.

Forecasting:
    ForecastInput and ForecastResult define the interface between the
    chronological feature pipeline and the NVIDIA forecasting model.

Feature History:
    FeatureHistoryBuilder and build_feature_history provide the Phase 2
    pipeline for converting MarketSnapshot objects into a unified,
    chronological feature DataFrame.

Design Principles
-----------------
- Keep shared data contracts centralized.
- Keep forecasting contracts separate from adverse-selection model contracts.
- Preserve backward compatibility with existing imports.
- Expose stable public interfaces while allowing internal implementation
  details to evolve.
- Avoid placing feature-extraction implementation directly in this module.
"""

# ---------------------------------------------------------------------------
# Core Data Contracts
# ---------------------------------------------------------------------------

from data.contracts import (
    OrderSide,
    OrderType,
    OrderStatus,
    LiquidityType,
    RiskCategory,
    ExecutionMode,
    ModelStatus,
    Trade,
    MarketSnapshot,
    Order,
    Fill,
    PredictionResult,
    ExecutionDecision,
    ExecutionResult,
    BacktestResult,
    ForecastInput,
    ForecastResult,
)


# ---------------------------------------------------------------------------
# Feature History API
# ---------------------------------------------------------------------------
#
# These imports are intentionally kept after the core contracts. The feature
# history module depends on MarketSnapshot and therefore belongs conceptually
# above the lower-level contract layer.
#
# The imports are wrapped in a guarded block so that the core data contracts
# remain importable during incremental development if feature_history.py has
# not yet been created.
# ---------------------------------------------------------------------------

try:
    from data.feature_history import (
        FeatureHistoryBuilder,
        build_feature_history,
    )

    _FEATURE_HISTORY_EXPORTS = [
        "FeatureHistoryBuilder",
        "build_feature_history",
    ]

except ImportError:
    _FEATURE_HISTORY_EXPORTS = []


# ---------------------------------------------------------------------------
# Public Package API
# ---------------------------------------------------------------------------

__all__ = [
    # Core enums
    "OrderSide",
    "OrderType",
    "OrderStatus",
    "LiquidityType",
    "RiskCategory",
    "ExecutionMode",
    "ModelStatus",

    # Core market and trading contracts
    "Trade",
    "MarketSnapshot",
    "Order",
    "Fill",

    # Existing prediction and execution contracts
    "PredictionResult",
    "ExecutionDecision",
    "ExecutionResult",
    "BacktestResult",

    # NVIDIA forecasting contracts
    "ForecastInput",
    "ForecastResult",

    # Phase 2 feature-history API
    *_FEATURE_HISTORY_EXPORTS,
]

