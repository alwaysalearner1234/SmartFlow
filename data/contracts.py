"""
SmartFlow Data Contracts.

This module defines the shared data contracts used throughout the SmartFlow
market-data, forecasting, execution, simulation, routing, and backtesting
pipeline.

The contracts are intentionally lightweight. They define stable interfaces
between pipeline stages and validate important structural invariants while
leaving implementation logic to the corresponding modules.

Core forecasting pipeline:

    MarketSnapshot
        ↓
    Feature Extraction
        ↓
    Chronological Feature History
        ↓
    SequenceDataset + TargetDataset
        ↓
    ForecastingDataset
        ↓
    ChronologicalDatasetSplit
        ↓
    Training-Only Normalization
        ↓
    ForecastInput
        ↓
    NVIDIA Forecasting Model
        ↓
    ForecastResult

Current NVIDIA forecasting contract:

    context_window = 20
    num_features   = 14
    forecast_horizon = 5

Therefore:

    single model input shape = (20, 14)

and a complete forecasting batch has shape:

    (N, 20, 14)

This module does not:

    - generate market data
    - calculate market features
    - build rolling sequences
    - construct forecasting targets
    - split datasets
    - normalize datasets
    - train forecasting models
    - perform inference
    - make execution decisions
    - submit orders
    - simulate fills
    - perform backtesting

Those responsibilities belong to the corresponding pipeline modules.
"""

from __future__ import annotations

# =============================================================================
# Standard Library Imports
# =============================================================================

from dataclasses import dataclass, field
from enum import Enum
from math import isfinite
from numbers import Real
from typing import Any, Dict, List, Optional, Sequence, Tuple

# =============================================================================
# Third-Party Imports
# =============================================================================

import pandas as pd


# =============================================================================
# Shared Validation Helpers
# =============================================================================


def _validate_positive_integer(
    value: int,
    field_name: str,
) -> None:
    """
    Validate that a value is a positive integer.
    """

    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(
            f"{field_name} must be an integer."
        )

    if value <= 0:
        raise ValueError(
            f"{field_name} must be greater than zero."
        )


def _validate_nonempty_string(
    value: str,
    field_name: str,
) -> None:
    """
    Validate that a value is a non-empty string.
    """

    if not isinstance(value, str):
        raise TypeError(
            f"{field_name} must be a string."
        )

    if not value.strip():
        raise ValueError(
            f"{field_name} must not be empty."
        )


def _validate_unique_strings(
    values: Sequence[str],
    field_name: str,
) -> None:
    """
    Validate a non-empty sequence of unique strings.
    """

    if not values:
        raise ValueError(
            f"{field_name} must contain at least one value."
        )

    for index, value in enumerate(values):
        if not isinstance(value, str):
            raise TypeError(
                f"{field_name}[{index}] must be a string."
            )

        if not value.strip():
            raise ValueError(
                f"{field_name}[{index}] must not be empty."
            )

    if len(set(values)) != len(values):
        raise ValueError(
            f"{field_name} must contain unique values."
        )


def _validate_numeric_value(
    value: Any,
    field_name: str,
) -> None:
    """
    Validate a finite real-valued scalar.
    """

    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(
            f"{field_name} must be numeric."
        )

    if not isfinite(float(value)):
        raise ValueError(
            f"{field_name} must be finite."
        )


def _validate_timestamp_sequence(
    timestamps: Sequence[float],
    field_name: str = "timestamps",
    *,
    require_unique: bool = True,
) -> None:
    """
    Validate a chronological timestamp sequence.
    """

    if timestamps is None:
        raise ValueError(
            f"{field_name} cannot be None."
        )

    if len(timestamps) == 0:
        raise ValueError(
            f"{field_name} must contain at least one timestamp."
        )

    for index, timestamp in enumerate(timestamps):
        _validate_numeric_value(
            timestamp,
            f"{field_name}[{index}]",
        )

    for index in range(1, len(timestamps)):
        previous = float(timestamps[index - 1])
        current = float(timestamps[index])

        if current < previous:
            raise ValueError(
                f"{field_name} must be in chronological order."
            )

        if require_unique and current == previous:
            raise ValueError(
                f"{field_name} must contain unique timestamps."
            )


def _validate_numeric_values(
    values: Sequence[Any],
    field_name: str,
    *,
    require_non_empty: bool = True,
) -> None:
    """
    Validate a collection of finite numeric values.
    """

    if values is None:
        raise ValueError(
            f"{field_name} cannot be None."
        )

    if require_non_empty and len(values) == 0:
        raise ValueError(
            f"{field_name} cannot be empty."
        )

    for index, value in enumerate(values):
        _validate_numeric_value(
            value,
            f"{field_name}[{index}]",
        )


def _validate_feature_rows(
    feature_sequence: Sequence[Sequence[float]],
    num_features: int,
    field_name: str = "feature_sequence",
) -> None:
    """
    Validate the rows of a model feature sequence.
    """

    if not feature_sequence:
        raise ValueError(
            f"{field_name} must contain at least one timestep."
        )

    for row_index, row in enumerate(feature_sequence):
        if not isinstance(row, (list, tuple)):
            raise TypeError(
                f"{field_name}[{row_index}] must be a list or tuple."
            )

        if len(row) != num_features:
            raise ValueError(
                f"{field_name}[{row_index}] must contain exactly "
                f"{num_features} features."
            )

        for feature_index, value in enumerate(row):
            _validate_numeric_value(
                value,
                (
                    f"{field_name}[{row_index}]"
                    f"[{feature_index}]"
                ),
            )


# =============================================================================
# Core Enumerations
# =============================================================================


class OrderSide(str, Enum):
    """
    Supported order directions.
    """

    BUY = "BUY"
    SELL = "SELL"


class OrderType(str, Enum):
    """
    Supported order types.
    """

    MARKET = "MARKET"
    LIMIT = "LIMIT"


class OrderStatus(str, Enum):
    """
    Lifecycle status of an order.
    """

    PENDING = "PENDING"
    SUBMITTED = "SUBMITTED"
    PARTIALLY_FILLED = "PARTIALLY_FILLED"
    FILLED = "FILLED"
    CANCELLED = "CANCELLED"
    REJECTED = "REJECTED"


# Backward-compatible execution-status name used by the newer execution
# contract while retaining OrderStatus for the original execution layer.
ExecutionStatus = OrderStatus


class LiquidityType(str, Enum):
    """
    Whether an execution provided or consumed liquidity.
    """

    MAKER = "MAKER"
    TAKER = "TAKER"


class RiskCategory(str, Enum):
    """
    Categorical representation of adverse-selection risk.
    """

    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"


class ExecutionMode(str, Enum):
    """
    Typed execution action returned by the execution engine.
    """

    AUTO = "auto"
    HOLD = "hold"
    PASSIVE = "passive"
    AGGRESSIVE = "aggressive"


class ModelStatus(str, Enum):
    """
    Standardized model availability and prediction status values.

    The original execution/model layer uses TRAINED, FALLBACK,
    FALLBACK_HEURISTIC, and UNAVAILABLE.

    The forecasting layer also supports READY, SUCCESS, and FAILED so that
    model-facing forecast contracts remain compatible with their callers.
    """

    TRAINED = "TRAINED"
    FALLBACK = "FALLBACK"
    FALLBACK_HEURISTIC = "FALLBACK_HEURISTIC"
    UNAVAILABLE = "UNAVAILABLE"

    READY = "READY"
    SUCCESS = "SUCCESS"
    FAILED = "FAILED"


class MarketRegime(str, Enum):
    """
    Deterministic market-regime classifications.
    """

    CALM = "Calm"
    NORMAL = "Normal"
    VOLATILE = "Volatile"
    HIGHLY_VOLATILE = "Highly Volatile"
    ILLIQUID = "Illiquid"


# =============================================================================
# ML / Model System Contracts
# =============================================================================


@dataclass
class ModelSystemStatus:
    """
    Standardized model-system status.

    This contract communicates model availability without exposing model
    implementation details to downstream components.
    """

    is_loaded: bool
    model_name: str
    model_version: str
    status: ModelStatus
    status_label: str
    prediction_available: bool
    details: str
    metrics: Dict[str, Any] = field(default_factory=dict)
    feature_importances: Dict[str, float] = field(default_factory=dict)


@dataclass
class PredictionResult:
    """
    Output from the adverse-selection / passive-exposure risk model.

    probability estimates the likelihood of adverse selection associated
    with passive exposure. It is not a directional price forecast.
    """

    probability: float
    timestamp: float
    prediction_horizon: int
    model_name: str
    model_status: ModelStatus
    risk_category: RiskCategory = RiskCategory.MEDIUM
    feature_importances: Dict[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """
        Validate probability and derive the categorical risk level.
        """

        _validate_numeric_value(
            self.probability,
            "probability",
        )

        _validate_numeric_value(
            self.timestamp,
            "timestamp",
        )

        _validate_positive_integer(
            self.prediction_horizon,
            "prediction_horizon",
        )

        _validate_nonempty_string(
            self.model_name,
            "model_name",
        )

        if not 0.0 <= self.probability <= 1.0:
            raise ValueError(
                "probability must be between 0.0 and 1.0."
            )

        if self.probability < 0.35:
            self.risk_category = RiskCategory.LOW
        elif self.probability > 0.65:
            self.risk_category = RiskCategory.HIGH
        else:
            self.risk_category = RiskCategory.MEDIUM


# =============================================================================
# NVIDIA Forecasting Feature Contract
# =============================================================================


NVIDIA_FORECAST_FEATURES: Tuple[str, ...] = (
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


DEFAULT_FORECAST_CONTEXT_WINDOW = 20
DEFAULT_FORECAST_HORIZON = 5
DEFAULT_FORECAST_TARGET_NAME = "future_mid_price_return"
DEFAULT_FORECAST_PRICE_COLUMN = "mid_price"


def validate_nvidia_feature_contract(
    feature_names: Sequence[str],
) -> None:
    """
    Validate the canonical NVIDIA forecasting feature contract.

    Feature order is part of the model contract.
    """

    normalized = tuple(feature_names)

    if normalized != NVIDIA_FORECAST_FEATURES:
        raise ValueError(
            "NVIDIA forecasting feature contract mismatch. "
            f"Expected {NVIDIA_FORECAST_FEATURES}, "
            f"received {normalized}."
        )


# =============================================================================
# Forecast Input Contract
# =============================================================================


@dataclass
class ForecastInput:
    """
    Model-facing forecasting input contract.

    This represents one chronological context window.

    Current SmartFlow contract:

        context_window = 20
        num_features   = 14

    Therefore:

        feature_sequence.shape = (20, 14)
    """

    feature_sequence: List[List[float]]
    feature_names: List[str]
    timestamps: List[float]
    context_window: int
    forecast_horizon: int

    def __post_init__(self) -> None:
        """
        Validate the forecasting model input.
        """

        _validate_positive_integer(
            self.context_window,
            "context_window",
        )

        _validate_positive_integer(
            self.forecast_horizon,
            "forecast_horizon",
        )

        if len(self.feature_sequence) != self.context_window:
            raise ValueError(
                "feature_sequence length must equal context_window."
            )

        if len(self.timestamps) != self.context_window:
            raise ValueError(
                "timestamps length must equal context_window."
            )

        _validate_unique_strings(
            self.feature_names,
            "feature_names",
        )

        _validate_feature_rows(
            self.feature_sequence,
            len(self.feature_names),
        )

        _validate_timestamp_sequence(
            self.timestamps,
            require_unique=True,
        )

        if self.context_window == DEFAULT_FORECAST_CONTEXT_WINDOW:
            validate_nvidia_feature_contract(
                self.feature_names
            )

    @property
    def num_features(self) -> int:
        """
        Return the number of model input features.
        """

        return len(self.feature_names)

    @property
    def shape(self) -> Tuple[int, int]:
        """
        Return the per-sample model input shape.

        Returns:

            (context_window, num_features)
        """

        return (
            self.context_window,
            self.num_features,
        )

    @property
    def input_shape(self) -> Tuple[int, int]:
        """
        Return the model-facing input shape.
        """

        return self.shape

    @property
    def latest_timestamp(self) -> float:
        """
        Return the most recent timestamp in the context window.
        """

        return float(self.timestamps[-1])

    @property
    def earliest_timestamp(self) -> float:
        """
        Return the earliest timestamp in the context window.
        """

        return float(self.timestamps[0])


# =============================================================================
# Forecast Output Contract
# =============================================================================


@dataclass
class ForecastResult:
    """
    Model-facing forecasting output contract.

    Represents a short-horizon directional forecast.

    The primary target is:

        future_mid_price_return =
            (future_mid_price - current_mid_price)
            / current_mid_price

    expected_move_bps is:

        predicted_return × 10,000
    """

    timestamp: float
    forecast_horizon: int
    predicted_return: float
    model_name: str
    model_version: str
    model_status: ModelStatus
    confidence: Optional[float] = None
    predicted_direction: Optional[str] = None
    expected_move_bps: Optional[float] = None
    feature_names: List[str] = field(
        default_factory=lambda: list(NVIDIA_FORECAST_FEATURES)
    )
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """
        Validate and derive forecast information.
        """

        _validate_numeric_value(
            self.timestamp,
            "timestamp",
        )

        _validate_positive_integer(
            self.forecast_horizon,
            "forecast_horizon",
        )

        _validate_numeric_value(
            self.predicted_return,
            "predicted_return",
        )

        _validate_nonempty_string(
            self.model_name,
            "model_name",
        )

        _validate_nonempty_string(
            self.model_version,
            "model_version",
        )

        _validate_unique_strings(
            self.feature_names,
            "feature_names",
        )

        if self.confidence is not None:
            _validate_numeric_value(
                self.confidence,
                "confidence",
            )

            if not 0.0 <= self.confidence <= 1.0:
                raise ValueError(
                    "confidence must be between 0 and 1."
                )

        if self.predicted_direction is None:
            if self.predicted_return > 0:
                self.predicted_direction = "UP"
            elif self.predicted_return < 0:
                self.predicted_direction = "DOWN"
            else:
                self.predicted_direction = "FLAT"

        else:
            _validate_nonempty_string(
                self.predicted_direction,
                "predicted_direction",
            )

            self.predicted_direction = (
                self.predicted_direction.upper()
            )

            if self.predicted_direction not in {
                "UP",
                "DOWN",
                "FLAT",
            }:
                raise ValueError(
                    "predicted_direction must be UP, DOWN, or FLAT."
                )

        if self.expected_move_bps is None:
            self.expected_move_bps = (
                float(self.predicted_return) * 10_000.0
            )

        else:
            _validate_numeric_value(
                self.expected_move_bps,
                "expected_move_bps",
            )


# =============================================================================
# Market Data Contracts
# =============================================================================


@dataclass
class Trade:
    """
    Represents an individual trade print in the market.
    """

    timestamp: float
    trade_id: str
    price: float
    size: float
    side: OrderSide


@dataclass
class MarketSnapshot:
    """
    Atomic snapshot of the multi-level order book and recent trade state.

    Bids are expected to be sorted descending by price.

    Asks are expected to be sorted ascending by price.

    The first element of each list is therefore the best available quote.
    """

    timestamp: float
    sequence_id: int
    bids: List[Tuple[float, float]]
    asks: List[Tuple[float, float]]
    last_trade_price: Optional[float] = None
    last_trade_size: Optional[float] = None
    last_trade_side: Optional[OrderSide] = None
    recent_trades: List[Trade] = field(default_factory=list)

    @property
    def best_bid(self) -> float:
        """
        Return the best bid price.
        """

        return self.bids[0][0] if self.bids else 0.0

    @property
    def best_bid_size(self) -> float:
        """
        Return quantity available at the best bid.
        """

        return self.bids[0][1] if self.bids else 0.0

    @property
    def best_ask(self) -> float:
        """
        Return the best ask price.
        """

        return self.asks[0][0] if self.asks else 0.0

    @property
    def best_ask_size(self) -> float:
        """
        Return quantity available at the best ask.
        """

        return self.asks[0][1] if self.asks else 0.0

    @property
    def mid_price(self) -> float:
        """
        Return the midpoint between the best bid and ask.
        """

        if self.best_bid > 0 and self.best_ask > 0:
            return (
                self.best_bid + self.best_ask
            ) / 2.0

        return self.best_bid or self.best_ask or 0.0

    @property
    def spread(self) -> float:
        """
        Return the absolute bid-ask spread.
        """

        if self.best_bid > 0 and self.best_ask > 0:
            return max(
                0.0,
                self.best_ask - self.best_bid,
            )

        return 0.0

    @property
    def total_bid_depth(self) -> float:
        """
        Return total displayed bid depth.
        """

        return sum(
            size
            for _, size in self.bids
        )

    @property
    def total_ask_depth(self) -> float:
        """
        Return total displayed ask depth.
        """

        return sum(
            size
            for _, size in self.asks
        )


# =============================================================================
# Order / Execution Contracts
# =============================================================================


@dataclass
class Order:
    """
    Represents a trading order placed in the market simulator.
    """

    order_id: str
    timestamp: float
    side: OrderSide
    order_type: OrderType
    quantity: float
    price: Optional[float] = None
    parent_id: Optional[str] = None
    filled_quantity: float = 0.0
    status: OrderStatus = OrderStatus.PENDING
    time_in_force: str = "GTC"
    created_at: float = field(default=0.0)
    updated_at: float = field(default=0.0)

    def __post_init__(self) -> None:
        """
        Initialize lifecycle timestamps when omitted.
        """

        if self.created_at == 0.0:
            self.created_at = self.timestamp

        if self.updated_at == 0.0:
            self.updated_at = self.timestamp

    @property
    def remaining_quantity(self) -> float:
        """
        Return the quantity that remains unfilled.
        """

        return max(
            0.0,
            self.quantity - self.filled_quantity,
        )

    @property
    def is_active(self) -> bool:
        """
        Return whether the order remains active.
        """

        return self.status in [
            OrderStatus.PENDING,
            OrderStatus.SUBMITTED,
            OrderStatus.PARTIALLY_FILLED,
        ]


@dataclass
class Fill:
    """
    Represents an execution fill record for an order.
    """

    fill_id: str
    order_id: str
    timestamp: float
    price: float
    quantity: float
    liquidity_type: LiquidityType
    parent_id: Optional[str] = None
    fee: float = 0.0
    slippage: float = 0.0


@dataclass
class ExecutionDecision:
    """
    Standardized decision returned by the dynamic execution engine.
    """

    timestamp: float
    strategy: str
    side: OrderSide
    quantity: float
    urgency: float
    risk_score: float
    reason: str
    remaining_quantity: float
    limit_price: Optional[float] = None
    expected_cost: float = 0.0
    execution_mode: ExecutionMode = ExecutionMode.AUTO
    cancel_active_orders: bool = False


@dataclass
class ExecutionResult:
    """
    Summary of one strategy execution run.
    """

    strategy_name: str
    side: OrderSide
    total_quantity: float
    executed_quantity: float
    avg_execution_price: float
    arrival_price: float
    implementation_shortfall: float
    implementation_shortfall_bps: float
    slippage: float
    market_impact: float
    total_cost: float
    fill_rate: float
    completion_rate: float
    execution_time_sec: float
    num_orders: int
    num_fills: int
    num_partial_fills: int
    adverse_selection_cost: float
    expected_cost: float
    fills: List[Fill] = field(
        default_factory=list
    )
    orders: List[Order] = field(
        default_factory=list
    )
    trajectory: List[Dict[str, Any]] = field(
        default_factory=list
    )


# =============================================================================
# Routing Contracts
# =============================================================================


@dataclass
class VenueQuote:
    """
    Quote and routing metrics for an individual simulated venue.
    """

    venue_id: str
    venue_name: str
    best_bid: float
    best_ask: float
    bid_depth: float
    ask_depth: float
    spread: float
    spread_bps: float
    liquidity_score: float
    execution_risk: float
    est_cost_bps: float
    routed_quantity: float = 0.0
    allocation_pct: float = 0.0


@dataclass
class VenueRoutingResult:
    """
    Result of multi-venue smart-order-routing allocation.
    """

    timestamp: float
    total_quantity: float
    side: OrderSide
    venues: List[VenueQuote]
    explanation: str
    avg_price: float = 0.0
    effective_spread_bps: float = 0.0
    total_cost_est: float = 0.0


# =============================================================================
# Backtesting Contracts
# =============================================================================


@dataclass
class BacktestResult:
    """
    Encapsulates comparative results across multiple strategies.
    """

    scenario_name: str
    scenario_description: str
    initial_arrival_price: float
    target_quantity: float
    execution_horizon_sec: float
    strategy_results: Dict[str, ExecutionResult]
    comparison_table: pd.DataFrame
    market_snapshots_count: int
    identical_market_guarantee: bool = True


# =============================================================================
# Sequence Dataset Contract
# =============================================================================


@dataclass
class SequenceDataset:
    """
    Rolling feature-sequence dataset.

    This contract represents historical model inputs before target alignment.

    Expected structure:

        sequences:
            Shape = (num_sequences, context_window, num_features)

        timestamps:
            One timestamp window per sequence.

    Current NVIDIA structure:

        (N, 20, 14)
    """

    sequences: Any
    timestamps: List[List[float]]
    feature_names: List[str]
    context_window: int

    def __post_init__(self) -> None:
        """
        Validate sequence-dataset structure.
        """

        _validate_positive_integer(
            self.context_window,
            "context_window",
        )

        _validate_unique_strings(
            self.feature_names,
            "feature_names",
        )

        if not hasattr(self.sequences, "shape"):
            raise TypeError(
                "sequences must provide a shape attribute."
            )

        if len(self.sequences.shape) != 3:
            raise ValueError(
                "sequences must be a 3-dimensional array."
            )

        if self.sequences.shape[0] <= 0:
            raise ValueError(
                "sequences must contain at least one sequence."
            )

        if self.sequences.shape[1] != self.context_window:
            raise ValueError(
                "Sequence context dimension does not match "
                "context_window."
            )

        if self.sequences.shape[2] != len(self.feature_names):
            raise ValueError(
                "Sequence feature dimension does not match "
                "feature_names."
            )

        if len(self.timestamps) != self.sequences.shape[0]:
            raise ValueError(
                "timestamps must contain one timestamp window per sequence."
            )

        for index, timestamp_window in enumerate(
            self.timestamps
        ):
            if len(timestamp_window) != self.context_window:
                raise ValueError(
                    f"timestamps[{index}] must contain exactly "
                    f"{self.context_window} timestamps."
                )

            _validate_timestamp_sequence(
                timestamp_window,
                f"timestamps[{index}]",
                require_unique=True,
            )

    @property
    def num_sequences(self) -> int:
        """
        Return the number of rolling sequences.
        """

        return int(
            self.sequences.shape[0]
        )

    @property
    def num_features(self) -> int:
        """
        Return the number of features per timestep.
        """

        return int(
            self.sequences.shape[2]
        )

    @property
    def shape(self) -> Tuple[int, int, int]:
        """
        Return the complete sequence tensor shape.
        """

        return tuple(
            self.sequences.shape
        )

    @property
    def input_shape(self) -> Tuple[int, int, int]:
        """
        Return the complete batch input shape.

        This remains the full tensor shape for backward compatibility with
        sequence-level callers.

        ForecastingDataset.input_shape is intentionally different: it returns
        the per-sample shape (20, 14).
        """

        return self.shape


# =============================================================================
# Target Dataset Contract
# =============================================================================


@dataclass
class TargetDataset:
    """
    Forecasting target dataset.

    Default target:

        future_mid_price_return =
            (future_mid_price - current_mid_price)
            / current_mid_price

    Target timestamps correspond to the current observation at which the
    prediction is made, not the future observation used to calculate the
    target.

    For N observations and horizon H:

        number of valid targets = N - H
    """

    targets: Any
    timestamps: List[float]
    target_name: str
    forecast_horizon: int
    price_column: str = DEFAULT_FORECAST_PRICE_COLUMN

    def __post_init__(self) -> None:
        """
        Validate target-dataset structure.
        """

        _validate_positive_integer(
            self.forecast_horizon,
            "forecast_horizon",
        )

        _validate_nonempty_string(
            self.target_name,
            "target_name",
        )

        _validate_nonempty_string(
            self.price_column,
            "price_column",
        )

        if not hasattr(self.targets, "__len__"):
            raise TypeError(
                "targets must be a sized one-dimensional collection."
            )

        if len(self.targets) <= 0:
            raise ValueError(
                "targets must contain at least one value."
            )

        if len(self.targets) != len(self.timestamps):
            raise ValueError(
                "targets and timestamps must contain the same number "
                "of values."
            )

        _validate_timestamp_sequence(
            self.timestamps,
            require_unique=True,
        )

        for index, target in enumerate(self.targets):
            _validate_numeric_value(
                target,
                f"targets[{index}]",
            )

    @property
    def num_targets(self) -> int:
        """
        Return the number of forecasting targets.
        """

        return int(
            len(self.targets)
        )

    @property
    def target_values(self) -> Any:
        """
        Return the target values.
        """

        return self.targets


# =============================================================================
# Forecasting Dataset Contract
# =============================================================================


@dataclass(frozen=True)
class ForecastingDataset:
    """
    Fully aligned forecasting dataset.

    Each sample contains:

        sequence
        target
        timestamp

    Alignment rule:

        sequence ending timestamp
            ==
        target timestamp

    Current NVIDIA model contract:

        sequences:
            (N, 20, 14)

        targets:
            (N,)

        timestamps:
            (N,)

    Important shape distinction:

        dataset.shape
            -> (N, 20, 14)

        dataset.input_shape
            -> (20, 14)

    The latter is the shape of one model input sample.
    """

    sequences: Any
    targets: Any
    timestamps: Any
    feature_names: Tuple[str, ...]
    context_window: int
    forecast_horizon: int
    target_name: str
    price_column: str = DEFAULT_FORECAST_PRICE_COLUMN

    def __post_init__(self) -> None:
        """
        Validate the aligned forecasting dataset.
        """

        _validate_positive_integer(
            self.context_window,
            "context_window",
        )

        _validate_positive_integer(
            self.forecast_horizon,
            "forecast_horizon",
        )

        _validate_nonempty_string(
            self.target_name,
            "target_name",
        )

        _validate_nonempty_string(
            self.price_column,
            "price_column",
        )

        _validate_unique_strings(
            self.feature_names,
            "feature_names",
        )

        if not hasattr(self.sequences, "ndim"):
            raise TypeError(
                "sequences must be a NumPy-compatible array."
            )

        if not hasattr(self.targets, "ndim"):
            raise TypeError(
                "targets must be a NumPy-compatible array."
            )

        if not hasattr(self.timestamps, "ndim"):
            raise TypeError(
                "timestamps must be a NumPy-compatible array."
            )

        if self.sequences.ndim != 3:
            raise ValueError(
                "sequences must be a 3-dimensional array."
            )

        if self.targets.ndim != 1:
            raise ValueError(
                "targets must be a 1-dimensional array."
            )

        if self.timestamps.ndim != 1:
            raise ValueError(
                "timestamps must be a 1-dimensional array."
            )

        if self.sequences.shape[0] <= 0:
            raise ValueError(
                "ForecastingDataset must contain at least one sample."
            )

        if self.sequences.shape[1] != self.context_window:
            raise ValueError(
                "Sequence context dimension does not match "
                "context_window."
            )

        if self.sequences.shape[2] != len(self.feature_names):
            raise ValueError(
                "Sequence feature dimension does not match "
                "feature_names."
            )

        if not (
            len(self.sequences)
            == len(self.targets)
            == len(self.timestamps)
        ):
            raise ValueError(
                "sequences, targets, and timestamps must contain "
                "the same number of samples."
            )

        if len(self.timestamps) == 0:
            raise ValueError(
                "timestamps must contain at least one sample."
            )

        if not hasattr(self.sequences, "dtype"):
            raise TypeError(
                "sequences must provide a numeric dtype."
            )

        if not hasattr(self.targets, "dtype"):
            raise TypeError(
                "targets must provide a numeric dtype."
            )

        if not hasattr(self.timestamps, "dtype"):
            raise TypeError(
                "timestamps must provide a numeric dtype."
            )

        if not pd.api.types.is_numeric_dtype(
            self.sequences.dtype
        ):
            raise ValueError(
                "sequences must contain numeric values."
            )

        if not pd.api.types.is_numeric_dtype(
            self.targets.dtype
        ):
            raise ValueError(
                "targets must contain numeric values."
            )

        if not pd.api.types.is_numeric_dtype(
            self.timestamps.dtype
        ):
            raise ValueError(
                "timestamps must contain numeric values."
            )

        if not self.sequences.size:
            raise ValueError(
                "sequences must contain data."
            )

        if not self.targets.size:
            raise ValueError(
                "targets must contain data."
            )

        if not self.timestamps.size:
            raise ValueError(
                "timestamps must contain data."
            )

        if not bool(
            pd.Series(
                self.sequences.reshape(-1)
            ).map(
                lambda value:
                    isinstance(value, Real)
                    and not isinstance(value, bool)
                    and isfinite(float(value))
            ).all()
        ):
            raise ValueError(
                "sequences must contain only finite numeric values."
            )

        if not bool(
            pd.Series(
                self.targets
            ).map(
                lambda value:
                    isinstance(value, Real)
                    and not isinstance(value, bool)
                    and isfinite(float(value))
            ).all()
        ):
            raise ValueError(
                "targets must contain only finite numeric values."
            )

        if not bool(
            pd.Series(
                self.timestamps
            ).map(
                lambda value:
                    isinstance(value, Real)
                    and not isinstance(value, bool)
                    and isfinite(float(value))
            ).all()
        ):
            raise ValueError(
                "timestamps must contain only finite numeric values."
            )

        timestamp_values = [
            float(timestamp)
            for timestamp in self.timestamps
        ]

        _validate_timestamp_sequence(
            timestamp_values,
            require_unique=True,
        )

    @property
    def num_samples(self) -> int:
        """
        Return the number of aligned forecasting samples.
        """

        return int(
            self.sequences.shape[0]
        )

    @property
    def num_features(self) -> int:
        """
        Return the number of model features.
        """

        return int(
            self.sequences.shape[2]
        )

    @property
    def shape(self) -> Tuple[int, int, int]:
        """
        Return the complete feature tensor shape.

        Returns:

            (num_samples, context_window, num_features)
        """

        return tuple(
            self.sequences.shape
        )

    @property
    def input_shape(self) -> Tuple[int, int]:
        """
        Return the per-sample model input shape.

        This is intentionally different from shape.

        Example:

            shape       = (100, 20, 14)
            input_shape = (20, 14)
        """

        return (
            self.context_window,
            self.num_features,
        )

    @property
    def target_values(self) -> Any:
        """
        Return the aligned target array.
        """

        return self.targets

    @property
    def earliest_timestamp(self) -> float:
        """
        Return the earliest forecasting sample timestamp.
        """

        return float(
            self.timestamps[0]
        )

    @property
    def latest_timestamp(self) -> float:
        """
        Return the latest forecasting sample timestamp.
        """

        return float(
            self.timestamps[-1]
        )

    @property
    def model_feature_shape(self) -> Tuple[int, int]:
        """
        Return the feature shape expected for each individual model sample.
        """

        return (
            self.context_window,
            self.num_features,
        )

    def to_forecast_input(
        self,
        index: int,
    ) -> ForecastInput:
        """
        Convert one dataset sample into a ForecastInput.

        Targets are intentionally excluded from ForecastInput so that target
        information cannot enter the model input during inference.
        """

        if not isinstance(index, int):
            raise TypeError(
                "index must be an integer."
            )

        if index < 0 or index >= self.num_samples:
            raise IndexError(
                "ForecastingDataset index is out of range."
            )

        sequence = self.sequences[index]

        timestamp_window = (
            self._timestamp_window_for_sample(index)
        )

        feature_sequence = [
            [
                float(value)
                for value in row
            ]
            for row in sequence
        ]

        timestamps = [
            float(timestamp)
            for timestamp in timestamp_window
        ]

        return ForecastInput(
            feature_sequence=feature_sequence,
            feature_names=list(self.feature_names),
            timestamps=timestamps,
            context_window=self.context_window,
            forecast_horizon=self.forecast_horizon,
        )

    def _timestamp_window_for_sample(
        self,
        index: int,
    ) -> Sequence[float]:
        """
        Return the original context-window timestamps for one sample.

        ForecastingDataset stores one aligned timestamp per sample. A full
        model-facing ForecastInput requires the original timestamp window.

        The optional timestamp_windows attribute is therefore supported by
        upstream integrations that retain those windows.
        """

        timestamp_windows = getattr(
            self,
            "timestamp_windows",
            None,
        )

        if timestamp_windows is None:
            raise ValueError(
                "Complete timestamp windows are not stored in "
                "ForecastingDataset. A ForecastInput requires the "
                "original context-window timestamps."
            )

        window = timestamp_windows[index]

        if len(window) != self.context_window:
            raise ValueError(
                "Stored timestamp window does not match context_window."
            )

        return window


# =============================================================================
# Forecasting Contract Validation
# =============================================================================


def validate_forecasting_dataset_contract(
    dataset: ForecastingDataset,
) -> None:
    """
    Validate that a ForecastingDataset satisfies the NVIDIA contract.
    """

    if not isinstance(
        dataset,
        ForecastingDataset,
    ):
        raise TypeError(
            "dataset must be a ForecastingDataset instance."
        )

    if dataset.context_window != (
        DEFAULT_FORECAST_CONTEXT_WINDOW
    ):
        raise ValueError(
            "Forecasting context window does not match the NVIDIA "
            f"contract. Expected "
            f"{DEFAULT_FORECAST_CONTEXT_WINDOW}, "
            f"received {dataset.context_window}."
        )

    if dataset.num_features != len(
        NVIDIA_FORECAST_FEATURES
    ):
        raise ValueError(
            "Forecasting feature count does not match the NVIDIA "
            f"contract. Expected "
            f"{len(NVIDIA_FORECAST_FEATURES)}, "
            f"received {dataset.num_features}."
        )

    validate_nvidia_feature_contract(
        dataset.feature_names
    )

    if dataset.forecast_horizon != (
        DEFAULT_FORECAST_HORIZON
    ):
        raise ValueError(
            "Forecast horizon does not match the NVIDIA forecasting "
            f"contract. Expected "
            f"{DEFAULT_FORECAST_HORIZON}, "
            f"received {dataset.forecast_horizon}."
        )

    if dataset.target_name != (
        DEFAULT_FORECAST_TARGET_NAME
    ):
        raise ValueError(
            "Forecast target name does not match the NVIDIA forecasting "
            f"contract. Expected "
            f"{DEFAULT_FORECAST_TARGET_NAME!r}, "
            f"received {dataset.target_name!r}."
        )

    if dataset.price_column != (
        DEFAULT_FORECAST_PRICE_COLUMN
    ):
        raise ValueError(
            "Forecast price column does not match the NVIDIA forecasting "
            f"contract. Expected "
            f"{DEFAULT_FORECAST_PRICE_COLUMN!r}, "
            f"received {dataset.price_column!r}."
        )


# =============================================================================
# Execution Context Contract
# =============================================================================

@dataclass(frozen=True)
class ExecutionContext:
    """
    Unified execution context passed between pipeline stages.

    Combines adverse-selection risk, NVIDIA forecast, Almgren-Chriss schedule
    data, and market microstructure features into a single immutable record
    used by the dynamic execution engine and proposed strategy.

    Invariants:
        - has_valid_forecast: forecast is not None and its model_status is in
          {TRAINED, SUCCESS, READY} and forecast.timestamp <= context timestamp
          (guards against future-data leakage).
        - is_ml_fallback: adverse_risk model_status is FALLBACK or
          FALLBACK_HEURISTIC (indicates no trained model available).
    """

    timestamp: float
    remaining_quantity: float
    initial_quantity: float
    elapsed_time: float
    total_horizon: float
    urgency: float
    ac_slice_quantity: float
    ac_expected_cost: float
    ac_schedule_index: int
    adverse_risk: PredictionResult
    forecast: Optional[ForecastResult] = None
    spread_bps: Optional[float] = None
    top_of_book_depth: Optional[float] = None
    volatility: Optional[float] = None

    @property
    def has_valid_forecast(self) -> bool:
        """
        Return True if the forecast is available, structurally valid, and not
        stale (no future-data leakage).

        A forecast is valid when:
            - it is not None,
            - its model_status is one of {TRAINED, SUCCESS, READY}, and
            - its timestamp does not exceed the execution context timestamp.
        """
        if self.forecast is None:
            return False
        if self.forecast.model_status not in {ModelStatus.TRAINED, ModelStatus.SUCCESS, ModelStatus.READY}:
            return False
        return self.forecast.timestamp <= self.timestamp

    @property
    def is_ml_fallback(self) -> bool:
        """
        Return True if the adverse-selection model is in a fallback/heuristic
        state, meaning no trained model is available.
        """
        return self.adverse_risk.model_status in {ModelStatus.FALLBACK, ModelStatus.FALLBACK_HEURISTIC}

    def to_dict(self) -> Dict[str, Any]:
        """
        Serialise the execution context to a dictionary for the API/dashboard.

        Returns a plain dict with all fields; Optional fields that are None
        are omitted from the returned mapping.
        """
        d: Dict[str, Any] = {
            "timestamp": self.timestamp,
            "remaining_quantity": self.remaining_quantity,
            "initial_quantity": self.initial_quantity,
            "elapsed_time": self.elapsed_time,
            "total_horizon": self.total_horizon,
            "urgency": self.urgency,
            "ac_slice_quantity": self.ac_slice_quantity,
            "ac_expected_cost": self.ac_expected_cost,
            "ac_schedule_index": self.ac_schedule_index,
            "adverse_risk_probability": self.adverse_risk.probability,
            "adverse_risk_model_status_label": self.adverse_risk.model_status,
            "adverse_risk_model_name": self.adverse_risk.model_name,
        }
        if self.forecast is not None:
            d["forecast_predicted_return"] = self.forecast.predicted_return
            d["forecast_model_status"] = self.forecast.model_status
            d["forecast_model_name"] = self.forecast.model_name
            d["forecast_horizon"] = self.forecast.forecast_horizon
            d["forecast_timestamp"] = self.forecast.timestamp
        else:
            d["forecast_predicted_return"] = None
            d["forecast_model_status"] = None
            d["forecast_model_name"] = None
            d["forecast_horizon"] = None
            d["forecast_timestamp"] = None
        if self.spread_bps is not None:
            d["spread_bps"] = self.spread_bps
        if self.top_of_book_depth is not None:
            d["top_of_book_depth"] = self.top_of_book_depth
        if self.volatility is not None:
            d["volatility"] = self.volatility
        return d


# =============================================================================
# Public Module API
# =============================================================================


__all__ = [
    # Core enumerations
    "OrderSide",
    "OrderType",
    "OrderStatus",
    "ExecutionStatus",
    "LiquidityType",
    "RiskCategory",
    "ExecutionMode",
    "ModelStatus",
    "MarketRegime",

    # Model/system contracts
    "ModelSystemStatus",
    "PredictionResult",

    # NVIDIA forecasting constants
    "NVIDIA_FORECAST_FEATURES",
    "DEFAULT_FORECAST_CONTEXT_WINDOW",
    "DEFAULT_FORECAST_HORIZON",
    "DEFAULT_FORECAST_TARGET_NAME",
    "DEFAULT_FORECAST_PRICE_COLUMN",

    # NVIDIA forecasting validation
    "validate_nvidia_feature_contract",
    "validate_forecasting_dataset_contract",

    # Forecasting model contracts
    "ForecastInput",
    "ForecastResult",

    # Market-data contracts
    "Trade",
    "MarketSnapshot",

    # Order/execution contracts
    "Order",
    "Fill",
    "ExecutionDecision",
    "ExecutionResult",

    # Routing contracts
    "VenueQuote",
    "VenueRoutingResult",

    # Backtesting contracts
    "BacktestResult",

    # Forecasting dataset contracts
    "SequenceDataset",
    "TargetDataset",
    "ForecastingDataset",

    # -------------------------------------------------------------------------
    # Execution context contract
    # -------------------------------------------------------------------------
    "ExecutionContext",
] 

