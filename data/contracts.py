"""
Core Domain Data Contracts for SmartFlow.

This module defines the strongly typed data structures shared across the
SmartFlow market-data, feature-engineering, forecasting, execution,
simulation, and backtesting layers.

The contracts are intentionally kept independent from implementation logic.
They provide stable interfaces between components while allowing individual
modules to evolve internally.

Contract Categories
-------------------
Core market contracts:
    Trade
    MarketSnapshot

Order and execution contracts:
    Order
    Fill
    ExecutionDecision
    ExecutionResult

Machine-learning contracts:
    PredictionResult
    ForecastInput
    ForecastResult
    ModelSystemStatus

Routing contracts:
    VenueQuote
    VenueRoutingResult

Backtesting contracts:
    BacktestResult

Sequence-data contracts:
    SequenceDataset

Forecasting Architecture
------------------------
SmartFlow uses two conceptually different ML outputs:

1. PredictionResult
   Represents adverse-selection / passive-exposure risk.

2. ForecastResult
   Represents short-term directional market movement.

These outputs must remain separate because a directional forecast is not the
same thing as the probability that a passive order will receive a toxic fill.

The forecasting data pipeline is:

    MarketSnapshot
        |
        v
    Feature Extraction
        |
        v
    Chronological Feature History
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

Design Principles
-----------------
- Use explicit dataclasses for shared interfaces.
- Keep contracts independent from implementation details.
- Validate structural invariants at object construction time.
- Keep adverse-selection prediction separate from directional forecasting.
- Preserve compatibility with the existing SmartFlow architecture.
- Avoid embedding model, execution, or simulator logic in contracts.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd


# ============================================================================
# Core Enumerations
# ============================================================================


class OrderSide(str, Enum):
    """Side of a market or execution order."""

    BUY = "BUY"
    SELL = "SELL"


class OrderType(str, Enum):
    """Supported order types in the simulator."""

    MARKET = "MARKET"
    LIMIT = "LIMIT"


class OrderStatus(str, Enum):
    """Lifecycle status of an order."""

    PENDING = "PENDING"
    SUBMITTED = "SUBMITTED"
    PARTIALLY_FILLED = "PARTIALLY_FILLED"
    FILLED = "FILLED"
    CANCELLED = "CANCELLED"
    REJECTED = "REJECTED"


class LiquidityType(str, Enum):
    """Whether an execution provided or consumed liquidity."""

    MAKER = "MAKER"
    TAKER = "TAKER"


class RiskCategory(str, Enum):
    """Categorical representation of adverse-selection risk."""

    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"


class ExecutionMode(str, Enum):
    """Typed execution action returned by the execution engine."""

    AUTO = "auto"
    HOLD = "hold"
    PASSIVE = "passive"
    AGGRESSIVE = "aggressive"


class ModelStatus(str, Enum):
    """Standardized ML model availability/status values."""

    TRAINED = "TRAINED"
    FALLBACK = "FALLBACK"
    FALLBACK_HEURISTIC = "FALLBACK_HEURISTIC"
    UNAVAILABLE = "UNAVAILABLE"


class MarketRegime(str, Enum):
    """Deterministic market-regime classifications."""

    CALM = "Calm"
    NORMAL = "Normal"
    VOLATILE = "Volatile"
    HIGHLY_VOLATILE = "Highly Volatile"
    ILLIQUID = "Illiquid"


# ============================================================================
# ML / Model Contracts
# ============================================================================


@dataclass
class ModelSystemStatus:
    """
    Standardized single source of truth for ML model status.

    This contract is used to communicate whether a model is loaded and
    whether predictions are currently available, without exposing model
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
class ForecastInput:
    """
    Input contract for one NVIDIA forecasting-model inference request.

    A ForecastInput represents exactly one chronological context window.

    The expected initial feature set contains 14 market microstructure and
    short-term price features:

        mid_price_return
        spread_bps
        best_bid_size
        best_ask_size
        depth_imbalance_l1
        depth_imbalance_multilevel
        ofi_instant
        ofi_sum_5
        trade_volume_imbalance
        momentum_ret_5
        momentum_ret_20
        volatility_std_10
        micro_price
        micro_price_divergence

    Data Shape
    ----------
    feature_sequence:
        A two-dimensional sequence represented as:

            [context_window][feature_count]

        For the current SmartFlow configuration:

            [20][14]

    timestamps:
        One timestamp for each observation in the context window.

    Important
    ---------
    The sequence must contain only information available at or before its
    final observation timestamp. Future observations must never be inserted
    into feature_sequence.
    """

    feature_sequence: List[List[float]]
    feature_names: List[str]
    timestamps: List[float]
    context_window: int
    forecast_horizon: int

    def __post_init__(self) -> None:
        """Validate the structural integrity of one forecasting input."""

        if self.context_window <= 0:
            raise ValueError(
                "context_window must be greater than zero."
            )

        if self.forecast_horizon <= 0:
            raise ValueError(
                "forecast_horizon must be greater than zero."
            )

        if not self.feature_sequence:
            raise ValueError(
                "feature_sequence cannot be empty."
            )

        if len(self.feature_sequence) != self.context_window:
            raise ValueError(
                "feature_sequence length must equal context_window."
            )

        if len(self.timestamps) != self.context_window:
            raise ValueError(
                "timestamps length must equal context_window."
            )

        if not self.feature_names:
            raise ValueError(
                "feature_names cannot be empty."
            )

        feature_count = len(self.feature_names)

        if len(set(self.feature_names)) != feature_count:
            raise ValueError(
                "feature_names must not contain duplicate names."
            )

        for row in self.feature_sequence:
            if len(row) != feature_count:
                raise ValueError(
                    "Every feature row must contain exactly one value "
                    "for each feature name."
                )

        if any(
            self.timestamps[index] > self.timestamps[index + 1]
            for index in range(len(self.timestamps) - 1)
        ):
            raise ValueError(
                "Forecasting timestamps must be in chronological order."
            )

        if len(set(self.timestamps)) != len(self.timestamps):
            raise ValueError(
                "Forecasting timestamps must be unique."
            )

        for timestamp in self.timestamps:
            if not isinstance(timestamp, (int, float)):
                raise TypeError(
                    "Forecasting timestamps must be numeric."
                )


@dataclass
class SequenceDataset:
    """
    Contract representing a collection of rolling forecasting sequences.

    Unlike ForecastInput, which represents one inference window,
    SequenceDataset represents the complete set of windows produced from a
    chronological feature history.

    Expected current shape:

        sequences.shape == (N, 20, 14)

    where:

        N  = number of rolling sequences
        20 = context window
        14 = number of forecasting features

    The sequence timestamps identify the observation timestamps belonging to
    each rolling window.

    This contract intentionally does not contain forecasting targets yet.
    Target construction belongs to Phase 4. Keeping targets separate here
    prevents sequence construction from accidentally introducing future
    information into the input-generation layer.
    """

    sequences: Any
    timestamps: List[List[float]]
    feature_names: List[str]
    context_window: int

    def __post_init__(self) -> None:
        """Validate the structural integrity of the sequence dataset."""

        if self.context_window <= 0:
            raise ValueError(
                "context_window must be greater than zero."
            )

        if not self.feature_names:
            raise ValueError(
                "feature_names cannot be empty."
            )

        if len(set(self.feature_names)) != len(self.feature_names):
            raise ValueError(
                "feature_names must not contain duplicate names."
            )

        if self.sequences is None:
            raise ValueError(
                "sequences cannot be None."
            )

        if not hasattr(self.sequences, "shape"):
            raise TypeError(
                "sequences must provide a shape attribute."
            )

        shape = tuple(self.sequences.shape)

        if len(shape) != 3:
            raise ValueError(
                "sequences must be a three-dimensional structure with "
                "shape (num_sequences, context_window, num_features)."
            )

        if shape[1] != self.context_window:
            raise ValueError(
                "The sequence context dimension must equal context_window."
            )

        if shape[2] != len(self.feature_names):
            raise ValueError(
                "The sequence feature dimension must equal the number "
                "of feature_names."
            )

        if len(self.timestamps) != shape[0]:
            raise ValueError(
                "timestamps must contain one entry for every sequence."
            )

        for sequence_timestamps in self.timestamps:
            if len(sequence_timestamps) != self.context_window:
                raise ValueError(
                    "Every sequence must contain exactly context_window "
                    "timestamps."
                )

            if any(
                sequence_timestamps[index]
                > sequence_timestamps[index + 1]
                for index in range(len(sequence_timestamps) - 1)
            ):
                raise ValueError(
                    "Each sequence's timestamps must be chronological."
                )

            if len(set(sequence_timestamps)) != len(sequence_timestamps):
                raise ValueError(
                    "Timestamps within a sequence must be unique."
                )

    @property
    def num_sequences(self) -> int:
        """Return the number of rolling sequences."""

        return int(self.sequences.shape[0])

    @property
    def num_features(self) -> int:
        """Return the number of features contained in each timestep."""

        return int(self.sequences.shape[2])

    @property
    def shape(self) -> Tuple[int, int, int]:
        """Return the complete sequence-dataset shape."""

        return tuple(self.sequences.shape)


@dataclass
class ForecastResult:
    """
    Output contract for the NVIDIA forecasting model.

    The primary forecast target is the future cumulative mid-price return
    over the configured forecast horizon:

        (future_mid_price - current_mid_price) / current_mid_price

    ForecastResult intentionally remains separate from PredictionResult.

    PredictionResult:
        Estimates adverse-selection / passive-exposure risk.

    ForecastResult:
        Estimates short-term directional market movement.
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
    feature_names: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate and derive basic forecast information."""

        if self.forecast_horizon <= 0:
            raise ValueError(
                "forecast_horizon must be greater than zero."
            )

        if self.confidence is not None:
            if not 0.0 <= self.confidence <= 1.0:
                raise ValueError(
                    "confidence must be between 0.0 and 1.0."
                )

        if self.predicted_direction is None:
            if self.predicted_return > 0:
                self.predicted_direction = "UP"
            elif self.predicted_return < 0:
                self.predicted_direction = "DOWN"
            else:
                self.predicted_direction = "FLAT"

        if self.expected_move_bps is None:
            self.expected_move_bps = self.predicted_return * 10_000


@dataclass
class PredictionResult:
    """
    Output from the adverse-selection / passive-exposure risk model.

    probability estimates how dangerous it is to rest a maker order at the
    current market state. It is not a directional forecast of price movement.
    """

    probability: float
    timestamp: float
    prediction_horizon: int
    model_name: str
    model_status: ModelStatus
    risk_category: RiskCategory = RiskCategory.MEDIUM
    feature_importances: Dict[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate probability and derive its categorical risk level."""

        if not 0.0 <= self.probability <= 1.0:
            raise ValueError(
                "probability must be between 0.0 and 1.0."
            )

        if self.prediction_horizon <= 0:
            raise ValueError(
                "prediction_horizon must be greater than zero."
            )

        if self.probability < 0.35:
            self.risk_category = RiskCategory.LOW
        elif self.probability > 0.65:
            self.risk_category = RiskCategory.HIGH
        else:
            self.risk_category = RiskCategory.MEDIUM


# ============================================================================
# Routing Contracts
# ============================================================================


@dataclass
class VenueQuote:
    """Quote and routing metrics for an individual simulated venue."""

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
    """Result of multi-venue smart-order-routing allocation."""

    timestamp: float
    total_quantity: float
    side: OrderSide
    venues: List[VenueQuote]
    explanation: str
    avg_price: float = 0.0
    effective_spread_bps: float = 0.0
    total_cost_est: float = 0.0


# ============================================================================
# Market Data Contracts
# ============================================================================


@dataclass
class Trade:
    """Represents an individual trade print in the market."""

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
        """Return the best bid price."""

        return self.bids[0][0] if self.bids else 0.0

    @property
    def best_bid_size(self) -> float:
        """Return quantity available at the best bid."""

        return self.bids[0][1] if self.bids else 0.0

    @property
    def best_ask(self) -> float:
        """Return the best ask price."""

        return self.asks[0][0] if self.asks else 0.0

    @property
    def best_ask_size(self) -> float:
        """Return quantity available at the best ask."""

        return self.asks[0][1] if self.asks else 0.0

    @property
    def mid_price(self) -> float:
        """Return the midpoint between the best bid and ask."""

        if self.best_bid > 0 and self.best_ask > 0:
            return (self.best_bid + self.best_ask) / 2.0

        return self.best_bid or self.best_ask or 0.0

    @property
    def spread(self) -> float:
        """Return the absolute bid-ask spread."""

        if self.best_bid > 0 and self.best_ask > 0:
            return max(0.0, self.best_ask - self.best_bid)

        return 0.0

    @property
    def total_bid_depth(self) -> float:
        """Return total displayed bid depth."""

        return sum(size for _, size in self.bids)

    @property
    def total_ask_depth(self) -> float:
        """Return total displayed ask depth."""

        return sum(size for _, size in self.asks)


# ============================================================================
# Order / Execution Contracts
# ============================================================================


@dataclass
class Order:
    """Represents a trading order placed in the market simulator."""

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
        """Initialize lifecycle timestamps when omitted."""

        if self.created_at == 0.0:
            self.created_at = self.timestamp

        if self.updated_at == 0.0:
            self.updated_at = self.timestamp

    @property
    def remaining_quantity(self) -> float:
        """Return the quantity that remains unfilled."""

        return max(0.0, self.quantity - self.filled_quantity)

    @property
    def is_active(self) -> bool:
        """Return whether the order remains active."""

        return self.status in [
            OrderStatus.PENDING,
            OrderStatus.SUBMITTED,
            OrderStatus.PARTIALLY_FILLED,
        ]


@dataclass
class Fill:
    """Represents an execution fill record for an order."""

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
    """Standardized decision returned by the dynamic execution engine."""

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
    """Summary of one strategy execution run."""

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
    fills: List[Fill] = field(default_factory=list)
    orders: List[Order] = field(default_factory=list)
    trajectory: List[Dict[str, Any]] = field(default_factory=list)


# ============================================================================
# Backtesting Contracts
# ============================================================================


@dataclass
class BacktestResult:
    """Encapsulates comparative results across multiple strategies."""

    scenario_name: str
    scenario_description: str
    initial_arrival_price: float
    target_quantity: float
    execution_horizon_sec: float
    strategy_results: Dict[str, ExecutionResult]
    comparison_table: pd.DataFrame
    market_snapshots_count: int
    identical_market_guarantee: bool = True

