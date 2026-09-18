"""
Core Domain Data Contracts for Smart Order Routing and Trade Execution.

Defines strongly-typed dataclasses and enums for order book snapshots, trades,
orders, fills, ML prediction results, execution decisions, backtest results,
and NVIDIA forecasting inputs and outputs.

The contracts in this module provide a shared interface between data
preparation, feature engineering, machine learning, execution, simulation,
and backtesting components.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Tuple, Dict, Any, Optional

import pandas as pd


class OrderSide(str, Enum):
    BUY = "BUY"
    SELL = "SELL"


class OrderType(str, Enum):
    MARKET = "MARKET"
    LIMIT = "LIMIT"


class OrderStatus(str, Enum):
    PENDING = "PENDING"
    SUBMITTED = "SUBMITTED"
    PARTIALLY_FILLED = "PARTIALLY_FILLED"
    FILLED = "FILLED"
    CANCELLED = "CANCELLED"
    REJECTED = "REJECTED"


class LiquidityType(str, Enum):
    MAKER = "MAKER"  # Passive limit order
    TAKER = "TAKER"  # Aggressive market order or crossing limit


class RiskCategory(str, Enum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"


class ExecutionMode(str, Enum):
    """Typed execution action. AUTO preserves legacy strategy-name inference."""

    AUTO = "auto"
    HOLD = "hold"
    PASSIVE = "passive"
    AGGRESSIVE = "aggressive"


class ModelStatus(str, Enum):
    TRAINED = "TRAINED"
    FALLBACK = "FALLBACK"
    FALLBACK_HEURISTIC = "FALLBACK_HEURISTIC"
    UNAVAILABLE = "UNAVAILABLE"


class MarketRegime(str, Enum):
    CALM = "Calm"
    NORMAL = "Normal"
    VOLATILE = "Volatile"
    HIGHLY_VOLATILE = "Highly Volatile"
    ILLIQUID = "Illiquid"


@dataclass
class ModelSystemStatus:
    """Standardized single source of truth for ML model status across the system."""

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
    Standardized input contract for the NVIDIA forecasting model.

    The forecasting model receives a chronological context window of engineered
    market features. Each row represents one historical feature snapshot and
    each column represents one agreed-upon forecasting feature.

    No future information may be included in `feature_sequence`.

    Expected initial feature set:

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

    The sequence is ordered chronologically from oldest observation to newest
    observation.
    """

    feature_sequence: List[List[float]]
    feature_names: List[str]
    timestamps: List[float]
    context_window: int
    forecast_horizon: int

    def __post_init__(self):
        """Validate the structural integrity of the forecasting input."""

        if self.context_window <= 0:
            raise ValueError("context_window must be greater than zero.")

        if self.forecast_horizon <= 0:
            raise ValueError("forecast_horizon must be greater than zero.")

        if len(self.feature_sequence) != self.context_window:
            raise ValueError(
                "feature_sequence length must equal context_window."
            )

        if len(self.timestamps) != self.context_window:
            raise ValueError(
                "timestamps length must equal context_window."
            )

        if not self.feature_sequence:
            raise ValueError("feature_sequence cannot be empty.")

        feature_count = len(self.feature_names)

        if feature_count == 0:
            raise ValueError("feature_names cannot be empty.")

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


@dataclass
class ForecastResult:
    """
    Standardized output contract for the NVIDIA forecasting model.

    The primary forecast target is the future cumulative mid-price return
    over the configured forecast horizon:

        (future_mid_price - current_mid_price) / current_mid_price

    The result intentionally remains separate from `PredictionResult`.

    `PredictionResult` represents adverse-selection / passive-exposure risk,
    while `ForecastResult` represents directional short-term market movement.
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

    def __post_init__(self):
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
class VenueQuote:
    """Represents quote and routing metrics for an individual simulated trading venue."""

    venue_id: str
    venue_name: str
    best_bid: float
    best_ask: float
    bid_depth: float
    ask_depth: float
    spread: float
    spread_bps: float
    liquidity_score: float  # 0.0 - 1.0 (relative depth quality)
    execution_risk: float   # 0.0 - 1.0 (volatility / adverse toxicity risk)
    est_cost_bps: float     # Estimated total execution cost in bps
    routed_quantity: float = 0.0
    allocation_pct: float = 0.0


@dataclass
class VenueRoutingResult:
    """Result of multi-venue smart order routing allocation."""

    timestamp: float
    total_quantity: float
    side: OrderSide
    venues: List[VenueQuote]
    explanation: str
    avg_price: float = 0.0
    effective_spread_bps: float = 0.0
    total_cost_est: float = 0.0


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
    Represents an atomic snapshot of the multi-level order book and recent
    trade state.

    Bids and asks are sorted lists of (price, size) tuples:

        bids: descending by price (best bid first)
        asks: ascending by price (best ask first)
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
        return self.bids[0][0] if self.bids else 0.0

    @property
    def best_bid_size(self) -> float:
        return self.bids[0][1] if self.bids else 0.0

    @property
    def best_ask(self) -> float:
        return self.asks[0][0] if self.asks else 0.0

    @property
    def best_ask_size(self) -> float:
        return self.asks[0][1] if self.asks else 0.0

    @property
    def mid_price(self) -> float:
        if self.best_bid > 0 and self.best_ask > 0:
            return (self.best_bid + self.best_ask) / 2.0

        return self.best_bid or self.best_ask or 0.0

    @property
    def spread(self) -> float:
        if self.best_bid > 0 and self.best_ask > 0:
            return max(0.0, self.best_ask - self.best_bid)

        return 0.0

    @property
    def total_bid_depth(self) -> float:
        return sum(size for _, size in self.bids)

    @property
    def total_ask_depth(self) -> float:
        return sum(size for _, size in self.asks)


@dataclass
class Order:
    """Represents a trading order placed in the market simulator."""

    order_id: str
    timestamp: float
    side: OrderSide
    order_type: OrderType
    quantity: float
    price: Optional[float] = None  # None for market orders
    parent_id: Optional[str] = None
    filled_quantity: float = 0.0
    status: OrderStatus = OrderStatus.PENDING
    time_in_force: str = "GTC"
    created_at: float = field(default=0.0)
    updated_at: float = field(default=0.0)

    def __post_init__(self):
        if self.created_at == 0.0:
            self.created_at = self.timestamp

        if self.updated_at == 0.0:
            self.updated_at = self.timestamp

    @property
    def remaining_quantity(self) -> float:
        return max(0.0, self.quantity - self.filled_quantity)

    @property
    def is_active(self) -> bool:
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
class PredictionResult:
    """
    Output from the adverse-selection / passive-exposure risk model.

    `probability` estimates how dangerous it is to rest a maker order right now
    (toxic-fill / winner's-curse risk). It is not a directional forecast of
    where price will go and must not, by itself, justify accelerating a trade.
    """

    probability: float
    timestamp: float
    prediction_horizon: int
    model_name: str
    model_status: ModelStatus
    risk_category: RiskCategory = RiskCategory.MEDIUM
    feature_importances: Dict[str, float] = field(default_factory=dict)

    def __post_init__(self):
        if self.probability < 0.35:
            self.risk_category = RiskCategory.LOW
        elif self.probability > 0.65:
            self.risk_category = RiskCategory.HIGH
        else:
            self.risk_category = RiskCategory.MEDIUM


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
    """Summary of a single strategy execution run."""

    strategy_name: str
    side: OrderSide
    total_quantity: float
    executed_quantity: float
    avg_execution_price: float
    arrival_price: float
    implementation_shortfall: float  # In dollars
    implementation_shortfall_bps: float  # In basis points
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


@dataclass
class BacktestResult:
    """Encapsulates the comparative results across multiple strategies."""

    scenario_name: str
    scenario_description: str
    initial_arrival_price: float
    target_quantity: float
    execution_horizon_sec: float
    strategy_results: Dict[str, ExecutionResult]
    comparison_table: pd.DataFrame
    market_snapshots_count: int
    identical_market_guarantee: bool = True

