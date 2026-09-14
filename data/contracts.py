"""
Core Domain Data Contracts for Smart Order Routing and Trade Execution.
Defines strongly-typed dataclasses and enums for order book snapshots, trades,
orders, fills, ML prediction results, execution decisions, and backtest results.
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
    Represents an atomic snapshot of the multi-level order book and recent trade state.
    Bids and asks are sorted lists of (price, size) tuples:
    bids: descending by price (best bid first)
    asks: ascending by price (best ask first)
    """
    timestamp: float
    sequence_id: int
    bids: List[Tuple[float, float]]  # [(price, size), ...]
    asks: List[Tuple[float, float]]  # [(price, size), ...]
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
        return self.status in [OrderStatus.PENDING, OrderStatus.SUBMITTED, OrderStatus.PARTIALLY_FILLED]


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
    """Represents the output from the ML adverse selection risk model."""
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
