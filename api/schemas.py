"""
Pydantic API Schemas conforming strictly to frontend/src/types.ts (Phase 1 contract).
"""

from typing import List, Optional, Literal
from pydantic import BaseModel, Field

# Domain Literal types
Side = Literal["buy", "sell"]
Strategy = Literal["market", "twap", "vwap", "almgren_chriss", "proposed"]
Action = Literal["passive", "aggressive", "wait"]


class BookLevel(BaseModel):
    price: float
    quantity: float


class MarketState(BaseModel):
    timestamp: str  # UTC ISO 8601
    symbol: str
    bids: List[BookLevel]
    asks: List[BookLevel]
    mid_price: float
    spread: float
    order_flow_imbalance: float


class FeaturePoint(BaseModel):
    """One chronological market-feature observation for Phase 3 charts."""

    timestamp: str  # UTC ISO 8601
    spread_bps: Optional[float] = None
    depth_imbalance_l1: Optional[float] = None  # [-1, 1]
    ofi_instant: Optional[float] = None  # signed quantity, not a normalized ratio
    volatility_std_10: Optional[float] = None  # standard deviation of log returns
    momentum_ret_5: Optional[float] = None  # five-tick simple return
    total_bid_depth: float
    total_ask_depth: float


class FeatureState(BaseModel):
    points: List[FeaturePoint]


class ExecutionPoint(BaseModel):
    timestamp: str
    remaining_quantity: float
    filled_quantity: float
    reference_price: float
    execution_price: Optional[float] = None
    action: Action


class ExecutionState(BaseModel):
    order_id: str
    symbol: str
    side: Side
    target_quantity: float
    filled_quantity: float
    remaining_quantity: float
    selected_action: Action
    execution_score: float
    trajectory: List[ExecutionPoint]


class RiskPoint(BaseModel):
    timestamp: str  # UTC ISO 8601
    adverse_selection_probability: float
    model_status: str


class RiskState(BaseModel):
    timestamp: str
    adverse_selection_probability: float
    fill_probability: Optional[float] = None
    prediction_horizon_ms: Optional[int] = None
    expected_market_impact_bps: Optional[float] = None
    expected_shortfall_bps: Optional[float] = None
    model_status: str = ""
    model_name: str = ""
    prediction_horizon: int = 10
    prediction_horizon_unit: Literal["ticks"] = "ticks"
    history: List[RiskPoint] = []


class ACSchedulePoint(BaseModel):
    elapsed_sec: float
    planned_remaining_quantity: float
    planned_slice_quantity: float


class ACScheduleState(BaseModel):
    total_quantity: float
    horizon_sec: float
    num_slices: int
    expected_cost: float
    cost_variance: float
    utility_cost: float
    cost_unit: Literal["USD"] = "USD"
    points: List[ACSchedulePoint]


class StrategyResult(BaseModel):
    strategy: Strategy
    filled_quantity: float
    fill_rate: float
    slippage_bps: float
    market_impact_bps: float
    implementation_shortfall_bps: float
    adverse_selection_bps: float


class PerformanceState(BaseModel):
    run_id: str
    completed_at: str
    results: List[StrategyResult]


class DashboardSnapshot(BaseModel):
    schema_version: Literal["1.0"] = "1.0"
    market: Optional[MarketState] = None
    features: Optional[FeatureState] = None
    execution: Optional[ExecutionState] = None
    risk: Optional[RiskState] = None
    ac_schedule: Optional[ACScheduleState] = None
    performance: Optional[PerformanceState] = None


class HealthResponse(BaseModel):
    status: str = "ok"
