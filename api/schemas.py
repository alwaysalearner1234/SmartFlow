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


class RiskState(BaseModel):
    timestamp: str
    adverse_selection_probability: float
    fill_probability: float
    prediction_horizon_ms: int
    expected_market_impact_bps: Optional[float] = None
    expected_shortfall_bps: Optional[float] = None


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
    execution: Optional[ExecutionState] = None
    risk: Optional[RiskState] = None
    performance: Optional[PerformanceState] = None


class HealthResponse(BaseModel):
    status: str = "ok"
