"""
Dashboard API Routes.
Exposes GET /api/v1/dashboard/snapshot adapting the existing SmartFlow
market simulator, ML predictor, execution engine, and backtester
to the React frontend contract (frontend/src/types.ts).
"""

from datetime import datetime, timezone, timedelta
from typing import Optional, List, Dict, Any
import numpy as np
from fastapi import APIRouter, Query

from data.contracts import OrderSide, MarketSnapshot, ExecutionResult, BacktestResult
from data.generator import MarketDataGenerator
from features import build_feature_pipeline
from models.predictor import AdverseSelectionPredictor
from simulator.execution_simulator import ExecutionSimulator
from backtest.backtester import Backtester

from api.schemas import (
    DashboardSnapshot,
    MarketState,
    BookLevel,
    FeaturePoint,
    FeatureState,
    ExecutionState,
    ExecutionPoint,
    RiskState,
    PerformanceState,
    StrategyResult,
    Action,
    Strategy,
    HealthResponse,
)

router = APIRouter(prefix="/api/v1", tags=["Dashboard"])

BASE_EPOCH = 1700000000.0


def _format_iso(timestamp: float) -> str:
    """Converts a float timestamp to UTC ISO 8601 string."""
    if timestamp > 1000000000.0:
        dt = datetime.fromtimestamp(timestamp, tz=timezone.utc)
    else:
        dt = datetime.fromtimestamp(BASE_EPOCH + timestamp, tz=timezone.utc)
    return dt.isoformat().replace("+00:00", "Z")


def _map_action(strategy_text: str, quantity: float = 1.0) -> Action:
    """Maps internal strategy name to frontend Action ('passive' | 'aggressive' | 'wait')."""
    s = strategy_text.lower()
    if quantity <= 0.0 or "withdrawn" in s:
        return "wait"
    if "passive" in s:
        return "passive"
    if "aggressive" in s or "market" in s:
        return "aggressive"
    return "passive"


def _finite_metric(value: Any) -> Optional[float]:
    """Keep unavailable feature values as null instead of inventing zeroes."""
    if value is None:
        return None
    number = float(value)
    return number if np.isfinite(number) else None


class DashboardDataService:
    """Coordinates existing backend modules to assemble a typed DashboardSnapshot."""

    def __init__(self):
        self.generator = MarketDataGenerator()
        self.predictor = AdverseSelectionPredictor()
        self.simulator = ExecutionSimulator()
        self.backtester = Backtester(execution_simulator=self.simulator)

        # In-memory cached snapshot for low-latency response
        self._cached_snapshot: Optional[DashboardSnapshot] = None
        self._cache_time: float = 0.0

    def generate_fresh_snapshot(
        self,
        scenario_key: str = "normal_market",
        order_size: float = 1000.0,
        horizon_sec: float = 60.0,
        side: OrderSide = OrderSide.BUY,
    ) -> DashboardSnapshot:
        """
        Executes market simulation, risk prediction, order routing,
        and multi-strategy backtesting to build the full dashboard snapshot.
        """
        # 1. Market Stream & Feature Pipeline
        snapshots = self.generator.generate_scenario_stream(
            scenario_key=scenario_key,
            num_ticks=int(max(100, (horizon_sec / 0.5) + 30)),
            dt=0.5,
        )
        df_features = build_feature_pipeline(snapshots)
        latest_snap: MarketSnapshot = snapshots[-1]
        latest_feats: Dict[str, Any] = df_features.iloc[-1].to_dict()

        # Build MarketState
        bids = [
            BookLevel(price=round(p, 2), quantity=round(s, 2))
            for p, s in sorted(latest_snap.bids, key=lambda x: x[0], reverse=True)[:5]
        ]
        asks = [
            BookLevel(price=round(p, 2), quantity=round(s, 2))
            for p, s in sorted(latest_snap.asks, key=lambda x: x[0])[:5]
        ]
        # The Phase 1 field name is retained for compatibility. Its bounded
        # value is L1 depth imbalance; raw order-flow imbalance is in features.
        ofi = float(np.clip(latest_feats.get("depth_imbalance_l1", 0.0), -1.0, 1.0))

        market_state = MarketState(
            timestamp=_format_iso(latest_snap.timestamp),
            symbol="BTC-USD",
            bids=bids,
            asks=asks,
            mid_price=round(latest_snap.mid_price, 2),
            spread=round(latest_snap.spread, 2),
            order_flow_imbalance=round(ofi, 4),
        )

        feature_state = FeatureState(
            points=[
                FeaturePoint(
                    timestamp=_format_iso(float(row["timestamp"])),
                    spread_bps=_finite_metric(row.get("spread_bps")),
                    depth_imbalance_l1=_finite_metric(row.get("depth_imbalance_l1")),
                    ofi_instant=_finite_metric(row.get("ofi_instant")),
                    volatility_std_10=_finite_metric(row.get("volatility_std_10")),
                    momentum_ret_5=_finite_metric(row.get("momentum_ret_5")),
                    total_bid_depth=float(row["total_bid_depth"]),
                    total_ask_depth=float(row["total_ask_depth"]),
                )
                for row in df_features.tail(80).to_dict(orient="records")
            ]
        )

        # 2. Risk State
        # Note: FillModel does not currently expose a dashboard fill_probability estimator,
        # and PredictionResult.prediction_horizon is measured in ticks (not milliseconds).
        # Per contract instructions, we do not fabricate values or invent conversions;
        # the risk section remains null.
        risk_state: Optional[RiskState] = None

        # 3. Execution Simulation via ExecutionSimulator
        exec_res: ExecutionResult = self.simulator.run_execution(
            strategy_name="Proposed (ML + AC)",
            snapshots=snapshots,
            total_quantity=order_size,
            horizon_sec=horizon_sec,
            side=side,
        )

        # Index actual fills by timestamp if available
        fills_by_ts: Dict[float, List[float]] = {}
        for fill in exec_res.fills:
            fills_by_ts.setdefault(fill.timestamp, []).append(fill.price)

        # Map trajectory points
        trajectory_points: List[ExecutionPoint] = []
        for pt in exec_res.trajectory:
            pt_ts = float(pt.get("timestamp", 0.0))
            action = _map_action(pt.get("decision_strategy", ""), pt.get("decision_quantity", 1.0))

            # Use actual execution price if a fill occurred at this trajectory point, otherwise None
            if pt_ts in fills_by_ts:
                actual_exec_price = round(float(np.mean(fills_by_ts[pt_ts])), 2)
            else:
                actual_exec_price = None

            trajectory_points.append(
                ExecutionPoint(
                    timestamp=_format_iso(pt_ts),
                    remaining_quantity=round(float(pt.get("remaining_quantity", 0.0)), 2),
                    filled_quantity=round(float(pt.get("filled_quantity", 0.0)), 2),
                    reference_price=round(float(pt.get("mid_price", 0.0)), 2),
                    execution_price=actual_exec_price,
                    action=action,
                )
            )

        last_action = trajectory_points[-1].action if trajectory_points else "passive"
        last_urgency = float(exec_res.trajectory[-1].get("urgency", 0.0)) if exec_res.trajectory else 0.0

        executed_qty = min(exec_res.total_quantity, exec_res.executed_quantity)
        rem_qty = max(0.0, exec_res.total_quantity - executed_qty)

        execution_state = ExecutionState(
            order_id=f"ORD-SMARTFLOW-{int(latest_snap.timestamp)}",
            symbol="BTC-USD",
            side="buy" if side == OrderSide.BUY else "sell",
            target_quantity=round(exec_res.total_quantity, 2),
            filled_quantity=round(executed_qty, 2),
            remaining_quantity=round(rem_qty, 2),
            selected_action=last_action,
            execution_score=round(float(np.clip(last_urgency, 0.0, 1.0)), 2),
            trajectory=trajectory_points,
        )

        # 4. Multi-Strategy Performance via Backtester
        backtest_res: BacktestResult = self.backtester.run_backtest(
            snapshots=snapshots,
            scenario_name="Normal Market",
            scenario_description="Balanced microstructure conditions",
            total_quantity=order_size,
            horizon_sec=horizon_sec,
            side=side,
        )

        strategy_map: Dict[str, Strategy] = {
            "Market": "market",
            "TWAP": "twap",
            "VWAP": "vwap",
            "Almgren-Chriss": "almgren_chriss",
            "Proposed (ML + AC)": "proposed",
        }

        perf_results: List[StrategyResult] = []
        for backend_name, fe_strat in strategy_map.items():
            res = backtest_res.strategy_results.get(backend_name)
            if res:
                notional = max(1.0, res.executed_quantity * res.arrival_price)
                slippage_bps = round((res.slippage / max(1e-4, res.arrival_price)) * 10000.0, 2)
                impact_bps = round((res.market_impact / notional) * 10000.0, 2)
                adverse_bps = round((res.adverse_selection_cost / notional) * 10000.0, 2)
                perf_results.append(
                    StrategyResult(
                        strategy=fe_strat,
                        filled_quantity=round(min(res.total_quantity, res.executed_quantity), 2),
                        fill_rate=round(res.fill_rate, 4),
                        slippage_bps=slippage_bps,
                        market_impact_bps=impact_bps,
                        implementation_shortfall_bps=round(res.implementation_shortfall_bps, 2),
                        adverse_selection_bps=adverse_bps,
                    )
                )

        perf_state = PerformanceState(
            run_id=f"BT-{int(latest_snap.timestamp)}",
            completed_at=_format_iso(latest_snap.timestamp),
            results=perf_results,
        )

        snapshot = DashboardSnapshot(
            schema_version="1.0",
            market=market_state,
            features=feature_state,
            execution=execution_state,
            risk=risk_state,
            performance=perf_state,
        )

        self._cached_snapshot = snapshot
        return snapshot

    def get_snapshot(self, refresh: bool = False) -> DashboardSnapshot:
        """Returns cached snapshot or generates fresh data."""
        if self._cached_snapshot is None or refresh:
            return self.generate_fresh_snapshot()
        return self._cached_snapshot


# Singleton service instance
service = DashboardDataService()


@router.get("/health", response_model=HealthResponse)
def health_check():
    """Health check endpoint."""
    return HealthResponse(status="ok")


@router.get("/dashboard/snapshot", response_model=DashboardSnapshot)
def get_dashboard_snapshot(refresh: bool = Query(False, description="Force re-generation of backend run")):
    """
    Returns the unified SmartFlow dashboard snapshot matching
    the frontend/src/types.ts Phase 1 contract.
    """
    return service.get_snapshot(refresh=refresh)
