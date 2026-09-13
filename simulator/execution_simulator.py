"""
End-to-End Execution Simulator.
Executes an individual strategy across a deterministic market snapshot stream,
recording all order states, fills, decisions, and trajectory checkpoints.
"""

from typing import List, Dict, Any, Optional
import numpy as np
import pandas as pd

from data.contracts import (
    MarketSnapshot,
    OrderSide,
    OrderType,
    ExecutionResult,
    ExecutionDecision,
    PredictionResult,
)
from features import extract_snapshot_features_dict
from execution.order_manager import OrderManager
from execution.strategy_engine import DynamicStrategyEngine
from simulator.fill_model import FillModel
from simulator.order_book_simulator import OrderBookSimulator


class ExecutionSimulator:
    """Orchestrates strategy execution against a market stream."""

    def __init__(
        self,
        strategy_engine: Optional[DynamicStrategyEngine] = None,
        fill_model: Optional[FillModel] = None,
        ob_simulator: Optional[OrderBookSimulator] = None,
    ):
        self.strategy_engine = strategy_engine or DynamicStrategyEngine()
        self.fill_model = fill_model or FillModel()
        self.ob_simulator = ob_simulator or OrderBookSimulator()

    def run_execution(
        self,
        strategy_name: str,
        snapshots: List[MarketSnapshot],
        total_quantity: float = 1000.0,
        horizon_sec: float = 60.0,
        side: OrderSide = OrderSide.BUY,
        precomputed_features: Optional[pd.DataFrame] = None,
    ) -> ExecutionResult:
        """
        Runs a complete execution loop for a target order quantity and horizon.
        """
        if not snapshots:
            raise ValueError("Snapshots list cannot be empty.")

        # Precompute features if not provided
        if precomputed_features is not None and len(precomputed_features) == len(snapshots):
            df_features = precomputed_features
        else:
            from features import build_feature_pipeline
            df_features = build_feature_pipeline(snapshots)

        order_manager = OrderManager()
        start_time = snapshots[0].timestamp
        end_time = start_time + horizon_sec
        arrival_price = snapshots[0].mid_price

        remaining_qty = total_quantity
        trajectory: List[Dict[str, Any]] = []

        last_decision_time = -1.0
        decision_interval_sec = 2.0  # Strategy evaluates every 2 seconds or on fill

        for tick_idx, snap in enumerate(snapshots):
            elapsed_time = snap.timestamp - start_time
            features_dict = df_features.iloc[tick_idx].to_dict()

            # 1. Process active resting orders against current market tick
            active_orders = order_manager.get_active_orders()
            if active_orders:
                pred = self.strategy_engine.predictor.predict(features_dict, side=side, timestamp=snap.timestamp)
                for active_ord in active_orders:
                    fill = self.fill_model.simulate_fill(
                        order=active_ord,
                        snapshot=snap,
                        adverse_risk=pred.probability,
                    )
                    if fill is not None:
                        order_manager.record_fill(
                            order_id=active_ord.order_id,
                            timestamp=fill.timestamp,
                            price=fill.price,
                            quantity=fill.quantity,
                            liquidity_type=fill.liquidity_type,
                            fee=fill.fee,
                            slippage=fill.slippage,
                        )
                        remaining_qty = max(0.0, total_quantity - order_manager.total_filled_quantity)
                        # Apply market impact to simulated book
                        snap = self.ob_simulator.apply_fill_impact(snap, fill, side)

            # Check if execution finished
            if remaining_qty <= 1e-4:
                break

            # Check horizon expiry
            if snap.timestamp >= end_time:
                # Urgent deadline market sweep for any remaining quantity
                if remaining_qty > 0:
                    sweep_ord = order_manager.create_order(
                        timestamp=snap.timestamp,
                        side=side,
                        order_type=OrderType.MARKET,
                        quantity=remaining_qty,
                    )
                    order_manager.submit_order(sweep_ord.order_id, snap.timestamp)
                    fill = self.fill_model.simulate_fill(sweep_ord, snap, adverse_risk=0.5)
                    if fill:
                        order_manager.record_fill(
                            order_id=sweep_ord.order_id,
                            timestamp=fill.timestamp,
                            price=fill.price,
                            quantity=fill.quantity,
                            liquidity_type=fill.liquidity_type,
                            slippage=fill.slippage,
                        )
                remaining_qty = max(0.0, total_quantity - order_manager.total_filled_quantity)
                break

            # 2. Trigger strategy decision at intervals
            if (snap.timestamp - last_decision_time >= decision_interval_sec) and remaining_qty > 0:
                decision = self.strategy_engine.evaluate_and_decide(
                    strategy_name=strategy_name,
                    snapshot=snap,
                    features_dict=features_dict,
                    remaining_quantity=remaining_qty,
                    initial_quantity=total_quantity,
                    elapsed_time=elapsed_time,
                    total_horizon=horizon_sec,
                    side=side,
                )
                last_decision_time = snap.timestamp

                # Record trajectory point
                trajectory.append({
                    "timestamp": snap.timestamp,
                    "elapsed_time": round(elapsed_time, 2),
                    "remaining_quantity": round(remaining_qty, 2),
                    "filled_quantity": round(order_manager.total_filled_quantity, 2),
                    "mid_price": snap.mid_price,
                    "decision_strategy": decision.strategy,
                    "decision_quantity": decision.quantity,
                    "urgency": decision.urgency,
                    "risk_score": decision.risk_score,
                    "reason": decision.reason,
                })

                # If decision specifies action and order size > 0
                if decision.quantity > 0:
                    # Cancel existing stale active orders if aggressive or new slice
                    if "aggressive" in decision.strategy.lower() or "market" in decision.strategy.lower():
                        order_manager.cancel_all_active_orders(snap.timestamp)
                        order_type = OrderType.MARKET
                    else:
                        order_type = OrderType.LIMIT

                    new_ord = order_manager.create_order(
                        timestamp=snap.timestamp,
                        side=side,
                        order_type=order_type,
                        quantity=decision.quantity,
                        price=decision.limit_price,
                    )
                    order_manager.submit_order(new_ord.order_id, snap.timestamp)

                    # Immediate fill check for market or crossing orders
                    if order_type == OrderType.MARKET or (
                        decision.limit_price and (
                            (side == OrderSide.BUY and decision.limit_price >= snap.best_ask) or
                            (side == OrderSide.SELL and decision.limit_price <= snap.best_bid)
                        )
                    ):
                        fill = self.fill_model.simulate_fill(new_ord, snap, adverse_risk=decision.risk_score)
                        if fill:
                            order_manager.record_fill(
                                order_id=new_ord.order_id,
                                timestamp=fill.timestamp,
                                price=fill.price,
                                quantity=fill.quantity,
                                liquidity_type=fill.liquidity_type,
                                slippage=fill.slippage,
                            )
                            remaining_qty = max(0.0, total_quantity - order_manager.total_filled_quantity)
                            snap = self.ob_simulator.apply_fill_impact(snap, fill, side)

        # 3. Calculate finalized metrics
        fills = order_manager.get_fill_history()
        orders = order_manager.get_order_history()
        executed_qty = sum(f.quantity for f in fills)

        if executed_qty > 0:
            avg_exec_price = sum(f.price * f.quantity for f in fills) / executed_qty
        else:
            avg_exec_price = arrival_price

        # Implementation shortfall
        if side == OrderSide.BUY:
            shortfall_dollar = (avg_exec_price - arrival_price) * executed_qty
        else:
            shortfall_dollar = (arrival_price - avg_exec_price) * executed_qty

        shortfall_bps = (shortfall_dollar / (executed_qty * arrival_price + 1e-6)) * 10000.0
        slippage = sum(f.slippage * f.quantity for f in fills) / max(1.0, executed_qty)
        total_cost = shortfall_dollar
        fill_rate = min(1.0, executed_qty / max(1.0, total_quantity))
        completion_rate = round(fill_rate * 100.0, 2)
        exec_time = (fills[-1].timestamp - start_time) if fills else 0.0
        num_partial = sum(1 for o in orders if o.status == OrderType.LIMIT and o.filled_quantity > 0 and o.remaining_quantity > 0)

        # Expected cost from Almgren-Chriss model
        ac_cost = self.strategy_engine.ac_model.estimate_cost(total_quantity, horizon_sec, arrival_price)["expected_cost"]

        return ExecutionResult(
            strategy_name=strategy_name,
            side=side,
            total_quantity=total_quantity,
            executed_quantity=round(executed_qty, 4),
            avg_execution_price=round(avg_exec_price, 4),
            arrival_price=round(arrival_price, 4),
            implementation_shortfall=round(shortfall_dollar, 4),
            implementation_shortfall_bps=round(shortfall_bps, 2),
            slippage=round(slippage, 4),
            market_impact=round(abs(shortfall_dollar * 0.4), 4),
            total_cost=round(total_cost, 4),
            fill_rate=round(fill_rate, 4),
            completion_rate=completion_rate,
            execution_time_sec=round(exec_time, 2),
            num_orders=len(orders),
            num_fills=len(fills),
            num_partial_fills=num_partial,
            adverse_selection_cost=round(max(0.0, shortfall_dollar * 0.35), 4),
            expected_cost=round(ac_cost, 4),
            fills=fills,
            orders=orders,
            trajectory=trajectory,
        )
