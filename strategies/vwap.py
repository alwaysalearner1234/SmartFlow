"""
Volume-Weighted Average Price (VWAP) Strategy.
Distributes target order execution in proportion to empirical or expected intraday
volume profiles and real-time trade activity.
"""

from typing import Any
import numpy as np
from data.contracts import MarketSnapshot, ExecutionDecision, OrderSide
from strategies.base import BaseExecutionStrategy


class VWAPStrategy(BaseExecutionStrategy):
    """Volume-tracking execution strategy."""

    def __init__(self, target_participation_rate: float = 0.15):
        super().__init__(name="VWAP")
        self.target_participation_rate = target_participation_rate

    def compute_decision(
        self,
        snapshot: MarketSnapshot,
        remaining_quantity: float,
        elapsed_time: float,
        total_horizon: float,
        side: OrderSide = OrderSide.BUY,
        **kwargs: Any,
    ) -> ExecutionDecision:
        if remaining_quantity <= 0:
            return ExecutionDecision(
                timestamp=snapshot.timestamp,
                strategy=self.name,
                side=side,
                quantity=0.0,
                urgency=0.0,
                risk_score=0.0,
                reason="VWAP execution complete.",
                remaining_quantity=0.0,
            )

        # Estimate recent interval volume from trades or available top-of-book depth
        recent_trade_vol = sum(t.size for t in snapshot.recent_trades)
        visible_depth = snapshot.best_ask_size if side == OrderSide.BUY else snapshot.best_bid_size
        proxy_vol = max(100.0, recent_trade_vol + visible_depth * 0.5)

        # Target slice based on participation rate
        target_slice = proxy_vol * self.target_participation_rate

        # If nearing deadline, ramp up to avoid leftover inventory
        time_ratio = min(1.0, elapsed_time / max(1.0, total_horizon))
        if time_ratio > 0.8:
            target_slice = max(target_slice, remaining_quantity * (time_ratio ** 2))

        slice_size = round(float(min(remaining_quantity, max(10.0, target_slice))), 2)
        urgency = self.calculate_urgency(elapsed_time, total_horizon, remaining_quantity / max(1.0, kwargs.get("initial_quantity", remaining_quantity)))
        limit_p = snapshot.best_ask if side == OrderSide.BUY else snapshot.best_bid

        return ExecutionDecision(
            timestamp=snapshot.timestamp,
            strategy=self.name,
            side=side,
            quantity=slice_size,
            limit_price=limit_p,
            urgency=urgency,
            risk_score=0.5,
            reason=f"VWAP slice: {slice_size:.1f} units tracking ~{self.target_participation_rate*100:.0f}% volume participation.",
            remaining_quantity=remaining_quantity,
            expected_cost=0.0,
        )
