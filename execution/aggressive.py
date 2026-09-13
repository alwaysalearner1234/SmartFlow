"""
Aggressive Spread-Crossing Execution Strategy.
Crosses the bid-ask spread to take immediate liquidity when adverse-selection risk
or deadline urgency justifies paying the half-spread cost.
"""

from typing import Any
from data.contracts import MarketSnapshot, ExecutionDecision, OrderSide
from strategies.base import BaseExecutionStrategy


class AggressiveStrategy(BaseExecutionStrategy):
    """Crosses the spread to execute immediately as taker."""

    def __init__(self, max_spread_bps: float = 50.0):
        super().__init__(name="Aggressive")
        self.max_spread_bps = max_spread_bps

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
                reason="Aggressive execution complete.",
                remaining_quantity=0.0,
            )

        risk_score = kwargs.get("risk_score", 0.7)
        initial_qty = kwargs.get("initial_quantity", remaining_quantity)
        rem_ratio = remaining_quantity / max(1.0, initial_qty)
        urgency = max(0.8, self.calculate_urgency(elapsed_time, total_horizon, rem_ratio))

        # Size: sweep visible opposite depth up to remaining
        opposite_depth = snapshot.best_ask_size if side == OrderSide.BUY else snapshot.best_bid_size
        slice_size = min(remaining_quantity, max(50.0, opposite_depth * 0.5))

        # Crossing price
        limit_p = snapshot.best_ask if side == OrderSide.BUY else snapshot.best_bid

        return ExecutionDecision(
            timestamp=snapshot.timestamp,
            strategy=self.name,
            side=side,
            quantity=round(slice_size, 2),
            limit_price=limit_p,
            urgency=urgency,
            risk_score=risk_score,
            reason=f"Aggressive sweep: crossing spread at {limit_p:.2f} for {slice_size:.1f} units due to high urgency/risk ({risk_score:.2f}).",
            remaining_quantity=remaining_quantity,
        )
