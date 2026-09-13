"""
Time-Weighted Average Price (TWAP) Strategy.
Divides the total order quantity uniformly across discrete time intervals
over the specified execution horizon.
"""

from typing import Any
from data.contracts import MarketSnapshot, ExecutionDecision, OrderSide
from strategies.base import BaseExecutionStrategy


class TWAPStrategy(BaseExecutionStrategy):
    """Uniform time-slicing execution strategy."""

    def __init__(self, num_slices: int = 12):
        super().__init__(name="TWAP")
        self.num_slices = num_slices

    def compute_decision(
        self,
        snapshot: MarketSnapshot,
        remaining_quantity: float,
        elapsed_time: float,
        total_horizon: float,
        side: OrderSide = OrderSide.BUY,
        **kwargs: Any,
    ) -> ExecutionDecision:
        if remaining_quantity <= 0 or total_horizon <= 0:
            return ExecutionDecision(
                timestamp=snapshot.timestamp,
                strategy=self.name,
                side=side,
                quantity=0.0,
                urgency=0.0,
                risk_score=0.0,
                reason="TWAP execution complete.",
                remaining_quantity=0.0,
            )

        interval_sec = total_horizon / self.num_slices
        current_interval = int(elapsed_time / max(0.1, interval_sec))
        remaining_intervals = max(1, self.num_slices - current_interval)

        # Slice size is proportional to remaining intervals
        slice_size = min(remaining_quantity, remaining_quantity / remaining_intervals)
        slice_size = round(slice_size, 2)

        # Urgent near deadline
        urgency = min(1.0, elapsed_time / total_horizon)
        limit_p = snapshot.best_ask if side == OrderSide.BUY else snapshot.best_bid

        return ExecutionDecision(
            timestamp=snapshot.timestamp,
            strategy=self.name,
            side=side,
            quantity=slice_size,
            limit_price=limit_p,
            urgency=round(urgency, 3),
            risk_score=0.5,
            reason=f"TWAP slice {current_interval + 1}/{self.num_slices}: {slice_size:.1f} units.",
            remaining_quantity=remaining_quantity,
            expected_cost=0.0,
        )
