"""
Order Book State Simulator.
Maintains order-book liquidity consumption and endogenous market impact.
"""

from typing import List, Tuple
from data.contracts import MarketSnapshot, OrderSide, Fill


class OrderBookSimulator:
    """Manages order-book depth depletion and endogenous price impact."""

    def __init__(self, permanent_impact_factor: float = 1e-5, temporary_impact_factor: float = 5e-5):
        self.permanent_impact = permanent_impact_factor
        self.temporary_impact = temporary_impact_factor
        self.cumulative_permanent_drift: float = 0.0

    def apply_fill_impact(self, snapshot: MarketSnapshot, fill: Fill, side: OrderSide) -> MarketSnapshot:
        """
        Adjusts the snapshot after a fill:
        - Depletes visible depth
        - Shifts prices according to market impact
        """
        # Directional impact (+ for buy pushes prices up, - for sell pushes down)
        sign = 1.0 if side == OrderSide.BUY else -1.0
        perm_delta = sign * self.permanent_impact * fill.quantity
        temp_delta = sign * self.temporary_impact * fill.quantity
        self.cumulative_permanent_drift += perm_delta

        total_shift = perm_delta + temp_delta

        # Apply price shifts to bids and asks
        new_bids: List[Tuple[float, float]] = []
        for p, s in snapshot.bids:
            new_p = round(p + total_shift, 4)
            # Deplete size if at top level
            new_s = max(1.0, s - (fill.quantity if side == OrderSide.SELL else 0.0))
            new_bids.append((new_p, new_s))

        new_asks: List[Tuple[float, float]] = []
        for p, s in snapshot.asks:
            new_p = round(p + total_shift, 4)
            new_s = max(1.0, s - (fill.quantity if side == OrderSide.BUY else 0.0))
            new_asks.append((new_p, new_s))

        # Ensure uncrossed
        if new_bids and new_asks and new_bids[0][0] >= new_asks[0][0]:
            diff = (new_bids[0][0] - new_asks[0][0]) + 0.01
            new_asks[0] = (round(new_asks[0][0] + diff, 4), new_asks[0][1])

        return MarketSnapshot(
            timestamp=snapshot.timestamp,
            sequence_id=snapshot.sequence_id,
            bids=new_bids,
            asks=new_asks,
            last_trade_price=fill.price,
            last_trade_size=fill.quantity,
            last_trade_side=side,
            recent_trades=snapshot.recent_trades,
        )
