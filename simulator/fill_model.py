"""
Configurable Order Fill Simulator.
Models realistic execution physics including:
- Market order depth walking with multi-level price impact and slippage
- Passive limit order queue position approximation and priority decay
- Adverse selection execution bias (higher fill probability when price plunges against maker)
- Full fills, partial fills, and non-fills.
"""

from typing import List, Tuple, Optional
import numpy as np
from data.contracts import (
    Order,
    OrderType,
    OrderSide,
    MarketSnapshot,
    Fill,
    LiquidityType,
)


class FillModel:
    """Simulates realistic fill outcomes against an order-book snapshot."""

    def __init__(
        self,
        execution_delay_ms: float = 15.0,
        queue_decay_rate: float = 0.5,
        adverse_fill_bias: float = 0.3,
        seed: int = 42,
    ):
        self.execution_delay_sec = execution_delay_ms / 1000.0
        self.queue_decay_rate = queue_decay_rate
        self.adverse_fill_bias = adverse_fill_bias
        self.rng = np.random.default_rng(seed)

    def simulate_fill(
        self,
        order: Order,
        snapshot: MarketSnapshot,
        adverse_risk: float = 0.5,
    ) -> Optional[Fill]:
        """
        Determines if and how an order fills given the snapshot.
        Returns a Fill object or None.
        """
        if not order.is_active or order.remaining_quantity <= 0:
            return None

        # Check execution latency: order must have arrived at exchange
        if snapshot.timestamp < order.timestamp + self.execution_delay_sec:
            return None

        if order.order_type == OrderType.MARKET:
            return self._simulate_market_fill(order, snapshot)
        else:
            return self._simulate_limit_fill(order, snapshot, adverse_risk)

    def _simulate_market_fill(self, order: Order, snapshot: MarketSnapshot) -> Fill:
        """
        Market order executes against the opposing book depth.
        Walks the book levels to calculate volume-weighted fill price.
        """
        target_qty = order.remaining_quantity
        opposing_levels = snapshot.asks if order.side == OrderSide.BUY else snapshot.bids

        if not opposing_levels:
            # Emergency fallback: best price + slippage
            p = snapshot.best_ask if order.side == OrderSide.BUY else snapshot.best_bid
            slippage = p * 0.0005
            exec_p = p + slippage if order.side == OrderSide.BUY else p - slippage
            return Fill(
                fill_id=f"F-{self.rng.integers(100000, 999999)}",
                order_id=order.order_id,
                parent_id=order.parent_id,
                timestamp=snapshot.timestamp,
                price=round(exec_p, 4),
                quantity=target_qty,
                liquidity_type=LiquidityType.TAKER,
                slippage=round(slippage, 4),
            )

        # Walk the book
        filled_qty = 0.0
        total_cost = 0.0
        arrival_p = opposing_levels[0][0]

        for lvl_p, lvl_size in opposing_levels:
            take_size = min(lvl_size, target_qty - filled_qty)
            total_cost += take_size * lvl_p
            filled_qty += take_size
            if filled_qty >= target_qty:
                break

        # If order size exceeded visible top levels, the remainder experiences extra market impact
        if filled_qty < target_qty:
            remaining_deficit = target_qty - filled_qty
            last_p = opposing_levels[-1][0]
            penalty = 0.001 * (remaining_deficit / 100.0)
            deep_p = last_p * (1.0 + penalty) if order.side == OrderSide.BUY else last_p * (1.0 - penalty)
            total_cost += remaining_deficit * deep_p
            filled_qty += remaining_deficit

        avg_price = total_cost / filled_qty
        slippage = abs(avg_price - arrival_p)

        return Fill(
            fill_id=f"F-{self.rng.integers(100000, 999999)}",
            order_id=order.order_id,
            parent_id=order.parent_id,
            timestamp=snapshot.timestamp,
            price=round(avg_price, 4),
            quantity=round(filled_qty, 4),
            liquidity_type=LiquidityType.TAKER,
            slippage=round(slippage, 4),
        )

    def _simulate_limit_fill(
        self,
        order: Order,
        snapshot: MarketSnapshot,
        adverse_risk: float,
    ) -> Optional[Fill]:
        """
        Passive Limit Order execution:
        Considers price level, queue priority, and adverse selection bias.
        """
        if order.price is None:
            return None

        # Price crossing check: does limit price cross the spread?
        if (order.side == OrderSide.BUY and order.price >= snapshot.best_ask) or \
           (order.side == OrderSide.SELL and order.price <= snapshot.best_bid):
            # Immediate crossing taker fill
            return self._simulate_market_fill(order, snapshot)

        # Resting maker order at or behind quote
        is_touch = (order.side == OrderSide.BUY and abs(order.price - snapshot.best_bid) < 1e-4) or \
                   (order.side == OrderSide.SELL and abs(order.price - snapshot.best_ask) < 1e-4)

        if not is_touch:
            # Order is deep behind the quote; only fills if massive trade flow hits it
            return None

        # Queue fill probability:
        # Base probability of being touched by incoming noise flow ~ 0.35
        # Adverse selection increases fill chance (the "winner's curse" phenomenon:
        # when adverse risk is high, toxic flow sweeps passive quotes)
        adverse_boost = (adverse_risk - 0.5) * self.adverse_fill_bias
        fill_prob = float(np.clip(0.35 + adverse_boost, 0.05, 0.90))

        if self.rng.random() > fill_prob:
            return None  # No fill this tick

        # Partial vs Full fill: fraction of remaining quantity executed
        fill_fraction = float(np.clip(self.rng.beta(2, 2), 0.25, 1.0))
        exec_qty = round(order.remaining_quantity * fill_fraction, 2)
        exec_qty = max(1.0, min(exec_qty, order.remaining_quantity))

        return Fill(
            fill_id=f"F-{self.rng.integers(100000, 999999)}",
            order_id=order.order_id,
            parent_id=order.parent_id,
            timestamp=snapshot.timestamp,
            price=order.price,
            quantity=exec_qty,
            liquidity_type=LiquidityType.MAKER,
            slippage=0.0,  # Maker gets exact limit price
        )
