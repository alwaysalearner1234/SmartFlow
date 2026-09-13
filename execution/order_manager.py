"""
Order Management System (OMS).
Manages order state machine, unique order IDs, child order dispatching,
execution fills, partial fills, cancellations, and remaining inventory tracking.
"""

import uuid
from typing import Dict, List, Optional
from data.contracts import (
    Order,
    OrderStatus,
    OrderSide,
    OrderType,
    Fill,
    LiquidityType,
)


class OrderManager:
    """Manages order lifecycles, states, fills, and ledger tracking."""

    def __init__(self, parent_order_id: Optional[str] = None):
        self.parent_order_id = parent_order_id or f"PARENT-{uuid.uuid4().hex[:8].upper()}"
        self.orders: Dict[str, Order] = {}
        self.fills: List[Fill] = []
        self._total_filled_quantity: float = 0.0

    def create_order(
        self,
        timestamp: float,
        side: OrderSide,
        order_type: OrderType,
        quantity: float,
        price: Optional[float] = None,
        time_in_force: str = "GTC",
    ) -> Order:
        """Creates a new child order in PENDING status."""
        order_id = f"ORD-{uuid.uuid4().hex[:8].upper()}"
        order = Order(
            order_id=order_id,
            parent_id=self.parent_order_id,
            timestamp=timestamp,
            side=side,
            order_type=order_type,
            quantity=round(quantity, 4),
            price=price,
            time_in_force=time_in_force,
            status=OrderStatus.PENDING,
        )
        self.orders[order_id] = order
        return order

    def submit_order(self, order_id: str, timestamp: float) -> Order:
        """Transitions order from PENDING to SUBMITTED."""
        order = self.orders[order_id]
        if order.status == OrderStatus.PENDING:
            order.status = OrderStatus.SUBMITTED
            order.updated_at = timestamp
        return order

    def record_fill(
        self,
        order_id: str,
        timestamp: float,
        price: float,
        quantity: float,
        liquidity_type: LiquidityType,
        fee: float = 0.0,
        slippage: float = 0.0,
    ) -> Fill:
        """Records a fill, updates order state and cumulative execution counts."""
        order = self.orders[order_id]
        fill_qty = min(quantity, order.remaining_quantity)

        fill = Fill(
            fill_id=f"FILL-{uuid.uuid4().hex[:8].upper()}",
            order_id=order_id,
            parent_id=self.parent_order_id,
            timestamp=timestamp,
            price=price,
            quantity=fill_qty,
            liquidity_type=liquidity_type,
            fee=fee,
            slippage=slippage,
        )

        self.fills.append(fill)
        order.filled_quantity += fill_qty
        self._total_filled_quantity += fill_qty
        order.updated_at = timestamp

        if order.remaining_quantity <= 1e-5:
            order.status = OrderStatus.FILLED
        else:
            order.status = OrderStatus.PARTIALLY_FILLED

        return fill

    def cancel_order(self, order_id: str, timestamp: float) -> Optional[Order]:
        """Cancels an active unfilled or partially filled order."""
        if order_id not in self.orders:
            return None
        order = self.orders[order_id]
        if order.is_active:
            order.status = OrderStatus.CANCELLED
            order.updated_at = timestamp
            return order
        return None

    def cancel_all_active_orders(self, timestamp: float) -> List[Order]:
        """Cancels all currently open/active orders."""
        cancelled = []
        for order in self.orders.values():
            if order.is_active:
                order.status = OrderStatus.CANCELLED
                order.updated_at = timestamp
                cancelled.append(order)
        return cancelled

    @property
    def total_filled_quantity(self) -> float:
        return self._total_filled_quantity

    def get_active_orders(self) -> List[Order]:
        return [o for o in self.orders.values() if o.is_active]

    def get_order_history(self) -> List[Order]:
        return list(self.orders.values())

    def get_fill_history(self) -> List[Fill]:
        return list(self.fills)
