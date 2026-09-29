"""
Execution context builder.

Builds an :class:`~data.contracts.ExecutionContext` once per decision cycle,
combining adverse-selection risk, NVIDIA forecast, Almgren-Chriss schedule
data, and market microstructure features from the snapshot and feature dict.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from data.contracts import (
    ExecutionContext,
    ForecastResult,
    MarketSnapshot,
    PredictionResult,
    ModelStatus,
)
from strategies.base import BaseExecutionStrategy
from execution.almgren_chriss import AlmgrenChrissModel


def build_execution_context(
    snapshot: MarketSnapshot,
    features_dict: Dict[str, Any],
    prediction: PredictionResult,
    forecast: Optional[ForecastResult],
    remaining_qty: float,
    initial_qty: float,
    elapsed: float,
    total_horizon: float,
    ac_model: AlmgrenChrissModel,
    side: OrderSide,
) -> ExecutionContext:
    """
    Build an :class:`~data.contracts.ExecutionContext` for the current decision tick.

    Computes urgency using :meth:`BaseExecutionStrategy.calculate_urgency`,
    generates the AC schedule on the remaining quantity/time, and fills
    spread/depth/volatility from ``features_dict``.

    Args:
        snapshot: Current market snapshot.
        features_dict: Feature dictionary from ``extract_snapshot_features_dict``.
        prediction: Adverse-selection ``PredictionResult`` from the predictor.
        forecast: Optional ``ForecastResult`` from the NVIDIA forecaster.
        remaining_qty: Remaining order quantity.
        initial_qty: Initial order quantity (used for ratio calculations).
        elapsed: Elapsed time since execution start.
    total_horizon: Total execution horizon in seconds.
        horizon: Total execution horizon in seconds.
        ac_model: Almgren-Chriss model for schedule generation.
        side: Order side (BUY or SELL).

    Returns:
        A populated :class:`~data.contracts.ExecutionContext` instance.
    """
    urgency = 0.5 * min(1.0, elapsed / total_horizon) + 0.5 * (remaining_qty / max(1.0, initial_qty))
    urgency = round(float(min(1.0, max(0.0, urgency))), 4)

    # Generate Almgren-Chriss schedule for the remaining quantity and remaining time
    time_rem = max(1.0, total_horizon - elapsed)
    sched = ac_model.generate_schedule(
        total_quantity=remaining_qty,
        horizon_sec=time_rem,
        num_slices=max(2, int(time_rem / 5.0)),
        initial_price=snapshot.mid_price,
    )

    ac_slice_quantity = float(sched.trade_sizes[0]) if len(sched.trade_sizes) > 0 else remaining_qty
    ac_expected_cost = float(sched.expected_cost)

    # Fill microstructure features from the features dict
    spread_bps = features_dict.get("spread_bps", None)
    top_of_book_depth = features_dict.get("depth_imbalance_l1", None)  # reuse key as placeholder
    volatility = features_dict.get("volatility_std_10", None)

    # If spread_bps not in features, try to derive from snapshot
    if spread_bps is None and snapshot.best_bid > 0 and snapshot.best_ask > 0:
        from features.spread import calculate_spread_bps
        spread_abs = snapshot.best_ask - snapshot.best_bid
        spread_bps = calculate_spread_bps(spread_abs, snapshot.mid_price)

    # If top_of_book_depth not provided, use best bid/ask sizes
    if top_of_book_depth is None:
        best_bid_size = snapshot.best_bid_size if snapshot.bids else 0.0
        best_ask_size = snapshot.best_ask_size if snapshot.asks else 0.0
        top_of_book_depth = min(best_bid_size, best_ask_size)

    if volatility is None:
        volatility = 0.20  # default fallback

    return ExecutionContext(
        timestamp=snapshot.timestamp,
        remaining_quantity=remaining_qty,
        initial_quantity=initial_qty,
        elapsed_time=elapsed,
        total_horizon=total_horizon,
        urgency=urgency,
        ac_slice_quantity=ac_slice_quantity,
        ac_expected_cost=ac_expected_cost,
        ac_schedule_index=0,
        adverse_risk=prediction,
        forecast=forecast,
        spread_bps=spread_bps,
        top_of_book_depth=top_of_book_depth,
        volatility=volatility,
    )