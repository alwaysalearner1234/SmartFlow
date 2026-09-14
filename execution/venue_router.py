"""
Simulated Multi-Venue Smart Order Router (SOR).
Models fragmented liquidity across multiple electronic execution venues:
- Venue A: Primary ECN (Continuous Order Book)
- Venue B: Dark Pool / ATS (Mid-point Crossing Network)
- Venue C: Retail Exchange (Fast Fill, Moderate Depth)
- Venue D: Institutional Block Pool (Deep Liquidity, Wider Spread)

Allocates order quantity across venues using multi-factor cost and risk optimization:
- Best price & spread crossing cost
- Available book depth
- Execution liquidity score
- Adverse-selection and toxicity risk
Generates transparent, plain-English routing explanations.
"""

from typing import List, Dict, Any, Optional
import numpy as np
from data.contracts import (
    MarketSnapshot,
    OrderSide,
    VenueQuote,
    VenueRoutingResult,
)


class VenueRouter:
    """Smart Order Router across simulated trading venues."""

    def __init__(self, venue_names: Optional[Dict[str, str]] = None):
        self.venue_names = venue_names or {
            "venue_a": "Venue A (Primary ECN)",
            "venue_b": "Venue B (Dark ATS)",
            "venue_c": "Venue C (Retail Flow)",
            "venue_d": "Venue D (Block Pool)",
        }

    def generate_venue_quotes(
        self,
        snapshot: MarketSnapshot,
        adverse_risk_score: float = 0.50,
        regime_volatility: float = 0.20,
    ) -> List[VenueQuote]:
        """
        Generates realistic, deterministic multi-venue quotes anchored to the primary market snapshot.
        """
        mid = snapshot.mid_price if snapshot.mid_price > 0 else 100.0
        base_spread = max(0.01, snapshot.spread)
        base_bid_depth = max(100.0, snapshot.total_bid_depth)
        base_ask_depth = max(100.0, snapshot.total_ask_depth)

        quotes: List[VenueQuote] = []

        # 1. Venue A: Primary ECN (Standard lit market)
        spread_a = base_spread
        bid_a = round(mid - spread_a / 2.0, 2)
        ask_a = round(mid + spread_a / 2.0, 2)
        b_depth_a = round(base_bid_depth * 0.45, 1)
        a_depth_a = round(base_ask_depth * 0.45, 1)
        spread_bps_a = round((spread_a / mid) * 10000.0, 1)
        risk_a = round(float(np.clip(adverse_risk_score, 0.05, 0.95)), 2)
        liq_a = round(float(np.clip((b_depth_a + a_depth_a) / 1000.0, 0.1, 1.0)), 2)
        est_cost_a = round(spread_bps_a * 0.5 + risk_a * 5.0, 1)

        quotes.append(VenueQuote(
            venue_id="venue_a",
            venue_name=self.venue_names["venue_a"],
            best_bid=bid_a,
            best_ask=ask_a,
            bid_depth=b_depth_a,
            ask_depth=a_depth_a,
            spread=round(spread_a, 2),
            spread_bps=spread_bps_a,
            liquidity_score=liq_a,
            execution_risk=risk_a,
            est_cost_bps=est_cost_a,
        ))

        # 2. Venue B: Dark ATS (Mid-point price improvement, lower toxic risk, but queue/depth constraint)
        spread_b = max(0.01, base_spread * 0.4)  # Sub-penny / mid-point pricing
        bid_b = round(mid - spread_b / 2.0, 2)
        ask_b = round(mid + spread_b / 2.0, 2)
        b_depth_b = round(base_bid_depth * 0.30, 1)
        a_depth_b = round(base_ask_depth * 0.30, 1)
        spread_bps_b = round((spread_b / mid) * 10000.0, 1)
        risk_b = round(float(np.clip(adverse_risk_score * 0.75, 0.05, 0.90)), 2)
        liq_b = round(float(np.clip((b_depth_b + a_depth_b) / 800.0, 0.1, 1.0)), 2)
        est_cost_b = round(spread_bps_b * 0.3 + risk_b * 3.5, 1)

        quotes.append(VenueQuote(
            venue_id="venue_b",
            venue_name=self.venue_names["venue_b"],
            best_bid=bid_b,
            best_ask=ask_b,
            bid_depth=b_depth_b,
            ask_depth=a_depth_b,
            spread=round(spread_b, 2),
            spread_bps=spread_bps_b,
            liquidity_score=liq_b,
            execution_risk=risk_b,
            est_cost_bps=est_cost_b,
        ))

        # 3. Venue C: Retail Flow Exchange (Tight inside spread, small depth, low adverse toxicity)
        spread_c = round(base_spread * 0.85, 2)
        bid_c = round(mid - spread_c / 2.0, 2)
        ask_c = round(mid + spread_c / 2.0, 2)
        b_depth_c = round(base_bid_depth * 0.18, 1)
        a_depth_c = round(base_ask_depth * 0.18, 1)
        spread_bps_c = round((spread_c / mid) * 10000.0, 1)
        risk_c = round(float(np.clip(adverse_risk_score * 0.60, 0.05, 0.85)), 2)
        liq_c = round(float(np.clip((b_depth_c + a_depth_c) / 500.0, 0.1, 0.8)), 2)
        est_cost_c = round(spread_bps_c * 0.45 + risk_c * 2.8, 1)

        quotes.append(VenueQuote(
            venue_id="venue_c",
            venue_name=self.venue_names["venue_c"],
            best_bid=bid_c,
            best_ask=ask_c,
            bid_depth=b_depth_c,
            ask_depth=a_depth_c,
            spread=round(spread_c, 2),
            spread_bps=spread_bps_c,
            liquidity_score=liq_c,
            execution_risk=risk_c,
            est_cost_bps=est_cost_c,
        ))

        # 4. Venue D: Institutional Block Pool (Deep depth, wider spread to compensate market maker)
        spread_d = round(base_spread * 1.35, 2)
        bid_d = round(mid - spread_d / 2.0, 2)
        ask_d = round(mid + spread_d / 2.0, 2)
        b_depth_d = round(base_bid_depth * 0.65, 1)
        a_depth_d = round(base_ask_depth * 0.65, 1)
        spread_bps_d = round((spread_d / mid) * 10000.0, 1)
        risk_d = round(float(np.clip(adverse_risk_score * 0.90, 0.05, 0.95)), 2)
        liq_d = round(float(np.clip((b_depth_d + a_depth_d) / 1200.0, 0.1, 1.0)), 2)
        est_cost_d = round(spread_bps_d * 0.6 + risk_d * 4.5, 1)

        quotes.append(VenueQuote(
            venue_id="venue_d",
            venue_name=self.venue_names["venue_d"],
            best_bid=bid_d,
            best_ask=ask_d,
            bid_depth=b_depth_d,
            ask_depth=a_depth_d,
            spread=round(spread_d, 2),
            spread_bps=spread_bps_d,
            liquidity_score=liq_d,
            execution_risk=risk_d,
            est_cost_bps=est_cost_d,
        ))

        return quotes

    def route_order(
        self,
        snapshot: MarketSnapshot,
        order_quantity: float,
        side: OrderSide = OrderSide.BUY,
        adverse_risk_score: float = 0.50,
        regime_volatility: float = 0.20,
    ) -> VenueRoutingResult:
        """
        Calculates optimal order routing allocations across venues.
        """
        venues = self.generate_venue_quotes(snapshot, adverse_risk_score, regime_volatility)
        total_qty = max(1.0, float(order_quantity))

        # Calculate routing desirability score for each venue:
        # Higher score = more depth, lower spread, lower risk, lower cost
        scores = []
        for v in venues:
            avail_depth = v.ask_depth if side == OrderSide.BUY else v.bid_depth
            depth_factor = np.sqrt(max(10.0, avail_depth))
            cost_penalty = 1.0 + (v.est_cost_bps / 10.0)
            risk_penalty = 1.0 + (v.execution_risk * 1.5)
            # Desirability score
            score = (depth_factor * v.liquidity_score) / (cost_penalty * risk_penalty)
            scores.append(max(0.01, score))

        sum_scores = sum(scores)
        raw_weights = [s / sum_scores for s in scores]

        # Allocate quantities
        allocated_quantities = []
        for i, v in enumerate(venues):
            qty = round(total_qty * raw_weights[i], 1)
            allocated_quantities.append(qty)

        # Rebalance discrepancy due to rounding
        diff = total_qty - sum(allocated_quantities)
        if abs(diff) > 0.01:
            # Add difference to venue with highest weight
            max_idx = int(np.argmax(raw_weights))
            allocated_quantities[max_idx] = round(allocated_quantities[max_idx] + diff, 1)

        # Update VenueQuote instances
        weighted_price_sum = 0.0
        weighted_spread_bps_sum = 0.0
        weighted_cost_sum = 0.0

        for i, v in enumerate(venues):
            v.routed_quantity = allocated_quantities[i]
            v.allocation_pct = round((allocated_quantities[i] / total_qty) * 100.0, 1)

            price = v.best_ask if side == OrderSide.BUY else v.best_bid
            weighted_price_sum += price * v.routed_quantity
            weighted_spread_bps_sum += v.spread_bps * v.routed_quantity
            weighted_cost_sum += (v.est_cost_bps / 10000.0 * price) * v.routed_quantity

        avg_price = round(weighted_price_sum / total_qty, 2)
        effective_spread_bps = round(weighted_spread_bps_sum / total_qty, 1)
        total_cost_est = round(weighted_cost_sum, 2)

        # Determine highest and lowest allocated venues for the explanation
        sorted_venues = sorted(venues, key=lambda x: x.routed_quantity, reverse=True)
        top_venue = sorted_venues[0]
        second_venue = sorted_venues[1]

        explanation = (
            f"**{top_venue.venue_name}** received the largest allocation ({top_venue.allocation_pct:.1f}% / {top_venue.routed_quantity:.0f} units) "
            f"because it currently offers superior available depth ({top_venue.ask_depth if side == OrderSide.BUY else top_venue.bid_depth:.0f} units) "
            f"and lower estimated execution cost ({top_venue.est_cost_bps:.1f} bps). "
            f"**{second_venue.venue_name}** was allocated {second_venue.allocation_pct:.1f}% to capture "
            f"{'mid-point price improvement' if 'Dark' in second_venue.venue_name else 'tight inside spread'} "
            f"while diversifying market impact across venues."
        )

        return VenueRoutingResult(
            timestamp=snapshot.timestamp,
            total_quantity=total_qty,
            side=side,
            venues=venues,
            explanation=explanation,
            avg_price=avg_price,
            effective_spread_bps=effective_spread_bps,
            total_cost_est=total_cost_est,
        )
