"""
Venue Routing Dashboard Component.
Displays simulated multi-venue quotes, smart order routing allocations,
cost/risk comparative matrix, and natural language routing rationale.
"""

import streamlit as st
import pandas as pd
from data.contracts import VenueRoutingResult, OrderSide
from dashboard.charts.plots import plot_venue_allocations


def render_venue_routing_component(routing_res: VenueRoutingResult):
    """
    Renders the simulated Venue Routing section.
    """
    st.markdown("## 🌐 Venue Routing")
    st.caption(
        "Simulated Smart Order Routing (SOR) evaluating multi-exchange quotes, available depth, "
        "spread crossing costs, and toxicity risk across fragmented liquidity venues."
    )

    # Top summary metrics
    v1, v2, v3, v4 = st.columns(4)
    v1.metric("Total Order Size", f"{routing_res.total_quantity:,.0f} units")
    v2.metric("Effective Avg Price", f"${routing_res.avg_price:.2f}")
    v3.metric("Effective Spread", f"{routing_res.effective_spread_bps:.1f} bps")
    v4.metric("Est. Execution Cost", f"${routing_res.total_cost_est:.2f}")

    # Layout: Table on left, Visual Allocation Chart on right
    col_tbl, col_chart = st.columns([1.3, 1.0])

    with col_tbl:
        st.markdown("##### Multi-Venue Liquidity & Execution Matrix")
        rows = []
        is_buy = routing_res.side == OrderSide.BUY

        for v in routing_res.venues:
            avail_depth = v.ask_depth if is_buy else v.bid_depth
            price = v.best_ask if is_buy else v.best_bid
            rows.append({
                "Venue": v.venue_name,
                "Price": f"${price:.2f}",
                "Avail Depth": f"{avail_depth:,.0f}",
                "Spread (bps)": f"{v.spread_bps:.1f}",
                "Est Cost": f"{v.est_cost_bps:.1f} bps",
                "Risk": f"{v.execution_risk:.2f}",
                "Liquidity": f"{v.liquidity_score:.2f}",
                "Routed Qty": f"{v.routed_quantity:,.0f}",
                "Allocation": f"{v.allocation_pct:.1f}%",
            })

        df_venues = pd.DataFrame(rows)
        st.dataframe(df_venues, use_container_width=True, hide_index=True)

    with col_chart:
        st.markdown("##### Order Allocation Distribution")
        fig_alloc = plot_venue_allocations(routing_res)
        st.plotly_chart(fig_alloc, use_container_width=True)

    # Dynamic explanation box
    st.markdown(
        f"""
        <div style="background-color: #141923; border: 1px solid #1E2638; border-left: 4px solid #00F0FF;
                    border-radius: 6px; padding: 12px 16px; margin-top: 8px;">
            <div style="font-size: 11px; color: #8E99AB; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 4px;">
                💡 Smart Order Routing Rationale
            </div>
            <div style="font-size: 13px; color: #E2E8F0; line-height: 1.5;">
                {routing_res.explanation}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
