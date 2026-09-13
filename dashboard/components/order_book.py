"""
Order Book Depth Ladder Component.
Renders visual bid/ask price ladders with depth bars and key quote metrics.
"""

import streamlit as st
import pandas as pd
from data.contracts import MarketSnapshot


def render_order_book_component(snapshot: MarketSnapshot):
    """Renders the live order book ladder and quote metrics."""
    st.markdown("### 📊 Live Order Book")

    # Quote metrics row
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Best Bid", f"${snapshot.best_bid:.2f}", delta=f"{snapshot.best_bid_size:.0f} units")
    m2.metric("Best Ask", f"${snapshot.best_ask:.2f}", delta=f"-{snapshot.best_ask_size:.0f} units", delta_color="inverse")
    m3.metric("Mid Price", f"${snapshot.mid_price:.2f}")
    spread_bps = (snapshot.spread / snapshot.mid_price * 10000) if snapshot.mid_price > 0 else 0
    m4.metric("Spread", f"${snapshot.spread:.2f}", delta=f"{spread_bps:.1f} bps", delta_color="off")

    col_bids, col_asks = st.columns(2)

    with col_bids:
        st.markdown("<p style='color:#00F0FF;font-weight:bold;margin-bottom:4px;'>BIDS (Buy Liquidity)</p>", unsafe_allow_html=True)
        if snapshot.bids:
            bid_df = pd.DataFrame(snapshot.bids[:5], columns=["Price ($)", "Size (Units)"])
            st.dataframe(
                bid_df.style.format({"Price ($)": "${:.2f}", "Size (Units)": "{:.1f}"}),
                use_container_width=True,
                hide_index=True,
            )

    with col_asks:
        st.markdown("<p style='color:#FF0055;font-weight:bold;margin-bottom:4px;'>ASKS (Sell Liquidity)</p>", unsafe_allow_html=True)
        if snapshot.asks:
            ask_df = pd.DataFrame(snapshot.asks[:5], columns=["Price ($)", "Size (Units)"])
            st.dataframe(
                ask_df.style.format({"Price ($)": "${:.2f}", "Size (Units)": "{:.1f}"}),
                use_container_width=True,
                hide_index=True,
            )
