"""
Execution Decision and Active Strategy Banner Component.
Displays active strategy state, urgency meters, order slicing,
and auditable real-time decision explanation text.
"""

import streamlit as st
import pandas as pd
from data.contracts import ExecutionDecision, ExecutionResult


def render_execution_decision_component(decision: ExecutionDecision):
    """
    Renders the live strategy selection banner and context explanation.
    """
    st.markdown("### ⚡ Dynamic Execution Engine Decision")

    # Strategy Badge styling
    strat_name = decision.strategy
    badge_color = "#00FFA3" if "Passive" in strat_name else ("#FF0055" if "Aggressive" in strat_name else "#3399FF")

    st.markdown(
        f"""
        <div style="background-color: #141923; border: 1px solid #2D3748; border-left: 5px solid {badge_color};
                    border-radius: 8px; padding: 14px; margin-bottom: 12px;">
            <div style="font-size: 12px; color: #8E99AB; text-transform: uppercase; letter-spacing: 1px;">Current Strategy Mode</div>
            <div style="font-size: 24px; font-weight: bold; color: {badge_color}; margin: 4px 0;">{strat_name}</div>
            <div style="font-size: 14px; color: #E2E8F0; margin-top: 6px; background-color: #0B0E14; padding: 10px; border-radius: 6px;">
                💡 <b>Reason:</b> {decision.reason}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Target Side", decision.side.value)
    c2.metric("Slice Quantity", f"{decision.quantity:.1f} units")
    c3.metric("Remaining Qty", f"{decision.remaining_quantity:.1f} units")
    c4.metric("Limit / Crossing Price", f"${decision.limit_price:.2f}" if decision.limit_price else "MKT")

    # Urgency Meter
    st.caption(f"Execution Urgency: {decision.urgency * 100:.0f}%")
    st.progress(min(1.0, max(0.0, decision.urgency)))


def render_execution_timeline_component(result: ExecutionResult):
    """
    Renders execution orders and fills history tables.
    """
    st.markdown("### 📜 Execution Ledger & Fills")

    if not result.fills:
        st.info("No fills executed yet.")
        return

    fill_records = []
    for f in result.fills:
        fill_records.append({
            "Fill ID": f.fill_id,
            "Timestamp": f"{f.timestamp:.1f}s",
            "Price": f"${f.price:.2f}",
            "Quantity": f.quantity,
            "Liquidity": f.liquidity_type.value,
            "Slippage": f"${f.slippage:.4f}",
        })

    df_fills = pd.DataFrame(fill_records)
    st.dataframe(df_fills, use_container_width=True, hide_index=True)
