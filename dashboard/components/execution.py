from typing import Optional, Dict, Any
import streamlit as st
import pandas as pd
from data.contracts import ExecutionDecision, ExecutionResult, MarketSnapshot


def render_why_this_decision_panel(
    decision: ExecutionDecision,
    features: Optional[Dict[str, Any]] = None,
    snapshot: Optional[MarketSnapshot] = None,
):
    """
    Renders the 'Why This Decision?' explanatory panel explaining the
    active strategy selection in plain English using actual system values.
    """
    st.markdown("### 💡 Why This Decision?")

    strat_lower = decision.strategy.lower()
    risk_pct = decision.risk_score * 100.0
    urgency_pct = decision.urgency * 100.0

    if "passive" in strat_lower and "withdrawn" not in strat_lower:
        mode_header = "PASSIVE selected"
        color = "#00FFA3"
        explanation = (
            f"Adverse-selection risk is low ({risk_pct:.1f}%) and available liquidity is healthy. "
            f"The engine is favoring passive limit execution ({decision.quantity:,.0f} units "
            f"at ${decision.limit_price:.2f} if limit available) to earn/save the spread and avoid immediate market impact."
        )
    elif "withdrawn" in strat_lower or decision.quantity == 0:
        mode_header = "WAIT / WITHDRAWN selected"
        color = "#F3BA2F"
        explanation = (
            f"Adverse-selection risk is elevated ({risk_pct:.1f}%) while liquidity is vulnerable. "
            f"Resting limit quotes have been paused/withdrawn to prevent the winner's curse on remaining "
            f"{decision.remaining_quantity:,.0f} units until market conditions stabilize."
        )
    elif "aggressive" in strat_lower or "urgent" in strat_lower:
        mode_header = "AGGRESSIVE selected"
        color = "#FF0055"
        if urgency_pct >= 70.0:
            explanation = (
                f"Execution urgency is high ({urgency_pct:.0f}%) with {decision.remaining_quantity:,.0f} units remaining. "
                f"The engine is prioritizing order completion by crossing the spread for {decision.quantity:,.0f} units."
            )
        else:
            explanation = (
                f"Adverse-selection risk is elevated ({risk_pct:.1f}%) with adverse order flow pressure. "
                f"The engine is executing aggressively for {decision.quantity:,.0f} units to lock in fills before price deteriorates further."
            )
    else:
        mode_header = "ALMGREN-CHRISS SLICED selected"
        color = "#3399FF"
        explanation = (
            f"Moderate market risk ({risk_pct:.1f}%) and balanced liquidity. "
            f"The engine is executing an optimal Almgren-Chriss trajectory slice of {decision.quantity:,.0f} units "
            f"to balance inventory variance and temporary price impact."
        )

    st.markdown(
        f"""
        <div style="background-color: #141923; border: 1px solid #2D3748; border-left: 4px solid {color};
                    border-radius: 6px; padding: 12px 16px; margin-bottom: 12px;">
            <div style="font-size: 14px; font-weight: 700; color: {color}; margin-bottom: 4px;">
                {mode_header}
            </div>
            <div style="font-size: 13px; color: #E2E8F0; line-height: 1.5;">
                {explanation}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_execution_decision_component(
    decision: ExecutionDecision,
    features: Optional[Dict[str, Any]] = None,
    snapshot: Optional[MarketSnapshot] = None,
):
    """
    Renders the live strategy selection banner, context explanation,
    metrics, and the 'Why This Decision?' panel.
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

    # Why This Decision? Panel
    render_why_this_decision_panel(decision, features, snapshot)


def render_strategy_timeline_component(result: Optional[ExecutionResult]):
    """
    Renders the Strategy Timeline showing when the execution engine
    changes strategy during a simulation run.
    """
    st.markdown("### ⏱ Strategy Timeline")
    st.caption("Chronological record of execution engine strategy transitions and adaptions.")

    if not result or not result.trajectory:
        st.info("No strategy execution history yet — click **Run Live Simulation** to execute.")
        return

    timeline_records = []
    last_strat = None

    for pt in result.trajectory:
        strat = pt.get("decision_strategy", "")
        # Clean strategy label
        clean_strat = "PASSIVE" if "passive" in strat.lower() else (
            "WAIT" if ("withdrawn" in strat.lower() or pt.get("decision_quantity", 0) == 0) else (
                "AGGRESSIVE" if "aggressive" in strat.lower() else "ALMGREN-CHRISS"
            )
        )
        elapsed = pt.get("elapsed_time", 0.0)
        mins = int(elapsed // 60)
        secs = int(elapsed % 60)
        time_str = f"{mins:02d}:{secs:02d}"

        timeline_records.append({
            "Time": time_str,
            "Elapsed (s)": f"{elapsed:.1f}s",
            "Strategy": clean_strat,
            "Mode Details": strat,
            "Slice Qty": f"{pt.get('decision_quantity', 0.0):,.0f}",
            "Remaining Qty": f"{pt.get('remaining_quantity', 0.0):,.0f}",
            "Risk Score": f"{pt.get('risk_score', 0.0) * 100:.1f}%",
        })
        last_strat = clean_strat

    df_timeline = pd.DataFrame(timeline_records)

    # Show compact transitions view
    col_t1, col_t2 = st.columns([1.2, 1.0])
    with col_t1:
        st.markdown("##### Strategy Transition Log")
        st.dataframe(
            df_timeline[["Time", "Strategy", "Slice Qty", "Remaining Qty", "Risk Score"]],
            use_container_width=True,
            hide_index=True,
        )

    with col_t2:
        st.markdown("##### Adaptive Behavior Summary")
        unique_strats = df_timeline["Strategy"].value_counts().to_dict()
        st.write(f"Total Decision Cycles: **{len(df_timeline)}**")
        for st_name, count in unique_strats.items():
            st.write(f"- **{st_name}**: {count} intervals ({count / len(df_timeline) * 100:.0f}%)")
        st.caption("Demonstrates continuous adaptation between passive maker and aggressive taker modes.")


def render_execution_timeline_component(result: Optional[ExecutionResult]):
    """
    Renders execution orders and fills history tables.
    """
    st.markdown("### 📜 Execution Ledger & Fills")

    if not result or not result.fills:
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
