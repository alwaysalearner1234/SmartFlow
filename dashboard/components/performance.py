"""
Performance and Benchmark Comparison Component.
Displays implementation shortfall, realized execution costs, slippage,
and side-by-side strategy benchmark tables.
"""

import streamlit as st
import pandas as pd
from data.contracts import BacktestResult
from dashboard.charts.plots import plot_comparison_metrics, plot_execution_trajectories


def render_performance_comparison_component(backtest_res: BacktestResult):
    """
    Renders the benchmark performance comparisons and charts.
    """
    st.markdown("### 🏆 Benchmark Strategy Comparison")

    st.markdown(
        f"""
        <div style="background-color: #141923; padding: 10px; border-radius: 6px; margin-bottom: 12px; border: 1px solid #2D3748;">
            <b>Scenario:</b> {backtest_res.scenario_name} &nbsp;|&nbsp;
            <b>Order Size:</b> {backtest_res.target_quantity:.0f} units &nbsp;|&nbsp;
            <b>Horizon:</b> {backtest_res.execution_horizon_sec:.0f}s &nbsp;|&nbsp;
            <b>Arrival Price:</b> ${backtest_res.initial_arrival_price:.2f} &nbsp;|&nbsp;
            <span style="color:#00FFA3;">✔ Identical Market Conditions Enforced</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Highlight metrics: Proposed vs TWAP savings
    strat_results = backtest_res.strategy_results
    prop = strat_results.get("Proposed (ML + AC)")
    twap = strat_results.get("TWAP")

    if prop and twap:
        savings = twap.total_cost - prop.total_cost
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Proposed Shortfall", f"{prop.implementation_shortfall_bps:.1f} bps", f"${prop.total_cost:.2f}")
        k2.metric("TWAP Shortfall", f"{twap.implementation_shortfall_bps:.1f} bps", f"${twap.total_cost:.2f}")
        k3.metric("Alpha Cost Savings", f"${savings:.2f}", delta=f"{twap.implementation_shortfall_bps - prop.implementation_shortfall_bps:.1f} bps improvement")
        k4.metric("Proposed Fill Rate", f"{prop.fill_rate * 100:.1f}%")

    # Interactive plots
    c_fig = plot_comparison_metrics(backtest_res.comparison_table)
    st.plotly_chart(c_fig, use_container_width=True)

    t_fig = plot_execution_trajectories(strat_results)
    st.plotly_chart(t_fig, use_container_width=True)

    # Full comparative table
    st.markdown("##### Detailed Metric Breakdown")
    st.dataframe(
        backtest_res.comparison_table.style.format({
            "Executed Qty": "{:.1f}",
            "Arrival Price": "${:.2f}",
            "Avg Exec Price": "${:.2f}",
            "Shortfall ($)": "${:.2f}",
            "Shortfall (bps)": "{:.1f}",
            "Slippage ($)": "${:.4f}",
            "Market Impact ($)": "${:.2f}",
            "Adverse Selection ($)": "${:.2f}",
            "Realized Cost ($)": "${:.2f}",
            "Expected AC Cost ($)": "${:.2f}",
            "Cost Gap (Realized - AC)": "${:.2f}",
            "Completion (%)": "{:.1f}%",
        }),
        use_container_width=True,
        hide_index=True,
    )
