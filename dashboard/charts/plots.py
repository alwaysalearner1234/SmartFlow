"""
Interactive Quantitative Trading Terminal Visualizations using Plotly.
Includes:
- Multi-level cumulative Order Book Depth chart
- Microstructure Features multi-panel time-series
- ML Adverse Selection Risk Gauge & Probability curve
- Strategy Execution Trajectory comparison
- Implementation Shortfall and Cost breakdown bar charts
"""

from typing import List, Dict, Any, Optional
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from data.contracts import MarketSnapshot, ExecutionResult, BacktestResult, VenueRoutingResult

DARK_TEMPLATE = "plotly_dark"
COLOR_BID = "#00F0FF"      # Cyan / Neon Blue for bids
COLOR_ASK = "#FF0055"      # Neon Pink / Red for asks
COLOR_MID = "#F3BA2F"      # Gold
COLOR_ACCENT = "#00FFA3"   # Neon Green
COLOR_BG = "#0B0E14"
COLOR_CARD = "#141923"


def plot_order_book_depth(snapshot: MarketSnapshot) -> go.Figure:
    """
    Plots cumulative order book depth chart (bid green/cyan, ask red/pink).
    """
    fig = go.Figure()

    if snapshot.bids and snapshot.asks:
        # Bids: sorted descending, cumulative size
        bids_sorted = sorted(snapshot.bids, key=lambda x: x[0], reverse=True)
        bid_prices = [p for p, _ in bids_sorted]
        bid_cum_sizes = np.cumsum([s for _, s in bids_sorted])

        # Asks: sorted ascending, cumulative size
        asks_sorted = sorted(snapshot.asks, key=lambda x: x[0])
        ask_prices = [p for p, _ in asks_sorted]
        ask_cum_sizes = np.cumsum([s for _, s in asks_sorted])

        # Add bid area
        fig.add_trace(go.Scatter(
            x=bid_prices,
            y=bid_cum_sizes,
            name="Cumulative Bid Depth",
            fill="tozeroy",
            mode="lines+markers",
            line=dict(color=COLOR_BID, width=2.5),
            fillcolor="rgba(0, 240, 255, 0.18)",
        ))

        # Add ask area
        fig.add_trace(go.Scatter(
            x=ask_prices,
            y=ask_cum_sizes,
            name="Cumulative Ask Depth",
            fill="tozeroy",
            mode="lines+markers",
            line=dict(color=COLOR_ASK, width=2.5),
            fillcolor="rgba(255, 0, 85, 0.18)",
        ))

        # Mid price line
        fig.add_vline(
            x=snapshot.mid_price,
            line_dash="dash",
            line_color=COLOR_MID,
            annotation_text=f"Mid ${snapshot.mid_price:.2f}",
            annotation_position="top",
        )

    fig.update_layout(
        title="<b>LIVE ORDER BOOK DEPTH LADDER</b>",
        template=DARK_TEMPLATE,
        paper_bgcolor=COLOR_BG,
        plot_bgcolor=COLOR_CARD,
        xaxis_title="Price ($)",
        yaxis_title="Cumulative Depth (Units)",
        hovermode="x unified",
        margin=dict(l=40, r=40, t=50, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        height=320,
    )
    return fig


def plot_features_timeseries(df_features: pd.DataFrame) -> go.Figure:
    """
    Multi-panel time-series for spread (bps), order flow imbalance, and realized volatility.
    """
    fig = make_subplots(
        rows=3,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.08,
        subplot_titles=(
            "<b>Bid-Ask Spread (basis points)</b>",
            "<b>Order Flow Imbalance (OFI)</b>",
            "<b>Rolling Realized Volatility</b>",
        ),
    )

    t = df_features["timestamp"] - df_features["timestamp"].iloc[0]

    # Panel 1: Spread
    if "spread_bps" in df_features.columns:
        fig.add_trace(
            go.Scatter(x=t, y=df_features["spread_bps"], name="Spread (bps)", line=dict(color=COLOR_MID, width=1.8)),
            row=1, col=1,
        )

    # Panel 2: OFI
    if "depth_imbalance_l1" in df_features.columns:
        fig.add_trace(
            go.Scatter(x=t, y=df_features["depth_imbalance_l1"], name="L1 Imbalance", line=dict(color=COLOR_BID, width=1.5)),
            row=2, col=1,
        )
    if "ofi_instant" in df_features.columns:
        fig.add_trace(
            go.Bar(x=t, y=df_features["ofi_instant"], name="Instant OFI", marker_color="rgba(0, 255, 163, 0.4)"),
            row=2, col=1,
        )

    # Panel 3: Volatility
    vol_col = "volatility_std_10" if "volatility_std_10" in df_features.columns else None
    if vol_col:
        fig.add_trace(
            go.Scatter(x=t, y=df_features[vol_col], name="Vol (10-tick)", line=dict(color="#B388FF", width=1.8)),
            row=3, col=1,
        )

    fig.update_layout(
        template=DARK_TEMPLATE,
        paper_bgcolor=COLOR_BG,
        plot_bgcolor=COLOR_CARD,
        height=450,
        margin=dict(l=40, r=40, t=50, b=30),
        showlegend=False,
    )
    return fig


def plot_risk_gauge(probability: float) -> go.Figure:
    """
    Plots an institutional adverse selection probability gauge.
    """
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=round(probability * 100, 1),
        number={"suffix": "%", "font": {"size": 28, "color": "#FFFFFF"}},
        title={"text": "<b>ADVERSE SELECTION RISK</b><br><span style='font-size:12px;color:#8E99AB'>Probability of Toxic Fill / Plunge</span>", "font": {"size": 14}},
        gauge={
            "axis": {"range": [0, 100], "tickwidth": 1, "tickcolor": "#5A6578"},
            "bar": {"color": "#FF0055" if probability > 0.65 else ("#00FFA3" if probability < 0.35 else "#F3BA2F"), "thickness": 0.28},
            "bgcolor": "#1A202C",
            "borderwidth": 1,
            "bordercolor": "#2D3748",
            "steps": [
                {"range": [0, 35], "color": "rgba(0, 255, 163, 0.15)"},
                {"range": [35, 65], "color": "rgba(243, 186, 47, 0.15)"},
                {"range": [65, 100], "color": "rgba(255, 0, 85, 0.22)"},
            ],
            "threshold": {
                "line": {"color": "#FF0055", "width": 3},
                "thickness": 0.8,
                "value": 65,
            },
        },
    ))

    fig.update_layout(
        template=DARK_TEMPLATE,
        paper_bgcolor=COLOR_BG,
        plot_bgcolor=COLOR_CARD,
        height=220,
        margin=dict(l=25, r=25, t=50, b=20),
    )
    return fig


def plot_execution_trajectories(strategy_results: Dict[str, ExecutionResult]) -> go.Figure:
    """
    Plots the remaining inventory trajectory curves for all competing strategies.
    """
    fig = go.Figure()
    colors = {
        "Market": "#FF3366",
        "TWAP": "#3399FF",
        "VWAP": "#9966FF",
        "Almgren-Chriss": "#FFCC00",
        "Proposed (ML + AC)": "#00FFA3",
    }

    for name, res in strategy_results.items():
        if res.trajectory:
            times = [pt["elapsed_time"] for pt in res.trajectory]
            rem_shares = [pt["remaining_quantity"] for pt in res.trajectory]
            color = colors.get(name, "#CCCCCC")
            fig.add_trace(go.Scatter(
                x=times,
                y=rem_shares,
                name=name,
                mode="lines+markers",
                line=dict(color=color, width=3.0 if "Proposed" in name else 1.8),
                marker=dict(size=4),
            ))

    fig.update_layout(
        title="<b>EXECUTION TRAJECTORY (Remaining Shares vs Time)</b>",
        template=DARK_TEMPLATE,
        paper_bgcolor=COLOR_BG,
        plot_bgcolor=COLOR_CARD,
        xaxis_title="Elapsed Time (seconds)",
        yaxis_title="Remaining Quantity",
        margin=dict(l=40, r=40, t=50, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        height=320,
    )
    return fig


def plot_comparison_metrics(df_comparison: pd.DataFrame) -> go.Figure:
    """
    Plots side-by-side bar charts comparing Implementation Shortfall (bps) and Realized Costs.
    """
    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=("<b>Implementation Shortfall (basis points)</b>", "<b>Realized Execution Cost ($)</b>"),
    )

    strategies = df_comparison["Strategy"]
    colors = [
        "#00FFA3" if "Proposed" in s else ("#FF3366" if "Market" in s else "#3399FF")
        for s in strategies
    ]

    # Panel 1: Shortfall bps
    fig.add_trace(
        go.Bar(
            x=strategies,
            y=df_comparison["Shortfall (bps)"],
            marker_color=colors,
            text=[f"{v:.1f} bps" for v in df_comparison["Shortfall (bps)"]],
            textposition="auto",
            name="Shortfall (bps)",
        ),
        row=1, col=1,
    )

    # Panel 2: Realized cost $
    fig.add_trace(
        go.Bar(
            x=strategies,
            y=df_comparison["Realized Cost ($)"],
            marker_color=colors,
            text=[f"${v:.2f}" for v in df_comparison["Realized Cost ($)"]],
            textposition="auto",
            name="Realized Cost ($)",
        ),
        row=1, col=2,
    )

    fig.update_layout(
        template=DARK_TEMPLATE,
        paper_bgcolor=COLOR_BG,
        plot_bgcolor=COLOR_CARD,
        height=340,
        margin=dict(l=40, r=40, t=50, b=40),
        showlegend=False,
    )
    return fig


def plot_venue_allocations(routing_result: VenueRoutingResult) -> go.Figure:
    """
    Renders a visual allocation bar chart showing order distribution across simulated venues.
    """
    venues = routing_result.venues
    venue_labels = [v.venue_name for v in venues]
    allocations = [v.routed_quantity for v in venues]
    percentages = [v.allocation_pct for v in venues]

    colors = ["#00FFA3", "#00F0FF", "#F3BA2F", "#A78BFA"]
    text_labels = [f"{q:,.0f} units ({pct:.1f}%)" for q, pct in zip(allocations, percentages)]

    fig = go.Figure(
        go.Bar(
            x=allocations,
            y=venue_labels,
            orientation="h",
            marker=dict(
                color=colors[:len(venues)],
                line=dict(color="#1E2638", width=1.5),
            ),
            text=text_labels,
            textposition="inside",
            insidetextanchor="middle",
        )
    )

    fig.update_layout(
        template=DARK_TEMPLATE,
        paper_bgcolor=COLOR_BG,
        plot_bgcolor=COLOR_CARD,
        height=240,
        margin=dict(l=20, r=20, t=30, b=30),
        xaxis=dict(
            title="Routed Quantity (Units)",
            showgrid=True,
            gridcolor="#1E2638",
        ),
        yaxis=dict(
            autorange="reversed",
            showgrid=False,
        ),
        showlegend=False,
    )
    return fig
