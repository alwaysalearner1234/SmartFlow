import sys
from pathlib import Path

# Add project root to sys.path for robust resolution regardless of launch directory
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st
import numpy as np
import pandas as pd

from config.config import SCENARIOS, DEFAULT_CONFIG, SAVED_MODELS_DIR
from data.generator import MarketDataGenerator
from data.contracts import OrderSide, PredictionResult, ModelStatus
from features import build_feature_pipeline, extract_snapshot_features_dict
from models.predictor import AdverseSelectionPredictor
from models.train import train_models
from simulator.execution_simulator import ExecutionSimulator
from backtest.backtester import Backtester

from dashboard.charts.plots import (
    plot_order_book_depth,
    plot_features_timeseries,
    plot_risk_gauge,
    plot_execution_trajectories,
    plot_comparison_metrics,
)
from dashboard.components.order_book import render_order_book_component
from dashboard.components.execution import (
    render_execution_decision_component,
    render_execution_timeline_component,
)
from dashboard.components.risk import render_risk_monitor_component
from dashboard.components.performance import render_performance_comparison_component

# Page configuration
st.set_page_config(
    page_title="SmartFlow | Smart Order Routing Terminal",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Refinitiv/Bloomberg-grade Terminal Custom CSS
st.markdown(
    """
    <style>
    /* Dark Theme Core */
    .stApp {
        background-color: #0B0E14;
        color: #E2E8F0;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    }
    /* Headers */
    h1, h2, h3, h4 {
        color: #FFFFFF !important;
        font-weight: 700;
        letter-spacing: -0.5px;
    }
    /* Metric Cards */
    [data-testid="stMetricValue"] {
        font-size: 24px !important;
        font-weight: 800;
        color: #00F0FF;
        font-family: "SF Mono", "Fira Code", monospace;
    }
    [data-testid="stMetricDelta"] {
        font-size: 12px !important;
    }
    /* Sidebar */
    [data-testid="stSidebar"] {
        background-color: #10141D;
        border-right: 1px solid #1E2638;
    }
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 12px;
        background-color: #10141D;
        padding: 6px;
        border-radius: 8px;
        border: 1px solid #1E2638;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 6px;
        color: #8E99AB;
        font-weight: 600;
        padding: 8px 16px;
    }
    .stTabs [aria-selected="true"] {
        background-color: #1E2638 !important;
        color: #00FFA3 !important;
    }
    /* Buttons */
    .stButton>button {
        background: linear-gradient(135deg, #1E2638 0%, #2A364F 100%);
        color: #FFFFFF;
        border: 1px solid #3B4B6E;
        border-radius: 6px;
        font-weight: 600;
        transition: all 0.2s ease;
    }
    .stButton>button:hover {
        border-color: #00FFA3;
        color: #00FFA3;
        box-shadow: 0 0 12px rgba(0, 255, 163, 0.25);
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def get_components():
    """Initializes shared predictor, generator, simulator, and backtester."""
    predictor = AdverseSelectionPredictor()
    generator = MarketDataGenerator()
    simulator = ExecutionSimulator()
    backtester = Backtester(execution_simulator=simulator)
    return predictor, generator, simulator, backtester


predictor, generator, simulator, backtester = get_components()

# ================= SIDEBAR CONFIGURATION =================
with st.sidebar:
    st.markdown(
        """
        <div style="padding-bottom: 15px; border-bottom: 1px solid #1E2638; margin-bottom: 15px;">
            <h2 style="margin: 0; color: #00FFA3 !important;">⚡ SmartFlow SOR</h2>
            <div style="font-size: 11px; color: #8E99AB; letter-spacing: 1px;">QUANTITATIVE EXECUTION TERMINAL</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("### 🛠 Execution Setup")
    scenario_keys = list(SCENARIOS.keys())
    selected_scenario_key = st.selectbox(
        "Market Scenario / Regime",
        options=scenario_keys,
        format_func=lambda k: SCENARIOS[k]["name"],
        index=0,
    )
    current_scenario = SCENARIOS[selected_scenario_key]
    st.caption(f"_{current_scenario['description']}_")

    side_choice = st.radio("Order Side", options=["BUY", "SELL"], horizontal=True)
    order_side = OrderSide.BUY if side_choice == "BUY" else OrderSide.SELL

    order_size = st.number_input(
        "Target Order Size",
        min_value=50.0,
        max_value=100000.0,
        value=float(current_scenario["order_size"]),
        step=100.0,
    )

    horizon_sec = st.number_input(
        "Execution Horizon (sec)",
        min_value=10.0,
        max_value=600.0,
        value=float(current_scenario["horizon_sec"]),
        step=10.0,
    )

    st.markdown("---")
    st.markdown("### 🚀 Operations")
    run_sim_btn = st.button("▶ Run Live Simulation", use_container_width=True)
    run_backtest_btn = st.button("⚡ Run Comparative Backtest", use_container_width=True)
    train_model_btn = st.button("🧠 Train ML Adverse Model", use_container_width=True)


# Handle ML model training
if train_model_btn:
    with st.spinner("Generating historical data & training Logistic Regression + XGBoost..."):
        # Generate rich training dataset across normal, volatile, and toxic regimes
        train_snaps = []
        train_snaps.extend(generator.generate_scenario_stream("normal_market", num_ticks=400, dt=0.5))
        train_snaps.extend(generator.generate_scenario_stream("high_volatility", num_ticks=400, dt=0.5))
        train_snaps.extend(generator.generate_scenario_stream("high_adverse_selection", num_ticks=400, dt=0.5))

        df_train_feats = build_feature_pipeline(train_snaps)
        results, path = train_models(df_train_feats, side=order_side, save_best=True)
        predictor.reload()
        st.session_state["training_results"] = results
        st.success(f"Trained & Saved **{results['best_model_name']}**! (ROC-AUC: {results['best_metrics']['roc_auc']:.4f})")


# App Tabs
tab1, tab2, tab3 = st.tabs([
    "📈 Live Execution Terminal",
    "🏆 Multi-Strategy Backtest",
    "🧠 Model Diagnostics & ML Metrics",
])


# Generate or load market data for current session
if "current_snapshots" not in st.session_state or run_sim_btn or run_backtest_btn:
    st.session_state["current_snapshots"] = generator.generate_scenario_stream(
        scenario_key=selected_scenario_key,
        num_ticks=int(max(100, (horizon_sec / 0.5) + 30)),
        dt=0.5,
    )
    st.session_state["features_df"] = build_feature_pipeline(st.session_state["current_snapshots"])

snapshots = st.session_state["current_snapshots"]
df_features = st.session_state["features_df"]


# ================= TAB 1: LIVE EXECUTION TERMINAL =================
with tab1:
    st.markdown(
        f"""
        <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:12px;">
            <div>
                <span style="font-size:20px; font-weight:700;">Asset: BTC-USD</span> &nbsp;|&nbsp;
                <span style="color:#8E99AB;">Regime: <b>{current_scenario['name']}</b></span> &nbsp;|&nbsp;
                <span style="color:#00F0FF;">Vol: {current_scenario['volatility']*100:.0f}%</span>
            </div>
            <div style="color:#8E99AB; font-size:13px;">Replay Ticks: {len(snapshots)}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Top Section: Live Order Book and Microstructure Features
    ob_col, feat_col = st.columns([1.1, 1.1])
    latest_snap = snapshots[-1]

    with ob_col:
        render_order_book_component(latest_snap)
        ob_fig = plot_order_book_depth(latest_snap)
        st.plotly_chart(ob_fig, use_container_width=True)

    with feat_col:
        st.markdown("### 🔬 Microstructure Feature Streams")
        feat_fig = plot_features_timeseries(df_features)
        st.plotly_chart(feat_fig, use_container_width=True)

    st.markdown("---")

    # Run Proposed Execution on stream to demonstrate adaptive SOR
    with st.spinner("Evaluating Dynamic Strategy Engine on live market stream..."):
        exec_result = simulator.run_execution(
            strategy_name="Proposed (ML + AC)",
            snapshots=snapshots,
            total_quantity=order_size,
            horizon_sec=horizon_sec,
            side=order_side,
        )

    # Mid Section: ML Risk & Current Strategy Decision
    risk_col, decision_col = st.columns([1.0, 1.2])

    latest_features = df_features.iloc[-1].to_dict()
    latest_pred = predictor.predict(latest_features, side=order_side, timestamp=latest_snap.timestamp)

    with risk_col:
        render_risk_monitor_component(latest_pred)

    with decision_col:
        # Get latest decision from execution trajectory
        if exec_result.trajectory:
            last_traj = exec_result.trajectory[-1]
            active_decision = simulator.strategy_engine.proposed.compute_decision(
                snapshot=latest_snap,
                remaining_quantity=last_traj["remaining_quantity"],
                elapsed_time=last_traj["elapsed_time"],
                total_horizon=horizon_sec,
                side=order_side,
                initial_quantity=order_size,
                features=latest_features,
            )
        else:
            active_decision = simulator.strategy_engine.proposed.compute_decision(
                snapshot=latest_snap,
                remaining_quantity=order_size,
                elapsed_time=0.0,
                total_horizon=horizon_sec,
                side=order_side,
                initial_quantity=order_size,
                features=latest_features,
            )

        render_execution_decision_component(active_decision)

    # Bottom Section: Fills Ledger
    render_execution_timeline_component(exec_result)


# ================= TAB 2: MULTI-STRATEGY BACKTEST =================
with tab2:
    st.markdown("## 📊 Head-to-Head Strategy Benchmark")
    st.write(
        "All strategies execute across the **exact same market dataset and tick sequence**, "
        "providing a strictly fair, unbiased comparison under identical microstructure conditions."
    )

    with st.spinner("Running synchronized multi-strategy backtest..."):
        backtest_res = backtester.run_backtest(
            snapshots=snapshots,
            scenario_name=current_scenario["name"],
            scenario_description=current_scenario["description"],
            total_quantity=order_size,
            horizon_sec=horizon_sec,
            side=order_side,
        )

    render_performance_comparison_component(backtest_res)


# ================= TAB 3: MODEL DIAGNOSTICS =================
with tab3:
    st.markdown("## 🔬 Adverse-Selection ML Diagnostics")

    # If training was performed in session, show actual results
    if "training_results" in st.session_state:
        res = st.session_state["training_results"]
        st.success(f"Best Validated Model: **{res['best_model_name']}**")

        d_col1, d_col2 = st.columns(2)
        with d_col1:
            st.markdown("### 🎯 Logistic Regression Baseline")
            lr = res["logistic_regression"]
            st.metric("ROC-AUC", f"{lr['roc_auc']:.4f}")
            st.metric("Accuracy", f"{lr['accuracy']*100:.2f}%")
            st.metric("F1 Score", f"{lr['f1_score']:.4f}")
            st.write(f"Precision: {lr['precision']:.4f} | Recall: {lr['recall']:.4f}")
            st.write("**Confusion Matrix:**", lr["confusion_matrix"])

        with d_col2:
            st.markdown("### ⚡ XGBoost Classifier")
            xgb_res = res["xgboost"]
            st.metric("ROC-AUC", f"{xgb_res['roc_auc']:.4f}")
            st.metric("Accuracy", f"{xgb_res['accuracy']*100:.2f}%")
            st.metric("F1 Score", f"{xgb_res['f1_score']:.4f}")
            st.write(f"Precision: {xgb_res['precision']:.4f} | Recall: {xgb_res['recall']:.4f}")
            st.write("**Confusion Matrix:**", xgb_res["confusion_matrix"])

        st.markdown("### 📈 Feature Importances")
        df_imp = pd.DataFrame(
            list(res["feature_importances"].items()),
            columns=["Microstructure Feature", "Importance Weight"],
        ).sort_values("Importance Weight", ascending=False)
        st.dataframe(df_imp, use_container_width=True, hide_index=True)

    else:
        st.info(
            "💡 Click **'Train ML Adverse Model'** in the sidebar to run the chronological training "
            "and cross-evaluation pipeline on multi-regime market microstructure data."
        )
        st.markdown(
            """
            ### Adverse Selection Labeling Logic:
            - **Horizon ($H$ ticks):** Configurable lookahead window (default: 10 ticks).
            - **Target Generation:**
              - For **BUY** orders: Adverse event ($Y=1$) if $P_{t+H} \\le P_t - \\theta \\cdot \\text{Spread}_t$
              - For **SELL** orders: Adverse event ($Y=1$) if $P_{t+H} \\ge P_t + \\theta \\cdot \\text{Spread}_t$
            - **Leakage Prevention:** Chronological non-shuffled train/validation/test partitions.
            - **Preprocessors:** Scaler fitted solely on training splits.
            """
        )
