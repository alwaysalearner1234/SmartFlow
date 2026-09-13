"""
Machine Learning Risk Monitor Component.
Renders adverse-selection probabilities, model verification status,
and key feature drivers.
"""

import streamlit as st
import pandas as pd
from data.contracts import PredictionResult, ModelStatus, RiskCategory
from dashboard.charts.plots import plot_risk_gauge


def render_risk_monitor_component(pred: PredictionResult):
    """
    Renders the ML adverse selection risk monitor and model status.
    """
    st.markdown("### 🤖 ML Adverse-Selection Risk Engine")

    # Status alert banner
    if pred.model_status == ModelStatus.TRAINED:
        st.success(f"✅ **Production ML Model Active:** {pred.model_name} (Horizon: {pred.prediction_horizon} ticks)")
    else:
        st.warning(
            f"⚠️ **Development Fallback Mode:** {pred.model_name} — "
            "Model has not yet been trained on historical data. Using calibrated microstructure statistical heuristic."
        )

    r_col1, r_col2 = st.columns([1.2, 1.0])

    with r_col1:
        gauge_fig = plot_risk_gauge(pred.probability)
        st.plotly_chart(gauge_fig, use_container_width=True)

    with r_col2:
        st.markdown("<p style='font-size:12px;color:#8E99AB;margin-top:16px;'>RISK CLASSIFICATION</p>", unsafe_allow_html=True)
        cat_color = "#FF0055" if pred.risk_category == RiskCategory.HIGH else ("#00FFA3" if pred.risk_category == RiskCategory.LOW else "#F3BA2F")
        st.markdown(f"<h2 style='color:{cat_color};margin-top:0;'>{pred.risk_category.value} RISK</h2>", unsafe_allow_html=True)
        st.write(f"Adverse Probability: **{pred.probability * 100:.1f}%**")
        st.caption(f"Evaluates likelihood that trade execution suffers adverse price selection over next {pred.prediction_horizon} ticks.")

    # Top Feature Importances / Microstructure Signals
    if pred.feature_importances:
        st.markdown("##### Key Microstructure Risk Drivers")
        imp_items = sorted(pred.feature_importances.items(), key=lambda x: abs(x[1]), reverse=True)[:5]
        df_imp = pd.DataFrame(imp_items, columns=["Feature", "Signal Importance"])
        st.dataframe(df_imp, use_container_width=True, hide_index=True)
