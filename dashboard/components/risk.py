"""
Machine Learning Risk Monitor Component.
Renders adverse-selection probabilities, model verification status,
and key feature drivers.
"""

from typing import Optional
import streamlit as st
import pandas as pd
from data.contracts import PredictionResult, ModelStatus, RiskCategory, ModelSystemStatus
from dashboard.charts.plots import plot_risk_gauge


def render_risk_monitor_component(
    pred: PredictionResult,
    status_info: Optional[ModelSystemStatus] = None,
):
    """
    Renders the ML adverse selection risk monitor and model status.
    Uses unified ModelSystemStatus as the single source of truth.
    """
    st.markdown("### 🤖 ML Adverse-Selection Risk Engine")

    # Unified Status alert banner
    if status_info is not None:
        if status_info.status == ModelStatus.TRAINED:
            st.markdown(
                f"""
                <div style="background-color: rgba(0, 255, 163, 0.1); border: 1px solid #00FFA3;
                            border-radius: 6px; padding: 10px 14px; margin-bottom: 14px;">
                    <span style="font-size: 15px; font-weight: 700; color: #00FFA3;">🟢 Trained Model Loaded</span> &nbsp;|&nbsp;
                    <span style="color: #FFFFFF; font-weight: 600;">{status_info.model_name}</span> &nbsp;
                    <span style="color: #8E99AB; font-size: 12px;">({status_info.model_version})</span>
                    <div style="font-size: 12px; color: #CBD5E1; margin-top: 4px;">{status_info.details}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        elif status_info.status in [ModelStatus.FALLBACK, ModelStatus.FALLBACK_HEURISTIC]:
            st.markdown(
                f"""
                <div style="background-color: rgba(243, 186, 47, 0.1); border: 1px solid #F3BA2F;
                            border-radius: 6px; padding: 10px 14px; margin-bottom: 14px;">
                    <span style="font-size: 15px; font-weight: 700; color: #F3BA2F;">🟡 Development Fallback</span> &nbsp;|&nbsp;
                    <span style="color: #FFFFFF; font-weight: 600;">{status_info.model_name}</span>
                    <div style="font-size: 12px; color: #CBD5E1; margin-top: 4px;">{status_info.details}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                """
                <div style="background-color: rgba(255, 0, 85, 0.1); border: 1px solid #FF0055;
                            border-radius: 6px; padding: 10px 14px; margin-bottom: 14px;">
                    <span style="font-size: 15px; font-weight: 700; color: #FF0055;">🔴 Model Unavailable</span>
                    <div style="font-size: 12px; color: #CBD5E1; margin-top: 4px;">Inference unavailable.</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
    else:
        if pred.model_status == ModelStatus.TRAINED:
            st.success(f"🟢 **Trained Model Loaded:** {pred.model_name} (Horizon: {pred.prediction_horizon} ticks)")
        else:
            st.warning(f"🟡 **Development Fallback:** {pred.model_name} (Calibrated Microstructure Heuristic)")

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
