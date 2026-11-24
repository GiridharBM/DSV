"""C. Severity Prediction using the actual trained model."""
from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
import streamlit as st

from dashboard.components.common import banner, section
from dashboard.theme import SEVERITY, plot_layout
from src import config
from src.explainability.shap_explain import shap_values_for
from src.features.engineering import frame_from_feature_values, time_of_day


def _options(features, col: str, limit: int = 40) -> list[str]:
    if col not in features.columns:
        return ["Unknown"]
    values = features[col].astype(str).value_counts().head(limit).index.tolist()
    return values if "Unknown" in values else values + ["Unknown"]


def render(state: dict) -> None:
    section("Severity Prediction", "Estimate collision severity from at-scene conditions using the trained classifier.")
    model = state["model"]
    if model is None:
        st.error("No trained model artifact found. Run `python scripts/run_pipeline.py` to train one. "
                 "The dashboard will not substitute a mock prediction.")
        return
    meta = state["model_meta"]
    if state["demo"]:
        banner("**DEMO INPUTS** — this dataset is synthetic. The model shown below was trained on real DfT data "
               "unless the pipeline was run against the demo frame.")
    else:
        banner(f"**Model:** {meta.get('best_model', 'unknown')} &nbsp;|&nbsp; "
               f"Selected by `{meta.get('selection_rule', 'macro-F1 based')}` &nbsp;|&nbsp; "
               f"Trained on {meta.get('train_rows', 0):,} rows, chronological holdout of {meta.get('test_rows', 0):,} rows. "
               "Features are pre-collision / at-scene context only; no casualty or post-collision fields are used.")

    features = state["features"]
    with st.form("prediction_form"):
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown("#### Event context")
            hour = st.slider("Hour of day", 0, 23, 17)
            month = st.selectbox("Month", list(range(1, 13)), index=5)
            day = st.selectbox("Day of week", _options(features, "day_of_week"), index=0)
            vehicles = st.number_input("Vehicles involved", 1, 12, 2)
        with c2:
            st.markdown("#### Road & environment")
            speed = st.selectbox("Speed limit (mph)", [10, 15, 20, 30, 40, 50, 60, 70], index=3)
            road_type = st.selectbox("Road type", _options(features, "road_type"))
            road_class = st.selectbox("Road class", _options(features, "first_road_class"))
            surface = st.selectbox("Road surface", _options(features, "road_surface_conditions"))
        with c3:
            st.markdown("#### Conditions")
            weather = st.selectbox("Weather", _options(features, "weather_conditions"))
            light = st.selectbox("Lighting", _options(features, "light_conditions"))
            junction = st.selectbox("Junction detail", _options(features, "junction_detail"))
            urban = st.selectbox("Urban / rural", _options(features, "urban_or_rural_area"))

        values = {
            "number_of_vehicles": vehicles, "speed_limit": speed, "hour": float(hour),
            "month": float(month), "is_weekend": float(day in ("Saturday", "Sunday")),
            "is_night": float(hour < 6 or hour >= 20),
            "day_of_week": day, "light_conditions": light, "weather_conditions": weather,
            "road_surface_conditions": surface, "road_type": road_type, "junction_detail": junction,
            "junction_control": "Unknown", "special_conditions_at_site": "None",
            "carriageway_hazards": "None", "urban_or_rural_area": urban,
            "first_road_class": road_class, "time_of_day_band": time_of_day(hour),
        }
        submitted = st.form_submit_button("Predict severity", width="stretch")

    if not submitted:
        st.info("Set the conditions above and press **Predict severity**.")
        return

    frame = frame_from_feature_values(values)
    try:
        proba = model.predict_proba(frame)[0]
    except Exception as exc:
        st.error(f"Prediction failed: {exc}")
        return
    classes = config.SEVERITY_ORDER
    pred_idx = int(np.argmax(proba))
    predicted = classes[pred_idx] if pred_idx < len(classes) else str(model.classes_[pred_idx])

    fig = go.Figure(go.Bar(
        x=proba, y=classes, orientation="h",
        marker=dict(color=[SEVERITY[c] for c in classes]),
        text=[f"{p*100:.2f}%" for p in proba], textposition="outside",
        hovertemplate="%{y}: %{x:.4f}",
    ))
    fig.update_xaxes(range=[0, max(proba) * 1.25 + 0.02], title="Predicted probability")
    st.plotly_chart(plot_layout(fig, 280, f"Predicted class: {predicted}"), width="stretch")
    st.success(f"**{predicted}** is the most likely severity for these conditions "
               f"(probability {proba[pred_idx]*100:.2f}%). This is an association-based estimate, not a statement of certainty.")

    st.markdown("---")
    st.subheader("Why the model predicted this")
    try:
        values_shap, names, kind = shap_values_for(model, frame)
        if values_shap is None:
            st.info(f"SHAP explainer unavailable ({kind}).")
            return
        contributions = values_shap[0, :, pred_idx]
        order = np.argsort(np.abs(contributions))[::-1][:12]
        labels = [str(names[i]).split("__", 1)[-1] for i in order]
        fig = go.Figure(go.Bar(
            x=contributions[order], y=labels, orientation="h",
            marker=dict(color=["#f87171" if v > 0 else "#38bdf8" for v in contributions[order]]),
            hovertemplate="%{y}: %{x:+.3f}",
        ))
        fig.update_xaxes(title="SHAP contribution to predicted class (log-odds)")
        st.plotly_chart(plot_layout(fig, 430, f"Feature contributions ({kind})"), width="stretch")
        st.caption("Positive values push toward the predicted class; negative values push away. "
                   "Contributions are from the actual trained model.")
    except Exception as exc:
        st.warning(f"Explanation could not be generated: {exc}")
