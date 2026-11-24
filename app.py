"""Road-Accident Severity & Intervention Prioritizer — Streamlit application.

Launch with:  streamlit run app.py
"""
from __future__ import annotations

from functools import partial

import streamlit as st

from dashboard import theme
from dashboard.data_state import get_state
from dashboard.pages import explorer, explainability, geography, overview, performance, prioritization, prediction

st.set_page_config(
    page_title="Road-Accident Severity & Intervention Prioritizer",
    page_icon=":rotating_light:",
    layout="wide",
    initial_sidebar_state="expanded",
)
theme.apply_theme()

state = get_state()

with st.sidebar:
    st.markdown("## :rotating_light: RAIP")
    st.caption("Road-Accident Severity & Intervention Prioritizer")
    st.markdown("**PREDICT → EXPLAIN → PRIORITISE → VISUALISE**")
    st.divider()
    st.markdown(
        f"<div class='banner' style='margin-bottom:10px'>"
        f"<strong>Data source:</strong> {state['label']}<br>"
        f"<strong>Years:</strong> {state['provenance'].get('years', 'n/a')}<br>"
        f"<strong>Scope:</strong> {state['provenance'].get('geography', 'Great Britain')}<br>"
        f"<strong>Model:</strong> {state['model_meta'].get('best_model', 'none')}</div>",
        unsafe_allow_html=True,
    )
    if state["demo"]:
        st.error("SYNTHETIC DEMO DATA — figures are not real-world results.")
    if state["model"] is None:
        st.warning("No trained model found. Run `python scripts/run_pipeline.py`.")

    uploaded = st.file_uploader("Load your own STATS19-style CSV", type=["csv"],
                                help="Optional. Provide a collision CSV with STATS19 column names.")
    if uploaded is not None:
        if st.session_state.get("uploaded_name") != uploaded.name:
            import pandas as pd
            try:
                st.session_state["uploaded_frame"] = pd.read_csv(uploaded, low_memory=False, encoding="utf-8-sig")
                st.session_state["uploaded_name"] = uploaded.name
                st.cache_data.clear()
                st.cache_resource.clear()
                st.rerun()
            except Exception as exc:
                st.error(f"Could not read upload: {exc}")
    if st.session_state.get("uploaded_frame") is not None:
        if st.button("Remove uploaded CSV"):
            st.session_state.pop("uploaded_frame", None)
            st.session_state.pop("uploaded_name", None)
            st.cache_data.clear()
            st.cache_resource.clear()
            st.rerun()

    st.divider()
    st.caption("Official data: GOV.UK / DfT STATS19 open data. Great Britain only. "
               "No causal claims; all prioritisation signals are associative.")

pages = [
    st.Page(partial(overview.render, state=state), title="Executive Overview", url_path="overview",
            icon=":material/dashboard:", default=True),
    st.Page(partial(explorer.render, state=state), title="Accident Explorer", url_path="explorer", icon=":material/search:"),
    st.Page(partial(prediction.render, state=state), title="Severity Prediction", url_path="prediction",
            icon=":material/model_training:"),
    st.Page(partial(explainability.render, state=state), title="Explainable AI", url_path="explainability",
            icon=":material/psychology:"),
    st.Page(partial(prioritization.render, state=state), title="Intervention Priority", url_path="prioritization",
            icon=":material/priority_high:"),
    st.Page(partial(performance.render, state=state), title="Model Performance", url_path="performance",
            icon=":material/analytics:"),
    st.Page(partial(geography.render, state=state), title="Geographic Intelligence", url_path="geography",
            icon=":material/map:"),
]
nav = st.navigation(pages)
nav.run()
