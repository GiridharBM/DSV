"""D. Explainable AI (SHAP)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from dashboard.components.common import banner, section
from dashboard.theme import CYAN, SEVERITY, plot_layout
from src import config
from src.explainability.shap_explain import transformed_matrix


def render(state: dict) -> None:
    section("Explainable AI", "What drives the model, globally and for individual collisions.")
    meta = state["shap_meta"]
    if not meta.get("available"):
        st.warning("SHAP outputs are not available for the current model. Run the analysis pipeline to generate them; "
                   "no substitute values are displayed.")
        return
    banner(f"**Explainer:** {meta.get('explainer', 'SHAP')} &nbsp;|&nbsp; **Model:** {meta.get('model')} "
           f"&nbsp;|&nbsp; **Rows explained:** {meta.get('n_explained', 0):,} from the chronological holdout. "
           "All values below are computed from the trained model, not estimated or invented.")

    global_df = pd.read_csv(config.REPORTS_DIR / "shap_global_importance.csv")
    cat_df = pd.read_csv(config.REPORTS_DIR / "shap_category_importance.csv")
    values = np.load(config.MODELS_DIR / "shap_values.npy")
    sample = pd.read_csv(config.PROCESSED_DIR / "shap_explain_sample.csv")
    names = meta["feature_names"]

    left, right = st.columns([1.1, 1])
    with left:
        top = global_df.head(18).sort_values("mean_abs_shap")
        labels = top["feature"].str.split("__").str[-1]
        fig = go.Figure(go.Bar(x=top["mean_abs_shap"], y=labels, orientation="h",
                               marker=dict(color=top["mean_abs_shap"], colorscale=[[0, "#155e75"], [1, CYAN]]),
                               hovertemplate="%{y}: %{x:.4f}"))
        fig.update_xaxes(title="Mean |SHAP value|")
        st.plotly_chart(plot_layout(fig, 520, "Global feature importance"), width="stretch")
    with right:
        if not cat_df.empty:
            top_cat = cat_df.nlargest(18, "e_raw").sort_values("e_raw")
            labels = top_cat["source_feature"] + " = " + top_cat["category"]
            fig = go.Figure(go.Bar(x=top_cat["e_raw"], y=labels, orientation="h",
                                   marker=dict(color=top_cat["e_raw"], colorscale=[[0, "#0f766e"], [1, "#fbbf24"]]),
                                   hovertemplate="%{y}: %{x:.4f}"))
            fig.update_xaxes(title="Mean |SHAP| for category")
            st.plotly_chart(plot_layout(fig, 520, "Condition-level importance (feeds IPS 'E')"), width="stretch")

    st.markdown("---")
    st.subheader("SHAP beeswarm")
    class_label = st.selectbox("Explain contribution to class", config.SEVERITY_ORDER, index=0)
    class_idx = config.SEVERITY_ORDER.index(class_label)
    top_features = global_df.head(12)["feature"].tolist()
    indices = [names.index(f) for f in top_features if f in names]
    matrix, matrix_names = transformed_matrix(state["model"], sample[[c for c in config.FEATURE_COLUMNS if c in sample.columns]])
    rng = np.random.default_rng(config.RANDOM_SEED)
    n_points = len(sample)
    fig = go.Figure()
    for i, feat_idx in enumerate(indices):
        col = matrix[:, feat_idx]
        span = np.ptp(col) if np.ptp(col) > 0 else 1
        norm = (col - col.min()) / span
        y = i + rng.uniform(-0.32, 0.32, n_points)
        fig.add_trace(go.Scatter(
            x=values[:, feat_idx, class_idx] if values.shape[2] > class_idx else values[:, feat_idx, 0],
            y=y, mode="markers", name=str(matrix_names[feat_idx]).split("__")[-1],
            marker=dict(color=norm, colorscale="Turbo", size=4, opacity=0.65,
                        colorbar=dict(title="Feature value", x=1.02) if i == 0 else None, showscale=i == 0),
            hovertemplate=f"{str(matrix_names[feat_idx]).split('__')[-1]}: %{{x:+.3f}}<extra></extra>",
        ))
    fig.update_yaxes(tickvals=list(range(len(indices))),
                     ticktext=[str(matrix_names[i]).split("__")[-1] for i in indices], autorange="reversed")
    fig.update_xaxes(title="SHAP value (log-odds contribution)")
    st.plotly_chart(plot_layout(fig, 620), width="stretch")
    st.caption("Each dot is one holdout collision. Horizontal position = that feature's contribution to the selected "
               "class; colour = the feature's value (low → high).")

    st.markdown("---")
    st.subheader("Individual prediction explanation")
    idx = st.slider("Explained holdout record", 0, max(0, n_points - 1), 0)
    actual = None
    if "actual_class_index" in sample.columns:
        actual = config.SEVERITY_ORDER[int(sample.iloc[idx]["actual_class_index"])]
    predicted_proba = state["model"].predict_proba(sample[[c for c in config.FEATURE_COLUMNS if c in sample.columns]].iloc[[idx]])[0]
    pred_label = config.SEVERITY_ORDER[int(np.argmax(predicted_proba))]
    st.write(f"**Actual:** {actual or 'not stored'} &nbsp;|&nbsp; **Predicted:** {pred_label} &nbsp;|&nbsp; "
             + " &nbsp;|&nbsp; ".join(f"{c}: {p*100:.1f}%" for c, p in zip(config.SEVERITY_ORDER, predicted_proba)))
    selected_idx = config.SEVERITY_ORDER.index(pred_label)
    contrib = values[idx, :, selected_idx] if values.shape[2] > selected_idx else values[idx, :, 0]
    order = np.argsort(np.abs(contrib))[::-1][:14]
    fig = go.Figure(go.Bar(x=contrib[order], y=[str(names[i]).split("__")[-1] for i in order], orientation="h",
                           marker=dict(color=["#f87171" if v > 0 else "#38bdf8" for v in contrib[order]])))
    fig.update_xaxes(title=f"Contribution to {pred_label} (log-odds)")
    st.plotly_chart(plot_layout(fig, 460), width="stretch")
