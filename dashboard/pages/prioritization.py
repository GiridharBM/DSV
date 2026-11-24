"""E. Intervention Priority Center — the visual centrepiece."""
from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from dashboard.components.common import banner, csv_download, kpi_grid, section
from dashboard.theme import BORDER, CYAN, PANEL, PANEL_2, SEVERITY, TEAL, TEXT, plot_layout
from src import config
from src.prioritization.ips import (
    DEFAULT_WEIGHTS, compute_condition_scores, normalise_weights, ranking_columns,
    ranking_stability, weight_sensitivity,
)


def _load_csv(name: str) -> pd.DataFrame:
    path = config.REPORTS_DIR / name
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


def render(state: dict) -> None:
    section("Intervention Priority Center", "PREDICT → EXPLAIN → PRIORITISE → VISUALISE")
    banner("**IPS = w₁·F + w₂·S + w₃·E.** F = normalised collision frequency, S = normalised severity impact, "
           "E = normalised SHAP importance. Weights are non-negative and normalised to sum to 1. "
           "This ranks *conditions for review* — it makes **no causal claim** and does not guarantee any "
           "intervention will reduce collisions.")

    clean = state["clean"].reset_index(drop=True)
    features = state["features"].reset_index(drop=True)
    cat_imp = _load_csv("shap_category_importance.csv")
    glob_imp = _load_csv("shap_global_importance.csv")
    if cat_imp.empty and glob_imp.empty:
        st.warning("SHAP importance outputs are missing, so the E component cannot be computed. "
                   "Run the analysis pipeline to generate genuine SHAP values.")
        return

    with st.expander("Adjust IPS weights", expanded=True):
        wf = st.slider("Frequency weight (F)", 0.0, 1.0, DEFAULT_WEIGHTS["frequency"], 0.05)
        ws = st.slider("Severity weight (S)", 0.0, 1.0, DEFAULT_WEIGHTS["severity"], 0.05)
        we = st.slider("SHAP weight (E)", 0.0, 1.0, DEFAULT_WEIGHTS["shap"], 0.05)
        try:
            weights = normalise_weights(wf, ws, we)
        except ValueError as exc:
            st.error(str(exc))
            return
        st.write(f"Normalised → **F {weights['frequency']:.2f}** · **S {weights['severity']:.2f}** · "
                 f"**E {weights['shap']:.2f}** (sum = 1.00)")

    with st.spinner("Recomputing priority ranking…"):
        ips = compute_condition_scores(clean, features, cat_imp, glob_imp, weights)
    if ips.empty:
        st.warning("Not enough condition rows to build a ranking.")
        return

    top = ips.iloc[0]
    kpi_grid([
        ("Highest-priority condition", str(top["condition"])[:26],
         f"{top['condition_type']} · IPS {top['IPS']:.3f}", CYAN),
        ("Conditions ranked", f"{len(ips):,}", "With ≥150 collisions each", "#a78bfa"),
        ("Its collision count", f"{int(top['collisions']):,}",
         f"Fatal+Serious {top['fatal_serious_pct']:.1f}% (n={int(top['collisions']):,})", SEVERITY["Serious"]),
        ("Its SHAP importance", f"{top['shap_importance']:.3f}", "Normalised E component", TEAL),
    ])
    st.markdown("<br>", unsafe_allow_html=True)

    top15 = ips.head(15).iloc[::-1]
    labels = top15["condition_type"] + " · " + top15["condition"]
    fig = go.Figure()
    for col, color, key in [("frequency", "#1d9bb5", "F (frequency)"),
                            ("severity", "#f59e0b", "S (severity)"),
                            ("shap_importance", "#14b8a6", "E (SHAP)")]:
        fig.add_trace(go.Bar(y=labels, x=weights[{"frequency": "frequency", "severity": "severity", "shap_importance": "shap"}[col]] * top15[col],
                             name=key, orientation="h", marker=dict(color=color),
                             customdata=top15["IPS"], hovertemplate="%{y}<br>" + key + ": %{x:.3f}<br>IPS: %{customdata:.3f}<extra></extra>"))
    fig.update_layout(barmode="stack")
    fig.update_xaxes(title="Weighted contribution to IPS")
    st.plotly_chart(plot_layout(fig, 520, "Top 15 conditions — IPS component breakdown"), width="stretch")

    st.markdown("---")
    tab1, tab2, tab3 = st.tabs(["Component scores", "Ranking comparisons", "Weight sensitivity"])
    with tab1:
        table = ips.head(30)[["rank", "condition_type", "condition", "collisions", "fatal", "serious",
                              "fatal_serious_pct", "frequency_score", "severity_score", "shap_importance_score", "IPS"]].copy()
        st.dataframe(table, width="stretch", height=520, hide_index=True,
                     column_config={
                         "rank": st.column_config.NumberColumn("Rank"),
                         "fatal_serious_pct": st.column_config.NumberColumn("F+S %", format="%.1f"),
                         "frequency_score": st.column_config.ProgressColumn("F (0-100)", min_value=0, max_value=100, format="%.1f"),
                         "severity_score": st.column_config.ProgressColumn("S (0-100)", min_value=0, max_value=100, format="%.1f"),
                         "shap_importance_score": st.column_config.ProgressColumn("E (0-100)", min_value=0, max_value=100, format="%.1f"),
                         "IPS": st.column_config.NumberColumn("IPS", format="%.3f"),
                     })
        csv_download("Download full intervention ranking (CSV)", ips, "intervention_priority_ranking.csv")
    with tab2:
        ranks = ranking_columns(ips)
        stability = ranking_stability(ips)
        shown = ranks.head(15).copy()
        shown["δ vs IPS (frequency)"] = shown["rank_IPS"] - shown["rank_frequency"]
        shown["δ vs IPS (severity)"] = shown["rank_IPS"] - shown["rank_severity"]
        st.dataframe(shown, width="stretch", height=430, hide_index=True)
        st.markdown("**Ranking stability** (Spearman correlation against the current IPS ranking):")
        st.dataframe(stability, width="stretch", hide_index=True)
        topn = ranks.head(10).iloc[::-1]
        fig = go.Figure()
        for col, color, name in [("rank_frequency", "#1d9bb5", "Frequency-only"),
                                 ("rank_severity", "#f59e0b", "Severity-only"),
                                 ("rank_shap", "#14b8a6", "SHAP-only"),
                                 ("rank_IPS", CYAN, "Combined IPS")]:
            fig.add_trace(go.Scatter(x=topn[col], y=topn["condition"], mode="lines+markers", name=name,
                                     line=dict(color=color, width=2), marker=dict(size=7)))
        fig.update_xaxes(title="Rank (1 = highest)", autorange="reversed")
        st.plotly_chart(plot_layout(fig, 460, "How the top 10 shifts across ranking strategies"), width="stretch")
    with tab3:
        sens = weight_sensitivity(clean, features, cat_imp, glob_imp, weights)
        if not sens.empty:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=sens["w_frequency"] * 100, y=sens["spearman_vs_baseline"],
                                     mode="lines+markers", name="Spearman ρ vs current ranking",
                                     line=dict(color=CYAN, width=3), marker=dict(size=8)))
            fig.add_trace(go.Scatter(x=sens["w_frequency"] * 100, y=sens["top10_overlap_with_baseline"] / 10,
                                     mode="lines+markers", name="Top-10 overlap (fraction)",
                                     line=dict(color="#f59e0b", width=2, dash="dot"), yaxis="y2"))
            fig.update_layout(yaxis=dict(title="Spearman ρ", range=[-1, 1]),
                              yaxis2=dict(title="Top-10 overlap", overlaying="y", side="right", range=[0, 1], showgrid=False))
            fig.update_xaxes(title="Frequency weight (%) — remainder split in current S:E ratio")
            st.plotly_chart(plot_layout(fig, 430, "Sensitivity of the ranking to weight changes"), width="stretch")
            st.dataframe(sens, width="stretch", hide_index=True)
        else:
            st.info("Sensitivity analysis unavailable.")

    with st.expander("Methodology & limitations"):
        st.markdown(f"""
- **Unit of analysis:** each condition is a (feature, category) pair such as `Weather = Fog or mist`, with at least 150 collisions.
- **F (frequency):** collisions in the condition ÷ the largest condition count. Denominators shown in the table.
- **S (severity):** mean analytic severity impact per collision (Fatal=5, Serious=3, Slight=1) ÷ the highest condition mean. These weights are an **analytic convention**, not official DfT costs.
- **E (SHAP):** mean absolute SHAP contribution for the matching encoded category (falling back to its parent feature), ÷ the largest value. Derived from the actual trained model.
- **IPS:** `{weights['frequency']:.3f}·F + {weights['severity']:.3f}·S + {weights['shap']:.3f}·E`.
- **Interpretation:** high frequency and high severity rate are different things; both are shown with denominators.
- **Limitations:** associations are not causes. Reporting practices, exposure (traffic volume) and confounding are not modelled. This tool does not claim that any intervention will be effective or guarantee reduced collisions.
""")
