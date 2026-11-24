"""F. Model Performance."""
from __future__ import annotations

import json

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from dashboard.components.common import banner, csv_download, kpi_grid, section
from dashboard.data_state import load_table
from dashboard.theme import CYAN, SEVERITY, TEAL, plot_layout
from src import config


def render(state: dict) -> None:
    section("Model Performance", "Honest evaluation on a chronological holdout — no resampling across years.")
    summary_path = config.REPORTS_DIR / "evaluation_summary.json"
    if not summary_path.exists():
        st.warning("No evaluation results found. Run the analysis pipeline first.")
        return
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    comparison = load_table("model_comparison.csv")
    classwise = load_table("model_classwise.csv")
    confusion = load_table("model_confusion.csv")
    best = summary.get("best_model")

    banner(f"**Selection rule:** {summary.get('selection_rule')} &nbsp;|&nbsp; **Best model:** {best} &nbsp;|&nbsp; "
           f"**Train:** through {summary.get('cleaning', {}).get('train_through_year', config.TRAIN_THROUGH_YEAR)} "
           f"&nbsp;|&nbsp; **Holdout rows:** {summary.get('test_rows', 0):,} &nbsp;|&nbsp; "
           f"**Training rows sampled:** {summary.get('train_rows', 0):,}. "
           "Macro-F1 and Fatal/Serious recall are emphasised because severe classes are rare.")

    if not comparison.empty:
        kpi_grid([
            ("Best model", str(best), "By weighted selection rule", CYAN),
            ("Accuracy", f"{comparison.loc[comparison['model'] == best, 'accuracy'].iloc[0]:.3f}",
             "Chronological holdout", TEAL),
            ("Macro-F1", f"{comparison.loc[comparison['model'] == best, 'macro_f1'].iloc[0]:.3f}",
             "Equal weight per class", "#a78bfa"),
            ("Fatal recall", f"{comparison.loc[comparison['model'] == best, 'fatal_recall'].iloc[0]:.3f}",
             "Share of true fatal collisions found", SEVERITY["Fatal"]),
        ])
        st.markdown("<br>", unsafe_allow_html=True)
        st.dataframe(comparison, width="stretch", hide_index=True)

        metrics = ["accuracy", "macro_f1", "weighted_f1", "fatal_recall", "serious_recall"]
        fig = go.Figure()
        for i, metric in enumerate(metrics):
            fig.add_trace(go.Bar(x=comparison["model"], y=comparison[metric], name=metric,
                                 marker_color=[CYAN, TEAL, "#a78bfa", SEVERITY["Fatal"], SEVERITY["Serious"]][i]))
        fig.update_layout(barmode="group")
        fig.update_yaxes(range=[0, 1], title="Score")
        st.plotly_chart(plot_layout(fig, 420, "Model comparison on the holdout year"), width="stretch")

    if not classwise.empty:
        fig = go.Figure()
        for label, color in zip(config.SEVERITY_ORDER, [SEVERITY[s] for s in config.SEVERITY_ORDER]):
            part = classwise[classwise["class"] == label]
            fig.add_trace(go.Bar(x=part["model"], y=part["recall"], name=f"{label} recall", marker_color=color))
        fig.update_layout(barmode="group")
        fig.update_yaxes(range=[0, 1], title="Recall")
        st.plotly_chart(plot_layout(fig, 400, "Class-wise recall"), width="stretch")
        st.dataframe(classwise, width="stretch", hide_index=True)

    if not confusion.empty:
        z = confusion.values.astype(float)
        fig = go.Figure(go.Heatmap(z=z, x=list(confusion.columns), y=list(confusion.index),
                                   colorscale=[[0, "#0e1a2d"], [0.5, "#0e7490"], [1, CYAN]],
                                   text=z.astype(int), texttemplate="%{text}", hovertemplate="%{y} → %{x}: %{z:,}"))
        fig.update_xaxes(title="Predicted")
        fig.update_yaxes(title="Actual", autorange="reversed")
        st.plotly_chart(plot_layout(fig, 430, "Confusion matrix (best model)"), width="stretch")

    results = summary.get("results", {})
    if best in results:
        st.subheader("Classification report")
        report = results[best].get("classification_report", {})
        rows = []
        for label in config.SEVERITY_ORDER:
            part = report.get(label, {})
            rows.append({"class": label, "precision": part.get("precision"), "recall": part.get("recall"),
                         "f1-score": part.get("f1-score"), "support": part.get("support")})
        st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)

    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        csv_download("Download model comparison (CSV)", comparison, "model_comparison.csv")
    with col2:
        csv_download("Download class-wise metrics (CSV)", classwise, "model_classwise.csv")
    st.caption("Accuracy alone is misleading with a ~1% fatal class. The selection rule adds an explicit "
               "Fatal/Serious recall bonus. All metrics come from the saved evaluation_summary.json.")
