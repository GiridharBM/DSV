"""A. Executive Overview."""
from __future__ import annotations

import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from dashboard.components.common import banner, insights_markdown, kpi_grid, section
from dashboard.data_state import load_table
from dashboard.theme import CYAN, MUTED, SEVERITY, plot_layout
from src import config


def render(state: dict) -> None:
    clean = state["clean"]
    prov = state["provenance"]
    section("Executive Overview", "Recorded collisions, severity mix and the conditions that shape the picture.")

    demo = state["demo"]
    banner(
        f"**Data:** {state['label']} &nbsp;|&nbsp; **Years:** {prov.get('years', 'n/a')} &nbsp;|&nbsp; "
        f"**Rows:** {len(clean):,} &nbsp;|&nbsp; **Coverage:** {prov.get('geography', 'Great Britain')}. "
        + ("**DEMO DATA — synthetic, not real-world.** " if demo else "")
        + "STATS19 records police-reported injury collisions on public roads in Great Britain."
    )

    counts = clean[config.TARGET].value_counts()
    total = len(clean)
    fatal = int(counts.get("Fatal", 0))
    serious = int(counts.get("Serious", 0))
    slight = int(counts.get("Slight", 0))
    kpi_grid([
        ("Total collisions", f"{total:,}", "All records in current dataset", CYAN),
        ("Fatal", f"{fatal:,}", f"{100*fatal/total:.2f}% of collisions", SEVERITY["Fatal"]),
        ("Serious", f"{serious:,}", f"{100*serious/total:.2f}% of collisions", SEVERITY["Serious"]),
        ("Slight", f"{slight:,}", f"{100*slight/total:.2f}% of collisions", SEVERITY["Slight"]),
    ])
    st.markdown("<br>", unsafe_allow_html=True)
    kpi_grid([
        ("Fatal + Serious", f"{fatal+serious:,}", f"{100*(fatal+serious)/total:.2f}% of collisions", "#fb923c"),
        ("Mean severity impact", f"{clean['severity_impact_weight'].mean():.2f}",
         "Analytic weight 5/3/1 — not an official DfT value", "#a78bfa"),
    ], columns=2)

    yearly = clean.groupby(["collision_year", config.TARGET]).size().unstack(fill_value=0).reindex(columns=config.SEVERITY_ORDER, fill_value=0)
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    for label in config.SEVERITY_ORDER:
        fig.add_trace(go.Bar(x=yearly.index.astype(str), y=yearly[label], name=label,
                             marker_color=SEVERITY[label], hovertemplate="%{x}: %{y:,} " + label), secondary_y=False)
    rates = 100 * (yearly["Fatal"] + yearly["Serious"]) / yearly.sum(axis=1)
    fig.add_trace(go.Scatter(x=yearly.index.astype(str), y=rates, name="Fatal+Serious %",
                             mode="lines+markers", line=dict(color=CYAN, width=3),
                             marker=dict(size=9), hovertemplate="%{x}: %{y:.2f}%"), secondary_y=True)
    fig.update_yaxes(title_text="Collisions", secondary_y=False)
    fig.update_yaxes(title_text="Fatal+Serious %", secondary_y=True, showgrid=False)
    st.plotly_chart(plot_layout(fig, 420, "Annual collisions by severity (with serious/fatal share)"), width="stretch")

    left, right = st.columns([1, 1.4])
    with left:
        fig = go.Figure(go.Pie(labels=config.SEVERITY_ORDER, values=[fatal, serious, slight], hole=0.62,
                               marker=dict(colors=[SEVERITY[s] for s in config.SEVERITY_ORDER]),
                               textinfo="label+percent", hovertemplate="%{label}: %{value:,} (%{percent})"))
        fig.add_annotation(text=f"<b>{total:,}</b><br><span style='font-size:11px'>collisions</span>",
                           x=0.5, y=0.5, showarrow=False, font=dict(size=22, color="#e6edf7"))
        st.plotly_chart(plot_layout(fig, 400, "Severity mix"), width="stretch")
    with right:
        # Use the engineered numeric hour: uploaded raw files only carry a string `time` column.
        hours = pd.to_numeric(state["features"]["hour"], errors="coerce").dropna().astype(int)
        hour_counts = hours.value_counts().sort_index()
        fig = go.Figure(go.Bar(x=[f"{h:02d}:00" for h in hour_counts.index], y=hour_counts.values,
                               marker=dict(color=hour_counts.values, colorscale=[[0, "#155e75"], [1, CYAN]]),
                               hovertemplate="%{x}: %{y:,} collisions"))
        fig.update_xaxes(tickangle=-45)
        st.plotly_chart(plot_layout(fig, 400, "Collisions by time of day"), width="stretch")

    def condition_chart(column: str, title: str, top: int = 12):
        if column not in clean.columns:
            return
        grouped = clean.groupby(column, observed=True).agg(
            collisions=(config.TARGET, "size"),
            fatal_serious=(config.TARGET, lambda s: s.isin(["Fatal", "Serious"]).sum()),
        ).sort_values("collisions", ascending=False).head(top)
        grouped["rate"] = 100 * grouped["fatal_serious"] / grouped["collisions"]
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        fig.add_trace(go.Bar(y=grouped.index.astype(str)[::-1], x=grouped["collisions"][::-1], orientation="h",
                             name="Collisions", marker_color="#1d9bb5", hovertemplate="%{y}: %{x:,}"), secondary_y=False)
        fig.add_trace(go.Scatter(y=grouped.index.astype(str)[::-1], x=grouped["rate"][::-1], mode="markers+lines",
                                 name="Fatal+Serious %", marker=dict(color="#f59e0b", size=8),
                                 line=dict(color="#f59e0b", width=2), hovertemplate="%{y}: %{x:.1f}%"), secondary_y=True)
        fig.update_yaxes(secondary_y=True, showgrid=False)
        st.plotly_chart(plot_layout(fig, 420, title), width="stretch")

    c1, c2 = st.columns(2)
    with c1:
        condition_chart("weather_conditions", "Weather conditions")
    with c2:
        condition_chart("road_surface_conditions", "Road surface conditions")
    c3, c4 = st.columns(2)
    with c3:
        condition_chart("light_conditions", "Lighting conditions")
    with c4:
        condition_chart("road_type", "Road type")

    st.markdown("---")
    st.subheader("Automatic insights")
    from dashboard.data_state import load_table
    from src.prioritization.ips import DEFAULT_WEIGHTS, compute_condition_scores
    from src.utils.insights import build_insights

    try:
        cat_imp = load_table("shap_category_importance.csv")
        glob_imp = load_table("shap_global_importance.csv")
        ips_live = compute_condition_scores(
            clean.reset_index(drop=True), state["features"].reset_index(drop=True),
            cat_imp, glob_imp, DEFAULT_WEIGHTS,
        )
        insights = build_insights(clean, state["features"], cat_imp, glob_imp, ips_live)
    except Exception:
        import json
        insight_path = config.REPORTS_DIR / "insights.json"
        insights = json.loads(insight_path.read_text(encoding="utf-8")) if insight_path.exists() else []
    if insights:
        insights_markdown(insights)
    else:
        st.info("Insights are generated from the loaded dataset.")
