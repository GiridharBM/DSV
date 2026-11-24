"""B. Accident Explorer."""
from __future__ import annotations

import pandas as pd
import streamlit as st

from dashboard.components.common import banner, csv_download, kpi_grid, section
from dashboard.theme import CYAN, SEVERITY, plot_layout
import plotly.graph_objects as go
from src import config


def _options(clean, column: str) -> list[str]:
    if column not in clean.columns:
        return []
    return sorted(clean[column].dropna().astype(str).unique().tolist())


def render(state: dict) -> None:
    clean = state["clean"]
    section("Accident Explorer", "Filter the underlying collision records and export the exact selection.")
    banner("All figures here are computed live from the filtered records. "
           + ("**DEMO DATA — synthetic.**" if state["demo"] else "Official DfT STATS19 collision records, Great Britain."))

    with st.expander("Filters", expanded=True):
        c1, c2, c3 = st.columns(3)
        with c1:
            years = sorted(clean["collision_year"].dropna().astype(int).unique().tolist())
            yr = st.multiselect("Collision year", years, default=years)
            date_min = clean["date"].min()
            date_max = clean["date"].max()
            dates = st.date_input("Date range", value=(date_min.date(), date_max.date()),
                                  min_value=date_min.date(), max_value=date_max.date())
        with c2:
            sev = st.multiselect("Severity", config.SEVERITY_ORDER, default=config.SEVERITY_ORDER, key="explorer_severity")
            weathers = st.multiselect("Weather", _options(clean, "weather_conditions"))
            roads = st.multiselect("Road type", _options(clean, "road_type"))
        with c3:
            lights = st.multiselect("Lighting", _options(clean, "light_conditions"))
            surfaces = st.multiselect("Road surface", _options(clean, "road_surface_conditions"))
            search = st.text_input("Search collision ID / authority", "")

    mask = clean["collision_year"].astype(int).isin(yr) & clean[config.TARGET].isin(sev)
    if isinstance(dates, tuple) and len(dates) == 2:
        mask &= clean["date"].between(pd.Timestamp(dates[0]), pd.Timestamp(dates[1]))
    for col, selected in [("weather_conditions", weathers), ("road_type", roads),
                          ("light_conditions", lights), ("road_surface_conditions", surfaces)]:
        if selected and col in clean.columns:
            mask &= clean[col].astype(str).isin(selected)
    if search.strip():
        text = clean.get("collision_index", clean.get("local_authority_district", "")).astype(str).str.lower()
        mask &= text.str.contains(search.strip().lower(), regex=False, na=False)

    view = clean[mask].copy()
    total = len(clean)
    kpi_grid([
        ("Matching collisions", f"{len(view):,}", f"{100*len(view)/total:.1f}% of {total:,} records", CYAN),
        ("Fatal", f"{int((view[config.TARGET]=='Fatal').sum()):,}", "In selection", SEVERITY["Fatal"]),
        ("Serious", f"{int((view[config.TARGET]=='Serious').sum()):,}", "In selection", SEVERITY["Serious"]),
        ("Slight", f"{int((view[config.TARGET]=='Slight').sum()):,}", "In selection", SEVERITY["Slight"]),
    ])
    st.markdown("<br>", unsafe_allow_html=True)

    if view.empty:
        st.warning("No collisions match the current filters. Widen the selection.")
        return

    fig = go.Figure()
    for label in config.SEVERITY_ORDER:
        part = view[view[config.TARGET] == label]
        fig.add_trace(go.Histogram(x=part["date"], name=label, marker_color=SEVERITY[label], nbinsx=40))
    fig.update_layout(barmode="stack")
    st.plotly_chart(plot_layout(fig, 300, "Filtered collisions over time"), width="stretch")

    display_cols = [c for c in ["collision_index", "date", "time", "collision_year", config.TARGET,
                                "weather_conditions", "road_surface_conditions", "light_conditions",
                                "road_type", "speed_limit", "latitude", "longitude"] if c in view.columns]
    st.dataframe(view[display_cols].head(5000), width="stretch", height=440,
                 column_config={config.TARGET: st.column_config.TextColumn("Severity")})
    if len(view) > 5000:
        st.caption(f"Showing first 5,000 of {len(view):,} matching records. Download the full selection below.")
    csv_download("Download filtered collisions (CSV)", view, "filtered_collisions.csv")
