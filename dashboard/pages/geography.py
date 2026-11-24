"""G. Geographic Intelligence — real coordinates only."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from dashboard.components.common import banner, kpi_grid, section
from dashboard.theme import CYAN, SEVERITY, plot_layout
from src import config


def render(state: dict) -> None:
    section("Geographic Intelligence", "Actual STATS19 grid-reference-derived coordinates; no synthetic locations.")
    clean = state["clean"]
    banner("**Map availability:** Great Britain only. Coordinates come from the source data; records without valid "
           "latitude/longitude are excluded from the map (and shown in the count). "
           + ("**DEMO DATA — synthetic coordinates suppressed.**" if state["demo"] else ""))

    if state["demo"]:
        st.warning("Geographic intelligence is unavailable for the synthetic demo dataset — real coordinates are "
                   "never generated. Load official STATS19 data to enable this page.")
        return
    if not {"latitude", "longitude"}.issubset(clean.columns):
        st.warning("The loaded dataset does not contain geographic coordinates. The map is unavailable.")
        return

    geo = clean.dropna(subset=["latitude", "longitude"]).copy()
    geo = geo[geo["latitude"].between(49, 61) & geo["longitude"].between(-9, 3)]
    if geo.empty:
        st.warning("No valid coordinates found in the loaded dataset. The map is unavailable.")
        return

    with st.expander("Geographic filters", expanded=True):
        c1, c2, c3 = st.columns(3)
        with c1:
            years = sorted(geo["collision_year"].astype(int).unique().tolist())
            yr = st.multiselect("Year", years, default=years, key="geo_year")
        with c2:
            sev = st.multiselect("Severity", config.SEVERITY_ORDER, default=config.SEVERITY_ORDER, key="geo_severity")
        with c3:
            if "police_force" in geo.columns:
                forces = sorted(geo["police_force"].dropna().astype(str).unique().tolist())
                selected_forces = st.multiselect("Police force (code)", forces)
            else:
                selected_forces = []

    view = geo[geo["collision_year"].astype(int).isin(yr) & geo[config.TARGET].isin(sev)]
    if selected_forces and "police_force" in view.columns:
        view = view[view["police_force"].astype(str).isin(selected_forces)]

    kpi_grid([
        ("Mapped collisions", f"{len(view):,}", f"of {len(clean):,} total records", CYAN),
        ("Valid coordinates", f"{100*len(geo)/max(1,len(clean)):.1f}%", "Of all loaded rows", "#a78bfa"),
        ("Fatal (mapped)", f"{int((view[config.TARGET]=='Fatal').sum()):,}", "In current map selection", SEVERITY["Fatal"]),
        ("Serious (mapped)", f"{int((view[config.TARGET]=='Serious').sum()):,}", "In current map selection", SEVERITY["Serious"]),
    ])
    st.markdown("<br>", unsafe_allow_html=True)

    map_df = view if len(view) <= 40000 else view.sample(n=40000, random_state=config.RANDOM_SEED)
    fig = go.Figure()
    for label in config.SEVERITY_ORDER:
        part = map_df[map_df[config.TARGET] == label]
        fig.add_trace(go.Scattermap(
            lat=part["latitude"], lon=part["longitude"], mode="markers", name=label,
            marker=dict(size=5 if label == "Slight" else 7, color=SEVERITY[label], opacity=0.6),
            text=part[config.TARGET].astype(str) if config.TARGET in part else None,
            hovertemplate=f"{label}<br>lat: %{{lat:.4f}}, lon: %{{lon:.4f}}<extra></extra>",
        ))
    fig.update_layout(map=dict(style="open-street-map", center=dict(lat=54, lon=-2), zoom=4.6))
    st.plotly_chart(plot_layout(fig, 600, "Collision locations by severity"), width="stretch")
    if len(view) > 40000:
        st.caption(f"Map shows a seeded 40,000-row sample of {len(view):,} mapped collisions for browser performance; "
                   "all counts and hotspot statistics use the full selection.")

    st.markdown("---")
    c1, c2 = st.columns([1, 1.2])
    with c1:
        st.subheader("Severity by police force area")
        if "police_force" in view.columns:
            area = view.groupby("police_force", observed=True)[config.TARGET].value_counts().unstack(fill_value=0)
            area = area.reindex(columns=config.SEVERITY_ORDER, fill_value=0)
            area = area.sort_values("Fatal", ascending=False).head(20)
            fig = go.Figure()
            for label in config.SEVERITY_ORDER:
                fig.add_trace(go.Bar(y=area.index.astype(str), x=area[label], name=label, orientation="h",
                                     marker_color=SEVERITY[label]))
            fig.update_layout(barmode="stack")
            st.plotly_chart(plot_layout(fig, 480), width="stretch")
        else:
            st.info("Police force area not available.")
    with c2:
        st.subheader("Hotspot analysis")
        grid = view.copy()
        grid["lat_bin"] = (np.floor(grid["latitude"] / 0.1) * 0.1).round(2)
        grid["lon_bin"] = (np.floor(grid["longitude"] / 0.1) * 0.1).round(2)
        hotspots = grid.groupby(["lat_bin", "lon_bin"], observed=True).agg(
            collisions=(config.TARGET, "size"),
            fatal=(config.TARGET, lambda s: int((s == "Fatal").sum())),
            serious=(config.TARGET, lambda s: int((s == "Serious").sum())),
        ).reset_index()
        hotspots["fatal_serious_pct"] = 100 * (hotspots["fatal"] + hotspots["serious"]) / hotspots["collisions"]
        hotspots = hotspots[hotspots["collisions"] >= 50].sort_values("collisions", ascending=False).head(15)
        if hotspots.empty:
            st.info("No grid cell with at least 50 collisions in the current selection.")
        else:
            st.dataframe(hotspots, width="stretch", hide_index=True)
            st.caption("Grid cells are ~0.1° latitude × 0.1° longitude. Counts are descriptive; "
                       "they are not adjusted for traffic exposure or population.")
