"""Premium dark visual theme shared by all dashboard pages."""
from __future__ import annotations

import plotly.graph_objects as go

from src import config

NAVY = "#070f1e"
PANEL = "#0e1a2d"
PANEL_2 = "#13223a"
BORDER = "#1e3554"
TEXT = "#e6edf7"
MUTED = "#94a3b8"
TEAL = "#14b8a6"
CYAN = "#22d3ee"
BLUE = "#3b82f6"
SEVERITY = config.SEVERITY_COLORS

CUSTOM_CSS = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=Space+Grotesk:wght@500;600;700&display=swap');
:root {{ color-scheme: dark; }}
.stApp {{ background: radial-gradient(1200px 600px at 15% -10%, #12304a 0%, {NAVY} 55%) !important; }}
h1, h2, h3, h4 {{ font-family: 'Space Grotesk', 'Inter', sans-serif !important; letter-spacing: -0.02em; }}
h1 {{ font-size: 2.15rem !important; font-weight: 700 !important; }}
h2 {{ font-size: 1.45rem !important; font-weight: 600 !important; }}
h3 {{ font-size: 1.1rem !important; font-weight: 600 !important; }}
.stMarkdown p, .stMarkdown li, .stMarkdown td, .stMarkdown th {{ font-family: 'Inter', sans-serif !important; }}
.material-symbols-outlined, .material-symbols-rounded {{ font-family: 'Material Symbols Outlined', sans-serif !important; }}
[data-testid="stSidebar"] {{ background: linear-gradient(180deg, #0b1729 0%, #08111f 100%) !important; border-right: 1px solid {BORDER}; }}
[data-testid="stSidebar"] h2 {{ color: {CYAN} !important; }}
.stMetric, .stDataFrame, [data-testid="stTable"] {{ background: {PANEL} !important; border: 1px solid {BORDER} !important; border-radius: 14px !important; }}
div[data-testid="stForm"] {{ background: {PANEL} !important; border: 1px solid {BORDER} !important; border-radius: 16px !important; padding: 1.2rem !important; }}
.kpi {{ background: linear-gradient(145deg, {PANEL} 0%, {PANEL_2} 100%); border: 1px solid {BORDER}; border-radius: 16px; padding: 16px 18px; box-shadow: 0 12px 30px rgba(0,0,0,.25); min-height: 112px; }}
.kpi .label {{ color: {MUTED}; font-size: .78rem; text-transform: uppercase; letter-spacing: .08em; font-weight: 600; }}
.kpi .value {{ font-family: 'Space Grotesk', sans-serif; font-size: 1.85rem; font-weight: 700; line-height: 1.15; margin-top: 6px; }}
.kpi .sub {{ color: {MUTED}; font-size: .76rem; margin-top: 4px; }}
.kpi .accent {{ height: 3px; width: 42px; border-radius: 4px; margin-top: 10px; }}
.banner {{ border: 1px solid {BORDER}; background: {PANEL}; border-radius: 14px; padding: 12px 16px; color: {MUTED}; font-size: .85rem; }}
.banner strong {{ color: {TEXT}; }}
hr {{ border-color: {BORDER} !important; }}
.stButton > button {{ border-radius: 10px !important; border: 1px solid {TEAL} !important; background: transparent !important; color: {TEAL} !important; font-weight: 600 !important; }}
.stButton > button:hover {{ background: {TEAL} !important; color: #03131a !important; }}
div.stDownloadButton > button {{ border-radius: 10px !important; }}
[data-testid="stSlider"] {{ padding-top: .4rem; }}
.insight {{ border-left: 3px solid {CYAN}; background: {PANEL}; padding: 10px 14px; border-radius: 0 12px 12px 0; margin-bottom: 10px; }}
.insight b {{ color: {CYAN}; }}
footer {{ visibility: hidden; }}
</style>
"""


def apply_theme() -> None:
    import streamlit as st
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


def plot_layout(fig: go.Figure, height: int = 400, title: str | None = None) -> go.Figure:
    fig.update_layout(
        template="plotly_dark",
        height=height,
        title=title,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, sans-serif", color=TEXT, size=12),
        margin=dict(l=55, r=25, t=45 if title else 20, b=45),
        hoverlabel=dict(bgcolor=PANEL_2, font_color=TEXT, bordercolor=BORDER),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0, font=dict(size=11)),
    )
    fig.update_xaxes(gridcolor="rgba(148,163,184,.10)", zeroline=False)
    fig.update_yaxes(gridcolor="rgba(148,163,184,.10)", zeroline=False)
    return fig


def kpi(label: str, value: str, sub: str = "", color: str = CYAN) -> str:
    return (
        f'<div class="kpi"><div class="label">{label}</div>'
        f'<div class="value" style="color:{color}">{value}</div>'
        f'<div class="sub">{sub}</div><div class="accent" style="background:{color}"></div></div>'
    )
