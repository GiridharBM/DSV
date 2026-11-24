"""Shared UI helpers: banners, KPI grids, insight cards, empty states."""
from __future__ import annotations

import streamlit as st

from dashboard.theme import kpi


def section(title: str, subtitle: str = "") -> None:
    st.title(title)
    if subtitle:
        st.markdown(f'<div style="color:#94a3b8;margin-top:-10px;margin-bottom:18px">{subtitle}</div>', unsafe_allow_html=True)


def banner(text: str) -> None:
    st.markdown(f'<div class="banner">{text}</div>', unsafe_allow_html=True)


def kpi_grid(cards: list[tuple[str, str, str, str]], columns: int | None = None) -> None:
    columns = columns or min(len(cards), 4)
    cols = st.columns(columns)
    for i, (label, value, sub, color) in enumerate(cards):
        with cols[i % columns]:
            st.markdown(kpi(label, value, sub, color), unsafe_allow_html=True)


def insights_markdown(insights: list[dict]) -> None:
    for item in insights:
        st.markdown(
            f'<div class="insight"><b>{item.get("title", "Insight")}</b><br>{item.get("detail", "")}</div>',
            unsafe_allow_html=True,
        )


def unavailable(title: str, detail: str) -> None:
    st.warning(f"**{title}** — {detail}")


def csv_download(label: str, frame, filename: str) -> None:
    if frame is None or getattr(frame, "empty", True):
        st.info("Nothing to download for the current selection.")
        return
    st.download_button(label, frame.to_csv(index=False).encode("utf-8-sig"), filename, "text/csv", width="stretch")
