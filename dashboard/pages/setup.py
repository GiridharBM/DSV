"""Dataset acquisition UI shown when no usable dataset is present."""
from __future__ import annotations

import streamlit as st

from dashboard import data_state
from dashboard.components.common import section
from dashboard.theme import CYAN

REQUIRED = ["collision_severity", "date"]
USEFUL = ["collision_index", "collision_year", "time", "latitude", "longitude",
          "weather_conditions", "road_surface_conditions", "light_conditions",
          "road_type", "speed_limit", "day_of_week"]


def render(state: dict) -> None:
    section("Dataset setup", "The dashboard needs collision records before analytics can be shown.")

    st.info(
        "No processed collision dataset was found. Upload an official STATS19 collision CSV below, "
        "or allow the app to acquire one automatically (controlled by `RAIP_AUTO_DOWNLOAD`). "
        "Nothing on this page is synthetic — no charts are drawn until real records are available.",
        icon=":material/dataset:",
    )

    with st.container(border=True):
        st.subheader("1. Upload a collision CSV")
        st.caption(
            "Required columns: `collision_severity` (1 = Fatal, 2 = Serious, 3 = Slight) and `date` "
            "(DD/MM/YYYY). Optional but useful: " + ", ".join(f"`{c}`" for c in USEFUL) + "."
        )
        uploaded = st.file_uploader("STATS19 collision CSV", type=["csv"], key="setup_uploader",
                                    help="Official DfT collision-level CSV, e.g. collision_2025.csv")
        data_state.handle_upload(uploaded)

        upload_error = st.session_state.get("upload_error")
        if upload_error:
            st.error(f"CSV validation failed: {upload_error}")
        for issue in state.get("validation_issues", []):
            st.error(issue)
        if state.get("error") and not upload_error:
            st.warning(state["error"])

    with st.container(border=True):
        st.subheader("2. Download official data")
        st.markdown(
            "Get the collision CSVs from the [DfT Road safety open data page]"
            "(https://www.gov.uk/government/statistical-data-sets/road-safety-open-data). "
            "The file named `dft-road-casualty-statistics-collision-<year>.csv` matches this schema.\n\n"
            "For a full local rebuild: `python -m src.data.download` then `python scripts/run_pipeline.py`."
        )
        st.caption("Scope: Great Britain (not India). STATS19 covers police-reported personal-injury collisions.")

    with st.container(border=True):
        st.subheader("Current environment")
        col1, col2, col3 = st.columns(3)
        col1.metric("Dataset", "available" if state.get("available") else "missing")
        col2.metric("Trained model", "available" if state.get("model") is not None else "missing")
        col3.metric("SHAP outputs", "available" if state.get("shap_meta", {}).get("available") else "missing")
        if state.get("model") is None:
            st.caption("Prediction and explainability pages will show an unavailable state until "
                       "`models/best_model.joblib` is present.")
        if state.get("bootstrap_warnings"):
            st.caption("Auto-download notes: " + "; ".join(state["bootstrap_warnings"][:3]))
