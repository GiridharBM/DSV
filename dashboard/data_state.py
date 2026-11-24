"""Single cached entry point for dataset, provenance and trained artifacts."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import streamlit as st

from src import config
from src.data.loader import PROVENANCE_FILE, load_json, load_processed, normalise_columns
from src.data.sample import make_demo_frame
from src.features.engineering import build_features
from src.models.train import load_best_model
from src.preprocessing.cleaner import clean
from src.utils.logging import get_logger

log = get_logger(__name__)


@st.cache_data(show_spinner=False)
def _load_processed_cached(mtime_ns: int) -> pd.DataFrame:
    return load_processed()


@st.cache_resource(show_spinner=False)
def _load_model_cached(mtime_ns: int):
    return load_best_model()


def _from_upload(frame: pd.DataFrame) -> dict:
    cleaned, _ = clean(frame)
    if cleaned.empty:
        return {}
    feats = build_features(cleaned)
    return {
        "clean": cleaned.reset_index(drop=True),
        "features": feats.reset_index(drop=True),
        "demo": False,
        "uploaded": True,
        "label": "User-uploaded CSV",
        "provenance": {
            "source": "Manual CSV upload", "geography": "As provided by the uploaded file",
            "years": f"{int(cleaned['collision_year'].min())}-{int(cleaned['collision_year'].max())}"
            if "collision_year" in cleaned else "unknown",
            "clean_rows": len(cleaned), "demo": False,
        },
    }


def _from_processed() -> dict:
    path = config.PROCESSED_DIR / "collisions_clean.csv"
    df = _load_processed_cached(path.stat().st_mtime_ns)
    if df.empty:
        return {}
    missing = [c for c in config.FEATURE_COLUMNS + [config.TARGET] if c not in df.columns]
    if missing:
        log.warning("Processed file missing columns: %s", missing[:5])
        return {}
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce", format="mixed")
    return {
        "clean": df,
        "features": df[config.FEATURE_COLUMNS].copy(),
        "demo": False, "uploaded": False, "label": "DfT STATS19 (official open data)",
        "provenance": load_json(PROVENANCE_FILE),
    }


def _from_demo() -> dict:
    raw = make_demo_frame()
    cleaned, _ = clean(raw)
    feats = build_features(cleaned)
    return {
        "clean": cleaned.reset_index(drop=True), "features": feats.reset_index(drop=True),
        "demo": True, "uploaded": False, "label": "SYNTHETIC DEMO (not real data)",
        "provenance": {"source": "Synthetic demo generator", "geography": "None — synthetic",
                       "years": "2021-2025", "clean_rows": len(cleaned), "demo": True},
    }


def get_state() -> dict:
    uploaded = st.session_state.get("uploaded_frame")
    state = {}
    if uploaded is not None:
        state = _from_upload(normalise_columns(uploaded))
        if not state:
            st.sidebar.error("Uploaded CSV could not be parsed. Expected STATS19-style columns.")
    if not state:
        state = _from_processed()
    if not state:
        state = _from_demo()
    state.setdefault("provenance", {})
    model_path = config.MODELS_DIR / "best_model.joblib"
    state["model"] = _load_model_cached(model_path.stat().st_mtime_ns) if model_path.exists() else None
    state["model_meta"] = load_json(config.MODELS_DIR / "model_meta.json")
    state["shap_meta"] = load_json(config.MODELS_DIR / "shap_meta.json")
    return state


def load_table(name: str) -> pd.DataFrame:
    path = config.REPORTS_DIR / name
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def load_eval_summary() -> dict:
    return load_json(config.REPORTS_DIR / "evaluation_summary.json")
