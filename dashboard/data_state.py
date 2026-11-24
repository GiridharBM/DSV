"""Single entry point for dataset, provenance and trained artifacts.

Priority: uploaded session data -> local processed data -> optional official
download -> explicit upload UI. This module never assumes a file exists.
"""
from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
import streamlit as st

from src import config
from src.data.loader import PROVENANCE_FILE, load_json, load_processed, normalise_columns
from src.features.engineering import build_features
from src.models.train import load_best_model
from src.preprocessing.cleaner import clean, validate
from src.utils.logging import get_logger

log = get_logger(__name__)

PROCESSED_FILE = config.PROCESSED_DIR / "collisions_clean.csv"
MODEL_FILE = config.MODELS_DIR / "best_model.joblib"

REQUIRED_UPLOAD_COLUMNS = {"collision_severity", "date"}
OPTIONAL_INDEX_COLUMNS = ("collision_index", "accident_index")


# --------------------------------------------------------------------------- validation
def validate_upload(frame: pd.DataFrame | None) -> list[str]:
    """Return human-readable problems; empty list means the CSV can be processed."""
    if frame is None or not isinstance(frame, pd.DataFrame):
        return ["No CSV file was provided."]
    if frame.empty:
        return ["The uploaded CSV contains no rows."]
    issues = validate(frame)
    missing = sorted(REQUIRED_UPLOAD_COLUMNS - set(frame.columns))
    if missing:
        issues.append("Missing required columns: " + ", ".join(missing))
    if not any(col in frame.columns for col in OPTIONAL_INDEX_COLUMNS):
        issues.append("No collision ID column found; duplicate removal will use whole rows.")
    if "collision_severity" in frame.columns:
        severity = pd.to_numeric(frame["collision_severity"], errors="coerce")
        if severity.notna().sum() == 0:
            issues.append("`collision_severity` has no numeric 1/2/3 codes.")
    return issues


# --------------------------------------------------------------------------- cached loaders
@st.cache_data(show_spinner=False)
def _load_processed_cached(mtime_ns: int) -> pd.DataFrame:
    return load_processed()


@st.cache_resource(show_spinner=False)
def _load_model_cached(mtime_ns: int):
    return load_best_model()


@st.cache_data(show_spinner=False)
def _bootstrap_cached():
    """Download and clean official data once per app instance (opt-in)."""
    from src.data.bootstrap import acquire_official_data
    return acquire_official_data()


# --------------------------------------------------------------------------- state builders
def _empty_state(label: str, error: str, provenance: dict | None = None) -> dict:
    return {
        "available": False,
        "clean": pd.DataFrame(),
        "features": pd.DataFrame(),
        "demo": False,
        "uploaded": False,
        "label": label,
        "provenance": provenance or {},
        "error": error,
        "validation_issues": [],
    }


def state_from_frame(frame: pd.DataFrame, label: str, *, uploaded: bool = False,
                     provenance: dict | None = None) -> dict:
    """Turn a raw frame into the dashboard state. Never raises on bad input."""
    issues = validate_upload(frame)
    if issues:
        return _empty_state(label, "Uploaded CSV could not be used.", provenance) | {
            "uploaded": uploaded, "validation_issues": issues,
        }
    try:
        raw = normalise_columns(frame)
        cleaned, report = clean(raw)
        if cleaned.empty:
            return _empty_state(label, "No valid collision rows after cleaning.", provenance) | {
                "uploaded": uploaded, "validation_issues": report.warnings,
            }
        features = build_features(cleaned).reset_index(drop=True)
        cleaned = cleaned.reset_index(drop=True)
        if "date" in cleaned.columns:
            cleaned["date"] = pd.to_datetime(cleaned["date"], errors="coerce", format="mixed")
        prov = dict(provenance or {})
        prov.setdefault("source", "Manual CSV upload" if uploaded else "Official DfT STATS19")
        prov.setdefault("geography", "Great Britain" if not uploaded else "As provided by uploaded file")
        prov.setdefault("years", str(int(cleaned["collision_year"].min())) if "collision_year" in cleaned else "unknown")
        prov.setdefault("clean_rows", len(cleaned))
        prov.setdefault("demo", False)
        return {
            "available": True, "clean": cleaned, "features": features,
            "demo": False, "uploaded": uploaded, "label": label,
            "provenance": prov, "error": None, "validation_issues": [],
        }
    except Exception as exc:
        log.exception("Failed to build state from frame")
        return _empty_state(label, f"Could not process the dataset: {exc}", provenance) | {
            "uploaded": uploaded, "validation_issues": [str(exc)],
        }


def _from_processed() -> dict:
    if not PROCESSED_FILE.exists():
        return _empty_state("No local dataset", "Processed dataset is not present.")
    try:
        df = _load_processed_cached(PROCESSED_FILE.stat().st_mtime_ns)
    except Exception as exc:
        log.exception("Failed to load processed dataset")
        return _empty_state("Local dataset unreadable", f"Could not read processed CSV: {exc}")
    missing = [c for c in config.FEATURE_COLUMNS + [config.TARGET] if c not in df.columns]
    if df.empty or missing:
        return _empty_state("Local dataset invalid", f"Processed CSV is missing columns: {', '.join(missing[:5])}")
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce", format="mixed")
    prov = load_json(PROVENANCE_FILE)
    prov.setdefault("source", "Official DfT STATS19 open data")
    prov.setdefault("geography", "Great Britain")
    prov.setdefault("years", str(int(df["collision_year"].min())) if "collision_year" in df else "unknown")
    return {
        "available": True, "clean": df, "features": df[config.FEATURE_COLUMNS].copy(),
        "demo": False, "uploaded": False, "label": "DfT STATS19 (official open data)",
        "provenance": prov, "error": None, "validation_issues": [],
    }


def _from_bootstrap() -> dict:
    if os.getenv("RAIP_AUTO_DOWNLOAD", "1").strip().lower() in {"0", "false", "no", "off"}:
        return _empty_state("No local dataset", "Automatic download is disabled.")
    try:
        frame, warnings = _bootstrap_cached()
    except Exception as exc:
        log.exception("Automatic data acquisition failed")
        return _empty_state("Automatic download failed", str(exc))
    if frame is None or frame.empty:
        return _empty_state("Automatic download failed", "; ".join(warnings) or "No data returned.")
    state = state_from_frame(frame, "DfT STATS19 (auto-downloaded)")
    if state["available"]:
        state["provenance"]["source"] = "DfT STATS19 open data (auto-downloaded)"
        state["provenance"]["warnings"] = warnings
        state["bootstrap_warnings"] = warnings
    return state


def handle_upload(uploaded) -> None:
    """Read, validate and store an uploaded CSV in session state."""
    if uploaded is None:
        return
    if st.session_state.get("uploaded_name") == uploaded.name and st.session_state.get("uploaded_frame") is not None:
        return
    try:
        frame = pd.read_csv(uploaded, low_memory=False, encoding="utf-8-sig")
    except Exception as exc:
        st.session_state["upload_error"] = f"Could not read CSV: {exc}"
        return
    issues = validate_upload(normalise_columns(frame))
    if issues:
        st.session_state["upload_error"] = " | ".join(issues)
        return
    st.session_state["uploaded_frame"] = frame
    st.session_state["uploaded_name"] = uploaded.name
    st.session_state.pop("upload_error", None)
    st.cache_data.clear()
    st.cache_resource.clear()
    st.rerun()


def clear_upload() -> None:
    st.session_state.pop("uploaded_frame", None)
    st.session_state.pop("uploaded_name", None)
    st.session_state.pop("upload_error", None)
    st.cache_data.clear()
    st.cache_resource.clear()


def get_state() -> dict:
    state = _empty_state("No dataset", "No dataset loaded.")

    uploaded = st.session_state.get("uploaded_frame")
    if uploaded is not None:
        state = state_from_frame(uploaded, st.session_state.get("uploaded_name", "Uploaded CSV"), uploaded=True)

    if not state["available"]:
        local = _from_processed()
        if local["available"]:
            state = local

    if not state["available"]:
        fetched = _from_bootstrap()
        if fetched["available"]:
            state = fetched
        elif fetched.get("error"):
            state["error"] = f"{state['error']} {fetched['error']}".strip()

    state.setdefault("provenance", {})
    state.setdefault("validation_issues", [])
    state.setdefault("bootstrap_warnings", [])
    if MODEL_FILE.exists():
        try:
            state["model"] = _load_model_cached(MODEL_FILE.stat().st_mtime_ns)
        except Exception as exc:
            log.warning("Could not load model: %s", exc)
            state["model"] = None
    else:
        state["model"] = None
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
