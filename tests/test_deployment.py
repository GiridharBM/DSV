"""Deployment-focused tests for safe dataset and artifact loading."""
from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

os.environ["RAIP_AUTO_DOWNLOAD"] = "0"

from dashboard import data_state  # noqa: E402
from src import config  # noqa: E402
from src.data.sample import make_demo_frame  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def small_raw() -> pd.DataFrame:
    return make_demo_frame(30)


def test_validate_upload_rules():
    assert data_state.validate_upload(None)
    assert data_state.validate_upload(pd.DataFrame())
    assert data_state.validate_upload(pd.DataFrame({"foo": [1]}))
    assert not data_state.validate_upload(small_raw())


def test_state_from_valid_and_invalid_frames():
    good = data_state.state_from_frame(small_raw(), "valid.csv", uploaded=True)
    assert good["available"] and not good["features"].empty
    bad = data_state.state_from_frame(pd.DataFrame({"foo": [1]}), "bad.csv", uploaded=True)
    assert not bad["available"] and bad["validation_issues"]


def test_from_processed_uses_exists_guard(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "PROCESSED_DIR", tmp_path)
    monkeypatch.setattr(data_state, "PROCESSED_FILE", tmp_path / "collisions_clean.csv")
    missing = data_state._from_processed()
    assert not missing["available"] and "not present" in missing["error"]

    frame = small_raw()
    from src.features.engineering import build_features
    from src.preprocessing.cleaner import clean
    cleaned, _ = clean(frame)
    processed = pd.concat([cleaned.reset_index(drop=True), build_features(cleaned).reset_index(drop=True)], axis=1)
    processed.to_csv(tmp_path / "collisions_clean.csv", index=False)
    data_state._load_processed_cached.clear()
    present = data_state._from_processed()
    assert present["available"] and not present["features"].empty


def _app_script() -> str:
    return f'''
import os, sys, tempfile
sys.path.insert(0, r"{ROOT}")
os.environ["RAIP_AUTO_DOWNLOAD"] = "0"
from pathlib import Path
from src import config
_tmp = Path(tempfile.mkdtemp())
config.PROCESSED_DIR = _tmp
config.MODELS_DIR = _tmp
import dashboard.data_state as ds
ds.PROCESSED_FILE = _tmp / "collisions_clean.csv"
ds.PROVENANCE_FILE = _tmp / "provenance.json"
ds.MODEL_FILE = _tmp / "best_model.joblib"
import streamlit as st
from dashboard.pages import setup, prediction
s = ds.get_state()
st.write("AVAILABLE:" + str(s["available"]))
st.write("MODEL:" + str(s["model"] is not None))
setup.render(s)
prediction.render(s)
'''


def _run(uploaded=None) -> AppTest:
    test = AppTest.from_string(_app_script(), default_timeout=180)
    if uploaded is not None:
        test.session_state["uploaded_frame"] = uploaded
    return test.run()


def test_app_starts_with_no_dataset_and_no_model():
    test = _run()
    assert not test.exception
    assert any("AVAILABLE:False" in str(m.value) for m in test.markdown)
    assert any("MODEL:False" in str(m.value) for m in test.markdown)


def test_valid_upload_is_used():
    test = _run(small_raw())
    assert not test.exception
    assert any("AVAILABLE:True" in str(m.value) for m in test.markdown)


def test_invalid_and_empty_uploads_do_not_crash():
    for frame in (pd.DataFrame({"foo": [1]}), pd.DataFrame()):
        test = _run(frame)
        assert not test.exception
        assert any("AVAILABLE:False" in str(m.value) for m in test.markdown)
