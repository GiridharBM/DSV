"""Fast unit tests for the core data, IPS and evaluation logic."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src import config
from src.data.sample import make_demo_frame
from src.evaluation.metrics import evaluate
from src.features.engineering import build_features, frame_from_feature_values, time_of_day
from src.preprocessing.cleaner import clean
from src.prioritization.ips import (
    compute_condition_scores, normalise_weights, ranking_stability, weight_sensitivity,
)


def raw_frame(n: int = 12) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    base = {
        "collision_index": [f"X{i}" for i in range(n)],
        "collision_severity": [1, 2, 3] * (n // 3) + [3] * (n % 3),
        "date": ["01/06/2024"] * n,
        "time": ["09:30"] * n,
        "collision_year": [2024] * n,
        "number_of_vehicles": rng.integers(1, 4, n),
        "speed_limit": [30] * n,
        "day_of_week": [1] * n,
        "light_conditions": [1] * n,
        "weather_conditions": [1] * n,
        "road_surface_conditions": [1] * n,
        "road_type": [6] * n,
        "junction_detail": [0] * n,
        "junction_control": [0] * n,
        "special_conditions_at_site": [0] * n,
        "carriageway_hazards": [0] * n,
        "urban_or_rural_area": [1] * n,
        "first_road_class": [3] * n,
        "latitude": rng.uniform(50, 58, n),
        "longitude": rng.uniform(-6, 1, n),
    }
    return pd.DataFrame(base)[:n]


def test_cleaning_maps_official_severity_codes_and_drops_duplicates():
    raw = raw_frame(9)
    raw.loc[8] = raw.loc[0]
    cleaned, report = clean(raw)
    assert report.duplicates_removed == 1
    assert set(cleaned[config.TARGET]) == {"Fatal", "Serious", "Slight"}
    assert cleaned["severity_impact_weight"].eq(5).sum() == 3


def test_cleaning_rejects_empty_and_missing_columns():
    cleaned, report = clean(pd.DataFrame())
    assert cleaned.empty and report.warnings


def test_feature_contract_and_time_bands():
    cleaned, _ = clean(raw_frame())
    features = build_features(cleaned)
    assert list(features.columns) == config.FEATURE_COLUMNS
    assert set(features["time_of_day_band"]) == {"Morning (06-12)"}
    assert time_of_day(23) == "Evening (18-24)"
    one = frame_from_feature_values({"hour": 3, "time_of_day_band": "Night (00-06)"})
    assert one["is_night"].iloc[0] == 1.0
    assert list(one.columns) == config.FEATURE_COLUMNS


def test_weights_normalise_and_reject_negative():
    w = normalise_weights(2, 1, 1)
    assert sum(w.values()) == pytest.approx(1.0)
    assert w["frequency"] == pytest.approx(0.5)
    with pytest.raises(ValueError):
        normalise_weights(-1, 1, 1)
    with pytest.raises(ValueError):
        normalise_weights(0, 0, 0)


def test_ips_scores_bounded_and_ranking_stability_shape():
    cleaned, _ = clean(raw_frame(60))
    features = build_features(cleaned)
    features.loc[0:29, "weather_conditions"] = "Fine no high winds"
    features.loc[30:59, "weather_conditions"] = "Fog or mist"
    cat = pd.DataFrame([{"source_feature": "weather_conditions", "category": "Fog or mist", "e_raw": 2.0},
                        {"source_feature": "weather_conditions", "category": "Fine no high winds", "e_raw": 0.5}])
    glob = pd.DataFrame([{"feature": "cat__weather_conditions_Fog or mist", "mean_abs_shap": 2.0}])
    ips = compute_condition_scores(cleaned, features, cat, glob, {"frequency": 0.4, "severity": 0.35, "shap": 0.25}, min_rows=10)
    assert not ips.empty
    for col in ["frequency", "severity", "shap_importance"]:
        assert ips[col].between(0, 1).all()
    assert ips["IPS"].between(0, 1).all()
    assert {"rank", "collisions", "fatal", "serious"}.issubset(ips.columns)
    stability = ranking_stability(ips)
    assert len(stability) == 3
    assert not weight_sensitivity(cleaned, features, cat, glob,
                                  {"frequency": 0.4, "severity": 0.35, "shap": 0.25}, min_rows=10).empty


def test_metrics_perfect_and_imbalanced():
    y = np.array([0] * 5 + [1] * 10 + [2] * 85)
    m = evaluate(y, y)
    assert m["accuracy"] == 1.0
    assert m["macro_f1"] == 1.0
    assert m["recall"]["Fatal"] == 1.0
    assert np.trace(m["confusion_matrix"]) == 100


def test_demo_is_deterministic_and_flagged():
    a, b = make_demo_frame(50), make_demo_frame(50)
    pd.testing.assert_frame_equal(a, b)
    assert (a["collision_index"].astype(str).str.startswith("DEMO-")).all()
