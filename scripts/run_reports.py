"""Rebuild IPS rankings, insights and reports from already-trained artifacts."""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from src import config
from src.data.loader import load_json
from src.evaluation.report import write_pdf, write_tables
from src.prioritization.ips import (
    DEFAULT_WEIGHTS, compute_condition_scores, ranking_stability, weight_sensitivity,
)
from src.utils.insights import build_insights
from src.utils.logging import get_logger

log = get_logger("reports")


def main() -> int:
    processed = pd.read_csv(config.PROCESSED_DIR / "collisions_clean.csv", low_memory=False)
    summary = load_json(config.MODELS_DIR / "model_meta.json")
    provenance = load_json(config.PROCESSED_DIR / "provenance.json")
    cat_imp = pd.read_csv(config.REPORTS_DIR / "shap_category_importance.csv")
    glob_imp = pd.read_csv(config.REPORTS_DIR / "shap_global_importance.csv")

    bundle = {
        "results": summary.get("results", {}),
        "best_name": summary.get("best_model", "unknown"),
        "train_rows": summary.get("train_rows", 0),
        "test_rows": summary.get("test_rows", 0),
    }
    clean = processed.reset_index(drop=True)
    features = processed[config.FEATURE_COLUMNS].reset_index(drop=True)
    ips = compute_condition_scores(clean, features, cat_imp, glob_imp, DEFAULT_WEIGHTS)
    stability = ranking_stability(ips)
    sensitivity = weight_sensitivity(clean, features, cat_imp, glob_imp, DEFAULT_WEIGHTS)
    insights = build_insights(clean, features, cat_imp, glob_imp, ips)
    frames = write_tables(bundle, ips, stability, sensitivity, insights, provenance.get("cleaning", {}), provenance)
    ok = write_pdf(bundle, frames, ips, insights, glob_imp, provenance)
    log.info("Reports rebuilt: %s conditions, PDF=%s", len(ips), ok)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
