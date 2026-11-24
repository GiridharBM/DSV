"""Run the full data -> model -> explain -> prioritise -> report pipeline."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from src import config
from src.data.loader import load_raw_files, save_provenance
from src.evaluation.report import write_pdf, write_tables
from src.explainability.shap_explain import generate_explainability
from src.features.engineering import build_features
from src.models.train import LABEL_TO_INT, save_artifacts, train_all
from src.preprocessing.cleaner import clean
from src.prioritization.ips import (
    DEFAULT_WEIGHTS, compute_condition_scores, ranking_stability, weight_sensitivity,
)
from src.utils.insights import build_insights
from src.utils.logging import get_logger

log = get_logger("pipeline")

KEEP = [
    "collision_index", "collision_year", "date", "time", "collision_severity", config.TARGET,
    "severity_impact_weight", "is_fatal_or_serious", "latitude", "longitude", "police_force",
    "local_authority_district", "number_of_casualties", "did_police_officer_attend_scene_of_accident",
]


def main() -> int:
    raw, raw_warnings = load_raw_files()
    if raw.empty:
        log.error("No raw data available. Run: python -m src.data.download")
        return 1

    cleaned, report = clean(raw)
    if cleaned.empty:
        log.error("Cleaning produced no usable rows")
        return 1
    features = build_features(cleaned)

    keep = [c for c in KEEP if c in cleaned.columns]
    processed = pd.concat([cleaned[keep].reset_index(drop=True), features.reset_index(drop=True)], axis=1)
    processed.to_csv(config.PROCESSED_DIR / "collisions_clean.csv", index=False)
    log.info("Saved processed dataset: %s rows x %s columns", *processed.shape)

    provenance = {
        "source": config.SOURCE_PAGE,
        "dataset": "DfT STATS19 road collision records (official open data)",
        "geography": "Great Britain (NOT India)",
        "years": f"{int(cleaned['collision_year'].min())}-{int(cleaned['collision_year'].max())}",
        "files": sorted(p.name for p in config.RAW_DIR.glob("collision_*.csv")),
        "raw_rows": len(raw), "clean_rows": len(cleaned),
        "demo": False, "raw_warnings": raw_warnings,
        "cleaning": report.as_dict(),
    }
    save_provenance(provenance)

    labels = cleaned[config.TARGET].map(LABEL_TO_INT).astype(int)
    years = cleaned["collision_year"].astype(int)
    bundle = train_all(features, labels, years)
    save_artifacts(bundle)

    shap_result = generate_explainability(bundle, bundle["X_test"])
    cat_imp = shap_result.get("category", pd.DataFrame()) if shap_result.get("available") else pd.DataFrame()
    glob_imp = shap_result.get("global", pd.DataFrame()) if shap_result.get("available") else pd.DataFrame()

    ips = compute_condition_scores(cleaned.reset_index(drop=True), features.reset_index(drop=True), cat_imp, glob_imp, DEFAULT_WEIGHTS)
    stability = ranking_stability(ips)
    sensitivity = weight_sensitivity(cleaned.reset_index(drop=True), features.reset_index(drop=True), cat_imp, glob_imp, DEFAULT_WEIGHTS)
    insights = build_insights(cleaned, features, cat_imp, glob_imp, ips)

    frames = write_tables(bundle, ips, stability, sensitivity, insights, report.as_dict(), provenance)
    ok = write_pdf(bundle, frames, ips, insights, glob_imp, provenance)
    log.info("Pipeline complete. PDF report: %s", "created" if ok else "unavailable")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
