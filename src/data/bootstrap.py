"""Optional acquisition of official STATS19 data for cloud/clean environments."""
from __future__ import annotations

import os

import pandas as pd

from src import config
from src.data.download import download_all
from src.data.loader import load_raw_files, save_provenance
from src.features.engineering import build_features
from src.preprocessing.cleaner import clean
from src.utils.logging import get_logger

log = get_logger(__name__)


def acquire_official_data() -> tuple[pd.DataFrame, list[str]]:
    """Download recent collision CSVs, clean them, and persist the processed file.

    Controlled by RAIP_AUTO_DOWNLOAD (set to 0 to disable) and
    RAIP_AUTO_DOWNLOAD_YEARS (comma-separated; default 2023,2024,2025).
    """
    years = tuple(int(y) for y in os.getenv("RAIP_AUTO_DOWNLOAD_YEARS", "2023,2024,2025").split(",") if y.strip())
    years = years or (2023, 2024, 2025)
    warnings: list[str] = []
    log.info("Bootstrapping official data for years %s", years)
    paths = download_all(years=years)
    if not paths:
        return pd.DataFrame(), ["Could not download any STATS19 collision CSV."]

    raw, raw_warnings = load_raw_files(paths)
    warnings.extend(raw_warnings)
    if raw.empty:
        return pd.DataFrame(), warnings + ["Downloaded files contained no readable collision rows."]

    cleaned, report = clean(raw)
    if cleaned.empty:
        return pd.DataFrame(), warnings + ["No valid rows after cleaning."]
    features = build_features(cleaned)
    processed = pd.concat([cleaned.reset_index(drop=True), features.reset_index(drop=True)], axis=1)
    processed.to_csv(config.PROCESSED_DIR / "collisions_clean.csv", index=False)
    save_provenance({
        "source": config.SOURCE_PAGE,
        "dataset": "DfT STATS19 road collision records (auto-downloaded)",
        "geography": "Great Britain (NOT India)",
        "years": f"{int(cleaned['collision_year'].min())}-{int(cleaned['collision_year'].max())}",
        "files": [p.name for p in paths],
        "raw_rows": len(raw), "clean_rows": len(cleaned), "demo": False,
        "cleaning": report.as_dict(), "warnings": warnings,
    })
    log.info("Bootstrap complete: %s rows", len(processed))
    return processed, warnings
