"""Load raw STATS19 CSVs, validate their schema, and load processed artifacts."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from src import config
from src.utils.logging import get_logger

log = get_logger(__name__)

REQUIRED_RAW_COLUMNS = {"collision_severity", "date", "latitude", "longitude", "accident_year"}

# DfT renamed accident_* to collision_* from 2024 onwards; support both spellings.
ALIASES = {
    "accident_index": "collision_index",
    "accident_year": "collision_year",
    "accident_reference": "collision_ref_no",
    "accident_severity": "collision_severity",
    "location_easting_osgr": "location_easting_osgr",
}


def normalise_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [c.strip().lower().replace(" ", "_").replace("-", "_") for c in df.columns]
    return df.rename(columns={k: v for k, v in ALIASES.items() if k in df.columns and v not in df.columns})


def read_raw_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, low_memory=False, encoding="utf-8-sig")
    df = normalise_columns(df)
    log.info("Read %s: %s rows, %s columns", path.name, len(df), df.shape[1])
    return df


def load_raw_files(paths: list[Path] | None = None) -> tuple[pd.DataFrame, list[str]]:
    """Load and concatenate raw collision files. Returns (frame, warnings)."""
    paths = list(paths or [config.RAW_DIR / f"collision_{y}.csv" for y in config.DATA_YEARS])
    paths = [p for p in paths if Path(p).exists()]
    warnings: list[str] = []
    frames = []
    for path in paths:
        try:
            frames.append(read_raw_csv(Path(path)))
        except Exception as exc:
            warnings.append(f"Could not read {Path(path).name}: {exc}")
    if not frames:
        return pd.DataFrame(), warnings + ["No raw collision files were found."]

    df = pd.concat(frames, ignore_index=True, sort=False)
    missing = sorted(REQUIRED_RAW_COLUMNS - set(df.columns))
    if missing:
        warnings.append(f"Raw data is missing expected columns: {', '.join(missing)}")
    return df, warnings


def processed_path(name: str) -> Path:
    return config.PROCESSED_DIR / name


def load_processed(name: str = "collisions_clean.csv") -> pd.DataFrame:
    path = processed_path(name)
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path, low_memory=False)


def load_json(path: Path) -> dict:
    if not Path(path).exists():
        return {}
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


PROVENANCE_FILE = config.PROCESSED_DIR / "provenance.json"


def load_provenance() -> dict:
    return load_json(PROVENANCE_FILE)


def save_provenance(payload: dict) -> None:
    with PROVENANCE_FILE.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, default=str)
