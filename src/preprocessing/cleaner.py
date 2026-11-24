"""Validation, cleaning, missing-value handling and duplicate removal for STATS19."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from src import config
from src.utils.logging import get_logger

log = get_logger(__name__)

# collision_index is optional: without it the cleaner de-duplicates on whole rows.
CORE_COLUMNS = ["collision_severity", "date"]
NUMERIC_RAW = [
    "number_of_vehicles", "speed_limit", "day_of_week", "junction_detail", "junction_control",
    "light_conditions", "weather_conditions", "road_surface_conditions", "road_type",
    "special_conditions_at_site", "carriageway_hazards", "urban_or_rural_area", "first_road_class",
    "latitude", "longitude", "collision_year",
]


@dataclass
class CleaningReport:
    rows_in: int = 0
    rows_out: int = 0
    duplicates_removed: int = 0
    invalid_severity_removed: int = 0
    invalid_date_removed: int = 0
    out_of_range_year_removed: int = 0
    numeric_missing_filled: dict = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "rows_in": self.rows_in, "rows_out": self.rows_out,
            "duplicates_removed": self.duplicates_removed,
            "invalid_severity_removed": self.invalid_severity_removed,
            "invalid_date_removed": self.invalid_date_removed,
            "out_of_range_year_removed": self.out_of_range_year_removed,
            "numeric_missing_filled": self.numeric_missing_filled,
            "warnings": self.warnings,
        }


def validate(raw: pd.DataFrame) -> list[str]:
    issues = []
    if raw is None or raw.empty:
        return ["Input frame is empty."]
    missing = [c for c in CORE_COLUMNS if c not in raw.columns]
    if missing:
        issues.append("Missing required columns: " + ", ".join(missing))
    if "collision_severity" in raw.columns:
        bad = set(pd.to_numeric(raw["collision_severity"], errors="coerce").dropna().unique()) - set(config.SEVERITY_MAP)
        if bad:
            issues.append(f"Unexpected severity codes: {sorted(bad)}")
    if not {"latitude", "longitude"}.issubset(raw.columns):
        issues.append("Geographic coordinates absent: map will show an unavailable state.")
    return issues


def _parse_dates(series: pd.Series) -> pd.Series:
    text = series.astype("string").str.strip()
    # STATS19 publishes DD/MM/YYYY; dayfirst=True is explicit.
    return pd.to_datetime(text, format="%d/%m/%Y", errors="coerce").fillna(
        pd.to_datetime(text, errors="coerce", format="mixed", dayfirst=True)
    )


def clean(raw: pd.DataFrame) -> tuple[pd.DataFrame, CleaningReport]:
    report = CleaningReport(rows_in=len(raw))
    df = raw.copy()
    issues = validate(df)
    report.warnings.extend(issues)
    if any("Missing required columns" in i for i in issues) or df.empty:
        return pd.DataFrame(), report

    for col in NUMERIC_RAW:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    before = len(df)
    df["collision_severity_label"] = df["collision_severity"].map(config.SEVERITY_MAP)
    report.invalid_severity_removed = int(df["collision_severity_label"].isna().sum())
    df = df.dropna(subset=["collision_severity_label"])

    df["date"] = _parse_dates(df["date"])
    report.invalid_date_removed = int(df["date"].isna().sum())
    df = df.dropna(subset=["date"])

    if "collision_year" not in df.columns:
        df["collision_year"] = df["date"].dt.year
    before_year = len(df)
    df = df[df["collision_year"].between(min(config.DATA_YEARS), max(config.DATA_YEARS))]
    report.out_of_range_year_removed = before_year - len(df)

    key = "collision_index" if "collision_index" in df.columns else None
    if key:
        report.duplicates_removed = int(df.duplicated(subset=[key]).sum())
        df = df.drop_duplicates(subset=[key], keep="first")
    else:
        report.duplicates_removed = int(df.duplicated().sum())
        df = df.drop_duplicates(keep="first")

    # Median imputation for model-relevant numeric fields. Missingness is recorded.
    for col in ["number_of_vehicles", "speed_limit"]:
        if col in df.columns:
            n_missing = int(df[col].isna().sum())
            if n_missing:
                report.numeric_missing_filled[col] = n_missing
                df[col] = df[col].fillna(df[col].median())

    # -1 is DfT's "data missing or out of range" code; keep it explicit.
    for col in ["latitude", "longitude"]:
        if col in df.columns:
            df[col] = df[col].replace(-1, np.nan)

    df = df.reset_index(drop=True)
    df["severity_impact_weight"] = df["collision_severity_label"].map(config.SEVERITY_IMPACT_WEIGHT)
    df["is_fatal_or_serious"] = df["collision_severity_label"].isin(["Fatal", "Serious"]).astype(int)
    report.rows_out = len(df)
    log.info("Cleaning: %s in -> %s out (%s duplicates, %s invalid severity)",
             report.rows_in, report.rows_out, report.duplicates_removed, report.invalid_severity_removed)
    return df, report
