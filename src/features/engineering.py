"""Deterministic, leakage-free feature engineering shared by training and inference."""
from __future__ import annotations

import numpy as np
import pandas as pd

from src import config

CODE_MAPS = {
    "day_of_week": config.DAY_OF_WEEK,
    "light_conditions": config.LIGHT_CONDITIONS,
    "weather_conditions": config.WEATHER_CONDITIONS,
    "road_surface_conditions": config.ROAD_SURFACE,
    "road_type": config.ROAD_TYPE,
    "junction_detail": config.JUNCTION_DETAIL,
    "junction_control": config.JUNCTION_CONTROL,
    "special_conditions_at_site": config.SPECIAL_CONDITIONS,
    "carriageway_hazards": config.CARRIAGEWAY_HAZARDS,
    "urban_or_rural_area": config.URBAN_RURAL,
    "first_road_class": config.ROAD_CLASS,
    "police_force": config.POLICE_FORCE,
}


def _map_code(series: pd.Series, mapping: dict) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    mapped = numeric.map(mapping)
    return mapped.fillna("Unknown").astype(str)


def _parse_time(series: pd.Series) -> pd.Series:
    text = series.astype("string").str.strip()
    parsed = pd.to_datetime(text, format="%H:%M", errors="coerce")
    fallback = pd.to_datetime(text, errors="coerce", format="mixed", dayfirst=True)
    return parsed.fillna(fallback)


def time_of_day(hour: float) -> str:
    if pd.isna(hour):
        return "Unknown"
    h = int(hour)
    if 6 <= h < 12:
        return "Morning (06-12)"
    if 12 <= h < 18:
        return "Afternoon (12-18)"
    if 18 <= h < 24:
        return "Evening (18-24)"
    return "Night (00-06)"


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Return a frame with the exact FEATURE_COLUMNS contract, plus metadata columns."""
    out = pd.DataFrame(index=df.index)
    time_dt = _parse_time(df["time"]) if "time" in df.columns else pd.Series(pd.NaT, index=df.index)
    date = pd.to_datetime(df["date"], errors="coerce") if "date" in df.columns else pd.Series(pd.NaT, index=df.index)
    hour = time_dt.dt.hour.astype("float64")

    # Numeric context available at the time of the collision.
    out["number_of_vehicles"] = pd.to_numeric(df.get("number_of_vehicles"), errors="coerce") if "number_of_vehicles" in df else np.nan
    out["speed_limit"] = pd.to_numeric(df.get("speed_limit"), errors="coerce") if "speed_limit" in df else np.nan
    out["hour"] = hour
    out["month"] = date.dt.month.astype("float64")
    dow = pd.to_numeric(df.get("day_of_week"), errors="coerce") if "day_of_week" in df else pd.Series(np.nan, index=df.index)
    out["is_weekend"] = dow.isin([6, 7]).astype(float).where(dow.notna(), np.nan)

    for col, mapping in CODE_MAPS.items():
        if col in df.columns:
            out[col] = _map_code(df[col], mapping)
        else:
            out[col] = "Unknown"

    out["time_of_day_band"] = hour.map(time_of_day).fillna("Unknown")
    out["is_night"] = ((hour < 6) | (hour >= 20)).astype(float).where(hour.notna(), np.nan)
    out["time_of_day_band"] = out["time_of_day_band"].astype(str)

    out = out[config.FEATURE_COLUMNS]
    # Impute here too so uploads with partial columns remain usable.
    for col in config.NUMERIC_FEATURES:
        out[col] = pd.to_numeric(out[col], errors="coerce").fillna(out[col].median())
    for col in config.CATEGORICAL_FEATURES:
        out[col] = out[col].fillna("Unknown").astype(str)
    return out


def frame_from_feature_values(values: dict):
    """Build the exact model input frame from already-mapped feature values."""
    import pandas as pd
    row = {col: values.get(col) for col in config.FEATURE_COLUMNS}
    frame = pd.DataFrame([row], columns=config.FEATURE_COLUMNS)
    # Derived fields come from the hour so the input can never be internally inconsistent.
    hour = pd.to_numeric(frame["hour"], errors="coerce").iloc[0]
    frame["is_night"] = float(hour < 6 or hour >= 20) if pd.notna(hour) else 0.0
    frame["time_of_day_band"] = time_of_day(hour)
    for col in config.NUMERIC_FEATURES:
        frame[col] = pd.to_numeric(frame[col], errors="coerce")
        if frame[col].isna().any():
            frame[col] = frame[col].fillna(0)
    for col in config.CATEGORICAL_FEATURES:
        frame[col] = frame[col].fillna("Unknown").astype(str)
    return frame
