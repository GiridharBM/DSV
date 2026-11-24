"""Small, deterministic DEMO dataset. Clearly labelled; never presented as real data."""
from __future__ import annotations

import numpy as np
import pandas as pd

from src import config


def make_demo_frame(n: int = 600) -> pd.DataFrame:
    """Generate a tiny synthetic frame solely to smoke-test the application.

    All figures derived from this frame are labelled DEMO in the UI and reports.
    """
    rng = np.random.default_rng(config.RANDOM_SEED)
    years = rng.choice(list(config.DATA_YEARS), n)
    hours = rng.integers(0, 24, n)
    weather = rng.choice([1, 2, 4, 7, 9], n, p=[0.55, 0.2, 0.1, 0.05, 0.1])
    light = rng.choice([1, 2, 4], n, p=[0.6, 0.3, 0.1])
    surface = rng.choice([1, 2, 5], n, p=[0.7, 0.27, 0.03])
    logit = (
        -1.2
        + 0.9 * ((weather == 7) | (weather == 9))
        + 0.8 * (light != 1)
        + 0.7 * (surface != 1)
        + 0.5 * ((hours < 6) | (hours >= 20))
    )
    p_fatal = 0.012 * np.exp(0.8 * logit)
    p_serious = 0.16 * np.exp(0.35 * logit)
    p_fatal = np.clip(p_fatal, 0, 0.4)
    p_serious = np.clip(p_serious, 0, 1 - p_fatal - 0.01)
    draw = rng.random(n)
    sev = np.where(draw < p_fatal, 1, np.where(draw < p_fatal + p_serious, 2, 3))
    frame = pd.DataFrame({
        "collision_index": [f"DEMO-{i:05d}" for i in range(n)],
        "collision_year": years,
        "collision_severity": sev,
        "date": pd.to_datetime({"year": years, "month": rng.integers(1, 13, n), "day": rng.integers(1, 28, n)}),
        "time": [f"{h:02d}:{m:02d}" for h, m in zip(hours, rng.integers(0, 60, n))],
        "day_of_week": rng.integers(1, 8, n),
        "number_of_vehicles": rng.integers(1, 5, n),
        "speed_limit": rng.choice([20, 30, 40, 50, 60, 70], n),
        "road_type": rng.choice([1, 2, 3, 6, 7], n),
        "junction_detail": rng.choice([0, 1, 3, 5, 8], n),
        "junction_control": rng.choice([0, 2, 4, 8], n),
        "light_conditions": light,
        "weather_conditions": weather,
        "road_surface_conditions": surface,
        "special_conditions_at_site": rng.choice([0, 1, 3, 5, 7], n),
        "carriageway_hazards": rng.choice([0, 2, 6, 9], n),
        "urban_or_rural_area": rng.choice([1, 2], n),
        "first_road_class": rng.choice([2, 3, 4, 6], n),
        "latitude": 51.0 + rng.normal(0, 0.5, n),
        "longitude": -1.0 + rng.normal(0, 0.7, n),
        # Real world coordinates are NOT generated: demo geo is marked unavailable.
        "local_authority_district": "DEMO",
    })
    return frame
