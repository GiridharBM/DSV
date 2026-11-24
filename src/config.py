"""Central configuration: paths, seeds, mappings, and feature contracts."""
from __future__ import annotations

import os
from pathlib import Path

# --------------------------------------------------------------------------- paths
ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

for _p in (RAW_DIR, PROCESSED_DIR, MODELS_DIR, REPORTS_DIR, FIGURES_DIR):
    _p.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------------------------------- runtime
RANDOM_SEED = 42
DATA_YEARS = (2021, 2022, 2023, 2024, 2025)
TRAIN_THROUGH_YEAR = 2024          # chronological holdout: 2025 is the test year
MAX_MODEL_ROWS = int(os.getenv("RAIP_MAX_MODEL_ROWS", "150000"))
SHAP_EXPLAIN_ROWS = int(os.getenv("RAIP_SHAP_ROWS", "1500"))

DFT_BASE_URL = "https://data.dft.gov.uk/road-accidents-safety-data"
COLLISION_URL_TEMPLATE = DFT_BASE_URL + "/dft-road-casualty-statistics-collision-{year}.csv"
SOURCE_PAGE = "https://www.gov.uk/government/statistical-data-sets/road-safety-open-data"

# STATS19 collision_severity official mapping = 1 Fatal, 2 Serious, 3 Slight
SEVERITY_MAP = {1: "Fatal", 2: "Serious", 3: "Slight"}
SEVERITY_ORDER = ["Fatal", "Serious", "Slight"]
SEVERITY_COLORS = {"Fatal": "#ef4444", "Serious": "#f59e0b", "Slight": "#38bdf8"}
SEVERITY_IMPACT_WEIGHT = {"Fatal": 5.0, "Serious": 3.0, "Slight": 1.0}

# --------------------------------------------------------------------------- codes
DAY_OF_WEEK = {
    1: "Monday", 2: "Tuesday", 3: "Wednesday", 4: "Thursday",
    5: "Friday", 6: "Saturday", 7: "Sunday",
}
LIGHT_CONDITIONS = {
    1: "Daylight", 2: "Darkness - lights lit", 3: "Darkness - lights unlit",
    4: "Darkness - no lighting", 5: "Darkness - lighting unknown", -1: "Data missing",
}
WEATHER_CONDITIONS = {
    1: "Fine no high winds", 2: "Raining no high winds", 3: "Snowing no high winds",
    4: "Fine + high winds", 5: "Raining + high winds", 6: "Snowing + high winds",
    7: "Fog or mist", 8: "Other", 9: "Unknown", -1: "Data missing",
}
ROAD_SURFACE = {
    1: "Dry", 2: "Wet or damp", 3: "Snow", 4: "Frozen", 5: "Flood over 3cm deep",
    6: "Ice", -1: "Data missing",
}
ROAD_TYPE = {
    1: "Roundabout", 2: "Mini roundabout", 3: "Dual carriageway",
    6: "Single carriageway", 7: "One way street", 8: "Slip road",
    9: "Unknown", 12: "Not within junction", -1: "Data missing",
}
SPECIAL_CONDITIONS = {
    0: "None", 1: "School gate", 2: "Construction", 3: "Roadworks",
    4: "Flood", 5: "Ice", 6: "Snow", 7: "Fog", 8: "Dazzling sun",
    9: "Strong wind", 10: "Other", -1: "Data missing",
}
CARRIAGEWAY_HAZARDS = {
    0: "None", 1: "Vehicle load on road", 2: "Other object on road",
    3: "Previous accident", 4: "Dog on road", 5: "Other animal on road",
    6: "Pedestrian in carriageway", 7: "Nobody in carriageway",
    8: "Any object on road", 9: "Road debris", 10: "Dislodged vehicle load",
    -1: "Data missing",
}
JUNCTION_DETAIL = {
    0: "Not at junction or within 20 metres", 1: "Roundabout",
    2: "Mini-roundabout", 3: "T or staggered junction", 4: "Slip road",
    5: "Crossroads", 6: "More than 4 arms", 7: "Private drive or entrance",
    8: "Other junction", 9: "Not known", -1: "Data missing",
}
JUNCTION_CONTROL = {
    0: "Not at junction or within 20 metres", 1: "Authorised person",
    2: "Auto traffic signal", 3: "Auto traffic signal with pedestrian lights",
    4: "Give way or uncontrolled", 5: "Pedestrian phase at traffic signal",
    6: "Stop sign", 7: "Traffic light not in use", 8: "Uncontrolled",
    9: "Not known", -1: "Data missing",
}
URBAN_RURAL = {1: "Urban", 2: "Rural", 3: "Unclassified", -1: "Data missing"}
ROAD_CLASS = {
    1: "Motorway", 2: "A(M)", 3: "A", 4: "B", 5: "C", 6: "Unclassified",
    -1: "Data missing",
}
POLICE_FORCE = {
    1: "Metropolitan", 2: "City of London", 3: "Greater Manchester",
    4: "Merseyside", 5: "South Yorkshire", 6: "Tyne and Wear",
    7: "West Midlands", 8: "West Yorkshire", 10: "Devon and Cornwall",
    11: "Avon and Somerset", 12: "Bedfordshire", 13: "Cambridgeshire",
    14: "Cheshire", 15: "Cleveland", 16: "Cumbria", 17: "Derbyshire",
    18: "Dorset", 19: "Durham", 20: "Essex", 21: "Gloucestershire",
    22: "Hampshire", 23: "Hertfordshire", 24: "Humberside", 25: "Kent",
    26: "Lancashire", 27: "Leicestershire", 28: "Lincolnshire",
    29: "Norfolk", 30: "North Yorkshire", 31: "Northamptonshire",
    32: "Nottinghamshire", 33: "Suffolk", 34: "Surrey", 35: "Sussex",
    36: "Thames Valley", 37: "Warwickshire", 38: "Wiltshire",
    39: "Dyfed Powys", 40: "Gwent", 41: "North Wales", 42: "South Wales",
    43: "Northern Ireland", 44: "Guernsey", 45: "Isle of Man", 46: "Jersey",
    51: "Scottish Borders", 52: "Central Scotland", 53: "Dumfries and Galloway",
    54: "Fife", 55: "Grampian", 56: "Highland", 57: "Lothian and Borders",
    58: "North Strathclyde", 59: "South Strathclyde", 60: "Tayside", -1: "Data missing",
}

# --------------------------------------------------------------------------- feature contract
# Pre-collision / at-scene context ONLY. No casualty, post-collision or attendance fields.
NUMERIC_FEATURES = ["number_of_vehicles", "speed_limit", "hour", "month", "is_weekend", "is_night"]
CATEGORICAL_FEATURES = [
    "day_of_week", "light_conditions", "weather_conditions", "road_surface_conditions",
    "road_type", "junction_detail", "junction_control", "special_conditions_at_site",
    "carriageway_hazards", "urban_or_rural_area", "first_road_class", "time_of_day_band",
]
FEATURE_COLUMNS = NUMERIC_FEATURES + CATEGORICAL_FEATURES
TARGET = "collision_severity_label"

# Human-readable condition groups used by the Intervention Priority Engine.
CONDITION_SPECS = [
    ("weather_conditions", "Weather"),
    ("road_surface_conditions", "Road surface"),
    ("light_conditions", "Lighting"),
    ("road_type", "Road type"),
    ("junction_detail", "Junction type"),
    ("speed_limit", "Speed limit"),
    ("urban_or_rural_area", "Urban / rural"),
    ("time_of_day_band", "Time of day"),
    ("day_of_week", "Day of week"),
    ("special_conditions_at_site", "Site conditions"),
    ("carriageway_hazards", "Carriageway hazards"),
    ("first_road_class", "Road class"),
]
