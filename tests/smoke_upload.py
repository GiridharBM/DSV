"""Reproduce the manual-upload path that previously crashed the overview page."""
from pathlib import Path

import pandas as pd
from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parents[1]
raw = pd.read_csv(ROOT / "data/raw/collision_2023.csv", nrows=500, low_memory=False)
test = AppTest.from_file(str(ROOT / "app.py"), default_timeout=300)
test.session_state["uploaded_frame"] = raw
test.session_state["uploaded_name"] = "dft-road-casualty-statistics-collision-2023.csv"
test.run()
print("exceptions:", len(test.exception))
for exc in test.exception:
    print(exc.message[:1200])
