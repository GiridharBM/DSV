"""Headless smoke test: render every dashboard page against real artifacts."""
from pathlib import Path

from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parents[1]
CODE = f'''
import sys
sys.path.insert(0, r"{ROOT}")
import streamlit as st
from dashboard.data_state import get_state
from dashboard.pages import overview, explorer, prediction, explainability, prioritization, performance, geography
s = get_state()
for m in (overview, explorer, prediction, explainability, prioritization, performance, geography):
    st.write("PAGE:" + m.__name__)
    m.render(s)
'''

if __name__ == "__main__":
    result = AppTest.from_string(CODE, default_timeout=900).run()
    print("exceptions:", len(result.exception))
    for exc in result.exception:
        print(exc.message[:1500])
    print("pages rendered:", [m.value for m in result.markdown if str(m.value).startswith("PAGE:")])
