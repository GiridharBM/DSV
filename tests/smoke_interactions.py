"""Headless interaction smoke test: prediction submit and IPS recompute."""
from pathlib import Path

from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parents[1]


def run(code: str, click: str | None = None):
    test = AppTest.from_string(f'import sys\nsys.path.insert(0, r"{ROOT}")\n' + code, default_timeout=600).run()
    if click:
        buttons = [b for b in test.button if b.label == click]
        if buttons:
            buttons[0].click().run()
        else:
            print("button not found:", click)
    print("exceptions:", len(test.exception))
    for exc in test.exception:
        print(exc.message[:1200])


if __name__ == "__main__":
    pred = '''
import streamlit as st
from dashboard.data_state import get_state
from dashboard.pages import prediction
prediction.render(get_state())
'''
    print("--- prediction submit ---")
    run(pred, "Predict severity")

    ips = '''
import streamlit as st
from dashboard.data_state import get_state
from dashboard.pages import prioritization
prioritization.render(get_state())
'''
    print("--- IPS default render ---")
    run(ips)
