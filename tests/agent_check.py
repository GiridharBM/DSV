"""One-shot agent API check against a headless Streamlit instance."""
import json
import os

import requests

port = os.getenv("RAIP_AGENT_PORT", "8765")
resp = requests.post(f"http://127.0.0.1:{port}/_stcore/agent/interact", json={}, timeout=180)
print("http", resp.status_code)
data = resp.json()
print("status", data.get("status"))
print("pages", [p.get("url_path") for p in data.get("pages", [])])
blob = json.dumps(data)
print("exception_present", '"type": "exception"' in blob or '"exception"' in blob)
print("material_icons", blob.count(":material/"))
for node in data.get("tree", {}).get("children", []):
    for child in node.get("children", []):
        if child.get("type") == "exception":
            print("exception detail", child.get("props"))
