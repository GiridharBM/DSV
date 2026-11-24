"""Manual smoke test for the optional official-data bootstrap path."""
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.environ["RAIP_AUTO_DOWNLOAD_YEARS"] = "2025"

from src import config  # noqa: E402

tmp = Path(tempfile.mkdtemp(prefix="raip_bootstrap_"))
config.RAW_DIR = tmp / "raw"
config.PROCESSED_DIR = tmp / "processed"
config.RAW_DIR.mkdir(parents=True, exist_ok=True)
config.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

from src.data.bootstrap import acquire_official_data  # noqa: E402

frame, warnings = acquire_official_data()
print("rows", len(frame))
print("files", config.RAW_DIR.name, sorted(p.name for p in config.RAW_DIR.glob('*.csv')))
print("warnings", warnings)
print("processed_exists", (config.PROCESSED_DIR / "collisions_clean.csv").exists())
