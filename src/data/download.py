"""Download official DfT STATS19 collision CSVs. Never fabricate a successful download."""
from __future__ import annotations

import time
from pathlib import Path

import requests

from src import config
from src.utils.logging import get_logger

log = get_logger(__name__)


def collision_path(year: int) -> Path:
    return config.RAW_DIR / f"collision_{year}.csv"


def download_year(year: int, *, force: bool = False, timeout: int = 180) -> Path | None:
    """Download one year of collision records. Returns the local path or None on failure."""
    dest = collision_path(year)
    if dest.exists() and dest.stat().st_size > 100_000 and not force:
        log.info("Using cached raw file %s", dest.name)
        return dest

    url = config.COLLISION_URL_TEMPLATE.format(year=year)
    tmp = dest.with_suffix(".part")
    try:
        log.info("Downloading %s", url)
        with requests.get(url, stream=True, timeout=timeout) as response:
            response.raise_for_status()
            with tmp.open("wb") as handle:
                for chunk in response.iter_content(chunk_size=1 << 20):
                    handle.write(chunk)
        if tmp.stat().st_size < 100_000:
            raise ValueError(f"Downloaded file implausibly small: {tmp.stat().st_size} bytes")
        tmp.replace(dest)
        log.info("Saved %s (%.1f MB)", dest.name, dest.stat().st_size / 1e6)
        return dest
    except Exception as exc:  # network, HTTP, disk
        log.warning("Download failed for %s: %s", year, exc)
        tmp.unlink(missing_ok=True)
        return None


def download_all(years=config.DATA_YEARS, *, force: bool = False) -> list[Path]:
    paths = []
    for year in years:
        result = download_year(year, force=force)
        if result is not None:
            paths.append(result)
        time.sleep(0.4)
    return paths


if __name__ == "__main__":
    found = download_all()
    print(f"Downloaded/available {len(found)} raw collision files")
