"""Download Binance spot monthly klines (1h and 5m) from data.binance.vision, verifying every .CHECKSUM.

Raw zips are kept under CACHE/binance/<same path as on the server>. A file already present with a matching
checksum is never downloaded again. The read-only ~/.cache/qal/binance tree is used as a source if it already
holds the exact file (copied, never modified). Generic User-Agent, no personal details.
"""

import hashlib
import logging
import shutil
import time
from concurrent.futures import ThreadPoolExecutor

import requests

from kvh import config

log = logging.getLogger(__name__)
BASE = "https://data.binance.vision"
UA = {"User-Agent": "Mozilla/5.0 (compatible; research-client/1.0)"}
INTERVALS = ("1h", "5m")


def months(first, last):
    y, m = first
    while (y, m) <= last:
        yield y, m
        m += 1
        if m == 13:
            y, m = y + 1, 1


def rel_path(symbol: str, interval: str, y: int, m: int) -> str:
    return f"data/spot/monthly/klines/{symbol}/{interval}/{symbol}-{interval}-{y:04d}-{m:02d}.zip"


def sha256(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _get(url: str, tries: int = 5):
    for k in range(tries):
        try:
            r = requests.get(url, headers=UA, timeout=60)
            if r.status_code == 404:
                return None
            r.raise_for_status()
            return r.content
        except requests.RequestException as e:  # pragma: no cover - network
            log.warning("retry %d %s: %s", k, url, e)
            time.sleep(2 ** k)
    raise RuntimeError(f"failed: {url}")


def fetch(rel: str) -> str:
    """Returns 'ok' (verified, cached), 'missing' (404) or raises on checksum mismatch."""
    dest = config.RAW / rel
    ck = dest.with_name(dest.name + ".CHECKSUM")
    if dest.exists() and ck.exists():
        if sha256(dest) == ck.read_text().split()[0]:
            return "ok"
    dest.parent.mkdir(parents=True, exist_ok=True)
    ck_bytes = _get(f"{BASE}/{rel}.CHECKSUM")
    if ck_bytes is None:
        return "missing"
    expected = ck_bytes.decode().split()[0]
    qal = config.QAL_CACHE / rel
    if qal.exists() and sha256(qal) == expected:  # reuse, read-only
        shutil.copyfile(qal, dest)
    else:
        body = _get(f"{BASE}/{rel}")
        if body is None:
            return "missing"
        tmp = dest.with_suffix(".part")
        tmp.write_bytes(body)
        if hashlib.sha256(body).hexdigest() != expected:
            tmp.unlink()
            raise RuntimeError(f"checksum mismatch: {rel}")
        tmp.replace(dest)
    ck.write_bytes(ck_bytes)
    return "ok"


def run() -> None:
    jobs = [rel_path(s, iv, y, m) for s in config.SYMBOLS for iv in INTERVALS
            for (y, m) in months(config.FIRST_MONTH, config.LAST_MONTH)]
    log.info("%d monthly files to check", len(jobs))
    counts = {"ok": 0, "missing": 0}
    missing = []
    with ThreadPoolExecutor(max_workers=4) as ex:
        for i, (rel, st) in enumerate(zip(jobs, ex.map(fetch, jobs))):
            counts[st] += 1
            if st == "missing":
                missing.append(rel)
            if (i + 1) % 100 == 0:
                log.info("%d/%d done %s", i + 1, len(jobs), counts)
    log.info("download finished: %s", counts)
    (config.RAW / "missing.txt").write_text("\n".join(missing))
    log.info("missing (not listed on the server, e.g. before listing): %d files, see %s", len(missing), config.RAW / "missing.txt")
