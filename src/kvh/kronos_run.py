"""Run Kronos forecasts, incrementally: one parquet per (model, tag, symbol, month). A crash resumes.

A forecast for day d is made at 00:00 UTC of day d from the CONTEXT_LEN hourly candles that closed at or before
that instant, and samples N paths of the 24 hourly candles of day d. Per path we keep the close-to-close and
Parkinson variances (see kronos_fast.path_variances); everything else is derived from those.
"""

import json
import logging
import time
from datetime import date, timedelta

import numpy as np
import pandas as pd
import torch

from kvh import config
from kvh.data import load_hourly
from kvh.kronos_fast import FastKronos, path_variances, seed_for, time_features

log = logging.getLogger(__name__)

# sampling settings tried on validation (paper Table 6: forecasting T=0.6/top-p 0.9; volatility T=0.9/top-p 0.9)
TAGS = {
    "T06": dict(T=0.6, top_p=0.9),
    "T09": dict(T=0.9, top_p=0.9),
    "T10": dict(T=1.0, top_p=1.0),
}
N_SAMPLES = 16  # paths per forecast (subsets of 4/8/16 are scored on validation)
BATCH_DAYS = {"mini": 8, "small": 4, "base": 1}


def period_days(period: str):
    if period == "val":
        return config.VAL_START, config.VAL_END
    if period == "test":
        return config.TEST_START, config.TEST_END
    if period == "distill":
        return config.DISTILL_TRAIN_START, config.VAL_START - timedelta(days=1)
    if period == "slice":
        return date(2024, 7, 1), date(2024, 7, 31)
    raise ValueError(period)


def context_for(hourly: pd.DataFrame, day: date, L: int):
    """Hourly candles with open time in [t0 - L h, t0), t0 = 00:00 UTC of `day` (all closed by t0)."""
    t0 = pd.Timestamp(day, tz="UTC")
    win = hourly.loc[t0 - pd.Timedelta(hours=L): t0 - pd.Timedelta(hours=1)]
    if len(win) != L:
        return None
    x = win[["open", "high", "low", "close", "volume", "amount"]].to_numpy(np.float64)
    fut = pd.date_range(t0, periods=config.PRED_LEN, freq="1h", tz="UTC")
    return x, time_features(win.index), time_features(fut)


def chunk_path(model: str, tag: str, symbol: str, y: int, m: int):
    return config.KRONOS_OUT / model / tag / symbol / f"{y:04d}-{m:02d}.parquet"


def forecast_days(fk: FastKronos, hourly: pd.DataFrame, days, symbol: str, tag: str, n_samples: int, batch: int):
    rows = []
    s = TAGS[tag]
    for i in range(0, len(days), batch):
        chunk = [d for d in days[i:i + batch]]
        items = [(d, context_for(hourly, d, config.CONTEXT_LEN)) for d in chunk]
        items = [(d, c) for d, c in items if c is not None]
        if not items:
            continue
        ctx = np.stack([c[0] for _, c in items])
        sx = np.stack([c[1] for _, c in items])
        sy = np.stack([c[2] for _, c in items])
        t0 = time.process_time()
        w0 = time.perf_counter()
        paths, recon = fk.sample_paths(ctx, sx, sy, config.PRED_LEN, n_samples, s["T"], s["top_p"],
                                       seed_for(fk.name, tag, symbol, items[0][0]), return_context_recon=True)
        cpu = (time.process_time() - t0) / len(items)
        wall = (time.perf_counter() - w0) / len(items)
        anchor_act = ctx[:, -1, 3]
        v_act = path_variances(paths, anchor_act)
        v_rec = path_variances(paths, recon[:, 3])
        for j, (d, _) in enumerate(items):
            rows.append(dict(date=d, symbol=symbol, model=fk.name, tag=tag, anchor=float(anchor_act[j]),
                             recon_anchor=float(recon[j, 3]),
                             cc_act=v_act["cc"][j].tolist(), cc_rec=v_rec["cc"][j].tolist(), park=v_act["park"][j].tolist(),
                             cpu_s=cpu, wall_s=wall))
    return rows


def run(models, symbols=None, period="val", tag=None, stride=1) -> None:
    symbols = symbols or config.SYMBOLS
    if period in ("test", "distill") and tag is not None:
        raise SystemExit("test / distill forecasts use the frozen validation setting only (no --tag)")
    if tag is None:
        frozen_path = config.RESULTS / "frozen_kronos.json"
        fz = json.loads(frozen_path.read_text())
        tags_for = lambda mdl: [fz[mdl]["tag"]]  # noqa: E731
        if period == "test":
            import hashlib
            from datetime import datetime, timezone
            entry = dict(timestamp_utc=datetime.now(timezone.utc).isoformat(timespec="seconds"), models=models,
                         symbols=symbols, stride=stride, frozen_kronos_sha256=hashlib.sha256(frozen_path.read_bytes()).hexdigest())
            with open(config.RESULTS / "kronos_test_generation.jsonl", "a") as f:
                f.write(json.dumps(entry) + "\n")
    else:
        tags_for = lambda mdl: tag.split(",")  # noqa: E731
    start, end = period_days(period)
    for model in models:
        fk = FastKronos(model)
        batch = BATCH_DAYS[model]
        for tg in tags_for(model):
            for symbol in symbols:
                hourly = load_hourly(symbol)
                y, m = start.year, start.month
                while date(y, m, 1) <= end:
                    out = chunk_path(model, tg, symbol, y, m)
                    if out.exists():
                        log.info("skip (done) %s", out.relative_to(config.KRONOS_OUT))
                    else:
                        d0 = max(date(y, m, 1), start)
                        nm = date(y + (m == 12), m % 12 + 1, 1)
                        d1 = min(nm - timedelta(days=1), end)
                        days = [d0 + timedelta(days=k) for k in range((d1 - d0).days + 1)]
                        days = [d for d in days if (d - start).days % stride == 0]
                        w0 = time.perf_counter()
                        rows = forecast_days(fk, hourly, days, symbol, tg, N_SAMPLES, batch)
                        out.parent.mkdir(parents=True, exist_ok=True)
                        pd.DataFrame(rows).to_parquet(out)
                        log.info("%s %s %s %04d-%02d: %d forecasts in %.0fs (%.2fs/forecast)", model, tg, symbol, y, m,
                                 len(rows), time.perf_counter() - w0, (time.perf_counter() - w0) / max(len(rows), 1))
                    y, m = (y + 1, 1) if m == 12 else (y, m + 1)


def load_forecasts(model: str, tag: str, symbols=None, start=None, end=None) -> pd.DataFrame:
    base = config.KRONOS_OUT / model / tag
    parts = []
    for s in symbols or config.SYMBOLS:
        for p in sorted((base / s).glob("*.parquet")):
            parts.append(pd.read_parquet(p))
    if not parts:
        return pd.DataFrame()
    df = pd.concat(parts, ignore_index=True)
    df["date"] = pd.to_datetime(df["date"]).dt.date
    if start is not None:
        df = df[df["date"] >= start]
    if end is not None:
        df = df[df["date"] <= end]
    return df


def throughput() -> None:
    """Seconds per forecast (24-hour, N paths) per model on 2 threads, written to results/throughput.json."""
    from kvh.data import build_hourly
    hourly = build_hourly("BTCUSDT")
    days = [date(2024, 7, 1) + timedelta(days=k) for k in range(8)]
    res = {"threads": torch.get_num_threads(), "context": config.CONTEXT_LEN, "pred_len": config.PRED_LEN, "runs": []}
    for model in ("mini", "small", "base"):
        fk = FastKronos(model)
        for n, batch in ((16, BATCH_DAYS[model]), (32, max(1, BATCH_DAYS[model] // 2))):
            sel = days[:batch]
            forecast_days(fk, hourly, sel[:1], "BTCUSDT", "T09", n, 1)  # warm-up
            w0, c0 = time.perf_counter(), time.process_time()
            reps = 2
            for _ in range(reps):
                forecast_days(fk, hourly, sel, "BTCUSDT", "T09", n, batch)
            wall = (time.perf_counter() - w0) / (reps * len(sel))
            cpu = (time.process_time() - c0) / (reps * len(sel))
            res["runs"].append(dict(model=model, params=fk.n_params(), n_samples=n, batch_days=batch, wall_s=wall, cpu_s=cpu))
            log.info("%s N=%d batch=%d: %.2fs wall, %.2fs CPU per forecast", model, n, batch, wall, cpu)
    config.RESULTS.mkdir(parents=True, exist_ok=True)
    (config.RESULTS / "throughput.json").write_text(json.dumps(res, indent=1))
