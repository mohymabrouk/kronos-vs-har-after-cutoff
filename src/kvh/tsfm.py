"""Zero-shot next-day variance forecasts from three newer time-series foundation models (PLAN.md section 13):
TTM (r2.1 release), TimesFM-2.5 and Chronos-2, each scored only after its release.

* Dates. `scoring_start(name)` = max(TEST_START, effective cutoff + one calendar month), with effective cutoff =
  the later of the weights date and the latest dated training source on the model card (config.TSFM_MODELS).
* Inputs. Specification H (Kronos-matched): the hourly candles Kronos sees at 00:00 UTC of day d, turned into an
  hourly volatility series (`cc` = |hourly log return|, `park` = sqrt of the hourly Parkinson variance).
  Specification D (as in arXiv 2607.05291): daily sigma = sqrt(RV) of the days before d.
* Output. The model's mean forecast of that series; raw daily variance = sum over the 24 hours of the squared
  (clipped at 0) hourly forecasts (H), or the squared one-day forecast (D), floored at TSFM_FLOOR.
* Forecasts are written incrementally, one parquet per (model, spec, variant, symbol, month), so a crash resumes.
"""

import calendar
import hashlib
import json
import logging
import os
import time
from datetime import date, datetime, timedelta, timezone

import numpy as np
import pandas as pd
import torch

from kvh import config
from kvh.data import load_daily, load_hourly
from kvh.kronos_run import context_for

os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
log = logging.getLogger(__name__)
TSFM_OUT = config.CACHE / "tsfm"
FREQ_TOKEN = {"H": "h", "D": "d"}
HORIZON = {"H": config.PRED_LEN, "D": 1}
POINT_QUANTILES = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)
BATCH = 32


# ---- dates ------------------------------------------------------------------------------------------------------
def add_months(d: date, k: int) -> date:
    """Calendar-month shift; the day is clamped to the month's length (2025-01-31 + 1 -> 2025-02-28)."""
    y, m = divmod(d.month - 1 + k, 12)
    y, m = d.year + y, m + 1
    return date(y, m, min(d.day, calendar.monthrange(y, m)[1]))


def effective_cutoff(name: str) -> date:
    m = config.TSFM_MODELS[name]
    return max(m["weights_date"], m["stated_data_end"] or m["weights_date"])


def embargo_end(cutoff: date) -> date:
    return add_months(cutoff, config.TSFM_EMBARGO_MONTHS)


def scoring_start(name: str) -> date:
    return max(config.TEST_START, embargo_end(effective_cutoff(name)))


def label(name: str, spec: str) -> str:
    return f"{config.TSFM_MODELS[name]['label']} ({'hourly' if spec == 'H' else 'daily RV'})"


def avg_label(spec: str) -> str:
    return f"Avg: {label('ttm', spec)} + log-HAR"


# ---- input series and the variance transform ---------------------------------------------------------------------
def hourly_vol(x: np.ndarray, variant: str) -> np.ndarray:
    """x [L, 6] hourly OHLCV + quote volume -> [L] hourly volatility proxy (decimal, one hour)."""
    o, h, lo_, c = (np.maximum(x[:, k], 1e-12) for k in range(4))
    if variant == "cc":
        prev = np.r_[o[0], c[:-1]]  # the first hour uses its own open
        return np.abs(np.log(c / prev))
    if variant == "park":
        hi = np.maximum.reduce([o, h, lo_, c])
        lo = np.minimum.reduce([o, h, lo_, c])
        return np.log(hi / lo) / (2.0 * np.sqrt(np.log(2.0)))
    raise ValueError(variant)


def daily_sigma(daily: pd.DataFrame, day: date, n: int):
    """sqrt(RV) of the n days before `day` (RV of day d-1 is known at 00:00 of day d), or None if fewer."""
    rv = daily["rv"]
    past = rv[rv.index < day].to_numpy()
    if len(past) < n:
        return None
    return np.sqrt(past[-n:])


def series_for(name: str, spec: str, variant: str, day: date, hourly=None, daily=None):
    m = config.TSFM_MODELS[name]
    if spec == "H":
        c = context_for(hourly, day, m["context_h"])
        return None if c is None else hourly_vol(c[0], variant)
    return daily_sigma(daily, day, m["context_d"])


def raw_variance(point: np.ndarray, spec: str) -> np.ndarray:
    """point [B, horizon] forecast of the volatility series -> raw daily variance [B] (not yet floored)."""
    s = np.maximum(np.asarray(point, float), 0.0)
    if spec == "H":
        return np.square(s[:, :config.PRED_LEN]).sum(axis=1)
    return np.square(s[:, 0])


def vol_average(a, b):
    """Equal-weight average on the volatility scale, squared (arXiv 2607.05291): ((sqrt a + sqrt b) / 2)^2."""
    return ((np.sqrt(np.asarray(a, float)) + np.sqrt(np.asarray(b, float))) / 2.0) ** 2


# ---- model wrappers: predict(list of 1-d series, horizon, spec) -> mean forecast [B, horizon] ----------------------
def _cache_dir() -> str:
    return str(config.HF_HOME)


class TTM:
    """granite-tsfm TinyTimeMixerForPrediction; context standard-scaled per window (as the card asks), frequency
    token 'h' or 'd' (the r2.1 ft checkpoint uses frequency prefix tuning)."""

    def __init__(self):
        from tsfm_public import TinyTimeMixerForPrediction
        m = config.TSFM_MODELS["ttm"]
        self.model = TinyTimeMixerForPrediction.from_pretrained(m["repo"], revision=m["revision"], cache_dir=_cache_dir()).eval()
        self.context = int(self.model.config.context_length)

    def n_params(self) -> int:
        return int(sum(p.numel() for p in self.model.parameters()))

    @torch.no_grad()
    def predict(self, series, horizon: int, spec: str) -> np.ndarray:
        from tsfm_public.toolkit.time_series_preprocessor import DEFAULT_FREQUENCY_MAPPING
        x = np.stack([np.asarray(s, np.float64) for s in series])
        assert x.shape[1] == self.context, f"TTM needs exactly {self.context} values, got {x.shape[1]}"
        mu, sd = x.mean(axis=1, keepdims=True), x.std(axis=1, keepdims=True)
        sd = np.where(sd > 0, sd, 1.0)
        z = torch.tensor(((x - mu) / sd)[..., None], dtype=torch.float32)
        tok = torch.full((len(x),), DEFAULT_FREQUENCY_MAPPING[FREQ_TOKEN[spec]], dtype=torch.long)
        out = self.model(past_values=z, freq_token=tok, return_loss=False).prediction_outputs[..., 0].numpy()
        return out[:, :horizon].astype(np.float64) * sd + mu


class TimesFM:
    """timesfm TimesFM_2p5_200M_torch with the model-card ForecastConfig; returns the mean channel (index 0)."""

    def __init__(self):
        import timesfm
        m = config.TSFM_MODELS["timesfm"]
        self._timesfm = timesfm
        self.model = timesfm.TimesFM_2p5_200M_torch.from_pretrained(m["repo"], revision=m["revision"], cache_dir=_cache_dir(),
                                                                    torch_compile=False)
        self._compiled_for = None

    def n_params(self) -> int:
        return int(sum(p.numel() for p in self.model.model.parameters()))

    def _compile(self, context: int) -> None:
        ctx = int(np.ceil(context / 32) * 32)
        if self._compiled_for == ctx:
            return
        self.model.compile(self._timesfm.ForecastConfig(
            max_context=ctx, max_horizon=128, normalize_inputs=True, use_continuous_quantile_head=True,
            force_flip_invariance=True, infer_is_positive=True, fix_quantile_crossing=True, per_core_batch_size=BATCH))
        self._compiled_for = ctx

    @torch.no_grad()
    def predict(self, series, horizon: int, spec: str) -> np.ndarray:
        self._compile(max(len(s) for s in series))
        _, q = self.model.forecast(horizon=horizon, inputs=[np.asarray(s, np.float64) for s in series])
        return np.asarray(q[..., 0], np.float64)  # [B, horizon]: the mean head


class Chronos2:
    """chronos-forecasting Chronos2Pipeline, univariate, no cross-learning; point = mean of the 0.1..0.9 quantiles."""

    def __init__(self):
        from chronos import Chronos2Pipeline
        m = config.TSFM_MODELS["chronos2"]
        self.pipe = Chronos2Pipeline.from_pretrained(m["repo"], revision=m["revision"], cache_dir=_cache_dir())
        self.pipe.model.float().eval()
        ql = list(self.pipe.quantiles)
        self.qidx = [int(np.argmin(np.abs(np.asarray(ql) - q))) for q in POINT_QUANTILES]
        assert np.allclose([ql[i] for i in self.qidx], POINT_QUANTILES)

    def n_params(self) -> int:
        return int(sum(p.numel() for p in self.pipe.model.parameters()))

    @torch.no_grad()
    def predict(self, series, horizon: int, spec: str) -> np.ndarray:
        preds = self.pipe.predict([torch.tensor(np.asarray(s), dtype=torch.float32) for s in series],
                                  prediction_length=horizon, batch_size=BATCH)
        arr = torch.stack([p[0] for p in preds]).float().numpy()  # [B, Q, horizon]
        return arr[:, self.qidx, :horizon].mean(axis=1).astype(np.float64)


WRAPPERS = {"ttm": TTM, "timesfm": TimesFM, "chronos2": Chronos2}


def cached_weights(name: str):
    """Path of the pinned weight file in the local HF cache, or None (tests that need the weights skip then)."""
    from huggingface_hub import try_to_load_from_cache
    m = config.TSFM_MODELS[name]
    p = try_to_load_from_cache(m["repo"], "model.safetensors", cache_dir=_cache_dir(), revision=m["revision"])
    return p if isinstance(p, str) and os.path.exists(p) else None


# ---- generation (resumable) -----------------------------------------------------------------------------------
def chunk_path(name: str, spec: str, variant: str, symbol: str, y: int, m: int):
    return TSFM_OUT / name / spec / variant / symbol / f"{y:04d}-{m:02d}.parquet"


def forecast_days(wrap, name: str, spec: str, variant: str, symbol: str, days, hourly=None, daily=None) -> list:
    rows = []
    for i in range(0, len(days), BATCH):
        items = [(d, series_for(name, spec, variant, d, hourly, daily)) for d in days[i:i + BATCH]]
        missing = [d for d, s in items if s is None]
        if missing:
            log.warning("%s %s %s: no full context for %s", name, spec, symbol, missing)
        items = [(d, s) for d, s in items if s is not None]
        if not items:
            continue
        t0, w0 = time.process_time(), time.perf_counter()
        point = wrap.predict([s for _, s in items], HORIZON[spec], spec)
        cpu = (time.process_time() - t0) / len(items)
        wall = (time.perf_counter() - w0) / len(items)
        raw = raw_variance(point, spec)
        for j, (d, _) in enumerate(items):
            rows.append(dict(date=d, symbol=symbol, model=name, spec=spec, variant=variant,
                             raw=float(max(raw[j], config.TSFM_FLOOR)), floored=bool(raw[j] < config.TSFM_FLOOR),
                             n_neg=int((point[j, :HORIZON[spec]] < 0).sum()), point=point[j, :HORIZON[spec]].tolist(),
                             cpu_s=cpu, wall_s=wall))
    return rows


def period_bounds(name: str, period: str):
    if period == "val":
        return config.VAL_START, config.VAL_END
    if period == "test":
        return scoring_start(name), config.TEST_END
    raise ValueError(period)


def frozen_tsfm_path():
    return config.RESULTS / "frozen_tsfm.json"


def load_frozen_tsfm() -> tuple:
    p = frozen_tsfm_path()
    text = p.read_text()
    digest = hashlib.sha256(text.encode()).hexdigest()
    stored = (config.RESULTS / "frozen_tsfm.sha256").read_text().strip()
    if digest != stored:
        raise RuntimeError("frozen_tsfm.json was modified after it was frozen (hash mismatch)")
    return json.loads(text), digest


def variants_for(name: str, spec: str, period: str, frozen=None):
    if spec == "D":
        return ["sigma"]
    if period == "test":
        return [frozen["models"][name]["H"]["variant"]]
    return list(config.TSFM_H_VARIANTS)


def run(models, period: str = "val", specs=("H", "D"), symbols=None) -> None:
    symbols = symbols or config.SYMBOLS
    frozen = None
    if period == "test":
        frozen, digest = load_frozen_tsfm()  # test forecasts: frozen settings only
        entry = dict(timestamp_utc=datetime.now(timezone.utc).isoformat(timespec="seconds"), models=list(models),
                     specs=list(specs), symbols=symbols, frozen_tsfm_sha256=digest,
                     windows={n: [str(scoring_start(n)), str(config.TEST_END)] for n in models})
        with open(config.RESULTS / "tsfm_test_generation.jsonl", "a") as f:
            f.write(json.dumps(entry) + "\n")
    params_path = config.RESULTS / "tsfm_params.json"
    params = json.loads(params_path.read_text()) if params_path.exists() else {}
    for name in models:
        wrap = WRAPPERS[name]()
        params[name] = wrap.n_params()
        params_path.write_text(json.dumps(params, indent=1))
        start, end = period_bounds(name, period)
        log.info("%s: %s period %s .. %s, %s parameters", name, period, start, end, f"{params[name]:,}")
        for spec in specs:
            for variant in variants_for(name, spec, period, frozen):
                for symbol in symbols:
                    hourly = load_hourly(symbol) if spec == "H" else None
                    daily = load_daily(symbol) if spec == "D" else None
                    y, m = start.year, start.month
                    while date(y, m, 1) <= end:
                        out = chunk_path(name, spec, variant, symbol, y, m)
                        if not out.exists():
                            d0 = max(date(y, m, 1), start)
                            d1 = min(add_months(date(y, m, 1), 1) - timedelta(days=1), end)
                            days = [d0 + timedelta(days=k) for k in range((d1 - d0).days + 1)]
                            w0 = time.perf_counter()
                            rows = forecast_days(wrap, name, spec, variant, symbol, days, hourly, daily)
                            out.parent.mkdir(parents=True, exist_ok=True)
                            pd.DataFrame(rows).to_parquet(out)
                            log.info("%s %s %s %s %04d-%02d: %d forecasts in %.1fs", name, spec, variant, symbol, y, m,
                                     len(rows), time.perf_counter() - w0)
                        y, m = (y + 1, 1) if m == 12 else (y, m + 1)


def load(name: str, spec: str, variant: str, start=None, end=None, symbols=None) -> pd.DataFrame:
    parts = [pd.read_parquet(p) for s in (symbols or config.SYMBOLS)
             for p in sorted((TSFM_OUT / name / spec / variant / s).glob("*.parquet"))]
    if not parts:
        return pd.DataFrame()
    df = pd.concat(parts, ignore_index=True)
    df["date"] = pd.to_datetime(df["date"]).dt.date
    if start is not None:
        df = df[df["date"] >= start]
    if end is not None:
        df = df[df["date"] <= end]
    return df.reset_index(drop=True)
