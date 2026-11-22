"""Distil Kronos-small into a tiny MLP.

Features at 00:00 UTC of day d (data closed by then only): HAR-style daily log-RV features and the last 24 hourly
candles' log squared ranges and returns. Label: log of the teacher's raw variance forecast (frozen setting).
The student's forecast is c_small x exp(output), with the teacher's frozen validation scale c_small, so student
and teacher live on the same scale.

Hyper-parameters (width, weight decay, epochs) are chosen on the validation period by QLIKE against realised RV;
the student is then refit on all pre-test teacher forecasts (distillation-train + validation subsample).
"""

import json
import logging
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from kvh import config
from kvh.data import load_daily, load_hourly

log = logging.getLogger(__name__)
N_HOURS = 24
GRID = [dict(width=w, wd=wd, epochs=e) for w in (32, 64) for wd in (1e-4, 1e-2) for e in (100, 300)]


def daily_features(daily: pd.DataFrame) -> pd.DataFrame:
    """Row d: features known at 00:00 of day d."""
    rv = daily["rv"]
    lr = np.log(rv)
    F = pd.DataFrame(index=daily.index)
    for k in range(1, 8):
        F[f"lrv_{k}"] = lr.shift(k)
    F["lrv_w"] = np.log(rv.shift(1).rolling(7).mean())
    F["lrv_m"] = np.log(rv.shift(1).rolling(30).mean())
    F["lpk_1"] = np.log(daily["pk"].shift(1).clip(lower=1e-10))
    F["ret_1"] = daily["ret"].shift(1) * 10
    F["absret_1"] = daily["ret"].shift(1).abs() * 10
    return F


def hourly_features(hourly: pd.DataFrame, days) -> pd.DataFrame:
    """Row d: last 24 hourly candles with open time < 00:00 of d: log squared range per hour (24 values) and
    log of the sum of squared hourly returns over the last 4/12/24 hours."""
    hi, lo, cl = (hourly[c].to_numpy() for c in ("high", "low", "close"))
    t = pd.DatetimeIndex(hourly.index).as_unit("ns").asi8  # hourly grid, sorted, in ns
    rows = {}
    for d in days:
        t0 = pd.Timestamp(d, tz="UTC").value
        j = int(np.searchsorted(t, t0, side="left"))  # first candle with open time >= t0 (excluded)
        i = j - N_HOURS
        if i < 1 or t[j - 1] != t0 - 3_600_000_000_000 or t[i] != t0 - N_HOURS * 3_600_000_000_000:
            continue
        w = np.log(np.maximum(np.log(hi[i:j] / lo[i:j]) ** 2, 1e-10))
        r2 = np.diff(np.log(cl[i - 1:j])) ** 2
        v = list(w)
        v += [np.log(max(r2[-k:].sum(), 1e-12)) for k in (4, 12, 24)]
        rows[d] = v
    cols = [f"lhl_{i}" for i in range(N_HOURS)] + ["lr2_4", "lr2_12", "lr2_24"]
    return pd.DataFrame.from_dict(rows, orient="index", columns=cols)


def features(symbol: str, days, daily=None, hourly=None) -> pd.DataFrame:
    daily = load_daily(symbol) if daily is None else daily
    hourly = load_hourly(symbol) if hourly is None else hourly
    lo = min(days) - pd.Timedelta(days=40).to_pytimedelta()  # 30-day lags + slack; only past rows are used
    F = daily_features(daily[(daily.index >= lo) & (daily.index <= max(days))]).reindex(days)
    H = hourly_features(hourly, days)
    X = F.join(H, how="inner").dropna()
    X.index.name = "date"
    return X


class MLP(nn.Module):
    def __init__(self, n_in: int, width: int):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(n_in, width), nn.SiLU(), nn.Linear(width, width), nn.SiLU(), nn.Linear(width, 1))
        self.register_buffer("mu", torch.zeros(n_in))
        self.register_buffer("sd", torch.ones(n_in))

    def forward(self, x):
        return self.net((x - self.mu) / self.sd).squeeze(-1)


def n_params(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters())


def train(X: np.ndarray, y: np.ndarray, width: int, wd: float, epochs: int, seed: int = config.SEED) -> MLP:
    torch.manual_seed(seed)
    m = MLP(X.shape[1], width)
    m.mu.copy_(torch.tensor(X.mean(0), dtype=torch.float32))
    m.sd.copy_(torch.tensor(X.std(0) + 1e-6, dtype=torch.float32))
    assert n_params(m) < config.STUDENT_MAX_PARAMS
    opt = torch.optim.AdamW(m.parameters(), lr=3e-3, weight_decay=wd)
    Xt, yt = torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)
    g = torch.Generator().manual_seed(seed)
    for ep in range(epochs):
        perm = torch.randperm(len(Xt), generator=g)
        for i in range(0, len(Xt), 128):
            b = perm[i:i + 128]
            loss = ((m(Xt[b]) - yt[b]) ** 2).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
    return m.eval()


def teacher_table(kfrozen: dict, periods, teacher: str = "small") -> pd.DataFrame:
    """Teacher labels: raw Kronos-small forecast with the frozen tag/variant/N, on the given periods."""
    from kvh.kronos_run import load_forecasts
    from kvh.pipeline import kronos_raw

    fr = kfrozen[teacher]
    parts = []
    for (a, b) in periods:
        df = load_forecasts(teacher, fr["tag"], start=a, end=b)
        if len(df):
            df["raw"] = kronos_raw(df, fr["variant"], fr["n"])
            parts.append(df[["date", "symbol", "raw"]])
    return pd.concat(parts, ignore_index=True)


def build_xy(teacher: pd.DataFrame):
    X, y, keys = [], [], []
    for s, g in teacher.groupby("symbol"):
        F = features(s, sorted(g["date"]))
        g = g.set_index("date").loc[F.index]
        X.append(F.to_numpy())
        y.append(np.log(g["raw"].to_numpy()))
        keys += [(d, s) for d in F.index]
    return np.vstack(X), np.concatenate(y), keys


def predict_days(m: MLP, symbol: str, days, scale: float) -> pd.Series:
    F = features(symbol, days)
    with torch.no_grad():
        out = m(torch.tensor(F.to_numpy(), dtype=torch.float32)).numpy()
    return pd.Series(scale * np.exp(out), index=F.index)


def validate_and_fit() -> None:
    """Choose hyper-parameters on validation (QLIKE vs realised RV), refit on all pre-test teacher forecasts,
    save models/student.pt and results/validation_student.csv. Uses no test-period data."""
    from datetime import timedelta

    from kvh.metrics import qlike

    kfrozen = json.loads((config.RESULTS / "frozen_kronos.json").read_text())
    c = kfrozen["small"]["scale"]
    tr = teacher_table(kfrozen, [(config.DISTILL_TRAIN_START, config.VAL_START - timedelta(days=1))])
    va = teacher_table(kfrozen, [(config.VAL_START, config.VAL_END)])
    Xtr, ytr, _ = build_xy(tr)
    Xva, yva, kva = build_xy(va)
    log.info("student: %d training labels, %d validation labels", len(ytr), len(yva))
    val_days = [config.VAL_START + timedelta(days=k) for k in range((config.VAL_END - config.VAL_START).days + 1)]
    vfeat, vrv = [], []
    for s in config.SYMBOLS:
        F = features(s, val_days)
        vfeat.append(F.to_numpy())
        vrv.append(load_daily(s)["rv"].reindex(F.index).to_numpy())
    vfeat, vrv = torch.tensor(np.vstack(vfeat), dtype=torch.float32), np.concatenate(vrv)
    rows = []
    for hp in GRID:
        m = train(Xtr, ytr, **hp)
        with torch.no_grad():
            f = c * np.exp(m(vfeat).numpy())
        losses = [qlike(vrv, f)]
        with torch.no_grad():
            pva = m(torch.tensor(Xva, dtype=torch.float32)).numpy()
        rows.append(dict(**hp, params=n_params(m), val_qlike_rv=float(np.mean(np.concatenate(losses))),
                         val_qlike_vs_teacher=float(np.mean(qlike(np.exp(yva), np.exp(pva)))),
                         val_r2_log_teacher=float(1 - np.mean((pva - yva) ** 2) / np.var(yva))))
        log.info("student %s: %s", hp, rows[-1])
    res = pd.DataFrame(rows)
    res.to_csv(config.RESULTS / "validation_student.csv", index=False)
    best = res.sort_values("val_qlike_rv").iloc[0]
    hp = dict(width=int(best["width"]), wd=float(best["wd"]), epochs=int(best["epochs"]))
    m = train(np.vstack([Xtr, Xva]), np.concatenate([ytr, yva]), **hp)
    (config.ROOT / "models").mkdir(exist_ok=True)
    torch.save({"state": m.state_dict(), "n_in": Xtr.shape[1], "hp": hp, "scale": c}, config.ROOT / "models" / "student.pt")
    (config.RESULTS / "frozen_student.json").write_text(json.dumps(dict(hp=hp, params=n_params(m), scale=c,
                                                                       n_train=int(len(ytr) + len(yva))), indent=1))
    log.info("student frozen: %s, %d params", hp, n_params(m))


def load_student() -> tuple:
    ck = torch.load(config.ROOT / "models" / "student.pt", weights_only=False)
    m = MLP(ck["n_in"], ck["hp"]["width"])
    m.load_state_dict(ck["state"])
    return m.eval(), ck["scale"]


def time_student(m: MLP, symbol: str = "BTCUSDT", reps: int = 2000) -> dict:
    """CPU microseconds per forecast: network forward at batch 1, and feature extraction + forward."""
    from datetime import date
    F = features(symbol, [date(2025, 9, 1)])
    x = torch.tensor(F.to_numpy(), dtype=torch.float32)
    with torch.no_grad():
        for _ in range(50):
            m(x)
        t0 = time.process_time()
        for _ in range(reps):
            m(x)
        fwd = (time.process_time() - t0) / reps
    daily, hourly = load_daily(symbol), load_hourly(symbol)
    t0 = time.process_time()
    for _ in range(50):
        F = features(symbol, [date(2025, 9, 1)], daily=daily, hourly=hourly)
    feat = (time.process_time() - t0) / 50
    return {"forward_us": fwd * 1e6, "features_us": feat * 1e6}
