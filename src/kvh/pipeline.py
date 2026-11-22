"""validate -> freeze -> test.

validate : chooses the HAR / log-HAR windows and, per Kronos model, the sampling tag, the variance-extraction
           variant, the number of paths and the scale c, all on the validation period. Writes
           results/validation_*.csv, results/frozen_har.json and results/frozen_kronos.json.
freeze   : collects every choice (+ all config constants) into results/frozen_config.json and its SHA-256.
test     : scores every model once on the test period; appends a line to results/test_log.jsonl.
"""

import hashlib
import json
import logging
import time
from datetime import date, datetime, timedelta, timezone

import numpy as np
import pandas as pd

from kvh import config
from kvh.baselines import all_baselines, har_forecast
from kvh.data import load_daily
from kvh.metrics import diebold_mariano, model_confidence_set, mz_r2, qlike, se

log = logging.getLogger(__name__)
VARIANTS = ("cc_act", "cc_rec", "park")
N_GRID = (4, 8, 16)
DM_LAG = 7
COMBO = "Combo: log-HAR x Kronos-small"


def days_between(a: date, b: date):
    return [a + timedelta(days=k) for k in range((b - a).days + 1)]


def kronos_raw(df: pd.DataFrame, variant: str, n: int) -> np.ndarray:
    """Mean over the first n sampled paths of the per-path variance (decimal, one UTC day)."""
    return np.stack(df[variant].to_numpy())[:, :n].mean(axis=1)


def qlike_scale(rv, raw) -> float:
    """The constant c minimising sum QLIKE(rv, c * raw) is mean(rv / raw)."""
    return float(np.mean(np.asarray(rv) / np.asarray(raw)))


def attach_rv(df: pd.DataFrame, dailies: dict) -> pd.DataFrame:
    rv = pd.concat([d["rv"].rename("rv").to_frame().assign(symbol=s) for s, d in dailies.items()])
    rv["date"] = rv.index
    return df.merge(rv.reset_index(drop=True), on=["date", "symbol"], how="inner")


# ---- validate --------------------------------------------------------------------------------------------------
def validate() -> None:
    from kvh.kronos_run import TAGS, load_forecasts

    dailies = {s: load_daily(s) for s in config.SYMBOLS}
    vdays = days_between(config.VAL_START, config.VAL_END)
    rows = []
    for log_form, name in ((False, "HAR"), (True, "log-HAR")):
        for w in config.HAR_WINDOW_GRID:
            L = []
            for s, d in dailies.items():
                f = har_forecast(d, vdays, w, log_form)
                L.append(qlike(d["rv"].reindex(f.index).to_numpy(), f.to_numpy()))
            rows.append(dict(model=name, window=w, val_qlike=float(np.mean(np.concatenate(L))), n=int(sum(map(len, L)))))
            log.info("validation %s window %d: QLIKE %.4f", name, w, rows[-1]["val_qlike"])
    vh = pd.DataFrame(rows)
    vh.to_csv(config.RESULTS / "validation_har.csv", index=False)
    frozen_har = {m: int(g.sort_values("val_qlike").iloc[0]["window"]) for m, g in vh.groupby("model")}
    (config.RESULTS / "frozen_har.json").write_text(json.dumps(frozen_har, indent=1))

    krows = []
    for model in ("mini", "small", "base"):
        frames = {tag: load_forecasts(model, tag, start=config.VAL_START, end=config.VAL_END) for tag in TAGS}
        frames = {t: f for t, f in frames.items() if not f.empty}
        if not frames:
            continue
        # compare settings on the same (date, symbol) pairs
        common = set.intersection(*(set(zip(f["date"], f["symbol"])) for f in frames.values()))
        for tag, df in frames.items():
            df = df[[k in common for k in zip(df["date"], df["symbol"])]]
            df = attach_rv(df, dailies)
            # HAR on the same (date, symbol) pairs, as a yardstick
            har = []
            for s, g in df.groupby("symbol"):
                f = har_forecast(dailies[s], sorted(g["date"]), frozen_har["HAR"])
                har.append(qlike(dailies[s]["rv"].reindex(f.index).to_numpy(), f.to_numpy()))
            har_q = float(np.mean(np.concatenate(har)))
            for variant in VARIANTS:
                for n in N_GRID:
                    raw = kronos_raw(df, variant, n)
                    c = qlike_scale(df["rv"], raw)
                    krows.append(dict(model=model, tag=tag, variant=variant, n=n, forecasts=len(df), scale=c,
                                      val_qlike_raw=float(np.mean(qlike(df["rv"], raw))),
                                      val_qlike_cal=float(np.mean(qlike(df["rv"], c * raw))),
                                      val_mz_r2=mz_r2(df["rv"], raw), val_qlike_har_same_days=har_q,
                                      cpu_s=float(df["cpu_s"].mean())))
            log.info("validation %s %s: best cal QLIKE %.4f (HAR same days %.4f)", model, tag,
                     min(r["val_qlike_cal"] for r in krows if r["model"] == model and r["tag"] == tag), har_q)
    vk = pd.DataFrame(krows)
    vk.to_csv(config.RESULTS / "validation_kronos.csv", index=False)
    frozen = {}
    for model in ("mini", "small"):
        g = vk[vk["model"] == model]
        if g.empty:
            continue
        # tags are compared on the same days; pick the lowest calibrated QLIKE
        b = g.sort_values("val_qlike_cal").iloc[0]
        frozen[model] = dict(tag=b["tag"], variant=b["variant"], n=int(b["n"]), scale=float(b["scale"]),
                             val_qlike_cal=float(b["val_qlike_cal"]), val_forecasts=int(b["forecasts"]))
    g = vk[(vk["model"] == "base") & (vk["tag"] == frozen.get("small", {}).get("tag"))]
    if len(g):  # base: same tag as small (same tokenizer); variant, N and scale on its own validation subset
        b = g.sort_values("val_qlike_cal").iloc[0]
        frozen["base"] = dict(tag=b["tag"], variant=b["variant"], n=int(b["n"]), scale=float(b["scale"]),
                              val_qlike_cal=float(b["val_qlike_cal"]), val_forecasts=int(b["forecasts"]))
    (config.RESULTS / "frozen_kronos.json").write_text(json.dumps(frozen, indent=1))
    log.info("frozen Kronos: %s", frozen)
    log.info("frozen HAR windows: %s", frozen_har)


# ---- freeze ----------------------------------------------------------------------------------------------------
def _jsonable_config() -> dict:
    out = {}
    for k in dir(config):
        if k.isupper():
            v = getattr(config, k)
            if isinstance(v, (int, float, str, bool, tuple, list, dict, date)):
                out[k] = v
    return json.loads(json.dumps(out, default=str))


def code_hash() -> str:
    h = hashlib.sha256()
    for p in sorted(list((config.ROOT / "src").rglob("*.py")) + list((config.ROOT / "third_party").rglob("*.py"))):
        h.update(p.relative_to(config.ROOT).as_posix().encode())
        h.update(p.read_bytes())
    return h.hexdigest()


def freeze() -> None:
    from kvh.kronos_run import BATCH_DAYS, N_SAMPLES, TAGS

    cfg = dict(config=_jsonable_config(),
               har=json.loads((config.RESULTS / "frozen_har.json").read_text()),
               kronos=json.loads((config.RESULTS / "frozen_kronos.json").read_text()),
               student=json.loads((config.RESULTS / "frozen_student.json").read_text()),
               kronos_tags=TAGS, kronos_n_samples=N_SAMPLES, kronos_batch_days=BATCH_DAYS)
    text = json.dumps(cfg, indent=1, sort_keys=True)
    digest = hashlib.sha256(text.encode()).hexdigest()
    (config.RESULTS / "frozen_config.json").write_text(text)
    (config.RESULTS / "frozen_config.sha256").write_text(digest + "\n")
    log.info("frozen config written, sha256 %s", digest)


def load_frozen() -> tuple:
    text = (config.RESULTS / "frozen_config.json").read_text()
    digest = hashlib.sha256(text.encode()).hexdigest()
    stored = (config.RESULTS / "frozen_config.sha256").read_text().strip()
    if digest != stored:
        raise RuntimeError("frozen_config.json was modified after freeze (hash mismatch)")
    return json.loads(text), digest


# ---- test ------------------------------------------------------------------------------------------------------
def collect_test_forecasts(cfg: dict, dailies: dict) -> tuple:
    from kvh.kronos_run import load_forecasts
    from kvh import student

    tdays = days_between(config.TEST_START, config.TEST_END)
    cost = {}
    base = all_baselines(dailies, tdays, cfg["har"])
    cost["baselines_cpu_s_per_forecast"] = all_baselines.cpu_s_per_forecast
    parts = [base]
    for model, label in (("mini", "Kronos-mini"), ("small", "Kronos-small"), ("base", "Kronos-base")):
        fr = cfg["kronos"].get(model)
        if fr is None:
            continue
        df = load_forecasts(model, fr["tag"], start=config.TEST_START, end=config.TEST_END)
        if df.empty:
            log.warning("no test forecasts for %s", model)
            continue
        raw = kronos_raw(df, fr["variant"], fr["n"])
        cost[label] = dict(cpu_s=float(df["cpu_s"].mean()), wall_s=float(df["wall_s"].mean()), forecasts=len(df))
        parts.append(pd.DataFrame({"date": df["date"], "symbol": df["symbol"], "model": label, "forecast": fr["scale"] * raw}))
        parts.append(pd.DataFrame({"date": df["date"], "symbol": df["symbol"], "model": label + " (raw)", "forecast": raw}))
    m, c = student.load_student()
    for s in config.SYMBOLS:
        f = student.predict_days(m, s, tdays, c)
        parts.append(pd.DataFrame({"date": list(f.index), "symbol": s, "model": "Student", "forecast": f.to_numpy()}))
    cost["student"] = student.time_student(m)
    cost["student_params"] = student.n_params(m)
    return pd.concat(parts, ignore_index=True), cost


def dm_pooled(wide_loss: pd.DataFrame, model: str, ref: str = "HAR"):
    """Pooled DM: daily cross-sectional mean of the loss differential, NW lag DM_LAG."""
    d = (wide_loss[model] - wide_loss[ref]).groupby(level="date").mean()
    return diebold_mariano(d.to_numpy(), np.zeros(len(d)), DM_LAG)


def score(long: pd.DataFrame, models) -> pd.DataFrame:
    """long: date, symbol, model, forecast, rv (common sample). One row per model."""
    rows = []
    w = long.pivot_table(index=["date", "symbol"], columns="model", values="forecast").dropna(subset=models)
    rv = long.drop_duplicates(["date", "symbol"]).set_index(["date", "symbol"])["rv"].reindex(w.index)
    ql = pd.DataFrame({m: qlike(rv.to_numpy(), w[m].to_numpy()) for m in models}, index=w.index)
    sq = pd.DataFrame({m: se(rv.to_numpy(), w[m].to_numpy()) * config.VAR_SCALE ** 2 for m in models}, index=w.index)
    for m in models:
        st, p, _ = dm_pooled(ql, m) if m != "HAR" else (np.nan, np.nan, 0)
        st2, p2, _ = dm_pooled(sq, m) if m != "HAR" else (np.nan, np.nan, 0)
        rows.append(dict(model=m, n=len(w), qlike=float(ql[m].mean()), mse=float(sq[m].mean()),
                         mz_r2=mz_r2(rv.to_numpy(), w[m].to_numpy()), dm_qlike=st, p_qlike=p, dm_mse=st2, p_mse=p2,
                         qlike_rel_har=float(ql[m].mean() / ql["HAR"].mean())))
    return pd.DataFrame(rows), ql


def test(force: bool = False) -> None:
    cfg, digest = load_frozen()
    log_path = config.RESULTS / "test_log.jsonl"
    prior = log_path.read_text().strip().splitlines() if log_path.exists() else []
    if prior and not force:
        raise SystemExit(f"test already ran {len(prior)} time(s); rerun only with --force (it is logged)")
    t_start = time.time()
    dailies = {s: load_daily(s) for s in config.SYMBOLS}
    long, cost = collect_test_forecasts(cfg, dailies)
    # pre-committed, untuned combination: geometric mean of log-HAR and calibrated Kronos-small forecasts
    w = long.pivot_table(index=["date", "symbol"], columns="model", values="forecast")[["log-HAR", "Kronos-small"]].dropna()
    combo = np.sqrt(w["log-HAR"] * w["Kronos-small"]).rename("forecast").reset_index().assign(model=COMBO)
    long = pd.concat([long, combo[["date", "symbol", "model", "forecast"]]], ignore_index=True)
    long = attach_rv(long, dailies)
    long.to_csv(config.RESULTS / "forecasts_test.csv", index=False)

    main = ["HAR", "log-HAR", "EWMA (RV)", "EWMA (r2)", "GARCH(1,1)", "LightGBM", "Kronos-mini", "Kronos-mini (raw)",
            "Kronos-small", "Kronos-small (raw)", "Student", COMBO]
    main = [m for m in main if m in set(long["model"])]
    perf, ql = score(long, main)
    perf.to_csv(config.RESULTS / "performance_test.csv", index=False)
    log.info("\n%s", perf.to_string())

    # per asset
    rows = []
    for s, g in long.groupby("symbol"):
        p, _ = score(g, main)
        p.insert(0, "symbol", s)
        rows.append(p)
    pd.concat(rows).to_csv(config.RESULTS / "performance_per_asset.csv", index=False)

    # splits: turbulent = top RV quintile per asset in the test period; majors vs alts
    q80 = long.drop_duplicates(["date", "symbol"]).groupby("symbol")["rv"].quantile(config.TURBULENT_QUANTILE)
    long["turbulent"] = long["rv"] > long["symbol"].map(q80)
    long["group"] = np.where(long["symbol"].isin(config.MAJORS), "BTC/ETH", "alts")
    rows = []
    for name, mask in (("calm", ~long["turbulent"]), ("turbulent", long["turbulent"]),
                       ("BTC/ETH", long["group"] == "BTC/ETH"), ("alts", long["group"] == "alts")):
        p, _ = score(long[mask], main)
        p.insert(0, "split", name)
        rows.append(p)
    pd.concat(rows).to_csv(config.RESULTS / "performance_splits.csv", index=False)

    # Kronos-base subset: same (date, symbol) pairs for every model
    if "Kronos-base" in set(long["model"]):
        sub = long[long["model"] == "Kronos-base"][["date", "symbol"]]
        lb = long.merge(sub, on=["date", "symbol"])
        p, _ = score(lb, main + ["Kronos-base", "Kronos-base (raw)"])
        p.to_csv(config.RESULTS / "performance_base_subset.csv", index=False)

    # Model Confidence Set on the daily cross-sectional mean QLIKE (calibrated models only, no raw variants)
    mcs_models = [m for m in main if "(raw)" not in m]
    daily_q = ql[mcs_models].groupby(level="date").mean()
    pv = model_confidence_set(daily_q.to_numpy(), mcs_models, alpha=0.10, block=7, reps=2000, seed=config.SEED)
    pd.DataFrame({"model": list(pv), "mcs_p": list(pv.values())}).assign(in_mcs_10pct=lambda x: x["mcs_p"] >= 0.10) \
        .to_csv(config.RESULTS / "mcs.csv", index=False)

    # distillation table
    w = long.pivot_table(index=["date", "symbol"], columns="model", values="forecast").dropna(subset=["Kronos-small", "Student"])
    lt, ls = np.log(w["Kronos-small"]), np.log(w["Student"])
    dist = dict(student_params=cost["student_params"], teacher_params=24741376 + 3958042,
                student_forward_us=cost["student"]["forward_us"], student_features_us=cost["student"]["features_us"],
                teacher_cpu_us=cost["Kronos-small"]["cpu_s"] * 1e6,
                r2_log_student_vs_teacher=float(1 - np.mean((ls - lt) ** 2) / np.var(lt)),
                corr_log_student_teacher=float(np.corrcoef(ls, lt)[0, 1]),
                qlike_student_to_teacher=float(np.mean(qlike(w["Kronos-small"], w["Student"]))))
    (config.RESULTS / "distillation_test.json").write_text(json.dumps(dist, indent=1))
    n_kron = {k: v for k, v in cost.items() if k.startswith("Kronos")}
    (config.RESULTS / "cost_test.json").write_text(json.dumps(cost, indent=1, default=float))

    entry = dict(timestamp_utc=datetime.now(timezone.utc).isoformat(timespec="seconds"), config_sha256=digest,
                 code_sha256=code_hash(), forced=bool(force), run=len(prior) + 1, n_forecast_rows=int(len(long)),
                 kronos=n_kron, seconds=round(time.time() - t_start, 1),
                 headline={r["model"]: round(r["qlike"], 4) for r in perf.to_dict("records")})
    with open(log_path, "a") as f:
        f.write(json.dumps(entry) + "\n")
    log.info("test logged: %s", entry)
