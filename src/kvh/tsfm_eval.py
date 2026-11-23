"""Validation and one-shot test scoring of the three newer foundation models (PLAN.md section 13).

validate : per model and input specification, the scale c = mean(RV / raw) on the validation period and (H only)
           the hourly variant with the lowest calibrated QLIKE. Writes results/validation_tsfm.csv and
           results/frozen_tsfm.json + .sha256 (pins, dates, scoring starts, choices, library versions).
test     : scores once, on each new model's own clean window, every row on the same asset-days; recomputes every
           existing model's test forecasts and compares them with results/forecasts_test.csv (reproduction check);
           appends a forced/added line to results/test_log.jsonl.
"""

import hashlib
import json
import logging
import time
from datetime import datetime, timezone
from importlib.metadata import version

import numpy as np
import pandas as pd

from kvh import config, pipeline, tsfm
from kvh.baselines import har_forecast
from kvh.data import load_daily
from kvh.metrics import diebold_mariano, model_confidence_set, mz_r2, qlike, se

log = logging.getLogger(__name__)
R = config.RESULTS
REFS = ["HAR", "log-HAR", "LightGBM", "Kronos-mini", "Kronos-small", "Student"]
DM_LAG = pipeline.DM_LAG
ADDED = "TSFM extension (PLAN.md section 13)"


def variants(spec: str):
    return list(config.TSFM_H_VARIANTS) if spec == "H" else ["sigma"]


# ---- validate -----------------------------------------------------------------------------------------------------
def validate() -> None:
    dailies = {s: load_daily(s) for s in config.SYMBOLS}
    rows = []
    for name in config.TSFM_MODELS:
        for spec in config.TSFM_SPECS:
            for variant in variants(spec):
                df = tsfm.load(name, spec, variant, config.VAL_START, config.VAL_END)
                if df.empty:
                    raise SystemExit(f"no validation forecasts for {name} {spec} {variant}")
                df = pipeline.attach_rv(df, dailies)
                c = pipeline.qlike_scale(df["rv"], df["raw"])
                rows.append(dict(model=name, spec=spec, variant=variant, forecasts=len(df), scale=c,
                                 val_qlike_raw=float(np.mean(qlike(df["rv"], df["raw"]))),
                                 val_qlike_cal=float(np.mean(qlike(df["rv"], c * df["raw"]))),
                                 val_mz_r2=mz_r2(df["rv"], df["raw"]), floored=int(df["floored"].sum()),
                                 cpu_s=float(df["cpu_s"].mean()), wall_s=float(df["wall_s"].mean())))
                log.info("validation %s %s %s: c %.3f, QLIKE raw %.4f, calibrated %.4f (%d forecasts)", name, spec, variant,
                         c, rows[-1]["val_qlike_raw"], rows[-1]["val_qlike_cal"], len(df))
    v = pd.DataFrame(rows)
    # yardsticks on the same validation asset-days (every day, all assets)
    fh = json.loads((R / "frozen_har.json").read_text())
    days = pipeline.days_between(config.VAL_START, config.VAL_END)
    for m, w, lf in (("HAR", fh["HAR"], False), ("log-HAR", fh["log-HAR"], True)):
        L = []
        for s, d in dailies.items():
            f = har_forecast(d, days, w, lf)
            L.append(qlike(d["rv"].reindex(f.index).to_numpy(), f.to_numpy()))
        q = np.concatenate(L)
        v = pd.concat([v, pd.DataFrame([dict(model=m, spec="", variant="", forecasts=len(q), val_qlike_cal=float(q.mean()))])],
                      ignore_index=True)
    v.to_csv(R / "validation_tsfm.csv", index=False)

    models = {}
    for name, m in config.TSFM_MODELS.items():
        entry = {k: (str(x) if hasattr(x, "isoformat") else x) for k, x in m.items()}
        entry.update(effective_cutoff=str(tsfm.effective_cutoff(name)), scoring_start=str(tsfm.scoring_start(name)))
        for spec in config.TSFM_SPECS:
            g = v[(v["model"] == name) & (v["spec"] == spec)].sort_values("val_qlike_cal")
            b = g.iloc[0]  # H: variants are compared on the same asset-days (every validation day, all assets)
            entry[spec] = dict(variant=b["variant"], scale=float(b["scale"]), val_qlike_cal=float(b["val_qlike_cal"]),
                               val_forecasts=int(b["forecasts"]))
        models[name] = entry
    frozen = dict(plan="PLAN.md section 13", models=models, specs=list(config.TSFM_SPECS),
                  embargo_months=config.TSFM_EMBARGO_MONTHS, floor=config.TSFM_FLOOR, crash_day=str(config.CRASH_DAY),
                  point_forecast={"ttm": "model output (MSE-trained)", "timesfm": "mean channel (index 0)",
                                  "chronos2": "mean of the 0.1..0.9 quantiles"},
                  average="((sqrt(F_TTM) + sqrt(F_log-HAR)) / 2)^2, F_TTM calibrated", references=REFS,
                  original_frozen_config_sha256=(R / "frozen_config.sha256").read_text().strip(),
                  versions={p: version(p) for p in ("torch", "numpy", "transformers", "timesfm", "chronos-forecasting", "granite-tsfm")})
    text = json.dumps(frozen, indent=1, sort_keys=True)
    tsfm.frozen_tsfm_path().write_text(text)
    (R / "frozen_tsfm.sha256").write_text(hashlib.sha256(text.encode()).hexdigest() + "\n")
    log.info("frozen TSFM choices: %s", {n: {s: models[n][s] for s in config.TSFM_SPECS} for n in models})


# ---- scoring helpers ------------------------------------------------------------------------------------------------
def dm_daily(d: pd.Series):
    """d: loss differential indexed by (date, symbol) -> pooled DM on the daily cross-sectional mean."""
    x = d.groupby(level="date").mean().to_numpy()
    return diebold_mariano(x, np.zeros_like(x), DM_LAG)


def reproduction_check(long: pd.DataFrame) -> dict:
    """Recomputed forecasts of the existing models against results/forecasts_test.csv and performance_test.csv."""
    old = pd.read_csv(R / "forecasts_test.csv")
    new = long[["date", "symbol", "model", "forecast", "rv"]].copy()
    new["date"] = new["date"].astype(str)
    old["date"] = old["date"].astype(str)
    m = old.merge(new, on=["date", "symbol", "model"], how="outer", suffixes=("_old", "_new"), indicator=True)
    both = m[m["_merge"] == "both"]
    rel = (both["forecast_new"] - both["forecast_old"]).abs() / both["forecast_old"].abs()
    per_model = rel.groupby(both["model"]).max()
    perf_old = pd.read_csv(R / "performance_test.csv").set_index("model")
    main = [x for x in perf_old.index]
    perf_new, _ = pipeline.score(long, main)
    perf_new = perf_new.set_index("model")
    cols = ["qlike", "mse", "mz_r2", "dm_qlike", "p_qlike", "dm_mse", "p_mse"]
    perf_diff = float(np.nanmax(np.abs(perf_new.loc[main, cols].to_numpy() - perf_old.loc[main, cols].to_numpy())))
    out = dict(rows_old=int(len(old)), rows_new=int(len(new)), rows_matched=int(len(both)),
               only_old=int((m["_merge"] == "left_only").sum()), only_new=int((m["_merge"] == "right_only").sum()),
               max_rel_diff_forecast=float(rel.max()), max_rel_diff_by_model={k: float(v) for k, v in per_model.items()},
               rv_identical=bool((both["rv_old"] == both["rv_new"]).all()),
               max_abs_diff_performance_test=perf_diff,
               headline_qlike_old={k: round(float(v), 4) for k, v in perf_old["qlike"].items()},
               headline_qlike_new={k: round(float(v), 4) for k, v in perf_new.loc[main, "qlike"].items()})
    out["unchanged"] = bool(out["only_old"] == 0 and out["only_new"] == 0 and out["max_rel_diff_forecast"] < 1e-9
                            and out["rv_identical"] and perf_diff < 1e-9)
    return out


def window_rows(win: str, frozen: dict) -> tuple:
    """Rows scored on the window of model `win`: references, every new model clean on the whole window (calibrated
    and raw, both specifications) and the two TTM + log-HAR averages."""
    start = tsfm.scoring_start(win)
    new = []
    for name in config.TSFM_MODELS:
        if tsfm.scoring_start(name) <= start:
            for spec in config.TSFM_SPECS:
                new += [tsfm.label(name, spec), tsfm.label(name, spec) + " (raw)"]
    avgs = [tsfm.avg_label(s) for s in config.TSFM_SPECS]
    return start, new, avgs


def score_window(allf: pd.DataFrame, win: str, frozen: dict) -> dict:
    start, new, avgs = window_rows(win, frozen)
    rows = REFS + new + avgs
    cal = [m for m in rows if "(raw)" not in m]
    sub = allf[allf["date"] >= start]
    w = sub.pivot_table(index=["date", "symbol"], columns="model", values="forecast")[rows].dropna()
    rv = sub.drop_duplicates(["date", "symbol"]).set_index(["date", "symbol"])["rv"].reindex(w.index)
    ql = pd.DataFrame({m: qlike(rv.to_numpy(), w[m].to_numpy()) for m in rows}, index=w.index)
    sq = pd.DataFrame({m: se(rv.to_numpy(), w[m].to_numpy()) * config.VAR_SCALE ** 2 for m in rows}, index=w.index)
    dates = w.index.get_level_values("date")
    has_crash = config.CRASH_DAY in set(dates)
    keep = dates != config.CRASH_DAY
    n_days = int(pd.Index(dates).nunique())
    kind = lambda m: "reference" if m in REFS else ("average" if m in avgs else ("raw" if "(raw)" in m else "new"))  # noqa: E731

    perf = []
    for m in rows:
        perf.append(dict(window=win, window_start=str(start), model=m, kind=kind(m), n=len(w), n_days=n_days,
                         qlike=float(ql[m].mean()), qlike_excl_crash=float(ql.loc[keep, m].mean()) if has_crash else np.nan,
                         mse=float(sq[m].mean()), mz_r2=mz_r2(rv.to_numpy(), w[m].to_numpy()),
                         qlike_rel_log_har=float(ql[m].mean() / ql["log-HAR"].mean())))
    dm = []
    ttm_of = {tsfm.avg_label(s): tsfm.label("ttm", s) for s in config.TSFM_SPECS}
    for m in new + avgs:
        versus = REFS + ([ttm_of[m]] if m in ttm_of else [])
        for ref in versus:
            for loss_name, L in (("QLIKE", ql), ("MSE", sq)):
                samples = [("all", slice(None))] + ([("excl_crash", keep)] if has_crash else [])
                for sname, mask in samples:
                    d = (L[m] - L[ref])[mask]
                    st, p, md = dm_daily(d)
                    dm.append(dict(window=win, model=m, versus=ref, loss=loss_name, sample=sname,
                                   n_days=int(d.index.get_level_values("date").nunique()), mean_diff=md, dm=st, p=p))
    pa = []
    for s, g in ql.groupby(level="symbol"):
        for m in rows:
            r = dict(window=win, symbol=s, model=m, n=len(g), qlike=float(g[m].mean()))
            for ref in ("HAR", "log-HAR"):
                if m == ref or m in REFS:
                    r[f"dm_vs_{ref}"], r[f"p_vs_{ref}"] = np.nan, np.nan
                    continue
                st, p, _ = diebold_mariano(g[m].to_numpy(), g[ref].to_numpy(), DM_LAG)
                r[f"dm_vs_{ref}"], r[f"p_vs_{ref}"] = st, p
            pa.append(r)
    mcs = []
    daily_q = ql[cal].groupby(level="date").mean()
    pv = model_confidence_set(daily_q.to_numpy(), cal, alpha=0.10, block=7, reps=2000, seed=config.SEED)
    mcs += [dict(window=win, scope="pooled", model=k, mcs_p=v, in_mcs_10pct=v >= 0.10) for k, v in pv.items()]
    for s, g in ql[cal].groupby(level="symbol"):
        pv = model_confidence_set(g.to_numpy(), cal, alpha=0.10, block=7, reps=2000, seed=config.SEED)
        mcs += [dict(window=win, scope=s, model=k, mcs_p=v, in_mcs_10pct=v >= 0.10) for k, v in pv.items()]
    return dict(perf=perf, dm=dm, per_asset=pa, mcs=mcs, n=len(w), n_days=n_days)


# ---- test (one run) -------------------------------------------------------------------------------------------------
def test(force: bool = False) -> None:
    cfg, digest = pipeline.load_frozen()
    frozen, tdigest = tsfm.load_frozen_tsfm()
    log_path = R / "test_log.jsonl"
    prior = [json.loads(x) for x in log_path.read_text().splitlines() if x.strip()]
    if any(e.get("added") == ADDED for e in prior) and not force:
        raise SystemExit("the TSFM extension was already scored; rerun only with --force (it is logged)")
    t_start = time.time()
    dailies = {s: load_daily(s) for s in config.SYMBOLS}

    # 1. every existing model, recomputed from scratch on the full window (reproduction check)
    long, _ = pipeline.collect_test_forecasts(cfg, dailies)
    w = long.pivot_table(index=["date", "symbol"], columns="model", values="forecast")[["log-HAR", "Kronos-small"]].dropna()
    combo = np.sqrt(w["log-HAR"] * w["Kronos-small"]).rename("forecast").reset_index().assign(model=pipeline.COMBO)
    long = pd.concat([long, combo[["date", "symbol", "model", "forecast"]]], ignore_index=True)
    long = pipeline.attach_rv(long, dailies)
    repro = reproduction_check(long)
    (R / "tsfm_reproduction.json").write_text(json.dumps(repro, indent=1))
    log.info("reproduction check: unchanged=%s, max relative forecast difference %.3g", repro["unchanged"], repro["max_rel_diff_forecast"])

    # 2. the new forecasts (frozen scale and variant), from each model's scoring start
    parts, cost = [], {}
    for name in config.TSFM_MODELS:
        start = tsfm.scoring_start(name)
        expected = len(pipeline.days_between(start, config.TEST_END)) * len(config.SYMBOLS)
        for spec in config.TSFM_SPECS:
            fz = frozen["models"][name][spec]
            df = tsfm.load(name, spec, fz["variant"], start=start, end=config.TEST_END)
            if len(df) != expected or df["date"].min() < start:
                raise SystemExit(f"{name} {spec}: {len(df)} test forecasts, expected {expected} from {start}")
            lab = tsfm.label(name, spec)
            parts.append(pd.DataFrame({"date": df["date"], "symbol": df["symbol"], "model": lab, "forecast": fz["scale"] * df["raw"]}))
            parts.append(pd.DataFrame({"date": df["date"], "symbol": df["symbol"], "model": lab + " (raw)", "forecast": df["raw"]}))
            cost[lab] = dict(cpu_s=float(df["cpu_s"].mean()), wall_s=float(df["wall_s"].mean()), forecasts=len(df),
                             floored=int(df["floored"].sum()), negative_points=int(df["n_neg"].sum()))
    newf = pd.concat(parts, ignore_index=True)
    ref = long[long["model"].isin(REFS)][["date", "symbol", "model", "forecast"]]
    both = pd.concat([ref, newf], ignore_index=True)
    wide = both.pivot_table(index=["date", "symbol"], columns="model", values="forecast")
    for spec in config.TSFM_SPECS:
        ttm = tsfm.label("ttm", spec)
        a = wide[[ttm, "log-HAR"]].dropna()
        avg = pd.DataFrame({"date": a.index.get_level_values("date"), "symbol": a.index.get_level_values("symbol"),
                            "model": tsfm.avg_label(spec), "forecast": tsfm.vol_average(a[ttm].to_numpy(), a["log-HAR"].to_numpy())})
        newf = pd.concat([newf, avg], ignore_index=True)
    newf = pipeline.attach_rv(newf, dailies)
    newf.sort_values(["model", "symbol", "date"]).to_csv(R / "tsfm_forecasts_test.csv", index=False)
    allf = pd.concat([long[long["model"].isin(REFS)], newf], ignore_index=True)

    # 3. score every window
    out = {k: [] for k in ("perf", "dm", "per_asset", "mcs")}
    head = {}
    for win in config.TSFM_MODELS:
        res = score_window(allf, win, frozen)
        for k in out:
            out[k] += res[k]
        head[win] = {r["model"]: round(r["qlike"], 4) for r in res["perf"] if r["kind"] != "raw"}
        log.info("window %s from %s: %d asset-days, %d days", win, tsfm.scoring_start(win), res["n"], res["n_days"])
    pd.DataFrame(out["perf"]).to_csv(R / "tsfm_performance.csv", index=False)
    pd.DataFrame(out["dm"]).to_csv(R / "tsfm_dm.csv", index=False)
    pd.DataFrame(out["per_asset"]).to_csv(R / "tsfm_per_asset.csv", index=False)
    pd.DataFrame(out["mcs"]).to_csv(R / "tsfm_mcs.csv", index=False)
    params = json.loads((R / "tsfm_params.json").read_text())
    (R / "tsfm_cost.json").write_text(json.dumps(dict(params=params, test=cost), indent=1))

    entry = dict(timestamp_utc=datetime.now(timezone.utc).isoformat(timespec="seconds"), config_sha256=digest,
                 tsfm_config_sha256=tdigest, code_sha256=pipeline.code_hash(), forced=True, added=ADDED, run=len(prior) + 1,
                 n_forecast_rows=int(len(newf)), reproduction_unchanged=repro["unchanged"],
                 seconds=round(time.time() - t_start, 1), headline=head)
    with open(log_path, "a") as f:
        f.write(json.dumps(entry) + "\n")
    log.info("TSFM test logged: %s", entry)
