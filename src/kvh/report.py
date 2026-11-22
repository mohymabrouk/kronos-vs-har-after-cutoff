"""Build results/tables.md, the figures and README.md, only from the files in results/.

README.md = scripts/README.template.md with {{PLACEHOLDERS}} replaced by tables and numbers read from results/.
"""

import json
import logging

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from kvh import config  # noqa: E402
from kvh.metrics import qlike  # noqa: E402

log = logging.getLogger(__name__)
R = config.RESULTS
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
COL = {"HAR": "#2a78d6", "Kronos-small": "#eb6834", "Kronos-mini": "#1baf7a", "Student": "#eda100",
       "LightGBM": "#e87ba4", "log-HAR": "#008300", "GARCH(1,1)": "#6250d6", "Kronos-base": "#e34948",
       "Combo: log-HAR x Kronos-small": "#8a8984"}
MAIN_ORDER = ["HAR", "log-HAR", "EWMA (RV)", "EWMA (r2)", "GARCH(1,1)", "LightGBM", "Kronos-mini", "Kronos-mini (raw)",
              "Kronos-small", "Kronos-small (raw)", "Student", "Combo: log-HAR x Kronos-small"]
PARAMS = {"HAR": "4", "log-HAR": "5", "EWMA (RV)": "0 (lambda fixed)", "EWMA (r2)": "0 (lambda fixed)", "GARCH(1,1)": "4",
          "LightGBM": "300 trees x 15 leaves", "Kronos-mini": "4.1M + 4.0M tok.", "Kronos-mini (raw)": "4.1M + 4.0M tok.",
          "Kronos-small": "24.7M + 4.0M tok.", "Kronos-small (raw)": "24.7M + 4.0M tok.",
          "Kronos-base": "102.3M + 4.0M tok.", "Kronos-base (raw)": "102.3M + 4.0M tok.",
          "Combo: log-HAR x Kronos-small": "log-HAR + Kronos-small"}


def fmt_p(p):
    if p is None or (isinstance(p, float) and np.isnan(p)):
        return ""
    return "<0.001" if p < 0.001 else f"{p:.3f}"


def fmt(x, d=3):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return ""
    return f"{x:.{d}f}"


def md_table(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    out = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for _, r in df.iterrows():
        out.append("| " + " | ".join(str(v) for v in r.values) + " |")
    return "\n".join(out)


def cpu_cost(model: str, cost: dict, student_us=None):
    b = cost.get("baselines_cpu_s_per_forecast", {})
    if model in b:
        return b[model]
    key = model.replace(" (raw)", "")
    if key in cost and isinstance(cost[key], dict):
        return cost[key]["cpu_s"]
    if model.startswith("Combo"):
        return cpu_cost("log-HAR", cost) + cpu_cost("Kronos-small", cost)
    if model == "Student" and student_us is not None:
        return student_us / 1e6
    return np.nan


def main_table(perf: pd.DataFrame, cost: dict, dist: dict) -> pd.DataFrame:
    st_us = dist["student_forward_us"] + dist["student_features_us"]
    PARAMS["Student"] = f"{dist['student_params']:,}"
    rows = []
    for m in [x for x in MAIN_ORDER if x in set(perf["model"])]:
        r = perf[perf["model"] == m].iloc[0]
        c = cpu_cost(m, cost, st_us)
        rows.append({"Model": m, "QLIKE": fmt(r["qlike"], 4), "QLIKE / HAR": fmt(r["qlike_rel_har"], 3),
                     "MSE ((%^2)^2)": fmt(r["mse"], 1), "MZ R^2 (log)": fmt(r["mz_r2"], 3),
                     "DM vs HAR (QLIKE)": fmt(r["dm_qlike"], 2), "p": fmt_p(r["p_qlike"]),
                     "DM vs HAR (MSE)": fmt(r["dm_mse"], 2), "p (MSE)": fmt_p(r["p_mse"]),
                     "CPU s / forecast": (("<4e-06 (timer floor)" if c == 0 else f"{c:.3g}") if np.isfinite(c) else ""), "Parameters": PARAMS.get(m, "")})
    return pd.DataFrame(rows)


def per_asset_table(pa: pd.DataFrame, models) -> pd.DataFrame:
    t = pa.pivot_table(index="symbol", columns="model", values="qlike")[models]
    dm = pa[pa["model"] == "Kronos-small"].set_index("symbol")[["dm_qlike", "p_qlike"]]
    out = t.map(lambda v: f"{v:.3f}")
    out["DM Kronos-small vs HAR"] = dm["dm_qlike"].map(lambda v: f"{v:.2f}")
    out["p"] = dm["p_qlike"].map(fmt_p)
    best = t.idxmin(axis=1)
    out["best"] = best
    out = out.reindex(config.SYMBOLS)
    out.insert(0, "Asset", out.index.str.replace("USDT", ""))
    return out.reset_index(drop=True)


def split_table(sp: pd.DataFrame, models) -> pd.DataFrame:
    rows = []
    for s, g in sp.groupby("split", sort=False):
        g = g.set_index("model")
        r = {"Split": s, "n": int(g["n"].iloc[0])}
        for m in models:
            r[m] = f"{g.loc[m, 'qlike']:.3f}" + ("" if m == "HAR" else f" ({g.loc[m, 'dm_qlike']:+.1f})")
        rows.append(r)
    return pd.DataFrame(rows)


# ---- figures ---------------------------------------------------------------------------------------------------------
def _style(ax):
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(INK2)
    ax.tick_params(colors=INK2, labelsize=9)


def fig_qlike(perf: pd.DataFrame):
    p = perf[~perf["model"].str.contains("raw")].sort_values("qlike")
    fig, ax = plt.subplots(figsize=(8, 4.2), dpi=140)
    colors = [COL.get(m, "#8a8984") for m in p["model"]]
    ax.barh(p["model"], p["qlike"], color=colors, height=0.6)
    har = float(perf.loc[perf["model"] == "HAR", "qlike"].iloc[0])
    ax.axvline(har, color=INK2, lw=1, ls="--")
    ax.text(har, -0.55, " HAR", color=INK2, fontsize=8, va="bottom")
    for y, (q, dm, pv) in enumerate(zip(p["qlike"], p["dm_qlike"], p["p_qlike"])):
        lab = f"{q:.3f}" + ("" if np.isnan(dm) else f"  DM {dm:+.1f}" + ("*" if pv < 0.05 else ""))
        ax.text(q, y, " " + lab, va="center", fontsize=8, color=INK)
    ax.invert_yaxis()
    ax.set_xlabel("Mean QLIKE on the test period, pooled over assets (lower is better)", color=INK2, fontsize=9)
    ax.set_title("Next-day realised variance, Aug 2025 - Oct 2026: QLIKE by model", loc="left", fontsize=11, color=INK)
    _style(ax)
    ax.grid(axis="x", color=GRID, lw=0.8)
    ax.grid(axis="y", visible=False)
    ax.set_xlim(0, p["qlike"].max() * 1.35)
    fig.tight_layout()
    fig.savefig(R / "fig_qlike_by_model.png")
    plt.close(fig)


def fig_btc(fc: pd.DataFrame):
    g = fc[fc["symbol"] == "BTCUSDT"]
    w = g.pivot_table(index="date", columns="model", values="forecast")
    rv = g.drop_duplicates("date").set_index("date")["rv"]
    fig, ax = plt.subplots(figsize=(9, 4), dpi=140)
    ax.plot(rv.index, rv * 1e4, color="#b9b8b3", lw=0.8, label="Realised variance (5-min)")
    for m in ("HAR", "Kronos-small", "Student"):
        if m in w:
            ax.plot(w.index, w[m] * 1e4, color=COL[m], lw=1.6, label=m)
    ax.set_yscale("log")
    ax.set_ylabel("Daily variance, %^2 (log scale)", color=INK2, fontsize=9)
    ax.set_title("BTCUSDT: realised variance and three forecasts, test period", loc="left", fontsize=11, color=INK)
    ax.legend(frameon=False, fontsize=8, ncol=4, loc="upper left")
    _style(ax)
    fig.tight_layout()
    fig.savefig(R / "fig_btc_forecasts.png")
    plt.close(fig)


def fig_rolling(fc: pd.DataFrame):
    w = fc.pivot_table(index=["date", "symbol"], columns="model", values="forecast")
    rv = fc.drop_duplicates(["date", "symbol"]).set_index(["date", "symbol"])["rv"].reindex(w.index)
    fig, ax = plt.subplots(figsize=(9, 4), dpi=140)
    for m in ("Kronos-small", "Kronos-mini", "Student"):
        if m not in w:
            continue
        d = pd.Series(qlike(rv.to_numpy(), w[m].to_numpy()) - qlike(rv.to_numpy(), w["HAR"].to_numpy()), index=w.index)
        d = d.groupby(level="date").mean().rolling(60, min_periods=30).mean()
        ax.plot(d.index, d.values, color=COL[m], lw=1.8, label=m)
    ax.axhline(0, color=INK2, lw=1)
    ax.set_ylabel("QLIKE(model) - QLIKE(HAR), 60-day mean", color=INK2, fontsize=9)
    ax.set_title("Below zero = better than HAR (pooled over 10 assets)", loc="left", fontsize=11, color=INK)
    ax.legend(frameon=False, fontsize=8, ncol=3, loc="upper left")
    _style(ax)
    fig.tight_layout()
    fig.savefig(R / "fig_rolling_qlike_vs_har.png")
    plt.close(fig)


def fig_cost(perf: pd.DataFrame, cost: dict, dist: dict):
    st_us = dist["student_forward_us"] + dist["student_features_us"]
    p = perf[~perf["model"].str.contains("raw")]
    fig, ax = plt.subplots(figsize=(7.5, 4.2), dpi=140)
    for _, r in p.iterrows():
        c = cpu_cost(r["model"], cost, st_us)
        if not np.isfinite(c):
            continue
        ax.scatter(c, r["qlike"], s=60, color=COL.get(r["model"], "#8a8984"), edgecolor="white", lw=1.5, zorder=3)
        ax.annotate(r["model"], (c, r["qlike"]), textcoords="offset points", xytext=(6, 4), fontsize=8, color=INK)
    ax.set_xscale("log")
    ax.set_xlabel("CPU seconds per forecast (2 threads, log scale)", color=INK2, fontsize=9)
    ax.set_ylabel("Test QLIKE (lower is better)", color=INK2, fontsize=9)
    ax.set_title("Accuracy against compute", loc="left", fontsize=11, color=INK)
    _style(ax)
    fig.tight_layout()
    fig.savefig(R / "fig_cost_vs_qlike.png")
    plt.close(fig)


def run() -> None:
    perf = pd.read_csv(R / "performance_test.csv")
    pa = pd.read_csv(R / "performance_per_asset.csv")
    sp = pd.read_csv(R / "performance_splits.csv")
    mcs = pd.read_csv(R / "mcs.csv")
    cost = json.loads((R / "cost_test.json").read_text())
    dist = json.loads((R / "distillation_test.json").read_text())
    fz = json.loads((R / "frozen_config.json").read_text())
    tlog = [json.loads(x) for x in (R / "test_log.jsonl").read_text().splitlines() if x.strip()]
    fc = pd.read_csv(R / "forecasts_test.csv", parse_dates=["date"])
    vk = pd.read_csv(R / "validation_kronos.csv")
    vh = pd.read_csv(R / "validation_har.csv")
    vs = pd.read_csv(R / "validation_student.csv")
    thr = json.loads((R / "throughput.json").read_text())
    base_path = R / "performance_base_subset.csv"

    T = {}
    T["MAIN"] = md_table(main_table(perf, cost, dist))
    models_pa = [m for m in ["HAR", "log-HAR", "GARCH(1,1)", "LightGBM", "Kronos-mini", "Kronos-small", "Student"] if m in set(pa["model"])]
    T["PER_ASSET"] = md_table(per_asset_table(pa, models_pa))
    T["SPLITS"] = md_table(split_table(sp, models_pa))
    T["MCS"] = md_table(mcs.assign(mcs_p=mcs["mcs_p"].map(lambda v: f"{v:.3f}")).rename(
        columns={"model": "Model", "mcs_p": "MCS p-value", "in_mcs_10pct": "In 90% MCS"}))
    if base_path.exists():
        b = pd.read_csv(base_path)
        keep = [m for m in ["HAR", "log-HAR", "LightGBM", "Kronos-mini", "Kronos-small", "Kronos-base", "Kronos-base (raw)", "Student"] if m in set(b["model"])]
        b = b.set_index("model").loc[keep].reset_index()
        T["BASE"] = md_table(pd.DataFrame({"Model": b["model"], "n": b["n"], "QLIKE": b["qlike"].map(lambda v: f"{v:.4f}"),
                                           "MSE": b["mse"].map(lambda v: f"{v:.1f}"), "MZ R^2": b["mz_r2"].map(lambda v: f"{v:.3f}"),
                                           "DM vs HAR": b["dm_qlike"].map(lambda v: "" if np.isnan(v) else f"{v:.2f}"),
                                           "p": b["p_qlike"].map(fmt_p)}))
    else:
        T["BASE"] = "(Kronos-base was not run on the test period.)"
    ks = perf.set_index("model")
    dist_rows = [
        ("Parameters", f"{dist['student_params']:,}", f"{dist['teacher_params']:,} (incl. tokenizer)", "4 (OLS)"),
        ("CPU time per forecast", f"{dist['student_forward_us']:.0f} us forward + {dist['student_features_us']:.0f} us features",
         f"{dist['teacher_cpu_us'] / 1e6:.2f} s (16 paths sampled, first {fz['kronos']['small']['n']} used)", f"{cost['baselines_cpu_s_per_forecast']['HAR'] * 1e6:.0f} us (refit daily)"),
        ("Speed-up vs teacher", f"{dist['teacher_cpu_us'] / (dist['student_forward_us'] + dist['student_features_us']):,.0f}x", "1x", ""),
        ("Test QLIKE", f"{ks.loc['Student', 'qlike']:.4f}", f"{ks.loc['Kronos-small', 'qlike']:.4f}", f"{ks.loc['HAR', 'qlike']:.4f}"),
        ("Test MZ R^2 (log)", f"{ks.loc['Student', 'mz_r2']:.3f}", f"{ks.loc['Kronos-small', 'mz_r2']:.3f}", f"{ks.loc['HAR', 'mz_r2']:.3f}"),
        ("DM vs HAR (QLIKE), p", f"{ks.loc['Student', 'dm_qlike']:.2f}, {fmt_p(ks.loc['Student', 'p_qlike'])}",
         f"{ks.loc['Kronos-small', 'dm_qlike']:.2f}, {fmt_p(ks.loc['Kronos-small', 'p_qlike'])}", ""),
        ("Fidelity to teacher on test: R^2 of log forecast", f"{dist['r2_log_student_vs_teacher']:.3f}", "1", ""),
        ("Fidelity: correlation of log forecasts", f"{dist['corr_log_student_teacher']:.3f}", "1", ""),
        ("Fidelity: QLIKE(teacher, student)", f"{dist['qlike_student_to_teacher']:.4f}", "0", ""),
    ]
    T["DISTILL"] = md_table(pd.DataFrame(dist_rows, columns=["", "Student (MLP)", "Teacher (Kronos-small)", "HAR"]))

    # validation summaries
    best = vk.sort_values("val_qlike_cal").groupby(["model", "tag"]).head(1).sort_values(["model", "val_qlike_cal"])
    T["VAL_KRONOS"] = md_table(pd.DataFrame({
        "Model": best["model"], "Setting": best["tag"], "Best variant": best["variant"], "N": best["n"],
        "Forecasts": best["forecasts"], "Scale c": best["scale"].map(lambda v: f"{v:.2f}"),
        "QLIKE raw": best["val_qlike_raw"].map(lambda v: f"{v:.4f}"), "QLIKE calibrated": best["val_qlike_cal"].map(lambda v: f"{v:.4f}"),
        "HAR, same days": best["val_qlike_har_same_days"].map(lambda v: f"{v:.4f}")}))
    var = vk.merge(vk.groupby(["model", "tag"])["val_qlike_cal"].min().rename("m").reset_index(), on=["model", "tag"])
    vv = vk[(vk["model"].isin(["mini", "small"]))].pivot_table(index=["model", "variant"], columns="n", values="val_qlike_cal", aggfunc="min")
    vv.columns = [f"N={c}" for c in vv.columns]
    T["VAL_VARIANT"] = md_table(vv.map(lambda v: f"{v:.4f}").reset_index())
    T["VAL_HAR"] = md_table(vh.assign(window=vh["window"].map(lambda w: "expanding" if w == 0 else f"{w} days"),
                                      val_qlike=vh["val_qlike"].map(lambda v: f"{v:.4f}")))
    T["VAL_STUDENT"] = md_table(vs.assign(**{c: vs[c].map(lambda v: f"{v:.4f}") for c in ("val_qlike_rv", "val_qlike_vs_teacher", "val_r2_log_teacher")}))
    thr_rows = pd.DataFrame([dict(Model=r["model"], N=r["n_samples"], **{"days per batch": r["batch_days"]},
                                  **{"wall s": f"{r['wall_s']:.2f}", "CPU s": f"{r['cpu_s']:.2f}"}) for r in thr["runs"]])
    T["THROUGHPUT"] = md_table(thr_rows)
    kc = {k: v for k, v in cost.items() if k.startswith("Kronos")}
    total_cpu_h = sum(v["cpu_s"] * v["forecasts"] for v in kc.values()) / 3600
    extra = R / "dm_extra.csv"
    if extra.exists():
        e = pd.read_csv(extra)
        T["EXTRA_DM"] = md_table(pd.DataFrame({"Model": e["model"], "versus": e["versus"], "n": e["n"],
                                               "Mean QLIKE difference": e["mean_qlike_diff"].map(lambda v: f"{v:+.4f}"),
                                               "DM": e["dm"].map(lambda v: f"{v:+.2f}"), "p": e["p"].map(fmt_p)}))
    else:
        T["EXTRA_DM"] = "(run scripts/extra_dm.py)"
    rob_p = R / "robustness_excl_crash.csv"
    if rob_p.exists():
        rb = pd.read_csv(rob_p)
        T["ROBUST"] = md_table(pd.DataFrame({
            "Model": rb["model"], "QLIKE, all 457 days": rb["qlike_all"].map(lambda v: f"{v:.4f}"),
            "QLIKE, without 2025-10-10": rb["qlike_excl_crash"].map(lambda v: f"{v:.4f}"),
            "DM vs HAR": rb["dm_vs_HAR"].map(lambda v: "" if np.isnan(v) else f"{v:+.2f}"), "p vs HAR": rb["p_vs_HAR"].map(fmt_p),
            "DM vs log-HAR": rb["dm_vs_log-HAR"].map(lambda v: "" if np.isnan(v) else f"{v:+.2f}"),
            "p vs log-HAR": rb["p_vs_log-HAR"].map(fmt_p)}))
        for m, key in (("Kronos-mini", "MINI"), ("Kronos-small", "SMALL"), ("Student", "STUDENT")):
            r = rb[rb["model"] == m].iloc[0]
            T[f"RDM_{key}"] = f"{r['dm_vs_HAR']:+.2f}"
    else:
        T["ROBUST"] = "(run scripts/extra_dm.py)"
    T["COST"] = md_table(pd.DataFrame([dict(Model=k, Forecasts=v["forecasts"], **{"CPU s / forecast": f"{v['cpu_s']:.2f}",
                                                                                 "Wall s / forecast": f"{v['wall_s']:.2f}",
                                                                                 "Total CPU h (test)": f"{v['cpu_s'] * v['forecasts'] / 3600:.2f}"})
                                       for k, v in kc.items()]))
    T["FROZEN"] = "```json\n" + json.dumps({"har_windows": fz["har"], "kronos": fz["kronos"], "student": fz["student"]}, indent=1) + "\n```"
    last = tlog[-1]
    T["TESTLOG"] = "\n".join(f"- run {e['run']}: {e['timestamp_utc']}, config {e['config_sha256'][:12]}, code {e['code_sha256'][:12]}, forced={e['forced']}" for e in tlog)
    n_rows = int(perf["n"].iloc[0])
    T["N_TEST"] = f"{n_rows:,}"
    T["N_DAYS"] = str(fc["date"].nunique())
    T["TOTAL_CPU_H"] = f"{total_cpu_h:.1f}"
    T["CONFIG_HASH"] = last["config_sha256"]
    for m in ("HAR", "log-HAR", "LightGBM", "Kronos-mini", "Kronos-small", "Student", "GARCH(1,1)", "Kronos-mini (raw)",
              "Kronos-small (raw)", "Combo: log-HAR x Kronos-small"):
        if m in ks.index:
            key = m.upper().replace("-", "_").replace(" ", "_").replace("(", "").replace(")", "").replace(",", "").replace(":", "")
            T[f"Q_{key}"] = f"{ks.loc[m, 'qlike']:.3f}"
            T[f"REL_{key}"] = f"{ks.loc[m, 'qlike_rel_har']:.2f}"
            T[f"DM_{key}"] = "" if np.isnan(ks.loc[m, "dm_qlike"]) else f"{ks.loc[m, 'dm_qlike']:.2f}"
            T[f"P_{key}"] = fmt_p(ks.loc[m, "p_qlike"])
            T[f"MZ_{key}"] = f"{ks.loc[m, 'mz_r2']:.3f}"

    T["STUDENT_US"] = f"{dist['student_forward_us']:.0f}"
    T["TEACHER_CPU"] = f"{dist['teacher_cpu_us'] / 1e6:.1f}"
    if extra.exists():
        r = e[(e["model"] == "Student") & (e["versus"] == "Kronos-small")].iloc[0]
        T["DM_STUDENT_TEACHER"] = f"{r['dm']:+.2f}"
        T["P_STUDENT_TEACHER"] = "p < 0.001" if r["p"] < 0.001 else f"p = {r['p']:.3f}"
    fig_qlike(perf)
    fig_btc(fc)
    fig_rolling(fc)
    fig_cost(perf, cost, dist)

    tables = ["# Tables (generated by `python -m kvh.cli report` from results/)", "",
              "## Main table (test, pooled)", T["MAIN"], "", "## Per asset (QLIKE)", T["PER_ASSET"], "",
              "## Splits (QLIKE, DM vs HAR in brackets)", T["SPLITS"], "", "## Model confidence set (QLIKE, 10%)", T["MCS"], "",
              "## Pairwise DM (post-hoc reading of the frozen test forecasts)", T["EXTRA_DM"], "",
              "## Without the 2025-10-10 crash day (post-hoc reading)", T["ROBUST"], "",
              "## Kronos-base subset", T["BASE"], "", "## Distillation", T["DISTILL"], "", "## Kronos compute on the test period", T["COST"], "",
              "## Validation: Kronos settings", T["VAL_KRONOS"], "", "## Validation: variance extraction x N", T["VAL_VARIANT"], "",
              "## Validation: HAR windows", T["VAL_HAR"], "", "## Validation: student", T["VAL_STUDENT"], "",
              "## Throughput (2 threads)", T["THROUGHPUT"], "", "## Frozen choices", T["FROZEN"], "", "## Test log", T["TESTLOG"], ""]
    (R / "tables.md").write_text("\n".join(tables), encoding="utf-8")

    tpl_path = config.ROOT / "scripts" / "README.template.md"
    if tpl_path.exists():
        txt = tpl_path.read_text(encoding="utf-8")
        for k, v in T.items():
            txt = txt.replace("{{" + k + "}}", v)
        missing = [x for x in set(__import__("re").findall(r"\{\{([A-Z_0-9]+)\}\}", txt))]
        if missing:
            log.warning("unfilled placeholders: %s", missing)
        (config.ROOT / "README.md").write_text(txt, encoding="utf-8")
    log.info("report written")
