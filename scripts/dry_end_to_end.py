"""Dry run of test + report on the VALIDATION period, with Kronos-mini standing in for every Kronos model and a
stand-in student. Writes only to a scratch directory (argv[1]); code check only, never part of the results."""
import json, shutil, sys
from datetime import date
from pathlib import Path
import numpy as np, torch
torch.set_num_threads(1)
from kvh import config
scratch = Path(sys.argv[1]); scratch.mkdir(parents=True, exist_ok=True)
real = config.RESULTS
config.RESULTS = scratch
config.TEST_START, config.TEST_END = date(2024, 7, 1), date(2025, 6, 30)
for f in ("frozen_har.json", "validation_kronos.csv", "validation_har.csv", "throughput.json"):
    shutil.copy(real / f, scratch / f)
fk = json.loads((real / "frozen_kronos.json").read_text())
fk["small"] = dict(fk["mini"]); fk["base"] = dict(fk["mini"])
(scratch / "frozen_kronos.json").write_text(json.dumps(fk))
import kvh.kronos_run as kr
_orig = kr.load_forecasts
def fake(model, tag, symbols=None, start=None, end=None):
    df = _orig("mini", tag, symbols, start, end)
    if model == "base":
        df = df[df.symbol.isin(["BTCUSDT", "ETHUSDT"])]
    return df.assign(model=model)
kr.load_forecasts = fake
from kvh import student, pipeline, report
tr = student.teacher_table(fk, [(date(2024, 7, 1), date(2025, 6, 30))], teacher="small")
X, y, _ = student.build_xy(tr)
m = student.train(X, y, 32, 1e-4, 50)
(config.ROOT / "models").mkdir(exist_ok=True)
real_pt = config.ROOT / "models" / "student.pt"
assert not real_pt.exists()
torch.save({"state": m.state_dict(), "n_in": X.shape[1], "hp": dict(width=32, wd=1e-4, epochs=50), "scale": fk["small"]["scale"]}, real_pt)
import pandas as pd
pd.DataFrame([dict(width=32, wd=1e-4, epochs=50, params=student.n_params(m), val_qlike_rv=0.3, val_qlike_vs_teacher=0.1, val_r2_log_teacher=0.5)]).to_csv(scratch / "validation_student.csv", index=False)
(scratch / "frozen_student.json").write_text(json.dumps(dict(hp=dict(width=32), params=student.n_params(m), scale=1.0, n_train=len(y))))
try:
    pipeline.freeze()
    pipeline.test()
    report.run.__globals__["R"] = scratch
    orig_root = config.ROOT
    report.run()
finally:
    real_pt.unlink()
print(open(scratch / "tables.md", encoding="utf-8").read()[:3000])
