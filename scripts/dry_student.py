"""Dry run of the student code path with Kronos-mini validation forecasts as a stand-in teacher (code check only)."""
import json, time
from datetime import date
import numpy as np, torch
torch.set_num_threads(1)
from kvh import config, student
fz = json.loads((config.RESULTS / "frozen_kronos.json").read_text())
tr = student.teacher_table(fz, [(date(2024, 7, 1), date(2024, 12, 31))], teacher="mini")
va = student.teacher_table(fz, [(date(2025, 1, 1), date(2025, 6, 30))], teacher="mini")
Xtr, ytr, _ = student.build_xy(tr); Xva, yva, _ = student.build_xy(va)
print(Xtr.shape, Xva.shape)
t = time.time(); m = student.train(Xtr, ytr, 64, 1e-4, 100); print("train s", round(time.time() - t, 1), "params", student.n_params(m))
with torch.no_grad(): p = m(torch.tensor(Xva, dtype=torch.float32)).numpy()
print("val R2 log teacher", 1 - np.mean((p - yva) ** 2) / np.var(yva))
print(student.time_student(m))
f = student.predict_days(m, "BTCUSDT", [date(2025, 1, k) for k in range(1, 6)], 2.0); print(f)
