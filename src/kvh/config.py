"""All constants of the study in one place. Every value here is part of the config hash in the test log.

Dates are UTC calendar days. A forecast "for day d" is made at 00:00 UTC of day d, with data up to that instant.
"""

import os
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CACHE = Path(os.environ.get("P3_CACHE", Path.home() / ".cache" / "quant-research" / "p3"))
QAL_CACHE = Path.home() / ".cache" / "qal" / "binance"  # read-only, never written
RESULTS = ROOT / "results"
LOGS = ROOT / "logs"
DATA = CACHE / "data"  # parquet built from the raw zips
RAW = CACHE / "binance"  # raw zips + .CHECKSUM, same layout as data.binance.vision
KRONOS_OUT = CACHE / "kronos"  # incremental Kronos forecasts, one parquet per (model, symbol, month, tag)
HF_HOME = CACHE / "hf"

# ---- assets ---------------------------------------------------------------------------------
SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT", "DOGEUSDT", "ADAUSDT", "LTCUSDT", "LINKUSDT", "TRXUSDT"]
MAJORS = ["BTCUSDT", "ETHUSDT"]
FIRST_MONTH = (2020, 1)  # first monthly file downloaded (SOLUSDT starts 2020-08)
LAST_MONTH = (2026, 10)  # latest complete month on data.binance.vision when the study ran (Nov 2026)

# ---- Kronos data cutoff (see PLAN.md section 2 for the evidence) -----------------------------
KRONOS_PAPER_DATA_END = date(2024, 6, 30)  # paper: "The pre-training data for Kronos extends up to June 2024"
KRONOS_WEIGHTS_UPLOADED = date(2025, 7, 1)  # HF "add model" commits: small/base 2025-06-30, mini/tokenizer-2k 2025-07-01
KRONOS_DATA_END = KRONOS_WEIGHTS_UPLOADED  # conservative bound used by the protocol (mini is not in the paper)

# ---- protocol dates ---------------------------------------------------------------------------
TRAIN_START = date(2020, 3, 1)  # first target day for baseline fitting (needs 30 days of RV lags)
VAL_START = date(2024, 7, 1)  # validation: after the paper's stated data end, before the weights upload
VAL_END = date(2025, 6, 30)
TEST_START = date(2025, 8, 1)  # one-month embargo after the conservative cutoff (2025-07-01)
TEST_END = date(2026, 10, 31)
DISTILL_TRAIN_START = date(2022, 1, 1)  # teacher forecasts on 2022-01-01 .. VAL_START-1 train the student
TURBULENT_QUANTILE = 0.8  # top RV quintile (per asset, test period) = turbulent

# ---- target ---------------------------------------------------------------------------------------
DAY_START_HOUR_UTC = 0  # realised variance over [00:00, 24:00) UTC, 288 five-minute log returns
MIN_5M_BARS = 260  # a day needs >= 260 of 288 five-minute bars to count (missing bars logged)
VAR_SCALE = 1e4  # variances reported in %^2 (decimal variance x 1e4)

# ---- HAR (Corsi 2009) ---------------------------------------------------------------------------
HAR_LAGS = (1, 7, 30)  # daily, weekly, monthly; 24/7 market so calendar week/month (Corsi: 1, 5, 22 trading days)
HAR_WINDOW_GRID = (365, 730, 0)  # rolling window in days; 0 = expanding. Chosen on validation.
EWMA_LAMBDA = 0.94
GARCH_WINDOW = 1000  # days of daily returns per GARCH(1,1) fit (refit at each month start)
LGBM_PARAMS = dict(objective="regression", learning_rate=0.03, num_leaves=15, min_data_in_leaf=50, bagging_fraction=0.8,
                   bagging_freq=1, feature_fraction=0.8, lambda_l2=1.0, verbose=-1, num_threads=2, seed=20260301)
LGBM_ROUNDS = 300

# ---- Kronos ----------------------------------------------------------------------------------------
KRONOS_MODELS = {
    # name: (model repo, model revision, tokenizer repo, tokenizer revision, max_context)
    "mini": ("NeoQuasar/Kronos-mini", "f4e68697d9d5aed55cef5c96aabc3376bcad9f81",
             "NeoQuasar/Kronos-Tokenizer-2k", "26966d0035065a0cae0ebad7af8ece35bc1fb51c", 2048),
    "small": ("NeoQuasar/Kronos-small", "901c26c1332695a2a8f243eb2f37243a37bea320",
              "NeoQuasar/Kronos-Tokenizer-base", "0e0117387f39004a9016484a186a908917e22426", 512),
    "base": ("NeoQuasar/Kronos-base", "2b554741eca47781b64468546e77fef3e85130e6",
             "NeoQuasar/Kronos-Tokenizer-base", "0e0117387f39004a9016484a186a908917e22426", 512),
}
KRONOS_REPO_COMMIT = "67b630e67f6a18c9e9be918d9b4337c960db1e9a"  # github.com/shiyu-coder/Kronos master, 2026-04-13
PRED_LEN = 24  # 24 hourly candles = one UTC day
CONTEXT_LEN = 488  # 512 - 24: the whole generation fits in the model window, so no sliding (exact KV cache)
CLIP = 5.0
SEED = 20260301
THREADS = 2

# ---- distillation student -----------------------------------------------------------------------------
STUDENT_MAX_PARAMS = 100_000

# ---- newer time-series foundation models, added after the original test (PLAN.md section 13) --------------------
# Not part of the original frozen config (results/frozen_config.json was written before these lines existed); the
# extension has its own frozen file, results/frozen_tsfm.json. weights_date = commit date of the first upload of
# the weights used (HF API, checked by LFS object id); stated_data_end = latest dated training source on the model
# card (None if the card states no date).
TSFM_EMBARGO_MONTHS = 1  # as for Kronos: cutoff 2025-07-01 -> TEST_START 2025-08-01
TSFM_MODELS = {
    "ttm": dict(label="TTM-r2.1", repo="ibm-granite/granite-timeseries-ttm-r2", branch="512-96-ft-r2.1",
                revision="a23813d5e5859987e522d084f7d1d1ef9f134c3e", weights_date=date(2025, 1, 31),
                stated_data_end=None, context_h=512, context_d=512,
                data_note="no date stated; fixed public datasets (Monash/Zenodo, LOTSA subsets), incl. a Bitcoin set ending 2021",
                weights_sha256="37fa184887f260cda022d3ae9f1c642fdba579d0aefeabdf751f83f45383a761"),
    "timesfm": dict(label="TimesFM-2.5", repo="google/timesfm-2.5-200m-pytorch", branch="main",
                    revision="d418f3e8a8fa79d655b391c158f0ee8d68fe68c9", weights_date=date(2025, 9, 2),
                    stated_data_end=date(2023, 11, 30), context_h=488, context_d=1000,
                    data_note="Wikimedia pageviews to Nov 2023, Google Trends to end-2022, GiftEvalPretrain, synthetic",
                    weights_sha256="2f776efe6245e42b24bc4153ffdf61810140210e4bd3b01fb21f7aa779ab6ce8"),
    "chronos2": dict(label="Chronos-2", repo="amazon/chronos-2", branch="main",
                     revision="95a9710e2596287d08352589f42634fa5abdf0a7", weights_date=date(2025, 10, 30),
                     stated_data_end=None, context_h=488, context_d=1000,
                     data_note="no date stated; subsets of Chronos datasets and GIFT-Eval Pretrain, synthetic",
                     weights_sha256="ddcda3c7508bf2528087723e98a20707cc04b7f370ae275a9fd88078ddba4f42"),
}
TSFM_SPECS = ("H", "D")  # H: Kronos-matched hourly candles; D: daily sqrt(RV), as in arXiv 2607.05291
TSFM_H_VARIANTS = ("cc", "park")
TSFM_FLOOR = 1e-10  # raw variance floor (the data-build floor)
CRASH_DAY = date(2025, 10, 10)
