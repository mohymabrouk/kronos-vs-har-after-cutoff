# PLAN: does Kronos beat HAR-RV at next-day realised variance after its data cutoff, and can a tiny student keep the edge?

Written before the modelling code (only the downloader and the Kronos inference wrapper existed when it was
drafted; the throughput section was filled in from the measurement in step 3). Things that change during the
build are logged at the end under "Changes during the build".

## 1. Question

Kronos (Shi et al., arXiv 2508.02739, AAAI 2026) is a decoder-only model pretrained on 12B candlesticks from
45 exchanges. The paper reports a 9% lower MAE than the best baseline for volatility forecasting. We ask, on
crypto data that Kronos cannot have seen:

1. Does a Kronos variance forecast (from sampled 24-hour candle paths) beat HAR-RV (Corsi 2009) at the
   next-day realised variance, measured by QLIKE (primary), MSE, Mincer-Zarnowitz R^2 and Diebold-Mariano?
2. Can a student network under 100k parameters, trained only on pre-test Kronos-small forecasts, keep whatever
   Kronos achieves, at a fraction of its CPU cost?

## 2. What Kronos is, and its data end date (evidence)

Read from the paper (arXiv HTML v1, the only version, 2 Aug 2025) and the repo (master 67b630e, 2026-04-13).

* **Pretraining data.** "over 12 billion K-line records from 45 global exchanges" (abstract); appendix: 7
  frequencies from 1-minute to weekly, stocks from 9 exchanges, **Binance spot crypto pairs**, 1,000+ forex
  pairs; per-exchange start dates back to 1990. **End date: "The pre-training data for Kronos extends up to June
  2024"**, and the paper's test data "begins in July 2024".
* **Weights.** HuggingFace commit history (API `/api/models/NeoQuasar/<m>/commits/main`):
  Kronos-small "add model" 2025-06-30 16:44 UTC, Kronos-base 2025-06-30 17:20, Kronos-Tokenizer-base
  2025-06-30 17:04, **Kronos-mini 2025-07-01 13:29**, Kronos-Tokenizer-2k 2025-07-01 08:53. All later commits
  only edit README.md (latest 2025-09-09), so the weights date from 2025-06-30/07-01.
* **Kronos-mini is not in the paper** (the paper has small/base/large), so its data end is undocumented. We
  therefore use the **conservative bound 2025-07-01** (weights upload) for all models, and start the test
  period on **2025-08-01** (one-month embargo). `config.KRONOS_DATA_END` / `config.TEST_START`; a pytest asserts
  `TEST_START > KRONOS_DATA_END`.
* **Tokenizer.** Transformer autoencoder (causal, RoPE) over 6 features per bar (OHLC, volume, amount), each
  z-scored over the context window and clipped to +-5. Binary Spherical Quantization with k = 20 bits, split into
  a coarse s1 and a fine s2 sub-token (10 bits each, vocab 1,024 each). Kronos-small/base use
  Kronos-Tokenizer-base, Kronos-mini uses Kronos-Tokenizer-2k. Each bar also carries minute/hour/weekday/day/
  month embeddings.
* **Generation.** Autoregressive: at each step sample s1 from the head, then s2 from a dependency-aware
  cross-attention layer conditioned on s1; temperature T and nucleus top-p apply to both. `predict()` defaults:
  T = 1.0, top_p = 0.9, sample_count = 1; the predictor **averages** the decoded sample paths.
* **Paper settings (Table 6).** Price/return forecasting T = 0.6, top-p 0.9, N = 10; **realised volatility
  T = 0.9, top-p 0.9, N = 1**; synthetic data T = 1.0, top-p 0.95.
* **How the paper evaluates volatility.** RV is computed from the model's predicted closes over the horizon as
  the sum of squared log returns; metrics MAE and R^2; baselines ARCH/GARCH (orders by BIC) and other
  time-series models; reported per exchange (including a "Crypto" row). No QLIKE, no HAR, no DM test.

## 3. Throughput first (2 threads), and the size of the run

Measured with `python -m kvh.cli throughput` (results/throughput.json). See the table that was pasted here
after the measurement. Budget: the whole Kronos run (validation grid + distillation teacher + test) <= ~3 hours
of wall clock at 2 threads, one heavy process at a time.

Measured 2026-11-21 on BTCUSDT contexts (488 hourly candles, 24-hour horizon), torch 2 threads, KV-cached
wrapper, after one warm-up forecast. CPU seconds = process time (about 2 x wall on 2 threads).

| Model | Params (model + tokenizer) | N paths | Days per batch | Wall s / forecast | CPU s / forecast |
|---|---|---|---|---|---|
| Kronos-mini | 4.1M + 4.0M | 16 | 8 | 0.30 | 0.60 |
| Kronos-mini | | 32 | 4 | 0.60 | 1.16 |
| Kronos-small | 24.7M + 4.0M | 16 | 4 | 1.15 | 2.24 |
| Kronos-small | | 32 | 2 | 2.06 | 4.01 |
| Kronos-base | 102.3M + 4.0M | 16 | 1 | 3.23 | 6.29 |
| Kronos-base | | 32 | 1 | 5.32 | 10.29 |

(The vendored original, without a cache and with one full-window pass per step and per sample, is roughly
L/2 ~ 200x more work per path; it would not fit the budget at all.) Cost is linear in N, so **N = 16** paths.

Arithmetic (wall clock, 2 threads; 10 assets; test 2025-08-01..2026-10-31 = 457 days = 4,570 forecasts):

| Job | Forecasts | s / forecast | Minutes |
|---|---|---|---|
| Validation grid, mini, 3 settings, every 6th day (61 days x 10) | 3 x 610 | 0.30 | 9 |
| Validation grid, small, 3 settings, every 9th day (41 days x 10) | 3 x 410 | 1.15 | 24 |
| Distillation teacher, small, 2022-01..2024-06 every 6th day (152 days x 10) | 1,520 | 1.15 | 29 |
| **Test, mini, every day, all assets** | 4,570 | 0.30 | 23 |
| **Test, small, every day, all assets** | 4,570 | 1.15 | 88 |
| Base: validation BTC+ETH every 6th day; test BTC+ETH every 4th day | 122 + 229 | 3.23 | 19 |
| **Total** | | | **~192 min = 3.2 h** |

Cuts already taken to fit ~3 h: validation on a day subsample (stride 6 / 9), teacher on every 6th day,
Kronos-base only on BTC/ETH at stride 4 in the test (its comparisons with HAR are on the same subset).
N is chosen on validation among subsets {4, 8, 16} of the 16 paths, at no extra cost.

## 4. Data (free, no keys)

* Binance spot monthly klines from data.binance.vision, `.CHECKSUM` (SHA-256) verified on every file,
  raw zips in `~/.cache/quant-research/p3/binance/` (the read-only `~/.cache/qal/binance/` holds only
  futures, so nothing is reused from it). 1h klines feed Kronos; 5m klines give the target.
* 10 USDT pairs with history well before the test: BTC, ETH, SOL, BNB, XRP, DOGE, ADA, LTC, LINK, TRX.
  Months 2020-01 to 2026-10 (SOL from 2020-08). Binance switched spot timestamps to microseconds on
  2025-01-01; the parser handles both.
* No equity ETFs (crypto only, as allowed).

## 5. Target and information set

* **Day** = UTC calendar day [00:00, 24:00). This matches Kronos's natural aggregation of 24 hourly candles
  with open times 00:00..23:00.
* **Target RV_d** = sum of the 288 squared 5-minute log returns of day d (first return: 00:00 open to the
  00:05 close; a gap inside the day is spanned by one return). Days with fewer than 260 bars are flagged,
  not dropped.
* **Forecast time** = 00:00 UTC of day d. Every model uses only data that is closed by then: hourly candles with
  open time < 00:00 of d, daily RVs up to day d-1. Same forecast times and dates for all models.

## 6. Models

| Model | Inputs at 00:00 of day d | Fit |
|---|---|---|
| HAR-RV | RV_{d-1}, mean RV over d-7..d-1, mean over d-30..d-1 (calendar week/month: 24/7 market; Corsi uses 5/22 trading days) | OLS per asset, refit every day on a rolling window (365 / 730 days or expanding, chosen on validation) |
| log-HAR | same in logs | OLS; forecast exp(fit + s^2/2) |
| EWMA | RiskMetrics lambda = 0.94 on daily RV (variance proxy = 5-minute RV) | none |
| GARCH(1,1) | daily close-to-close returns (normal QMLE, constant mean) | `arch`, 1000-day window, refit at each month start, filtered daily |
| LightGBM | HAR features (logs) + 1-day return + daily Parkinson | pooled over assets, log target, refit at each month start, expanding |
| Kronos-mini / -small (-base on a subset) | last 488 hourly candles (OHLCV + quote volume) | zero-shot; N paths of 24 hourly candles |
| Student | HAR-style log features + last 48 hourly log-ranges/returns | distilled from Kronos-small on pre-test forecasts |

**Kronos variance extraction.** Per sampled path: (a) `cc` = sum of the 24 squared hourly close-to-close log
returns (first return from the close at the forecast time); (b) `park` = Parkinson on the predicted hourly
high/low. Forecast = mean over paths. Hourly close-to-close variance and 5-minute RV differ in level
(microstructure, intraday autocorrelation) and Kronos's decoded paths are smoother than real ones, so the raw
forecast has a scale bias. We report the **raw** forecast and a **calibrated** one, `c x raw`, with a single
pooled scale c per model fitted on the validation period (the variance-proxy ratio; exp of the mean log ratio,
the QLIKE-optimal constant up to a small Jensen term) and frozen. The calibrated version is the headline
Kronos row; this is one number per model, chosen on validation, never on test.

**Kronos inference.** The vendored original re-runs the full window at each of the 24 steps and once per
sample, and returns the average path. `kvh/kronos_fast.py` computes the same thing with a key/value cache: the
context is encoded and prefilled once, then copied to the N paths; each new token costs one position. Context
L = 488 so L + 24 = 512 never exceeds the window (no sliding, so the cache is exact). A pytest checks the
greedy-decoded paths against the original. Seeds are derived from (model, tag, symbol, first day of the batch).

## 7. Protocol (no tuning on test)

| Period | Dates | Use |
|---|---|---|
| Baseline history | 2020-03 .. | HAR/GARCH/LightGBM training windows (rolling, always <= day d-1) |
| Distillation train | 2022-01-01 .. 2024-06-30 | Kronos-small teacher forecasts (stride as throughput allows) -> student training set |
| Validation | 2024-07-01 .. 2025-06-30 | Kronos sampling settings, extraction variant, scale c, sample count; HAR window; student hyper-parameters |
| Embargo | 2025-07 | unused (weights uploaded 2025-06-30 / 07-01) |
| **Test** | **2025-08-01 .. 2026-10-31** | scored once |

Validation choices (all by pooled QLIKE): Kronos tag in {T=0.6/p=0.9, T=0.9/p=0.9, T=1.0/p=1.0} (on a
subsample of validation days if throughput requires), variant in {cc, park}, N in {4, 8, 16} (subsets of the 16
sampled paths), HAR window in {365, 730, expanding}, student width/epochs/learning rate. Note: the validation
year is after the paper's stated data end for small/base but may be inside Kronos-mini's training data, so
validation is optimistic for mini; this only affects choices, not test scores.

`freeze` writes `results/frozen_config.json` and its SHA-256. `test` writes one line per run to
`results/test_log.jsonl` (UTC timestamp, config hash, code hash = SHA-256 of src/ + third_party/), refuses a
second run without `--force`, and logs the forced rerun.

## 8. Metrics

* QLIKE = RV/F - log(RV/F) - 1 (Patton 2011, robust to noise in the proxy), primary; MSE of RV in (%^2)^2.
* Mincer-Zarnowitz R^2: regress log RV on log F.
* Diebold-Mariano vs HAR: loss differential d_t = L(model) - L(HAR); pooled = cross-sectional mean per day
  (so cross-asset correlation is handled), Newey-West HAC with lag 7 (6 + 1 for 1-step forecasts with
  overlapping-information persistence); per asset the same on the asset series. Negative = better than HAR.
* Model Confidence Set (Hansen, Lunde & Nason 2011), T_max statistic, block bootstrap (block 7, 2,000 reps),
  alpha 10%, on pooled daily QLIKE.
* Splits: calm vs turbulent days (turbulent = top RV quintile per asset in the test period, defined ex post on
  the realised target, a standard conditional evaluation), BTC/ETH vs alts.
* Cost: parameters, CPU seconds per forecast (process time on 2 threads), total compute.

## 9. Distillation

1. Teacher: Kronos-small with the frozen setting on 2022-01..2024-06 (pre-validation) and on validation.
2. Student: MLP on 30 features: log RV lags (1, 7, 30), log of the last 7 daily RVs, last 24 hourly log
   ranges and squared returns summarised (log mean of last 4/12/24 h), log Parkinson of the last day; target =
   log(teacher calibrated forecast). < 100k parameters (about 5k actually).
3. Hyper-parameters chosen on validation by QLIKE of the student against realised RV (the teacher is the
   training label; the realised RV is the selection criterion), refit on all pre-test teacher forecasts.
4. Report: parameters, CPU microseconds per forecast vs Kronos, accuracy vs teacher (QLIKE to teacher, R^2 of
   log) and vs HAR on test.

## 10. Tests (pytest)

1. RV from 5-minute klines on a toy series. 2. HAR OLS recovers simulated coefficients. 3. QLIKE and DM on fixed
inputs. 4. Leak test: changing data after the forecast time leaves Kronos inputs, HAR features and the
student's features/forecast unchanged. 5. Kronos wrapper: shape, determinism, match with the original.
6. `TEST_START > KRONOS_DATA_END` on the config.

## 11. Risks and fallbacks

* **Throughput** (biggest risk): if small/base is too slow, cut in this order: Kronos-base to a subset (BTC/ETH,
  stride), validation grid to a day subsample, sample count, distillation-train stride. Never the test period.
* Watchdog stalls: every long job runs in the background with per-chunk logging and per-(model, symbol, month)
  parquet output, so a crash resumes.
* Scale bias of raw Kronos variance: handled by the frozen validation scale; raw also reported.
* Kronos-mini trained on unknown data: conservative cutoff covers it.
* Binance data gaps: flagged days kept; hourly gaps filled flat (zero volume) and counted.
* Memory: KV cache sized as batch x N x layers x 512 x d; batch per model kept small.

## 12. Changes during the build

(appended as they happen)
* **2026-11-21 23:20 - Kronos inference re-implemented with a KV cache** (`kvh/kronos_fast.py`) after reading the
  original: it averages the sample paths before returning (useless for a variance forecast) and recomputes the
  full window per step and per sample. Greedy-decoding equivalence with the vendored original is a pytest
  (mini and small agree to float tolerance).
* **23:27 - thin slice done** (BTC, July 2024, HAR vs Kronos-mini, 31 days): pipeline runs end to end; raw
  Kronos-mini close-to-close variance is ~65% of the 5-minute RV level, Parkinson ~94%, confirming the need
  for the validation scale c.
* **23:28 - LightGBM** uses the native `lgb.train` API (the sklearn wrapper needs scikit-learn, not installed).
* **23:29 - HAR floor replaced by the insanity filter of Bollerslev, Patton & Quaedvlieg (2016).** The first
  version floored level-HAR forecasts at the 0.1% quantile of the *whole* RV series (a look-ahead, and too low:
  on validation TRXUSDT had QLIKE 537 on one day). Now a forecast outside [min, max] of the RVs in its own
  fitting window is replaced by that window's mean (past data only). Seen on validation only. The data-build
  floor for zero-RV days was also changed from 1% of the full-sample median to an absolute 1e-10 (0 days hit
  either floor, so the built data are unchanged).
* **23:29 - two EWMA rows**: RiskMetrics on squared daily returns (the textbook version) and the same recursion on
  5-minute RV (the fairer benchmark). Both lambda = 0.94, nothing tuned.
* **23:30 - Kronos test/distill forecasts can only be generated with the frozen validation setting** (`--tag` is
  refused for those periods); every test-period generation run is logged to `results/kronos_test_generation.jsonl`
  with the SHA-256 of `frozen_kronos.json`. The full `frozen_config.json` (with the student) is written by
  `freeze` before the test-period Kronos generation starts.
* **23:34 - Kronos settings are compared on the same asset-days** (intersection across settings) in `validate`.
* **23:37 - run order.** To put the most important job first, the background chain runs the test-period Kronos
  generation (mini, then small; frozen settings from `frozen_kronos.json`) before the distillation teacher and
  Kronos-base. The student never reads test-period files (its labels are pre-test teacher forecasts and its
  selection uses validation RV), and `freeze` + `test` (the scoring) still come after the student is fitted.
* **23:50 - one pre-committed combination row** (decided before any test scoring, nothing tuned): the geometric
  mean of the log-HAR and calibrated Kronos-small forecasts. It asks whether Kronos carries information that
  HAR lacks (if so, the combination beats both). Same scoring as every other model.
* **23:50 - student feature extraction** reads arrays by position (searchsorted) instead of pandas slicing; the
  student's CPU cost is reported as forward pass and feature extraction separately.
* **00:15 - validation choices (frozen in `results/frozen_kronos.json`, `frozen_har.json`):**
  * Kronos-mini: T = 0.9 / top-p 0.9 (the paper's volatility setting), close-to-close variant anchored on the
    reconstructed last close, N = 16, scale c = 2.31; validation QLIKE 0.217 vs HAR 0.243 on the same 610
    asset-days.
  * Kronos-small: T = 0.6 / top-p 0.9, Parkinson variant, N = 8, c = 1.86; validation QLIKE 0.252 vs HAR
    0.228 on the same 410 asset-days (worse than HAR on validation already).
  * HAR rolling 730 days; log-HAR expanding.
  * Observation: the temperature sets the raw variance level of the generated paths (mini close-to-close:
    c = 8.5 at T = 0.6, 2.3 at T = 0.9, 0.83 at T = 1.0), which is why a scale is needed at all; and
    calibrated QLIKE still falls from N = 8 to 16 for mini, so more paths could help further.
* **00:49-03:06 - Kronos chain finished**: test mini (4,570 forecasts, 34 min), test small (4,570, 98 min),
  teacher (small, 1,520 pre-validation forecasts, 24 min), Kronos-base validation (BTC/ETH, 122) and test (BTC/ETH
  every 4th day, 230), `validate` re-run (adds base: T = 0.6 / top-p 0.9 by rule, Parkinson, N = 8, c = 1.25;
  mini/small choices unchanged). Three generation runs logged in `results/kronos_test_generation.jsonl`.
* **03:06-10:40 - pause.** The session was cut by a network error (API ENOTFOUND). No process was running and no
  stage was half-done; the remaining stages were resumed from the files on disk.
* **10:41 - student**: grid of 8 (width 32/64, weight decay 1e-4/1e-2, 100/300 epochs) chosen by validation QLIKE
  against RV -> width 32, wd 0.01, 100 epochs, 2,369 parameters, refit on 1,930 pre-test teacher labels.
* **10:41 - freeze** (`results/frozen_config.json`, SHA-256 aba34451afb7...) and the **single test run**
  (`results/test_log.jsonl`, run 1, forced = false, code SHA-256 a0cc4e422d02...). No rerun.
* **10:45 - post-hoc readings, added after the test and labelled as such in the README** (no refit, no new
  forecasts, the frozen test forecasts only): pairwise DM tests other than "vs HAR" (`results/dm_extra.csv`)
  and the scores without the 2025-10-10 crash day (`results/robustness_excl_crash.csv`), because the rolling
  QLIKE figure showed that one day (cross-asset mean RV ~220x the median day) moves every model's mean QLIKE by
  0.26-0.36 and decides the sign of Kronos vs level-HAR.
* **10:45 - report formatting only** (after the test run, so the code hash in the test log predates it):
  CPU cost below the Windows timer floor printed as "<4e-06", the teacher cost labelled "16 paths sampled, first
  8 used", and the extra tables added to `tables.md`/README.
* **2026-11-22, after publication on GitHub - README wording only:** the headline now leads with the
  distillation result, and the cost-against-accuracy figure moved under it. The "series" line was removed. Same
  numbers, same files; no model, test or result changed.
