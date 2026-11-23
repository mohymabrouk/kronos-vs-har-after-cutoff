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

## 13. Change E1 (declared 2026-11-23, before any forecast of the new models): three newer foundation models, each scored only after its release

Written before the TSFM code and before any forecast (validation or test) of these models. Nothing in sections
1-12 changes: the original test run (results/test_log.jsonl run 1) and its files stay as they are.

### 13.1 Models, pinned weights and the dates that bound their training data

Evidence: HuggingFace API `/api/models/<repo>/commits/<branch>` and `/tree/<revision>` (LFS object id of every
file), and each model card, read 2026-11-23.

| Model | Repo, branch | Pinned revision (the commit that uploaded the weights used) | Weights date | Later commits | Training data stated on the card |
|---|---|---|---|---|---|
| TTM (r2.1 release) | `ibm-granite/granite-timeseries-ttm-r2`, branch `512-96-ft-r2.1` (the checkpoint 2607.05291 uses) | `a23813d5e5859987e522d084f7d1d1ef9f134c3e` | 2025-01-31 | branch head `3442406f` (2025-02-26) has the same `model.safetensors` and `config.json` objects | fixed public datasets (Monash/Zenodo collections, LOTSA subsets, PEMS...), including the Monash "Bitcoin" set that ends in 2021; no date cutoff stated |
| TimesFM-2.5 | `google/timesfm-2.5-200m-pytorch`, main | `d418f3e8a8fa79d655b391c158f0ee8d68fe68c9` (2025-10-01, "Refreshed model with speed optimizations": fused QKV) | **2025-09-02** (first upload `d35e0e45`) | checked: the 2025-10-01 file is the 2025-09-02 file with query/key/value concatenated, 232 of 232 tensors bit-identical; later commits edit the README only | GiftEvalPretrain; Wikimedia pageviews "cutoff Nov 2023"; Google Trends "cutoff EoY 2022"; synthetic data |
| Chronos-2 | `amazon/chronos-2`, main | `95a9710e2596287d08352589f42634fa5abdf0a7` ("Add model weights and config", 2025-10-30) | 2025-10-30 | `model.safetensors` and `config.json` keep the same object ids at every later commit up to main `29ec3766` (2026-06-05); the later commits edit the README and tags | subsets of Chronos datasets and GIFT-Eval Pretrain, synthetic data; no date cutoff stated |

All three are Apache 2.0. The weights date is an upper bound on the end of the training data. Effective cutoff =
the later of the weights date and the latest dated source on the card (TimesFM: Nov 2023 < 2025-09-02, so the
weights date). **Scoring start = max(TEST_START, effective cutoff + one calendar month)**, the convention of the
Kronos test (2025-07-01 + one month = 2025-08-01). Hence:

| Model | Effective cutoff | Scoring start | Test days | Contains 2025-10-10 |
|---|---|---|---|---|
| TTM | 2025-01-31 | 2025-08-01 (whole test) | 457 | yes |
| TimesFM-2.5 | 2025-09-02 | 2025-10-02 | 395 | yes |
| Chronos-2 | 2025-10-30 | 2025-11-30 | 336 | no |

Test-period forecasts of a model are generated only from its scoring start. A pytest checks the dates.

### 13.2 Input, output and variance forecast (zero-shot, nothing fine-tuned)

Two input specifications, both pre-declared and both reported:

* **H, Kronos-matched (the primary specification).** The same hourly candles Kronos sees at 00:00 UTC of day d:
  the last 488 hourly candles (TimesFM-2.5: 488 values, left-padded and masked to 512 by its own API;
  Chronos-2: 488). TTM's checkpoint has a fixed context of 512, so it gets the last 512 candles (24 older
  hours, all closed before the forecast time). The three models are univariate, so the candles are turned into
  an hourly volatility series, in one of two variants (as Kronos's close-to-close / Parkinson):
  * `cc`: |log C_h - log C_{h-1}| (the first hour uses its own open);
  * `park`: ln(H_h / L_h) / (2 sqrt(ln 2)), the square root of the hourly Parkinson variance.
  The model forecasts the next 24 hourly values s_h; raw daily variance = sum over the 24 hours of max(s_h, 0)^2.
  The variant is chosen on validation (calibrated QLIKE), as the Kronos variant was.
* **D, as in arXiv 2607.05291.** The daily volatility sigma_t = sqrt(RV_t) (5-minute RV, the target itself) of
  days up to d-1; context 512 days for TTM (the r2.1 maximum, as in the paper) and 1,000 days for TimesFM-2.5
  (padded/masked to 1,024 by its API) and Chronos-2 (the paper's context for its other models); horizon 1;
  raw daily variance = max(sigma_hat, 0)^2.

Point forecast = the model's mean: TTM's output (MSE-trained); TimesFM-2.5's mean channel (index 0 of its
quantile output, not the median it returns as "point"); Chronos-2: the average of its 0.1..0.9 quantiles (the
paper's rule for Chronos-Bolt and Moirai; Chronos-2's own "mean" is its median). Settings: TTM frequency token
"h" (H) or "d" (D), and the context standard-scaled per window as its card asks; TimesFM-2.5 with the
model-card ForecastConfig (normalize_inputs, continuous quantile head, flip invariance, infer_is_positive,
fix_quantile_crossing), max_horizon 128, no torch.compile; Chronos-2 with its defaults (univariate, no
cross-learning), float32. A raw forecast below 1e-10 is floored there (the data-build floor) and counted.

**Calibration, as for Kronos:** one pooled scale c per model and specification, c = mean(RV / raw) on the
validation period (2024-07-01 .. 2025-06-30, every day, 10 assets), frozen. Headline rows = c x raw; raw rows
are also scored. The validation year is before every model's weights date, so it may be inside their training
data; that can only affect c and the variant, not the test.

### 13.3 The TTM + log-HAR average (the 2607.05291 claim)

2607.05291 (v2, 2026-11-20): nine zero-shot TSFMs against eight econometric models on 50 VOLARE assets; only
TTM (512-96-ft-r2.1, daily sqrt(RV), 512 days) beats log-HAR in average QLIKE, narrowly, at every horizon; the
equal-weight average of the two "squares the average of the two volatility forecasts" and is in the 10% MCS for
100% of the assets at h = 1. Pre-declared rows, one per specification:
**Avg = ((sqrt(F_TTM) + sqrt(F_log-HAR)) / 2)^2**, with F_TTM the calibrated TTM forecast. Read-outs: TTM (D)
against log-HAR (QLIKE, DM); the average against log-HAR and against TTM (DM); the average in the pooled 10% MCS
and in the per-asset 10% MCS (count of assets out of 10).

### 13.4 Scoring (one run)

* For each new model, on its own window (13.1), every row is scored on the same asset-days: HAR, log-HAR,
  LightGBM, Kronos-mini, Kronos-small, the student, every new model that is clean on the whole window (both
  specifications, calibrated and raw) and the two averages.
* Metrics: QLIKE (primary), MSE, MZ R^2; pooled DM (daily cross-sectional mean of the loss differential,
  Newey-West lag 7) of each new row against each of HAR, log-HAR, LightGBM, Kronos-mini, Kronos-small and the
  student, on QLIKE and MSE, and of each average against its TTM row; the same with 2025-10-10 removed when the
  window contains it; per asset QLIKE and DM against HAR and log-HAR; MCS (T_max, block 7, 2,000 draws, 10%)
  pooled and per asset over the calibrated rows of the window.
* `python -m kvh.cli tsfm-test` scores once. It re-checks the SHA-256 of `frozen_config.json` and of the new
  `frozen_tsfm.json`, refuses a second run without `--force`, and appends to `results/test_log.jsonl` with
  `forced = true` and `added = "TSFM extension (PLAN.md section 13)"` (the original test already ran once).
* **Reproduction check** in the same run: every existing model's test forecasts are recomputed from scratch on
  the full window and compared with `results/forecasts_test.csv` and `results/performance_test.csv`
  (`results/tsfm_reproduction.json`). The original result files are not rewritten.

### 13.5 Environment

granite-tsfm (TTM) pins `torch<2.12`; the project is locked at torch 2.15.0+cpu. To keep the environment of
the existing results, `[tool.uv] override-dependencies` lifts that upper bound instead of downgrading torch;
TTM outputs are checked against a separate torch 2.11 environment on the same inputs. Weights live in the HF
cache under the home directory (`~/.cache/quant-research/p3/hf`), never in the repo. 2 threads, one heavy
process at a time.

### 13.6 Changes during the extension

(appended as they happen)
* **2026-11-23 11:58 UTC - throughput** (smoke test on 32 validation days of BTC, 2 threads, wall seconds per
  forecast, batch 32): TTM 0.002 (H) / 0.001 (D); TimesFM-2.5 0.19 / 0.34; Chronos-2 0.09 / 0.18. So validation
  on every day (3,650 asset-days per series) and the full test windows fit in about 2 hours; no stride.
* **11:58 - parameter counts** (counted from the loaded models): TTM 855,100; TimesFM-2.5 231,289,280;
  Chronos-2 119,477,664.
* **12:06 - torch check done** (`scripts/ttm_torch_check.py`, `results/tsfm_ttm_torch_check.json`): TTM under
  torch 2.11 (granite-tsfm's declared range, separate environment) against the project's torch 2.15 on 120
  validation inputs per series: largest relative difference 2.5e-7 over 5,880 values (float32 rounding).
* **12:12 - the weight file of each pinned revision is checked by SHA-256** (the LFS object id, in
  `config.TSFM_MODELS`) in a pytest that skips when the weights are not cached (CI).
* **12:00 - no local absolute paths in scripts.** The four existing chain scripts did `cd` to an absolute local
  path; they now `cd` relative to the script. No behaviour change.
* **12:01-13:09 - validation forecasts** (`scripts/run_tsfm_val.sh`): every validation day, 10 assets, H (cc and
  park) and D, for the three models; TTM 0.5 min, TimesFM-2.5 38 min, Chronos-2 30 min. No forecast hit the floor.
* **13:08 - dry run of the scoring and report code on validation forecasts only** (fake windows inside the
  validation year, output outside the repo), to debug the code before the single test run. No test-period
  forecast existed at that point.
* **13:10 - `tsfm-validate`, frozen** (`results/frozen_tsfm.json`, SHA-256 7378c155e058...): the Parkinson
  series (`park`) wins on validation for all three hourly inputs. Scales c: TTM 1.400 (H) / 1.097 (D);
  TimesFM-2.5 1.642 / 1.237; Chronos-2 1.703 / 1.236. Calibrated validation QLIKE: TTM 0.245 / 0.244,
  TimesFM-2.5 0.229 / 0.231, Chronos-2 0.240 / 0.242, against HAR 0.306 and log-HAR 0.260 on the same
  asset-days (the validation year is before the weights dates, so possibly inside the training data).
* **13:10 - test-period chain started** (`scripts/run_tsfm_test.sh`): forecasts from each scoring start with
  the frozen settings, then `tsfm-test` once, then `report`.
* **13:10-14:36 - test forecasts** (logged in `results/tsfm_test_generation.jsonl`): TTM 1 min, TimesFM-2.5
  58 min, Chronos-2 26 min. Two other projects ran on the machine at the same time, so these runs were slower
  than the throughput check (TimesFM-2.5 daily input about 0.6 s wall per forecast instead of 0.34).
* **14:37 - the single scoring run** (`tsfm-test`; `results/test_log.jsonl` run 2, forced = true, added = "TSFM
  extension (PLAN.md section 13)", original config aba34451afb7..., TSFM config 7378c155e058..., code
  a5a54f703c90...). Reproduction check: all 55,300 original
  forecast rows recomputed from scratch, largest relative difference 0, `performance_test.csv` reproduced
  exactly. Read-outs (all from the run's files): with the daily RV input, TTM beats log-HAR on the whole test
  and TimesFM-2.5 on its window (and LightGBM too); with the hourly input no model beats log-HAR
  significantly; every new model scores below Kronos and the student; the TTM + log-HAR average beats log-HAR
  but not TTM. No rerun.
* **After the scoring run - report and README only** (so the code hash in the test log predates these):
  three counts computed from `results/tsfm_dm.csv` for the README prose (significant DM tests against
  Kronos/student, HAR and on MSE), parameter counts and scoring starts as placeholders instead of typed
  numbers, and `scripts/timesfm_weights_check.py` (the TimesFM-2.5 weight-equivalence check of 13.1, moved
  from a scratch script into the repo; its output is `results/tsfm_timesfm_weights_check.json`). No forecast,
  choice or score changed.
