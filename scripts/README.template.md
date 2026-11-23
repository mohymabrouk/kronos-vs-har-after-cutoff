# Kronos vs HAR-RV: next-day crypto volatility after the foundation model's data cutoff

[![tests](https://github.com/mohymabrouk/kronos-vs-har-after-cutoff/actions/workflows/tests.yml/badge.svg)](https://github.com/mohymabrouk/kronos-vs-har-after-cutoff/actions/workflows/tests.yml)

Every number below is read from `results/` by `python -m kvh.cli report`; nothing in the tables was typed by
hand.

## Headline

**A 2,369-parameter student beats the 28.7M-parameter Kronos model it was trained to copy, and neither beats
the classical volatility models.** The test covers 15 months of Binance data published after the Kronos
weights: Aug 2025 to Oct 2026, 10 USDT pairs, {{N_TEST}} asset-days per model.

![Cost against accuracy](results/fig_cost_vs_qlike.png)

* **The student beats its teacher.** The MLP was trained only to imitate pre-test Kronos-small forecasts. On
  test it scores QLIKE {{Q_STUDENT}} against the teacher's {{Q_KRONOS_SMALL}} (DM {{DM_STUDENT_TEACHER}},
  {{P_STUDENT_TEACHER}}).
  * Its forward pass takes {{STUDENT_US}} microseconds, against {{TEACHER_CPU}} CPU-seconds per teacher
    forecast.
  * It does not significantly beat HAR (DM {{DM_STUDENT}}, p = {{P_STUDENT}}), and it stays behind log-HAR.
* **Kronos does not beat HAR-RV.** Zero-shot Kronos variance forecasts, even with a scale fitted on
  validation, score pooled QLIKE {{Q_KRONOS_MINI}} (mini) and {{Q_KRONOS_SMALL}} (small), against
  {{Q_HAR}} for HAR. The DM statistics are {{DM_KRONOS_MINI}} and {{DM_KRONOS_SMALL}} (p = {{P_KRONOS_MINI}} and
  {{P_KRONOS_SMALL}}), so the difference is not significant.
* **It is significantly worse than the stronger classical models.** Those are log-HAR ({{Q_LOG_HAR}}) and
  LightGBM on HAR features ({{Q_LIGHTGBM}}, the best model). Kronos-base, on a BTC/ETH subset, is worse
  still.
* **Kronos adds nothing to log-HAR.** Averaging log-HAR with Kronos-small gives {{Q_COMBO_LOG_HAR_X_KRONOS_SMALL}},
  worse than log-HAR alone.
* **Turbulent days decide the comparison with HAR.**
  * On the calmer 80% of days, Kronos beats level-HAR. On the top RV quintile it loses heavily.
  * One day dominates: the 2025-10-10 crash. Its cross-asset mean RV was about 220 times the median day's,
    and 22 times the next-worst day's. It adds 0.26 to 0.36 to every model's mean QLIKE.
  * Dropping that day is a post-hoc reading, not the test. Without it, Kronos edges ahead of HAR, but still
    not significantly (DM {{RDM_MINI}} and {{RDM_SMALL}}). It stays significantly worse than log-HAR. The
    student then beats HAR significantly (DM {{RDM_STUDENT}}).
* **Answer to the question.** Does Kronos beat HAR-RV after its data cutoff? No. The student keeps, and
  improves on, whatever Kronos knows about tomorrow's variance, but what Kronos knows is not an edge over the
  2009 benchmark.

**Extension (added 2026-11-23): three newer, general time-series foundation models do better than Kronos.
Fed the daily realised-variance series, TTM and TimesFM-2.5 beat log-HAR at the 5% level after their release
dates, but not after correcting for the {{HOLM_N}} tests run. Fed the hourly candles Kronos sees, none of the three beats
log-HAR significantly.** Each model is scored only from one
month after its weights were published. The settings were declared in PLAN.md section 13 before any forecast,
and the test was scored once. Details are in the section "Newer foundation models".

* **Daily RV input (the setup of arXiv 2607.05291).**
  * TTM ({{TSFM_PARAMS_TTM}} parameters), on the whole test: QLIKE {{TQ_TTM_TTM_R2_1_DAILY_RV}} against
    log-HAR's {{TQ_TTM_LOG_HAR}} (DM {{TDM_TTM_TTM_R2_1_DAILY_RV_VS_LOG_HAR}}, p =
    {{TDMP_TTM_TTM_R2_1_DAILY_RV_VS_LOG_HAR}}). It ties LightGBM (DM {{TDM_TTM_TTM_R2_1_DAILY_RV_VS_LIGHTGBM}}).
  * TimesFM-2.5, from {{TSFM_START_TIMESFM}}: {{TQ_TIMESFM_TIMESFM_2_5_DAILY_RV}} against log-HAR's
    {{TQ_TIMESFM_LOG_HAR}} (DM {{TDM_TIMESFM_TIMESFM_2_5_DAILY_RV_VS_LOG_HAR}}, p =
    {{TDMP_TIMESFM_TIMESFM_2_5_DAILY_RV_VS_LOG_HAR}}). It also beats LightGBM, the best model of the original
    test ({{TQ_TIMESFM_LIGHTGBM}}, DM {{TDM_TIMESFM_TIMESFM_2_5_DAILY_RV_VS_LIGHTGBM}}).
  * Chronos-2, from {{TSFM_START_CHRONOS2}}: {{TQ_CHRONOS2_CHRONOS_2_DAILY_RV}} against log-HAR's
    {{TQ_CHRONOS2_LOG_HAR}}. The difference is not significant (DM {{TDM_CHRONOS2_CHRONOS_2_DAILY_RV_VS_LOG_HAR}},
    p = {{TDMP_CHRONOS2_CHRONOS_2_DAILY_RV_VS_LOG_HAR}}).
  * Without the 2025-10-10 crash day, TTM and TimesFM-2.5 still beat log-HAR (DM
    {{TDMX_TTM_TTM_R2_1_DAILY_RV_VS_LOG_HAR}} and {{TDMX_TIMESFM_TIMESFM_2_5_DAILY_RV_VS_LOG_HAR}}).
  * **Multiple testing.** These are {{HOLM_N}} pre-declared tests against log-HAR: 3 models x 2 inputs, each
    on its own window. After a Holm correction for all {{HOLM_N}}, {{HOLM_N_SIG}} stay significant at 5%
    (smallest adjusted p = {{HOLM_MIN_P}}). The 10% model confidence set drops log-HAR only in the TTM window
    and keeps it in the TimesFM-2.5 and Chronos-2 windows. Read the daily-input wins as suggestive, not
    established.
* **Hourly input (Kronos-matched).** No model is significantly better than log-HAR on its own window: DM
  {{TDM_TTM_TTM_R2_1_HOURLY_VS_LOG_HAR}} for TTM (p = {{TDMP_TTM_TTM_R2_1_HOURLY_VS_LOG_HAR}}),
  {{TDM_TIMESFM_TIMESFM_2_5_HOURLY_VS_LOG_HAR}} for TimesFM-2.5 (p =
  {{TDMP_TIMESFM_TIMESFM_2_5_HOURLY_VS_LOG_HAR}}) and {{TDM_CHRONOS2_CHRONOS_2_HOURLY_VS_LOG_HAR}} for
  Chronos-2 (p = {{TDMP_CHRONOS2_CHRONOS_2_HOURLY_VS_LOG_HAR}}).
  * TimesFM-2.5 (hourly) still has the lowest QLIKE of its window ({{TQ_TIMESFM_TIMESFM_2_5_HOURLY}}).
    Without the crash day, its daily-RV version is lower ({{TQX_TIMESFM_TIMESFM_2_5_DAILY_RV}} against
    {{TQX_TIMESFM_TIMESFM_2_5_HOURLY}}).
* **Against Kronos, the student and HAR.** Every new model, with either input and in every window, has a
  lower QLIKE than Kronos-mini, Kronos-small and the student.
  * Against those three, {{TSFM_KRONOS_SIG}} DM tests are significant at 5%. The others are all in the
    TimesFM-2.5 window, which includes the crash day.
  * Against HAR, {{TSFM_HAR_SIG}} are significant.
* **The 2607.05291 claim, on crypto with no training-data overlap.**
  * In the paper, only TTM had a lower average QLIKE than log-HAR, by a narrow margin. Here TTM beats log-HAR
    by more: a QLIKE ratio of {{TR_TTM_TTM_R2_1_DAILY_RV}}, against {{TSFM_PAPER_TTM_RATIO}} there.
    TimesFM-2.5 beats it too.
  * The equal-weight TTM + log-HAR average also beats log-HAR (ratio {{TR_TTM_AVG_TTM_R2_1_DAILY_RV_LOG_HAR}},
    DM {{TDM_TTM_AVG_TTM_R2_1_DAILY_RV_LOG_HAR_VS_LOG_HAR}}). But it does not improve on TTM alone (DM
    {{TDM_TTM_AVG_TTM_R2_1_DAILY_RV_LOG_HAR_VS_TTM_R2_1_DAILY_RV}}, p =
    {{TDMP_TTM_AVG_TTM_R2_1_DAILY_RV_LOG_HAR_VS_TTM_R2_1_DAILY_RV}}). In the paper its ratio was below TTM's
    ({{TSFM_PAPER_AVG_RATIO}} against {{TSFM_PAPER_TTM_RATIO}}).
  * The average is in the per-asset 10% MCS for {{TSFM_AVG_ASSETS_MCS_D}} of 10 assets (paper:
    {{TSFM_PAPER_AVG_MCS}}% of 50). TTM is in it for {{TSFM_TTM_ASSETS_MCS_D}} and log-HAR for
    {{TSFM_LOGHAR_ASSETS_MCS}}.
* **Caveats.**
  * These are single pairwise tests on 10 correlated assets. The pooled 10% MCS still keeps LightGBM in every
    window, and log-HAR in the two later ones.
  * On MSE, {{TSFM_MSE_SIG}} DM tests of the new models against log-HAR are significant at 5%: a few extreme
    days dominate MSE.
  * The windows of TimesFM-2.5 and Chronos-2 are shorter ({{TSFM_DAYS_TIMESFM}} and {{TSFM_DAYS_CHRONOS2}}
    days).
* **Reproduction.** Recomputing every original model in the new environment gives identical forecasts and
  scores (`results/tsfm_reproduction.json`).

## Question

Kronos (Shi et al., "Kronos: A Foundation Model for the Language of Financial Markets", arXiv 2508.02739,
AAAI 2026) is a decoder-only transformer pretrained on 12 billion candlesticks from 45 exchanges, Binance spot
crypto included. The paper reports a 9% lower MAE than the best baseline for volatility forecasting. Here we ask
two questions on data Kronos could not have seen:

1. Does a Kronos variance forecast beat the HAR-RV model of Corsi (2009) at next-day realised variance?
2. Can a student network under 100k parameters, distilled from Kronos-small on pre-test data only, keep
   Kronos's edge at a fraction of the cost?

An extension added after the original test asks a third question. Do three newer general time-series
foundation models (TTM, TimesFM-2.5, Chronos-2) do better? Each is scored only on days after its own
release. And does the TTM + log-HAR average that arXiv 2607.05291 found best on 50 VOLARE assets hold up on
crypto, with no overlap with the training data?

## Design in one screen

* **Data.** Binance spot, 10 USDT pairs: BTC, ETH, SOL, BNB, XRP, DOGE, ADA, LTC, LINK and TRX. Monthly
  1h and 5m klines from data.binance.vision, 2020-01 to 2026-10, each SHA-256 `.CHECKSUM` verified.
* **Target.** RV of UTC day d = the sum of its 288 squared 5-minute log returns.
* **Forecast time.** 00:00 UTC of day d. Every model sees only candles closed by then.
* **Kronos data end.**
  * The paper says "the pre-training data for Kronos extends up to June 2024".
  * Kronos-mini is not in the paper. So the cutoff we use is the HuggingFace weights upload, 2025-06-30 for
    small and base and 2025-07-01 for mini and its tokenizer. Every later commit only edits the README.
  * **Test: 2025-08-01 to 2026-10-31**, after a one-month embargo. That is {{N_DAYS}} days and {{N_TEST}}
    asset-days scored by every model.
  * Validation: 2024-07-01 to 2025-06-30.
* **Kronos forecast.**
  * Input: the last 488 hourly candles (OHLCV and quote volume).
  * Output: N sampled 24-hour candle paths. The variance forecast is the mean over paths of the per-path
    variance: either the sum of squared hourly close-to-close log returns, or Parkinson on the predicted
    high/low.
  * Raw paths underestimate 5-minute RV, so the headline Kronos rows multiply by a single scale c per model.
    c is fitted on validation and frozen; it is the QLIKE-optimal constant, mean(RV / raw). Raw rows are
    also shown.
* **Chosen on validation only:** sampling temperature and top-p, the variance variant, N, c, the HAR
  windows and the student hyper-parameters. The frozen config's hash is
  `{{CONFIG_HASH}}`.
* **Baselines.**
  * HAR-RV (Corsi 2009; lags 1, 7 and 30 days because crypto trades 24/7) and log-HAR, both OLS refitted
    every day.
  * EWMA with RiskMetrics lambda = 0.94, on squared returns and on RV.
  * GARCH(1,1), refitted monthly on 1,000 days.
  * LightGBM on HAR features, pooled across assets and refitted monthly.
* **Student.**
  * Model: an MLP on 39 features. They are HAR-style daily log-RV lags plus the last 24 hourly log
    squared ranges.
  * Training: on Kronos-small forecasts from 2022-01 to 2025-06 only (teacher on every 6th day), then
    scored on the test period like every other model.
* **Kronos inference on 2 CPU threads.**
  * The vendored model is re-run with a key/value cache: encode and prefill once, then copy to N paths.
  * A pytest checks it against the original code under greedy decoding.
  * Speed: about 0.3 s of wall time per 16-path forecast for mini, 1.2 s for small and 3.2 s for base
    (throughput test). A full-window
    pass per step and per sample would cost roughly 200 times more.

## Main result (test period, pooled over 10 assets)

Read the table as follows:

* QLIKE (primary) = RV/F - log(RV/F) - 1. MSE is in (%^2)^2.
* MZ R^2 regresses log RV on log F.
* DM is the Diebold-Mariano statistic against HAR. It is computed on the daily cross-sectional mean of the
  loss differential, with Newey-West lag 7. Negative means better than HAR.
* CPU cost is process time per forecast on 2 threads; Kronos counts as 2 CPU-seconds per wall second. Every
  Kronos run sampled 16 paths. Kronos-small and -base use the first 8 (the validation choice), so at N = 8
  their cost would be roughly halved.
* "(raw)" means the Kronos forecast without the validation scale.

{{MAIN}}

![QLIKE by model](results/fig_qlike_by_model.png)

**Model confidence set** (Hansen, Lunde & Nason 2011). It uses the T_max statistic on daily pooled QLIKE,
a 7-day block bootstrap with 2,000 draws, and alpha = 10%.

{{MCS}}

At 10% the MCS removes only Kronos-small. Daily pooled QLIKE is dominated by a few crash days, so the set is
wide: even level-HAR stays in it.

**Pairwise DM tests other than "vs HAR".** These are a post-hoc reading of the same frozen test forecasts
(`scripts/extra_dm.py` -> `results/dm_extra.csv`): no refit and no new forecast. Each test uses the same
pooled statistic as the main table, and a positive value means the first model is worse.

{{EXTRA_DM}}

**Without the 2025-10-10 crash day.** This is a post-hoc reading of the same frozen forecasts
(`results/robustness_excl_crash.csv`), not the pre-registered test.

{{ROBUST}}

## Per asset (QLIKE; DM and p for Kronos-small against HAR)

{{PER_ASSET}}

## Splits (QLIKE; DM statistic against HAR in brackets)

* Turbulent days are the top RV quintile per asset over the test period, defined on the realised target.
  Calm days are the rest.
* BTC/ETH are compared with the eight alts.

{{SPLITS}}

![Rolling QLIKE difference against HAR](results/fig_rolling_qlike_vs_har.png)

![BTC forecasts](results/fig_btc_forecasts.png)

## Kronos-base (subset)

Kronos-base was run only on BTC and ETH, every 4th test day, to fit the compute budget. Every model is
scored on those same asset-days.

{{BASE}}

## Distillation

{{DISTILL}}

The cost-against-accuracy figure is under the headline.

## Newer foundation models, each scored only after its release

This section was added after the original test (PLAN.md section 13). Its settings were declared there before
any forecast of these models was made. Three zero-shot time-series foundation models, all newer than Kronos,
are scored only on test days after their weights were published, plus the same one-month embargo as Kronos.
Within each table, every row is scored on the same asset-days.

{{TSFM_PINS}}

* **Pinned weights.** Each model is pinned to the HuggingFace commit that uploaded the weight file used, and a
  test checks that file's SHA-256.
  * TTM is the `512-96-ft-r2.1` checkpoint of the r2 repo, the one 2607.05291 uses.
  * TimesFM-2.5's 2025-10-01 file is a re-export with fused query/key/value matrices. Once its query, key and
    value are concatenated, the 2025-09-02 upload equals it ({{TSFM_TIMESFM_WEIGHTS}},
    `scripts/timesfm_weights_check.py`), so the weights date is 2025-09-02.
  * Chronos-2 is pinned to its original 2025-10-30 commit. Its weight and config files have the same object
    ids on main (2026-06-05); the later commits edit the README.
* **Scoring start** = the later of the weights date and the latest dated training source on the model card,
  plus one calendar month: the Kronos rule (2025-07-01 + one month = 2025-08-01). The 2025-10-10 crash is in
  the TTM and TimesFM-2.5 windows, not in Chronos-2's.
* **Two input specifications**, both declared in advance and both reported.
  * **H (hourly), Kronos-matched, the primary one.** The hourly candles Kronos sees at 00:00 UTC: 488 of them
    (TTM: 512, its fixed context). The models are univariate, so the candles become an hourly volatility
    series: the absolute hourly log return (`cc`), or the square root of the hourly Parkinson variance
    (`park`), chosen on validation. The model forecasts the next 24 hours; the variance forecast is the sum of
    the squared hourly forecasts.
  * **D (daily RV), as in 2607.05291.** The daily volatility sqrt(RV) of the previous days: 512 days for TTM,
    1,000 for TimesFM-2.5 and Chronos-2. The variance forecast is the squared one-day forecast.
* **Point forecast** = the model's mean: TTM's output, TimesFM-2.5's mean head, and for Chronos-2 the average
  of its 0.1 to 0.9 quantiles (2607.05291's rule for quantile models).
* **Calibration, as for Kronos.** One scale c per model and input, c = mean(RV / raw), fitted on the
  validation year and frozen. Headline rows are calibrated; raw rows are in the last table of this section.
* **The TTM + log-HAR average**, as in 2607.05291: ((sqrt(F_TTM) + sqrt(F_log-HAR)) / 2)^2, with the
  calibrated TTM forecast.

Validation (2024-07-01 to 2025-06-30, every day, 10 assets). On the same asset-days HAR scores
{{TSFM_VAL_HAR}} and log-HAR {{TSFM_VAL_LOGHAR}}. This year is before every weights date, so it may be inside
the models' training data; that can affect c and the series choice, not the test.

{{TSFM_VAL}}

How to read the window tables:

* A DM cell is the statistic with its p-value. It is the pooled test of the main table: the daily
  cross-sectional mean of the QLIKE differential, Newey-West lag 7.
* Negative means the row's model is better than the column's model.
* "DM vs its TTM row" compares each average with its own TTM forecast.

### TTM window: {{TSFM_START_TTM}} to 2026-10-31 ({{TSFM_DAYS_TTM}} days, {{TSFM_N_TTM}} asset-days)

{{TSFM_WIN_TTM}}

### TimesFM-2.5 window: {{TSFM_START_TIMESFM}} to 2026-10-31 ({{TSFM_DAYS_TIMESFM}} days, {{TSFM_N_TIMESFM}} asset-days)

{{TSFM_WIN_TIMESFM}}

### Chronos-2 window: {{TSFM_START_CHRONOS2}} to 2026-10-31 ({{TSFM_DAYS_CHRONOS2}} days, {{TSFM_N_CHRONOS2}} asset-days)

{{TSFM_WIN_CHRONOS2}}

### Without the 2025-10-10 crash day

This covers the two windows that contain the crash, and it is part of the declared scoring.

{{TSFM_CRASH}}

### Model confidence set (10%), pooled and per asset

The set is computed over the calibrated rows of each window: T_max statistic, 7-day blocks, 2,000 draws.
"Assets in MCS" counts the per-asset sets that contain the model.

{{TSFM_MCS}}

### The 2607.05291 claim on crypto (TTM window)

{{TSFM_CLAIM}}

### Per asset (QLIKE; the lowest per asset in bold)

TTM window:

{{TSFM_PA_TTM}}

TimesFM-2.5 window:

{{TSFM_PA_TIMESFM}}

Chronos-2 window:

{{TSFM_PA_CHRONOS2}}

### Raw against calibrated, and compute

{{TSFM_RAW}}

{{TSFM_COST}}

Total CPU time of the new models on the test period: {{TSFM_CPU_H}} CPU-hours, on 2 threads. The costs
were measured during the run, while other jobs shared the machine, so they are higher than in the
throughput check before the run (PLAN.md 13.6).

* **Reproduction check.** In the same run, every existing model's test forecasts were recomputed from scratch
  in the new environment: {{TSFM_REPRO}}.
* **Environment.** granite-tsfm declares `torch<2.12`; the project keeps torch 2.15 through a uv override, so
  the original environment is unchanged. The TTM forecasts were repeated under torch 2.11 in a separate
  environment: {{TSFM_TORCH}}.

## Compute

Kronos on the test period, measured during the run:

{{COST}}

Total Kronos CPU time on the test period: {{TOTAL_CPU_H}} CPU-hours, all on 2 threads.

Throughput measured before sizing the run:

{{THROUGHPUT}}

## Validation (all choices made here)

Kronos sampling settings. For each model and setting, the row shows the best variant and N. Settings are
compared on the same asset-days; the QLIKE is after the scale c.

* T06 means T = 0.6, top-p 0.9 (the paper's forecasting setting).
* T09 means T = 0.9, top-p 0.9 (the paper's volatility setting).
* T10 means T = 1.0, top-p 1.0 (pure sampling).

{{VAL_KRONOS}}

Variance extraction x number of paths (calibrated QLIKE, best setting):

{{VAL_VARIANT}}

HAR windows:

{{VAL_HAR}}

Student hyper-parameters (selected by QLIKE against realised RV on validation):

{{VAL_STUDENT}}

Frozen choices:

{{FROZEN}}

Test log (`results/test_log.jsonl`):

{{TESTLOG}}

## What is new here

* **A test Kronos could not have seen, with the cutoff documented.**
  * Cutoff evidence: the paper's stated data end (June 2024), plus HuggingFace commit dates showing the
    weights unchanged since 2025-06-30 / 07-01.
  * The test starts after a one-month embargo and runs 15 months.
  * A pytest asserts the ordering.
* **Volatility-forecasting evaluation of a financial foundation model.** The paper reports MAE and R^2
  against ARCH/GARCH only. Here it is scored against the HAR family with QLIKE, DM with HAC errors and a
  model confidence set.
* **Variance read from sampled candle paths, not from an averaged path.**
  * The released predictor averages its sample paths before returning them, which shrinks the variance. We
    keep every path.
  * Validation shows that the sampling temperature, not the model, sets the variance level of the paths: the
    close-to-close scale c goes from 8.5 at T = 0.6 to 0.8 at T = 1.0.
  * So a zero-shot Kronos variance needs a calibration step before it is comparable with HAR.
* **Exact, cached CPU inference.** A key/value-cache re-implementation of Kronos sampling, checked against
  the original, makes 16-path, 24-step forecasts cost about a second on 2 threads.
* **A distilled student of a few thousand parameters.** It is trained only on pre-test teacher outputs and
  scored like every other model.
* **Three newer foundation models, each scored only after its release.**
  * Weights are pinned to HuggingFace commits and checked by SHA-256. A tested rule sets each model's scoring
    start. TimesFM-2.5's re-exported weights were shown to equal the original upload.
  * This is a direct test of the TTM + log-HAR claim of 2607.05291, on an asset class and period that no
    model could have seen.

## Limits

* **One asset class, one venue.** 10 Binance USDT pairs over 15 months. The assets are highly correlated, so
  the effective sample is far smaller than {{N_TEST}} asset-days. The pooled DM test averages across assets
  each day for that reason.
* **Zero-shot only.** Kronos was not fine-tuned. The repo ships fine-tuning scripts, and a fine-tuned Kronos
  could do better.
* **Small settings search.**
  * Context is fixed at 488 hours.
  * N is at most 16 paths, and mini's validation QLIKE was still falling from N = 8 to 16.
  * The sampling grid has 3 points, searched on a validation subsample (410-610 asset-days per model).
* **The validation year may be inside Kronos-mini's training data.** Mini is not in the paper. This can only
  make mini's validation look better; the test is clean.
* **The calibration scale c is one constant per model, fitted on one year.** If the level bias of the paths
  drifts, c is stale. The raw rows show the uncalibrated forecast.
* **Kronos-base was run only on a small subset**, so its row has wide confidence bands.
* **The student's feature extraction is pandas-bound** (milliseconds). Its network forward pass is tens of
  microseconds.
* **The newer foundation models get one series each.** Here they see a single series derived from the
  candles, not the candles themselves. TTM and Chronos-2 can take several channels and covariates; that was
  not tried.
* **The newer models have shorter windows.** TimesFM-2.5 is scored on {{TSFM_DAYS_TIMESFM}} days and Chronos-2
  on {{TSFM_DAYS_CHRONOS2}}, so their tests have less power than TTM's. Chronos-2's window contains no crash
  like 2025-10-10.

## Simplifications and deviations

**Scope**
* **Crypto only.** There are no equity ETFs, which the brief made optional.
* **Kronos-base on a subset only.** It ran on BTC and ETH, every 4th test day. Its validation used every 6th
  validation day and reused Kronos-small's sampling setting, since both share a tokenizer. Variant, N and
  scale were fitted on base's own validation subset.

**Kronos validation and training data**
* **Validation on a day subsample, to fit about 3 hours of Kronos compute on 2 threads.** Every 6th
  validation day for mini (610 forecasts per setting) and every 9th for small (410). The settings grid is
  small: T/top-p in {0.6/0.9, 0.9/0.9, 1.0/1.0}. The context length is fixed at 488 hours for every size;
  Kronos-mini could take 2,048, but that was not tried.
* **Teacher on a subsample.** The distillation teacher ran on every 6th day from 2022-01 to 2024-06, plus the
  validation subsample. Those inputs predate the paper's June 2024 data end, so the teacher may be partly
  in-sample there. That is harmless for distillation: the student copies the teacher's function. The student
  is still scored only on the test period.

**Calibration and the target**
* **One pooled scale c per Kronos model**, not one per asset. c is fitted on validation as the
  QLIKE-optimal constant. Raw Kronos rows are reported too.
* **HAR lags are 1, 7 and 30 calendar days**, not Corsi's 1, 5 and 22 trading days, because crypto trades
  every day.
* **Insanity filter on HAR.** A level-HAR forecast outside the range of its own fitting window is replaced
  by that window's mean (Bollerslev, Patton & Quaedvlieg 2016). Without it, one TRX day had QLIKE above 500
  on validation.
* **GARCH(1,1)** uses a normal QMLE with a constant mean, on close-to-close daily returns at 00:00 UTC. It is
  refitted monthly and filtered daily in between.
* **The target is the 5-minute RV**, a noisy proxy of the latent variance. QLIKE and MSE still rank forecasts
  consistently under such a proxy (Patton 2011).

**Statistics and cost accounting**
* **Turbulent days are defined ex post**, as the top RV quintile per asset over the test period. This is a
  conditional evaluation, not a tradeable regime signal.
* **DM tests** use Newey-West lag 7 without the Harvey-Leybourne-Newbold small-sample correction (n = 457
  days). The pooled test averages the loss differential across assets each day. DM p-values are not adjusted
  for multiple comparisons; the MCS covers that.
* **CPU cost.** For Kronos it is the process time of the sampling call, excluding data loading. For baselines
  it is total fitting and forecasting time divided by the number of forecasts. The student is timed at batch
  size 1, forward pass and feature extraction reported separately.
* **Data repairs.** Hourly gaps (31 hours in 6.8 years) are filled flat at the last close with zero volume.
  The 7 days with fewer than 260 of 288 five-minute bars are kept and flagged.

**Newer foundation models (PLAN.md section 13)**
* **Departures from 2607.05291.**
  * The data are crypto, which trades 24/7, so days are calendar days. There are 10 assets and one horizon (1
    day).
  * A validation scale c is applied on top of the raw forecast; the raw rows are reported too.
  * Forecasts are not winsorised to their past range.
  * The model confidence set covers the 6 reference models plus the new rows, not the paper's 17 models.
  * Chronos-2 is not in that paper. It gets the 1,000-day context the paper gives its other models.
* **TTM checkpoint.** We use the `512-96-ft-r2.1` branch (frequency-tuned, r2.1 release), not the repo's main
  branch (512-96-r2), because it is the checkpoint of 2607.05291 and it takes hourly and daily data. Its context
  is fixed at 512 steps, so with hourly input it sees 24 more candles than Kronos.
* **TimesFM-2.5's mean head** is the point forecast, not the median the library returns as its "point"
  output.
* **Only the validation-chosen hourly series was generated on test.** Test forecasts of each model start at
  its scoring start; nothing earlier was generated.

**Order of runs.** The Kronos test-period forecasts were generated (with the frozen Kronos settings) before
the student was fitted and the full config frozen. The student reads no test-period file, and all scoring
happened once, after `freeze`. See PLAN.md, "Changes during the build".

## How to run

```bash
git clone https://github.com/mohymabrouk/kronos-vs-har-after-cutoff.git
cd kronos-vs-har-after-cutoff
export PYTHONIOENCODING=utf-8 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
uv sync
uv run python -m kvh.cli download          # Binance 1h + 5m monthly klines, checksums verified (~8 min)
uv run python -m kvh.cli build             # parquet: hourly candles, daily 5-minute RV
uv run python -m kvh.cli throughput        # seconds per forecast per Kronos size on 2 threads
uv run python -m kvh.cli kronos --period val --models mini --tag T06,T09,T10 --stride 6
uv run python -m kvh.cli kronos --period val --models small --tag T06,T09,T10 --stride 9
uv run python -m kvh.cli validate          # choose Kronos setting/variant/N/scale and HAR windows
uv run python -m kvh.cli kronos --period distill --models small --stride 6     # teacher, pre-validation
uv run python -m kvh.cli kronos --period val --models base --symbols BTCUSDT,ETHUSDT --tag <small tag> --stride 6
uv run python -m kvh.cli validate          # adds Kronos-base (mini/small choices unchanged)
uv run python -m kvh.cli distill           # student: choose hyper-parameters on validation, refit pre-test
uv run python -m kvh.cli freeze            # results/frozen_config.json + sha256
uv run python -m kvh.cli kronos --period test --models mini      # frozen settings only (no --tag allowed)
uv run python -m kvh.cli kronos --period test --models small
uv run python -m kvh.cli kronos --period test --models base --symbols BTCUSDT,ETHUSDT --stride 4
uv run python -m kvh.cli test              # scores once; a rerun needs --force and is logged
# newer foundation models (PLAN.md section 13), each scored only after its release
uv run python -m kvh.cli tsfm --period val           # TTM, TimesFM-2.5, Chronos-2: every validation day, both inputs
uv run python -m kvh.cli tsfm-validate               # scale c and hourly series -> results/frozen_tsfm.json + sha256
uv run python -m kvh.cli tsfm --period test          # frozen settings, from each model's scoring start
uv run python -m kvh.cli tsfm-test                   # scores once (an added, forced run in the test log) + reproduction check
uv run python -m kvh.cli report            # tables.md, figures, this README
uv run pytest                              # 9 groups of tests (the TSFM weight tests skip without local weights)
```

The Kronos jobs write one parquet file per (model, setting, asset, month) under
`~/.cache/quant-research/p3/kronos/`, so an interrupted run resumes where it stopped; the TSFM jobs do the
same under `~/.cache/quant-research/p3/tsfm/`. The model code is vendored in `third_party/kronos/` (MIT
licence, commit 67b630e). All weights are pinned to HuggingFace commits in `src/kvh/config.py` and downloaded
to `~/.cache/quant-research/p3/hf/`, outside the repo. `scripts/ttm_torch_check.py` repeats the TTM forecasts
under torch 2.11 (the range granite-tsfm declares) in a separate environment.

## Related work (2026)

* **Brini, "Forecasting Realized Volatility with Time Series Foundation Models: A Comparison with Econometric
  Benchmarks", [arXiv 2607.05291](https://arxiv.org/abs/2607.05291)** (July 2026, revised November 2026).
  * Setup: nine zero-shot foundation models against eight econometric models, on realised volatility of 50
    VOLARE assets (equities, currencies, futures), 1, 5 and 22 days ahead.
  * Findings: the foundation models do not beat the econometric models as a class. Only TTM, the smallest,
    beats log-HAR in average QLIKE, narrowly, at every horizon. The equal-weight TTM + log-HAR average matches
    the best single model and is in the model confidence set for nearly every asset.
  * The section "Newer foundation models" above tests that claim on crypto, after TTM's weights date.
* **Upadhyay, "Do Time-Series Foundation Models Forecast Index-Futures Returns? A Preregistered Evaluation of
  Models, Metrics, and Input Representations", [arXiv 2611.12141](https://arxiv.org/abs/2611.12141)**
  (November 2026). On E-mini Nasdaq-100 and S&P 500 futures, 5 to 120 minutes ahead, 0 of 160
  model-representation-horizon-market cells beat a no-change forecast on a blind set opened once. Kronos is
  evaluated in separate experiments.
* **Bulut, "Conflicting History in Foundation Model Return Forecasts",
  [arXiv 2610.25822](https://arxiv.org/abs/2610.25822)** (October 2026). Older observations can teach a
  foundation model a predictive relationship but mislead it once the relationship changes. In simulations
  this "conflicting history" lowers the chance of beating a zero forecast; on 56 assets, neither 256 nor 64
  returns of context beat zero.
* **Noguer i Alonso and Pereira Franklin, "Pretrained Time-Series Foundation Models for Financial Return
  Forecasting", [arXiv 2606.27100](https://arxiv.org/abs/2606.27100)** (June 2026). TimeGPT,
  TimesFM-2.5, Moirai-2.0, Chronos and Chronos-2 against neural models trained from scratch, on five US
  stocks. Pretrained models win most rankings, but their gains over a random walk are small and sparse.
* **Shi et al., "Kronos: A Foundation Model for the Language of Financial Markets",
  [arXiv 2508.02739](https://arxiv.org/abs/2508.02739)** (August 2025, AAAI 2026). The model of the
  original study.
