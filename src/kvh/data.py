"""Build tidy parquet files from the raw monthly kline zips.

* hourly/<SYM>.parquet : full hourly grid (open time, UTC) with open, high, low, close, volume, amount
                          (amount = quote-asset volume). Missing hours are filled flat at the last close with
                          zero volume and flagged (`filled`).
* daily/<SYM>.parquet  : one row per UTC day: rv (sum of squared 5-minute log returns over [00:00, 24:00)),
                          n5 (number of 5-minute bars), close (last 5-minute close of the day), ret (daily log
                          return, close to close), pk (daily Parkinson variance from the 5m high/low).

Binance spot files switched from millisecond to microsecond timestamps on 2025-01-01; both are handled.
"""

import io
import logging
import zipfile

import numpy as np
import pandas as pd

from kvh import config
from kvh.download import months, rel_path

log = logging.getLogger(__name__)
COLS = ["open_time", "open", "high", "low", "close", "volume", "close_time", "amount", "count", "tb_base", "tb_quote", "ignore"]


def read_zip(path) -> pd.DataFrame:
    with zipfile.ZipFile(path) as z:
        name = z.namelist()[0]
        raw = z.read(name)
    df = pd.read_csv(io.BytesIO(raw), header=None)
    if not np.issubdtype(df.dtypes.iloc[0], np.number):  # some files carry a header row
        df = pd.read_csv(io.BytesIO(raw))
    df = df.iloc[:, :12]
    df.columns = COLS
    t = df["open_time"].astype("int64")
    t = np.where(t > 10 ** 14, t // 1000, t)  # microseconds -> milliseconds
    df["open_time"] = pd.to_datetime(t, unit="ms", utc=True)
    return df[["open_time", "open", "high", "low", "close", "volume", "amount"]]


def load_interval(symbol: str, interval: str) -> pd.DataFrame:
    parts = []
    for y, m in months(config.FIRST_MONTH, config.LAST_MONTH):
        p = config.RAW / rel_path(symbol, interval, y, m)
        if p.exists():
            parts.append(read_zip(p))
    df = pd.concat(parts).drop_duplicates("open_time").sort_values("open_time").set_index("open_time")
    return df.astype("float64")


def build_hourly(symbol: str) -> pd.DataFrame:
    df = load_interval(symbol, "1h")
    grid = pd.date_range(df.index[0], df.index[-1], freq="1h", tz="UTC")
    out = df.reindex(grid)
    filled = out["close"].isna()
    out["close"] = out["close"].ffill()
    for c in ("open", "high", "low"):
        out[c] = out[c].fillna(out["close"])
    out[["volume", "amount"]] = out[["volume", "amount"]].fillna(0.0)
    out["filled"] = filled
    out.index.name = "open_time"
    return out


def daily_from_5m(df5: pd.DataFrame) -> pd.DataFrame:
    """Realised variance per UTC day from 5-minute bars (pure function, unit tested)."""
    df5 = df5.sort_index()
    day = df5.index.floor("D")
    lc = np.log(df5["close"].to_numpy())
    lo = np.log(df5["open"].to_numpy())
    r = np.empty_like(lc)
    r[1:] = lc[1:] - lc[:-1]
    first = np.r_[True, day[1:] != day[:-1]]
    r[first] = lc[first] - lo[first]  # first bar of each day: open (= 00:00 price) to close
    hl = (np.log(df5["high"].to_numpy()) - np.log(df5["low"].to_numpy())) ** 2 / (4 * np.log(2))
    g = pd.DataFrame({"r2": r ** 2, "hl": hl, "close": df5["close"].to_numpy(), "open": df5["open"].to_numpy()}, index=day)
    agg = g.groupby(level=0).agg(rv=("r2", "sum"), n5=("r2", "size"), close=("close", "last"), open=("open", "first"))
    agg["pk"] = g["hl"].groupby(level=0).sum()
    agg["ret"] = np.log(agg["close"]).diff()
    first_ret = np.log(agg["close"].iloc[0] / agg["open"].iloc[0])
    agg.iloc[0, agg.columns.get_loc("ret")] = first_ret
    agg.index = agg.index.tz_localize(None).date if agg.index.tz is not None else agg.index.date
    agg.index.name = "date"
    return agg.drop(columns=["open"])


def build_daily(symbol: str) -> pd.DataFrame:
    df5 = load_interval(symbol, "5m")
    d = daily_from_5m(df5)
    # drop incomplete first day (listing) and flag thin days
    d = d.iloc[1:]
    thin = d["n5"] < config.MIN_5M_BARS
    if thin.any():
        log.info("%s: %d days with < %d five-minute bars (kept, flagged): %s", symbol, int(thin.sum()), config.MIN_5M_BARS,
                 [str(x) for x in d.index[thin][:10]])
    d["thin"] = thin
    # a day with zero variance (exchange halt) would break logs: absolute floor (no look-ahead); 0 days hit it
    floor = 1e-10
    n_floor = int((d["rv"] < floor).sum())
    if n_floor:
        log.info("%s: %d days floored at %g", symbol, n_floor, floor)
    d["rv"] = d["rv"].clip(lower=floor)
    return d


def build_all() -> None:
    (config.DATA / "hourly").mkdir(parents=True, exist_ok=True)
    (config.DATA / "daily").mkdir(parents=True, exist_ok=True)
    rows = []
    for s in config.SYMBOLS:
        h = build_hourly(s)
        h.to_parquet(config.DATA / "hourly" / f"{s}.parquet")
        d = build_daily(s)
        d.to_parquet(config.DATA / "daily" / f"{s}.parquet")
        rows.append(dict(symbol=s, hourly_first=str(h.index[0]), hourly_last=str(h.index[-1]), hours=len(h),
                         hours_filled=int(h["filled"].sum()), daily_first=str(d.index[0]), daily_last=str(d.index[-1]),
                         days=len(d), thin_days=int(d["thin"].sum()), median_rv_pct2=float(d["rv"].median() * config.VAR_SCALE)))
        log.info("built %s: %s", s, rows[-1])
    config.RESULTS.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(config.RESULTS / "data_summary.csv", index=False)


def load_hourly(symbol: str) -> pd.DataFrame:
    return pd.read_parquet(config.DATA / "hourly" / f"{symbol}.parquet")


def load_daily(symbol: str) -> pd.DataFrame:
    d = pd.read_parquet(config.DATA / "daily" / f"{symbol}.parquet")
    d.index = pd.to_datetime(d.index).date
    d.index.name = "date"
    return d
