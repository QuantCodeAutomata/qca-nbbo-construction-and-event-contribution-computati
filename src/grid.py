"""
grid.py - 10-second grid aggregation for OFI, price-impact, and trade-imbalance.

Builds a uniform 2340-bin (10-second) grid over the regular trading session
09:30–16:00 and computes per-bin order-flow imbalance (OFI), price change
(delta_P), trade imbalance (TI), and volume (VOL).  Also provides trade
classification via quote test and tick test, and half-hourly average depth.

References:
    Cont, Kukanov & Stoikov (2014). The price impact of order book events.
    Lee & Ready (1991). Inferring trade direction from intraday data.
"""

import os

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.regression.linear_model import OLS


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MARKET_OPEN_NS = pd.Timestamp("1970-01-01 09:30:00")
MARKET_CLOSE_NS = pd.Timestamp("1970-01-01 16:00:00")
SESSION_SECONDS = int((MARKET_CLOSE_NS - MARKET_OPEN_NS).total_seconds())  # 23400
BIN_WIDTH = 10  # seconds
N_BINS = SESSION_SECONDS // BIN_WIDTH  # 2340

MARKET_OPEN_TIME = pd.Timestamp("09:30:00").time()
MARKET_CLOSE_TIME = pd.Timestamp("16:00:00").time()

EXCLUDED_TRADE_CONDITIONS = {"O", "Z", "B", "T", "L", "G", "W", "J", "K"}

HALF_HOUR_LABELS = [
    "09:30", "10:00", "10:30", "11:00", "11:30", "12:00",
    "12:30", "13:00", "13:30", "14:00", "14:30", "15:00", "15:30",
]  # 13 half-hour slots (indices 0–12)


# ---------------------------------------------------------------------------
# Trade filtering
# ---------------------------------------------------------------------------


def apply_trade_filters(df: pd.DataFrame) -> pd.DataFrame:
    """Filter TAQ trade records to retain only clean, reportable trades.

    Applies the following filters:
      1. Time window: [09:30, 16:00] inclusive on the time-of-day component.
      2. Price strictly positive.
      3. Size strictly positive.
      4. Correction indicator ``corr`` ≤ 2 (regular and minor-correction prints).
      5. Trade condition code not in
         ``{'O', 'Z', 'B', 'T', 'L', 'G', 'W', 'J', 'K'}``.

    Parameters
    ----------
    df : pd.DataFrame
        Raw TAQ trade data.  Expected columns:
          - ``timestamp`` : datetime64[ns] – trade timestamp
          - ``price``     : float          – execution price
          - ``size``      : float/int      – shares traded
          - ``corr``      : int            – TAQ correction indicator
          - ``cond``      : str            – trade condition code

    Returns
    -------
    pd.DataFrame
        Filtered copy of *df*, sorted by ``timestamp``, with the original
        index reset.

    Raises
    ------
    ValueError
        If any required column is missing.
    """
    required = {"timestamp", "price", "size", "corr", "cond"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"apply_trade_filters: missing columns {missing}")

    ts = pd.to_datetime(df["timestamp"])
    time_of_day = ts.dt.time

    mask = (
        (time_of_day >= MARKET_OPEN_TIME)
        & (time_of_day <= MARKET_CLOSE_TIME)
        & (df["price"] > 0)
        & (df["size"] > 0)
        & (df["corr"] <= 2)
        & (~df["cond"].isin(EXCLUDED_TRADE_CONDITIONS))
    )

    out = df.loc[mask].copy()
    out["timestamp"] = pd.to_datetime(out["timestamp"])
    out = out.sort_values("timestamp").reset_index(drop=True)
    return out


# ---------------------------------------------------------------------------
# Trade classification – quote test
# ---------------------------------------------------------------------------


def classify_trades_quote_test(
    trades_df: pd.DataFrame,
    nbbo_df: pd.DataFrame,
) -> pd.DataFrame:
    """Classify each trade as buy, sell, or unclassified using the quote test.

    Uses :func:`pandas.merge_asof` (Context7) with ``direction='backward'``
    and a ``tolerance`` of 1 second to align each trade to the most recent
    NBBO snapshot.

    Classification rules:
      - ``trade_dir = 1`` (buy)  if ``price ≥ nbbo_ask``
      - ``trade_dir = -1`` (sell) if ``price ≤ nbbo_bid``
      - ``trade_dir = 0`` (unclassified) otherwise

    Parameters
    ----------
    trades_df : pd.DataFrame
        Filtered trade data (output of :func:`apply_trade_filters`).
        Required columns: ``timestamp``, ``price``, ``size``.
    nbbo_df : pd.DataFrame
        NBBO sequence.  Required columns: ``timestamp``, ``nbbo_bid``,
        ``nbbo_ask``.

    Returns
    -------
    pd.DataFrame
        *trades_df* augmented with columns ``nbbo_bid``, ``nbbo_ask``, and
        ``trade_dir`` (int: 1 buy, -1 sell, 0 unclassified).

    Raises
    ------
    ValueError
        If required columns are absent.
    """
    req_t = {"timestamp", "price", "size"}
    req_n = {"timestamp", "nbbo_bid", "nbbo_ask"}
    missing_t = req_t - set(trades_df.columns)
    missing_n = req_n - set(nbbo_df.columns)
    if missing_t:
        raise ValueError(f"classify_trades_quote_test: trades missing {missing_t}")
    if missing_n:
        raise ValueError(f"classify_trades_quote_test: nbbo missing {missing_n}")

    trades_sorted = trades_df.sort_values("timestamp").copy()
    nbbo_sorted = nbbo_df[["timestamp", "nbbo_bid", "nbbo_ask"]].sort_values(
        "timestamp"
    ).copy()

    # Context7: pandas.merge_asof for backward time-join with tolerance
    merged = pd.merge_asof(
        trades_sorted,
        nbbo_sorted,
        on="timestamp",
        direction="backward",
        tolerance=pd.Timedelta("1s"),
    )

    price = merged["price"].to_numpy(dtype=float)
    size = merged["size"].to_numpy(dtype=float)
    nbbo_ask = merged["nbbo_ask"].to_numpy(dtype=float)
    nbbo_bid = merged["nbbo_bid"].to_numpy(dtype=float)

    # NaN NBBO after merge → unclassified (no matching quote within tolerance)
    has_nbbo = ~(np.isnan(nbbo_ask) | np.isnan(nbbo_bid))

    is_buy = has_nbbo & (price >= nbbo_ask)
    is_sell = has_nbbo & (price <= nbbo_bid)

    trade_dir = np.where(is_buy, 1, np.where(is_sell, -1, 0))
    merged["trade_dir"] = trade_dir
    merged["b_n"] = np.where(is_buy, size, 0.0)
    merged["s_n"] = np.where(is_sell, size, 0.0)
    return merged.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Trade classification – tick test
# ---------------------------------------------------------------------------


def classify_trades_tick_test(trades_df: pd.DataFrame) -> pd.DataFrame:
    """Classify each trade using the Lee–Ready tick test.

    Algorithm:
      1. Compare each trade price to the immediately preceding trade price.
      2. Uptick  (price > prev price)  → buy  (+1).
      3. Downtick (price < prev price) → sell (−1).
      4. Zero-tick (price == prev price) → carry forward the direction of
         the most recent non-zero tick.  If no prior non-zero tick exists
         the direction is set to 0.

    Parameters
    ----------
    trades_df : pd.DataFrame
        Filtered, timestamp-sorted trade data.  Required column: ``price``.

    Returns
    -------
    pd.DataFrame
        *trades_df* with added columns ``trade_dir_tick`` (int: 1 buy,
        -1 sell, 0 unclassified), ``b_n`` (buy volume, 0 otherwise), and
        ``s_n`` (sell volume, 0 otherwise).

    Raises
    ------
    ValueError
        If the ``price`` column is absent.
    """
    if "price" not in trades_df.columns:
        raise ValueError("classify_trades_tick_test: 'price' column not found")

    if trades_df.empty:
        out = trades_df.copy()
        out["trade_dir_tick"] = pd.array([], dtype="int64")
        out["b_n"] = pd.array([], dtype=float)
        out["s_n"] = pd.array([], dtype=float)
        return out

    sort_col = "timestamp" if "timestamp" in trades_df.columns else trades_df.columns[0]
    out = trades_df.sort_values(sort_col).copy().reset_index(drop=True)
    prices = out["price"].to_numpy(dtype=float)
    n = len(prices)
    dirs = np.zeros(n, dtype=int)

    last_nonzero = 0
    for i in range(1, n):
        if prices[i] > prices[i - 1]:
            dirs[i] = 1
            last_nonzero = 1
        elif prices[i] < prices[i - 1]:
            dirs[i] = -1
            last_nonzero = -1
        else:
            dirs[i] = last_nonzero  # zero-tick: carry forward

    sizes = out["size"].to_numpy(dtype=float) if "size" in out.columns else np.ones(n)
    out["trade_dir_tick"] = dirs
    out["b_n"] = np.where(dirs == 1, sizes, 0.0)
    out["s_n"] = np.where(dirs == -1, sizes, 0.0)
    return out


# ---------------------------------------------------------------------------
# 10-second grid construction
# ---------------------------------------------------------------------------


def _timestamp_to_bin(ts_series: pd.Series, date: str) -> np.ndarray:
    """Map timestamps to 10-second bin indices relative to 09:30 on *date*.

    Parameters
    ----------
    ts_series : pd.Series
        Datetime64 timestamps.
    date : str
        Calendar date string ``YYYY-MM-DD``.

    Returns
    -------
    np.ndarray
        Integer bin indices in [0, N_BINS).  Values outside the session are
        clipped to this range.
    """
    session_start = pd.Timestamp(f"{date} 09:30:00")
    elapsed = (ts_series - session_start).dt.total_seconds().to_numpy(dtype=float)
    bins = (elapsed // BIN_WIDTH).astype(int)
    return np.clip(bins, 0, N_BINS - 1)


def build_10sec_grid(
    nbbo_df: pd.DataFrame,
    trades_df: pd.DataFrame,
    date: str,
) -> pd.DataFrame:
    """Construct a 2340-bin, 10-second grid for one stock-day.

    Each bin ``k`` covers ``[09:30 + k*10s, 09:30 + (k+1)*10s)``.  The
    following quantities are computed per bin:

    - ``OFI_k``    = Σ e_n for NBBO events in bin k (NaN e_n values skipped).
    - ``mid_k``    = last NBBO midpoint in bin k (or carry-forward from prior
                     bin if the bin contains no NBBO update).
    - ``delta_P_k``= (mid_k − mid_{k−1}) / 0.01  (price change in tick units).
    - ``TI_k``     = Σ buy_volume − Σ sell_volume for trades in bin k.
    - ``VOL_k``    = Σ buy_volume + Σ sell_volume for trades in bin k.
    - ``half_hour``= half-hour slot index 0–12.

    ``delta_P_k`` is NaN for k = 0 (no prior midpoint).  The grid always
    contains exactly 2340 rows.

    Parameters
    ----------
    nbbo_df : pd.DataFrame
        NBBO sequence for the stock-day.  Required columns:
        ``timestamp``, ``nbbo_bid``, ``nbbo_ask``, ``e_n``.
    trades_df : pd.DataFrame
        Trade data with quote-test classification.  Required columns:
        ``timestamp``, ``size``, ``trade_dir``.
    date : str
        Calendar date ``YYYY-MM-DD``.

    Returns
    -------
    pd.DataFrame
        2340 rows × columns: ``bin_k``, ``bin_start``, ``half_hour``,
        ``OFI_k``, ``mid_k``, ``delta_P_k``, ``TI_k``, ``VOL_k``.

    Raises
    ------
    ValueError
        If required columns are absent.
    """
    req_n = {"timestamp", "nbbo_bid", "nbbo_ask", "e_n"}
    # Accept either (b_n, s_n) pair or trade_dir + size
    has_bn_sn = {"b_n", "s_n"}.issubset(set(trades_df.columns))
    has_trade_dir = {"timestamp", "size", "trade_dir"}.issubset(set(trades_df.columns))
    missing_n = req_n - set(nbbo_df.columns)
    if missing_n:
        raise ValueError(f"build_10sec_grid: nbbo missing {missing_n}")
    if not trades_df.empty and not has_bn_sn and not has_trade_dir:
        raise ValueError(
            "build_10sec_grid: trades must have either (b_n, s_n) or "
            "(size, trade_dir) columns"
        )

    session_start = pd.Timestamp(f"{date} 09:30:00")

    # --- NBBO side ---
    nbbo_s = nbbo_df.copy()
    nbbo_s["timestamp"] = pd.to_datetime(nbbo_s["timestamp"])
    nbbo_s["mid"] = (nbbo_s["nbbo_bid"] + nbbo_s["nbbo_ask"]) / 2.0
    nbbo_s["_bin"] = _timestamp_to_bin(nbbo_s["timestamp"], date)

    # OFI per bin: sum of e_n (ignoring NaN)
    ofi_per_bin = (
        nbbo_s.groupby("_bin")["e_n"]
        .apply(lambda x: np.nansum(x.to_numpy(dtype=float)))
        .rename("OFI_k")
    )

    # Last mid per bin (for delta_P computation)
    last_mid_per_bin = nbbo_s.groupby("_bin")["mid"].last().rename("mid_last")

    # --- Trade side ---
    trades_s = trades_df.copy()
    if not trades_s.empty:
        trades_s["timestamp"] = pd.to_datetime(trades_s["timestamp"])
        trades_s["_bin"] = _timestamp_to_bin(trades_s["timestamp"], date)
        if has_bn_sn:
            # Paper notation: b_n = buy volume per trade, s_n = sell volume
            trades_s["buy_vol"] = trades_s["b_n"].astype(float)
            trades_s["sell_vol"] = trades_s["s_n"].astype(float)
        else:
            vol = trades_s["size"].astype(float)
            trades_s["buy_vol"] = np.where(trades_s["trade_dir"] == 1, vol, 0.0)
            trades_s["sell_vol"] = np.where(trades_s["trade_dir"] == -1, vol, 0.0)

        ti_per_bin = (
            trades_s.groupby("_bin")
            .apply(lambda g: g["buy_vol"].sum() - g["sell_vol"].sum())
            .rename("TI_k")
        )
        vol_per_bin = (
            trades_s.groupby("_bin")
            .apply(lambda g: g["buy_vol"].sum() + g["sell_vol"].sum())
            .rename("VOL_k")
        )
    else:
        ti_per_bin = pd.Series(name="TI_k", dtype=float)
        vol_per_bin = pd.Series(name="VOL_k", dtype=float)

    # --- Assemble full grid ---
    bin_k = np.arange(N_BINS)
    grid = pd.DataFrame({"bin_k": bin_k})
    grid = grid.join(ofi_per_bin.rename_axis("bin_k"), on="bin_k")
    grid = grid.join(last_mid_per_bin.rename_axis("bin_k"), on="bin_k")
    grid = grid.join(ti_per_bin.rename_axis("bin_k") if not ti_per_bin.empty else ti_per_bin, on="bin_k")
    grid = grid.join(vol_per_bin.rename_axis("bin_k") if not vol_per_bin.empty else vol_per_bin, on="bin_k")

    # Fill bins with no NBBO events with 0 OFI
    grid["OFI_k"] = grid["OFI_k"].fillna(0.0)
    # Carry forward mid price for bins with no NBBO update
    grid["mid_k"] = grid["mid_last"].ffill()
    grid.drop(columns=["mid_last"], inplace=True)
    # Fill trade-side zeros for bins with no trades
    grid["TI_k"] = grid["TI_k"].fillna(0.0)
    grid["VOL_k"] = grid["VOL_k"].fillna(0.0)

    # delta_P_k = (mid_k - mid_{k-1}) / 0.01
    mid_vals = grid["mid_k"].to_numpy(dtype=float)
    delta_p = np.empty(N_BINS, dtype=float)
    delta_p[0] = np.nan
    delta_p[1:] = (mid_vals[1:] - mid_vals[:-1]) / 0.01
    grid["delta_P_k"] = delta_p

    # Half-hour slot: 13 slots of 180 bins each (180 × 10s = 30 min)
    grid["half_hour"] = (grid["bin_k"] // 180).clip(0, 12).astype(int)

    # Bin start timestamps
    grid["bin_start"] = grid["bin_k"].apply(
        lambda k: session_start + pd.Timedelta(seconds=k * BIN_WIDTH)
    )

    return grid[
        ["bin_k", "bin_start", "half_hour", "OFI_k", "mid_k", "delta_P_k",
         "TI_k", "VOL_k"]
    ].reset_index(drop=True)


# ---------------------------------------------------------------------------
# Average depth
# ---------------------------------------------------------------------------


def compute_average_depth(nbbo_df: pd.DataFrame) -> pd.DataFrame:
    """Compute time-weighted average quoted depth for each half-hour interval.

    For each half-hour slot i (0–12, corresponding to 09:30–10:00, …,
    15:30–16:00), computes:

        AD_i = Σ(bid_size_n + ask_size_n) / (2 × (N_i − 1))

    where N_i is the number of NBBO records in slot i and the sum runs over
    all N_i records.  Slots with fewer than 2 records receive NaN.

    Parameters
    ----------
    nbbo_df : pd.DataFrame
        NBBO sequence with columns ``timestamp``, ``nbbo_bid_size``,
        ``nbbo_ask_size``.

    Returns
    -------
    pd.DataFrame
        13 rows × columns: ``half_hour`` (0–12), ``AD_i``.

    Raises
    ------
    ValueError
        If required columns are absent.
    """
    required = {"timestamp", "nbbo_bid_size", "nbbo_ask_size"}
    missing = required - set(nbbo_df.columns)
    if missing:
        raise ValueError(f"compute_average_depth: missing columns {missing}")

    # Empty input → return empty frame (no slots computable)
    if nbbo_df.empty:
        return pd.DataFrame({"half_hour": pd.Series([], dtype=int), "AD_i": pd.Series([], dtype=float)})

    out = nbbo_df.copy()
    out["timestamp"] = pd.to_datetime(out["timestamp"])

    # Half-hour slot based on minutes past 09:30
    session_start = out["timestamp"].dt.normalize() + pd.Timedelta("9h30min")
    elapsed_min = (out["timestamp"] - session_start).dt.total_seconds() / 60.0
    out["half_hour"] = (elapsed_min // 30).clip(0, 12).astype(int)

    def _ad(g: pd.DataFrame) -> float:
        n = len(g)
        if n < 2:
            return np.nan
        total = (g["nbbo_bid_size"] + g["nbbo_ask_size"]).sum()
        return float(total / (2.0 * (n - 1)))

    result = (
        out.groupby("half_hour")
        .apply(_ad, include_groups=False)
        .reset_index()
    )
    result.columns = ["half_hour", "AD_i"]

    # Ensure all 13 slots are present
    full = pd.DataFrame({"half_hour": list(range(13))})
    result = full.merge(result, on="half_hour", how="left")
    return result


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------


def save_grid_data(
    grid_df: pd.DataFrame,
    ticker: str,
    output_dir: str,
) -> str:
    """Save a grid DataFrame to a Parquet file.

    The file is written to ``{output_dir}/{ticker}_grid.parquet``.

    Parameters
    ----------
    grid_df : pd.DataFrame
        Grid data to persist (typically the output of
        :func:`build_10sec_grid`).
    ticker : str
        Stock ticker symbol used to name the output file.
    output_dir : str
        Directory path where the file will be written.  Created if absent.

    Returns
    -------
    str
        Absolute path of the written file.

    Raises
    ------
    ValueError
        If *grid_df* is empty, *ticker* is blank, or *output_dir* is blank.
    OSError
        If the directory cannot be created or the file cannot be written.
    """
    if grid_df.empty:
        raise ValueError("save_grid_data: grid_df is empty")
    if not ticker:
        raise ValueError("save_grid_data: ticker must be a non-empty string")
    if not output_dir:
        raise ValueError("save_grid_data: output_dir must be a non-empty string")

    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, f"{ticker}_grid.parquet")
    grid_df.to_parquet(path, index=False)
    return os.path.abspath(path)
