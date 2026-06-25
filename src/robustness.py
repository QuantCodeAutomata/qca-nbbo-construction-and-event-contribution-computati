"""
robustness.py - Robustness tests for NBBO price-impact analysis.

Provides alternative grid constructions (event-based, arbitrary time frequency),
timescale sensitivity analysis, exchange-scope comparison (NBBO vs NYSE-only),
and trade-classification comparison (quote test vs tick test).

References:
    Cont, Kukanov & Stoikov (2014). The price impact of order book events.
    Lee & Ready (1991). Inferring trade direction from intraday data.
"""

import warnings

import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from statsmodels.regression.linear_model import OLS
from scipy.stats import pearsonr

from .nbbo import (
    apply_quote_filters,
    build_nbbo_sequence,
    apply_spread_filter,
    compute_event_contribution,
)
from .regression import (
    run_ols_hc0,
    run_ofi_regression,
    run_ti_regression,
)


# ---------------------------------------------------------------------------
# Event-based grid
# ---------------------------------------------------------------------------


def build_event_grid(
    nbbo_df: pd.DataFrame,
    window_size: int,
) -> pd.DataFrame:
    """Group consecutive NBBO events into non-overlapping windows of fixed length.

    Each window of ``window_size`` consecutive events is treated as a single
    observation with:
      - ``OFI_k``     = Σ e_n across the window (NaN values ignored).
      - ``mid_start`` = midpoint at the first event of the window.
      - ``mid_end``   = midpoint at the last event of the window.
      - ``delta_P_k`` = (mid_end − mid_start) / 0.01.
      - ``window``    = sequential window index.

    The last partial window (if ``len(nbbo_df) % window_size ≠ 0``) is
    discarded.

    Parameters
    ----------
    nbbo_df : pd.DataFrame
        NBBO sequence with columns ``timestamp``, ``nbbo_bid``, ``nbbo_ask``,
        ``e_n``.
    window_size : int
        Number of consecutive NBBO events per window (must be ≥ 2).

    Returns
    -------
    pd.DataFrame
        Columns: ``window``, ``timestamp_start``, ``timestamp_end``,
        ``OFI_k``, ``mid_start``, ``mid_end``, ``delta_P_k``.

    Raises
    ------
    ValueError
        If required columns are absent, *window_size* < 2, or *nbbo_df*
        contains fewer than *window_size* rows.
    """
    required = {"timestamp", "nbbo_bid", "nbbo_ask", "e_n"}
    missing = required - set(nbbo_df.columns)
    if missing:
        raise ValueError(f"build_event_grid: missing columns {missing}")
    if window_size < 2:
        raise ValueError("build_event_grid: window_size must be ≥ 2")
    if len(nbbo_df) < window_size:
        raise ValueError(
            f"build_event_grid: nbbo_df has {len(nbbo_df)} rows, "
            f"need at least {window_size}"
        )

    work = nbbo_df.copy().reset_index(drop=True)
    work["timestamp"] = pd.to_datetime(work["timestamp"])
    work["mid"] = (work["nbbo_bid"] + work["nbbo_ask"]) / 2.0

    n_windows = len(work) // window_size
    records = []
    for w in range(n_windows):
        lo = w * window_size
        hi = lo + window_size
        chunk = work.iloc[lo:hi]
        e_vals = chunk["e_n"].to_numpy(dtype=float)
        ofi = float(np.nansum(e_vals))
        mid_start = float(chunk["mid"].iloc[0])
        mid_end = float(chunk["mid"].iloc[-1])
        delta_p = (mid_end - mid_start) / 0.01
        records.append(
            {
                "window": w,
                "timestamp_start": chunk["timestamp"].iloc[0],
                "timestamp_end": chunk["timestamp"].iloc[-1],
                "OFI_k": ofi,
                "mid_start": mid_start,
                "mid_end": mid_end,
                "delta_P_k": delta_p,
            }
        )

    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# Arbitrary time-frequency grid
# ---------------------------------------------------------------------------


def build_time_grid(
    nbbo_df: pd.DataFrame,
    trades_df: pd.DataFrame,
    freq_seconds: int,
) -> pd.DataFrame:
    """Resample NBBO and trade data onto a uniform time grid of arbitrary frequency.

    Creates time bins of width *freq_seconds* starting at 09:30 and ending
    at 16:00.  For each bin:
      - ``OFI_k``     = Σ e_n for NBBO events in the bin.
      - ``mid_k``     = last NBBO midpoint in the bin (carry-forward if empty).
      - ``delta_P_k`` = (mid_k − mid_{k−1}) / 0.01 (NaN for first bin).
      - ``TI_k``      = Σ buy_volume − Σ sell_volume in the bin.
      - ``VOL_k``     = Σ buy_volume + Σ sell_volume in the bin.

    Parameters
    ----------
    nbbo_df : pd.DataFrame
        NBBO sequence with columns ``timestamp``, ``nbbo_bid``, ``nbbo_ask``,
        ``e_n``.
    trades_df : pd.DataFrame
        Trade data with columns ``timestamp``, ``size``, ``trade_dir``.
    freq_seconds : int
        Bin width in seconds (must be a positive divisor or multiple of
        session length 23400 s).

    Returns
    -------
    pd.DataFrame
        Columns: ``bin_index``, ``bin_start``, ``OFI_k``, ``mid_k``,
        ``delta_P_k``, ``TI_k``, ``VOL_k``.

    Raises
    ------
    ValueError
        If required columns are absent or *freq_seconds* is not positive.
    """
    req_n = {"timestamp", "nbbo_bid", "nbbo_ask", "e_n"}
    req_t = {"timestamp", "size", "trade_dir"}
    missing_n = req_n - set(nbbo_df.columns)
    missing_t = req_t - set(trades_df.columns)
    if missing_n:
        raise ValueError(f"build_time_grid: nbbo missing {missing_n}")
    if missing_t:
        raise ValueError(f"build_time_grid: trades missing {missing_t}")
    if freq_seconds <= 0:
        raise ValueError("build_time_grid: freq_seconds must be positive")

    SESSION_SECONDS = 23400
    n_bins = int(np.ceil(SESSION_SECONDS / freq_seconds))

    nbbo_s = nbbo_df.copy()
    nbbo_s["timestamp"] = pd.to_datetime(nbbo_s["timestamp"])
    nbbo_s["mid"] = (nbbo_s["nbbo_bid"] + nbbo_s["nbbo_ask"]) / 2.0

    # Infer date from first timestamp
    first_ts = nbbo_s["timestamp"].iloc[0]
    session_start = first_ts.normalize() + pd.Timedelta("9h30min")

    def _to_bin(ts_series: pd.Series) -> np.ndarray:
        elapsed = (ts_series - session_start).dt.total_seconds().to_numpy(float)
        return np.clip((elapsed // freq_seconds).astype(int), 0, n_bins - 1)

    nbbo_s["_bin"] = _to_bin(nbbo_s["timestamp"])

    ofi_per_bin = (
        nbbo_s.groupby("_bin")["e_n"]
        .apply(lambda x: np.nansum(x.to_numpy(dtype=float)))
        .rename("OFI_k")
    )
    last_mid = nbbo_s.groupby("_bin")["mid"].last().rename("mid_last")

    trades_s = trades_df.copy()
    trades_s["timestamp"] = pd.to_datetime(trades_s["timestamp"])
    trades_s["_bin"] = _to_bin(trades_s["timestamp"])
    trades_s["buy_vol"] = np.where(
        trades_s["trade_dir"] == 1, trades_s["size"].astype(float), 0.0
    )
    trades_s["sell_vol"] = np.where(
        trades_s["trade_dir"] == -1, trades_s["size"].astype(float), 0.0
    )
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

    grid = pd.DataFrame({"bin_index": np.arange(n_bins)})
    grid = grid.join(ofi_per_bin, on="bin_index")
    grid = grid.join(last_mid, on="bin_index")
    grid = grid.join(ti_per_bin, on="bin_index")
    grid = grid.join(vol_per_bin, on="bin_index")

    grid["OFI_k"] = grid["OFI_k"].fillna(0.0)
    grid["mid_k"] = grid["mid_last"].ffill()
    grid.drop(columns=["mid_last"], inplace=True)
    grid["TI_k"] = grid["TI_k"].fillna(0.0)
    grid["VOL_k"] = grid["VOL_k"].fillna(0.0)

    mid_vals = grid["mid_k"].to_numpy(dtype=float)
    delta_p = np.empty(n_bins, dtype=float)
    delta_p[0] = np.nan
    delta_p[1:] = (mid_vals[1:] - mid_vals[:-1]) / 0.01
    grid["delta_P_k"] = delta_p

    grid["bin_start"] = grid["bin_index"].apply(
        lambda k: session_start + pd.Timedelta(seconds=k * freq_seconds)
    )

    return grid[
        ["bin_index", "bin_start", "OFI_k", "mid_k", "delta_P_k", "TI_k", "VOL_k"]
    ].reset_index(drop=True)


# ---------------------------------------------------------------------------
# Timescale robustness
# ---------------------------------------------------------------------------


def run_timescale_robustness(
    nbbo_df: pd.DataFrame,
    trades_df: pd.DataFrame,
    timescales: dict,
) -> dict:
    """Run OFI price-impact regressions across multiple timescales.

    Parameters
    ----------
    nbbo_df : pd.DataFrame
        NBBO sequence with columns ``timestamp``, ``nbbo_bid``, ``nbbo_ask``,
        ``e_n``.
    trades_df : pd.DataFrame
        Classified trade data with columns ``timestamp``, ``size``,
        ``trade_dir``.
    timescales : dict
        Mapping ``label -> (type, value)`` where:
          - ``type = 'time'``  → use :func:`build_time_grid` with
            ``freq_seconds = value``
          - ``type = 'event'`` → use :func:`build_event_grid` with
            ``window_size = value``

    Returns
    -------
    dict
        Mapping ``label -> regression_result_dict`` (from
        :func:`~regression.run_ofi_regression`), plus a ``'grid_size'`` key
        recording the number of bins/windows produced.

    Raises
    ------
    ValueError
        If *timescales* is empty or contains an unrecognised type string.
    """
    if not timescales:
        raise ValueError("run_timescale_robustness: timescales dict is empty")

    results: dict = {}
    for label, (kind, value) in timescales.items():
        if kind == "time":
            grid = build_time_grid(nbbo_df, trades_df, freq_seconds=int(value))
        elif kind == "event":
            grid = build_event_grid(nbbo_df, window_size=int(value))
        else:
            raise ValueError(
                f"run_timescale_robustness: unknown timescale type '{kind}'. "
                "Expected 'time' or 'event'."
            )

        try:
            reg = run_ofi_regression(grid)
        except ValueError as exc:
            warnings.warn(
                f"run_timescale_robustness: regression failed for '{label}': {exc}"
            )
            reg = {"beta_ofi": np.nan, "rsquared": np.nan, "nobs": 0}

        reg["grid_size"] = len(grid)
        results[label] = reg

    return results


# ---------------------------------------------------------------------------
# NYSE-only OFI
# ---------------------------------------------------------------------------


def compute_nyse_only_ofi(quotes_df: pd.DataFrame) -> pd.DataFrame:
    """Build a NYSE-only OFI sequence from raw TAQ quote data.

    Filters the quote DataFrame to NYSE records only (``ex == 'N'``), then
    applies the full NBBO pipeline (filter → build NBBO → spread filter →
    event contribution).  Because only one exchange is present the NBBO
    collapses to the NYSE best bid/offer, but the construction logic remains
    identical to the multi-exchange case for consistency.

    Parameters
    ----------
    quotes_df : pd.DataFrame
        Raw TAQ quotes.  Required columns: same as
        :func:`~nbbo.apply_quote_filters`, plus ``ex``.

    Returns
    -------
    pd.DataFrame
        NBBO-style DataFrame restricted to NYSE quotes, with the same
        columns as the output of :func:`~nbbo.compute_event_contribution`.

    Raises
    ------
    ValueError
        If ``ex`` column is absent, or no NYSE quotes survive filtering.
    """
    if "ex" not in quotes_df.columns:
        raise ValueError("compute_nyse_only_ofi: 'ex' column not found")

    nyse_quotes = quotes_df.loc[quotes_df["ex"] == "N"].copy()
    if nyse_quotes.empty:
        raise ValueError(
            "compute_nyse_only_ofi: no NYSE quotes (ex=='N') found in quotes_df"
        )

    filtered = apply_quote_filters(nyse_quotes)
    if filtered.empty:
        raise ValueError(
            "compute_nyse_only_ofi: no valid NYSE quotes after filtering"
        )

    nbbo = build_nbbo_sequence(filtered)
    nbbo = apply_spread_filter(nbbo)
    nbbo = compute_event_contribution(nbbo)
    return nbbo


# ---------------------------------------------------------------------------
# Exchange robustness comparison
# ---------------------------------------------------------------------------


def run_exchange_robustness(
    nbbo_df: pd.DataFrame,
    nyse_ofi_df: pd.DataFrame,
) -> dict:
    """Compare OFI price-impact regressions: full NBBO vs NYSE-only.

    Aligns the two OFI sequences to a common 10-second time grid, sums e_n
    within each bin, and runs :func:`~regression.run_ofi_regression` on
    each.  Returns both individual results and a Pearson correlation between
    the two OFI series.

    Parameters
    ----------
    nbbo_df : pd.DataFrame
        Full NBBO sequence (all exchanges) with columns ``timestamp``,
        ``nbbo_bid``, ``nbbo_ask``, ``e_n``.
    nyse_ofi_df : pd.DataFrame
        NYSE-only NBBO sequence with the same column structure.

    Returns
    -------
    dict
        Keys:
          - ``'nbbo_result'``   : dict from :func:`~regression.run_ofi_regression`
          - ``'nyse_result'``   : dict from :func:`~regression.run_ofi_regression`
          - ``'ofi_correlation'``: Pearson correlation between the two OFI series

    Raises
    ------
    ValueError
        If required columns are absent.
    """
    required = {"timestamp", "nbbo_bid", "nbbo_ask", "e_n"}
    for name, df in [("nbbo_df", nbbo_df), ("nyse_ofi_df", nyse_ofi_df)]:
        missing = required - set(df.columns)
        if missing:
            raise ValueError(
                f"run_exchange_robustness: {name} missing columns {missing}"
            )

    BIN_SECONDS = 10
    SESSION_SECONDS = 23400
    N_BINS = SESSION_SECONDS // BIN_SECONDS

    def _bin_ofi(df: pd.DataFrame) -> np.ndarray:
        df = df.copy()
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        first_ts = df["timestamp"].iloc[0]
        session_start = first_ts.normalize() + pd.Timedelta("9h30min")
        elapsed = (df["timestamp"] - session_start).dt.total_seconds().to_numpy(float)
        bin_idx = np.clip((elapsed // BIN_SECONDS).astype(int), 0, N_BINS - 1)
        ofi_bins = np.zeros(N_BINS, dtype=float)
        for i, b in enumerate(bin_idx):
            v = df["e_n"].iloc[i]
            if np.isfinite(v):
                ofi_bins[b] += v
        # Midpoint for delta_P
        df["mid"] = (df["nbbo_bid"] + df["nbbo_ask"]) / 2.0
        return ofi_bins, df, bin_idx, session_start

    def _build_mini_grid(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        first_ts = df["timestamp"].iloc[0]
        session_start = first_ts.normalize() + pd.Timedelta("9h30min")
        elapsed = (df["timestamp"] - session_start).dt.total_seconds().to_numpy(float)
        bin_idx = np.clip((elapsed // BIN_SECONDS).astype(int), 0, N_BINS - 1)
        df["_bin"] = bin_idx
        df["mid"] = (df["nbbo_bid"] + df["nbbo_ask"]) / 2.0

        ofi = (
            df.groupby("_bin")["e_n"]
            .apply(lambda x: np.nansum(x.to_numpy(dtype=float)))
            .reindex(range(N_BINS), fill_value=0.0)
            .rename("OFI_k")
        )
        last_mid = (
            df.groupby("_bin")["mid"].last()
            .reindex(range(N_BINS))
            .ffill()
            .rename("mid_k")
        )

        grid = pd.DataFrame({"OFI_k": ofi, "mid_k": last_mid})
        mid_vals = grid["mid_k"].to_numpy(dtype=float)
        dp = np.empty(N_BINS, dtype=float)
        dp[0] = np.nan
        dp[1:] = (mid_vals[1:] - mid_vals[:-1]) / 0.01
        grid["delta_P_k"] = dp
        return grid.reset_index(drop=True)

    grid_nbbo = _build_mini_grid(nbbo_df)
    grid_nyse = _build_mini_grid(nyse_ofi_df)

    nbbo_result = run_ofi_regression(grid_nbbo)
    nyse_result = run_ofi_regression(grid_nyse)

    # Pearson correlation between the two OFI bin series
    ofi_nbbo = grid_nbbo["OFI_k"].to_numpy(dtype=float)
    ofi_nyse = grid_nyse["OFI_k"].to_numpy(dtype=float)
    valid = np.isfinite(ofi_nbbo) & np.isfinite(ofi_nyse)
    if valid.sum() >= 2:
        corr, _ = pearsonr(ofi_nbbo[valid], ofi_nyse[valid])
    else:
        corr = np.nan

    return {
        "nbbo_result": nbbo_result,
        "nyse_result": nyse_result,
        "ofi_correlation": float(corr),
    }


# ---------------------------------------------------------------------------
# Quote-test vs tick-test robustness
# ---------------------------------------------------------------------------


def run_ticktest_robustness(
    grid_df_quote: pd.DataFrame,
    grid_df_tick: pd.DataFrame,
) -> dict:
    """Compare TI regressions from quote-test vs tick-test trade classification.

    Runs :func:`~regression.run_ti_regression` on both grid DataFrames and
    computes the Pearson correlation between the two ``TI_k`` series.

    Parameters
    ----------
    grid_df_quote : pd.DataFrame
        10-second grid built using quote-test trade classification.  Must
        contain ``delta_P_k`` and ``TI_k``.
    grid_df_tick : pd.DataFrame
        10-second grid built using tick-test trade classification.  Must
        contain ``delta_P_k`` and ``TI_k``.

    Returns
    -------
    dict
        Keys:
          - ``'quote_test_result'`` : dict from :func:`~regression.run_ti_regression`
          - ``'tick_test_result'``  : dict from :func:`~regression.run_ti_regression`
          - ``'ti_correlation'``    : Pearson correlation of the TI_k series

    Raises
    ------
    ValueError
        If required columns are absent from either DataFrame.
    """
    required = {"delta_P_k", "TI_k"}
    for name, df in [("grid_df_quote", grid_df_quote), ("grid_df_tick", grid_df_tick)]:
        missing = required - set(df.columns)
        if missing:
            raise ValueError(
                f"run_ticktest_robustness: {name} missing columns {missing}"
            )

    quote_result = run_ti_regression(grid_df_quote)
    tick_result = run_ti_regression(grid_df_tick)

    ti_quote = grid_df_quote["TI_k"].to_numpy(dtype=float)
    ti_tick = grid_df_tick["TI_k"].to_numpy(dtype=float)

    min_len = min(len(ti_quote), len(ti_tick))
    ti_quote = ti_quote[:min_len]
    ti_tick = ti_tick[:min_len]
    valid = np.isfinite(ti_quote) & np.isfinite(ti_tick)
    if valid.sum() >= 2:
        corr, _ = pearsonr(ti_quote[valid], ti_tick[valid])
    else:
        corr = np.nan

    return {
        "quote_test_result": quote_result,
        "tick_test_result": tick_result,
        "ti_correlation": float(corr),
    }
