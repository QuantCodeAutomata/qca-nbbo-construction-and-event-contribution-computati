"""
exp_8_robustness.py - Experiment 8: Robustness checks.

Three independent robustness tests verify that the H1 OFI price-impact
finding is not an artefact of specific methodological choices.

Test 1 – Timescale robustness
  Rebuild the OFI grid with alternative bin widths
  {5, 10, 20, 30, 60} seconds and re-estimate the OFI regression.
  Result: β and R² should be stable across scales.

Test 2 – NYSE-only exchange filter
  Rebuild the NBBO using only NYSE ('N') quotes and re-run the analysis.
  Compares to the multi-exchange baseline.

Test 3 – Tick test vs quote test trade classification
  Re-classify trades using the Lee-Ready tick test instead of the
  quote test and rebuild the grid.  Compares R² and β.

Outputs
-------
- ``{results_dir}/robustness_timescale.csv``   – β and R² across bin widths.
- ``{results_dir}/robustness_exchange.csv``    – multi-exchange vs NYSE-only.
- ``{results_dir}/robustness_ticktest.csv``    – quote test vs tick test.

Usage
-----
    python exp/exp_8_robustness.py
    python exp/exp_8_robustness.py --tickers_subset SLB AMD BAC COP AXP \\
                                   --nbbo_dir results/nbbo \\
                                   --grid_dir results/grid \\
                                   --trade_dir data/raw \\
                                   --data_dir data/raw \\
                                   --results_dir results
"""

import argparse
import os
import sys
import warnings

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.nbbo import (
    apply_quote_filters,
    build_nbbo_sequence,
    apply_spread_filter,
    compute_event_contribution,
)
from src.grid import (
    apply_trade_filters,
    classify_trades_quote_test,
    classify_trades_tick_test,
    build_10sec_grid,
    SESSION_SECONDS,
)
from src.regression import run_ofi_regression, summarize_subsamples
from data.synthetic_data import TICKERS_50, TRADING_DAYS_APRIL_2010


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

TIMESCALES = [5, 10, 20, 30, 60]  # seconds


# ---------------------------------------------------------------------------
# Custom grid builder (variable bin width)
# ---------------------------------------------------------------------------


def _build_grid_custom_bins(
    nbbo_df: pd.DataFrame,
    trades_df: pd.DataFrame,
    date: str,
    bin_width: int,
) -> pd.DataFrame:
    """Build a price-impact grid with a custom bin width.

    Mirrors the logic of :func:`src.grid.build_10sec_grid` but accepts
    an arbitrary *bin_width* in seconds.  All returned column names are
    identical to those of :func:`src.grid.build_10sec_grid`.

    Parameters
    ----------
    nbbo_df : pd.DataFrame
        NBBO events with ``timestamp``, ``nbbo_bid``, ``nbbo_ask``, ``e_n``.
    trades_df : pd.DataFrame
        Trade data with ``timestamp``, ``size``, ``trade_dir``.
    date : str
        Calendar date ``YYYY-MM-DD``.
    bin_width : int
        Bin duration in seconds (e.g., 5, 10, 20, 30, 60).

    Returns
    -------
    pd.DataFrame
        Columns: ``bin_index``, ``bin_start``, ``half_hour``,
        ``OFI_k``, ``mid_k``, ``delta_P_k``, ``TI_k``, ``VOL_k``.
    """
    session_start = pd.Timestamp(f"{date} 09:30:00")
    n_bins = int(SESSION_SECONDS // bin_width)

    # --- NBBO side ---
    nbbo = nbbo_df.copy()
    nbbo["timestamp"] = pd.to_datetime(nbbo["timestamp"])
    nbbo["mid"] = (nbbo["nbbo_bid"] + nbbo["nbbo_ask"]) / 2.0
    elapsed_n = (nbbo["timestamp"] - session_start).dt.total_seconds().to_numpy(float)
    nbbo["_bin"] = np.clip((elapsed_n // bin_width).astype(int), 0, n_bins - 1)

    ofi_per_bin = (
        nbbo.groupby("_bin")["e_n"]
        .apply(lambda x: np.nansum(x.to_numpy(dtype=float)))
        .rename("OFI_k")
    )
    last_mid_per_bin = nbbo.groupby("_bin")["mid"].last().rename("mid_last")

    # --- Trade side ---
    trades = trades_df.copy()
    trades["timestamp"] = pd.to_datetime(trades["timestamp"])
    elapsed_t = (trades["timestamp"] - session_start).dt.total_seconds().to_numpy(float)
    trades["_bin"] = np.clip((elapsed_t // bin_width).astype(int), 0, n_bins - 1)
    trades["buy_vol"] = np.where(
        trades["trade_dir"] == 1, trades["size"].astype(float), 0.0
    )
    trades["sell_vol"] = np.where(
        trades["trade_dir"] == -1, trades["size"].astype(float), 0.0
    )

    buy_sum = trades.groupby("_bin")["buy_vol"].sum()
    sell_sum = trades.groupby("_bin")["sell_vol"].sum()
    ti_per_bin = (buy_sum - sell_sum).rename("TI_k")
    vol_per_bin = (buy_sum + sell_sum).rename("VOL_k")

    # --- Assemble ---
    bin_index = np.arange(n_bins)
    grid = pd.DataFrame({"bin_index": bin_index})
    grid = grid.join(ofi_per_bin, on="bin_index")
    grid = grid.join(last_mid_per_bin, on="bin_index")
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

    # Half-hour slot: how many bins per 30 minutes
    bins_per_30min = max(1, int(1800 / bin_width))
    grid["half_hour"] = (grid["bin_index"] // bins_per_30min).clip(0, 12).astype(int)

    grid["bin_start"] = [
        session_start + pd.Timedelta(seconds=int(k) * bin_width)
        for k in grid["bin_index"]
    ]

    return grid[
        ["bin_index", "bin_start", "half_hour",
         "OFI_k", "mid_k", "delta_P_k", "TI_k", "VOL_k"]
    ].reset_index(drop=True)


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------


def _load_nbbo(ticker: str, date: str, nbbo_dir: str) -> pd.DataFrame | None:
    path = os.path.join(nbbo_dir, f"{ticker}_{date}_nbbo.parquet")
    if not os.path.exists(path):
        return None
    df = pd.read_parquet(path)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df


def _load_grid(ticker: str, date: str, grid_dir: str) -> pd.DataFrame | None:
    path = os.path.join(grid_dir, f"{ticker}_{date}_grid.parquet")
    if not os.path.exists(path):
        return None
    return pd.read_parquet(path)


def _load_quotes_raw(ticker: str, date: str, data_dir: str) -> pd.DataFrame | None:
    path = os.path.join(data_dir, "quotes", f"{ticker}_{date}_quotes.parquet")
    if not os.path.exists(path):
        return None
    df = pd.read_parquet(path)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df


def _load_trades_raw(ticker: str, date: str, data_dir: str) -> pd.DataFrame | None:
    path = os.path.join(data_dir, "trades", f"{ticker}_{date}_trades.parquet")
    if not os.path.exists(path):
        return None
    df = pd.read_parquet(path)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df


def _run_regression_safe(grid: pd.DataFrame) -> dict | None:
    """Run OFI regression; return None on failure."""
    subdf = grid[["OFI_k", "delta_P_k"]].dropna(subset=["delta_P_k"])
    if len(subdf) < 10:
        return None
    try:
        return run_ofi_regression(subdf)
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Test 1: Timescale robustness
# ---------------------------------------------------------------------------


def _test_timescale(
    tickers: list[str],
    dates: list[str],
    nbbo_dir: str,
    grid_dir: str,
    data_dir: str,
) -> pd.DataFrame:
    """Re-build grid at multiple timescales and compare regression results."""
    rows: list[dict] = []

    for ticker in tickers:
        for date in dates:
            nbbo_df = _load_nbbo(ticker, date, nbbo_dir)
            if nbbo_df is None:
                continue

            # We need classified trades to rebuild the grid
            trades_raw = _load_trades_raw(ticker, date, data_dir)
            if trades_raw is None:
                continue
            trades_filt = apply_trade_filters(trades_raw)
            if trades_filt.empty:
                continue
            try:
                classified = classify_trades_quote_test(trades_filt, nbbo_df)
            except Exception:
                continue

            for bw in TIMESCALES:
                try:
                    grid = _build_grid_custom_bins(nbbo_df, classified, date, bw)
                    res = _run_regression_safe(grid)
                    if res is None:
                        continue
                    rows.append(
                        {
                            "ticker": ticker,
                            "date": date,
                            "bin_width_sec": bw,
                            "n_bins": int(SESSION_SECONDS // bw),
                            "beta_ofi": res["beta_ofi"],
                            "t_beta_ofi": res["t_beta_ofi"],
                            "rsquared": res["rsquared"],
                            "nobs": res["nobs"],
                        }
                    )
                except Exception as exc:
                    warnings.warn(f"Timescale {bw}s {ticker} {date}: {exc}")

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Test 2: NYSE-only exchange filter
# ---------------------------------------------------------------------------


def _test_exchange(
    tickers: list[str],
    dates: list[str],
    nbbo_dir: str,
    data_dir: str,
) -> pd.DataFrame:
    """Compare full-NBBO vs NYSE-only NBBO regression."""
    rows: list[dict] = []

    for ticker in tickers:
        for date in dates:
            quotes_raw = _load_quotes_raw(ticker, date, data_dir)
            trades_raw = _load_trades_raw(ticker, date, data_dir)
            if quotes_raw is None or trades_raw is None:
                continue

            trades_filt = apply_trade_filters(trades_raw)
            if trades_filt.empty:
                continue

            # --- Baseline: full NBBO (from exp1) ---
            nbbo_full = _load_nbbo(ticker, date, nbbo_dir)
            if nbbo_full is not None:
                try:
                    classified_full = classify_trades_quote_test(trades_filt, nbbo_full)
                    grid_full = build_10sec_grid(nbbo_full, classified_full, date)
                    res_full = _run_regression_safe(grid_full)
                    if res_full:
                        rows.append(
                            {
                                "ticker": ticker,
                                "date": date,
                                "exchange_filter": "all",
                                "beta_ofi": res_full["beta_ofi"],
                                "t_beta_ofi": res_full["t_beta_ofi"],
                                "rsquared": res_full["rsquared"],
                                "nobs": res_full["nobs"],
                            }
                        )
                except Exception as exc:
                    warnings.warn(f"Full-NBBO {ticker} {date}: {exc}")

            # --- NYSE-only NBBO ---
            try:
                quotes_nyse = quotes_raw[quotes_raw["ex"] == "N"].copy()
                if len(quotes_nyse) < 10:
                    continue
                filt_nyse = apply_quote_filters(quotes_nyse)
                if filt_nyse.empty:
                    continue
                nbbo_nyse = build_nbbo_sequence(filt_nyse)
                nbbo_nyse = apply_spread_filter(nbbo_nyse)
                nbbo_nyse = compute_event_contribution(nbbo_nyse)

                classified_nyse = classify_trades_quote_test(trades_filt, nbbo_nyse)
                grid_nyse = build_10sec_grid(nbbo_nyse, classified_nyse, date)
                res_nyse = _run_regression_safe(grid_nyse)
                if res_nyse:
                    rows.append(
                        {
                            "ticker": ticker,
                            "date": date,
                            "exchange_filter": "NYSE_only",
                            "beta_ofi": res_nyse["beta_ofi"],
                            "t_beta_ofi": res_nyse["t_beta_ofi"],
                            "rsquared": res_nyse["rsquared"],
                            "nobs": res_nyse["nobs"],
                        }
                    )
            except Exception as exc:
                warnings.warn(f"NYSE-only {ticker} {date}: {exc}")

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Test 3: Tick test vs quote test
# ---------------------------------------------------------------------------


def _test_ticktest(
    tickers: list[str],
    dates: list[str],
    nbbo_dir: str,
    grid_dir: str,
    data_dir: str,
) -> pd.DataFrame:
    """Compare quote-test vs tick-test trade classification."""
    rows: list[dict] = []

    for ticker in tickers:
        for date in dates:
            nbbo_df = _load_nbbo(ticker, date, nbbo_dir)
            if nbbo_df is None:
                continue

            trades_raw = _load_trades_raw(ticker, date, data_dir)
            if trades_raw is None:
                continue
            trades_filt = apply_trade_filters(trades_raw)
            if trades_filt.empty:
                continue

            # --- Baseline: quote test (use precomputed grid) ---
            grid_qt = _load_grid(ticker, date, grid_dir)
            if grid_qt is not None:
                res_qt = _run_regression_safe(grid_qt)
                if res_qt:
                    rows.append(
                        {
                            "ticker": ticker,
                            "date": date,
                            "classification": "quote_test",
                            "beta_ofi": res_qt["beta_ofi"],
                            "t_beta_ofi": res_qt["t_beta_ofi"],
                            "rsquared": res_qt["rsquared"],
                            "nobs": res_qt["nobs"],
                        }
                    )

            # --- Tick test classification ---
            try:
                tick_classified = classify_trades_tick_test(trades_filt)
                # Rename trade_dir_tick → trade_dir so build_10sec_grid accepts it
                tick_classified = tick_classified.rename(
                    columns={"trade_dir_tick": "trade_dir"}
                )
                grid_tt = build_10sec_grid(nbbo_df, tick_classified, date)
                res_tt = _run_regression_safe(grid_tt)
                if res_tt:
                    rows.append(
                        {
                            "ticker": ticker,
                            "date": date,
                            "classification": "tick_test",
                            "beta_ofi": res_tt["beta_ofi"],
                            "t_beta_ofi": res_tt["t_beta_ofi"],
                            "rsquared": res_tt["rsquared"],
                            "nobs": res_tt["nobs"],
                        }
                    )
            except Exception as exc:
                warnings.warn(f"Tick-test {ticker} {date}: {exc}")

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Core pipeline
# ---------------------------------------------------------------------------


def run_exp8(
    tickers_subset: list[str],
    dates: list[str],
    nbbo_dir: str,
    grid_dir: str,
    trade_dir: str,
    results_dir: str,
) -> dict:
    """Run all three robustness checks.

    Parameters
    ----------
    tickers_subset : list[str]
        Small subset of tickers (≤ 5 for speed).
    dates : list[str]
        Trading dates in ``YYYY-MM-DD`` format.
    nbbo_dir : str
        Directory with NBBO parquets from Exp 1.
    grid_dir : str
        Directory with grid parquets from Exp 2.
    trade_dir : str
        Root data directory containing a ``trades/`` sub-directory.
        Also used as *data_dir* (which must also have a ``quotes/`` sub-dir).
    results_dir : str
        Output directory for robustness tables.

    Returns
    -------
    dict
        Keys: ``'timescale'``, ``'exchange'``, ``'ticktest'`` (DataFrames).
    """
    os.makedirs(results_dir, exist_ok=True)

    n = len(tickers_subset)
    print(f"  Robustness subset: {tickers_subset} ({n} stocks × {len(dates)} dates)")

    # -------------------------------------------------------------------------
    # Test 1: Timescale
    # -------------------------------------------------------------------------
    print("\n--- Test 1: Timescale robustness ---")
    ts_df = _test_timescale(
        tickers_subset, dates, nbbo_dir, grid_dir, trade_dir
    )
    if not ts_df.empty:
        ts_df.to_csv(
            os.path.join(results_dir, "robustness_timescale.csv"), index=False
        )
        print(f"  Saved: {results_dir}/robustness_timescale.csv")

        # Summary by bin width
        ts_summary = (
            ts_df.groupby("bin_width_sec")[["beta_ofi", "t_beta_ofi", "rsquared"]]
            .mean()
            .round(6)
        )
        print("\n  Timescale summary (mean across stock-days):")
        print(ts_summary.to_string())
    else:
        print("  WARNING: No timescale results produced.")

    # -------------------------------------------------------------------------
    # Test 2: Exchange filter
    # -------------------------------------------------------------------------
    print("\n--- Test 2: NYSE-only exchange filter ---")
    ex_df = _test_exchange(tickers_subset, dates, nbbo_dir, trade_dir)
    if not ex_df.empty:
        ex_df.to_csv(
            os.path.join(results_dir, "robustness_exchange.csv"), index=False
        )
        print(f"  Saved: {results_dir}/robustness_exchange.csv")

        ex_summary = (
            ex_df.groupby("exchange_filter")[["beta_ofi", "t_beta_ofi", "rsquared"]]
            .mean()
            .round(6)
        )
        print("\n  Exchange summary (mean across stock-days):")
        print(ex_summary.to_string())
    else:
        print("  WARNING: No exchange robustness results produced.")

    # -------------------------------------------------------------------------
    # Test 3: Trade classification
    # -------------------------------------------------------------------------
    print("\n--- Test 3: Tick test vs quote test ---")
    tt_df = _test_ticktest(tickers_subset, dates, nbbo_dir, grid_dir, trade_dir)
    if not tt_df.empty:
        tt_df.to_csv(
            os.path.join(results_dir, "robustness_ticktest.csv"), index=False
        )
        print(f"  Saved: {results_dir}/robustness_ticktest.csv")

        tt_summary = (
            tt_df.groupby("classification")[["beta_ofi", "t_beta_ofi", "rsquared"]]
            .mean()
            .round(6)
        )
        print("\n  Tick-test summary (mean across stock-days):")
        print(tt_summary.to_string())
    else:
        print("  WARNING: No tick-test results produced.")

    # -------------------------------------------------------------------------
    # Console summary
    # -------------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("Exp 8 — Robustness Checks Summary")
    print("=" * 60)
    if not ts_df.empty:
        for bw in TIMESCALES:
            subset = ts_df[ts_df["bin_width_sec"] == bw]
            if subset.empty:
                continue
            print(
                f"  Timescale {bw:3d}s: "
                f"β = {subset['beta_ofi'].mean():.6f}, "
                f"R² = {subset['rsquared'].mean():.4f}, "
                f"|t| = {subset['t_beta_ofi'].abs().mean():.2f}"
            )
    if not ex_df.empty:
        for exf in ["all", "NYSE_only"]:
            subset = ex_df[ex_df["exchange_filter"] == exf]
            if subset.empty:
                continue
            print(
                f"  Exchange [{exf:10s}]: "
                f"β = {subset['beta_ofi'].mean():.6f}, "
                f"R² = {subset['rsquared'].mean():.4f}"
            )
    if not tt_df.empty:
        for cls in ["quote_test", "tick_test"]:
            subset = tt_df[tt_df["classification"] == cls]
            if subset.empty:
                continue
            print(
                f"  Classify [{cls:11s}]: "
                f"β = {subset['beta_ofi'].mean():.6f}, "
                f"R² = {subset['rsquared'].mean():.4f}"
            )
    print("=" * 60 + "\n")

    return {"timescale": ts_df, "exchange": ex_df, "ticktest": tt_df}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Exp 8: Robustness checks (timescale, exchange, tick-test)."
    )
    p.add_argument(
        "--tickers_subset",
        nargs="+",
        default=None,
        help="Tickers to use (default: first 5 from TICKERS_50).",
    )
    p.add_argument("--nbbo_dir", default="results/nbbo")
    p.add_argument("--grid_dir", default="results/grid")
    p.add_argument("--trade_dir", default="data/raw")
    p.add_argument("--results_dir", default="results")
    return p


def main() -> None:
    parser = _build_arg_parser()
    args = parser.parse_args()

    tickers_subset = (
        args.tickers_subset if args.tickers_subset else TICKERS_50[:5]
    )
    dates = TRADING_DAYS_APRIL_2010

    print(
        f"Exp 8: Robustness checks for {tickers_subset} × "
        f"{len(dates)} dates …\n"
    )
    run_exp8(
        tickers_subset=tickers_subset,
        dates=dates,
        nbbo_dir=args.nbbo_dir,
        grid_dir=args.grid_dir,
        trade_dir=args.trade_dir,
        results_dir=args.results_dir,
    )


if __name__ == "__main__":
    main()
