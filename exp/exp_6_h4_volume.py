"""
exp_6_h4_volume.py - Experiment 6: H4 – Volume scaling of price impact.

Tests **Hypothesis 4**: the magnitude of price changes scales with trading
volume according to a power law with exponent H (analogous to the Kyle 1985
λ but estimated from data).

Per-stock estimation
--------------------
Via :func:`src.regression.run_volume_regression` which executes:

    Stage 1: log|Δp_k| = const + H · log(VOL_k) + ε   (H = Hurst exponent)
    Stage 2: |Δp_k|    = c₀   + c · VOL_k^H   + ε

Three absolute-value regressions (per stock-day)
-------------------------------------------------
1. |Δp_k| = α + β · |OFI_k| + ε    (via :func:`src.regression.run_abs_ofi_regression`)
2. |Δp_k| = α + γ · |TI_k|  + ε    (custom HC0)
3. |Δp_k| = α + c · VOL_k^Ĥ + ε    (Stage 2 from volume regression)

Outputs
-------
- ``{results_dir}/table5_volume.csv``  – per-subsample results.
- ``{results_dir}/table5_summary.csv`` – mean coefficients and R² across subsamples.

Usage
-----
    python exp/exp_6_h4_volume.py
    python exp/exp_6_h4_volume.py --n_stocks 5 --grid_dir results/grid \\
                                   --results_dir results
"""

import argparse
import os
import sys
import warnings

import numpy as np
import pandas as pd
import statsmodels.api as sm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.regression import (
    run_abs_ofi_regression,
    run_volume_regression,
    run_ols_hc0,
    summarize_subsamples,
)
from data.synthetic_data import TICKERS_50, TRADING_DAYS_APRIL_2010


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _load_grid(ticker: str, date: str, grid_dir: str) -> pd.DataFrame | None:
    path = os.path.join(grid_dir, f"{ticker}_{date}_grid.parquet")
    if not os.path.exists(path):
        return None
    return pd.read_parquet(path)


def _run_abs_ti_regression(subdf: pd.DataFrame) -> dict:
    """Regress |Δp_k| on |TI_k| using HC0 standard errors."""
    work = subdf[["delta_P_k", "TI_k"]].copy().replace(
        [np.inf, -np.inf], np.nan
    ).dropna()
    if len(work) < 5:
        raise ValueError("_run_abs_ti_regression: insufficient observations")

    work["abs_dP"] = np.abs(work["delta_P_k"])
    work["abs_ti"] = np.abs(work["TI_k"])
    y = work["abs_dP"].to_numpy(dtype=float)
    X = sm.add_constant(
        work["abs_ti"].to_numpy(dtype=float).reshape(-1, 1), has_constant="add"
    )
    res = run_ols_hc0(y, X)
    return {
        "alpha": float(res["params"][0]),
        "beta_abs_ti": float(res["params"][1]),
        "t_alpha": float(res["tvalues"][0]),
        "t_beta_abs_ti": float(res["tvalues"][1]),
        "rsquared": float(res["rsquared"]),
        "fvalue": float(res["fvalue"]),
        "nobs": res["nobs"],
    }


# ---------------------------------------------------------------------------
# Core pipeline
# ---------------------------------------------------------------------------


def run_exp6(
    tickers: list[str],
    dates: list[str],
    grid_dir: str,
    results_dir: str,
) -> dict:
    """Estimate H and run absolute-value regressions for all stock-days.

    Parameters
    ----------
    tickers : list[str]
        Ticker symbols to process.
    dates : list[str]
        Trading dates in ``YYYY-MM-DD`` format.
    grid_dir : str
        Directory containing grid parquets from Exp 2.
    results_dir : str
        Output directory for tables.

    Returns
    -------
    dict
        Keys: ``'abs_ofi_summary'``, ``'abs_ti_summary'``,
        ``'volume_summary'``, ``'n_subsamples'``.
    """
    os.makedirs(results_dir, exist_ok=True)

    abs_ofi_results: list[dict] = []
    abs_ti_results: list[dict] = []
    vol_results: list[dict] = []
    table5_rows: list[dict] = []

    total = len(tickers) * len(dates)
    done = 0

    for ticker in tickers:
        for date in dates:
            done += 1
            pct = 100 * done / total

            grid = _load_grid(ticker, date, grid_dir)
            if grid is None or grid.empty:
                continue

            req_cols = {"OFI_k", "delta_P_k", "TI_k", "VOL_k"}
            if not req_cols.issubset(grid.columns):
                continue

            subdf = grid[list(req_cols)].dropna(subset=["delta_P_k"])
            if len(subdf) < 10:
                continue

            row_base = {"ticker": ticker, "date": date}

            # Regression 1: |Δp| ~ |OFI|
            try:
                res_aofi = run_abs_ofi_regression(subdf)
                abs_ofi_results.append(res_aofi)
                table5_rows.append(
                    {
                        **row_base,
                        "regression": "abs_OFI",
                        "beta": res_aofi["beta_abs_ofi"],
                        "t_beta": res_aofi["t_beta_abs_ofi"],
                        "rsquared": res_aofi["rsquared"],
                        "nobs": res_aofi["nobs"],
                        "H": float("nan"),
                    }
                )
            except (ValueError, Exception) as exc:
                warnings.warn(f"|OFI| regression failed {ticker} {date}: {exc}")

            # Regression 2: |Δp| ~ |TI|
            try:
                res_ati = _run_abs_ti_regression(subdf)
                abs_ti_results.append(res_ati)
                table5_rows.append(
                    {
                        **row_base,
                        "regression": "abs_TI",
                        "beta": res_ati["beta_abs_ti"],
                        "t_beta": res_ati["t_beta_abs_ti"],
                        "rsquared": res_ati["rsquared"],
                        "nobs": res_ati["nobs"],
                        "H": float("nan"),
                    }
                )
            except (ValueError, Exception) as exc:
                warnings.warn(f"|TI| regression failed {ticker} {date}: {exc}")

            # Regression 3: volume scaling (H estimation + power regression)
            try:
                res_vol = run_volume_regression(subdf)
                vol_results.append(res_vol)
                table5_rows.append(
                    {
                        **row_base,
                        "regression": "vol_power",
                        "beta": res_vol["c"],
                        "t_beta": res_vol["t_c"],
                        "rsquared": res_vol["rsquared_stage2"],
                        "nobs": res_vol["nobs_stage2"],
                        "H": res_vol["H"],
                    }
                )
            except (ValueError, Exception) as exc:
                warnings.warn(f"Volume regression failed {ticker} {date}: {exc}")

            if done % 20 == 0 or done == total:
                print(f"  [{pct:5.1f}%] {ticker} {date}", flush=True)

    # -------------------------------------------------------------------------
    # Summaries
    # -------------------------------------------------------------------------
    abs_ofi_summary: dict = {}
    abs_ti_summary: dict = {}
    vol_summary: dict = {}

    if abs_ofi_results:
        abs_ofi_summary = summarize_subsamples(abs_ofi_results)
    if abs_ti_results:
        abs_ti_summary = summarize_subsamples(abs_ti_results)
    if vol_results:
        vol_summary = summarize_subsamples(vol_results)

    # -------------------------------------------------------------------------
    # Save outputs
    # -------------------------------------------------------------------------
    if table5_rows:
        pd.DataFrame(table5_rows).to_csv(
            os.path.join(results_dir, "table5_volume.csv"), index=False
        )
        print(f"\n  Saved: {results_dir}/table5_volume.csv")

    # Summary across subsamples
    sum_rows = []
    for label, summ in [
        ("abs_OFI", abs_ofi_summary),
        ("abs_TI", abs_ti_summary),
        ("vol_power", vol_summary),
    ]:
        if not summ:
            continue
        means = summ.get("means", {})
        mat = summ.get("mean_abs_t", {})
        sum_rows.append(
            {
                "regression": label,
                "mean_r2": means.get("rsquared", float("nan")),
                "mean_H": means.get("H", float("nan")),
                "mean_beta": (
                    means.get("beta_abs_ofi")
                    or means.get("beta_abs_ti")
                    or means.get("c")
                    or float("nan")
                ),
                "mean_abs_t": (
                    mat.get("t_beta_abs_ofi")
                    or mat.get("t_beta_abs_ti")
                    or mat.get("t_c")
                    or float("nan")
                ),
            }
        )
    if sum_rows:
        pd.DataFrame(sum_rows).to_csv(
            os.path.join(results_dir, "table5_summary.csv"), index=False
        )
        print(f"  Saved: {results_dir}/table5_summary.csv")

    # -------------------------------------------------------------------------
    # Console summary
    # -------------------------------------------------------------------------
    n_sub = len(vol_results)
    print("\n" + "=" * 60)
    print("Exp 6 — H4 Volume Scaling Summary")
    print("=" * 60)
    print(f"  Subsamples processed : {n_sub}")
    if vol_summary:
        vm = vol_summary.get("means", {})
        print(f"  Mean Hurst exponent H  : {vm.get('H', float('nan')):.4f}")
        print(f"  Mean Stage-1 R²        : {vm.get('rsquared_stage1', float('nan')):.4f}")
        print(f"  Mean Stage-2 R²        : {vm.get('rsquared_stage2', float('nan')):.4f}")
    if abs_ofi_summary:
        am = abs_ofi_summary.get("means", {})
        print(f"  |OFI| mean R²          : {am.get('rsquared', float('nan')):.4f}")
    if abs_ti_summary:
        tm = abs_ti_summary.get("means", {})
        print(f"  |TI|  mean R²          : {tm.get('rsquared', float('nan')):.4f}")
    print("=" * 60 + "\n")

    return {
        "abs_ofi_summary": abs_ofi_summary,
        "abs_ti_summary": abs_ti_summary,
        "volume_summary": vol_summary,
        "n_subsamples": n_sub,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Exp 6: H4 – Volume scaling regressions (Table 5)."
    )
    p.add_argument("--n_stocks", type=int, default=5)
    p.add_argument("--grid_dir", default="results/grid")
    p.add_argument("--results_dir", default="results")
    return p


def main() -> None:
    parser = _build_arg_parser()
    args = parser.parse_args()

    tickers = TICKERS_50[: args.n_stocks]
    dates = TRADING_DAYS_APRIL_2010

    print(f"Exp 6: H4 volume regressions for {len(tickers)} stocks × {len(dates)} dates …\n")
    run_exp6(
        tickers=tickers,
        dates=dates,
        grid_dir=args.grid_dir,
        results_dir=args.results_dir,
    )


if __name__ == "__main__":
    main()
