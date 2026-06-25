"""
exp_5_h3_ofi_ti.py - Experiment 5: H3 – OFI and TI jointly explain price changes.

Tests **Hypothesis 3**: OFI and trade imbalance (TI) each carry incremental
information about price changes beyond what the other provides.

Three regressions run per stock-day subsample
----------------------------------------------
1. **TI-only**:
   Δp_k = α + β_TI · TI_k + ε_k

2. **OFI-only**:
   Δp_k = α + β_OFI · OFI_k + ε_k

3. **Joint OFI + TI**:
   Δp_k = α + β_OFI · OFI_k + β_TI · TI_k + ε_k

All regressions use HC0 heteroskedasticity-robust standard errors.

Expected results (Table 4, Panel A)
-------------------------------------
- R²(TI-only)  < R²(OFI-only)  < R²(Joint).
- Both β_OFI and β_TI are positive and significant in the joint model.

Outputs
-------
- ``{results_dir}/table4_panel_A.csv``   – per-subsample results for all 3 models.
- ``{results_dir}/table4_summary.csv``   – mean R² comparison across models.

Usage
-----
    python exp/exp_5_h3_ofi_ti.py
    python exp/exp_5_h3_ofi_ti.py --n_stocks 5 --grid_dir results/grid \\
                                   --results_dir results
"""

import argparse
import os
import sys
import warnings

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.regression import (
    run_ofi_regression,
    run_ti_regression,
    run_joint_ofi_ti_regression,
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


# ---------------------------------------------------------------------------
# Core pipeline
# ---------------------------------------------------------------------------


def run_exp5(
    tickers: list[str],
    dates: list[str],
    grid_dir: str,
    results_dir: str,
) -> dict:
    """Run TI-only, OFI-only, and joint OFI+TI regressions.

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
        Keys: ``'ti_summary'``, ``'ofi_summary'``, ``'joint_summary'``,
        ``'n_subsamples'``.
    """
    os.makedirs(results_dir, exist_ok=True)

    ti_results: list[dict] = []
    ofi_results: list[dict] = []
    joint_results: list[dict] = []
    panel_A_rows: list[dict] = []

    total = len(tickers) * len(dates)
    done = 0

    for ticker in tickers:
        for date in dates:
            done += 1
            pct = 100 * done / total

            grid = _load_grid(ticker, date, grid_dir)
            if grid is None or grid.empty:
                continue

            subdf = grid[["OFI_k", "delta_P_k", "TI_k"]].dropna(subset=["delta_P_k"])
            if len(subdf) < 10:
                continue

            row_base = {"ticker": ticker, "date": date}

            # TI-only
            ti_r2 = float("nan")
            try:
                res_ti = run_ti_regression(subdf)
                ti_results.append(res_ti)
                ti_r2 = res_ti["rsquared"]
                panel_A_rows.append(
                    {
                        **row_base,
                        "model": "TI_only",
                        "rsquared": ti_r2,
                        "beta_ofi": float("nan"),
                        "t_beta_ofi": float("nan"),
                        "beta_ti": res_ti["beta_ti"],
                        "t_beta_ti": res_ti["t_beta_ti"],
                        "nobs": res_ti["nobs"],
                    }
                )
            except (ValueError, Exception) as exc:
                warnings.warn(f"TI regression failed {ticker} {date}: {exc}")

            # OFI-only
            ofi_r2 = float("nan")
            try:
                res_ofi = run_ofi_regression(subdf)
                ofi_results.append(res_ofi)
                ofi_r2 = res_ofi["rsquared"]
                panel_A_rows.append(
                    {
                        **row_base,
                        "model": "OFI_only",
                        "rsquared": ofi_r2,
                        "beta_ofi": res_ofi["beta_ofi"],
                        "t_beta_ofi": res_ofi["t_beta_ofi"],
                        "beta_ti": float("nan"),
                        "t_beta_ti": float("nan"),
                        "nobs": res_ofi["nobs"],
                    }
                )
            except (ValueError, Exception) as exc:
                warnings.warn(f"OFI regression failed {ticker} {date}: {exc}")

            # Joint OFI + TI
            joint_r2 = float("nan")
            try:
                res_joint = run_joint_ofi_ti_regression(subdf)
                joint_results.append(res_joint)
                joint_r2 = res_joint["rsquared"]
                panel_A_rows.append(
                    {
                        **row_base,
                        "model": "Joint_OFI_TI",
                        "rsquared": joint_r2,
                        "beta_ofi": res_joint["beta_ofi"],
                        "t_beta_ofi": res_joint["t_beta_ofi"],
                        "beta_ti": res_joint["beta_ti"],
                        "t_beta_ti": res_joint["t_beta_ti"],
                        "nobs": res_joint["nobs"],
                    }
                )
            except (ValueError, Exception) as exc:
                warnings.warn(f"Joint regression failed {ticker} {date}: {exc}")

            if done % 20 == 0 or done == total:
                print(
                    f"  [{pct:5.1f}%] {ticker} {date}  "
                    f"R²: TI={ti_r2:.4f}, OFI={ofi_r2:.4f}, Joint={joint_r2:.4f}",
                    flush=True,
                )

    # -------------------------------------------------------------------------
    # Summaries
    # -------------------------------------------------------------------------
    ti_summary: dict = {}
    ofi_summary: dict = {}
    joint_summary: dict = {}

    if ti_results:
        ti_summary = summarize_subsamples(ti_results)
    if ofi_results:
        ofi_summary = summarize_subsamples(ofi_results)
    if joint_results:
        joint_summary = summarize_subsamples(joint_results)

    # -------------------------------------------------------------------------
    # Save outputs
    # -------------------------------------------------------------------------
    if panel_A_rows:
        pd.DataFrame(panel_A_rows).to_csv(
            os.path.join(results_dir, "table4_panel_A.csv"), index=False
        )
        print(f"\n  Saved: {results_dir}/table4_panel_A.csv")

    # Summary table (mean R² by model)
    summary_rows = []
    for model_name, summ in [
        ("TI_only", ti_summary),
        ("OFI_only", ofi_summary),
        ("Joint_OFI_TI", joint_summary),
    ]:
        if summ:
            means = summ.get("means", {})
            mat = summ.get("mean_abs_t", {})
            sig = summ.get("sig_rate", {})
            summary_rows.append(
                {
                    "model": model_name,
                    "mean_r2": means.get("rsquared", float("nan")),
                    "mean_beta_ofi": means.get("beta_ofi", float("nan")),
                    "mean_beta_ti": means.get("beta_ti", float("nan")),
                    "mat_beta_ofi": mat.get("t_beta_ofi", float("nan")),
                    "mat_beta_ti": mat.get("t_beta_ti", float("nan")),
                    "sig_beta_ofi": sig.get("t_beta_ofi", float("nan")),
                    "sig_beta_ti": sig.get("t_beta_ti", float("nan")),
                }
            )

    if summary_rows:
        pd.DataFrame(summary_rows).to_csv(
            os.path.join(results_dir, "table4_summary.csv"), index=False
        )
        print(f"  Saved: {results_dir}/table4_summary.csv")

    # -------------------------------------------------------------------------
    # Console summary
    # -------------------------------------------------------------------------
    n_sub = len(ofi_results)
    print("\n" + "=" * 60)
    print("Exp 5 — H3 OFI vs TI Summary (mean R² across subsamples)")
    print("=" * 60)
    print(f"  Subsamples processed : {n_sub}")
    if ti_summary:
        ti_m = ti_summary.get("means", {})
        print(f"  TI-only  R²  : {ti_m.get('rsquared', float('nan')):.4f}")
    if ofi_summary:
        ofi_m = ofi_summary.get("means", {})
        print(f"  OFI-only R²  : {ofi_m.get('rsquared', float('nan')):.4f}")
    if joint_summary:
        jt_m = joint_summary.get("means", {})
        jt_mat = joint_summary.get("mean_abs_t", {})
        jt_sig = joint_summary.get("sig_rate", {})
        print(f"  Joint    R²  : {jt_m.get('rsquared', float('nan')):.4f}")
        print(
            f"  Joint β_OFI  : {jt_m.get('beta_ofi', float('nan')):.6f}  "
            f"|t| = {jt_mat.get('t_beta_ofi', float('nan')):.2f}  "
            f"sig% = {100 * jt_sig.get('t_beta_ofi', float('nan')):.1f}"
        )
        print(
            f"  Joint β_TI   : {jt_m.get('beta_ti', float('nan')):.6f}  "
            f"|t| = {jt_mat.get('t_beta_ti', float('nan')):.2f}  "
            f"sig% = {100 * jt_sig.get('t_beta_ti', float('nan')):.1f}"
        )
    print("=" * 60 + "\n")

    return {
        "ti_summary": ti_summary,
        "ofi_summary": ofi_summary,
        "joint_summary": joint_summary,
        "n_subsamples": n_sub,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Exp 5: H3 – OFI vs TI explanatory power (Table 4)."
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

    print(f"Exp 5: H3 regressions for {len(tickers)} stocks × {len(dates)} dates …\n")
    run_exp5(
        tickers=tickers,
        dates=dates,
        grid_dir=args.grid_dir,
        results_dir=args.results_dir,
    )


if __name__ == "__main__":
    main()
