"""
exp_4_h2_depth.py - Experiment 4: H2 – Price-impact beta scales with depth.

Tests **Hypothesis 2**: the price-impact coefficient β_s is inversely
proportional to average market depth AD_s across stocks.

Two-stage procedure
-------------------
Stage 1 — Log-log cross-stock regression (HAC standard errors):

    log(β_s) = c + λ · log(AD_s) + ε_s

Expected result: λ ≈ −1 (β ∝ 1/AD).

Stage 2 — Power regression (levels, HAC standard errors):

    β_s = c₀ + c · AD_s^{λ̂} + ε_s

Inputs
------
- ``{exp3_results_path}``    – ``exp3_betas.csv`` from Exp 3 (per-stock β̂ and AD̅).
- ``{grid_dir}``             – grid parquets from Exp 2 (for per-day scatter).

Outputs
-------
- ``{results_dir}/table3_depth_regression.csv``   – two-stage results.
- ``{results_dir}/scatter_logbeta_logAD.png``     – cross-stock log-log scatter.
- ``{results_dir}/scatter_daylogbeta_logAD_{ticker}.png`` – per-day scatter for
  the reference ticker (SLB or first available).

Usage
-----
    python exp/exp_4_h2_depth.py
    python exp/exp_4_h2_depth.py --n_stocks 5 \\
                                  --grid_dir results/grid \\
                                  --exp3_results_path results/exp3_betas.csv \\
                                  --results_dir results
"""

import argparse
import os
import sys
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.regression import (
    run_depth_log_regression,
    run_depth_power_regression,
    run_ofi_regression,
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


def _compute_per_day_beta_ad(
    ticker: str,
    dates: list[str],
    grid_dir: str,
) -> pd.DataFrame:
    """Compute per-day β and AD for a single ticker from grid files."""
    records: list[dict] = []
    for date in dates:
        g = _load_grid(ticker, date, grid_dir)
        if g is None or g.empty:
            continue
        subdf = g[["OFI_k", "delta_P_k"]].dropna(subset=["delta_P_k"])
        if len(subdf) < 10:
            continue
        try:
            res = run_ofi_regression(subdf)
            beta_d = res["beta_ofi"]
        except Exception:
            beta_d = float("nan")

        ad_d = float(g["AD"].mean(skipna=True)) if "AD" in g.columns else float("nan")
        records.append({"date": date, "beta_d": beta_d, "ad_d": ad_d})
    return pd.DataFrame(records)


def _loglog_scatter(
    ad_arr: np.ndarray,
    beta_arr: np.ndarray,
    lambda_hat: float,
    const_hat: float,
    labels: list[str],
    title: str,
    out_path: str,
) -> None:
    """Log-log scatter of (log AD, log β) with fitted regression line."""
    mask = (ad_arr > 0) & (beta_arr > 0) & np.isfinite(ad_arr) & np.isfinite(beta_arr)
    if mask.sum() < 2:
        return

    log_ad = np.log(ad_arr[mask])
    log_beta = np.log(beta_arr[mask])

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(log_ad, log_beta, color="steelblue", s=50, zorder=5)
    for i, (lad, lb, lbl) in enumerate(
        zip(log_ad, log_beta, [l for l, m in zip(labels, mask) if m])
    ):
        ax.annotate(lbl, (lad, lb), fontsize=7, textcoords="offset points",
                    xytext=(4, 4))

    # Regression line
    x_fit = np.linspace(log_ad.min() - 0.1, log_ad.max() + 0.1, 200)
    ax.plot(x_fit, const_hat + lambda_hat * x_fit, color="crimson",
            linewidth=1.5, label=f"λ = {lambda_hat:.3f}")
    ax.set_xlabel("log(AD)  [log shares]")
    ax.set_ylabel("log(β̂)  [log ticks/share]")
    ax.set_title(title)
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"  Saved: {out_path}")


# ---------------------------------------------------------------------------
# Core pipeline
# ---------------------------------------------------------------------------


def run_exp4(
    tickers: list[str],
    grid_dir: str,
    exp3_results_path: str,
    results_dir: str,
) -> dict:
    """Run H2 depth scaling regression.

    Parameters
    ----------
    tickers : list[str]
        Ticker symbols (same subset used in Exp 3).
    grid_dir : str
        Directory containing grid parquets from Exp 2.
    exp3_results_path : str
        Path to ``exp3_betas.csv`` produced by Exp 3.
    results_dir : str
        Output directory for tables and plots.

    Returns
    -------
    dict
        Keys: ``'stage1'``, ``'stage2'`` (result dicts from regression
        functions), ``'n_stocks'``.
    """
    os.makedirs(results_dir, exist_ok=True)

    # -------------------------------------------------------------------------
    # Load per-stock β and AD from Exp 3 output
    # -------------------------------------------------------------------------
    if not os.path.exists(exp3_results_path):
        raise FileNotFoundError(
            f"Exp 3 betas file not found: {exp3_results_path}\n"
            "Run exp_3_h1_regression.py first."
        )

    betas_df = pd.read_csv(exp3_results_path)
    # Filter to the tickers we're processing and that have valid data
    betas_df = betas_df[betas_df["ticker"].isin(tickers)].copy()
    betas_df = betas_df.dropna(subset=["beta_hat", "mean_ad"])
    betas_df = betas_df[(betas_df["beta_hat"] > 0) & (betas_df["mean_ad"] > 0)]

    if betas_df.empty:
        print("  WARNING: No valid (β, AD) pairs found.  Skipping depth regression.")
        return {"stage1": {}, "stage2": {}, "n_stocks": 0}

    beta_arr = betas_df["beta_hat"].to_numpy(dtype=float)
    ad_arr = betas_df["mean_ad"].to_numpy(dtype=float)
    stock_labels = betas_df["ticker"].tolist()
    n_stocks = len(betas_df)

    print(f"  Cross-stock regression: n = {n_stocks} stocks")
    for tkr, b, a in zip(stock_labels, beta_arr, ad_arr):
        print(f"    {tkr}: β̂ = {b:.6f}, AD = {a:.1f}")

    # -------------------------------------------------------------------------
    # Stage 1: log-log regression  log(β) = c + λ log(AD)
    # -------------------------------------------------------------------------
    stage1: dict = {}
    lambda_hat = float("nan")
    const1 = float("nan")

    try:
        stage1 = run_depth_log_regression(beta_arr, ad_arr)
        lambda_hat = stage1["lambda_hat"]
        # Estimate constant from the data
        log_beta_vals = np.log(beta_arr[(beta_arr > 0) & (ad_arr > 0)])
        log_ad_vals = np.log(ad_arr[(beta_arr > 0) & (ad_arr > 0)])
        const1 = float(np.mean(log_beta_vals - lambda_hat * log_ad_vals))

        print(
            f"\n  Stage 1 (log-log): λ̂ = {lambda_hat:.4f}  "
            f"|t| = {abs(stage1['t_lambda']):.2f}  "
            f"R² = {stage1['rsquared']:.4f}"
        )
    except ValueError as exc:
        warnings.warn(f"Stage 1 depth regression failed: {exc}")

    # -------------------------------------------------------------------------
    # Stage 2: power regression  β = c₀ + c · AD^{λ̂}
    # -------------------------------------------------------------------------
    stage2: dict = {}

    if np.isfinite(lambda_hat):
        try:
            stage2 = run_depth_power_regression(beta_arr, ad_arr, lambda_hat)
            print(
                f"  Stage 2 (power):   ĉ = {stage2['c_hat']:.6f}  "
                f"|t| = {abs(stage2['t_c']):.2f}  "
                f"R² = {stage2['rsquared']:.4f}"
            )
        except ValueError as exc:
            warnings.warn(f"Stage 2 depth regression failed: {exc}")

    # -------------------------------------------------------------------------
    # Per-day scatter for reference ticker (intra-stock depth variation)
    # -------------------------------------------------------------------------
    plot_ticker = "SLB" if "SLB" in tickers else tickers[0]
    day_df = _compute_per_day_beta_ad(plot_ticker, TRADING_DAYS_APRIL_2010, grid_dir)

    if not day_df.empty and np.isfinite(lambda_hat):
        day_df = day_df[(day_df["beta_d"] > 0) & (day_df["ad_d"] > 0)].dropna()
        if len(day_df) >= 2:
            day_log_ad = np.log(day_df["ad_d"].to_numpy(dtype=float))
            day_mean_beta = np.log(day_df["beta_d"].to_numpy(dtype=float))
            day_const = float(np.mean(day_mean_beta - lambda_hat * day_log_ad))

            _loglog_scatter(
                day_df["ad_d"].to_numpy(),
                day_df["beta_d"].to_numpy(),
                lambda_hat,
                day_const,
                labels=day_df["date"].tolist(),
                title=f"{plot_ticker}: log(β_d) vs log(AD_d) by day",
                out_path=os.path.join(
                    results_dir, f"scatter_daylogbeta_logAD_{plot_ticker}.png"
                ),
            )

    # Cross-stock scatter
    if np.isfinite(lambda_hat):
        _loglog_scatter(
            ad_arr, beta_arr, lambda_hat, const1,
            labels=stock_labels,
            title="Cross-stock: log(β̂_s) vs log(AD_s)",
            out_path=os.path.join(results_dir, "scatter_logbeta_logAD.png"),
        )

    # -------------------------------------------------------------------------
    # Save table3_depth_regression.csv
    # -------------------------------------------------------------------------
    rows = []
    if stage1:
        rows.append(
            {
                "stage": "1_log_log",
                "estimand": "lambda",
                "estimate": stage1.get("lambda_hat", float("nan")),
                "t_stat": stage1.get("t_lambda", float("nan")),
                "rsquared": stage1.get("rsquared", float("nan")),
                "nobs": stage1.get("nobs", 0),
            }
        )
    if stage2:
        rows.append(
            {
                "stage": "2_power",
                "estimand": "c",
                "estimate": stage2.get("c_hat", float("nan")),
                "t_stat": stage2.get("t_c", float("nan")),
                "rsquared": stage2.get("rsquared", float("nan")),
                "nobs": stage2.get("nobs", 0),
            }
        )
    if rows:
        table3 = pd.DataFrame(rows)
        table3.to_csv(
            os.path.join(results_dir, "table3_depth_regression.csv"), index=False
        )
        print(f"\n  Saved: {results_dir}/table3_depth_regression.csv")

    # -------------------------------------------------------------------------
    # Console summary
    # -------------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("Exp 4 — H2 Depth Scaling Summary")
    print("=" * 60)
    print(f"  Stocks in cross-stock regression : {n_stocks}")
    if stage1:
        print(f"  Stage 1 λ̂  (expected ≈ −1)     : {stage1.get('lambda_hat', float('nan')):.4f}")
        print(f"  Stage 1 |t_λ|                   : {abs(stage1.get('t_lambda', float('nan'))):.2f}")
        print(f"  Stage 1 R²                      : {stage1.get('rsquared', float('nan')):.4f}")
    if stage2:
        print(f"  Stage 2 ĉ                       : {stage2.get('c_hat', float('nan')):.6f}")
        print(f"  Stage 2 R²                      : {stage2.get('rsquared', float('nan')):.4f}")
    print("=" * 60 + "\n")

    return {"stage1": stage1, "stage2": stage2, "n_stocks": n_stocks}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Exp 4: H2 – depth scaling of price-impact beta (Table 3)."
    )
    p.add_argument("--n_stocks", type=int, default=5)
    p.add_argument("--grid_dir", default="results/grid")
    p.add_argument("--exp3_results_path", default="results/exp3_betas.csv")
    p.add_argument("--results_dir", default="results")
    return p


def main() -> None:
    parser = _build_arg_parser()
    args = parser.parse_args()

    tickers = TICKERS_50[: args.n_stocks]
    print(f"Exp 4: H2 depth regression for {len(tickers)} stocks …\n")
    run_exp4(
        tickers=tickers,
        grid_dir=args.grid_dir,
        exp3_results_path=args.exp3_results_path,
        results_dir=args.results_dir,
    )


if __name__ == "__main__":
    main()
