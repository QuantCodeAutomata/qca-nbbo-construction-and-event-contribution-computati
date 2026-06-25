"""
exp_7_h5_seasonality.py - Experiment 7: H5 – Intraday seasonality of β and depth.

Tests **Hypothesis 5**: price-impact β and quoted depth AD exhibit systematic
intraday patterns that are inversely related — when depth is low (near open
and close), price impact is high, and vice versa.

Methodology
-----------
For each half-hour slot h (0–12, i.e., 09:30–10:00 … 15:30–16:00):

  1. Extract all 10-second grid bins belonging to slot h.
  2. Estimate per-slot OFI regression β_h via
     :func:`src.regression.run_ofi_regression`.
  3. Record mean quoted depth AD_h for slot h.
  4. Decompose variance of Δp_k into explained (β_h² · Var(OFI_k)) and residual.

The per-slot quantities are averaged across all stock-days and normalised to
unit mean for Figure 5.

Figure 5 — Normalised intraday β and AD
  Two-panel line chart: normalised β_h (left) and normalised AD_h (right),
  plotted against the 13 half-hour labels 09:30 … 15:30.

Figure 6 — Variance decomposition
  Three stacked-line chart: realised variance of Δp_k, explained variance
  β_h² · Var(OFI_k), and residual variance, normalised to daily mean.

Outputs
-------
- ``{results_dir}/intraday_beta.csv``      – per-slot (h, mean_beta, mean_AD).
- ``{results_dir}/intraday_variance.csv``  – per-slot variance components.
- ``{results_dir}/figure5_intraday_beta_AD.png``
- ``{results_dir}/figure6_variance_decomp.png``

Usage
-----
    python exp/exp_7_h5_seasonality.py
    python exp/exp_7_h5_seasonality.py --n_stocks 5 \\
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

from src.regression import run_ofi_regression
from src.grid import HALF_HOUR_LABELS
from data.synthetic_data import TICKERS_50, TRADING_DAYS_APRIL_2010


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

N_SLOTS = 13  # half-hour slots 0–12


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _load_grid(ticker: str, date: str, grid_dir: str) -> pd.DataFrame | None:
    path = os.path.join(grid_dir, f"{ticker}_{date}_grid.parquet")
    if not os.path.exists(path):
        return None
    return pd.read_parquet(path)


def _slot_regression(
    grid: pd.DataFrame,
    slot: int,
) -> tuple[float, float, float, float, float]:
    """Run OFI regression for a single half-hour slot.

    Returns
    -------
    tuple
        (beta_h, r2_h, var_dp_h, var_ofi_h, mean_ad_h)
        All float, NaN if insufficient data.
    """
    slot_data = grid[grid["half_hour"] == slot].copy()
    slot_data = slot_data.dropna(subset=["delta_P_k"])

    nan5 = (float("nan"),) * 5

    if len(slot_data) < 5:
        return nan5

    try:
        res = run_ofi_regression(slot_data)
        beta_h = res["beta_ofi"]
        r2_h = res["rsquared"]
    except Exception:
        return nan5

    var_dp_h = float(slot_data["delta_P_k"].var())
    var_ofi_h = float(slot_data["OFI_k"].var())
    mean_ad_h = (
        float(slot_data["AD"].mean(skipna=True))
        if "AD" in slot_data.columns
        else float("nan")
    )
    return beta_h, r2_h, var_dp_h, var_ofi_h, mean_ad_h


def _normalise(arr: np.ndarray) -> np.ndarray:
    """Divide by the nanmean so the series has mean ≈ 1."""
    m = np.nanmean(arr)
    if m == 0 or not np.isfinite(m):
        return arr
    return arr / m


def _figure5(
    beta_norm: np.ndarray,
    ad_norm: np.ndarray,
    slot_labels: list[str],
    out_path: str,
) -> None:
    """Figure 5: normalised intraday β and AD."""
    x = np.arange(N_SLOTS)
    fig, ax1 = plt.subplots(figsize=(9, 4))

    color_beta = "crimson"
    color_ad = "steelblue"

    ax1.plot(x, beta_norm, "o-", color=color_beta, linewidth=1.8,
             markersize=5, label="Normalised β")
    ax1.set_ylabel("Normalised price-impact β", color=color_beta)
    ax1.tick_params(axis="y", labelcolor=color_beta)
    ax1.set_xticks(x)
    ax1.set_xticklabels(slot_labels, rotation=45, ha="right", fontsize=8)
    ax1.set_xlabel("Half-hour slot (09:30 – 16:00)")
    ax1.set_title("Figure 5: Intraday patterns of β and average depth")

    ax2 = ax1.twinx()
    ax2.plot(x, ad_norm, "s--", color=color_ad, linewidth=1.8,
             markersize=5, label="Normalised AD")
    ax2.set_ylabel("Normalised average depth AD", color=color_ad)
    ax2.tick_params(axis="y", labelcolor=color_ad)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper center", fontsize=9)

    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"  Saved: {out_path}")


def _figure6(
    var_total_norm: np.ndarray,
    var_expl_norm: np.ndarray,
    var_resid_norm: np.ndarray,
    slot_labels: list[str],
    out_path: str,
) -> None:
    """Figure 6: intraday variance decomposition."""
    x = np.arange(N_SLOTS)
    fig, ax = plt.subplots(figsize=(9, 4))

    ax.plot(x, var_total_norm, "o-", color="black", linewidth=1.8,
            markersize=5, label="Total Var(Δp)")
    ax.plot(x, var_expl_norm, "s-", color="steelblue", linewidth=1.8,
            markersize=5, label="Explained: β² · Var(OFI)")
    ax.plot(x, var_resid_norm, "^-", color="tomato", linewidth=1.8,
            markersize=5, label="Residual")

    ax.set_xticks(x)
    ax.set_xticklabels(slot_labels, rotation=45, ha="right", fontsize=8)
    ax.set_xlabel("Half-hour slot (09:30 – 16:00)")
    ax.set_ylabel("Normalised variance (daily mean = 1)")
    ax.set_title("Figure 6: Intraday variance decomposition")
    ax.legend(fontsize=9, loc="upper center")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"  Saved: {out_path}")


# ---------------------------------------------------------------------------
# Core pipeline
# ---------------------------------------------------------------------------


def run_exp7(
    tickers: list[str],
    dates: list[str],
    grid_dir: str,
    exp3_results_path: str,
    results_dir: str,
) -> dict:
    """Compute and plot intraday seasonality in β and depth.

    Parameters
    ----------
    tickers : list[str]
        Ticker symbols to process.
    dates : list[str]
        Trading dates in ``YYYY-MM-DD`` format.
    grid_dir : str
        Directory containing grid parquets from Exp 2.
    exp3_results_path : str
        Path to ``exp3_betas.csv`` from Exp 3 (used for reference only).
    results_dir : str
        Output directory for tables and figures.

    Returns
    -------
    dict
        Keys: ``'intraday_beta'``, ``'intraday_variance'`` (DataFrames).
    """
    os.makedirs(results_dir, exist_ok=True)

    # Accumulators per slot: list of per-stock-day values
    slot_betas: list[list[float]] = [[] for _ in range(N_SLOTS)]
    slot_ads: list[list[float]] = [[] for _ in range(N_SLOTS)]
    slot_var_dp: list[list[float]] = [[] for _ in range(N_SLOTS)]
    slot_var_ofi: list[list[float]] = [[] for _ in range(N_SLOTS)]

    total = len(tickers) * len(dates)
    done = 0

    for ticker in tickers:
        for date in dates:
            done += 1
            pct = 100 * done / total

            grid = _load_grid(ticker, date, grid_dir)
            if grid is None or grid.empty:
                continue

            if "half_hour" not in grid.columns:
                continue

            for h in range(N_SLOTS):
                beta_h, r2_h, var_dp_h, var_ofi_h, mean_ad_h = _slot_regression(
                    grid, h
                )
                if np.isfinite(beta_h):
                    slot_betas[h].append(beta_h)
                if np.isfinite(mean_ad_h):
                    slot_ads[h].append(mean_ad_h)
                if np.isfinite(var_dp_h):
                    slot_var_dp[h].append(var_dp_h)
                if np.isfinite(var_ofi_h):
                    slot_var_ofi[h].append(var_ofi_h)

            if done % 20 == 0 or done == total:
                print(f"  [{pct:5.1f}%] {ticker} {date}", flush=True)

    # -------------------------------------------------------------------------
    # Aggregate across stock-days
    # -------------------------------------------------------------------------
    mean_beta = np.array(
        [np.nanmean(slot_betas[h]) if slot_betas[h] else float("nan")
         for h in range(N_SLOTS)]
    )
    mean_ad = np.array(
        [np.nanmean(slot_ads[h]) if slot_ads[h] else float("nan")
         for h in range(N_SLOTS)]
    )
    mean_var_dp = np.array(
        [np.nanmean(slot_var_dp[h]) if slot_var_dp[h] else float("nan")
         for h in range(N_SLOTS)]
    )
    mean_var_ofi = np.array(
        [np.nanmean(slot_var_ofi[h]) if slot_var_ofi[h] else float("nan")
         for h in range(N_SLOTS)]
    )

    # Explained variance = beta² × Var(OFI) averaged over stock-days per slot
    mean_var_expl = np.array(
        [
            np.nanmean([b**2 * v
                        for b, v in zip(slot_betas[h], slot_var_ofi[h])
                        if np.isfinite(b) and np.isfinite(v)])
            if slot_betas[h] else float("nan")
            for h in range(N_SLOTS)
        ]
    )
    mean_var_resid = np.maximum(mean_var_dp - mean_var_expl, 0.0)

    # Normalise
    beta_norm = _normalise(mean_beta)
    ad_norm = _normalise(mean_ad)
    var_total_norm = _normalise(mean_var_dp)
    var_expl_norm = _normalise(mean_var_expl)
    var_resid_norm = _normalise(mean_var_resid)

    # -------------------------------------------------------------------------
    # Save intraday tables
    # -------------------------------------------------------------------------
    intraday_beta_df = pd.DataFrame(
        {
            "half_hour": list(range(N_SLOTS)),
            "label": HALF_HOUR_LABELS,
            "mean_beta": mean_beta,
            "mean_AD": mean_ad,
            "beta_normalised": beta_norm,
            "AD_normalised": ad_norm,
        }
    )
    intraday_beta_path = os.path.join(results_dir, "intraday_beta.csv")
    intraday_beta_df.to_csv(intraday_beta_path, index=False)
    print(f"\n  Saved: {intraday_beta_path}")

    intraday_var_df = pd.DataFrame(
        {
            "half_hour": list(range(N_SLOTS)),
            "label": HALF_HOUR_LABELS,
            "var_dp": mean_var_dp,
            "var_explained": mean_var_expl,
            "var_residual": mean_var_resid,
            "var_dp_norm": var_total_norm,
            "var_expl_norm": var_expl_norm,
            "var_resid_norm": var_resid_norm,
        }
    )
    intraday_var_path = os.path.join(results_dir, "intraday_variance.csv")
    intraday_var_df.to_csv(intraday_var_path, index=False)
    print(f"  Saved: {intraday_var_path}")

    # -------------------------------------------------------------------------
    # Figures
    # -------------------------------------------------------------------------
    _figure5(
        beta_norm, ad_norm, HALF_HOUR_LABELS,
        out_path=os.path.join(results_dir, "figure5_intraday_beta_AD.png"),
    )
    _figure6(
        var_total_norm, var_expl_norm, var_resid_norm, HALF_HOUR_LABELS,
        out_path=os.path.join(results_dir, "figure6_variance_decomp.png"),
    )

    # -------------------------------------------------------------------------
    # Console summary
    # -------------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("Exp 7 — H5 Intraday Seasonality Summary")
    print("=" * 60)
    print(f"  Slots with data   : {int(np.sum(np.isfinite(mean_beta)))}")
    print(f"  Mean β  (all slots): {np.nanmean(mean_beta):.6f}")
    print(f"  Max  β  (open/close effect): {np.nanmax(mean_beta):.6f}")
    print(f"  Min  β  (mid-day min)       : {np.nanmin(mean_beta):.6f}")
    print(f"  Mean AD (all slots): {np.nanmean(mean_ad):.1f} shares")
    print("=" * 60 + "\n")

    return {
        "intraday_beta": intraday_beta_df,
        "intraday_variance": intraday_var_df,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Exp 7: H5 – Intraday seasonality of β and depth (Figs 5 & 6)."
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
    dates = TRADING_DAYS_APRIL_2010

    print(
        f"Exp 7: H5 seasonality for {len(tickers)} stocks × {len(dates)} dates …\n"
    )
    run_exp7(
        tickers=tickers,
        dates=dates,
        grid_dir=args.grid_dir,
        exp3_results_path=args.exp3_results_path,
        results_dir=args.results_dir,
    )


if __name__ == "__main__":
    main()
