"""
exp_3_h1_regression.py - Experiment 3: H1 – OFI predicts price changes.

Tests **Hypothesis 1**: order-flow imbalance (OFI) positively and
significantly predicts 10-second mid-quote price changes.

Regressions run per stock-day subsample
----------------------------------------
1. **Linear OFI**:
   Δp_k = α + β·OFI_k + ε_k

2. **Signed-quadratic OFI**:
   Δp_k = α + β₁·OFI_k + β₂·OFI_k·|OFI_k| + ε_k

Both regressions use HC0 (White) heteroskedasticity-robust standard errors.

Additional analyses
-------------------
- **ACF** of OFI_k and Δp_k for the first available ticker (or SLB).
- **Scatter plot** of OFI_k vs Δp_k for that ticker with the fitted line.

Outputs
-------
- ``{results_dir}/table2_main_regression.csv``  – per-subsample results.
- ``{results_dir}/table2_summary.csv``          – means & sig-rates across subsamples.
- ``{results_dir}/exp3_betas.csv``              – per-stock pooled β (for Exp 4 & 7).
- ``{results_dir}/acf_ofi_{ticker}.png``        – ACF bar chart.
- ``{results_dir}/scatter_ofi_vs_dP_{ticker}.png`` – OFI scatter plot.

Usage
-----
    python exp/exp_3_h1_regression.py
    python exp/exp_3_h1_regression.py --n_stocks 5 --grid_dir results/grid \\
                                      --results_dir results
"""

import argparse
import os
import sys
import warnings

import matplotlib
matplotlib.use("Agg")  # non-interactive backend for headless servers
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import acf

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.regression import (
    run_ofi_regression,
    run_ofi_quadratic_regression,
    summarize_subsamples,
    run_ols_hc0,
)
from data.synthetic_data import TICKERS_50, TRADING_DAYS_APRIL_2010

import statsmodels.api as sm


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _load_grid(ticker: str, date: str, grid_dir: str) -> pd.DataFrame | None:
    """Load one stock-day grid; return None if file not found."""
    path = os.path.join(grid_dir, f"{ticker}_{date}_grid.parquet")
    if not os.path.exists(path):
        return None
    df = pd.read_parquet(path)
    df["timestamp"] = pd.to_datetime(df["bin_start"])
    return df


def _pool_grids_for_ticker(ticker: str, dates: list[str], grid_dir: str) -> pd.DataFrame:
    """Load and vertically concatenate all daily grids for a single ticker."""
    frames: list[pd.DataFrame] = []
    for date in dates:
        g = _load_grid(ticker, date, grid_dir)
        if g is not None:
            g = g.copy()
            g["date"] = date
            frames.append(g)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def _acf_plot(series: np.ndarray, nlags: int, title: str, out_path: str) -> None:
    """Compute and save an ACF bar-chart up to *nlags* lags."""
    series_clean = series[np.isfinite(series)]
    if len(series_clean) < nlags + 5:
        return
    acf_vals, confint = acf(series_clean, nlags=nlags, alpha=0.05)
    lags = np.arange(len(acf_vals))
    lb = confint[:, 0] - acf_vals          # lower 95 % bound offset
    ub = confint[:, 1] - acf_vals

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(lags, acf_vals, color="steelblue", alpha=0.8, label="ACF")
    ax.fill_between(lags, lb, ub, color="grey", alpha=0.3, label="95% CI")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Lag (10-second bins)")
    ax.set_ylabel("Autocorrelation")
    ax.set_title(title)
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"  Saved ACF plot: {out_path}")


def _scatter_plot(
    ofi: np.ndarray,
    delta_p: np.ndarray,
    beta: float,
    alpha: float,
    ticker: str,
    out_path: str,
) -> None:
    """Scatter OFI vs Δp with regression line; cap outliers for readability."""
    mask = np.isfinite(ofi) & np.isfinite(delta_p)
    ofi_c = ofi[mask]
    dp_c = delta_p[mask]

    # Cap at 99th percentile for visual clarity
    ofi_lo, ofi_hi = np.percentile(ofi_c, [1, 99])
    dp_lo, dp_hi = np.percentile(dp_c, [1, 99])
    vis = (ofi_c >= ofi_lo) & (ofi_c <= ofi_hi) & (dp_c >= dp_lo) & (dp_c <= dp_hi)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(ofi_c[vis], dp_c[vis], alpha=0.08, s=3, color="steelblue", rasterized=True)

    x_range = np.linspace(ofi_lo, ofi_hi, 200)
    ax.plot(x_range, alpha + beta * x_range, color="crimson", linewidth=1.5,
            label=f"β = {beta:.4f}")
    ax.axhline(0, color="grey", linewidth=0.5, linestyle="--")
    ax.axvline(0, color="grey", linewidth=0.5, linestyle="--")
    ax.set_xlabel("OFI (shares)")
    ax.set_ylabel("Δp (ticks)")
    ax.set_title(f"{ticker}: OFI vs Price Change")
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"  Saved scatter plot: {out_path}")


# ---------------------------------------------------------------------------
# Core pipeline
# ---------------------------------------------------------------------------


def run_exp3(
    tickers: list[str],
    dates: list[str],
    grid_dir: str,
    results_dir: str,
) -> dict:
    """Run H1 OFI regression analysis.

    Parameters
    ----------
    tickers : list[str]
        Ticker symbols to process.
    dates : list[str]
        Trading dates in ``YYYY-MM-DD`` format.
    grid_dir : str
        Directory containing grid parquets from Exp 2.
    results_dir : str
        Directory where output CSV files and plots are saved.

    Returns
    -------
    dict
        Keys: ``'linear_summary'``, ``'quad_summary'``,
        ``'per_stock_betas'`` (DataFrame), ``'n_subsamples'``.
    """
    os.makedirs(results_dir, exist_ok=True)

    linear_results: list[dict] = []
    quad_results: list[dict] = []
    per_sd_records: list[dict] = []

    # -- per-stock pooled β (for Exp 4 and Exp 7) --
    stock_pooled: dict[str, dict] = {}

    # Choose reference ticker for plots (SLB preferred)
    plot_ticker = "SLB" if "SLB" in tickers else tickers[0]

    total = len(tickers) * len(dates)
    done = 0

    for ticker in tickers:
        stock_linear: list[dict] = []

        for date in dates:
            done += 1
            pct = 100 * done / total
            grid = _load_grid(ticker, date, grid_dir)

            if grid is None or grid.empty:
                continue

            # Only use non-NaN rows where both OFI_k and delta_P_k are present
            subdf = grid[["OFI_k", "delta_P_k", "TI_k", "VOL_k", "half_hour"]].copy()
            subdf = subdf.dropna(subset=["delta_P_k"])

            if len(subdf) < 10:
                continue

            # Linear regression
            try:
                res_lin = run_ofi_regression(subdf)
                res_lin["ticker"] = ticker
                res_lin["date"] = date
                linear_results.append(res_lin)
                stock_linear.append(res_lin)
                per_sd_records.append(
                    {
                        "ticker": ticker,
                        "date": date,
                        "model": "linear",
                        "beta_ofi": res_lin["beta_ofi"],
                        "t_beta_ofi": res_lin["t_beta_ofi"],
                        "rsquared": res_lin["rsquared"],
                        "nobs": res_lin["nobs"],
                    }
                )
            except (ValueError, Exception) as exc:
                warnings.warn(f"Linear OFI failed {ticker} {date}: {exc}")

            # Quadratic regression
            try:
                res_quad = run_ofi_quadratic_regression(subdf)
                res_quad["ticker"] = ticker
                res_quad["date"] = date
                quad_results.append(res_quad)
                per_sd_records.append(
                    {
                        "ticker": ticker,
                        "date": date,
                        "model": "quadratic",
                        "beta_ofi": res_quad["beta_ofi"],
                        "t_beta_ofi": res_quad["t_beta_ofi"],
                        "rsquared": res_quad["rsquared"],
                        "nobs": res_quad["nobs"],
                    }
                )
            except (ValueError, Exception) as exc:
                warnings.warn(f"Quad OFI failed {ticker} {date}: {exc}")

            if done % 20 == 0 or done == total:
                print(f"  [{pct:5.1f}%] {ticker} {date}", flush=True)

        # -- Pooled β for this stock (across all days) --
        if stock_linear:
            all_ofi_vals = [r["beta_ofi"] for r in stock_linear if np.isfinite(r.get("beta_ofi", float("nan")))]
            mean_beta = float(np.mean(all_ofi_vals)) if all_ofi_vals else float("nan")

            # Pooled regression across all days for this stock
            pooled_grid = _pool_grids_for_ticker(ticker, dates, grid_dir)
            pooled_subdf = (
                pooled_grid[["OFI_k", "delta_P_k"]].dropna()
                if not pooled_grid.empty
                else pd.DataFrame()
            )
            if len(pooled_subdf) >= 10:
                try:
                    pool_res = run_ofi_regression(pooled_subdf)
                    pooled_beta = pool_res["beta_ofi"]
                    pooled_r2 = pool_res["rsquared"]
                except Exception:
                    pooled_beta = mean_beta
                    pooled_r2 = float("nan")
            else:
                pooled_beta = mean_beta
                pooled_r2 = float("nan")

            # Mean AD for the stock (from grid AD column)
            if not pooled_grid.empty and "AD" in pooled_grid.columns:
                mean_ad = float(pooled_grid["AD"].mean(skipna=True))
            else:
                mean_ad = float("nan")

            stock_pooled[ticker] = {
                "ticker": ticker,
                "beta_hat": pooled_beta,
                "mean_ad": mean_ad,
                "pooled_r2": pooled_r2,
                "n_days": len(stock_linear),
            }

    # -------------------------------------------------------------------------
    # ACF and scatter plots for reference ticker
    # -------------------------------------------------------------------------
    plot_grid = _pool_grids_for_ticker(plot_ticker, dates, grid_dir)
    if not plot_grid.empty:
        ofi_arr = plot_grid["OFI_k"].to_numpy(dtype=float)
        dp_arr = plot_grid["delta_P_k"].to_numpy(dtype=float)

        # ACF plots
        _acf_plot(
            ofi_arr,
            nlags=20,
            title=f"{plot_ticker}: ACF of OFI (10-second bins)",
            out_path=os.path.join(results_dir, f"acf_ofi_{plot_ticker}.png"),
        )
        _acf_plot(
            dp_arr,
            nlags=20,
            title=f"{plot_ticker}: ACF of Δp (10-second bins)",
            out_path=os.path.join(results_dir, f"acf_dP_{plot_ticker}.png"),
        )

        # Scatter plot
        if plot_ticker in stock_pooled:
            beta_plot = stock_pooled[plot_ticker]["beta_hat"]
        elif linear_results:
            beta_plot = np.mean([r["beta_ofi"] for r in linear_results
                                 if r.get("ticker") == plot_ticker and np.isfinite(r.get("beta_ofi", float("nan")))])
        else:
            beta_plot = 0.0

        valid_mask = np.isfinite(ofi_arr) & np.isfinite(dp_arr)
        if valid_mask.sum() >= 5:
            X_scat = sm.add_constant(ofi_arr[valid_mask].reshape(-1, 1), has_constant="add")
            try:
                scatter_res = run_ols_hc0(dp_arr[valid_mask], X_scat)
                alpha_scat = float(scatter_res["params"][0])
                beta_scat = float(scatter_res["params"][1])
            except Exception:
                alpha_scat = 0.0
                beta_scat = beta_plot

            _scatter_plot(
                ofi_arr, dp_arr, beta_scat, alpha_scat,
                plot_ticker,
                os.path.join(results_dir, f"scatter_ofi_vs_dP_{plot_ticker}.png"),
            )

    # -------------------------------------------------------------------------
    # Summaries
    # -------------------------------------------------------------------------
    linear_summary: dict = {}
    quad_summary: dict = {}

    if linear_results:
        clean_lin = [{k: v for k, v in r.items() if k not in ("ticker", "date")}
                     for r in linear_results]
        linear_summary = summarize_subsamples(clean_lin)
    if quad_results:
        clean_quad = [{k: v for k, v in r.items() if k not in ("ticker", "date")}
                      for r in quad_results]
        quad_summary = summarize_subsamples(clean_quad)

    # -------------------------------------------------------------------------
    # Save outputs
    # -------------------------------------------------------------------------
    # Per-subsample table
    if per_sd_records:
        pd.DataFrame(per_sd_records).to_csv(
            os.path.join(results_dir, "table2_main_regression.csv"), index=False
        )
        print(f"\n  Saved: {results_dir}/table2_main_regression.csv")

    # Summary table
    summary_rows = []
    for model_name, summ in [("linear", linear_summary), ("quadratic", quad_summary)]:
        if summ:
            row = {"model": model_name}
            row.update({f"mean_{k}": v for k, v in summ.get("means", {}).items()})
            row.update({f"mat_{k}": v for k, v in summ.get("mean_abs_t", {}).items()})
            row.update({f"sig_{k}": v for k, v in summ.get("sig_rate", {}).items()})
            summary_rows.append(row)

    if summary_rows:
        pd.DataFrame(summary_rows).to_csv(
            os.path.join(results_dir, "table2_summary.csv"), index=False
        )
        print(f"  Saved: {results_dir}/table2_summary.csv")

    # Per-stock pooled β (for Exp 4)
    betas_df = pd.DataFrame(list(stock_pooled.values()))
    if not betas_df.empty:
        betas_df.to_csv(
            os.path.join(results_dir, "exp3_betas.csv"), index=False
        )
        print(f"  Saved: {results_dir}/exp3_betas.csv")

    # -------------------------------------------------------------------------
    # Console summary
    # -------------------------------------------------------------------------
    n_sub = len(linear_results)
    print("\n" + "=" * 60)
    print("Exp 3 — H1 Regression Summary")
    print("=" * 60)
    print(f"  Subsamples (stock-days) : {n_sub}")
    if linear_summary:
        means = linear_summary.get("means", {})
        mat = linear_summary.get("mean_abs_t", {})
        sig = linear_summary.get("sig_rate", {})
        print(f"  Linear OFI:")
        print(f"    Mean β_OFI    : {means.get('beta_ofi', float('nan')):.6f}")
        print(f"    Mean |t_β|    : {mat.get('t_beta_ofi', float('nan')):.2f}")
        print(f"    % |t| > 1.96  : {100 * sig.get('t_beta_ofi', float('nan')):.1f} %")
        print(f"    Mean R²       : {means.get('rsquared', float('nan')):.4f}")
    print("=" * 60 + "\n")

    return {
        "linear_summary": linear_summary,
        "quad_summary": quad_summary,
        "per_stock_betas": betas_df,
        "n_subsamples": n_sub,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Exp 3: H1 – OFI regression (Table 2)."
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

    print(f"Exp 3: H1 regression for {len(tickers)} stocks × {len(dates)} dates …\n")
    run_exp3(
        tickers=tickers,
        dates=dates,
        grid_dir=args.grid_dir,
        results_dir=args.results_dir,
    )


if __name__ == "__main__":
    main()
