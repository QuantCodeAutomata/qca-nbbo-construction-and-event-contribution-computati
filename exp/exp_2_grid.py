"""
exp_2_grid.py - Experiment 2: Trade classification → 10-second grid construction.

Pipeline
--------
For each (ticker, date) pair:
  1. Load NBBO events parquet from Exp 1
     (``{nbbo_dir}/{ticker}_{date}_nbbo.parquet``).
  2. Load raw trades parquet (``{trade_dir}/trades/{ticker}_{date}_trades.parquet``).
  3. :func:`src.grid.apply_trade_filters`           – remove invalid prints.
  4. :func:`src.grid.classify_trades_quote_test`    – label buys (+1) / sells (−1).
  5. :func:`src.grid.build_10sec_grid`              – 2340-bin grid.
  6. :func:`src.grid.compute_average_depth`         – per-half-hour AD.
  7. Merge AD into the grid on ``half_hour``.
  8. Save grid to ``{output_dir}/{ticker}_{date}_grid.parquet``.

Validation checkpoint
---------------------
After processing all stock-days the script prints:

- **Total volume** across all generated stock-days (expected ≈ 7,512,376 shares
  for 5 stocks × 21 days).
- **Mean stock price** (expected ≈ $51.75).

Usage
-----
    python exp/exp_2_grid.py
    python exp/exp_2_grid.py --n_stocks 10 \\
                              --nbbo_dir results/nbbo \\
                              --trade_dir data/raw \\
                              --output_dir results/grid
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.grid import (
    apply_trade_filters,
    classify_trades_quote_test,
    build_10sec_grid,
    compute_average_depth,
)
from data.synthetic_data import TICKERS_50, TRADING_DAYS_APRIL_2010


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _load_nbbo(ticker: str, date: str, nbbo_dir: str) -> pd.DataFrame:
    path = os.path.join(nbbo_dir, f"{ticker}_{date}_nbbo.parquet")
    if not os.path.exists(path):
        raise FileNotFoundError(f"NBBO file not found: {path}")
    df = pd.read_parquet(path)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df


def _load_trades(ticker: str, date: str, trade_dir: str) -> pd.DataFrame:
    path = os.path.join(trade_dir, "trades", f"{ticker}_{date}_trades.parquet")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Trade file not found: {path}")
    df = pd.read_parquet(path)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df


# ---------------------------------------------------------------------------
# Core pipeline
# ---------------------------------------------------------------------------


def run_exp2(
    tickers: list[str],
    dates: list[str],
    nbbo_dir: str,
    trade_dir: str,
    output_dir: str,
) -> dict:
    """Build 10-second grids for all (ticker, date) pairs.

    Parameters
    ----------
    tickers : list[str]
        Ticker symbols to process.
    dates : list[str]
        Trading dates in ``YYYY-MM-DD`` format.
    nbbo_dir : str
        Directory containing NBBO parquets from Exp 1.
    trade_dir : str
        Root data directory containing a ``trades/`` sub-directory.
    output_dir : str
        Directory where grid parquet files are written.

    Returns
    -------
    dict
        Validation statistics: ``'total_volume'``, ``'mean_price'``,
        ``'n_processed'``, ``'n_failed'``.
    """
    os.makedirs(output_dir, exist_ok=True)

    total_volume_all: float = 0.0
    all_prices: list[float] = []
    n_processed = 0
    n_failed = 0

    total = len(tickers) * len(dates)
    done = 0

    for ticker in tickers:
        for date in dates:
            done += 1
            pct = 100 * done / total
            grid_path = os.path.join(output_dir, f"{ticker}_{date}_grid.parquet")

            try:
                # Load NBBO events from Exp 1
                nbbo_df = _load_nbbo(ticker, date, nbbo_dir)

                # Load and filter trades
                trades_raw = _load_trades(ticker, date, trade_dir)
                trades_filtered = apply_trade_filters(trades_raw)

                # Classify trades with quote test
                # Handle case where no valid trades pass the filter
                if trades_filtered.empty:
                    classified_trades = trades_filtered.copy()
                    classified_trades["trade_dir"] = pd.Series(dtype=int)
                else:
                    classified_trades = classify_trades_quote_test(
                        trades_filtered, nbbo_df
                    )

                # Build 10-second grid (2340 bins)
                grid = build_10sec_grid(nbbo_df, classified_trades, date)

                # Compute per-half-hour average depth and merge into grid
                ad_df = compute_average_depth(nbbo_df)
                grid = grid.merge(ad_df, on="half_hour", how="left")

                # Metadata columns
                grid["ticker"] = ticker
                grid["date"] = date

                # Persist
                grid.to_parquet(grid_path, index=False)

                # Validation statistics
                day_volume = float(grid["VOL_k"].sum())
                total_volume_all += day_volume
                valid_mids = grid["mid_k"].dropna()
                if not valid_mids.empty:
                    all_prices.append(float(valid_mids.mean()))

                n_processed += 1
                print(
                    f"  [{pct:5.1f}%] {ticker} {date}: "
                    f"volume={day_volume:,.0f} shs, "
                    f"mid=${valid_mids.mean():.2f}",
                    flush=True,
                )

            except FileNotFoundError as exc:
                print(f"  [{pct:5.1f}%] SKIP {ticker} {date} – {exc}")
                n_failed += 1
            except ValueError as exc:
                print(f"  [{pct:5.1f}%] ERROR {ticker} {date} – {exc}")
                n_failed += 1

    # -------------------------------------------------------------------------
    # Validation checkpoint
    # -------------------------------------------------------------------------
    mean_price = float(np.nanmean(all_prices)) if all_prices else float("nan")

    print("\n" + "=" * 60)
    print("Exp 2 — Validation Checkpoint")
    print("=" * 60)
    print(f"  Stock-days processed  : {n_processed:,}")
    print(f"  Stock-days failed     : {n_failed:,}")
    print(
        f"  Total volume          : {total_volume_all:,.0f} shares  "
        f"(expected ≈ 7,512,376 for 5 stocks × 21 days)"
    )
    print(
        f"  Mean stock price      : ${mean_price:.2f}  "
        f"(expected ≈ $51.75)"
    )
    print("=" * 60 + "\n")

    return {
        "total_volume": total_volume_all,
        "mean_price": mean_price,
        "n_processed": n_processed,
        "n_failed": n_failed,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Exp 2: Classify trades and build 10-second OFI grids."
    )
    p.add_argument("--n_stocks", type=int, default=5)
    p.add_argument("--nbbo_dir", default="results/nbbo")
    p.add_argument("--trade_dir", default="data/raw")
    p.add_argument("--output_dir", default="results/grid")
    return p


def main() -> None:
    parser = _build_arg_parser()
    args = parser.parse_args()

    tickers = TICKERS_50[: args.n_stocks]
    dates = TRADING_DAYS_APRIL_2010

    print(
        f"Exp 2: Processing {len(tickers)} stocks × {len(dates)} dates …\n"
    )
    run_exp2(
        tickers=tickers,
        dates=dates,
        nbbo_dir=args.nbbo_dir,
        trade_dir=args.trade_dir,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
