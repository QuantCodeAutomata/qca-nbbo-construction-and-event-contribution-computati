"""
exp_1_nbbo.py - Experiment 1: Quote filtering → NBBO construction → OFI events.

Pipeline
--------
For each (ticker, date) pair:
  1. Load raw synthetic quotes from ``{data_dir}/quotes/{ticker}_{date}_quotes.parquet``.
  2. :func:`src.nbbo.apply_quote_filters`   – remove invalid mode codes and
     out-of-hours records.
  3. :func:`src.nbbo.build_nbbo_sequence`   – compute per-exchange NBBO state
     after each quote update.
  4. :func:`src.nbbo.apply_spread_filter`   – flag records whose spread exceeds
     the 95th percentile.
  5. :func:`src.nbbo.compute_event_contribution` – compute per-event OFI
     contribution ``e_n``.
  6. Save NBBO events to ``{output_dir}/{ticker}_{date}_nbbo.parquet``.

Validation checkpoint
---------------------
After processing all stock-days the script prints:

- **Mean quote-to-trade ratio** (expected ≈ 40:1).
- **Mean best-quote depth** (expected ≈ 227 shares per side).

Usage
-----
    python exp/exp_1_nbbo.py
    python exp/exp_1_nbbo.py --n_stocks 10 --data_dir data/raw \\
                              --output_dir results/nbbo
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd

# Ensure project root is on the path when run as a script
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.nbbo import (
    EXCLUDED_MODE_CODES,
    apply_quote_filters,
    apply_spread_filter,
    build_nbbo_sequence,
    compute_event_contribution,
    process_stock_day,
)
from src.grid import apply_trade_filters
from data.synthetic_data import TICKERS_50, TRADING_DAYS_APRIL_2010


# ---------------------------------------------------------------------------
# Core pipeline
# ---------------------------------------------------------------------------


def _load_quotes(ticker: str, date: str, data_dir: str) -> pd.DataFrame:
    """Load raw quotes parquet for a single stock-day."""
    path = os.path.join(data_dir, "quotes", f"{ticker}_{date}_quotes.parquet")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Quote file not found: {path}")
    return pd.read_parquet(path)


def _load_trades(ticker: str, date: str, data_dir: str) -> pd.DataFrame:
    """Load raw trades parquet for a single stock-day (for validation only)."""
    path = os.path.join(data_dir, "trades", f"{ticker}_{date}_trades.parquet")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Trade file not found: {path}")
    return pd.read_parquet(path)


def run_exp1(
    tickers: list[str],
    dates: list[str],
    data_dir: str,
    output_dir: str,
) -> dict:
    """Run the NBBO pipeline for all (ticker, date) pairs.

    Parameters
    ----------
    tickers : list[str]
        Ticker symbols to process.
    dates : list[str]
        Trading dates in ``YYYY-MM-DD`` format.
    data_dir : str
        Root directory containing ``quotes/`` and ``trades/`` sub-directories.
    output_dir : str
        Directory where NBBO parquet files are written.

    Returns
    -------
    dict
        Validation statistics:
        ``'mean_q2t_ratio'``, ``'mean_depth'``, ``'n_processed'``,
        ``'n_failed'``.
    """
    os.makedirs(output_dir, exist_ok=True)

    q2t_ratios: list[float] = []
    depths: list[float] = []
    n_processed = 0
    n_failed = 0

    total = len(tickers) * len(dates)
    done = 0

    for ticker in tickers:
        for date in dates:
            done += 1
            pct = 100 * done / total

            nbbo_path = os.path.join(output_dir, f"{ticker}_{date}_nbbo.parquet")

            try:
                # Load and filter quotes
                quotes_raw = _load_quotes(ticker, date, data_dir)
                filtered_quotes = apply_quote_filters(quotes_raw)

                if filtered_quotes.empty:
                    print(
                        f"  [{pct:5.1f}%] SKIP {ticker} {date} – no valid quotes after filtering"
                    )
                    n_failed += 1
                    continue

                # Full NBBO pipeline (steps 3–5)
                nbbo_df = process_stock_day(quotes_raw, ticker, date)

                # Save NBBO events
                nbbo_df.to_parquet(nbbo_path, index=False)

                # --- Validation statistics ---
                # Depth: mean of (bid_size + ask_size) / 2 per NBBO event
                depth_series = (
                    nbbo_df["nbbo_bid_size"] + nbbo_df["nbbo_ask_size"]
                ) / 2.0
                mean_depth_day = float(depth_series.mean())

                # Quote-to-trade ratio: filtered quotes / filtered trades
                try:
                    trades_raw = _load_trades(ticker, date, data_dir)
                    filtered_trades = apply_trade_filters(trades_raw)
                    n_filt_q = len(filtered_quotes)
                    n_filt_t = max(len(filtered_trades), 1)
                    q2t = n_filt_q / n_filt_t
                except FileNotFoundError:
                    q2t = float("nan")

                if np.isfinite(q2t):
                    q2t_ratios.append(q2t)
                depths.append(mean_depth_day)
                n_processed += 1

                print(
                    f"  [{pct:5.1f}%] {ticker} {date}: "
                    f"{len(nbbo_df):,} NBBO events, "
                    f"depth={mean_depth_day:.0f} shs, "
                    f"Q/T={q2t:.1f}",
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
    mean_q2t = float(np.nanmean(q2t_ratios)) if q2t_ratios else float("nan")
    mean_depth = float(np.nanmean(depths)) if depths else float("nan")

    print("\n" + "=" * 60)
    print("Exp 1 — Validation Checkpoint")
    print("=" * 60)
    print(f"  Stock-days processed : {n_processed:,}")
    print(f"  Stock-days failed    : {n_failed:,}")
    print(
        f"  Mean quote-to-trade ratio : {mean_q2t:.1f}  "
        f"(expected ≈ 40)"
    )
    print(
        f"  Mean best-quote depth     : {mean_depth:.0f} shares  "
        f"(expected ≈ 227)"
    )
    print("=" * 60 + "\n")

    return {
        "mean_q2t_ratio": mean_q2t,
        "mean_depth": mean_depth,
        "n_processed": n_processed,
        "n_failed": n_failed,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Exp 1: Build NBBO event sequences from synthetic TAQ quotes."
    )
    p.add_argument(
        "--n_stocks",
        type=int,
        default=5,
        help="Number of stocks to process from TICKERS_50 (default: 5).",
    )
    p.add_argument(
        "--data_dir",
        default="data/raw",
        help="Root data directory with quotes/ and trades/ sub-dirs (default: data/raw).",
    )
    p.add_argument(
        "--output_dir",
        default="results/nbbo",
        help="Directory for NBBO parquet output (default: results/nbbo).",
    )
    return p


def main() -> None:
    parser = _build_arg_parser()
    args = parser.parse_args()

    tickers = TICKERS_50[: args.n_stocks]
    dates = TRADING_DAYS_APRIL_2010

    print(
        f"Exp 1: Processing {len(tickers)} stocks × {len(dates)} dates "
        f"from '{args.data_dir}' …\n"
    )
    stats = run_exp1(
        tickers=tickers,
        dates=dates,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
    )

    if stats["n_failed"] > 0:
        print(
            f"Warning: {stats['n_failed']} stock-day(s) failed.  "
            "Check that synthetic data has been generated first:\n"
            "  python -m data.synthetic_data\n"
        )


if __name__ == "__main__":
    main()
