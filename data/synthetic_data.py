"""
synthetic_data.py - Synthetic TAQ quote and trade data generator.

Generates realistic intraday quote and trade data for April 2010 for a
configurable subset of S&P 500 stocks.  All column names match the schema
consumed by src.nbbo and src.grid.

Schema
------
Quotes: timestamp (datetime64[ns]), bid, ask (float64),
        bid_size, ask_size (int64), mode (int64), ex (str)
Trades: timestamp (datetime64[ns]), price (float64),
        size (int64), corr (int64), cond (str)

Usage
-----
    python -m data.synthetic_data                   # generate for 5 stocks
    python -m data.synthetic_data --n_stocks 10     # generate for 10 stocks
    python -m data.synthetic_data --output_dir data/raw
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd

# Ensure project root is importable when run as __main__
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.nbbo import apply_quote_filters, build_nbbo_sequence, apply_spread_filter

# ---------------------------------------------------------------------------
# Universe constants
# ---------------------------------------------------------------------------

TICKERS_50: list[str] = [
    "AMD", "APOL", "AXP", "AZO", "BAC", "BDX", "BK", "BSX", "BTU", "CAT",
    "CB", "CCL", "CINF", "CME", "COH", "COP", "CVH", "DNR", "DVN", "EFX",
    "ETN", "FISV", "HAS", "HCP", "HOT", "KSS", "LLL", "LMT", "M", "MAR",
    "MFE", "MHP", "MHS", "MRK", "MRO", "MWV", "NEM", "OMC", "PCS", "PHM",
    "PKI", "R", "RAI", "SLB", "TE", "TWC", "WHR", "WIN", "WPI", "XTO",
]

TRADING_DAYS_APRIL_2010: list[str] = [
    "2010-04-01", "2010-04-05", "2010-04-06", "2010-04-07", "2010-04-08",
    "2010-04-09", "2010-04-12", "2010-04-13", "2010-04-14", "2010-04-15",
    "2010-04-16", "2010-04-19", "2010-04-20", "2010-04-21", "2010-04-22",
    "2010-04-23", "2010-04-26", "2010-04-27", "2010-04-28", "2010-04-29",
    "2010-04-30",
]

EXCHANGE_CODES: list[str] = [
    "N", "Q", "A", "C", "D", "I", "K", "M", "P", "T", "X", "Y", "Z",
]

# Weights for exchange activity (NYSE and NASDAQ dominant)
_EXCHANGE_WEIGHTS: np.ndarray = np.array(
    [0.25, 0.20, 0.10, 0.07, 0.07, 0.06, 0.05, 0.04, 0.04, 0.03, 0.03, 0.03, 0.03],
    dtype=float,
)
_EXCHANGE_WEIGHTS /= _EXCHANGE_WEIGHTS.sum()

# Mode codes that are filtered out as invalid
_INVALID_MODE_CODES: list[int] = [4, 7, 9, 11, 13, 14, 15, 19, 20, 27, 28]

# Trade condition codes that are filtered out
_INVALID_COND_CODES: list[str] = ["O", "Z", "B", "T", "L", "G", "W", "J", "K"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _ticker_seed(ticker: str) -> int:
    """Derive a deterministic integer seed from a ticker string."""
    return sum(ord(c) * (i + 1) for i, c in enumerate(ticker)) % (2**31)


def _random_walk_prices(
    rng: np.random.Generator,
    n: int,
    start: float,
    vol_per_step: float = 5e-4,
) -> np.ndarray:
    """Geometric Brownian-motion random walk for mid-price.

    Parameters
    ----------
    rng : np.random.Generator
    n : int
        Number of price steps.
    start : float
        Starting mid-price.
    vol_per_step : float
        Per-step return volatility (fraction of price).

    Returns
    -------
    np.ndarray of shape (n,)
    """
    log_returns = rng.normal(0.0, vol_per_step, size=n)
    log_prices = np.log(start) + np.cumsum(log_returns)
    return np.exp(log_prices)


# ---------------------------------------------------------------------------
# Quote generation
# ---------------------------------------------------------------------------


def generate_synthetic_quotes(
    ticker: str,
    date: str,
    seed: int | None = None,
) -> pd.DataFrame:
    """Generate realistic synthetic TAQ quote records for one stock-day.

    Generates approximately 10 000 quote records spanning 09:15–16:10,
    with ~95 % of records within regular trading hours (09:30–16:00).
    The price process is a geometric random walk starting near $51.75.

    Parameters
    ----------
    ticker : str
        Stock ticker symbol.
    date : str
        Trading date in ``YYYY-MM-DD`` format.
    seed : int, optional
        Random seed for reproducibility.  Defaults to a hash of *ticker*.

    Returns
    -------
    pd.DataFrame
        Columns: ``timestamp``, ``bid``, ``ask``, ``bid_size``, ``ask_size``,
        ``mode``, ``ex``.  Not yet filtered; contains invalid-mode and
        out-of-hours records.

    Notes
    -----
    - ~8 % of records carry an invalid mode code drawn from
      {4,7,9,11,13,14,15,19,20,27,28}.
    - ~5 % of records fall outside [09:30, 16:00].
    - Spread = max($0.01, 0.06 % × mid).
    - Bid/ask sizes are log-normally distributed; NBBO size target ~227 shares.
    """
    if seed is None:
        seed = _ticker_seed(ticker)
    rng = np.random.default_rng(seed)

    date_ts = pd.Timestamp(date)
    market_open = date_ts + pd.Timedelta("9h30m")
    market_close = date_ts + pd.Timedelta("16h0m")
    pre_start = date_ts + pd.Timedelta("9h15m")
    post_end = date_ts + pd.Timedelta("16h10m")
    session_secs = int((market_close - market_open).total_seconds())       # 23400
    pre_secs = int((market_open - pre_start).total_seconds())             # 900
    post_secs = int((post_end - market_close).total_seconds())            # 600

    n_total = 10_000
    n_pre_post = int(n_total * 0.05)
    n_pre = n_pre_post // 2
    n_post = n_pre_post - n_pre
    n_market = n_total - n_pre_post

    # Timestamps ----------------------------------------------------------------
    ts_market = market_open + pd.to_timedelta(
        rng.integers(0, session_secs, size=n_market), unit="s"
    )
    ts_pre = pre_start + pd.to_timedelta(
        rng.integers(0, pre_secs, size=n_pre), unit="s"
    )
    ts_post = market_close + pd.to_timedelta(
        rng.integers(0, post_secs, size=n_post), unit="s"
    )
    timestamps = np.concatenate([ts_market.values, ts_pre.values, ts_post.values])
    sort_idx = np.argsort(timestamps)
    timestamps = timestamps[sort_idx]

    n = len(timestamps)

    # Mid-price random walk (start near $51.75, ticker-specific drift)
    start_price = 51.75 + rng.uniform(-5.0, 5.0)
    mid_prices = _random_walk_prices(rng, n, start_price, vol_per_step=5e-4)

    # Spreads: half-spread ≈ max(0.5 cents, 0.03 % of mid)
    half_spreads = np.maximum(0.005, mid_prices * 0.0003)

    # Exchange-level noise shifts the mid slightly
    ex_noise = rng.normal(0.0, 0.004, size=n)

    bids = np.round(mid_prices - half_spreads + ex_noise, 2)
    asks = np.round(mid_prices + half_spreads + ex_noise, 2)
    # Enforce minimum spread of $0.01
    asks = np.maximum(asks, bids + 0.01)

    # Sizes: log-normal targeting ~227 shares at NBBO per side
    # lognormal(mu=log(200), sigma=0.5) → E[X] ≈ 225 per exchange → depth ≈ 227
    raw_bid_sizes = rng.lognormal(np.log(200.0), 0.5, size=n)
    raw_ask_sizes = rng.lognormal(np.log(200.0), 0.5, size=n)
    # Round to nearest 100-share lot, floor at 100
    bid_sizes = (np.round(raw_bid_sizes / 100.0) * 100).astype(int)
    bid_sizes = np.maximum(bid_sizes, 100)
    ask_sizes = (np.round(raw_ask_sizes / 100.0) * 100).astype(int)
    ask_sizes = np.maximum(ask_sizes, 100)

    # Exchange codes (weighted)
    ex_arr = rng.choice(EXCHANGE_CODES, size=n, p=_EXCHANGE_WEIGHTS)

    # Mode codes: mostly 0, ~8 % invalid
    modes = np.zeros(n, dtype=int)
    invalid_mask = rng.random(n) < 0.08
    modes[invalid_mask] = rng.choice(
        _INVALID_MODE_CODES, size=int(invalid_mask.sum())
    )

    df = pd.DataFrame(
        {
            "timestamp": pd.DatetimeIndex(timestamps),
            "bid": bids,
            "ask": asks,
            "bid_size": bid_sizes,
            "ask_size": ask_sizes,
            "mode": modes,
            "ex": ex_arr,
        }
    )
    return df


# ---------------------------------------------------------------------------
# Trade generation
# ---------------------------------------------------------------------------


def generate_synthetic_trades(
    ticker: str,
    date: str,
    nbbo_df: pd.DataFrame,
    seed: int | None = None,
) -> pd.DataFrame:
    """Generate realistic synthetic TAQ trade records for one stock-day.

    Trade prices are anchored to the prevailing NBBO at each simulated
    trade time, producing a realistic bid-ask bounce pattern.  Approximately
    250 trades are generated per stock-day.

    Parameters
    ----------
    ticker : str
        Stock ticker symbol.
    date : str
        Trading date in ``YYYY-MM-DD`` format.
    nbbo_df : pd.DataFrame
        NBBO sequence for the stock-day (output of
        :func:`src.nbbo.build_nbbo_sequence` after filtering).
        Required columns: ``timestamp``, ``nbbo_bid``, ``nbbo_ask``.
    seed : int, optional
        Random seed.  Defaults to a hash of *ticker* + *date*.

    Returns
    -------
    pd.DataFrame
        Columns: ``timestamp``, ``price``, ``size``, ``corr``, ``cond``.
        Contains only records within regular trading hours.

    Notes
    -----
    - ~3 % of records have ``corr`` = 1 (minor correction),
      ~1 % have ``corr`` = 2 (corrected trade), and
      ~0.5 % have an out-of-range ``corr`` value (→ filtered in exp_2).
    - ~2 % of records carry an invalid ``cond`` code (→ filtered in exp_2).
    - Trade sizes target a mean of ~286 shares (lognormal with
      mu=log(250), sigma=0.7).
    """
    if seed is None:
        seed = (_ticker_seed(ticker) + hash(date)) % (2**31)
    rng = np.random.default_rng(seed)

    n_trades = 250
    date_ts = pd.Timestamp(date)
    market_open = date_ts + pd.Timedelta("9h30m")
    market_close = date_ts + pd.Timedelta("16h0m")
    session_secs = int((market_close - market_open).total_seconds())

    # Trade timestamps (uniform within session, sorted)
    trade_ts = market_open + pd.to_timedelta(
        rng.integers(0, session_secs, size=n_trades), unit="s"
    )
    trade_ts = trade_ts.sort_values()

    # Align each trade to the most recent NBBO using merge_asof
    nbbo_sorted = (
        nbbo_df[["timestamp", "nbbo_bid", "nbbo_ask"]]
        .sort_values("timestamp")
        .copy()
    )
    trade_frame = pd.DataFrame({"timestamp": trade_ts})
    merged = pd.merge_asof(
        trade_frame,
        nbbo_sorted,
        on="timestamp",
        direction="backward",
    )

    # Price: 50 % at ask (buy), 40 % at bid (sell), 10 % at mid
    roll = rng.random(n_trades)
    mid = (merged["nbbo_bid"] + merged["nbbo_ask"]) / 2.0
    price = np.where(
        roll < 0.50,
        merged["nbbo_ask"],
        np.where(roll < 0.90, merged["nbbo_bid"], mid),
    )
    # Add tiny sub-penny noise and round to 2 dp
    price = np.round(price + rng.normal(0.0, 0.001, n_trades), 2)
    # Clip to [nbbo_bid - 0.05, nbbo_ask + 0.05]
    price = np.clip(
        price,
        merged["nbbo_bid"].to_numpy() - 0.05,
        merged["nbbo_ask"].to_numpy() + 0.05,
    )

    # Sizes: lognormal targeting mean ≈ 286 shares
    raw_sizes = rng.lognormal(np.log(250.0), 0.7, size=n_trades)
    sizes = (np.round(raw_sizes / 100.0) * 100).astype(int)
    sizes = np.maximum(sizes, 100)

    # Correction codes
    corr_roll = rng.random(n_trades)
    corr = np.where(
        corr_roll < 0.955, 0,
        np.where(corr_roll < 0.985, 1,
        np.where(corr_roll < 0.995, 2, 5))  # last 0.5 % → invalid (>2)
    ).astype(int)

    # Condition codes: mostly empty string, ~2 % invalid
    cond = np.full(n_trades, "", dtype=object)
    invalid_cond_mask = rng.random(n_trades) < 0.02
    cond[invalid_cond_mask] = rng.choice(_INVALID_COND_CODES, size=int(invalid_cond_mask.sum()))

    df = pd.DataFrame(
        {
            "timestamp": trade_ts.values,
            "price": price,
            "size": sizes,
            "corr": corr,
            "cond": cond,
        }
    )
    return df


# ---------------------------------------------------------------------------
# Batch generation
# ---------------------------------------------------------------------------


def generate_all_stock_data(
    tickers: list[str],
    dates: list[str],
    output_dir: str,
    n_stocks: int = 5,
) -> None:
    """Generate and persist synthetic quotes and trades for a stock subset.

    For each of the first *n_stocks* tickers × each date in *dates*:

    1. Generate raw quote records via :func:`generate_synthetic_quotes`.
    2. Apply :func:`src.nbbo.apply_quote_filters`, build NBBO via
       :func:`src.nbbo.build_nbbo_sequence`, and apply
       :func:`src.nbbo.apply_spread_filter`.
    3. Generate trade records via :func:`generate_synthetic_trades`
       using the NBBO from step 2.
    4. Write quotes to ``{output_dir}/quotes/{ticker}_{date}_quotes.parquet``.
    5. Write trades to ``{output_dir}/trades/{ticker}_{date}_trades.parquet``.

    Parameters
    ----------
    tickers : list[str]
        Full ticker list; only the first *n_stocks* are processed.
    dates : list[str]
        List of date strings in ``YYYY-MM-DD`` format.
    output_dir : str
        Root output directory.  Sub-directories ``quotes/`` and ``trades/``
        are created automatically.
    n_stocks : int, optional
        Number of stocks to generate (default 5 for speed).

    Raises
    ------
    ValueError
        If *n_stocks* > len(*tickers*).
    OSError
        If output directories cannot be created.
    """
    if n_stocks > len(tickers):
        raise ValueError(
            f"generate_all_stock_data: n_stocks={n_stocks} > len(tickers)={len(tickers)}"
        )

    quotes_dir = os.path.join(output_dir, "quotes")
    trades_dir = os.path.join(output_dir, "trades")
    os.makedirs(quotes_dir, exist_ok=True)
    os.makedirs(trades_dir, exist_ok=True)

    subset = tickers[:n_stocks]
    total = len(subset) * len(dates)
    done = 0

    for ticker in subset:
        for date in dates:
            # --- Quotes ---
            quotes_raw = generate_synthetic_quotes(ticker, date)
            quotes_path = os.path.join(quotes_dir, f"{ticker}_{date}_quotes.parquet")
            quotes_raw.to_parquet(quotes_path, index=False)

            # --- Build minimal NBBO for trade anchoring ---
            try:
                filtered = apply_quote_filters(quotes_raw)
                if filtered.empty:
                    nbbo_for_trades = pd.DataFrame(
                        columns=["timestamp", "nbbo_bid", "nbbo_ask"]
                    )
                else:
                    nbbo_seq = build_nbbo_sequence(filtered)
                    nbbo_for_trades = apply_spread_filter(nbbo_seq)
            except Exception:
                nbbo_for_trades = pd.DataFrame(
                    columns=["timestamp", "nbbo_bid", "nbbo_ask"]
                )

            # --- Trades ---
            if nbbo_for_trades.empty:
                # Fallback: generate trades with constant mid-price
                nbbo_fallback = pd.DataFrame(
                    {
                        "timestamp": [pd.Timestamp(f"{date} 09:30:00")],
                        "nbbo_bid": [51.74],
                        "nbbo_ask": [51.76],
                    }
                )
                trades = generate_synthetic_trades(ticker, date, nbbo_fallback)
            else:
                trades = generate_synthetic_trades(ticker, date, nbbo_for_trades)

            trades_path = os.path.join(trades_dir, f"{ticker}_{date}_trades.parquet")
            trades.to_parquet(trades_path, index=False)

            done += 1
            pct = 100 * done / total
            print(
                f"  [{pct:5.1f}%] generated {ticker} {date} "
                f"({len(quotes_raw):,} quotes, {len(trades):,} trades)",
                flush=True,
            )

    print(
        f"\nDone. {total} stock-days saved under '{output_dir}'.",
        flush=True,
    )


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Generate synthetic April 2010 TAQ data for NBBO analysis."
    )
    p.add_argument(
        "--n_stocks",
        type=int,
        default=5,
        help="Number of stocks to generate (default: 5).",
    )
    p.add_argument(
        "--output_dir",
        type=str,
        default="data/raw",
        help="Root output directory (default: data/raw).",
    )
    return p


def main() -> None:
    parser = _build_arg_parser()
    args = parser.parse_args()

    print(
        f"Generating synthetic data for {args.n_stocks} stocks × "
        f"{len(TRADING_DAYS_APRIL_2010)} trading days …"
    )
    generate_all_stock_data(
        tickers=TICKERS_50,
        dates=TRADING_DAYS_APRIL_2010,
        output_dir=args.output_dir,
        n_stocks=args.n_stocks,
    )


if __name__ == "__main__":
    main()
