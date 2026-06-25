"""
download_data.py - WRDS TAQ data download pipeline.

Downloads raw TAQ (Trade and Quote) data from the Wharton Research Data
Services (WRDS) TAQ database for a specified universe of stocks and dates.

WRDS Setup
----------
1. Install the WRDS Python library::

       pip install wrds

2. Create a WRDS account at https://wrds-www.wharton.upenn.edu/ and request
   access to the *TAQ* (Trade and Quote) dataset.

3. Store your credentials in ``~/.pgpass`` or supply the username to
   :func:`download_all_data`; the library will prompt for your password
   on first run and cache the session.

   Alternatively set the environment variables::

       WRDS_USERNAME=yourname
       WRDS_PASSWORD=yourpassword  # NOT recommended; use pgpass instead

4. TAQ data lives in the ``taqmsec`` (millisecond) or ``taq`` (daily)
   schemas depending on your subscription.  This module targets the
   **Daily TAQ** (``taqmsec``) schema available from 2003 onwards.

Column mapping from TAQ to project schema
------------------------------------------
Quotes (``ctm_*`` tables):
   - ``TIME_M``    → ``timestamp``
   - ``BID``       → ``bid``
   - ``ASK``       → ``ask``
   - ``BIDSIZ``    → ``bid_size``
   - ``ASKSIZ``    → ``ask_size``
   - ``QU_COND``   → ``mode``   (cast to int via condition-to-code map)
   - ``EX``        → ``ex``

Trades (``ct_*`` tables):
   - ``TIME_M``    → ``timestamp``
   - ``PRICE``     → ``price``
   - ``SIZE``      → ``size``
   - ``TR_CORR``   → ``corr``
   - ``TR_SCN_CD`` → ``cond``
"""

import os
import sys
from pathlib import Path

import pandas as pd

# ---------------------------------------------------------------------------
# Quote condition-to-mode-code mapping (TAQ ``QU_COND`` field)
# ---------------------------------------------------------------------------

_QU_COND_TO_MODE: dict[str, int] = {
    "R": 0,   # regular
    "A": 0,   # automatic
    "B": 4,   # invalid (fast market)
    "C": 7,   # invalid (closing)
    "D": 9,   # invalid
    "E": 11,  # invalid
    "F": 13,  # invalid
    "G": 14,  # invalid
    "H": 15,  # invalid
    "I": 19,  # invalid
    "J": 20,  # invalid
    "K": 27,  # invalid
    "L": 28,  # invalid
    "M": 0,   # manual
    "N": 0,   # non-firm
    "O": 0,   # opening
    "Q": 0,   # regular
    "S": 0,   # specialist ask
    "T": 0,   # terminal
    "U": 0,   # unpriced
    "W": 0,   # weighted average
    "X": 0,   # cross trade
    "Y": 0,   # yes, regular
    "Z": 0,   # regular
}

_DEFAULT_MODE: int = 0


# ---------------------------------------------------------------------------
# Quote download
# ---------------------------------------------------------------------------


def wrds_download_quotes(
    ticker: str,
    date: str,
    db_connection,
) -> pd.DataFrame:
    """Download TAQ quote records for a single stock-day from WRDS.

    Queries the ``taqmsec.ctm_{date_nodash}`` table (millisecond TAQ
    Daily) for the specified ticker.  The raw TAQ columns are renamed and
    cast to the project schema consumed by ``src.nbbo.apply_quote_filters``.

    Parameters
    ----------
    ticker : str
        Stock ticker symbol (e.g., ``'SLB'``).
    date : str
        Trading date in ``YYYY-MM-DD`` format.
    db_connection : wrds.Connection
        An open WRDS database connection, obtained via::

            import wrds
            db = wrds.Connection(wrds_username='yourname')

    Returns
    -------
    pd.DataFrame
        Columns: ``timestamp`` (datetime64[ns]), ``bid`` (float64),
        ``ask`` (float64), ``bid_size`` (int64), ``ask_size`` (int64),
        ``mode`` (int64), ``ex`` (str).
        All rows are included (no filtering applied here).

    Raises
    ------
    ValueError
        If *ticker* or *date* is empty.
    RuntimeError
        If the WRDS query fails or returns an empty result.
    """
    if not ticker:
        raise ValueError("wrds_download_quotes: ticker must be non-empty")
    if not date:
        raise ValueError("wrds_download_quotes: date must be non-empty")

    date_nodash = date.replace("-", "")
    table = f"taqmsec.ctm_{date_nodash}"
    sym_upper = ticker.upper()

    sql = f"""
        SELECT
            (DATE '{date}' + TIME_M) AS timestamp,
            BID                      AS bid,
            ASK                      AS ask,
            BIDSIZ                   AS bid_size,
            ASKSIZ                   AS ask_size,
            QU_COND                  AS qu_cond,
            EX                       AS ex
        FROM {table}
        WHERE SYM_ROOT = '{sym_upper}'
          AND BID > 0
          AND ASK > 0
    """

    try:
        raw = db_connection.raw_sql(sql, date_cols=["timestamp"])
    except Exception as exc:
        raise RuntimeError(
            f"wrds_download_quotes: query failed for {ticker} on {date}: {exc}"
        ) from exc

    if raw is None or raw.empty:
        raise RuntimeError(
            f"wrds_download_quotes: no records returned for {ticker} on {date}"
        )

    # Map condition code to integer mode code
    raw["mode"] = (
        raw["qu_cond"]
        .str.strip()
        .map(_QU_COND_TO_MODE)
        .fillna(_DEFAULT_MODE)
        .astype(int)
    )
    raw.drop(columns=["qu_cond"], inplace=True)
    raw["ex"] = raw["ex"].str.strip().str.upper()

    # Cast numeric columns
    raw["bid"] = pd.to_numeric(raw["bid"], errors="coerce")
    raw["ask"] = pd.to_numeric(raw["ask"], errors="coerce")
    raw["bid_size"] = pd.to_numeric(raw["bid_size"], errors="coerce").fillna(0).astype(int)
    raw["ask_size"] = pd.to_numeric(raw["ask_size"], errors="coerce").fillna(0).astype(int)

    raw = raw.dropna(subset=["timestamp", "bid", "ask"])
    raw = raw.sort_values("timestamp").reset_index(drop=True)

    return raw[["timestamp", "bid", "ask", "bid_size", "ask_size", "mode", "ex"]]


# ---------------------------------------------------------------------------
# Trade download
# ---------------------------------------------------------------------------


def wrds_download_trades(
    ticker: str,
    date: str,
    db_connection,
) -> pd.DataFrame:
    """Download TAQ trade records for a single stock-day from WRDS.

    Queries the ``taqmsec.ct_{date_nodash}`` table for the specified ticker.
    Raw TAQ columns are renamed and cast to the project schema consumed by
    ``src.grid.apply_trade_filters``.

    Parameters
    ----------
    ticker : str
        Stock ticker symbol.
    date : str
        Trading date in ``YYYY-MM-DD`` format.
    db_connection : wrds.Connection
        Open WRDS database connection.

    Returns
    -------
    pd.DataFrame
        Columns: ``timestamp`` (datetime64[ns]), ``price`` (float64),
        ``size`` (int64), ``corr`` (int64), ``cond`` (str).
        All rows are included (no filtering applied here).

    Raises
    ------
    ValueError
        If *ticker* or *date* is empty.
    RuntimeError
        If the WRDS query fails or returns an empty result.
    """
    if not ticker:
        raise ValueError("wrds_download_trades: ticker must be non-empty")
    if not date:
        raise ValueError("wrds_download_trades: date must be non-empty")

    date_nodash = date.replace("-", "")
    table = f"taqmsec.ct_{date_nodash}"
    sym_upper = ticker.upper()

    sql = f"""
        SELECT
            (DATE '{date}' + TIME_M) AS timestamp,
            PRICE                    AS price,
            SIZE                     AS size,
            TR_CORR                  AS corr,
            TR_SCN_CD                AS cond
        FROM {table}
        WHERE SYM_ROOT = '{sym_upper}'
          AND PRICE > 0
          AND SIZE  > 0
    """

    try:
        raw = db_connection.raw_sql(sql, date_cols=["timestamp"])
    except Exception as exc:
        raise RuntimeError(
            f"wrds_download_trades: query failed for {ticker} on {date}: {exc}"
        ) from exc

    if raw is None or raw.empty:
        raise RuntimeError(
            f"wrds_download_trades: no records returned for {ticker} on {date}"
        )

    raw["price"] = pd.to_numeric(raw["price"], errors="coerce")
    raw["size"] = pd.to_numeric(raw["size"], errors="coerce").fillna(0).astype(int)
    raw["corr"] = pd.to_numeric(raw["corr"], errors="coerce").fillna(0).astype(int)
    raw["cond"] = raw["cond"].fillna("").str.strip().str.upper()

    raw = raw.dropna(subset=["timestamp", "price"])
    raw = raw.sort_values("timestamp").reset_index(drop=True)

    return raw[["timestamp", "price", "size", "corr", "cond"]]


# ---------------------------------------------------------------------------
# Full download pipeline
# ---------------------------------------------------------------------------


def download_all_data(
    tickers: list[str],
    dates: list[str],
    wrds_username: str,
    output_dir: str,
) -> None:
    """Download TAQ quotes and trades for all tickers × dates from WRDS.

    Iterates over every (ticker, date) pair, downloads quotes and trades via
    :func:`wrds_download_quotes` and :func:`wrds_download_trades`, and
    persists each result as a Parquet file.

    Output files
    ------------
    - ``{output_dir}/quotes/{ticker}_{date}_quotes.parquet``
    - ``{output_dir}/trades/{ticker}_{date}_trades.parquet``

    Skips any (ticker, date) pair for which a file already exists on disk
    (resume-friendly).

    Parameters
    ----------
    tickers : list[str]
        List of ticker symbols to download.
    dates : list[str]
        List of date strings in ``YYYY-MM-DD`` format.
    wrds_username : str
        WRDS account username.  The WRDS library will prompt for your
        password on first run (stored securely in the system keyring).
    output_dir : str
        Root directory for output files.

    Raises
    ------
    ImportError
        If the ``wrds`` package is not installed.
    RuntimeError
        If the WRDS connection cannot be established.

    Notes
    -----
    A rate-limit of ~1 query/second is enforced automatically by the WRDS
    library.  For the full 50-stock × 21-day universe (~2100 queries) expect
    roughly 35–60 minutes of download time on a standard internet connection.
    """
    try:
        import wrds  # type: ignore[import]
    except ImportError as exc:
        raise ImportError(
            "The 'wrds' package is required for live data download.  "
            "Install it with: pip install wrds"
        ) from exc

    quotes_dir = os.path.join(output_dir, "quotes")
    trades_dir = os.path.join(output_dir, "trades")
    os.makedirs(quotes_dir, exist_ok=True)
    os.makedirs(trades_dir, exist_ok=True)

    print(f"Connecting to WRDS as '{wrds_username}' …")
    try:
        db = wrds.Connection(wrds_username=wrds_username)
    except Exception as exc:
        raise RuntimeError(f"WRDS connection failed: {exc}") from exc

    total = len(tickers) * len(dates)
    done = 0
    errors: list[str] = []

    for ticker in tickers:
        for date in dates:
            done += 1
            pct = 100 * done / total

            quotes_path = os.path.join(quotes_dir, f"{ticker}_{date}_quotes.parquet")
            trades_path = os.path.join(trades_dir, f"{ticker}_{date}_trades.parquet")

            # Skip already-downloaded pairs
            if Path(quotes_path).exists() and Path(trades_path).exists():
                print(f"  [{pct:5.1f}%] skip {ticker} {date} (exists)")
                continue

            print(f"  [{pct:5.1f}%] downloading {ticker} {date} …", end="", flush=True)

            # Quotes
            if not Path(quotes_path).exists():
                try:
                    q_df = wrds_download_quotes(ticker, date, db)
                    q_df.to_parquet(quotes_path, index=False)
                    print(f" quotes({len(q_df):,})", end="", flush=True)
                except RuntimeError as exc:
                    msg = f"{ticker} {date} quotes: {exc}"
                    errors.append(msg)
                    print(f" QUOTES_ERROR({exc})", end="", flush=True)

            # Trades
            if not Path(trades_path).exists():
                try:
                    t_df = wrds_download_trades(ticker, date, db)
                    t_df.to_parquet(trades_path, index=False)
                    print(f" trades({len(t_df):,})", end="", flush=True)
                except RuntimeError as exc:
                    msg = f"{ticker} {date} trades: {exc}"
                    errors.append(msg)
                    print(f" TRADES_ERROR({exc})", end="", flush=True)

            print()

    db.close()

    print(f"\nDownload complete.  {total - len(errors)} / {total} pairs succeeded.")
    if errors:
        print(f"Errors ({len(errors)}):")
        for e in errors:
            print(f"  {e}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    import argparse

    from data.synthetic_data import TICKERS_50, TRADING_DAYS_APRIL_2010

    parser = argparse.ArgumentParser(
        description="Download April 2010 TAQ data from WRDS."
    )
    parser.add_argument("--wrds_username", required=True, help="WRDS account username.")
    parser.add_argument(
        "--output_dir", default="data/raw", help="Root output directory."
    )
    parser.add_argument(
        "--tickers",
        nargs="+",
        default=None,
        help="Subset of tickers (default: all 50).",
    )
    args = parser.parse_args()

    tickers = args.tickers if args.tickers else TICKERS_50
    download_all_data(
        tickers=tickers,
        dates=TRADING_DAYS_APRIL_2010,
        wrds_username=args.wrds_username,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
