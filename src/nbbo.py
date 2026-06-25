"""
nbbo.py - NBBO construction and event contribution module.

Implements the full pipeline for building the National Best Bid and Offer
(NBBO) sequence from TAQ quote data, computing per-event OFI contributions,
and validating the resulting output.

References:
    Cont, Kukanov & Stoikov (2014). The price impact of order book events.
    TAQ data processing follows standard market-microstructure conventions.
"""

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.regression.linear_model import OLS


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MARKET_OPEN = pd.Timestamp("09:30:00").time()
MARKET_CLOSE = pd.Timestamp("16:00:00").time()

EXCLUDED_MODE_CODES = {4, 7, 9, 11, 13, 14, 15, 19, 20, 27, 28}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def apply_quote_filters(df: pd.DataFrame) -> pd.DataFrame:
    """Filter TAQ quote records to retain only clean, actionable quotes.

    Applies the following filters in sequence:
      1. Time window: [09:30, 16:00] inclusive on the time-of-day component.
      2. Positive bid price, ask price, bid size, and ask size.
      3. Exclusion of quotes whose mode code is in the set
         {4, 7, 9, 11, 13, 14, 15, 19, 20, 27, 28}.

    Parameters
    ----------
    df : pd.DataFrame
        Raw TAQ quote data.  Expected columns:
          - ``timestamp`` : datetime64[ns] – quote timestamp
          - ``bid``       : float          – bid price
          - ``ask``       : float          – ask price
          - ``bid_size``  : float/int      – bid depth (shares)
          - ``ask_size``  : float/int      – ask depth (shares)
          - ``mode``      : int            – TAQ quote condition / mode code
          - ``ex``        : str            – exchange identifier

    Returns
    -------
    pd.DataFrame
        Filtered copy of *df*, sorted by ``timestamp``, with the original
        index reset.

    Raises
    ------
    ValueError
        If any required column is missing from *df*.
    """
    required = {"timestamp", "bid", "ask", "bid_size", "ask_size", "mode", "ex"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"apply_quote_filters: missing columns {missing}")

    ts = pd.to_datetime(df["timestamp"])
    time_of_day = ts.dt.time

    mask = (
        (time_of_day >= MARKET_OPEN)
        & (time_of_day <= MARKET_CLOSE)
        & (df["bid"] > 0)
        & (df["ask"] > 0)
        & (df["bid_size"] > 0)
        & (df["ask_size"] > 0)
        & (~df["mode"].isin(EXCLUDED_MODE_CODES))
    )

    out = df.loc[mask].copy()
    out["timestamp"] = pd.to_datetime(out["timestamp"])
    out = out.sort_values("timestamp").reset_index(drop=True)
    return out


def build_nbbo_sequence(filtered_df: pd.DataFrame) -> pd.DataFrame:
    """Construct the NBBO sequence by scanning quotes chronologically.

    For each incoming quote the per-exchange best bid/ask is updated in an
    in-memory ``exchange_matrix`` dictionary.  After every update the NBBO
    is derived as:
      - ``nbbo_bid``      = max over all exchanges of their best bid
      - ``nbbo_ask``      = min over all exchanges of their best ask
      - ``nbbo_bid_size`` = sum of bid sizes for exchanges quoting at nbbo_bid
      - ``nbbo_ask_size`` = sum of ask sizes for exchanges quoting at nbbo_ask
      - ``spread``        = nbbo_ask − nbbo_bid

    Each row in the returned DataFrame represents the NBBO *after* the
    corresponding quote update.

    Parameters
    ----------
    filtered_df : pd.DataFrame
        Quote data that has already passed through :func:`apply_quote_filters`.
        Required columns: ``timestamp``, ``ex``, ``bid``, ``ask``,
        ``bid_size``, ``ask_size``.

    Returns
    -------
    pd.DataFrame
        Columns: ``timestamp``, ``nbbo_bid``, ``nbbo_bid_size``,
        ``nbbo_ask``, ``nbbo_ask_size``, ``spread``.
        Index is reset and consecutive.

    Raises
    ------
    ValueError
        If *filtered_df* is empty or missing required columns.
    """
    required = {"timestamp", "ex", "bid", "ask", "bid_size", "ask_size"}
    missing = required - set(filtered_df.columns)
    if missing:
        raise ValueError(f"build_nbbo_sequence: missing columns {missing}")
    if filtered_df.empty:
        return pd.DataFrame(columns=[
            "timestamp", "nbbo_bid", "nbbo_bid_size",
            "nbbo_ask", "nbbo_ask_size", "spread",
        ])

    # exchange_matrix: ex -> [bid, bid_size, ask, ask_size]
    exchange_matrix: dict = {}

    timestamps = filtered_df["timestamp"].to_numpy()
    exchanges = filtered_df["ex"].to_numpy()
    bids = filtered_df["bid"].to_numpy(dtype=float)
    asks = filtered_df["ask"].to_numpy(dtype=float)
    bid_sizes = filtered_df["bid_size"].to_numpy(dtype=float)
    ask_sizes = filtered_df["ask_size"].to_numpy(dtype=float)

    # Pre-allocate at max size; we may emit fewer rows when crossed markets
    # arise (nbbo_bid >= nbbo_ask).  Crossed/locked quotes are standard TAQ
    # artefacts filtered before the spread-percentile step.
    n = len(filtered_df)
    out_ts: list = []
    out_nbbo_bid: list = []
    out_nbbo_bid_size: list = []
    out_nbbo_ask: list = []
    out_nbbo_ask_size: list = []

    for i in range(n):
        ex = exchanges[i]
        exchange_matrix[ex] = [bids[i], bid_sizes[i], asks[i], ask_sizes[i]]

        # Derive NBBO from all quoted exchanges
        nbbo_bid = -np.inf
        nbbo_ask = np.inf
        for vals in exchange_matrix.values():
            if vals[0] > nbbo_bid:
                nbbo_bid = vals[0]
            if vals[2] < nbbo_ask:
                nbbo_ask = vals[2]

        # Skip crossed/locked markets (bid >= ask) — standard TAQ artefact filter
        if nbbo_bid >= nbbo_ask:
            continue

        # Aggregate sizes at the NBBO prices
        nbbo_bid_size = 0.0
        nbbo_ask_size = 0.0
        for vals in exchange_matrix.values():
            if vals[0] == nbbo_bid:
                nbbo_bid_size += vals[1]
            if vals[2] == nbbo_ask:
                nbbo_ask_size += vals[3]

        out_ts.append(timestamps[i])
        out_nbbo_bid.append(nbbo_bid)
        out_nbbo_bid_size.append(nbbo_bid_size)
        out_nbbo_ask.append(nbbo_ask)
        out_nbbo_ask_size.append(nbbo_ask_size)

    spread = np.array(out_nbbo_ask) - np.array(out_nbbo_bid)

    return pd.DataFrame(
        {
            "timestamp": out_ts,
            "nbbo_bid": out_nbbo_bid,
            "nbbo_bid_size": out_nbbo_bid_size,
            "nbbo_ask": out_nbbo_ask,
            "nbbo_ask_size": out_nbbo_ask_size,
            "spread": spread,
        }
    )


def apply_spread_filter(nbbo_df: pd.DataFrame) -> pd.DataFrame:
    """Flag NBBO records whose spread exceeds the 95th-percentile threshold.

    The 95th percentile is computed across *all* rows in *nbbo_df* (i.e.,
    across all stock-days supplied).  A new column ``spread_filtered_flag``
    is added: 1 if the spread is strictly greater than the 95th percentile,
    0 otherwise.

    Parameters
    ----------
    nbbo_df : pd.DataFrame
        DataFrame with at least a ``spread`` column, as returned by
        :func:`build_nbbo_sequence`.

    Returns
    -------
    pd.DataFrame
        Copy of *nbbo_df* with an added ``spread_filtered_flag`` column.

    Raises
    ------
    ValueError
        If ``spread`` column is absent.
    """
    if "spread" not in nbbo_df.columns:
        raise ValueError("apply_spread_filter: 'spread' column not found")

    out = nbbo_df.copy()
    p95 = out["spread"].quantile(0.95)
    out["spread_filtered_flag"] = (out["spread"] > p95).astype(int)
    return out


def compute_event_contribution(nbbo_df: pd.DataFrame) -> pd.DataFrame:
    """Compute per-event OFI contribution e_n for each NBBO update.

    Implements the event-contribution formula from Cont, Kukanov & Stoikov
    (2014).  For consecutive NBBO records (n−1, n) within the *same calendar
    day*:

    Bid contribution
    ~~~~~~~~~~~~~~~~
    - P_n^B > P_{n−1}^B  →  +q_n^B
    - P_n^B < P_{n−1}^B  →  −q_{n−1}^B
    - P_n^B = P_{n−1}^B  →  q_n^B − q_{n−1}^B

    Ask contribution
    ~~~~~~~~~~~~~~~~
    - P_n^A < P_{n−1}^A  →  −q_n^A
    - P_n^A > P_{n−1}^A  →  +q_{n−1}^A
    - P_n^A = P_{n−1}^A  →  −(q_n^A − q_{n−1}^A)

    e_n = bid_contrib + ask_contrib

    The first event of each calendar day receives e_n = NaN because there is
    no prior NBBO state from which to compute a difference.

    Computation is vectorised using ``numpy.where`` for performance.

    Parameters
    ----------
    nbbo_df : pd.DataFrame
        NBBO sequence with columns ``timestamp``, ``nbbo_bid``,
        ``nbbo_bid_size``, ``nbbo_ask``, ``nbbo_ask_size``.

    Returns
    -------
    pd.DataFrame
        *nbbo_df* with an added ``e_n`` column (float, NaN for day-first rows).

    Raises
    ------
    ValueError
        If required columns are absent or the DataFrame is empty.
    """
    required = {"timestamp", "nbbo_bid", "nbbo_bid_size", "nbbo_ask", "nbbo_ask_size"}
    missing = required - set(nbbo_df.columns)
    if missing:
        raise ValueError(f"compute_event_contribution: missing columns {missing}")
    if nbbo_df.empty:
        raise ValueError("compute_event_contribution: nbbo_df is empty")

    out = nbbo_df.copy().reset_index(drop=True)
    out["timestamp"] = pd.to_datetime(out["timestamp"])
    out["_date"] = out["timestamp"].dt.date

    # --- vectorised numpy arrays ---
    pb = out["nbbo_bid"].to_numpy(dtype=float)
    qb = out["nbbo_bid_size"].to_numpy(dtype=float)
    pa = out["nbbo_ask"].to_numpy(dtype=float)
    qa = out["nbbo_ask_size"].to_numpy(dtype=float)
    dates = out["_date"].to_numpy()

    # Shifted (previous) values – index 0 placeholder will be masked later
    pb_prev = np.empty_like(pb)
    qb_prev = np.empty_like(qb)
    pa_prev = np.empty_like(pa)
    qa_prev = np.empty_like(qa)
    pb_prev[1:] = pb[:-1]
    qb_prev[1:] = qb[:-1]
    pa_prev[1:] = pa[:-1]
    qa_prev[1:] = qa[:-1]

    # Identify first events per day (same-day mask: True when same day as prev)
    same_day = np.empty(len(out), dtype=bool)
    same_day[0] = False
    same_day[1:] = dates[1:] == dates[:-1]

    # --- Bid contribution (Context7: numpy.where for conditional vectorisation)
    bid_contrib = np.where(
        pb > pb_prev,
        qb,
        np.where(
            pb < pb_prev,
            -qb_prev,
            qb - qb_prev,
        ),
    )

    # --- Ask contribution
    ask_contrib = np.where(
        pa < pa_prev,
        -qa,
        np.where(
            pa > pa_prev,
            qa_prev,
            -(qa - qa_prev),
        ),
    )

    e_n = bid_contrib + ask_contrib

    # First event of each day → NaN
    e_n = e_n.astype(float)
    e_n[~same_day] = np.nan

    out["e_n"] = e_n
    out.drop(columns=["_date"], inplace=True)
    return out


def process_stock_day(
    quotes_df: pd.DataFrame,
    ticker: str,
    date: str,
) -> pd.DataFrame:
    """Run the full NBBO pipeline for a single stock-day.

    Executes the following steps:
      1. :func:`apply_quote_filters`
      2. :func:`build_nbbo_sequence`
      3. :func:`apply_spread_filter`
      4. :func:`compute_event_contribution`

    The returned DataFrame has two extra metadata columns: ``ticker`` and
    ``date``.

    Parameters
    ----------
    quotes_df : pd.DataFrame
        Raw TAQ quote records for the stock-day.
    ticker : str
        Stock ticker symbol (e.g., ``'AAPL'``).
    date : str
        Calendar date string in ``YYYY-MM-DD`` format.

    Returns
    -------
    pd.DataFrame
        Fully annotated NBBO sequence with columns:
        ``timestamp``, ``nbbo_bid``, ``nbbo_bid_size``, ``nbbo_ask``,
        ``nbbo_ask_size``, ``spread``, ``spread_filtered_flag``, ``e_n``,
        ``ticker``, ``date``.

    Raises
    ------
    ValueError
        If the quote data cannot be filtered to any valid records.
    """
    if not isinstance(ticker, str) or not ticker:
        raise ValueError("process_stock_day: ticker must be a non-empty string")

    filtered = apply_quote_filters(quotes_df)
    if filtered.empty:
        raise ValueError(
            f"process_stock_day: no valid quotes for {ticker} on {date} "
            "after filtering"
        )

    nbbo = build_nbbo_sequence(filtered)
    nbbo = apply_spread_filter(nbbo)
    nbbo = compute_event_contribution(nbbo)

    nbbo["ticker"] = ticker
    nbbo["date"] = date
    return nbbo


def validate_nbbo_output(all_stocks_stats: dict) -> dict:
    """Validate the NBBO output by computing diagnostic summary statistics.

    Computes two key validation metrics across the supplied per-stock
    statistics dictionary:

    1. **Quote-to-trade ratio** – average number of NBBO updates per trade.
    2. **Average depth** – mean of (nbbo_bid_size + nbbo_ask_size) / 2
       across all records.

    Parameters
    ----------
    all_stocks_stats : dict
        Mapping of ``ticker -> stats_dict`` where each ``stats_dict``
        contains:
          - ``n_quotes``  : int   – number of post-filter NBBO events
          - ``n_trades``  : int   – number of trades for the stock-day
          - ``bid_sizes`` : array-like – NBBO bid sizes per event
          - ``ask_sizes`` : array-like – NBBO ask sizes per event

    Returns
    -------
    dict
        ``{'quote_to_trade_ratio': float, 'avg_depth': float,
           'per_stock': dict}``

    Raises
    ------
    ValueError
        If *all_stocks_stats* is empty.
    """
    if not all_stocks_stats:
        raise ValueError("validate_nbbo_output: all_stocks_stats is empty")

    total_quotes = 0
    total_trades = 0
    depth_values: list = []
    per_stock: dict = {}

    for ticker, stats in all_stocks_stats.items():
        nq = int(stats.get("n_quotes", 0))
        nt = int(stats.get("n_trades", 1))  # avoid /0
        bs = np.asarray(stats.get("bid_sizes", []), dtype=float)
        as_ = np.asarray(stats.get("ask_sizes", []), dtype=float)

        qtr = nq / max(nt, 1)
        avg_d = float(np.mean((bs + as_) / 2.0)) if len(bs) > 0 else np.nan

        per_stock[ticker] = {"quote_to_trade_ratio": qtr, "avg_depth": avg_d}
        total_quotes += nq
        total_trades += nt
        if len(bs) > 0:
            depth_values.extend(((bs + as_) / 2.0).tolist())

    global_qtr = total_quotes / max(total_trades, 1)
    global_depth = float(np.mean(depth_values)) if depth_values else np.nan

    return {
        "quote_to_trade_ratio": global_qtr,
        "avg_depth": global_depth,
        "per_stock": per_stock,
    }
