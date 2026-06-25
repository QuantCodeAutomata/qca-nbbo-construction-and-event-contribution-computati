"""
test_grid.py - Unit tests for src/grid.py

Tests cover:
  - apply_trade_filters: time window, price/size/corr filtering, condition codes
  - classify_trades_quote_test: buy/sell/unclassified assignment
  - classify_trades_tick_test: tick-test direction rules including forward fill
  - build_10sec_grid: bin count, OFI aggregation, delta_P, TI, VOL
  - compute_average_depth: AD formula with N-1 denominator
"""

import numpy as np
import pandas as pd
import pytest

from src.grid import (
    apply_trade_filters,
    classify_trades_quote_test,
    classify_trades_tick_test,
    build_10sec_grid,
    compute_average_depth,
    N_BINS,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ts(time_str: str, date_str: str = "2010-04-01") -> pd.Timestamp:
    return pd.Timestamp(f"{date_str} {time_str}")


def _make_trade_df(**overrides) -> pd.DataFrame:
    n = overrides.pop("n", 5)
    base = {
        "timestamp": pd.date_range("2010-04-01 10:00:00", periods=n, freq="1min"),
        "price": [50.0] * n,
        "size": [100] * n,
        "corr": [0] * n,
        "cond": [""] * n,
    }
    base.update(overrides)
    return pd.DataFrame(base)


def _make_nbbo_df(n: int = 10, date_str: str = "2010-04-01") -> pd.DataFrame:
    """Minimal NBBO DataFrame for trade classification tests.

    Uses 1-second spacing so that a trade at 09:30:05 sits within the
    1-second backward-merge tolerance of the NBBO record at 09:30:05.
    """
    timestamps = pd.date_range(f"{date_str} 09:30:00", periods=n, freq="1s")
    return pd.DataFrame({
        "timestamp": timestamps,
        "nbbo_bid": [49.90] * n,
        "nbbo_bid_size": [100] * n,
        "nbbo_ask": [50.10] * n,
        "nbbo_ask_size": [100] * n,
        "spread": [0.20] * n,
        "spread_filtered_flag": [0] * n,
        "e_n": [np.nan] + [10.0] * (n - 1),
    })


# ---------------------------------------------------------------------------
# apply_trade_filters
# ---------------------------------------------------------------------------

class TestApplyTradeFilters:

    def test_rejects_missing_columns(self):
        df = pd.DataFrame({"timestamp": [], "price": []})
        with pytest.raises(ValueError, match="missing columns"):
            apply_trade_filters(df)

    def test_removes_pre_market_trades(self):
        df = _make_trade_df(n=3, timestamp=[
            _ts("09:00:00"), _ts("09:29:59"), _ts("09:30:00")
        ])
        result = apply_trade_filters(df)
        assert len(result) == 1

    def test_removes_post_market_trades(self):
        df = _make_trade_df(n=2, timestamp=[
            _ts("16:00:00"), _ts("16:00:01")
        ])
        result = apply_trade_filters(df)
        assert len(result) == 1  # exactly 16:00:00 passes

    def test_removes_zero_price(self):
        df = _make_trade_df(n=2)
        df.loc[0, "price"] = 0.0
        result = apply_trade_filters(df)
        assert len(result) == 1

    def test_removes_zero_size(self):
        df = _make_trade_df(n=2)
        df.loc[1, "size"] = 0
        result = apply_trade_filters(df)
        assert len(result) == 1

    def test_removes_high_corr_indicator(self):
        df = _make_trade_df(n=3, corr=[0, 2, 3])
        result = apply_trade_filters(df)
        assert len(result) == 2  # corr=3 excluded

    def test_removes_excluded_conditions(self):
        excluded = ["O", "Z", "B", "T", "L", "G", "W", "J", "K"]
        df = _make_trade_df(n=len(excluded) + 1)
        df["cond"] = excluded + [""]
        result = apply_trade_filters(df)
        assert len(result) == 1

    def test_sorted_by_timestamp(self):
        df = _make_trade_df(n=3, timestamp=[
            _ts("11:00:00"), _ts("09:45:00"), _ts("14:30:00")
        ])
        result = apply_trade_filters(df)
        ts = result["timestamp"].tolist()
        assert ts == sorted(ts)

    def test_empty_input(self):
        df = pd.DataFrame({
            "timestamp": pd.Series([], dtype="datetime64[ns]"),
            "price": pd.Series([], dtype=float),
            "size": pd.Series([], dtype=float),
            "corr": pd.Series([], dtype=int),
            "cond": pd.Series([], dtype=str),
        })
        result = apply_trade_filters(df)
        assert len(result) == 0


# ---------------------------------------------------------------------------
# classify_trades_quote_test
# ---------------------------------------------------------------------------

class TestClassifyTradesQuoteTest:

    def _setup(self):
        nbbo = _make_nbbo_df(n=20)
        # NBBO: bid=49.90, ask=50.10
        trades = pd.DataFrame({
            "timestamp": [
                _ts("09:30:05"),  # buy: price >= ask
                _ts("09:30:05"),  # sell: price <= bid
                _ts("09:30:05"),  # inside: unclassified
            ],
            "price": [50.10, 49.90, 50.00],
            "size":  [200,   300,   100],
            "corr": [0, 0, 0],
            "cond": ["", "", ""],
        })
        return nbbo, trades

    def test_buy_classification(self):
        nbbo, trades = self._setup()
        result = classify_trades_quote_test(trades, nbbo)
        buy_row = result[result["price"] == 50.10].iloc[0]
        assert buy_row["b_n"] == 200
        assert buy_row["s_n"] == 0

    def test_sell_classification(self):
        nbbo, trades = self._setup()
        result = classify_trades_quote_test(trades, nbbo)
        sell_row = result[result["price"] == 49.90].iloc[0]
        assert sell_row["b_n"] == 0
        assert sell_row["s_n"] == 300

    def test_inside_spread_unclassified(self):
        nbbo, trades = self._setup()
        result = classify_trades_quote_test(trades, nbbo)
        inside_row = result[result["price"] == 50.00].iloc[0]
        assert inside_row["b_n"] == 0
        assert inside_row["s_n"] == 0

    def test_no_matching_nbbo_leaves_unclassified(self):
        nbbo = _make_nbbo_df(n=5)
        # trade 5 minutes before any NBBO
        trades = pd.DataFrame({
            "timestamp": [pd.Timestamp("2010-04-01 09:25:00")],
            "price": [50.10],
            "size": [100],
            "corr": [0],
            "cond": [""],
        })
        result = classify_trades_quote_test(trades, nbbo)
        assert result["b_n"].iloc[0] == 0
        assert result["s_n"].iloc[0] == 0

    def test_output_has_b_n_and_s_n_columns(self):
        nbbo, trades = self._setup()
        result = classify_trades_quote_test(trades, nbbo)
        assert "b_n" in result.columns
        assert "s_n" in result.columns


# ---------------------------------------------------------------------------
# classify_trades_tick_test
# ---------------------------------------------------------------------------

class TestClassifyTradesTickTest:

    def test_uptick_is_buy(self):
        trades = pd.DataFrame({
            "timestamp": pd.date_range("2010-04-01 10:00:00", periods=2, freq="1s"),
            "price": [50.0, 50.1],
            "size": [100, 200],
            "corr": [0, 0],
            "cond": ["", ""],
        })
        result = classify_trades_tick_test(trades)
        assert result["b_n"].iloc[1] == 200
        assert result["s_n"].iloc[1] == 0

    def test_downtick_is_sell(self):
        trades = pd.DataFrame({
            "timestamp": pd.date_range("2010-04-01 10:00:00", periods=2, freq="1s"),
            "price": [50.1, 50.0],
            "size": [100, 200],
            "corr": [0, 0],
            "cond": ["", ""],
        })
        result = classify_trades_tick_test(trades)
        assert result["b_n"].iloc[1] == 0
        assert result["s_n"].iloc[1] == 200

    def test_zero_tick_carries_forward(self):
        """Zero tick should carry forward the previous classification."""
        trades = pd.DataFrame({
            "timestamp": pd.date_range("2010-04-01 10:00:00", periods=3, freq="1s"),
            "price": [50.0, 50.1, 50.1],  # uptick then zero-tick
            "size": [100, 200, 300],
            "corr": [0, 0, 0],
            "cond": ["", "", ""],
        })
        result = classify_trades_tick_test(trades)
        # Third trade: zero-tick, should carry forward "buy" from second trade
        assert result["b_n"].iloc[2] == 300
        assert result["s_n"].iloc[2] == 0

    def test_output_has_required_columns(self):
        trades = _make_trade_df(n=3)
        result = classify_trades_tick_test(trades)
        assert "b_n" in result.columns
        assert "s_n" in result.columns

    def test_first_trade_is_unclassified(self):
        """First trade has no predecessor, should be unclassified."""
        trades = _make_trade_df(n=3)
        result = classify_trades_tick_test(trades)
        assert result["b_n"].iloc[0] == 0
        assert result["s_n"].iloc[0] == 0


# ---------------------------------------------------------------------------
# build_10sec_grid
# ---------------------------------------------------------------------------

class TestBuild10SecGrid:

    def _make_full_day_nbbo(self, date_str: str = "2010-04-01") -> pd.DataFrame:
        """Generate a simple NBBO covering the full session."""
        # 50 NBBO events at 5-minute intervals starting at 09:30
        n = 78  # roughly one per 5 minutes over 6.5 hours
        timestamps = pd.date_range(f"{date_str} 09:30:00", periods=n, freq="5min")
        # Keep within session
        timestamps = timestamps[timestamps.time <= pd.Timestamp("16:00:00").time()]
        n = len(timestamps)
        price = 50.0 + np.cumsum(np.random.default_rng(0).normal(0, 0.01, n))
        spread = 0.05
        return pd.DataFrame({
            "timestamp": timestamps,
            "nbbo_bid": price,
            "nbbo_bid_size": 100,
            "nbbo_ask": price + spread,
            "nbbo_ask_size": 100,
            "spread": spread,
            "spread_filtered_flag": 0,
            "e_n": np.concatenate([[np.nan], np.random.randint(-50, 50, n - 1).astype(float)]),
        })

    def test_returns_2340_bins(self):
        nbbo = self._make_full_day_nbbo()
        trades = pd.DataFrame({
            "timestamp": pd.Series([], dtype="datetime64[ns]"),
            "price": pd.Series([], dtype=float),
            "size": pd.Series([], dtype=float),
            "b_n": pd.Series([], dtype=float),
            "s_n": pd.Series([], dtype=float),
        })
        result = build_10sec_grid(nbbo, trades, "2010-04-01")
        assert len(result) == N_BINS, f"Expected {N_BINS} bins, got {len(result)}"

    def test_ofi_k_zero_when_no_events(self):
        nbbo = pd.DataFrame({
            "timestamp": pd.Series([], dtype="datetime64[ns]"),
            "nbbo_bid": pd.Series([], dtype=float),
            "nbbo_bid_size": pd.Series([], dtype=float),
            "nbbo_ask": pd.Series([], dtype=float),
            "nbbo_ask_size": pd.Series([], dtype=float),
            "spread": pd.Series([], dtype=float),
            "spread_filtered_flag": pd.Series([], dtype=int),
            "e_n": pd.Series([], dtype=float),
        })
        trades = pd.DataFrame({
            "timestamp": pd.Series([], dtype="datetime64[ns]"),
            "price": pd.Series([], dtype=float),
            "size": pd.Series([], dtype=float),
            "b_n": pd.Series([], dtype=float),
            "s_n": pd.Series([], dtype=float),
        })
        result = build_10sec_grid(nbbo, trades, "2010-04-01")
        assert (result["OFI_k"].fillna(0) == 0).all()

    def test_vol_k_zero_when_no_trades(self):
        nbbo = self._make_full_day_nbbo()
        trades = pd.DataFrame({
            "timestamp": pd.Series([], dtype="datetime64[ns]"),
            "price": pd.Series([], dtype=float),
            "size": pd.Series([], dtype=float),
            "b_n": pd.Series([], dtype=float),
            "s_n": pd.Series([], dtype=float),
        })
        result = build_10sec_grid(nbbo, trades, "2010-04-01")
        assert (result["VOL_k"].fillna(0) == 0).all()

    def test_half_hour_index_range(self):
        nbbo = self._make_full_day_nbbo()
        trades = pd.DataFrame({
            "timestamp": pd.Series([], dtype="datetime64[ns]"),
            "price": pd.Series([], dtype=float),
            "size": pd.Series([], dtype=float),
            "b_n": pd.Series([], dtype=float),
            "s_n": pd.Series([], dtype=float),
        })
        result = build_10sec_grid(nbbo, trades, "2010-04-01")
        assert result["half_hour"].min() == 0
        assert result["half_hour"].max() == 12

    def test_ofi_k_aggregation(self):
        """OFI_k should sum all e_n falling in the same 10-second bin."""
        # Place two NBBO events in the same 10-second bin
        nbbo = pd.DataFrame({
            "timestamp": [
                _ts("09:30:01"), _ts("09:30:05"),  # both in bin 0 [09:30:00, 09:30:10)
                _ts("09:30:11"),                    # bin 1
            ],
            "nbbo_bid": [50.0, 50.0, 50.1],
            "nbbo_bid_size": [100, 100, 100],
            "nbbo_ask": [50.05, 50.05, 50.15],
            "nbbo_ask_size": [100, 100, 100],
            "spread": [0.05, 0.05, 0.05],
            "spread_filtered_flag": [0, 0, 0],
            "e_n": [np.nan, 10.0, 5.0],
        })
        trades = pd.DataFrame({
            "timestamp": pd.Series([], dtype="datetime64[ns]"),
            "price": pd.Series([], dtype=float),
            "size": pd.Series([], dtype=float),
            "b_n": pd.Series([], dtype=float),
            "s_n": pd.Series([], dtype=float),
        })
        result = build_10sec_grid(nbbo, trades, "2010-04-01")
        # Bin 0 should have OFI = 10 (skip NaN); bin 1 = 5
        assert result.loc[result["bin_k"] == 0, "OFI_k"].iloc[0] == 10.0
        assert result.loc[result["bin_k"] == 1, "OFI_k"].iloc[0] == 5.0


# ---------------------------------------------------------------------------
# compute_average_depth
# ---------------------------------------------------------------------------

class TestComputeAverageDepth:

    def test_formula_uses_n_minus_1_denominator(self):
        """AD_i = sum(bid_size + ask_size) / (2 * (N_i - 1))."""
        n_events = 5
        bid_size = 100
        ask_size = 200
        nbbo = pd.DataFrame({
            "timestamp": pd.date_range("2010-04-01 09:30:00", periods=n_events, freq="30s"),
            "nbbo_bid_size": [bid_size] * n_events,
            "nbbo_ask_size": [ask_size] * n_events,
            "spread_filtered_flag": [0] * n_events,
        })
        result = compute_average_depth(nbbo)
        # First slot (09:30-10:00): all n_events events
        # AD = sum / (2 * (N-1)) = 5*300 / (2*4) = 1500/8 = 187.5
        expected = n_events * (bid_size + ask_size) / (2 * (n_events - 1))
        slot0 = result[result["half_hour"] == 0]
        if len(slot0) > 0:
            np.testing.assert_allclose(slot0["AD_i"].iloc[0], expected, rtol=0.01)

    def test_returns_13_slots(self):
        """Should return at most 13 half-hour slots."""
        n = 100
        timestamps = pd.date_range("2010-04-01 09:30:00", periods=n, freq="2min")
        timestamps = timestamps[timestamps.time <= pd.Timestamp("16:00:00").time()]
        nbbo = pd.DataFrame({
            "timestamp": timestamps,
            "nbbo_bid_size": 100,
            "nbbo_ask_size": 200,
            "spread_filtered_flag": 0,
        })
        result = compute_average_depth(nbbo)
        assert len(result) <= 13
        assert result["half_hour"].between(0, 12).all()

    def test_nan_when_fewer_than_2_events_in_slot(self):
        """Slots with N_i <= 1 should yield NaN AD_i."""
        nbbo = pd.DataFrame({
            "timestamp": [_ts("09:30:00")],
            "nbbo_bid_size": [100],
            "nbbo_ask_size": [200],
            "spread_filtered_flag": [0],
        })
        result = compute_average_depth(nbbo)
        assert result["AD_i"].isna().all()

    def test_empty_nbbo_returns_empty(self):
        nbbo = pd.DataFrame({
            "timestamp": pd.Series([], dtype="datetime64[ns]"),
            "nbbo_bid_size": pd.Series([], dtype=float),
            "nbbo_ask_size": pd.Series([], dtype=float),
            "spread_filtered_flag": pd.Series([], dtype=int),
        })
        result = compute_average_depth(nbbo)
        assert len(result) == 0
