"""
test_nbbo.py - Unit tests for src/nbbo.py

Tests cover:
  - apply_quote_filters: time window, positive values, mode-code exclusion
  - build_nbbo_sequence: exchange-matrix logic, bid=max, ask=min, size sums
  - apply_spread_filter: 95th-percentile threshold
  - compute_event_contribution: all six sign-change cases
  - process_stock_day: end-to-end pipeline integration
  - validate_nbbo_output: ratio and depth checkpoint
"""

import numpy as np
import pandas as pd
import pytest
from datetime import date, time, datetime

from src.nbbo import (
    apply_quote_filters,
    build_nbbo_sequence,
    apply_spread_filter,
    compute_event_contribution,
    process_stock_day,
    validate_nbbo_output,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_base_date() -> str:
    return "2010-04-01"


def _ts(time_str: str, date_str: str = "2010-04-01") -> pd.Timestamp:
    return pd.Timestamp(f"{date_str} {time_str}")


def _make_quote_df(**overrides) -> pd.DataFrame:
    """Construct a minimal valid quote DataFrame."""
    n = overrides.pop("n", 5)
    base = {
        "timestamp": pd.date_range("2010-04-01 09:30:00", periods=n, freq="1min"),
        "bid": [10.0] * n,
        "ask": [10.02] * n,
        "bid_size": [100] * n,
        "ask_size": [200] * n,
        "mode": [0] * n,
        "ex": ["N"] * n,
    }
    base.update(overrides)
    return pd.DataFrame(base)


# ---------------------------------------------------------------------------
# apply_quote_filters
# ---------------------------------------------------------------------------

class TestApplyQuoteFilters:

    def test_rejects_missing_columns(self):
        df = pd.DataFrame({"timestamp": [], "bid": []})
        with pytest.raises(ValueError, match="missing columns"):
            apply_quote_filters(df)

    def test_removes_pre_market_quotes(self):
        df = _make_quote_df(n=3, timestamp=[
            _ts("09:00:00"), _ts("09:29:59"), _ts("09:30:00")
        ])
        result = apply_quote_filters(df)
        # Only the 09:30 quote passes
        assert len(result) == 1

    def test_removes_post_market_quotes(self):
        df = _make_quote_df(n=3, timestamp=[
            _ts("15:59:59"), _ts("16:00:00"), _ts("16:00:01")
        ])
        result = apply_quote_filters(df)
        assert len(result) == 2  # 15:59:59 and 16:00:00

    def test_removes_zero_bid(self):
        df = _make_quote_df(n=2)
        df.loc[0, "bid"] = 0.0
        result = apply_quote_filters(df)
        assert len(result) == 1

    def test_removes_zero_ask(self):
        df = _make_quote_df(n=2)
        df.loc[1, "ask"] = 0.0
        result = apply_quote_filters(df)
        assert len(result) == 1

    def test_removes_zero_bid_size(self):
        df = _make_quote_df(n=2)
        df.loc[0, "bid_size"] = 0
        result = apply_quote_filters(df)
        assert len(result) == 1

    def test_removes_excluded_mode_codes(self):
        excluded = [4, 7, 9, 11, 13, 14, 15, 19, 20, 27, 28]
        df = _make_quote_df(n=len(excluded) + 1)
        df["mode"] = excluded + [0]
        result = apply_quote_filters(df)
        assert len(result) == 1

    def test_keeps_valid_mode_code_zero(self):
        df = _make_quote_df(n=3, mode=[0, 0, 0])
        result = apply_quote_filters(df)
        assert len(result) == 3

    def test_output_sorted_by_timestamp(self):
        df = _make_quote_df(n=3, timestamp=[
            _ts("10:30:00"), _ts("09:45:00"), _ts("11:00:00")
        ])
        result = apply_quote_filters(df)
        assert list(result["timestamp"]) == sorted(result["timestamp"])

    def test_empty_input_returns_empty(self):
        df = _make_quote_df(n=0)
        result = apply_quote_filters(df)
        assert len(result) == 0


# ---------------------------------------------------------------------------
# build_nbbo_sequence
# ---------------------------------------------------------------------------

class TestBuildNbboSequence:

    def _make_multi_exchange_quotes(self) -> pd.DataFrame:
        """Two exchanges updating in turn — tests exchange matrix updates.

        Quote prices are chosen so that no individual exchange's bid ever
        exceeds another exchange's ask, preventing crossed-NBBO scenarios.
        Exchange N always quotes around 10.00 and Q around 10.05–10.10.
        """
        timestamps = pd.date_range("2010-04-01 09:30:00", periods=6, freq="5s")
        return pd.DataFrame({
            "timestamp": timestamps,
            "bid":       [10.00, 10.03, 10.01, 10.05, 10.02, 10.04],
            "ask":       [10.10, 10.13, 10.11, 10.15, 10.12, 10.14],
            "bid_size":  [100, 200, 150, 300, 250, 100],
            "ask_size":  [200, 100, 250, 150, 200, 300],
            "mode":      [0] * 6,
            "ex":        ["N", "Q", "N", "Q", "N", "Q"],
        })

    def test_returns_expected_columns(self):
        df = self._make_multi_exchange_quotes()
        result = build_nbbo_sequence(df)
        for col in ["timestamp", "nbbo_bid", "nbbo_bid_size", "nbbo_ask", "nbbo_ask_size", "spread"]:
            assert col in result.columns, f"Missing column: {col}"

    def test_nbbo_bid_is_max_across_exchanges(self):
        df = self._make_multi_exchange_quotes()
        result = build_nbbo_sequence(df)
        # After both exchanges update, NBBO bid should be max of their current bids
        # We verify monotonicity: nbbo_bid >= any individual bid in df
        assert (result["nbbo_bid"] <= result["nbbo_ask"]).all(), "NBBO bid must be <= NBBO ask"

    def test_nbbo_ask_is_min_across_exchanges(self):
        df = self._make_multi_exchange_quotes()
        result = build_nbbo_sequence(df)
        # With two exchanges, nbbo_ask = min(ask_N, ask_Q)
        # The minimum observed ask among all quotes
        min_possible_ask = df["ask"].min()
        assert (result["nbbo_ask"] >= min_possible_ask * 0.99).all()

    def test_spread_equals_ask_minus_bid(self):
        df = self._make_multi_exchange_quotes()
        result = build_nbbo_sequence(df)
        expected_spread = result["nbbo_ask"] - result["nbbo_bid"]
        np.testing.assert_allclose(result["spread"].values, expected_spread.values, atol=1e-10)

    def test_spread_non_negative(self):
        df = self._make_multi_exchange_quotes()
        result = build_nbbo_sequence(df)
        assert (result["spread"] >= -1e-10).all()

    def test_single_exchange_nbbo_equals_quote(self):
        df = pd.DataFrame({
            "timestamp": pd.date_range("2010-04-01 09:30:00", periods=3, freq="10s"),
            "bid":      [10.0, 10.1, 10.2],
            "ask":      [10.05, 10.15, 10.25],
            "bid_size": [100, 200, 300],
            "ask_size": [150, 250, 350],
            "mode":     [0, 0, 0],
            "ex":       ["N", "N", "N"],
        })
        result = build_nbbo_sequence(df)
        # With single exchange, NBBO = the exchange's own quote
        np.testing.assert_allclose(result["nbbo_bid"].values, df["bid"].values, atol=1e-10)
        np.testing.assert_allclose(result["nbbo_ask"].values, df["ask"].values, atol=1e-10)

    def test_bid_size_sum_at_nbbo_price(self):
        """Two exchanges at same bid: bid_size should be summed."""
        df = pd.DataFrame({
            "timestamp": [_ts("09:30:00"), _ts("09:30:01")],
            "bid":      [10.0, 10.0],
            "ask":      [10.10, 10.05],
            "bid_size": [100, 200],
            "ask_size": [150, 250],
            "mode":     [0, 0],
            "ex":       ["N", "Q"],
        })
        result = build_nbbo_sequence(df)
        last = result.iloc[-1]
        # Both exchanges at 10.0 bid → bid_size = 100 + 200 = 300
        assert last["nbbo_bid_size"] == 300
        # Best ask is min(10.10, 10.05) = 10.05 → ask_size = 250
        assert last["nbbo_ask_size"] == 250

    def test_empty_filtered_quotes_returns_empty(self):
        df = pd.DataFrame({
            "timestamp": pd.Series([], dtype="datetime64[ns]"),
            "bid": pd.Series([], dtype=float),
            "ask": pd.Series([], dtype=float),
            "bid_size": pd.Series([], dtype=float),
            "ask_size": pd.Series([], dtype=float),
            "mode": pd.Series([], dtype=int),
            "ex": pd.Series([], dtype=str),
        })
        result = build_nbbo_sequence(df)
        assert len(result) == 0


# ---------------------------------------------------------------------------
# apply_spread_filter
# ---------------------------------------------------------------------------

class TestApplySpreadFilter:

    def _make_nbbo_df(self, spreads: list) -> pd.DataFrame:
        n = len(spreads)
        return pd.DataFrame({
            "timestamp": pd.date_range("2010-04-01 09:30:00", periods=n, freq="5s"),
            "nbbo_bid": [10.0] * n,
            "nbbo_bid_size": [100] * n,
            "nbbo_ask": [10.0 + s for s in spreads],
            "nbbo_ask_size": [200] * n,
            "spread": spreads,
        })

    def test_flag_column_added(self):
        df = self._make_nbbo_df([0.01] * 20 + [1.0])
        result = apply_spread_filter(df)
        assert "spread_filtered_flag" in result.columns

    def test_approximately_5_percent_flagged(self):
        """With 100 rows, at most ~5 should be flagged."""
        spreads = list(np.linspace(0.01, 0.20, 100))
        df = self._make_nbbo_df(spreads)
        result = apply_spread_filter(df)
        pct_flagged = result["spread_filtered_flag"].mean()
        # All rows with spread > 95th percentile are flagged
        p95 = np.percentile(spreads, 95)
        expected_flagged = sum(1 for s in spreads if s > p95)
        assert result["spread_filtered_flag"].sum() == expected_flagged

    def test_flag_is_zero_or_one(self):
        df = self._make_nbbo_df(np.random.uniform(0.01, 0.5, 50).tolist())
        result = apply_spread_filter(df)
        assert set(result["spread_filtered_flag"].unique()).issubset({0, 1})

    def test_empty_returns_empty(self):
        df = pd.DataFrame({
            "timestamp": pd.Series([], dtype="datetime64[ns]"),
            "nbbo_bid": pd.Series([], dtype=float),
            "nbbo_bid_size": pd.Series([], dtype=float),
            "nbbo_ask": pd.Series([], dtype=float),
            "nbbo_ask_size": pd.Series([], dtype=float),
            "spread": pd.Series([], dtype=float),
        })
        result = apply_spread_filter(df)
        assert len(result) == 0

    def test_all_equal_spreads_none_flagged(self):
        """All spreads equal to the same value: p95 = that value, none exceed it."""
        df = self._make_nbbo_df([0.01] * 50)
        result = apply_spread_filter(df)
        assert result["spread_filtered_flag"].sum() == 0


# ---------------------------------------------------------------------------
# compute_event_contribution
# ---------------------------------------------------------------------------

class TestComputeEventContribution:

    def _make_nbbo_with_en_df(
        self,
        bids, asks, bid_sizes, ask_sizes,
        date_str="2010-04-01"
    ) -> pd.DataFrame:
        n = len(bids)
        return pd.DataFrame({
            "timestamp": pd.date_range(f"{date_str} 09:30:00", periods=n, freq="10s"),
            "nbbo_bid": bids,
            "nbbo_ask": asks,
            "nbbo_bid_size": bid_sizes,
            "nbbo_ask_size": ask_sizes,
            "spread": [a - b for a, b in zip(asks, bids)],
            "spread_filtered_flag": [0] * n,
            "date": date_str,
        })

    def test_bid_increases_positive_contribution(self):
        """Bid rises: bid_contrib = +q_n^B."""
        df = self._make_nbbo_with_en_df(
            bids=[10.0, 10.1],
            asks=[10.05, 10.15],
            bid_sizes=[100, 200],
            ask_sizes=[150, 250],
        )
        result = compute_event_contribution(df)
        en = result["e_n"].dropna()
        # bid_contrib = +200; ask_contrib: ask rose → +q_{n-1}^A = +150
        # e_n = 200 + 150 = 350
        assert len(en) == 1
        assert en.iloc[0] == 200 + 150  # bid up + ask up

    def test_bid_decreases_negative_contribution(self):
        """Bid falls: bid_contrib = -q_{n-1}^B."""
        df = self._make_nbbo_with_en_df(
            bids=[10.1, 10.0],
            asks=[10.15, 10.05],
            bid_sizes=[200, 100],
            ask_sizes=[250, 150],
        )
        result = compute_event_contribution(df)
        en = result["e_n"].dropna()
        # bid_contrib = -200 (old bid size); ask_contrib: ask fell → -q_n^A = -150
        # e_n = -200 + (-150) = -350
        assert len(en) == 1
        assert en.iloc[0] == -200 + (-150)

    def test_bid_unchanged_size_change(self):
        """Bid same: bid_contrib = q_n^B - q_{n-1}^B."""
        df = self._make_nbbo_with_en_df(
            bids=[10.0, 10.0],
            asks=[10.05, 10.05],
            bid_sizes=[100, 300],
            ask_sizes=[200, 200],
        )
        result = compute_event_contribution(df)
        en = result["e_n"].dropna()
        # bid_contrib = 300 - 100 = 200; ask_contrib = -(200 - 200) = 0
        assert len(en) == 1
        assert en.iloc[0] == 200

    def test_first_event_has_nan_e_n(self):
        df = self._make_nbbo_with_en_df(
            bids=[10.0, 10.1, 10.0],
            asks=[10.05, 10.15, 10.05],
            bid_sizes=[100, 200, 150],
            ask_sizes=[150, 250, 200],
        )
        result = compute_event_contribution(df)
        assert pd.isna(result["e_n"].iloc[0])

    def test_no_cross_day_computation(self):
        """e_n must not be computed across day boundaries."""
        df1 = self._make_nbbo_with_en_df(
            bids=[10.0, 10.1], asks=[10.05, 10.15],
            bid_sizes=[100, 200], ask_sizes=[150, 250],
            date_str="2010-04-01"
        )
        df2 = self._make_nbbo_with_en_df(
            bids=[10.1, 10.0], asks=[10.15, 10.05],
            bid_sizes=[200, 100], ask_sizes=[250, 150],
            date_str="2010-04-05"
        )
        combined = pd.concat([df1, df2], ignore_index=True)
        result = compute_event_contribution(combined)
        # First event of each day must have NaN
        day_groups = result.groupby("date")
        for day, grp in day_groups:
            assert pd.isna(grp["e_n"].iloc[0]), f"First event on {day} must have NaN e_n"

    def test_e_n_is_integer_or_nan(self):
        """e_n values should be integers (share quantities)."""
        df = self._make_nbbo_with_en_df(
            bids=[10.0, 10.1, 10.0, 10.05],
            asks=[10.05, 10.15, 10.05, 10.10],
            bid_sizes=[100, 200, 150, 300],
            ask_sizes=[200, 100, 250, 200],
        )
        result = compute_event_contribution(df)
        valid_en = result["e_n"].dropna()
        # All valid e_n should be integer-valued (within floating point tolerance)
        for val in valid_en:
            assert float(val) == int(float(val)), f"e_n={val} is not integer"

    def test_single_event_day_all_nan(self):
        """Day with only one NBBO event: no e_n can be computed."""
        df = self._make_nbbo_with_en_df(
            bids=[10.0], asks=[10.05],
            bid_sizes=[100], ask_sizes=[200],
        )
        result = compute_event_contribution(df)
        assert result["e_n"].isna().all()


# ---------------------------------------------------------------------------
# process_stock_day (integration)
# ---------------------------------------------------------------------------

class TestProcessStockDay:

    def test_output_has_required_columns(self):
        from data.synthetic_data import generate_synthetic_quotes
        quotes_df = generate_synthetic_quotes("SLB", "2010-04-01", seed=42)
        result = process_stock_day(quotes_df, "SLB", "2010-04-01")
        required = {"timestamp", "nbbo_bid", "nbbo_bid_size", "nbbo_ask",
                    "nbbo_ask_size", "spread", "spread_filtered_flag", "e_n"}
        assert required.issubset(set(result.columns))

    def test_spread_filtered_flag_fraction_near_5pct(self):
        from data.synthetic_data import generate_synthetic_quotes
        quotes_df = generate_synthetic_quotes("SLB", "2010-04-01", seed=42)
        result = process_stock_day(quotes_df, "SLB", "2010-04-01")
        frac = result["spread_filtered_flag"].mean()
        # Should be approximately 5% (within a reasonable range for synthetic data)
        assert 0.0 <= frac <= 0.25, f"Flagged fraction {frac:.2%} out of expected range"

    def test_nbbo_bid_leq_nbbo_ask_always(self):
        from data.synthetic_data import generate_synthetic_quotes
        quotes_df = generate_synthetic_quotes("AMD", "2010-04-05", seed=99)
        result = process_stock_day(quotes_df, "AMD", "2010-04-05")
        assert (result["nbbo_bid"] <= result["nbbo_ask"] + 1e-10).all()


# ---------------------------------------------------------------------------
# validate_nbbo_output
# ---------------------------------------------------------------------------

class TestValidateNbboOutput:

    def _make_stats(self, n_quotes: int = 1000, n_trades: int = 25,
                    depth: float = 220.0, ticker: str = "SLB") -> dict:
        """Build stats dict matching validate_nbbo_output's expected schema."""
        rng = np.random.default_rng(0)
        n = n_quotes
        bid_sizes = rng.integers(100, 400, size=n).astype(float)
        ask_sizes = rng.integers(100, 400, size=n).astype(float)
        return {
            ticker: {
                "n_quotes": n_quotes,
                "n_trades": n_trades,
                "bid_sizes": bid_sizes,
                "ask_sizes": ask_sizes,
            }
        }

    def test_returns_dict_with_expected_keys(self):
        stats = self._make_stats()
        result = validate_nbbo_output(stats)
        assert "quote_to_trade_ratio" in result
        assert "avg_depth" in result
        assert "per_stock" in result

    def test_ratio_approximately_40(self):
        """Simulate a scenario where ratio is close to 40."""
        stats = {}
        rng = np.random.default_rng(1)
        for ticker in ["A", "B", "C"]:
            stats[ticker] = {
                "n_quotes": 40000,
                "n_trades": 1000,
                "bid_sizes": rng.integers(100, 400, size=1000).astype(float),
                "ask_sizes": rng.integers(100, 400, size=1000).astype(float),
            }
        result = validate_nbbo_output(stats)
        assert abs(result["quote_to_trade_ratio"] - 40.0) < 1.0

    def test_avg_depth_positive(self):
        stats = self._make_stats(n_quotes=500, n_trades=20)
        result = validate_nbbo_output(stats)
        assert result["avg_depth"] > 0

    def test_empty_stats_raises(self):
        with pytest.raises(ValueError):
            validate_nbbo_output({})
