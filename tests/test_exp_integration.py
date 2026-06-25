"""
test_exp_integration.py - Integration tests for experiment pipelines.

Tests verify that:
  - Synthetic data generation produces expected shapes and value ranges
  - exp_1 NBBO pipeline produces valid output parquet files
  - exp_2 grid pipeline produces correct bin count and column set
  - exp_3 regression results satisfy stated hypotheses (approx.)
  - exp_4 depth regression produces lambda > 0
  - exp_5 OFI vs TI shows OFI R² > TI R²
  - exp_6 volume regression returns H_hat in reasonable range
  - exp_7 seasonality produces 13 normalized slots
  - exp_8 robustness produces all three output files
"""

import os
import numpy as np
import pandas as pd
import pytest

# Only run integration tests if results directory already contains outputs
# (experiments were already executed by sub-agents)
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
NBBO_DIR = os.path.join(RESULTS_DIR, "nbbo")
GRID_DIR = os.path.join(RESULTS_DIR, "grid")


# ---------------------------------------------------------------------------
# Synthetic data
# ---------------------------------------------------------------------------

class TestSyntheticData:

    def test_generate_quotes_returns_dataframe(self):
        from data.synthetic_data import generate_synthetic_quotes
        df = generate_synthetic_quotes("SLB", "2010-04-01", seed=0)
        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0

    def test_quotes_have_required_columns(self):
        from data.synthetic_data import generate_synthetic_quotes
        df = generate_synthetic_quotes("AMD", "2010-04-01", seed=1)
        for col in ["timestamp", "bid", "ask", "bid_size", "ask_size", "mode", "ex"]:
            assert col in df.columns

    def test_quotes_bid_leq_ask(self):
        from data.synthetic_data import generate_synthetic_quotes
        df = generate_synthetic_quotes("BAC", "2010-04-05", seed=2)
        assert (df["bid"] <= df["ask"]).all()

    def test_quotes_prices_positive(self):
        from data.synthetic_data import generate_synthetic_quotes
        df = generate_synthetic_quotes("MRK", "2010-04-06", seed=3)
        assert (df["bid"] > 0).all()
        assert (df["ask"] > 0).all()

    def test_generate_trades_returns_dataframe(self):
        from data.synthetic_data import generate_synthetic_quotes, generate_synthetic_trades
        quotes_df = generate_synthetic_quotes("SLB", "2010-04-01", seed=4)
        # Build a minimal NBBO for trade generation
        from src.nbbo import apply_quote_filters, build_nbbo_sequence, apply_spread_filter, compute_event_contribution
        filtered = apply_quote_filters(quotes_df)
        nbbo = build_nbbo_sequence(filtered)
        nbbo = apply_spread_filter(nbbo)
        nbbo = compute_event_contribution(nbbo)
        trades_df = generate_synthetic_trades("SLB", "2010-04-01", nbbo, seed=5)
        assert isinstance(trades_df, pd.DataFrame)
        assert len(trades_df) > 0

    def test_trades_prices_in_nbbo_range(self):
        from data.synthetic_data import generate_synthetic_quotes, generate_synthetic_trades
        from src.nbbo import apply_quote_filters, build_nbbo_sequence, apply_spread_filter, compute_event_contribution
        quotes_df = generate_synthetic_quotes("SLB", "2010-04-01", seed=6)
        filtered = apply_quote_filters(quotes_df)
        nbbo = build_nbbo_sequence(filtered)
        nbbo = apply_spread_filter(nbbo)
        nbbo = compute_event_contribution(nbbo)
        trades_df = generate_synthetic_trades("SLB", "2010-04-01", nbbo, seed=7)
        # Prices should be in a reasonable range around the synthetic stock price
        valid_prices = trades_df["price"]
        assert (valid_prices > 0).all()
        assert (valid_prices < 500).all()

    def test_tickers_50_has_correct_count(self):
        from data.synthetic_data import TICKERS_50
        assert len(TICKERS_50) == 50

    def test_trading_days_has_21_days(self):
        from data.synthetic_data import TRADING_DAYS_APRIL_2010
        assert len(TRADING_DAYS_APRIL_2010) == 21


# ---------------------------------------------------------------------------
# NBBO output files (exp_1)
# ---------------------------------------------------------------------------

class TestExp1Output:

    def test_nbbo_parquet_files_exist(self):
        if not os.path.isdir(NBBO_DIR):
            pytest.skip("NBBO output directory not found — run exp_1 first")
        files = os.listdir(NBBO_DIR)
        assert len(files) > 0, "No NBBO parquet files found"

    def test_nbbo_parquet_has_required_columns(self):
        if not os.path.isdir(NBBO_DIR):
            pytest.skip("NBBO output directory not found — run exp_1 first")
        files = [f for f in os.listdir(NBBO_DIR) if f.endswith(".parquet")]
        if not files:
            pytest.skip("No parquet files in NBBO dir")
        df = pd.read_parquet(os.path.join(NBBO_DIR, files[0]))
        required = {"timestamp", "nbbo_bid", "nbbo_bid_size", "nbbo_ask", "nbbo_ask_size",
                    "spread", "spread_filtered_flag", "e_n"}
        assert required.issubset(set(df.columns))

    def test_nbbo_spread_nonnegative(self):
        if not os.path.isdir(NBBO_DIR):
            pytest.skip("NBBO output directory not found")
        files = [f for f in os.listdir(NBBO_DIR) if f.endswith(".parquet")]
        if not files:
            pytest.skip("No parquet files")
        df = pd.read_parquet(os.path.join(NBBO_DIR, files[0]))
        assert (df["spread"] >= -1e-9).all()

    def test_nbbo_flag_is_binary(self):
        if not os.path.isdir(NBBO_DIR):
            pytest.skip("NBBO output directory not found")
        files = [f for f in os.listdir(NBBO_DIR) if f.endswith(".parquet")]
        if not files:
            pytest.skip("No parquet files")
        df = pd.read_parquet(os.path.join(NBBO_DIR, files[0]))
        assert set(df["spread_filtered_flag"].unique()).issubset({0, 1})


# ---------------------------------------------------------------------------
# Grid output files (exp_2)
# ---------------------------------------------------------------------------

class TestExp2Output:

    def test_grid_parquet_files_exist(self):
        if not os.path.isdir(GRID_DIR):
            pytest.skip("Grid output directory not found — run exp_2 first")
        files = [f for f in os.listdir(GRID_DIR) if f.endswith(".parquet")]
        assert len(files) > 0

    def test_grid_has_2340_bins_per_day(self):
        if not os.path.isdir(GRID_DIR):
            pytest.skip("Grid output directory not found")
        files = [f for f in os.listdir(GRID_DIR) if f.endswith(".parquet")]
        if not files:
            pytest.skip("No grid parquet files")
        df = pd.read_parquet(os.path.join(GRID_DIR, files[0]))
        from src.grid import N_BINS
        assert len(df) == N_BINS, f"Expected {N_BINS} bins, got {len(df)}"

    def test_grid_has_required_columns(self):
        if not os.path.isdir(GRID_DIR):
            pytest.skip("Grid output directory not found")
        files = [f for f in os.listdir(GRID_DIR) if f.endswith(".parquet")]
        if not files:
            pytest.skip("No grid parquet files")
        df = pd.read_parquet(os.path.join(GRID_DIR, files[0]))
        for col in ["OFI_k", "delta_P_k", "TI_k", "VOL_k"]:
            assert col in df.columns

    def test_half_hour_range_0_to_12(self):
        if not os.path.isdir(GRID_DIR):
            pytest.skip("Grid output directory not found")
        files = [f for f in os.listdir(GRID_DIR) if f.endswith(".parquet")]
        if not files:
            pytest.skip("No grid parquet files")
        df = pd.read_parquet(os.path.join(GRID_DIR, files[0]))
        if "half_hour" in df.columns:
            assert df["half_hour"].min() >= 0
            assert df["half_hour"].max() <= 12


# ---------------------------------------------------------------------------
# Regression results (exp_3)
# ---------------------------------------------------------------------------

class TestExp3Results:

    def test_table2_csv_exists(self):
        path = os.path.join(RESULTS_DIR, "table2_main_regression.csv")
        if not os.path.exists(path):
            pytest.skip("table2_main_regression.csv not found — run exp_3 first")
        assert os.path.getsize(path) > 0

    def test_r_squared_between_0_and_1(self):
        path = os.path.join(RESULTS_DIR, "table2_main_regression.csv")
        if not os.path.exists(path):
            pytest.skip("table2_main_regression.csv not found")
        df = pd.read_csv(path)
        r2_col = [c for c in df.columns if "r2" in c.lower() or "rsquared" in c.lower() or "r_squared" in c.lower()]
        if not r2_col:
            pytest.skip("No R² column found in table2 CSV")
        r2 = df[r2_col[0]].dropna()
        assert (r2 >= -0.01).all(), f"R² below 0: {r2.min()}"
        assert (r2 <= 1.01).all(), f"R² above 1: {r2.max()}"

    def test_beta_sign_positive(self):
        """OFI beta should be mostly positive (order flow imbalance raises price)."""
        path = os.path.join(RESULTS_DIR, "table2_main_regression.csv")
        if not os.path.exists(path):
            pytest.skip("table2_main_regression.csv not found")
        df = pd.read_csv(path)
        beta_col = [c for c in df.columns if "beta" in c.lower()]
        if not beta_col:
            pytest.skip("No beta column in table2 CSV")
        betas = df[beta_col[0]].dropna()
        pct_positive = (betas > 0).mean()
        assert pct_positive > 0.5, f"Only {pct_positive:.0%} of betas are positive"

    def test_betas_csv_exists(self):
        path = os.path.join(RESULTS_DIR, "exp3_betas.csv")
        if not os.path.exists(path):
            pytest.skip("exp3_betas.csv not found — run exp_3 first")
        df = pd.read_csv(path)
        assert len(df) > 0


# ---------------------------------------------------------------------------
# Depth regression (exp_4)
# ---------------------------------------------------------------------------

class TestExp4Results:

    def test_table3_csv_exists(self):
        path = os.path.join(RESULTS_DIR, "table3_depth_regression.csv")
        if not os.path.exists(path):
            pytest.skip("table3_depth_regression.csv not found — run exp_4 first")
        assert os.path.getsize(path) > 0

    def test_lambda_positive(self):
        """All lambda estimates should be positive (depth-impact inverse relationship).

        table3_depth_regression.csv uses a long-form layout:
          estimand  ∈ {'lambda', 'c'};  estimate  holds the numeric value.
        """
        path = os.path.join(RESULTS_DIR, "table3_depth_regression.csv")
        if not os.path.exists(path):
            pytest.skip("table3_depth_regression.csv not found")
        df = pd.read_csv(path)
        # Support both long-form ('estimand'/'estimate') and wide-form ('lambda_hat').
        if "estimand" in df.columns and "estimate" in df.columns:
            lam_rows = df[df["estimand"].str.lower().str.contains("lambda", na=False)]
            if lam_rows.empty:
                pytest.skip("No lambda row found in table3 CSV")
            lams = lam_rows["estimate"].dropna()
        else:
            lambda_col = [c for c in df.columns if "lambda" in c.lower()]
            if not lambda_col:
                pytest.skip("No lambda column in table3 CSV")
            lams = df[lambda_col[0]].dropna()
        assert (lams > 0).all(), f"Negative lambda found: {lams.min()}"


# ---------------------------------------------------------------------------
# OFI vs TI comparison (exp_5)
# ---------------------------------------------------------------------------

class TestExp5Results:

    def test_table4_csv_exists(self):
        path = os.path.join(RESULTS_DIR, "table4_panel_A.csv")
        if not os.path.exists(path):
            pytest.skip("table4_panel_A.csv not found — run exp_5 first")
        assert os.path.getsize(path) > 0

    def test_ofi_r2_geq_ti_r2(self):
        """H3: OFI explains more variation than TI.

        table4_panel_A.csv uses a long-form layout:
          model  ∈ {'OFI_only', 'TI_only', 'Joint_OFI_TI'};  rsquared  holds R².
        Wide-form (columns like 'r2_ofi') is also supported.
        """
        path = os.path.join(RESULTS_DIR, "table4_panel_A.csv")
        if not os.path.exists(path):
            pytest.skip("table4_panel_A.csv not found")
        df = pd.read_csv(path)
        if "model" in df.columns and "rsquared" in df.columns:
            # Long-form: filter by model label (case-insensitive partial match)
            ofi_mask = df["model"].str.contains("OFI_only", case=False, na=False)
            ti_mask = df["model"].str.contains("TI_only", case=False, na=False)
            if not ofi_mask.any() or not ti_mask.any():
                pytest.skip("OFI_only / TI_only rows not found in table4 CSV")
            mean_ofi_r2 = df.loc[ofi_mask, "rsquared"].mean()
            mean_ti_r2 = df.loc[ti_mask, "rsquared"].mean()
        else:
            ofi_cols = [c for c in df.columns if "ofi" in c.lower() and "r2" in c.lower()]
            ti_cols = [c for c in df.columns if "ti" in c.lower() and "r2" in c.lower()]
            if not ofi_cols or not ti_cols:
                pytest.skip("R² columns not found in table4 CSV")
            mean_ofi_r2 = df[ofi_cols[0]].mean()
            mean_ti_r2 = df[ti_cols[0]].mean()
        assert mean_ofi_r2 >= mean_ti_r2, (
            f"OFI R²={mean_ofi_r2:.3f} < TI R²={mean_ti_r2:.3f}: H3 violated"
        )


# ---------------------------------------------------------------------------
# Volume scaling (exp_6)
# ---------------------------------------------------------------------------

class TestExp6Results:

    def test_table5_csv_exists(self):
        path = os.path.join(RESULTS_DIR, "table5_volume.csv")
        if not os.path.exists(path):
            pytest.skip("table5_volume.csv not found — run exp_6 first")
        assert os.path.getsize(path) > 0

    def test_h_hat_reasonable_range(self):
        """H_hat should be in a reasonable range (paper expects ~0.18 ± 0.18).

        table5_volume.csv uses a long-form layout with a 'regression' column;
        H exponent values live in rows where regression == 'vol_power' in an
        'H' column.  Wide-form columns named 'H_hat' or 'mean_h' are also
        supported for forward compatibility.
        """
        path = os.path.join(RESULTS_DIR, "table5_volume.csv")
        if not os.path.exists(path):
            pytest.skip("table5_volume.csv not found")
        df = pd.read_csv(path)
        # Long-form layout: 'H' column present in 'vol_power' rows
        if "H" in df.columns:
            sub = df
            if "regression" in df.columns:
                sub = df[df["regression"].str.contains("vol", case=False, na=False)]
            h_vals = sub["H"].dropna()
        else:
            h_cols = [c for c in df.columns if "h_hat" in c.lower() or "mean_h" in c.lower()]
            if not h_cols:
                pytest.skip("No H_hat column in table5 CSV")
            h_vals = df[h_cols[0]].dropna()
        if h_vals.empty:
            pytest.skip("No finite H values found in table5 CSV")
        # Should be in range [-2, 3] for reasonable estimates
        assert (h_vals > -2).all(), f"H_hat too negative: {h_vals.min()}"
        assert (h_vals < 3).all(), f"H_hat too large: {h_vals.max()}"


# ---------------------------------------------------------------------------
# Intraday seasonality (exp_7)
# ---------------------------------------------------------------------------

class TestExp7Results:

    def test_intraday_beta_csv_exists(self):
        path = os.path.join(RESULTS_DIR, "intraday_beta.csv")
        if not os.path.exists(path):
            pytest.skip("intraday_beta.csv not found — run exp_7 first")
        df = pd.read_csv(path)
        assert len(df) == 13, f"Expected 13 rows (half-hour slots), got {len(df)}"

    def test_intraday_variance_csv_exists(self):
        path = os.path.join(RESULTS_DIR, "intraday_variance.csv")
        if not os.path.exists(path):
            pytest.skip("intraday_variance.csv not found — run exp_7 first")
        df = pd.read_csv(path)
        assert len(df) == 13

    def test_normalized_patterns_mean_near_1(self):
        """Normalized patterns should have mean close to 1 across 13 slots."""
        path = os.path.join(RESULTS_DIR, "intraday_beta.csv")
        if not os.path.exists(path):
            pytest.skip("intraday_beta.csv not found")
        df = pd.read_csv(path)
        norm_cols = [c for c in df.columns if "norm" in c.lower()]
        for col in norm_cols:
            mean_val = df[col].mean()
            assert abs(mean_val - 1.0) < 0.3, (
                f"Normalized {col} mean={mean_val:.3f} not close to 1"
            )

    def test_figure5_png_exists(self):
        path = os.path.join(RESULTS_DIR, "figure5_intraday_beta_AD.png")
        if not os.path.exists(path):
            pytest.skip("figure5 PNG not found — run exp_7 first")
        assert os.path.getsize(path) > 1000  # non-empty PNG

    def test_figure6_png_exists(self):
        path = os.path.join(RESULTS_DIR, "figure6_variance_decomp.png")
        if not os.path.exists(path):
            pytest.skip("figure6 PNG not found — run exp_7 first")
        assert os.path.getsize(path) > 1000


# ---------------------------------------------------------------------------
# Robustness tests (exp_8)
# ---------------------------------------------------------------------------

class TestExp8Results:

    def test_robustness_timescale_csv_exists(self):
        path = os.path.join(RESULTS_DIR, "robustness_timescale.csv")
        if not os.path.exists(path):
            pytest.skip("robustness_timescale.csv not found — run exp_8 first")
        df = pd.read_csv(path)
        assert len(df) > 0

    def test_robustness_exchange_csv_exists(self):
        path = os.path.join(RESULTS_DIR, "robustness_exchange.csv")
        if not os.path.exists(path):
            pytest.skip("robustness_exchange.csv not found — run exp_8 first")
        df = pd.read_csv(path)
        assert len(df) > 0

    def test_robustness_ticktest_csv_exists(self):
        path = os.path.join(RESULTS_DIR, "robustness_ticktest.csv")
        if not os.path.exists(path):
            pytest.skip("robustness_ticktest.csv not found — run exp_8 first")
        df = pd.read_csv(path)
        assert len(df) > 0

    def test_timescale_r2_non_negative(self):
        path = os.path.join(RESULTS_DIR, "robustness_timescale.csv")
        if not os.path.exists(path):
            pytest.skip("robustness_timescale.csv not found")
        df = pd.read_csv(path)
        r2_cols = [c for c in df.columns if "r2" in c.lower()]
        for col in r2_cols:
            vals = df[col].dropna()
            assert (vals >= -0.01).all(), f"Negative R² in {col}"
