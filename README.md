# NBBO Construction and Order Flow Imbalance Analysis

Replication of **Cont, Kukanov & Stoikov (2014)** — *"The Price Impact of Order Book Events"* — using NYSE TAQ data for 50 S&P 500 stocks over 21 trading days in April 2010.

## Research Hypotheses

| ID | Hypothesis | Expected Result |
|----|-----------|----------------|
| H1 | Mid-price changes are linearly related to OFI | Mean R² ≈ 65%, β ≈ 0.04, |t| ≈ 11.5 |
| H2 | Price impact β is inversely proportional to depth AD | λ ≈ 0.98 ≈ 1, R² ≈ 74% |
| H3 | OFI explains price changes better than trade imbalance | OFI R² ≈ 65% vs TI R² ≈ 32% |
| H4 | Square-root price-volume relation is a statistical artifact | |OFI| R² ≈ 58% > VOL^H R² ≈ 23% |
| H5 | Intraday volatility patterns are explained by β²·Var[OFI] | corr(var[ΔP], β²·var[OFI]) > 0.9 |

## Repository Structure

```
├── data/                        # Data loading and synthetic generation
│   ├── synthetic_data.py        # Synthetic TAQ data generator (used when WRDS unavailable)
│   └── download_data.py         # WRDS download pipeline (requires institutional access)
├── src/                         # Core implementation
│   ├── nbbo.py                  # NBBO construction and event contribution (exp_1)
│   ├── grid.py                  # 10-second grid aggregation (exp_2)
│   ├── regression.py            # OLS/HC0/HAC regression utilities (exp_3–6)
│   ├── seasonality.py           # Intraday pattern computation (exp_7)
│   └── robustness.py            # Timescale/exchange/tick-test robustness (exp_8)
├── exp/                         # Experiment scripts
│   ├── exp_1_nbbo.py            # H0: NBBO construction and OFI events
│   ├── exp_2_grid.py            # 10-second grid aggregation
│   ├── exp_3_h1_regression.py   # H1: OFI→ΔP linear regression (Table 2)
│   ├── exp_4_h2_depth.py        # H2: β ~ 1/AD depth-impact (Table 3)
│   ├── exp_5_h3_ofi_ti.py       # H3: OFI vs TI comparison (Table 4)
│   ├── exp_6_h4_volume.py       # H4: Volume scaling artifact (Table 5)
│   ├── exp_7_h5_seasonality.py  # H5: Intraday seasonality (Figures 5–6)
│   └── exp_8_robustness.py      # Robustness: timescale, exchange, tick test
├── tests/                       # pytest test suite
│   ├── test_nbbo.py             # Unit tests for src/nbbo.py
│   ├── test_grid.py             # Unit tests for src/grid.py
│   ├── test_regression.py       # Unit tests for src/regression.py
│   └── test_exp_integration.py  # Integration tests for experiment outputs
├── results/                     # Outputs: tables, plots, logs
│   └── RESULTS.md               # Summary of experiments and findings
└── AGENTS.md                    # Agent memory (persistent context)
```

## Installation

```bash
pip install numpy pandas statsmodels scipy matplotlib seaborn pyarrow
```

All dependencies are available in the project container (see requirements list in system prompt).

## Running Experiments

### With Synthetic Data (no WRDS required)

Run the full pipeline sequentially:

```bash
# Step 1: Build NBBO sequences
python exp/exp_1_nbbo.py --n_stocks 5 --data_dir data/raw --output_dir results/nbbo

# Step 2: Build 10-second grid
python exp/exp_2_grid.py --n_stocks 5 --nbbo_dir results/nbbo --trade_dir data/raw --output_dir results/grid

# Step 3: H1 - OFI regression (Table 2)
python exp/exp_3_h1_regression.py --n_stocks 5 --grid_dir results/grid --results_dir results

# Step 4: H2 - Depth-impact regression (Table 3)
python exp/exp_4_h2_depth.py --n_stocks 5 --grid_dir results/grid \
    --exp3_results_path results/exp3_betas.csv --results_dir results

# Step 5: H3 - OFI vs TI (Table 4)
python exp/exp_5_h3_ofi_ti.py --n_stocks 5 --grid_dir results/grid --results_dir results

# Step 6: H4 - Volume scaling (Table 5)
python exp/exp_6_h4_volume.py --n_stocks 5 --grid_dir results/grid --results_dir results

# Step 7: H5 - Intraday seasonality (Figures 5–6)
python exp/exp_7_h5_seasonality.py --n_stocks 5 --grid_dir results/grid \
    --exp3_results_path results/exp3_betas.csv --results_dir results

# Step 8: Robustness checks
python exp/exp_8_robustness.py --tickers_subset AMD APOL AXP AZO BAC \
    --nbbo_dir results/nbbo --grid_dir results/grid \
    --trade_dir data/raw --results_dir results
```

### With WRDS Data (institutional access required)

1. Set up WRDS credentials: `export WRDS_USERNAME=your_username`
2. Use `data/download_data.py` to fetch real TAQ data
3. Run experiments with `--n_stocks 50`

See `data/download_data.py` for detailed WRDS connection instructions.

## Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run specific test file
pytest tests/test_nbbo.py -v

# Run only integration tests (requires experiment outputs)
pytest tests/test_exp_integration.py -v
```

Current status: **139 passed · 0 skipped · 0 failed**

## Key Methodology

### NBBO Construction (exp_1)
- Filter quotes: time [09:30, 16:00], positive prices/sizes, exclude mode codes {4,7,9,11,13,14,15,19,20,27,28}
- Sequential scan over exchange matrix: NBBO bid = max(bids), NBBO ask = min(asks)
- Spread filter: remove events with spread > 95th percentile
- Event contribution `e_n`: bid-side + ask-side indicator function

### OFI Regression (exp_3)
- Per-half-hour subsample: `ΔP_k = α + β·OFI_k + ε`
- White HC0 heteroscedasticity-robust standard errors
- Quadratic robustness: `ΔP_k = α + β·OFI_k + γ·OFI_k|OFI_k| + ε`

### Depth-Impact Scaling (exp_4)
- Stage 1: log(β̂_i) = α_L - λ·log(AD_i) + ε_L (Newey-West SE)
- Stage 2: β̂_i = α_M + c·AD_i^{-λ̂} + ε_M (Newey-West SE)

## Results Summary

See [results/RESULTS.md](results/RESULTS.md) for detailed experiment results and comparisons to paper benchmarks.

## References

Cont, R., Kukanov, A., & Stoikov, S. (2014). The price impact of order book events. *Journal of Financial Econometrics*, 12(1), 47–88.
