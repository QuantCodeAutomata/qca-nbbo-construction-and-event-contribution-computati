# NBBO Analysis Project — Agent Memory

## Project Overview
Replication of Cont, Kukanov & Stoikov (2014) "The Price Impact of Order Book Events"
using April 2010 TAQ (Trade and Quote) data for 50 S&P 500 stocks.

## Directory Layout
```
project/
├── AGENTS.md
├── data/
│   ├── __init__.py
│   ├── synthetic_data.py   # synthetic TAQ data generator
│   └── download_data.py    # WRDS TAQ download pipeline
├── exp/
│   ├── __init__.py
│   ├── exp_1_nbbo.py       # Quote filter → NBBO → OFI events
│   ├── exp_2_grid.py       # Trade classify → 10-sec grid
│   ├── exp_3_h1_regression.py  # H1: OFI → returns (Table 2)
│   ├── exp_4_h2_depth.py   # H2: β ~ 1/AD (Table 3)
│   ├── exp_5_h3_ofi_ti.py  # H3: OFI vs TI (Table 4A)
│   ├── exp_6_h4_volume.py  # H4: volume regression (Table 5)
│   ├── exp_7_h5_seasonality.py # H5: intraday patterns (Fig 5, 6)
│   └── exp_8_robustness.py # Robustness checks
├── src/
│   ├── __init__.py
│   ├── nbbo.py             # NBBO construction from raw TAQ quotes
│   ├── grid.py             # 10-sec grid + trade classification
│   ├── regression.py       # OLS utilities (HC0, HAC, depth scaling)
│   ├── seasonality.py      # Intraday U-shape patterns + plots
│   └── robustness.py       # Timescale / exchange / ticktest checks
├── results/                # All output CSVs and PNGs land here
└── tests/
```

## Key Data Structures

### Raw Quotes (DataFrame) — input to `src/nbbo.apply_quote_filters`
Required columns: `timestamp` (datetime64), `bid` (float64), `ask` (float64),
                  `bid_size` (float/int), `ask_size` (float/int),
                  `mode` (int), `ex` (str)

### NBBO Events (DataFrame) — output of `src/nbbo.compute_event_contribution`
Columns: `timestamp`, `nbbo_bid`, `nbbo_bid_size`, `nbbo_ask`, `nbbo_ask_size`,
         `spread`, `spread_filtered_flag`, `e_n` (OFI contribution, NaN for
         day-first event), optionally `ticker`, `date`

### Raw Trades (DataFrame) — input to `src/grid.apply_trade_filters`
Required columns: `timestamp`, `price` (float), `size` (float/int),
                  `corr` (int ≤ 2 for valid), `cond` (str trade condition)

### Classified Trades (DataFrame) — output of quote/tick test classifiers
Extra columns: `trade_dir` (1=buy, -1=sell, 0=unclassified) for quote test;
               `trade_dir_tick` for tick test

### 10-sec Grid (DataFrame) — output of `src/grid.build_10sec_grid`
2340 rows (bins 0–2339), columns:
  `bin_index`, `bin_start`, `half_hour` (0–12), `OFI_k`, `mid_k`,
  `delta_P_k` (price change in tick=0.01 units), `TI_k`, `VOL_k`

### Average Depth (DataFrame) — output of `src/grid.compute_average_depth`
13 rows (slots 0–12), columns: `half_hour`, `AD`

## Parquet File Paths
- Raw quotes:   `{data_dir}/quotes/{ticker}_{date}_quotes.parquet`
- Raw trades:   `{data_dir}/trades/{ticker}_{date}_trades.parquet`
- NBBO events:  `{nbbo_dir}/{ticker}_{date}_nbbo.parquet`
- Grid:         `{grid_dir}/{ticker}_{date}_grid.parquet`

## Invalid Quote Mode Codes
`EXCLUDED_MODE_CODES = {4, 7, 9, 11, 13, 14, 15, 19, 20, 27, 28}`

## Excluded Trade Condition Codes
`EXCLUDED_TRADE_CONDITIONS = {'O','Z','B','T','L','G','W','J','K'}`

## Exchange Codes
`['N','Q','A','C','D','I','K','M','P','T','X','Y','Z']`

## Session Constants
- Market open: 09:30, close: 16:00
- Session length: 23,400 seconds
- Grid bins: 2340 (10-second bins)
- Half-hour slots: 13 (indices 0–12, labels '09:30'–'15:30')

## Validation Targets (synthetic data, 5 stocks × 21 days)
- Quote-to-trade ratio: ~40:1
- Mean best-quote depth: ~227 shares
- Mean daily total volume: ~7,512,376 shares (across all stocks)
- Mean stock price: ~$51.75

## src Module Public APIs

### src/nbbo.py
- `apply_quote_filters(df)` → filtered quotes DataFrame
- `build_nbbo_sequence(filtered_df)` → NBBO DataFrame
- `apply_spread_filter(nbbo_df)` → adds `spread_filtered_flag` column
- `compute_event_contribution(nbbo_df)` → adds `e_n` column (OFI, NaN for first/day)
- `process_stock_day(quotes_df, ticker, date)` → full pipeline, adds `ticker`/`date`
- `validate_nbbo_output(all_stocks_stats)` → dict with qtr, avg_depth, per_stock

### src/grid.py
- `apply_trade_filters(df)` → filtered trades DataFrame
- `classify_trades_quote_test(trades_df, nbbo_df)` → adds `trade_dir` (merge_asof)
- `classify_trades_tick_test(trades_df)` → adds `trade_dir_tick`
- `build_10sec_grid(nbbo_df, trades_df, date)` → 2340-row grid DataFrame
- `compute_average_depth(nbbo_df)` → 13-row AD DataFrame
- `save_grid_data(grid_df, ticker, output_dir)` → saves parquet, returns path

### src/regression.py
- `run_ols_hc0(y, X)` → dict{params, tvalues, bse, rsquared, fvalue, nobs}
- `run_ols_hac(y, X, n_lags=None)` → dict (adds n_lags key); default BW=floor(4*(T/100)^(2/9))
- `run_ofi_regression(subdf)` → {alpha, beta_ofi, t_*, rsquared, fvalue, nobs}
- `run_ofi_quadratic_regression(subdf)` → adds beta_ofi2 (OFI_k*|OFI_k| term)
- `run_ti_regression(subdf)` → {alpha, beta_ti, ...}
- `run_joint_ofi_ti_regression(subdf)` → {alpha, beta_ofi, beta_ti, ...}
- `run_abs_ofi_regression(subdf)` → |delta_P| on |OFI|
- `run_volume_regression(subdf)` → two-stage H estimation + c; {H, t_H, c, t_c, ...}
- `summarize_subsamples(results_list)` → {means, mean_abs_t, sig_rate}
- `run_depth_log_regression(beta_arr, ad_arr)` → {lambda_hat, t_lambda, ...} (NW HAC)
- `run_depth_power_regression(beta_arr, ad_arr, lambda_hat)` → {c_hat, t_c, ...} (NW HAC)

### src/seasonality.py
- `compute_intraday_beta_pattern(beta_df)` → length-13 Series (normalised mean=1)
- `compute_intraday_depth_pattern(ad_df)` → length-13 Series
- `compute_intraday_variance_patterns(grid_df)` → (var_dP_norm, var_OFI_norm) Series pair
- `compute_predicted_volatility(beta_by_slot, var_ofi_by_slot)` → length-13 Series
- `cross_stock_average(pattern_list)` → length-13 Series mean across stocks
- `plot_beta_depth_patterns(beta_norm, ad_norm, output_path)` → Figure 5; Agg backend
- `plot_volatility_decomposition(var_dP_norm, var_OFI_norm, pred_norm, output_path)` → Figure 6

### src/robustness.py
- `build_event_grid(nbbo_df, window_size)` → DataFrame of non-overlapping event windows
- `build_time_grid(nbbo_df, trades_df, freq_seconds)` → arbitrary-frequency grid
- `run_timescale_robustness(nbbo_df, trades_df, timescales)` → dict per label
- `compute_nyse_only_ofi(quotes_df)` → NBBO-pipeline restricted to ex=='N'
- `run_exchange_robustness(nbbo_df, nyse_ofi_df)` → {nbbo_result, nyse_result, ofi_correlation}
- `run_ticktest_robustness(grid_df_quote, grid_df_tick)` → {quote/tick results, ti_correlation}

## Pipeline Order
exp_1 → exp_2 → exp_3 → exp_4 → exp_5 → exp_6 → exp_7 → exp_8

Run full pipeline from project root:
```bash
python -m data.synthetic_data          # generate synthetic TAQ data
python exp/exp_1_nbbo.py               # NBBO events
python exp/exp_2_grid.py               # 10-sec grids
python exp/exp_3_h1_regression.py      # Table 2
python exp/exp_4_h2_depth.py           # Table 3
python exp/exp_5_h3_ofi_ti.py          # Table 4
python exp/exp_6_h4_volume.py          # Table 5
python exp/exp_7_h5_seasonality.py     # Figures 5, 6
python exp/exp_8_robustness.py         # Robustness tables
```

## Verified Smoke-Test Results (5 stocks × 21 days)
| Script | Key metric | Observed |
|--------|-----------|----------|
| exp_1  | Quote-to-trade ratio | 35.9 |
| exp_1  | Mean best-quote depth | 271 shares |
| exp_2  | Mean stock price | $51.45 |
| exp_3  | Mean β_OFI (linear) | 0.0041 |
| exp_3  | Mean \|t_β\| | 24.3 |
| exp_3  | Mean R² | 0.386 |
| exp_4  | Stage-1 λ̂ | −0.630 (expected ≈ −1) |
| exp_8  | Timescale 10s β | 0.00412 |
| exp_8  | Timescale 60s β | 0.00472 |

## Experiment Output Files (results/)
- `table2_main_regression.csv`  – H1 per-subsample OFI regression results
- `table2_summary.csv`          – H1 mean β, |t|, sig-rate across subsamples
- `exp3_betas.csv`              – per-stock pooled β̂ and mean AD (used by exp4, exp7)
- `table3_depth_regression.csv` – H2 Stage-1 log-log and Stage-2 power results
- `table4_panel_A.csv`          – H3 TI-only, OFI-only, Joint per-subsample results
- `table4_summary.csv`          – H3 mean R² comparison
- `table5_volume.csv`           – H4 |OFI|, |TI|, volume-power per-subsample
- `table5_summary.csv`          – H4 summary
- `intraday_beta.csv`           – H5 per-slot normalised β and AD
- `intraday_variance.csv`       – H5 per-slot variance decomposition
- `robustness_timescale.csv`    – Test 1: β/R² across {5,10,20,30,60}-sec grids
- `robustness_exchange.csv`     – Test 2: all-exchanges vs NYSE-only
- `robustness_ticktest.csv`     – Test 3: quote-test vs tick-test classification
- `acf_ofi_{ticker}.png`        – ACF of OFI (first ticker or AMD)
- `acf_dP_{ticker}.png`         – ACF of Δp
- `scatter_ofi_vs_dP_{ticker}.png` – OFI scatter with fitted line
- `scatter_logbeta_logAD.png`   – cross-stock log-log depth scatter
- `scatter_daylogbeta_logAD_{ticker}.png` – per-day depth scatter for reference ticker
- `figure5_intraday_beta_AD.png` – Figure 5: normalised β and AD intraday patterns
- `figure6_variance_decomp.png`  – Figure 6: realised / explained / residual variance

## Notes
- All scripts must be run from `/workspace/project/` so that `src.*` imports resolve.
- Synthetic data uses seed=42 for reproducibility; each ticker uses seed=hash(ticker).
- `build_10sec_grid`: bins with no NBBO update carry the last known mid forward (ffill).
  Bins with no trades get TI_k=0, VOL_k=0. First bin always has delta_P_k=NaN.
- OFI (e_n) is computed in share units (not dollar-weighted) to match the paper.
- `compute_event_contribution` uses vectorised numpy.where; the first row of each
  calendar day has e_n=NaN (no prior NBBO state).
- `classify_trades_quote_test` uses pd.merge_asof with direction='backward', tolerance=1s.
- `classify_trades_tick_test`: zero-tick carries forward last non-zero direction; first
  trade always gets direction=0.
- `matplotlib.use('Agg')` is set at import in both seasonality.py and robustness.py to
  avoid display errors in headless environments.
- `run_ols_hac` default n_lags: floor(4*(T/100)^(2/9)) per Andrews/Newey-West rule.
- `build_event_grid` discards trailing partial windows (len % window_size remainder).
