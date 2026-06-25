"""
regression.py - OLS regression utilities for NBBO price-impact analysis.

Implements the suite of cross-sectional and time-series regressions used to
quantify the price impact of order-flow imbalance (OFI), trade imbalance (TI),
and volume.  Includes heteroskedasticity-robust (HC0) and HAC (Newey-West)
standard error variants, two-stage depth scaling regressions, and subsample
summary utilities.

References:
    Cont, Kukanov & Stoikov (2014). The price impact of order book events.
    White (1980). A heteroskedasticity-consistent covariance matrix estimator.
    Newey & West (1987). A simple positive semi-definite HAC estimator.
"""

import math
import warnings

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import pearsonr
from statsmodels.regression.linear_model import OLS


# ---------------------------------------------------------------------------
# Low-level OLS helpers
# ---------------------------------------------------------------------------


def run_ols_hc0(y: np.ndarray, X: np.ndarray) -> dict:
    """Ordinary Least Squares with White HC0 heteroskedasticity-robust SE.

    Parameters
    ----------
    y : np.ndarray, shape (T,)
        Dependent variable vector.
    X : np.ndarray, shape (T, k)
        Regressor matrix.  A constant column must already be included if an
        intercept is desired (use :func:`statsmodels.api.add_constant`).

    Returns
    -------
    dict
        Keys: ``'params'`` (ndarray), ``'tvalues'`` (ndarray),
        ``'rsquared'`` (float), ``'fvalue'`` (float),
        ``'nobs'`` (int), ``'bse'`` (ndarray).

    Raises
    ------
    ValueError
        If *y* and *X* have incompatible lengths or contain all-NaN values.
    """
    y = np.asarray(y, dtype=float)
    X = np.asarray(X, dtype=float)
    if y.ndim != 1:
        raise ValueError("run_ols_hc0: y must be 1-dimensional")
    if X.ndim != 2:
        raise ValueError("run_ols_hc0: X must be 2-dimensional")
    if len(y) != X.shape[0]:
        raise ValueError(
            f"run_ols_hc0: y length {len(y)} != X rows {X.shape[0]}"
        )

    # Drop rows with any NaN (Context7: statsmodels OLS)
    valid = np.isfinite(y) & np.all(np.isfinite(X), axis=1)
    if valid.sum() < X.shape[1] + 1:
        raise ValueError(
            "run_ols_hc0: insufficient non-NaN observations for regression"
        )

    model = OLS(y[valid], X[valid])
    result = model.fit(cov_type="HC0")

    return {
        "params": result.params,
        "tvalues": result.tvalues,
        "bse": result.bse,
        "rsquared": result.rsquared,
        "fvalue": result.fvalue,
        "nobs": int(result.nobs),
    }


def run_ols_hac(
    y: np.ndarray,
    X: np.ndarray,
    n_lags: int | None = None,
) -> dict:
    """Ordinary Least Squares with Newey-West HAC standard errors.

    The default bandwidth follows Andrews (1991) / the common empirical
    rule:  ``floor(4 × (T / 100) ^ (2/9))``.

    Parameters
    ----------
    y : np.ndarray, shape (T,)
        Dependent variable vector.
    X : np.ndarray, shape (T, k)
        Regressor matrix (include a constant column if needed).
    n_lags : int or None, optional
        Number of lags for the Newey-West kernel.  If *None* the automatic
        bandwidth formula is used.

    Returns
    -------
    dict
        Same keys as :func:`run_ols_hc0`.

    Raises
    ------
    ValueError
        If inputs are incompatible.
    """
    y = np.asarray(y, dtype=float)
    X = np.asarray(X, dtype=float)
    if y.ndim != 1:
        raise ValueError("run_ols_hac: y must be 1-dimensional")
    if X.ndim != 2:
        raise ValueError("run_ols_hac: X must be 2-dimensional")
    if len(y) != X.shape[0]:
        raise ValueError(
            f"run_ols_hac: y length {len(y)} != X rows {X.shape[0]}"
        )

    valid = np.isfinite(y) & np.all(np.isfinite(X), axis=1)
    if valid.sum() < X.shape[1] + 1:
        raise ValueError(
            "run_ols_hac: insufficient non-NaN observations for regression"
        )

    T = valid.sum()
    if n_lags is None:
        # Andrews / Newey-West automatic bandwidth selection
        n_lags = int(math.floor(4.0 * (T / 100.0) ** (2.0 / 9.0)))

    model = OLS(y[valid], X[valid])
    result = model.fit(cov_type="HAC", cov_kwds={"maxlags": n_lags})

    return {
        "params": result.params,
        "tvalues": result.tvalues,
        "bse": result.bse,
        "rsquared": result.rsquared,
        "fvalue": result.fvalue,
        "nobs": int(result.nobs),
        "n_lags": n_lags,
    }


# ---------------------------------------------------------------------------
# Subsample-level regression wrappers
# ---------------------------------------------------------------------------

def _prep_xy(
    subdf: pd.DataFrame,
    y_col: str,
    x_cols: list[str],
    add_const: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Extract and validate (y, X) from *subdf*, dropping NaN rows.

    Parameters
    ----------
    subdf : pd.DataFrame
        Subsample grid data.
    y_col : str
        Name of the dependent variable column.
    x_cols : list[str]
        Names of the regressor columns.
    add_const : bool
        If True prepend a constant column via
        :func:`statsmodels.api.add_constant`.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        ``(y, X)`` arrays with NaN rows removed.

    Raises
    ------
    ValueError
        If required columns are absent or fewer than 5 valid observations
        remain.
    """
    all_cols = [y_col] + x_cols
    missing = set(all_cols) - set(subdf.columns)
    if missing:
        raise ValueError(f"_prep_xy: missing columns {missing}")

    sub = subdf[all_cols].replace([np.inf, -np.inf], np.nan).dropna()
    if len(sub) < 5:
        raise ValueError(
            f"_prep_xy: only {len(sub)} valid rows after dropping NaN"
        )

    y = sub[y_col].to_numpy(dtype=float)
    X_raw = sub[x_cols].to_numpy(dtype=float)
    X = sm.add_constant(X_raw, has_constant="add") if add_const else X_raw
    return y, X


def run_ofi_regression(subdf: pd.DataFrame) -> dict:
    """Regress price change on OFI: delta_P_k = alpha + beta * OFI_k + eps.

    Uses HC0 standard errors.

    Parameters
    ----------
    subdf : pd.DataFrame
        Grid data with columns ``delta_P_k`` and ``OFI_k``.

    Returns
    -------
    dict
        Keys: ``'alpha'``, ``'beta_ofi'``, ``'t_alpha'``, ``'t_beta_ofi'``,
        ``'rsquared'``, ``'fvalue'``, ``'nobs'``.

    Raises
    ------
    ValueError
        If required columns are absent or insufficient observations.
    """
    y, X = _prep_xy(subdf, "delta_P_k", ["OFI_k"])
    res = run_ols_hc0(y, X)
    return {
        "alpha": float(res["params"][0]),
        "beta_ofi": float(res["params"][1]),
        "t_alpha": float(res["tvalues"][0]),
        "t_beta_ofi": float(res["tvalues"][1]),
        "rsquared": float(res["rsquared"]),
        "fvalue": float(res["fvalue"]),
        "nobs": res["nobs"],
    }


def run_ofi_quadratic_regression(subdf: pd.DataFrame) -> dict:
    """OFI regression with a signed-quadratic term.

    Fits: delta_P_k = alpha + beta1 * OFI_k + beta2 * OFI_k * |OFI_k| + eps

    The extra term ``OFI_k * |OFI_k|`` captures convex/concave price impact.
    HC0 standard errors are used.

    Parameters
    ----------
    subdf : pd.DataFrame
        Grid data with columns ``delta_P_k`` and ``OFI_k``.

    Returns
    -------
    dict
        Keys: ``'alpha'``, ``'beta_ofi'``, ``'beta_ofi2'``, ``'t_alpha'``,
        ``'t_beta_ofi'``, ``'t_beta_ofi2'``, ``'rsquared'``, ``'fvalue'``,
        ``'nobs'``.

    Raises
    ------
    ValueError
        If required columns are absent or insufficient observations.
    """
    work = subdf[["delta_P_k", "OFI_k"]].copy().replace(
        [np.inf, -np.inf], np.nan
    ).dropna()
    if len(work) < 5:
        raise ValueError("run_ofi_quadratic_regression: insufficient observations")

    work["OFI_sq"] = work["OFI_k"] * np.abs(work["OFI_k"])
    y, X = _prep_xy(work, "delta_P_k", ["OFI_k", "OFI_sq"])
    res = run_ols_hc0(y, X)
    return {
        "alpha": float(res["params"][0]),
        "beta_ofi": float(res["params"][1]),
        "beta_ofi2": float(res["params"][2]),
        "t_alpha": float(res["tvalues"][0]),
        "t_beta_ofi": float(res["tvalues"][1]),
        "t_beta_ofi2": float(res["tvalues"][2]),
        "rsquared": float(res["rsquared"]),
        "fvalue": float(res["fvalue"]),
        "nobs": res["nobs"],
    }


def run_ti_regression(subdf: pd.DataFrame) -> dict:
    """Regress price change on trade imbalance: delta_P_k = alpha + beta * TI_k + eps.

    Uses HC0 standard errors.

    Parameters
    ----------
    subdf : pd.DataFrame
        Grid data with columns ``delta_P_k`` and ``TI_k``.

    Returns
    -------
    dict
        Keys: ``'alpha'``, ``'beta_ti'``, ``'t_alpha'``, ``'t_beta_ti'``,
        ``'rsquared'``, ``'fvalue'``, ``'nobs'``.

    Raises
    ------
    ValueError
        If required columns are absent or insufficient observations.
    """
    y, X = _prep_xy(subdf, "delta_P_k", ["TI_k"])
    res = run_ols_hc0(y, X)
    return {
        "alpha": float(res["params"][0]),
        "beta_ti": float(res["params"][1]),
        "t_alpha": float(res["tvalues"][0]),
        "t_beta_ti": float(res["tvalues"][1]),
        "rsquared": float(res["rsquared"]),
        "fvalue": float(res["fvalue"]),
        "nobs": res["nobs"],
    }


def run_joint_ofi_ti_regression(subdf: pd.DataFrame) -> dict:
    """Joint regression of price change on OFI and TI.

    Fits: delta_P_k = alpha + beta_ofi * OFI_k + beta_ti * TI_k + eps

    Uses HC0 standard errors.

    Parameters
    ----------
    subdf : pd.DataFrame
        Grid data with columns ``delta_P_k``, ``OFI_k``, and ``TI_k``.

    Returns
    -------
    dict
        Keys: ``'alpha'``, ``'beta_ofi'``, ``'beta_ti'``, ``'t_alpha'``,
        ``'t_beta_ofi'``, ``'t_beta_ti'``, ``'rsquared'``, ``'fvalue'``,
        ``'nobs'``.

    Raises
    ------
    ValueError
        If required columns are absent or insufficient observations.
    """
    y, X = _prep_xy(subdf, "delta_P_k", ["OFI_k", "TI_k"])
    res = run_ols_hc0(y, X)
    return {
        "alpha": float(res["params"][0]),
        "beta_ofi": float(res["params"][1]),
        "beta_ti": float(res["params"][2]),
        "t_alpha": float(res["tvalues"][0]),
        "t_beta_ofi": float(res["tvalues"][1]),
        "t_beta_ti": float(res["tvalues"][2]),
        "rsquared": float(res["rsquared"]),
        "fvalue": float(res["fvalue"]),
        "nobs": res["nobs"],
    }


def run_abs_ofi_regression(subdf: pd.DataFrame) -> dict:
    """Regress absolute price change on absolute OFI.

    Fits: |delta_P_k| = alpha + beta * |OFI_k| + eps

    Uses HC0 standard errors.

    Parameters
    ----------
    subdf : pd.DataFrame
        Grid data with columns ``delta_P_k`` and ``OFI_k``.

    Returns
    -------
    dict
        Keys: ``'alpha'``, ``'beta_abs_ofi'``, ``'t_alpha'``,
        ``'t_beta_abs_ofi'``, ``'rsquared'``, ``'fvalue'``, ``'nobs'``.

    Raises
    ------
    ValueError
        If required columns are absent or insufficient observations.
    """
    work = subdf[["delta_P_k", "OFI_k"]].copy().replace(
        [np.inf, -np.inf], np.nan
    ).dropna()
    if len(work) < 5:
        raise ValueError("run_abs_ofi_regression: insufficient observations")

    work["abs_dP"] = np.abs(work["delta_P_k"])
    work["abs_ofi"] = np.abs(work["OFI_k"])
    y, X = _prep_xy(work, "abs_dP", ["abs_ofi"])
    res = run_ols_hc0(y, X)
    return {
        "alpha": float(res["params"][0]),
        "beta_abs_ofi": float(res["params"][1]),
        "t_alpha": float(res["tvalues"][0]),
        "t_beta_abs_ofi": float(res["tvalues"][1]),
        "rsquared": float(res["rsquared"]),
        "fvalue": float(res["fvalue"]),
        "nobs": res["nobs"],
    }


def run_volume_regression(subdf: pd.DataFrame) -> dict:
    """Two-stage volume scaling regression for the Hurst exponent H.

    Stage 1 – estimate H:
        log|delta_P_k| = const + H * log(VOL_k) + eps
        (only bins with VOL_k > 0 and |delta_P_k| > 0 are used)

    Stage 2 – estimate the proportionality constant c:
        |delta_P_k| = c * VOL_k ^ H + eps  (OLS through origin not used;
        a constant is included for robustness)

    HC0 standard errors are used for both stages.

    Parameters
    ----------
    subdf : pd.DataFrame
        Grid data with columns ``delta_P_k`` and ``VOL_k``.

    Returns
    -------
    dict
        Keys: ``'H'`` (float), ``'t_H'`` (float), ``'c'`` (float),
        ``'t_c'`` (float), ``'rsquared_stage1'``, ``'rsquared_stage2'``,
        ``'nobs_stage1'``, ``'nobs_stage2'``.

    Raises
    ------
    ValueError
        If required columns are absent or insufficient observations remain
        after taking logarithms.
    """
    required = {"delta_P_k", "VOL_k"}
    missing = required - set(subdf.columns)
    if missing:
        raise ValueError(f"run_volume_regression: missing columns {missing}")

    work = subdf[["delta_P_k", "VOL_k"]].copy().replace(
        [np.inf, -np.inf], np.nan
    ).dropna()
    pos_mask = (work["VOL_k"] > 0) & (np.abs(work["delta_P_k"]) > 0)
    work_log = work.loc[pos_mask].copy()
    if len(work_log) < 5:
        raise ValueError(
            "run_volume_regression: insufficient positive-valued observations "
            "for log regression"
        )

    work_log["log_abs_dP"] = np.log(np.abs(work_log["delta_P_k"]))
    work_log["log_vol"] = np.log(work_log["VOL_k"])

    y1, X1 = _prep_xy(work_log, "log_abs_dP", ["log_vol"])
    res1 = run_ols_hc0(y1, X1)
    H = float(res1["params"][1])

    # Stage 2: |delta_P_k| = alpha + c * VOL_k^H
    work2 = work.loc[work["VOL_k"] > 0].copy()
    if len(work2) < 5:
        raise ValueError("run_volume_regression: insufficient observations for stage 2")
    work2["vol_H"] = work2["VOL_k"].to_numpy(dtype=float) ** H
    work2["abs_dP"] = np.abs(work2["delta_P_k"])

    y2, X2 = _prep_xy(work2, "abs_dP", ["vol_H"])
    res2 = run_ols_hc0(y2, X2)

    return {
        "H": H,
        "t_H": float(res1["tvalues"][1]),
        "c": float(res2["params"][1]),
        "t_c": float(res2["tvalues"][1]),
        "rsquared_stage1": float(res1["rsquared"]),
        "rsquared_stage2": float(res2["rsquared"]),
        "nobs_stage1": res1["nobs"],
        "nobs_stage2": res2["nobs"],
    }


# ---------------------------------------------------------------------------
# Subsample summary
# ---------------------------------------------------------------------------


def summarize_subsamples(results_list: list[dict]) -> dict:
    """Aggregate regression results across multiple subsamples.

    For each numeric key present in the individual result dictionaries,
    computes:
      - mean across subsamples
      - mean |t-statistic| for keys ending in ``'t_'`` prefix
      - significance rate: fraction of subsamples with |t| > 1.96

    Significance rates are reported for each ``t_*`` key.

    Parameters
    ----------
    results_list : list[dict]
        List of result dictionaries, one per subsample (e.g., per
        stock-day).  All dicts must share the same set of numeric keys.

    Returns
    -------
    dict
        Nested: ``{'means': {...}, 'mean_abs_t': {...}, 'sig_rate': {...}}``.

    Raises
    ------
    ValueError
        If *results_list* is empty.
    """
    if not results_list:
        raise ValueError("summarize_subsamples: results_list is empty")

    all_keys = [k for k, v in results_list[0].items() if isinstance(v, (int, float))]

    means: dict = {}
    mean_abs_t: dict = {}
    sig_rate: dict = {}

    for key in all_keys:
        vals = np.array(
            [r[key] for r in results_list if key in r and r[key] is not None],
            dtype=float,
        )
        vals = vals[np.isfinite(vals)]
        means[key] = float(np.mean(vals)) if len(vals) > 0 else np.nan

    t_keys = [k for k in all_keys if k.startswith("t_")]
    for tkey in t_keys:
        t_vals = np.array(
            [r[tkey] for r in results_list if tkey in r and r[tkey] is not None],
            dtype=float,
        )
        t_vals = t_vals[np.isfinite(t_vals)]
        mean_abs_t[tkey] = float(np.mean(np.abs(t_vals))) if len(t_vals) > 0 else np.nan
        sig_rate[tkey] = float(np.mean(np.abs(t_vals) > 1.96)) if len(t_vals) > 0 else np.nan

    return {"means": means, "mean_abs_t": mean_abs_t, "sig_rate": sig_rate}


# ---------------------------------------------------------------------------
# Depth scaling regressions (two-stage)
# ---------------------------------------------------------------------------


def run_depth_log_regression(
    beta_hat_arr: np.ndarray,
    ad_arr: np.ndarray,
) -> dict:
    """Stage 1 log-log regression of price-impact beta on average depth.

    Fits: log(beta_hat_i) = const - lambda * log(AD_i) + eps

    The economic relationship is beta = c / AD^lambda (inverse depth-impact),
    so the OLS slope on log(AD) equals -lambda.  ``lambda_hat`` is returned
    as a positive number: lambda_hat = -(OLS slope on log AD).

    Estimates ``lambda`` (depth scaling exponent) with Newey-West HAC
    standard errors.  Only positive, finite pairs are included.

    Parameters
    ----------
    beta_hat_arr : np.ndarray
        Per-slot (or per-stock) estimated price-impact coefficients.
    ad_arr : np.ndarray
        Corresponding average depths (same length as *beta_hat_arr*).

    Returns
    -------
    dict
        Keys: ``'lambda_hat'`` (float), ``'t_lambda'`` (float),
        ``'rsquared'``, ``'nobs'``.

    Raises
    ------
    ValueError
        If arrays have different lengths or insufficient positive observations.
    """
    beta_hat_arr = np.asarray(beta_hat_arr, dtype=float)
    ad_arr = np.asarray(ad_arr, dtype=float)
    if len(beta_hat_arr) != len(ad_arr):
        raise ValueError(
            "run_depth_log_regression: beta_hat_arr and ad_arr must have the same length"
        )

    valid = (
        np.isfinite(beta_hat_arr)
        & np.isfinite(ad_arr)
        & (beta_hat_arr > 0)
        & (ad_arr > 0)
    )
    if valid.sum() < 3:
        raise ValueError(
            "run_depth_log_regression: fewer than 3 valid positive pairs"
        )

    log_beta = np.log(beta_hat_arr[valid])
    log_ad = np.log(ad_arr[valid])
    X = sm.add_constant(log_ad.reshape(-1, 1), has_constant="add")

    res = run_ols_hac(log_beta, X)

    # Slope on log(AD) equals -lambda; negate so lambda_hat > 0.
    return {
        "lambda_hat": -float(res["params"][1]),
        "t_lambda": -float(res["tvalues"][1]),
        "rsquared": float(res["rsquared"]),
        "nobs": res["nobs"],
    }


def run_depth_power_regression(
    beta_hat_arr: np.ndarray,
    ad_arr: np.ndarray,
    lambda_hat: float,
) -> dict:
    """Stage 2 power regression of price-impact beta on depth^lambda.

    Fits: beta_hat_i = const + c * AD_i ^ lambda_hat + eps

    Uses the ``lambda_hat`` estimated in Stage 1 (:func:`run_depth_log_regression`).
    Newey-West HAC standard errors are used.

    Parameters
    ----------
    beta_hat_arr : np.ndarray
        Per-slot estimated price-impact coefficients.
    ad_arr : np.ndarray
        Corresponding average depths.
    lambda_hat : float
        Depth scaling exponent from Stage 1.

    Returns
    -------
    dict
        Keys: ``'c_hat'`` (float), ``'t_c'`` (float),
        ``'rsquared'``, ``'nobs'``.

    Raises
    ------
    ValueError
        If arrays have different lengths or insufficient valid observations.
    """
    beta_hat_arr = np.asarray(beta_hat_arr, dtype=float)
    ad_arr = np.asarray(ad_arr, dtype=float)
    if len(beta_hat_arr) != len(ad_arr):
        raise ValueError(
            "run_depth_power_regression: arrays must have the same length"
        )
    if not np.isfinite(lambda_hat):
        raise ValueError("run_depth_power_regression: lambda_hat must be finite")

    valid = np.isfinite(beta_hat_arr) & np.isfinite(ad_arr) & (ad_arr > 0)
    if valid.sum() < 3:
        raise ValueError(
            "run_depth_power_regression: fewer than 3 valid observations"
        )

    # Predictor is z_i = AD_i^{-lambda_hat}  (inverse depth-impact: beta = c/AD^lambda)
    ad_powered = ad_arr[valid] ** (-lambda_hat)
    X = sm.add_constant(ad_powered.reshape(-1, 1), has_constant="add")
    y = beta_hat_arr[valid]

    res = run_ols_hac(y, X)

    return {
        "c_hat": float(res["params"][1]),
        "t_c": float(res["tvalues"][1]),
        "rsquared": float(res["rsquared"]),
        "nobs": res["nobs"],
    }
