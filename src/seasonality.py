"""
seasonality.py - Intraday seasonality patterns for OFI price-impact analysis.

Computes and visualises the U-shaped intraday patterns of the OFI price-impact
coefficient (beta), quoted depth (AD), return variance, OFI variance, and
predicted volatility.  Output figures correspond to Figures 5 and 6 of
Cont, Kukanov & Stoikov (2014).

References:
    Cont, Kukanov & Stoikov (2014). The price impact of order book events.
    Wood, McInish & Ord (1985). An investigation of transactions data.
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from statsmodels.tsa.stattools import acf


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

N_SLOTS = 13  # half-hour slots 0–12 covering 09:30–16:00
SLOT_LABELS = [
    "09:30", "10:00", "10:30", "11:00", "11:30", "12:00",
    "12:30", "13:00", "13:30", "14:00", "14:30", "15:00", "15:30",
]


# ---------------------------------------------------------------------------
# Intraday pattern computation
# ---------------------------------------------------------------------------


def compute_intraday_beta_pattern(beta_df: pd.DataFrame) -> pd.Series:
    """Compute the normalised intraday pattern of OFI price-impact beta.

    Groups per-subsample beta estimates by half-hour slot, averages within
    each slot, then normalises so that the cross-slot mean equals 1.

    Parameters
    ----------
    beta_df : pd.DataFrame
        One row per subsample (stock-day or day).  Required columns:
          - ``half_hour``  : int (0–12) – half-hour slot index
          - ``beta_ofi``   : float      – OFI price-impact estimate

    Returns
    -------
    pd.Series
        Length-13 Series indexed 0–12 with normalised mean beta per slot.

    Raises
    ------
    ValueError
        If required columns are absent.
    """
    required = {"half_hour", "beta_ofi"}
    missing = required - set(beta_df.columns)
    if missing:
        raise ValueError(f"compute_intraday_beta_pattern: missing columns {missing}")

    mean_by_slot = (
        beta_df.groupby("half_hour")["beta_ofi"]
        .mean()
        .reindex(range(N_SLOTS), fill_value=np.nan)
    )
    grand_mean = mean_by_slot.mean()
    if grand_mean == 0 or not np.isfinite(grand_mean):
        return mean_by_slot.rename("beta_norm")

    return (mean_by_slot / grand_mean).rename("beta_norm")


def compute_intraday_depth_pattern(ad_df: pd.DataFrame) -> pd.Series:
    """Compute the normalised intraday pattern of average quoted depth.

    Groups per-slot average depth values, takes the cross-subsample mean for
    each slot, then normalises to unit mean.

    Parameters
    ----------
    ad_df : pd.DataFrame
        One row per (stock-day, slot) observation.  Required columns:
          - ``half_hour`` : int (0–12)
          - ``AD``        : float – average quoted depth (bid_size + ask_size)/2

    Returns
    -------
    pd.Series
        Length-13 Series indexed 0–12 with normalised mean depth per slot.

    Raises
    ------
    ValueError
        If required columns are absent.
    """
    required = {"half_hour", "AD"}
    missing = required - set(ad_df.columns)
    if missing:
        raise ValueError(f"compute_intraday_depth_pattern: missing columns {missing}")

    mean_by_slot = (
        ad_df.groupby("half_hour")["AD"]
        .mean()
        .reindex(range(N_SLOTS), fill_value=np.nan)
    )
    grand_mean = mean_by_slot.mean()
    if grand_mean == 0 or not np.isfinite(grand_mean):
        return mean_by_slot.rename("ad_norm")

    return (mean_by_slot / grand_mean).rename("ad_norm")


def compute_intraday_variance_patterns(
    grid_df: pd.DataFrame,
) -> tuple[pd.Series, pd.Series]:
    """Compute normalised intraday variance patterns of delta_P and OFI.

    For each (half-hour slot, trading day) cell:
      - Compute ``var[delta_P_k]`` and ``var[OFI_k]`` across the 180 bins.

    Then for each slot, average the cell variances across all days, yielding
    slot-level average variances.  Both series are normalised to unit mean.

    Parameters
    ----------
    grid_df : pd.DataFrame
        Full grid data with columns ``half_hour``, ``date`` (or
        ``bin_start`` from which a date can be derived), ``delta_P_k``,
        ``OFI_k``.

    Returns
    -------
    tuple[pd.Series, pd.Series]
        ``(var_dP_norm, var_OFI_norm)`` – each a length-13 Series indexed
        0–12.

    Raises
    ------
    ValueError
        If required columns are absent.
    """
    required = {"half_hour", "delta_P_k", "OFI_k"}
    missing = required - set(grid_df.columns)
    if missing:
        raise ValueError(
            f"compute_intraday_variance_patterns: missing columns {missing}"
        )

    work = grid_df.copy()

    # Derive a date column if absent
    if "date" not in work.columns:
        if "bin_start" in work.columns:
            work["date"] = pd.to_datetime(work["bin_start"]).dt.date
        else:
            raise ValueError(
                "compute_intraday_variance_patterns: need 'date' or 'bin_start' column"
            )

    # Per (slot, day) variance
    grouped = work.groupby(["half_hour", "date"])
    var_dP_per_cell = grouped["delta_P_k"].var()
    var_ofi_per_cell = grouped["OFI_k"].var()

    # Average across days per slot
    var_dP_by_slot = (
        var_dP_per_cell.groupby(level="half_hour")
        .mean()
        .reindex(range(N_SLOTS), fill_value=np.nan)
    )
    var_ofi_by_slot = (
        var_ofi_per_cell.groupby(level="half_hour")
        .mean()
        .reindex(range(N_SLOTS), fill_value=np.nan)
    )

    def _normalise(s: pd.Series) -> pd.Series:
        m = s.mean()
        if m == 0 or not np.isfinite(m):
            return s
        return s / m

    return _normalise(var_dP_by_slot).rename("var_dP_norm"), \
           _normalise(var_ofi_by_slot).rename("var_OFI_norm")


def compute_predicted_volatility(
    beta_by_slot: pd.Series,
    var_ofi_by_slot: pd.Series,
) -> pd.Series:
    """Compute normalised predicted return volatility: beta^2 * var_OFI.

    The predicted variance of price changes due to OFI is beta_k^2 * Var[OFI_k].
    The result is normalised so that its cross-slot mean equals 1.

    Parameters
    ----------
    beta_by_slot : pd.Series
        Mean (or normalised) OFI beta per slot, length 13.
    var_ofi_by_slot : pd.Series
        Mean (or normalised) OFI variance per slot, length 13.

    Returns
    -------
    pd.Series
        Length-13 Series of normalised predicted volatility per slot.

    Raises
    ------
    ValueError
        If the two series have different lengths.
    """
    if len(beta_by_slot) != len(var_ofi_by_slot):
        raise ValueError(
            "compute_predicted_volatility: beta_by_slot and var_ofi_by_slot "
            "must have the same length"
        )

    pred = beta_by_slot.to_numpy(dtype=float) ** 2 * var_ofi_by_slot.to_numpy(
        dtype=float
    )
    pred_series = pd.Series(pred, index=range(N_SLOTS), name="pred_vol_norm")
    grand_mean = pred_series.mean()
    if grand_mean == 0 or not np.isfinite(grand_mean):
        return pred_series
    return pred_series / grand_mean


def cross_stock_average(pattern_list: list[pd.Series]) -> pd.Series:
    """Compute the cross-stock mean of per-stock intraday patterns.

    Parameters
    ----------
    pattern_list : list[pd.Series]
        Each element is a length-13 Series (one per stock) representing a
        normalised intraday pattern indexed 0–12.

    Returns
    -------
    pd.Series
        Length-13 Series of element-wise means across all stocks.

    Raises
    ------
    ValueError
        If *pattern_list* is empty or any Series has a length other than 13.
    """
    if not pattern_list:
        raise ValueError("cross_stock_average: pattern_list is empty")

    for i, s in enumerate(pattern_list):
        if len(s) != N_SLOTS:
            raise ValueError(
                f"cross_stock_average: series at index {i} has length "
                f"{len(s)}, expected {N_SLOTS}"
            )

    mat = np.vstack([s.reindex(range(N_SLOTS)).to_numpy(dtype=float)
                     for s in pattern_list])
    mean_pattern = np.nanmean(mat, axis=0)
    return pd.Series(mean_pattern, index=range(N_SLOTS), name="cross_stock_avg")


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def plot_beta_depth_patterns(
    beta_norm: pd.Series,
    ad_norm: pd.Series,
    output_path: str,
) -> None:
    """Plot normalised OFI beta and average depth patterns (Figure 5 equivalent).

    Generates a single-panel line chart with two y-axes showing the intraday
    U-shape of OFI beta and the inverse U-shape of quoted depth across the
    13 half-hour slots.

    Parameters
    ----------
    beta_norm : pd.Series
        Normalised OFI beta per slot (mean = 1), length 13.
    ad_norm : pd.Series
        Normalised average depth per slot (mean = 1), length 13.
    output_path : str
        Full file path for the saved PNG figure (directory must exist).

    Raises
    ------
    ValueError
        If the input series do not have length 13.
    """
    if len(beta_norm) != N_SLOTS or len(ad_norm) != N_SLOTS:
        raise ValueError(
            "plot_beta_depth_patterns: both series must have length 13"
        )

    x = np.arange(N_SLOTS)
    beta_vals = beta_norm.to_numpy(dtype=float)
    ad_vals = ad_norm.to_numpy(dtype=float)

    fig, ax1 = plt.subplots(figsize=(10, 5))
    ax2 = ax1.twinx()

    color_beta = "#1f77b4"
    color_ad = "#d62728"

    ax1.plot(x, beta_vals, marker="o", color=color_beta, linewidth=2,
             label="Normalised OFI Beta")
    ax1.set_ylabel("Normalised OFI Beta", color=color_beta, fontsize=11)
    ax1.tick_params(axis="y", labelcolor=color_beta)

    ax2.plot(x, ad_vals, marker="s", linestyle="--", color=color_ad,
             linewidth=2, label="Normalised Avg Depth")
    ax2.set_ylabel("Normalised Avg Depth", color=color_ad, fontsize=11)
    ax2.tick_params(axis="y", labelcolor=color_ad)

    ax1.set_xticks(x)
    ax1.set_xticklabels(SLOT_LABELS, rotation=45, ha="right", fontsize=9)
    ax1.set_xlabel("Half-Hour Slot", fontsize=11)
    ax1.axhline(1.0, color="grey", linestyle=":", linewidth=0.8, alpha=0.6)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right", fontsize=9)

    plt.title("Intraday OFI Beta and Average Depth Patterns", fontsize=13)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_volatility_decomposition(
    var_dP_norm: pd.Series,
    var_OFI_norm: pd.Series,
    pred_norm: pd.Series,
    output_path: str,
) -> None:
    """Plot normalised return variance, OFI variance, and predicted volatility.

    Generates a line chart with three curves for the 13 half-hour slots,
    corresponding to Figure 6 of Cont, Kukanov & Stoikov (2014).

    Parameters
    ----------
    var_dP_norm : pd.Series
        Normalised variance of delta_P_k per slot (mean = 1), length 13.
    var_OFI_norm : pd.Series
        Normalised variance of OFI_k per slot (mean = 1), length 13.
    pred_norm : pd.Series
        Normalised predicted volatility (beta^2 × var_OFI) per slot, length 13.
    output_path : str
        Full file path for the saved PNG figure.

    Raises
    ------
    ValueError
        If any input series does not have length 13.
    """
    for name, s in [("var_dP_norm", var_dP_norm), ("var_OFI_norm", var_OFI_norm),
                    ("pred_norm", pred_norm)]:
        if len(s) != N_SLOTS:
            raise ValueError(
                f"plot_volatility_decomposition: '{name}' must have length 13"
            )

    x = np.arange(N_SLOTS)
    fig, ax = plt.subplots(figsize=(10, 5))

    ax.plot(x, var_dP_norm.to_numpy(dtype=float), marker="o", linewidth=2,
            label="Var[ΔP] (normalised)", color="#1f77b4")
    ax.plot(x, var_OFI_norm.to_numpy(dtype=float), marker="s", linestyle="--",
            linewidth=2, label="Var[OFI] (normalised)", color="#2ca02c")
    ax.plot(x, pred_norm.to_numpy(dtype=float), marker="^", linestyle="-.",
            linewidth=2, label="Predicted Vol β²·Var[OFI] (normalised)",
            color="#d62728")

    ax.axhline(1.0, color="grey", linestyle=":", linewidth=0.8, alpha=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels(SLOT_LABELS, rotation=45, ha="right", fontsize=9)
    ax.set_xlabel("Half-Hour Slot", fontsize=11)
    ax.set_ylabel("Normalised Value", fontsize=11)
    ax.legend(fontsize=9, loc="upper right")
    plt.title("Intraday Volatility Decomposition", fontsize=13)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close(fig)
