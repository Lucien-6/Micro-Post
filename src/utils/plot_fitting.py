"""
Fitting results plotting module.

This module provides journal-quality plotting for MSD and MSAD
fitting results with proper annotations and styling.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from typing import Dict, Tuple, Optional
from pathlib import Path

from src.core.curve_fitting import (
    MODEL_DRIFT, MODEL_ACTIVE,
    msd_drift, msd_active, msad_model
)
from src.utils.logger import get_logger


# Plotting configuration
FIGURE_SIZE = (8, 10)  # inches
DPI = 300
FONT_FAMILY = "Arial"
FONT_SIZE_LABEL = 12
FONT_SIZE_TICK = 10
FONT_SIZE_LEGEND = 10
FONT_SIZE_ANNOTATION = 10

# Colors
COLOR_DATA = "#1f77b4"  # Deep blue
COLOR_FIT = "#d62728"  # Red
COLOR_ERROR_BAND = "#aec7e8"  # Light blue
COLOR_FIT_RANGE = "#7f7f7f"  # Gray

# Line styles
LINE_WIDTH_DATA = 1.5
LINE_WIDTH_FIT = 2.0
MARKER_SIZE = 4


def plot_fitting_results(
    output_path: str,
    tau: np.ndarray,
    mean_msd: np.ndarray,
    std_msd: np.ndarray,
    mean_msad: np.ndarray,
    std_msad: np.ndarray,
    model_type: str,
    fit_params: Dict,
    fit_range: Tuple[int, int]
) -> str:
    """
    Plot MSD and MSAD fitting results and save to file.

    Creates a two-panel figure with MSD on top and MSAD on bottom,
    showing data with error bands and fitted curves with annotations.

    Args:
        output_path: Directory path to save the figure.
        tau: Lag time array.
        mean_msd: Mean MSD values.
        std_msd: Standard deviation of MSD.
        mean_msad: Mean MSAD values.
        std_msad: Standard deviation of MSAD.
        model_type: MODEL_DRIFT or MODEL_ACTIVE.
        fit_params: Dictionary with fitting parameters and R² values.
        fit_range: Tuple of (min_tau, max_tau) for fitting.

    Returns:
        Path to the saved figure.
    """
    logger = get_logger()

    # Configure matplotlib
    plt.rcParams["font.family"] = FONT_FAMILY
    plt.rcParams["font.size"] = FONT_SIZE_LABEL
    plt.rcParams["axes.linewidth"] = 1.0
    plt.rcParams["xtick.major.width"] = 1.0
    plt.rcParams["ytick.major.width"] = 1.0

    # Create figure with two subplots
    fig, (ax_msd, ax_msad) = plt.subplots(
        2, 1,
        figsize=FIGURE_SIZE,
        dpi=DPI
    )

    # Filter valid data (non-NaN)
    valid_mask = ~np.isnan(mean_msd) & ~np.isnan(mean_msad)
    tau_valid = tau[valid_mask]
    msd_valid = mean_msd[valid_mask]
    std_msd_valid = std_msd[valid_mask]
    msad_valid = mean_msad[valid_mask]
    std_msad_valid = std_msad[valid_mask]

    # Generate smooth fitting curves
    tau_fit_range = np.linspace(fit_range[0], fit_range[1], 200)

    # === MSD Plot ===
    _plot_msd_panel(
        ax_msd, tau_valid, msd_valid, std_msd_valid,
        tau_fit_range, model_type, fit_params, fit_range
    )

    # === MSAD Plot ===
    _plot_msad_panel(
        ax_msad, tau_valid, msad_valid, std_msad_valid,
        tau_fit_range, fit_params, fit_range
    )

    # Adjust layout
    plt.tight_layout()
    plt.subplots_adjust(hspace=0.25)

    # Save figure
    output_file = Path(output_path) / "Fitting_Results.png"
    fig.savefig(
        output_file,
        dpi=DPI,
        bbox_inches="tight",
        facecolor="white",
        edgecolor="none"
    )
    plt.close(fig)

    logger.info(f"Fitting plot saved to: {output_file}")
    return str(output_file)


def _plot_msd_panel(
    ax: plt.Axes,
    tau: np.ndarray,
    mean_msd: np.ndarray,
    std_msd: np.ndarray,
    tau_fit: np.ndarray,
    model_type: str,
    fit_params: Dict,
    fit_range: Tuple[int, int]
) -> None:
    """
    Plot MSD panel with data, error band, and fitted curve.

    Args:
        ax: Matplotlib axes object.
        tau: Lag time array.
        mean_msd: Mean MSD values.
        std_msd: Standard deviation of MSD.
        tau_fit: Tau array for fitted curve.
        model_type: Fitting model type.
        fit_params: Fitting parameters dictionary.
        fit_range: Fitting range tuple.
    """
    # Plot error band
    ax.fill_between(
        tau, mean_msd - std_msd, mean_msd + std_msd,
        color=COLOR_ERROR_BAND, alpha=0.5, label="Std Dev"
    )

    # Plot data points with line
    ax.plot(
        tau, mean_msd,
        color=COLOR_DATA, linewidth=LINE_WIDTH_DATA,
        marker="o", markersize=MARKER_SIZE,
        label="Mean MSD"
    )

    # Generate and plot fitted curve
    if model_type == MODEL_DRIFT:
        D_T = fit_params["D_T"]
        V = fit_params["V"]
        msd_fit = msd_drift(tau_fit, D_T, V)
        formula = r"$\mathrm{MSD}(t) = 4D_T t + V^2 t^2$"
        params_text = (
            f"$D_T = {D_T:.4f}$ μm²/s\n"
            f"$V = {V:.4f}$ μm/s"
        )
    else:  # MODEL_ACTIVE
        D_eff = fit_params["D_eff"]
        tau_r = fit_params["tau_r"]
        msd_fit = msd_active(tau_fit, D_eff, tau_r)
        formula = r"$\mathrm{MSD}(t) = 4D_{\mathrm{eff}}(t - \tau_r(1-e^{-t/\tau_r}))$"
        params_text = (
            f"$D_{{\\mathrm{{eff}}}} = {D_eff:.4f}$ μm²/s\n"
            f"$\\tau_r = {tau_r:.4f}$ s"
        )

    r_squared = fit_params["MSD_R2"]

    ax.plot(
        tau_fit, msd_fit,
        color=COLOR_FIT, linewidth=LINE_WIDTH_FIT,
        linestyle="--", label="Fitted curve"
    )

    # Mark fitting range
    ax.axvline(
        x=fit_range[0], color=COLOR_FIT_RANGE,
        linestyle=":", linewidth=1.0, alpha=0.7
    )
    ax.axvline(
        x=fit_range[1], color=COLOR_FIT_RANGE,
        linestyle=":", linewidth=1.0, alpha=0.7
    )

    # Add annotations
    annotation_text = (
        f"{formula}\n"
        f"{params_text}\n"
        f"$R^2 = {r_squared:.4f}$\n"
        f"Fit range: {fit_range[0]} - {fit_range[1]} s"
    )

    # Determine annotation position (avoid data)
    _add_annotation(ax, annotation_text, tau, mean_msd)

    # Labels and formatting
    ax.set_xlabel(r"Lag time $\tau$ (s)", fontsize=FONT_SIZE_LABEL)
    ax.set_ylabel(r"MSD (μm²)", fontsize=FONT_SIZE_LABEL)
    ax.tick_params(axis="both", labelsize=FONT_SIZE_TICK)
    ax.legend(loc="lower right", fontsize=FONT_SIZE_LEGEND)
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)


def _plot_msad_panel(
    ax: plt.Axes,
    tau: np.ndarray,
    mean_msad: np.ndarray,
    std_msad: np.ndarray,
    tau_fit: np.ndarray,
    fit_params: Dict,
    fit_range: Tuple[int, int]
) -> None:
    """
    Plot MSAD panel with data, error band, and fitted curve.

    Args:
        ax: Matplotlib axes object.
        tau: Lag time array.
        mean_msad: Mean MSAD values.
        std_msad: Standard deviation of MSAD.
        tau_fit: Tau array for fitted curve.
        fit_params: Fitting parameters dictionary.
        fit_range: Fitting range tuple.
    """
    # Plot error band
    ax.fill_between(
        tau, mean_msad - std_msad, mean_msad + std_msad,
        color=COLOR_ERROR_BAND, alpha=0.5, label="Std Dev"
    )

    # Plot data points with line
    ax.plot(
        tau, mean_msad,
        color=COLOR_DATA, linewidth=LINE_WIDTH_DATA,
        marker="o", markersize=MARKER_SIZE,
        label="Mean MSAD"
    )

    # Generate and plot fitted curve
    D_R = fit_params["D_R"]
    msad_fit = msad_model(tau_fit, D_R)

    formula = r"$\mathrm{MSAD}(t) = 2D_R t$"
    params_text = f"$D_R = {D_R:.4f}$ rad²/s"
    r_squared = fit_params["MSAD_R2"]

    ax.plot(
        tau_fit, msad_fit,
        color=COLOR_FIT, linewidth=LINE_WIDTH_FIT,
        linestyle="--", label="Fitted curve"
    )

    # Mark fitting range
    ax.axvline(
        x=fit_range[0], color=COLOR_FIT_RANGE,
        linestyle=":", linewidth=1.0, alpha=0.7
    )
    ax.axvline(
        x=fit_range[1], color=COLOR_FIT_RANGE,
        linestyle=":", linewidth=1.0, alpha=0.7
    )

    # Add annotations
    annotation_text = (
        f"{formula}\n"
        f"{params_text}\n"
        f"$R^2 = {r_squared:.4f}$\n"
        f"Fit range: {fit_range[0]} - {fit_range[1]} s"
    )

    # Determine annotation position (avoid data)
    _add_annotation(ax, annotation_text, tau, mean_msad)

    # Labels and formatting
    ax.set_xlabel(r"Lag time $\tau$ (s)", fontsize=FONT_SIZE_LABEL)
    ax.set_ylabel(r"MSAD (rad²)", fontsize=FONT_SIZE_LABEL)
    ax.tick_params(axis="both", labelsize=FONT_SIZE_TICK)
    ax.legend(loc="lower right", fontsize=FONT_SIZE_LEGEND)
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)


def _add_annotation(
    ax: plt.Axes,
    text: str,
    tau: np.ndarray,
    data: np.ndarray
) -> None:
    """
    Add annotation text box at optimal position.

    Places the annotation in the upper left or upper right corner,
    choosing the position that is less likely to overlap with data.

    Args:
        ax: Matplotlib axes object.
        text: Annotation text.
        tau: X data for determining position.
        data: Y data for determining position.
    """
    # Determine if data rises steeply at the beginning
    # If so, place annotation on the right side
    if len(data) > 5:
        early_slope = (data[5] - data[0]) / (tau[5] - tau[0] + 1e-10)
        late_slope = (data[-1] - data[-6]) / (tau[-1] - tau[-6] + 1e-10)

        if early_slope > late_slope * 1.5:
            # Data rises more steeply early, put annotation on right
            x_pos, ha = 0.97, "right"
        else:
            # Place on left
            x_pos, ha = 0.03, "left"
    else:
        x_pos, ha = 0.97, "right"

    ax.text(
        x_pos, 0.97, text,
        transform=ax.transAxes,
        fontsize=FONT_SIZE_ANNOTATION,
        verticalalignment="top",
        horizontalalignment=ha,
        bbox=dict(
            boxstyle="round,pad=0.5",
            facecolor="white",
            edgecolor="gray",
            alpha=0.9
        )
    )
