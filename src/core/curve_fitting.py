"""
Curve fitting module for MSD and MSAD analysis.

This module provides fitting functions for two diffusion models:
- Constant Velocity Drift Model
- Active Diffusion Model
"""

import numpy as np
from scipy.optimize import curve_fit
from typing import Tuple, List

from src.utils.logger import get_logger


# Model type constants
MODEL_DRIFT = "Constant Velocity Drift"
MODEL_ACTIVE = "Active Diffusion"


class FittingError(Exception):
    """Custom exception for fitting failures."""

    def __init__(self, message: str, model: str = "", parameter: str = ""):
        """
        Initialize FittingError.

        Args:
            message: Error description.
            model: Model type that failed.
            parameter: Parameter that caused the failure.
        """
        self.message = message
        self.model = model
        self.parameter = parameter
        super().__init__(self.message)


def msd_drift(t: np.ndarray, D_T: float, V: float) -> np.ndarray:
    """
    Constant velocity drift model for MSD.

    MSD(t) = 4 * D_T * t + V² * t²

    Args:
        t: Time array (lag time tau).
        D_T: Translational diffusion coefficient (μm²/s).
        V: Drift velocity (μm/s).

    Returns:
        MSD values array.
    """
    return 4 * D_T * t + V ** 2 * t ** 2


def msd_active(t: np.ndarray, D_eff: float, tau_r: float) -> np.ndarray:
    """
    Active diffusion model for MSD.

    MSD(t) = 4 * D_eff * (t - tau_r * (1 - exp(-t / tau_r)))

    Args:
        t: Time array (lag time tau).
        D_eff: Effective diffusion coefficient (μm²/s).
        tau_r: Direction persistence time (s).

    Returns:
        MSD values array.
    """
    # Handle tau_r near zero to avoid division issues
    if tau_r < 1e-10:
        return 4 * D_eff * t
    return 4 * D_eff * (t - tau_r * (1 - np.exp(-t / tau_r)))


def msad_model(t: np.ndarray, D_R: float) -> np.ndarray:
    """
    Rotational diffusion model for MSAD.

    MSAD(t) = 2 * D_R * t

    Args:
        t: Time array (lag time tau).
        D_R: Rotational diffusion coefficient (rad²/s).

    Returns:
        MSAD values array.
    """
    return 2 * D_R * t


def calculate_r_squared(y_actual: np.ndarray, y_predicted: np.ndarray) -> float:
    """
    Calculate coefficient of determination (R²).

    R² = 1 - SS_res / SS_tot

    Args:
        y_actual: Actual data values.
        y_predicted: Predicted values from model.

    Returns:
        R² value (0 to 1, can be negative for poor fits).
    """
    ss_res = np.sum((y_actual - y_predicted) ** 2)
    ss_tot = np.sum((y_actual - np.mean(y_actual)) ** 2)

    if ss_tot < 1e-15:
        return 0.0

    return 1 - ss_res / ss_tot


def validate_parameters(
    params: List[float],
    param_names: List[str]
) -> None:
    """
    Validate that all fitted parameters are non-negative.

    Args:
        params: List of parameter values.
        param_names: List of parameter names.

    Raises:
        FittingError: If any parameter is negative.
    """
    for value, name in zip(params, param_names):
        if value < 0:
            raise FittingError(
                f"Fitting resulted in negative {name}: {value:.6f}. "
                f"This is physically invalid.",
                parameter=name
            )


def fit_msd_drift(
    tau: np.ndarray,
    msd: np.ndarray,
    max_fit_tau: int
) -> Tuple[float, float, float]:
    """
    Fit MSD data using constant velocity drift model.

    Args:
        tau: Lag time array.
        msd: Mean MSD values array.
        max_fit_tau: Maximum lag time for fitting.

    Returns:
        Tuple of (D_T, V, R²).

    Raises:
        FittingError: If fitting fails or parameters are invalid.
    """
    logger = get_logger()

    # Select fitting range (excluding tau=0)
    mask = (tau >= 1) & (tau <= max_fit_tau)
    tau_fit = tau[mask]
    msd_fit = msd[mask]

    if len(tau_fit) < 2:
        raise FittingError(
            f"Insufficient data points for MSD fitting. "
            f"Need at least 2 points, got {len(tau_fit)}.",
            model=MODEL_DRIFT
        )

    try:
        # Initial guess: D_T=1, V=1
        popt, _ = curve_fit(
            msd_drift,
            tau_fit,
            msd_fit,
            p0=[1.0, 1.0],
            bounds=([0, 0], [np.inf, np.inf]),
            maxfev=5000
        )
        D_T, V = popt

        # Validate parameters
        validate_parameters([D_T, V], ["D_T", "V"])

        # Calculate R²
        msd_predicted = msd_drift(tau_fit, D_T, V)
        r_squared = calculate_r_squared(msd_fit, msd_predicted)

        logger.info(
            f"MSD drift fit: D_T={D_T:.6f} μm²/s, "
            f"V={V:.6f} μm/s, R²={r_squared:.4f}"
        )

        return D_T, V, r_squared

    except RuntimeError as e:
        raise FittingError(
            f"MSD fitting failed to converge: {str(e)}",
            model=MODEL_DRIFT
        )
    except ValueError as e:
        raise FittingError(
            f"MSD fitting error: {str(e)}",
            model=MODEL_DRIFT
        )


def fit_msd_active(
    tau: np.ndarray,
    msd: np.ndarray,
    max_fit_tau: int
) -> Tuple[float, float, float]:
    """
    Fit MSD data using active diffusion model.

    Args:
        tau: Lag time array.
        msd: Mean MSD values array.
        max_fit_tau: Maximum lag time for fitting.

    Returns:
        Tuple of (D_eff, tau_r, R²).

    Raises:
        FittingError: If fitting fails or parameters are invalid.
    """
    logger = get_logger()

    # Select fitting range (excluding tau=0)
    mask = (tau >= 1) & (tau <= max_fit_tau)
    tau_fit = tau[mask]
    msd_fit = msd[mask]

    if len(tau_fit) < 2:
        raise FittingError(
            f"Insufficient data points for MSD fitting. "
            f"Need at least 2 points, got {len(tau_fit)}.",
            model=MODEL_ACTIVE
        )

    try:
        # Initial guess: D_eff=1, tau_r=1
        popt, _ = curve_fit(
            msd_active,
            tau_fit,
            msd_fit,
            p0=[1.0, 1.0],
            bounds=([0, 1e-10], [np.inf, np.inf]),
            maxfev=5000
        )
        D_eff, tau_r = popt

        # Validate parameters
        validate_parameters([D_eff, tau_r], ["D_eff", "τ_r"])

        # Calculate R²
        msd_predicted = msd_active(tau_fit, D_eff, tau_r)
        r_squared = calculate_r_squared(msd_fit, msd_predicted)

        logger.info(
            f"MSD active fit: D_eff={D_eff:.6f} μm²/s, "
            f"τ_r={tau_r:.6f} s, R²={r_squared:.4f}"
        )

        return D_eff, tau_r, r_squared

    except RuntimeError as e:
        raise FittingError(
            f"MSD fitting failed to converge: {str(e)}",
            model=MODEL_ACTIVE
        )
    except ValueError as e:
        raise FittingError(
            f"MSD fitting error: {str(e)}",
            model=MODEL_ACTIVE
        )


def fit_msad(
    tau: np.ndarray,
    msad: np.ndarray,
    max_fit_tau: int
) -> Tuple[float, float]:
    """
    Fit MSAD data using rotational diffusion model.

    Args:
        tau: Lag time array.
        msad: Mean MSAD values array.
        max_fit_tau: Maximum lag time for fitting.

    Returns:
        Tuple of (D_R, R²).

    Raises:
        FittingError: If fitting fails or parameters are invalid.
    """
    logger = get_logger()

    # Select fitting range (excluding tau=0)
    mask = (tau >= 1) & (tau <= max_fit_tau)
    tau_fit = tau[mask]
    msad_fit = msad[mask]

    if len(tau_fit) < 1:
        raise FittingError(
            f"Insufficient data points for MSAD fitting. "
            f"Need at least 1 point, got {len(tau_fit)}.",
            model="MSAD"
        )

    try:
        # Initial guess: D_R=0.1
        popt, _ = curve_fit(
            msad_model,
            tau_fit,
            msad_fit,
            p0=[0.1],
            bounds=([0], [np.inf]),
            maxfev=5000
        )
        D_R = popt[0]

        # Validate parameters
        validate_parameters([D_R], ["D_R"])

        # Calculate R²
        msad_predicted = msad_model(tau_fit, D_R)
        r_squared = calculate_r_squared(msad_fit, msad_predicted)

        logger.info(
            f"MSAD fit: D_R={D_R:.6f} rad²/s, R²={r_squared:.4f}"
        )

        return D_R, r_squared

    except RuntimeError as e:
        raise FittingError(
            f"MSAD fitting failed to converge: {str(e)}",
            model="MSAD"
        )
    except ValueError as e:
        raise FittingError(
            f"MSAD fitting error: {str(e)}",
            model="MSAD"
        )
