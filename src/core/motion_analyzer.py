"""
Motion analysis module.

This module provides comprehensive motion analysis for bacterial
trajectories, including displacement, velocity, MSD, MSAD,
ellipse fitting, oscillation index calculations, and curve fitting.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from sklearn.decomposition import PCA

from src.core.data_manager import DataManager
from src.core.curve_fitting import (
    MODEL_DRIFT, MODEL_ACTIVE, FittingError,
    fit_msd_drift, fit_msd_active, fit_msad
)
from src.utils.ellipse_fitting import (
    fit_minimum_bounding_ellipse,
    calculate_ellipse_aspect_ratio
)
from src.utils.logger import get_logger


class MotionAnalyzer:
    """
    Analyzes motion characteristics of bacterial trajectories.

    Performs calculations for displacement, velocity, angular displacement,
    MSD, MSAD, trajectory ellipse fitting, and oscillation index.
    """

    def __init__(self, data_manager: DataManager):
        """
        Initialize MotionAnalyzer with DataManager.

        Args:
            data_manager: DataManager instance containing trajectory data.
        """
        self.logger = get_logger()
        self.data_manager = data_manager
        self.summary_data: Dict[str, Any] = {}

        # Fitting related attributes
        self.fitting_model: str = MODEL_DRIFT
        self.fit_results: Dict[str, float] = {}
        self.fit_range: Tuple[int, int] = (0, 0)
        self.max_tau_global: int = 0

        # Store MSD/MSAD data for plotting
        self._tau_array: Optional[np.ndarray] = None
        self._mean_msd: Optional[np.ndarray] = None
        self._std_msd: Optional[np.ndarray] = None
        self._mean_msad: Optional[np.ndarray] = None
        self._std_msad: Optional[np.ndarray] = None

    def set_fitting_model(self, model_type: str) -> None:
        """
        Set the fitting model type.

        Args:
            model_type: MODEL_DRIFT or MODEL_ACTIVE.
        """
        if model_type not in [MODEL_DRIFT, MODEL_ACTIVE]:
            raise ValueError(f"Invalid fitting model: {model_type}")
        self.fitting_model = model_type
        self.logger.info(f"Fitting model set to: {model_type}")

    def analyze_single_object(self, object_id: str) -> pd.DataFrame:
        """
        Perform complete motion analysis for a single object.

        Args:
            object_id: Object identifier (e.g., "Object_1").

        Returns:
            DataFrame with original data plus analysis columns.
        """
        df = self.data_manager.get_object_data(object_id)
        if df is None or len(df) < 2:
            self.logger.warning(
                f"Insufficient data for {object_id}"
            )
            return df

        df = df.copy()

        # Extract base data
        time = df["time (s)"].values
        x = df["center_x (μm)"].values
        y = df["center_y (μm)"].values
        major_axis = df["major axis length (μm)"].values
        minor_axis = df["minor axis length (μm)"].values
        posture_angle = df["posture angle (°)"].values

        n_points = len(time)
        max_tau = n_points - 1

        # 1. Aspect ratio (minor/major) for each time point
        df["aspect_ratio"] = minor_axis / major_axis

        # 2. Displacement relative to start point
        x0, y0 = x[0], y[0]
        df["dx (μm)"] = x - x0
        df["dy (μm)"] = y - y0
        df["displacement (μm)"] = np.sqrt(
            (x - x0) ** 2 + (y - y0) ** 2
        )

        # 3. Angular displacement (handle 180° wrapping)
        angular_disp = self._calculate_angular_displacement(posture_angle)
        df["angular_displacement (rad)"] = angular_disp

        # 4. Instantaneous velocity
        vx, vy, speed = self._calculate_velocity(time, x, y)
        df["vx (μm/s)"] = vx
        df["vy (μm/s)"] = vy
        df["speed (μm/s)"] = speed

        # 5. Lag-time dependent quantities (stored in separate columns)
        positions = np.column_stack([x, y])
        lag_results = self._calculate_lag_time_quantities(
            time, positions, posture_angle, max_tau
        )

        # Add lag-time columns (tau from 0 to max_tau)
        df["tau (s)"] = np.nan
        df["mean_vx (μm/s)"] = np.nan
        df["mean_vy (μm/s)"] = np.nan
        df["mean_speed (μm/s)"] = np.nan
        df["mean_angular_displacement (rad)"] = np.nan
        df["MSD (μm²)"] = np.nan
        df["MSAD (rad²)"] = np.nan

        # Fill lag-time data (one tau per row)
        for i, tau in enumerate(range(max_tau + 1)):
            if i < len(df):
                df.loc[df.index[i], "tau (s)"] = tau
                df.loc[df.index[i], "mean_vx (μm/s)"] = lag_results["mean_vx"][i]
                df.loc[df.index[i], "mean_vy (μm/s)"] = lag_results["mean_vy"][i]
                df.loc[df.index[i], "mean_speed (μm/s)"] = lag_results["mean_speed"][i]
                df.loc[df.index[i], "mean_angular_displacement (rad)"] = (
                    lag_results["mean_angular_disp"][i]
                )
                df.loc[df.index[i], "MSD (μm²)"] = lag_results["msd"][i]
                df.loc[df.index[i], "MSAD (rad²)"] = lag_results["msad"][i]

        # 6. Maximum displacement
        max_dx = np.max(np.abs(df["dx (μm)"].values))
        max_dy = np.max(np.abs(df["dy (μm)"].values))

        # 7. Trajectory ellipse fitting
        ellipse_result = fit_minimum_bounding_ellipse(positions)
        if ellipse_result:
            ellipse_major = ellipse_result[2]
            ellipse_minor = ellipse_result[3]
            ellipse_ar = calculate_ellipse_aspect_ratio(
                ellipse_major, ellipse_minor
            )
        else:
            ellipse_major = np.nan
            ellipse_minor = np.nan
            ellipse_ar = np.nan

        # 8. Oscillation index
        oscillation_idx = self._calculate_oscillation_index(positions)

        # Add trajectory-level metrics as additional columns
        # (stored in first row, rest are NaN)
        df["max_dx (μm)"] = np.nan
        df["max_dy (μm)"] = np.nan
        df["ellipse_major (μm)"] = np.nan
        df["ellipse_minor (μm)"] = np.nan
        df["ellipse_aspect_ratio"] = np.nan
        df["oscillation_index"] = np.nan

        df.loc[df.index[0], "max_dx (μm)"] = max_dx
        df.loc[df.index[0], "max_dy (μm)"] = max_dy
        df.loc[df.index[0], "ellipse_major (μm)"] = ellipse_major
        df.loc[df.index[0], "ellipse_minor (μm)"] = ellipse_minor
        df.loc[df.index[0], "ellipse_aspect_ratio"] = ellipse_ar
        df.loc[df.index[0], "oscillation_index"] = oscillation_idx

        return df

    def _calculate_angular_displacement(
        self,
        angles: np.ndarray
    ) -> np.ndarray:
        """
        Calculate cumulative angular displacement with 180° wrapping.

        For 180° symmetric objects (bacteria), angular changes >90°
        are adjusted to account for head-tail ambiguity.

        Args:
            angles: Array of posture angles in degrees (0-180°).

        Returns:
            Array of cumulative angular displacement in radians.
        """
        n = len(angles)
        angular_disp = np.zeros(n)

        cumulative = 0.0
        for i in range(1, n):
            delta = angles[i] - angles[i - 1]

            # Handle 180° wrapping
            if delta > 90:
                delta -= 180
            elif delta < -90:
                delta += 180

            cumulative += delta
            angular_disp[i] = cumulative

        # Convert to radians
        return np.deg2rad(angular_disp)

    def _calculate_velocity(
        self,
        time: np.ndarray,
        x: np.ndarray,
        y: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Calculate instantaneous velocity components.

        Args:
            time: Time array in seconds.
            x: X position array in μm.
            y: Y position array in μm.

        Returns:
            Tuple of (vx, vy, speed) arrays.
        """
        n = len(time)
        vx = np.zeros(n)
        vy = np.zeros(n)
        speed = np.zeros(n)

        for i in range(1, n):
            dt = time[i] - time[i - 1]
            if dt > 0:
                vx[i] = (x[i] - x[i - 1]) / dt
                vy[i] = (y[i] - y[i - 1]) / dt
                speed[i] = np.sqrt(vx[i] ** 2 + vy[i] ** 2)

        return vx, vy, speed

    def _calculate_lag_time_quantities(
        self,
        time: np.ndarray,
        positions: np.ndarray,
        angles: np.ndarray,
        max_tau: int
    ) -> Dict[str, np.ndarray]:
        """
        Calculate lag-time dependent quantities.

        Args:
            time: Time array.
            positions: N×2 position array.
            angles: Posture angle array in degrees.
            max_tau: Maximum lag time.

        Returns:
            Dictionary with arrays for each quantity.
        """
        n = len(time)

        # Initialize result arrays
        msd = np.zeros(max_tau + 1)
        msad = np.zeros(max_tau + 1)
        mean_vx = np.zeros(max_tau + 1)
        mean_vy = np.zeros(max_tau + 1)
        mean_speed = np.zeros(max_tau + 1)
        mean_angular_disp = np.zeros(max_tau + 1)

        # tau = 0: all values are 0
        # Already initialized to 0

        # Calculate cumulative angular displacement for MSAD
        angular_disp_rad = self._calculate_angular_displacement(angles)

        for tau in range(1, max_tau + 1):
            # Number of valid pairs
            n_pairs = n - tau

            if n_pairs <= 0:
                msd[tau] = np.nan
                msad[tau] = np.nan
                mean_vx[tau] = np.nan
                mean_vy[tau] = np.nan
                mean_speed[tau] = np.nan
                mean_angular_disp[tau] = np.nan
                continue

            # Position displacements
            dx = positions[tau:, 0] - positions[:-tau, 0]
            dy = positions[tau:, 1] - positions[:-tau, 1]
            squared_disp = dx ** 2 + dy ** 2

            # Angular displacements (in radians)
            d_angle = angular_disp_rad[tau:] - angular_disp_rad[:-tau]
            squared_angle_disp = d_angle ** 2

            # MSD and MSAD
            msd[tau] = np.mean(squared_disp)
            msad[tau] = np.mean(squared_angle_disp)

            # Mean velocities for this lag time
            dt = tau  # Assuming 1 second intervals
            mean_vx[tau] = np.mean(dx / dt)
            mean_vy[tau] = np.mean(dy / dt)
            mean_speed[tau] = np.mean(np.sqrt(dx ** 2 + dy ** 2) / dt)
            mean_angular_disp[tau] = np.mean(np.abs(d_angle))

        return {
            "msd": msd,
            "msad": msad,
            "mean_vx": mean_vx,
            "mean_vy": mean_vy,
            "mean_speed": mean_speed,
            "mean_angular_disp": mean_angular_disp
        }

    def _calculate_oscillation_index(
        self,
        positions: np.ndarray
    ) -> float:
        """
        Calculate oscillation index using PCA.

        The oscillation index is the mean squared perpendicular
        distance from trajectory points to the principal axis.

        Args:
            positions: N×2 position array.

        Returns:
            Oscillation index value.
        """
        if len(positions) < 3:
            return np.nan

        try:
            # Center the positions
            centroid = np.mean(positions, axis=0)
            centered = positions - centroid

            # PCA to find principal axis
            pca = PCA(n_components=2)
            pca.fit(centered)

            # Principal axis direction (first component)
            principal_axis = pca.components_[0]

            # Calculate perpendicular distances
            # For each point, project onto principal axis
            # Perpendicular distance = |centered - projection|
            projections = np.outer(
                np.dot(centered, principal_axis),
                principal_axis
            )
            perpendicular = centered - projections
            squared_distances = np.sum(perpendicular ** 2, axis=1)

            # Oscillation index = mean squared distance
            return np.mean(squared_distances)

        except Exception as e:
            self.logger.warning(
                f"Oscillation index calculation failed: {e}"
            )
            return np.nan

    def analyze_all_objects(self):
        """Perform analysis on all objects in the dataset."""
        object_ids = self.data_manager.get_all_object_ids()
        total = len(object_ids)

        self.logger.info(f"Starting analysis of {total} objects")

        for i, obj_id in enumerate(object_ids):
            analyzed_df = self.analyze_single_object(obj_id)
            self.data_manager.update_object_data(obj_id, analyzed_df)

            if (i + 1) % 10 == 0 or (i + 1) == total:
                self.logger.info(
                    f"Analyzed {i + 1}/{total} objects"
                )

        self.logger.info("All objects analyzed")

    def compute_summary_statistics(self) -> pd.DataFrame:
        """
        Compute summary statistics across all objects and perform fitting.

        Returns:
            DataFrame with summary statistics including means,
            standard deviations, counts for each lag time, and fitting results.

        Raises:
            FittingError: If curve fitting fails.
        """
        object_ids = self.data_manager.get_all_object_ids()
        n_objects = len(object_ids)

        if n_objects == 0:
            return pd.DataFrame()

        # Collect data from all objects
        all_areas = []
        all_aspect_ratios = []
        all_max_dx = []
        all_max_dy = []
        all_ellipse_major = []
        all_ellipse_minor = []
        all_ellipse_ar = []
        all_oscillation_idx = []

        # Find maximum tau across all objects
        self.max_tau_global = 0
        for obj_id in object_ids:
            df = self.data_manager.get_object_data(obj_id)
            if df is not None:
                valid_tau = df["tau (s)"].dropna()
                if len(valid_tau) > 0:
                    self.max_tau_global = max(
                        self.max_tau_global, int(valid_tau.max())
                    )

        # Initialize lag-time data storage
        lag_data = {tau: {
            "mean_vx": [], "mean_vy": [], "mean_speed": [],
            "mean_angular_disp": [], "msd": [], "msad": []
        } for tau in range(self.max_tau_global + 1)}

        # Collect statistics from each object
        for obj_id in object_ids:
            df = self.data_manager.get_object_data(obj_id)
            if df is None:
                continue

            # Time-averaged values (mean across all time points)
            all_areas.append(df["area (μm²)"].mean())
            all_aspect_ratios.append(df["aspect_ratio"].mean())

            # Trajectory-level values (from first row)
            max_dx = df["max_dx (μm)"].iloc[0]
            max_dy = df["max_dy (μm)"].iloc[0]
            ellipse_major = df["ellipse_major (μm)"].iloc[0]
            ellipse_minor = df["ellipse_minor (μm)"].iloc[0]
            ellipse_ar = df["ellipse_aspect_ratio"].iloc[0]
            osc_idx = df["oscillation_index"].iloc[0]

            if not np.isnan(max_dx):
                all_max_dx.append(max_dx)
            if not np.isnan(max_dy):
                all_max_dy.append(max_dy)
            if not np.isnan(ellipse_major):
                all_ellipse_major.append(ellipse_major)
            if not np.isnan(ellipse_minor):
                all_ellipse_minor.append(ellipse_minor)
            if not np.isnan(ellipse_ar):
                all_ellipse_ar.append(ellipse_ar)
            if not np.isnan(osc_idx):
                all_oscillation_idx.append(osc_idx)

            # Lag-time dependent values
            for i, row in df.iterrows():
                tau = row["tau (s)"]
                if pd.isna(tau):
                    continue
                tau = int(tau)
                if tau > self.max_tau_global:
                    continue

                for key in ["mean_vx", "mean_vy", "mean_speed",
                            "mean_angular_disp", "msd", "msad"]:
                    col_map = {
                        "mean_vx": "mean_vx (μm/s)",
                        "mean_vy": "mean_vy (μm/s)",
                        "mean_speed": "mean_speed (μm/s)",
                        "mean_angular_disp": "mean_angular_displacement (rad)",
                        "msd": "MSD (μm²)",
                        "msad": "MSAD (rad²)"
                    }
                    val = row[col_map[key]]
                    if not pd.isna(val):
                        lag_data[tau][key].append(val)

        # Compute Mean MSD and MSAD arrays for fitting
        mean_msd_list = []
        std_msd_list = []
        mean_msad_list = []
        std_msad_list = []
        tau_list = []

        for tau in range(self.max_tau_global + 1):
            ld = lag_data[tau]
            tau_list.append(tau)

            if len(ld["msd"]) > 0:
                mean_msd_list.append(np.mean(ld["msd"]))
                std_msd_list.append(np.std(ld["msd"]))
            else:
                mean_msd_list.append(np.nan)
                std_msd_list.append(np.nan)

            if len(ld["msad"]) > 0:
                mean_msad_list.append(np.mean(ld["msad"]))
                std_msad_list.append(np.std(ld["msad"]))
            else:
                mean_msad_list.append(np.nan)
                std_msad_list.append(np.nan)

        # Convert to numpy arrays
        self._tau_array = np.array(tau_list)
        self._mean_msd = np.array(mean_msd_list)
        self._std_msd = np.array(std_msd_list)
        self._mean_msad = np.array(mean_msad_list)
        self._std_msad = np.array(std_msad_list)

        # Perform curve fitting
        max_fit_tau = self.max_tau_global // 2
        self.fit_range = (0, max_fit_tau)

        self.logger.info(
            f"Performing {self.fitting_model} fitting "
            f"(range: 0-{max_fit_tau} s)"
        )

        # Fit MSD
        if self.fitting_model == MODEL_DRIFT:
            D_T, V, msd_r2 = fit_msd_drift(
                self._tau_array, self._mean_msd, max_fit_tau
            )
            self.fit_results["D_T"] = D_T
            self.fit_results["V"] = V
        else:  # MODEL_ACTIVE
            D_eff, tau_r, msd_r2 = fit_msd_active(
                self._tau_array, self._mean_msd, max_fit_tau
            )
            self.fit_results["D_eff"] = D_eff
            self.fit_results["tau_r"] = tau_r

        self.fit_results["MSD_R2"] = msd_r2

        # Fit MSAD (same for both models)
        D_R, msad_r2 = fit_msad(
            self._tau_array, self._mean_msad, max_fit_tau
        )
        self.fit_results["D_R"] = D_R
        self.fit_results["MSAD_R2"] = msad_r2

        # Update data manager with fitting info
        self.data_manager.set_fitting_info(
            self.fitting_model,
            f"0 - {max_fit_tau}"
        )

        # Global statistics (same for all tau, stored in first row)
        mean_area = np.mean(all_areas) if all_areas else np.nan
        std_area = np.std(all_areas) if all_areas else np.nan
        mean_ar = np.mean(all_aspect_ratios) if all_aspect_ratios else np.nan
        std_ar = np.std(all_aspect_ratios) if all_aspect_ratios else np.nan
        mean_max_dx = np.mean(all_max_dx) if all_max_dx else np.nan
        std_max_dx = np.std(all_max_dx) if all_max_dx else np.nan
        mean_max_dy = np.mean(all_max_dy) if all_max_dy else np.nan
        std_max_dy = np.std(all_max_dy) if all_max_dy else np.nan
        mean_ell_major = np.mean(all_ellipse_major) if all_ellipse_major else np.nan
        std_ell_major = np.std(all_ellipse_major) if all_ellipse_major else np.nan
        mean_ell_minor = np.mean(all_ellipse_minor) if all_ellipse_minor else np.nan
        std_ell_minor = np.std(all_ellipse_minor) if all_ellipse_minor else np.nan
        mean_ell_ar = np.mean(all_ellipse_ar) if all_ellipse_ar else np.nan
        std_ell_ar = np.std(all_ellipse_ar) if all_ellipse_ar else np.nan
        mean_osc = np.mean(all_oscillation_idx) if all_oscillation_idx else np.nan
        std_osc = np.std(all_oscillation_idx) if all_oscillation_idx else np.nan

        # Build summary DataFrame with column-based storage
        # Order: Global stats -> Fitting results -> tau/Count -> Lag-time stats
        summary_data = {
            "Mean Area (μm²)": [],
            "Std Area (μm²)": [],
            "Mean Aspect Ratio": [],
            "Std Aspect Ratio": [],
            "Mean Max dx (μm)": [],
            "Std Max dx (μm)": [],
            "Mean Max dy (μm)": [],
            "Std Max dy (μm)": [],
            "Mean Ellipse Major (μm)": [],
            "Std Ellipse Major (μm)": [],
            "Mean Ellipse Minor (μm)": [],
            "Std Ellipse Minor (μm)": [],
            "Mean Ellipse Aspect Ratio": [],
            "Std Ellipse Aspect Ratio": [],
            "Mean Oscillation Index": [],
            "Std Oscillation Index": [],
        }

        # Add fitting result columns based on model
        if self.fitting_model == MODEL_DRIFT:
            summary_data["D_T (μm²/s)"] = []
            summary_data["V (μm/s)"] = []
            summary_data["MSD R²"] = []
            summary_data["D_R (rad²/s)"] = []
            summary_data["MSAD R²"] = []
        else:  # MODEL_ACTIVE
            summary_data["D_eff (μm²/s)"] = []
            summary_data["τ_r (s)"] = []
            summary_data["MSD R²"] = []
            summary_data["D_R (rad²/s)"] = []
            summary_data["MSAD R²"] = []

        # Continue with tau and lag-time columns
        summary_data.update({
            "tau (s)": [],
            "Count": [],
            "Mean vx (μm/s)": [],
            "Std vx (μm/s)": [],
            "Mean vy (μm/s)": [],
            "Std vy (μm/s)": [],
            "Mean Speed (μm/s)": [],
            "Std Speed (μm/s)": [],
            "Mean Angular Disp (rad)": [],
            "Std Angular Disp (rad)": [],
            "Mean MSD (μm²)": [],
            "Std MSD (μm²)": [],
            "Mean MSAD (rad²)": [],
            "Std MSAD (rad²)": []
        })

        # Build rows for each tau
        for tau in range(self.max_tau_global + 1):
            ld = lag_data[tau]
            n_count = len(ld["msd"])

            # Global stats only in first row
            if tau == 0:
                summary_data["Mean Area (μm²)"].append(mean_area)
                summary_data["Std Area (μm²)"].append(std_area)
                summary_data["Mean Aspect Ratio"].append(mean_ar)
                summary_data["Std Aspect Ratio"].append(std_ar)
                summary_data["Mean Max dx (μm)"].append(mean_max_dx)
                summary_data["Std Max dx (μm)"].append(std_max_dx)
                summary_data["Mean Max dy (μm)"].append(mean_max_dy)
                summary_data["Std Max dy (μm)"].append(std_max_dy)
                summary_data["Mean Ellipse Major (μm)"].append(mean_ell_major)
                summary_data["Std Ellipse Major (μm)"].append(std_ell_major)
                summary_data["Mean Ellipse Minor (μm)"].append(mean_ell_minor)
                summary_data["Std Ellipse Minor (μm)"].append(std_ell_minor)
                summary_data["Mean Ellipse Aspect Ratio"].append(mean_ell_ar)
                summary_data["Std Ellipse Aspect Ratio"].append(std_ell_ar)
                summary_data["Mean Oscillation Index"].append(mean_osc)
                summary_data["Std Oscillation Index"].append(std_osc)

                # Fitting results only in first row
                if self.fitting_model == MODEL_DRIFT:
                    summary_data["D_T (μm²/s)"].append(self.fit_results["D_T"])
                    summary_data["V (μm/s)"].append(self.fit_results["V"])
                else:
                    summary_data["D_eff (μm²/s)"].append(self.fit_results["D_eff"])
                    summary_data["τ_r (s)"].append(self.fit_results["tau_r"])
                summary_data["MSD R²"].append(self.fit_results["MSD_R2"])
                summary_data["D_R (rad²/s)"].append(self.fit_results["D_R"])
                summary_data["MSAD R²"].append(self.fit_results["MSAD_R2"])
            else:
                summary_data["Mean Area (μm²)"].append(np.nan)
                summary_data["Std Area (μm²)"].append(np.nan)
                summary_data["Mean Aspect Ratio"].append(np.nan)
                summary_data["Std Aspect Ratio"].append(np.nan)
                summary_data["Mean Max dx (μm)"].append(np.nan)
                summary_data["Std Max dx (μm)"].append(np.nan)
                summary_data["Mean Max dy (μm)"].append(np.nan)
                summary_data["Std Max dy (μm)"].append(np.nan)
                summary_data["Mean Ellipse Major (μm)"].append(np.nan)
                summary_data["Std Ellipse Major (μm)"].append(np.nan)
                summary_data["Mean Ellipse Minor (μm)"].append(np.nan)
                summary_data["Std Ellipse Minor (μm)"].append(np.nan)
                summary_data["Mean Ellipse Aspect Ratio"].append(np.nan)
                summary_data["Std Ellipse Aspect Ratio"].append(np.nan)
                summary_data["Mean Oscillation Index"].append(np.nan)
                summary_data["Std Oscillation Index"].append(np.nan)

                # NaN for fitting results
                if self.fitting_model == MODEL_DRIFT:
                    summary_data["D_T (μm²/s)"].append(np.nan)
                    summary_data["V (μm/s)"].append(np.nan)
                else:
                    summary_data["D_eff (μm²/s)"].append(np.nan)
                    summary_data["τ_r (s)"].append(np.nan)
                summary_data["MSD R²"].append(np.nan)
                summary_data["D_R (rad²/s)"].append(np.nan)
                summary_data["MSAD R²"].append(np.nan)

            # tau and Count columns
            summary_data["tau (s)"].append(tau)
            summary_data["Count"].append(n_count)

            # Lag-time dependent stats
            if n_count > 0:
                summary_data["Mean vx (μm/s)"].append(np.mean(ld["mean_vx"]))
                summary_data["Std vx (μm/s)"].append(np.std(ld["mean_vx"]))
                summary_data["Mean vy (μm/s)"].append(np.mean(ld["mean_vy"]))
                summary_data["Std vy (μm/s)"].append(np.std(ld["mean_vy"]))
                summary_data["Mean Speed (μm/s)"].append(np.mean(ld["mean_speed"]))
                summary_data["Std Speed (μm/s)"].append(np.std(ld["mean_speed"]))
                summary_data["Mean Angular Disp (rad)"].append(
                    np.mean(ld["mean_angular_disp"])
                )
                summary_data["Std Angular Disp (rad)"].append(
                    np.std(ld["mean_angular_disp"])
                )
                summary_data["Mean MSD (μm²)"].append(np.mean(ld["msd"]))
                summary_data["Std MSD (μm²)"].append(np.std(ld["msd"]))
                summary_data["Mean MSAD (rad²)"].append(np.mean(ld["msad"]))
                summary_data["Std MSAD (rad²)"].append(np.std(ld["msad"]))
            else:
                summary_data["Mean vx (μm/s)"].append(np.nan)
                summary_data["Std vx (μm/s)"].append(np.nan)
                summary_data["Mean vy (μm/s)"].append(np.nan)
                summary_data["Std vy (μm/s)"].append(np.nan)
                summary_data["Mean Speed (μm/s)"].append(np.nan)
                summary_data["Std Speed (μm/s)"].append(np.nan)
                summary_data["Mean Angular Disp (rad)"].append(np.nan)
                summary_data["Std Angular Disp (rad)"].append(np.nan)
                summary_data["Mean MSD (μm²)"].append(np.nan)
                summary_data["Std MSD (μm²)"].append(np.nan)
                summary_data["Mean MSAD (rad²)"].append(np.nan)
                summary_data["Std MSAD (rad²)"].append(np.nan)

        return pd.DataFrame(summary_data)

    def get_fitting_info(self) -> Dict[str, Any]:
        """
        Get fitting information for plotting.

        Returns:
            Dictionary containing tau array, MSD/MSAD data,
            fitting model, parameters, and range.
        """
        return {
            "tau": self._tau_array,
            "mean_msd": self._mean_msd,
            "std_msd": self._std_msd,
            "mean_msad": self._mean_msad,
            "std_msad": self._std_msad,
            "model_type": self.fitting_model,
            "fit_params": self.fit_results,
            "fit_range": self.fit_range
        }

    def save_analysis_results(self):
        """Save all analysis results to the Excel file."""
        summary_df = self.compute_summary_statistics()
        self.data_manager.save_analysis_results(summary_df)
        self.logger.info("Analysis results saved successfully")
