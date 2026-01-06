"""
Minimum bounding ellipse fitting module using OpenCV.

This module provides functions to fit a minimum bounding ellipse
to a set of trajectory points using convex hull and ellipse fitting.
"""

import numpy as np
import cv2
from typing import Tuple, Optional

from src.utils.logger import get_logger


def fit_minimum_bounding_ellipse(
    points: np.ndarray
) -> Optional[Tuple[float, float, float, float, float]]:
    """
    Fit a minimum bounding ellipse to trajectory points.

    Uses OpenCV's convex hull and fitEllipse functions to find
    the minimum bounding ellipse for the given points.

    Args:
        points: N×2 array of (x, y) coordinates in micrometers.

    Returns:
        Tuple of (center_x, center_y, major_axis, minor_axis, angle) or None.
        - center_x, center_y: Center of the ellipse (μm)
        - major_axis: Length of major axis (μm)
        - minor_axis: Length of minor axis (μm)
        - angle: Rotation angle of the ellipse (degrees)

        Returns None if fitting fails (e.g., insufficient points).
    """
    logger = get_logger()

    if points is None or len(points) < 5:
        logger.debug(
            f"Insufficient points for ellipse fitting: {len(points) if points is not None else 0}"
        )
        return None

    # Convert to float32 for OpenCV
    points_f32 = np.array(points, dtype=np.float32)

    try:
        # Compute convex hull
        hull = cv2.convexHull(points_f32)

        if len(hull) < 5:
            logger.debug(
                f"Convex hull has insufficient points: {len(hull)}"
            )
            return None

        # Fit ellipse to convex hull points
        ellipse = cv2.fitEllipse(hull)
        center, axes, angle = ellipse

        # axes is (width, height), determine major and minor
        major_axis = max(axes)
        minor_axis = min(axes)

        return (center[0], center[1], major_axis, minor_axis, angle)

    except cv2.error as e:
        logger.warning(f"OpenCV ellipse fitting failed: {e}")
        return None
    except Exception as e:
        logger.warning(f"Ellipse fitting failed: {e}")
        return None


def calculate_ellipse_aspect_ratio(
    major_axis: float,
    minor_axis: float
) -> float:
    """
    Calculate aspect ratio of an ellipse (minor/major).

    Args:
        major_axis: Length of major axis.
        minor_axis: Length of minor axis.

    Returns:
        Aspect ratio (minor/major), or NaN if major_axis is zero.
    """
    if major_axis <= 0:
        return np.nan
    return minor_axis / major_axis
