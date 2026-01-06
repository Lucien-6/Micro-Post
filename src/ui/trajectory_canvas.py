"""
Trajectory visualization canvas module.

This module provides an interactive Matplotlib canvas for visualizing
and interacting with bacterial trajectories using PyQt6.
"""

import numpy as np
from typing import Dict, List, Set, Optional

from PyQt6.QtWidgets import QWidget, QVBoxLayout
from PyQt6.QtCore import pyqtSignal

import matplotlib
matplotlib.use('QtAgg')
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure

from src.core.data_manager import DataManager
from src.utils.logger import get_logger
from src.ui.theme import Theme


class TrajectoryCanvas(QWidget):
    """
    Interactive canvas for trajectory visualization.

    Provides Matplotlib-based trajectory plotting with mouse interaction:
    - Hover: Show trajectory ID within 5px
    - Left click: Hide clicked trajectory
    - Right click: Hide all other trajectories
    - Double click empty area: Show all trajectories
    - Scroll wheel: Zoom in/out centered on mouse position
    - Middle click: Reset zoom to original view

    Signals:
        trajectory_clicked: Emitted when a trajectory is clicked.
    """

    trajectory_clicked = pyqtSignal(str)

    # Color palette for trajectories (vibrant colors for dark background)
    COLORS = [
        '#00d4aa', '#ff7f0e', '#4fc3f7', '#ff6b6b', '#a29bfe',
        '#ffeaa7', '#fd79a8', '#74b9ff', '#55efc4', '#fdcb6e',
        '#e17055', '#00cec9', '#6c5ce7', '#fab1a0', '#81ecec'
    ]

    def __init__(self, parent=None):
        """Initialize the trajectory canvas."""
        super().__init__(parent)
        self.logger = get_logger()

        self.data_manager: Optional[DataManager] = None
        self.canvas_size_um: float = 100.0
        self.displayed_objects: List[str] = []
        self.trajectory_lines: Dict[str, any] = {}
        self.trajectory_markers: Dict[str, List] = {}
        self.hidden_objects: Set[str] = set()
        self.hover_annotation = None
        self.object_colors: Dict[str, str] = {}

        # Zoom-related variables
        self.zoom_factor: float = 1.2
        self.original_xlim: tuple = (0, 100.0)
        self.original_ylim: tuple = (100.0, 0)
        self.current_xlim: tuple = (0, 100.0)
        self.current_ylim: tuple = (100.0, 0)

        self._setup_ui()
        self._connect_events()

    def _setup_ui(self):
        """Set up the Matplotlib figure and canvas with dark theme."""
        self.figure = Figure(figsize=(6, 6), dpi=100)
        self.figure.patch.set_facecolor(Theme.BG_SECONDARY)

        self.canvas = FigureCanvasQTAgg(self.figure)
        self.ax = self.figure.add_subplot(111)
        self.ax.set_facecolor(Theme.BG_CARD)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.addWidget(self.canvas)

        self._setup_axes()

    def _setup_axes(self, preserve_zoom: bool = False):
        """
        Configure axes properties with dark theme colors.

        Args:
            preserve_zoom: If True, maintain current zoom level.
        """
        if preserve_zoom and hasattr(self, 'current_xlim'):
            # Restore previous zoom state
            self.ax.set_xlim(self.current_xlim)
            self.ax.set_ylim(self.current_ylim)
        else:
            # Set to full canvas size and save as original
            self.ax.set_xlim(0, self.canvas_size_um)
            self.ax.set_ylim(self.canvas_size_um, 0)  # Origin at top-left
            self.original_xlim = (0, self.canvas_size_um)
            self.original_ylim = (self.canvas_size_um, 0)
            self.current_xlim = self.original_xlim
            self.current_ylim = self.original_ylim

        self.ax.set_aspect('equal')
        self.ax.set_facecolor(Theme.BG_CARD)

        # Axis labels with light color
        self.ax.set_xlabel(
            'X (μm)', fontsize=12, fontname='Arial',
            color=Theme.TEXT_PRIMARY
        )
        self.ax.set_ylabel(
            'Y (μm)', fontsize=12, fontname='Arial',
            color=Theme.TEXT_PRIMARY
        )

        # Tick parameters for dark theme
        self.ax.tick_params(
            labelsize=10,
            colors=Theme.TEXT_SECONDARY,
            which='both'
        )

        # Spine colors
        for spine in self.ax.spines.values():
            spine.set_color(Theme.BORDER)

        # Grid with subtle color
        self.ax.grid(
            True, linestyle='--', alpha=0.3,
            color=Theme.TEXT_SECONDARY
        )

        # Set tick font
        for label in self.ax.get_xticklabels() + self.ax.get_yticklabels():
            label.set_fontname('Arial')
            label.set_color(Theme.TEXT_SECONDARY)

    def _connect_events(self):
        """Connect Matplotlib event handlers."""
        self.canvas.mpl_connect('motion_notify_event', self._on_mouse_move)
        self.canvas.mpl_connect('button_press_event', self._on_mouse_click)
        self.canvas.mpl_connect('scroll_event', self._on_scroll)

    def set_data_manager(self, data_manager: DataManager):
        """
        Set the data manager instance.

        Args:
            data_manager: DataManager containing trajectory data.
        """
        self.data_manager = data_manager

    def set_canvas_size(self, size_um: float):
        """
        Set the canvas size in micrometers.

        Args:
            size_um: Canvas side length in μm.
        """
        self.canvas_size_um = size_um
        self._setup_axes()
        self.redraw_current()

    def draw_trajectories(self, object_ids: List[str]):
        """
        Draw specified trajectories on the canvas.

        Args:
            object_ids: List of object IDs to draw.
        """
        if self.data_manager is None:
            return

        self.ax.clear()
        self._setup_axes()

        self.trajectory_lines.clear()
        self.trajectory_markers.clear()
        self.hidden_objects.clear()
        self.displayed_objects = object_ids.copy()
        self.object_colors.clear()

        for i, obj_id in enumerate(object_ids):
            df = self.data_manager.get_object_data(obj_id)
            if df is None:
                continue

            x = df['center_x (μm)'].values
            y = df['center_y (μm)'].values
            color = self.COLORS[i % len(self.COLORS)]
            self.object_colors[obj_id] = color

            # Draw trajectory line with picker enabled
            line, = self.ax.plot(
                x, y, '-',
                linewidth=1.5,
                color=color,
                picker=5,  # 5 pixel tolerance
                label=obj_id
            )
            self.trajectory_lines[obj_id] = line

            # Start point marker (triangle)
            start_marker, = self.ax.plot(
                x[0], y[0],
                marker='^',
                markersize=8,
                color=color,
                markeredgecolor=Theme.BG_CARD,
                markeredgewidth=1
            )

            # End point marker (circle)
            end_marker, = self.ax.plot(
                x[-1], y[-1],
                marker='o',
                markersize=6,
                color=color,
                markeredgecolor=Theme.BG_CARD,
                markeredgewidth=1
            )

            self.trajectory_markers[obj_id] = [start_marker, end_marker]

        # Create hover annotation (hidden initially) with dark theme
        self.hover_annotation = self.ax.annotate(
            '',
            xy=(0, 0),
            xytext=(15, 15),
            textcoords='offset points',
            bbox=dict(
                boxstyle='round,pad=0.4',
                fc=Theme.BG_PRIMARY,
                ec=Theme.ACCENT,
                alpha=0.95
            ),
            fontsize=10,
            fontname='Arial',
            color=Theme.ACCENT
        )
        self.hover_annotation.set_visible(False)

        self.canvas.draw()

    def redraw_random(self, n: int):
        """
        Randomly select and draw n trajectories.

        Args:
            n: Number of trajectories to display.
        """
        if self.data_manager is None:
            return

        object_ids = self.data_manager.get_random_objects(n)
        self.draw_trajectories(object_ids)

    def redraw_current(self):
        """Redraw currently displayed trajectories."""
        if self.displayed_objects:
            visible_objects = [
                obj_id for obj_id in self.displayed_objects
                if obj_id not in self.hidden_objects
            ]
            self._redraw_visible(visible_objects)

    def _redraw_visible(self, visible_ids: List[str]):
        """Redraw only visible trajectories while preserving zoom state."""
        if self.data_manager is None:
            return

        self.ax.clear()
        self._setup_axes(preserve_zoom=True)

        temp_lines = {}
        temp_markers = {}

        for obj_id in self.displayed_objects:
            if obj_id in self.hidden_objects:
                continue

            df = self.data_manager.get_object_data(obj_id)
            if df is None:
                continue

            x = df['center_x (μm)'].values
            y = df['center_y (μm)'].values
            color = self.object_colors.get(
                obj_id,
                self.COLORS[0]
            )

            line, = self.ax.plot(
                x, y, '-',
                linewidth=1.5,
                color=color,
                picker=5,
                label=obj_id
            )
            temp_lines[obj_id] = line

            start_marker, = self.ax.plot(
                x[0], y[0],
                marker='^',
                markersize=8,
                color=color,
                markeredgecolor=Theme.BG_CARD,
                markeredgewidth=1
            )
            end_marker, = self.ax.plot(
                x[-1], y[-1],
                marker='o',
                markersize=6,
                color=color,
                markeredgecolor=Theme.BG_CARD,
                markeredgewidth=1
            )
            temp_markers[obj_id] = [start_marker, end_marker]

        self.trajectory_lines = temp_lines
        self.trajectory_markers = temp_markers

        # Recreate hover annotation with dark theme
        self.hover_annotation = self.ax.annotate(
            '',
            xy=(0, 0),
            xytext=(15, 15),
            textcoords='offset points',
            bbox=dict(
                boxstyle='round,pad=0.4',
                fc=Theme.BG_PRIMARY,
                ec=Theme.ACCENT,
                alpha=0.95
            ),
            fontsize=10,
            fontname='Arial',
            color=Theme.ACCENT
        )
        self.hover_annotation.set_visible(False)

        self.canvas.draw()

    def _on_mouse_move(self, event):
        """Handle mouse move event for hover detection."""
        if event.inaxes != self.ax:
            self._hide_hover_label()
            return

        # Check if mouse is near any visible trajectory
        for obj_id, line in self.trajectory_lines.items():
            if obj_id in self.hidden_objects:
                continue

            contains, _ = line.contains(event)
            if contains:
                self._show_hover_label(obj_id, event.xdata, event.ydata)
                return

        self._hide_hover_label()

    def _on_mouse_click(self, event):
        """Handle mouse click events."""
        # Middle button - reset zoom
        if event.button == 2:
            self._reset_zoom()
            return

        if event.inaxes != self.ax:
            # Double click outside axes - show all
            if event.dblclick:
                self.show_all_trajectories()
            return

        # Check for double click on empty area
        if event.dblclick:
            clicked_obj = self._find_clicked_trajectory(event)
            if clicked_obj is None:
                self.show_all_trajectories()
            return

        # Single click
        clicked_obj = self._find_clicked_trajectory(event)

        if clicked_obj:
            if event.button == 1:  # Left click - hide this trajectory
                self.hide_trajectory(clicked_obj)
            elif event.button == 3:  # Right click - hide others
                self.hide_other_trajectories(clicked_obj)

    def _on_scroll(self, event):
        """
        Handle scroll event for zooming.

        Zoom is centered on the mouse position.
        Cannot zoom out beyond the original canvas size.
        """
        if event.inaxes != self.ax:
            return

        # Get mouse position in data coordinates
        xdata, ydata = event.xdata, event.ydata
        if xdata is None or ydata is None:
            return

        # Get current axis limits
        cur_xlim = self.ax.get_xlim()
        cur_ylim = self.ax.get_ylim()

        # Determine zoom direction
        if event.button == 'up':
            # Zoom in
            scale = 1.0 / self.zoom_factor
        elif event.button == 'down':
            # Zoom out
            scale = self.zoom_factor
        else:
            return

        # Calculate new limits centered on mouse position
        new_xmin = xdata - (xdata - cur_xlim[0]) * scale
        new_xmax = xdata + (cur_xlim[1] - xdata) * scale
        new_ymin = ydata - (ydata - cur_ylim[0]) * scale
        new_ymax = ydata + (cur_ylim[1] - ydata) * scale

        # Clamp to original limits (cannot zoom out beyond original)
        orig_xmin, orig_xmax = self.original_xlim
        orig_ymin, orig_ymax = self.original_ylim  # Note: ymin > ymax due to inverted axis

        # For X axis
        if new_xmin < orig_xmin:
            new_xmin = orig_xmin
        if new_xmax > orig_xmax:
            new_xmax = orig_xmax

        # For Y axis (inverted: orig_ymin is larger value, orig_ymax is smaller)
        if new_ymin > orig_ymin:
            new_ymin = orig_ymin
        if new_ymax < orig_ymax:
            new_ymax = orig_ymax

        # Check if the new range is larger than original (zooming out too much)
        if (new_xmax - new_xmin) >= (orig_xmax - orig_xmin):
            new_xmin, new_xmax = orig_xmin, orig_xmax
        if (new_ymin - new_ymax) >= (orig_ymin - orig_ymax):
            new_ymin, new_ymax = orig_ymin, orig_ymax

        # Apply new limits
        self.ax.set_xlim(new_xmin, new_xmax)
        self.ax.set_ylim(new_ymin, new_ymax)

        # Update current limits
        self.current_xlim = (new_xmin, new_xmax)
        self.current_ylim = (new_ymin, new_ymax)

        self.canvas.draw_idle()

    def _reset_zoom(self):
        """Reset zoom to original canvas size."""
        self.ax.set_xlim(self.original_xlim)
        self.ax.set_ylim(self.original_ylim)
        self.current_xlim = self.original_xlim
        self.current_ylim = self.original_ylim
        self.canvas.draw_idle()
        self.logger.debug("Zoom reset to original view")

    def _find_clicked_trajectory(self, event) -> Optional[str]:
        """Find which trajectory was clicked."""
        for obj_id, line in self.trajectory_lines.items():
            if obj_id in self.hidden_objects:
                continue
            contains, _ = line.contains(event)
            if contains:
                return obj_id
        return None

    def _show_hover_label(self, obj_id: str, x: float, y: float):
        """Show hover label with trajectory ID."""
        if self.hover_annotation is None:
            return

        self.hover_annotation.set_text(obj_id)
        self.hover_annotation.xy = (x, y)
        self.hover_annotation.set_visible(True)
        self.canvas.draw_idle()

    def _hide_hover_label(self):
        """Hide the hover label."""
        if self.hover_annotation and self.hover_annotation.get_visible():
            self.hover_annotation.set_visible(False)
            self.canvas.draw_idle()

    def hide_trajectory(self, obj_id: str):
        """
        Hide a specific trajectory.

        Args:
            obj_id: Object ID to hide.
        """
        if obj_id not in self.hidden_objects:
            self.hidden_objects.add(obj_id)
            self._redraw_visible([
                oid for oid in self.displayed_objects
                if oid not in self.hidden_objects
            ])
            self.logger.debug(f"Hidden trajectory: {obj_id}")

    def hide_other_trajectories(self, obj_id: str):
        """
        Hide all trajectories except the specified one.

        Args:
            obj_id: Object ID to keep visible.
        """
        for oid in self.displayed_objects:
            if oid != obj_id:
                self.hidden_objects.add(oid)

        self._redraw_visible([obj_id])
        self.logger.debug(f"Showing only: {obj_id}")

    def show_all_trajectories(self):
        """Show all trajectories (unhide all)."""
        self.hidden_objects.clear()
        self._redraw_visible(self.displayed_objects)
        self.logger.debug("Showing all trajectories")

    def get_displayed_count(self) -> int:
        """Get number of currently visible trajectories."""
        return len(self.displayed_objects) - len(self.hidden_objects)
