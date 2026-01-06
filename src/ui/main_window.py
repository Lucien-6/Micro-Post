"""
Main window module for Micro Post application.

This module provides the main GUI window with all controls
for data loading, trajectory preview, and motion analysis.
"""

import os
from pathlib import Path
from typing import Optional

from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QGroupBox, QLabel, QLineEdit, QPushButton, QSpinBox,
    QDoubleSpinBox, QFileDialog, QMessageBox, QStatusBar,
    QSplitter, QProgressDialog, QApplication
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QUrl
from PyQt6.QtGui import QFont, QDesktopServices

from src.core.data_loader import DataLoader
from src.core.data_manager import DataManager
from src.core.motion_analyzer import MotionAnalyzer
from src.ui.trajectory_canvas import TrajectoryCanvas
from src.utils.logger import setup_logger, get_logger


class AnalysisWorker(QThread):
    """Worker thread for motion analysis."""

    progress = pyqtSignal(int, str)
    finished = pyqtSignal(bool, str)

    def __init__(self, analyzer: MotionAnalyzer):
        super().__init__()
        self.analyzer = analyzer

    def run(self):
        """Run the analysis in background."""
        try:
            self.progress.emit(10, "Analyzing trajectories...")
            self.analyzer.analyze_all_objects()

            self.progress.emit(80, "Computing summary statistics...")
            self.analyzer.save_analysis_results()

            self.progress.emit(100, "Complete")
            self.finished.emit(True, "Analysis completed successfully!")

        except Exception as e:
            self.finished.emit(False, f"Analysis failed: {str(e)}")


class MainWindow(QMainWindow):
    """
    Main application window.

    Provides the complete GUI for trajectory data loading,
    preview, filtering, and motion analysis.
    """

    def __init__(self):
        """Initialize the main window."""
        super().__init__()

        self.logger: Optional[any] = None
        self.data_loader: Optional[DataLoader] = None
        self.data_manager: Optional[DataManager] = None
        self.analyzer: Optional[MotionAnalyzer] = None
        self.analysis_worker: Optional[AnalysisWorker] = None

        self._setup_ui()
        self._connect_signals()

    def _setup_ui(self):
        """Set up the user interface with dark tech theme."""
        self.setWindowTitle("Micro Post - Bacterial Trajectory Motion Analyzer")
        self.setMinimumSize(1100, 750)

        # Central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        # Main layout with splitter
        main_layout = QHBoxLayout(central_widget)
        main_layout.setContentsMargins(12, 12, 12, 12)
        main_layout.setSpacing(12)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        main_layout.addWidget(splitter)

        # Left panel: Trajectory canvas
        self.canvas = TrajectoryCanvas()
        splitter.addWidget(self.canvas)

        # Right panel: Controls (fixed width)
        control_panel = QWidget()
        control_panel.setFixedWidth(320)
        control_layout = QVBoxLayout(control_panel)
        control_layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        control_layout.setSpacing(12)
        control_layout.setContentsMargins(10, 0, 0, 0)  # Left margin for splitter gap

        # Define label width for alignment
        LABEL_WIDTH = 120

        # Data Loading Group
        load_group = QGroupBox("📁  Data Loading")
        load_group.setObjectName("loadGroup")
        load_layout = QVBoxLayout(load_group)
        load_layout.setSpacing(10)

        # Folder selection
        folder_layout = QHBoxLayout()
        folder_layout.setSpacing(8)
        folder_label = QLabel("Folder:")
        folder_label.setFixedWidth(50)
        folder_layout.addWidget(folder_label)
        self.folder_edit = QLineEdit()
        self.folder_edit.setReadOnly(True)
        self.folder_edit.setPlaceholderText("Select data folder...")
        folder_layout.addWidget(self.folder_edit, 1)
        self.browse_btn = QPushButton("Browse...")
        self.browse_btn.setObjectName("browseBtn")
        self.browse_btn.setFixedWidth(80)
        folder_layout.addWidget(self.browse_btn)
        load_layout.addLayout(folder_layout)

        # Minimum duration
        duration_layout = QHBoxLayout()
        duration_layout.setSpacing(8)
        duration_label = QLabel("Min Duration (s):")
        duration_label.setFixedWidth(LABEL_WIDTH)
        duration_layout.addWidget(duration_label)
        self.duration_spin = QSpinBox()
        self.duration_spin.setRange(2, 200)
        self.duration_spin.setValue(10)
        self.duration_spin.setFixedWidth(85)
        duration_layout.addWidget(self.duration_spin)
        duration_layout.addStretch()
        load_layout.addLayout(duration_layout)

        # Load button
        self.load_btn = QPushButton("⬇  Load Data")
        self.load_btn.setObjectName("loadBtn")
        self.load_btn.setEnabled(False)
        load_layout.addWidget(self.load_btn)

        control_layout.addWidget(load_group)

        # Preview Settings Group
        preview_group = QGroupBox("👁  Preview Settings")
        preview_group.setObjectName("previewGroup")
        preview_layout = QVBoxLayout(preview_group)
        preview_layout.setSpacing(10)

        # Canvas size
        size_layout = QHBoxLayout()
        size_layout.setSpacing(8)
        size_label = QLabel("Canvas Size (μm):")
        size_label.setFixedWidth(LABEL_WIDTH)
        size_layout.addWidget(size_label)
        self.size_spin = QDoubleSpinBox()
        self.size_spin.setRange(10, 1000)
        self.size_spin.setValue(100)
        self.size_spin.setSingleStep(10)
        self.size_spin.setFixedWidth(85)
        size_layout.addWidget(self.size_spin)
        size_layout.addStretch()
        preview_layout.addLayout(size_layout)

        # Number of trajectories
        count_layout = QHBoxLayout()
        count_layout.setSpacing(8)
        count_label = QLabel("Display Count (N):")
        count_label.setFixedWidth(LABEL_WIDTH)
        count_layout.addWidget(count_label)
        self.count_spin = QSpinBox()
        self.count_spin.setRange(1, 100)
        self.count_spin.setValue(10)
        self.count_spin.setFixedWidth(85)
        count_layout.addWidget(self.count_spin)
        count_layout.addStretch()
        preview_layout.addLayout(count_layout)

        # Redraw button
        self.redraw_btn = QPushButton("🔄  Redraw")
        self.redraw_btn.setObjectName("redrawBtn")
        self.redraw_btn.setEnabled(False)
        preview_layout.addWidget(self.redraw_btn)

        # Interaction hints
        hint_label = QLabel(
            "💡 Interactions:\n"
            "   • Hover: Show ID\n"
            "   • Left-click: Hide trajectory\n"
            "   • Right-click: Hide others\n"
            "   • Double-click empty: Show all"
        )
        hint_label.setObjectName("hintLabel")
        preview_layout.addWidget(hint_label)

        control_layout.addWidget(preview_group)

        # Exclude Objects Group
        exclude_group = QGroupBox("🚫  Exclude Objects")
        exclude_group.setObjectName("excludeGroup")
        exclude_layout = QVBoxLayout(exclude_group)
        exclude_layout.setSpacing(10)

        self.exclude_edit = QLineEdit()
        self.exclude_edit.setPlaceholderText("e.g., 3, 7-10, 15")
        exclude_layout.addWidget(self.exclude_edit)

        self.exclude_btn = QPushButton("✗  Confirm Exclusion")
        self.exclude_btn.setObjectName("excludeBtn")
        self.exclude_btn.setEnabled(False)
        exclude_layout.addWidget(self.exclude_btn)

        control_layout.addWidget(exclude_group)

        # Analysis Group with help button
        analysis_group = QGroupBox("📊  Motion Analysis")
        analysis_group.setObjectName("analysisGroup")
        analysis_layout = QVBoxLayout(analysis_group)
        analysis_layout.setSpacing(12)

        self.analyze_btn = QPushButton("▶  Run Motion Analysis")
        self.analyze_btn.setObjectName("analyzeBtn")
        self.analyze_btn.setEnabled(False)
        analysis_layout.addWidget(self.analyze_btn)

        # Help button in top-right corner of the group
        self.help_btn = QPushButton("❓")
        self.help_btn.setObjectName("helpBtn")
        self.help_btn.setFixedSize(24, 24)
        self.help_btn.setToolTip("Open Documentation")
        self.help_btn.setStyleSheet("""
            QPushButton#helpBtn {
                background-color: transparent;
                border: none;
                font-size: 14px;
                padding: 0px;
            }
            QPushButton#helpBtn:hover {
                background-color: rgba(0, 212, 170, 0.2);
                border-radius: 12px;
            }
        """)
        # Position help button at top-right of group box
        self.help_btn.setParent(analysis_group)
        self.help_btn.move(analysis_group.width() - 35, 2)

        control_layout.addWidget(analysis_group)

        # Spacer
        control_layout.addStretch()

        splitter.addWidget(control_panel)

        # Set splitter proportions
        splitter.setSizes([750, 350])

        # Status bar
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("✨ Ready. Please select a data folder.")

    def _connect_signals(self):
        """Connect widget signals to slots."""
        self.browse_btn.clicked.connect(self._on_browse)
        self.load_btn.clicked.connect(self._on_load_data)
        self.size_spin.valueChanged.connect(self._on_canvas_size_changed)
        self.count_spin.valueChanged.connect(self._on_count_changed)
        self.redraw_btn.clicked.connect(self._on_redraw)
        self.exclude_btn.clicked.connect(self._on_exclude)
        self.analyze_btn.clicked.connect(self._on_analyze)
        self.help_btn.clicked.connect(self._on_help)

    def _on_browse(self):
        """Handle browse button click."""
        folder = QFileDialog.getExistingDirectory(
            self,
            "Select Data Folder",
            "",
            QFileDialog.Option.ShowDirsOnly
        )

        if folder:
            self.folder_edit.setText(folder)
            self.load_btn.setEnabled(True)
            self.status_bar.showMessage(f"Selected folder: {folder}")

    def _on_load_data(self):
        """Handle load data button click."""
        folder = self.folder_edit.text()
        if not folder:
            return

        # Initialize logger
        self.logger = setup_logger(folder)
        self.logger.info(f"Loading data from: {folder}")

        # Create data loader
        self.data_loader = DataLoader(folder)

        # Search for files
        self.status_bar.showMessage("Searching for trajectory files...")
        QApplication.processEvents()

        files = self.data_loader.search_excel_files()
        if not files:
            QMessageBox.warning(
                self,
                "No Files Found",
                "No trajectory files found in the selected folder.\n"
                "Please ensure files are named 'Trajectories_Results_*.xlsx'"
            )
            self.status_bar.showMessage("No files found.")
            return

        # Validate parameters
        self.status_bar.showMessage("Validating parameters...")
        QApplication.processEvents()

        is_consistent, inconsistencies = self.data_loader.validate_parameters()

        if not is_consistent:
            # Show detailed error message
            msg = "Parameter inconsistencies detected!\n\n"
            for inc in inconsistencies[:10]:  # Show first 10
                msg += (
                    f"File: {os.path.basename(inc['file'])}\n"
                    f"  Parameter: {inc['param']}\n"
                    f"  Expected: {inc['expected']}\n"
                    f"  Actual: {inc['actual']}\n\n"
                )
            if len(inconsistencies) > 10:
                msg += f"... and {len(inconsistencies) - 10} more.\n"

            QMessageBox.critical(
                self,
                "Parameter Mismatch",
                msg
            )
            self.status_bar.showMessage("Parameter validation failed.")
            return

        # Create data manager and filter data
        self.status_bar.showMessage("Filtering and merging data...")
        QApplication.processEvents()

        self.data_manager = DataManager(self.data_loader)
        min_duration = self.duration_spin.value()
        total_objects = self.data_manager.filter_and_merge(min_duration)

        if total_objects == 0:
            QMessageBox.warning(
                self,
                "No Objects",
                f"No objects with duration >= {min_duration}s found."
            )
            self.status_bar.showMessage("No valid objects found.")
            return

        # Save summary Excel
        self.status_bar.showMessage("Saving summary file...")
        QApplication.processEvents()

        summary_path = self.data_manager.save_summary_excel()

        # Set up canvas
        self.canvas.set_data_manager(self.data_manager)
        self.count_spin.setMaximum(total_objects)

        # Draw initial trajectories
        n_display = min(self.count_spin.value(), total_objects)
        self.canvas.redraw_random(n_display)

        # Enable controls
        self.redraw_btn.setEnabled(True)
        self.exclude_btn.setEnabled(True)
        self.analyze_btn.setEnabled(True)

        # Update status
        self.status_bar.showMessage(
            f"Loaded {total_objects} objects | "
            f"Displaying {n_display} trajectories | "
            f"Saved to: {os.path.basename(summary_path)}"
        )

        QMessageBox.information(
            self,
            "Data Loaded",
            f"Successfully loaded {total_objects} objects.\n"
            f"Summary saved to:\n{summary_path}"
        )

    def _on_canvas_size_changed(self, value: float):
        """Handle canvas size change."""
        self.canvas.set_canvas_size(value)

    def _on_count_changed(self, value: int):
        """Handle display count change."""
        if self.data_manager:
            self.canvas.redraw_random(value)
            self._update_status()

    def _on_redraw(self):
        """Handle redraw button click."""
        if self.data_manager:
            n = self.count_spin.value()
            self.canvas.redraw_random(n)
            self._update_status()

    def _on_exclude(self):
        """Handle exclude button click."""
        if not self.data_manager:
            return

        input_text = self.exclude_edit.text().strip()
        if not input_text:
            QMessageBox.warning(
                self,
                "No Input",
                "Please enter object numbers to exclude."
            )
            return

        # Parse input
        numbers = self.data_manager.parse_exclude_input(input_text)
        if not numbers:
            QMessageBox.warning(
                self,
                "Invalid Input",
                "Could not parse any valid object numbers.\n"
                "Use format: 3, 7-10, 15"
            )
            return

        # Convert to object IDs
        object_ids = [f"Object_{n}" for n in numbers]

        # Confirm exclusion
        reply = QMessageBox.question(
            self,
            "Confirm Exclusion",
            f"Exclude {len(object_ids)} object(s)?\n"
            f"IDs: {', '.join(object_ids[:10])}"
            f"{'...' if len(object_ids) > 10 else ''}",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )

        if reply == QMessageBox.StandardButton.Yes:
            excluded = self.data_manager.exclude_objects(object_ids)

            # Update count spin max
            remaining = self.data_manager.get_object_count()
            self.count_spin.setMaximum(remaining)

            # Redraw
            n = min(self.count_spin.value(), remaining)
            self.canvas.redraw_random(n)

            # Clear input
            self.exclude_edit.clear()

            self.status_bar.showMessage(
                f"Excluded {excluded} objects | "
                f"{remaining} objects remaining"
            )

            if remaining == 0:
                self.analyze_btn.setEnabled(False)
                QMessageBox.warning(
                    self,
                    "No Objects",
                    "All objects have been excluded."
                )

    def _on_analyze(self):
        """Handle analyze button click."""
        if not self.data_manager:
            return

        object_count = self.data_manager.get_object_count()
        if object_count == 0:
            QMessageBox.warning(
                self,
                "No Objects",
                "No objects to analyze."
            )
            return

        # Confirm analysis
        reply = QMessageBox.question(
            self,
            "Confirm Analysis",
            f"Run motion analysis on {object_count} objects?\n"
            f"This may take a few minutes.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )

        if reply != QMessageBox.StandardButton.Yes:
            return

        # Create analyzer
        self.analyzer = MotionAnalyzer(self.data_manager)

        # Create progress dialog
        progress = QProgressDialog(
            "Analyzing trajectories...",
            None,  # No cancel button
            0, 100,
            self
        )
        progress.setWindowTitle("Motion Analysis")
        progress.setWindowModality(Qt.WindowModality.WindowModal)
        progress.setMinimumDuration(0)
        progress.setValue(0)

        # Create worker thread
        self.analysis_worker = AnalysisWorker(self.analyzer)
        self.analysis_worker.progress.connect(
            lambda val, msg: self._on_analysis_progress(progress, val, msg)
        )
        self.analysis_worker.finished.connect(
            lambda success, msg: self._on_analysis_finished(progress, success, msg)
        )

        # Start analysis
        self.analyze_btn.setEnabled(False)
        self.analysis_worker.start()

    def _on_analysis_progress(
        self,
        progress: QProgressDialog,
        value: int,
        message: str
    ):
        """Handle analysis progress update."""
        progress.setValue(value)
        progress.setLabelText(message)
        self.status_bar.showMessage(message)

    def _on_analysis_finished(
        self,
        progress: QProgressDialog,
        success: bool,
        message: str
    ):
        """Handle analysis completion."""
        progress.close()
        self.analyze_btn.setEnabled(True)

        if success:
            QMessageBox.information(
                self,
                "Analysis Complete",
                f"{message}\n\n"
                f"Results saved to:\n{self.data_manager.summary_path}"
            )
            self.status_bar.showMessage(
                f"Analysis complete | Results saved to "
                f"{os.path.basename(self.data_manager.summary_path)}"
            )
        else:
            QMessageBox.critical(
                self,
                "Analysis Failed",
                message
            )
            self.status_bar.showMessage("Analysis failed.")

    def _update_status(self):
        """Update status bar with current state."""
        if self.data_manager:
            total = self.data_manager.get_object_count()
            displayed = self.canvas.get_displayed_count()
            self.status_bar.showMessage(
                f"Total: {total} objects | Displaying: {displayed} trajectories"
            )

    def _on_help(self):
        """Open the documentation HTML file."""
        # Get the path to the documentation file
        docs_path = Path(__file__).parent.parent.parent / "docs" / "manual.html"

        if docs_path.exists():
            # Open in default browser
            url = QUrl.fromLocalFile(str(docs_path.resolve()))
            QDesktopServices.openUrl(url)
            self.status_bar.showMessage("Documentation opened in browser")
        else:
            QMessageBox.warning(
                self,
                "Documentation Not Found",
                f"Could not find documentation file:\n{docs_path}"
            )

    def showEvent(self, event):
        """Handle show event to position help button."""
        super().showEvent(event)
        # Reposition help button after window is shown
        self._reposition_help_button()

    def resizeEvent(self, event):
        """Handle resize event to reposition help button."""
        super().resizeEvent(event)
        self._reposition_help_button()

    def _reposition_help_button(self):
        """Reposition help button to top-right of analysis group."""
        if hasattr(self, 'help_btn') and self.help_btn.parent():
            parent = self.help_btn.parent()
            self.help_btn.move(parent.width() - 22, 8)
