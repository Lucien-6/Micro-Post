#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Micro Post - Bacterial Trajectory Motion Analyzer

A desktop application for comprehensive motion analysis of bacterial
trajectories from tracking data.

Copyright (c) 2026 Lucien
MIT License
"""

import sys
import os
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from PyQt6.QtWidgets import QApplication
from PyQt6.QtGui import QFont, QPalette, QColor, QIcon
from PyQt6.QtCore import Qt

from src.ui.main_window import MainWindow
from src.ui.styles import get_main_stylesheet
from src.ui.theme import Theme


def main():
    """Main entry point for the application."""
    # Enable high DPI scaling
    QApplication.setHighDpiScaleFactorRoundingPolicy(
        Qt.HighDpiScaleFactorRoundingPolicy.PassThrough
    )

    # Create application
    app = QApplication(sys.argv)
    app.setApplicationName("Micro Post")
    app.setApplicationVersion("1.2.0")
    app.setOrganizationName("Lucien")

    # Set Fusion style (works best with custom stylesheets)
    app.setStyle("Fusion")

    # Set dark palette for native dialogs
    palette = QPalette()
    palette.setColor(QPalette.ColorRole.Window, QColor(Theme.BG_PRIMARY))
    palette.setColor(QPalette.ColorRole.WindowText, QColor(Theme.TEXT_PRIMARY))
    palette.setColor(QPalette.ColorRole.Base, QColor(Theme.BG_INPUT))
    palette.setColor(QPalette.ColorRole.AlternateBase, QColor(Theme.BG_SECONDARY))
    palette.setColor(QPalette.ColorRole.ToolTipBase, QColor(Theme.BG_CARD))
    palette.setColor(QPalette.ColorRole.ToolTipText, QColor(Theme.TEXT_PRIMARY))
    palette.setColor(QPalette.ColorRole.Text, QColor(Theme.TEXT_PRIMARY))
    palette.setColor(QPalette.ColorRole.Button, QColor(Theme.BG_SECONDARY))
    palette.setColor(QPalette.ColorRole.ButtonText, QColor(Theme.TEXT_PRIMARY))
    palette.setColor(QPalette.ColorRole.BrightText, QColor(Theme.ACCENT))
    palette.setColor(QPalette.ColorRole.Link, QColor(Theme.ACCENT))
    palette.setColor(QPalette.ColorRole.Highlight, QColor(Theme.ACCENT))
    palette.setColor(QPalette.ColorRole.HighlightedText, QColor(Theme.TEXT_DARK))
    app.setPalette(palette)

    # Set default font
    font = QFont("Arial", 10)
    app.setFont(font)

    # Apply custom stylesheet
    app.setStyleSheet(get_main_stylesheet())

    # Set application icon (handle both development and frozen executable)
    if getattr(sys, 'frozen', False):
        # Running as compiled executable
        base_path = Path(sys._MEIPASS)
    else:
        # Running in development
        base_path = Path(__file__).parent
    
    icon_path = base_path / "src" / "ui" / "resources" / "icon.png"
    if icon_path.exists():
        app_icon = QIcon(str(icon_path))
        app.setWindowIcon(app_icon)

    # Create and show main window
    window = MainWindow()
    window.setWindowIcon(app.windowIcon())
    window.show()

    # Run event loop
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
