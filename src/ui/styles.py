"""
QSS Stylesheet module for Micro Post application.

This module provides the complete Qt Style Sheet (QSS) for
the dark tech theme.
"""

from pathlib import Path

from src.ui.theme import Theme


def _get_resource_path() -> str:
    """
    Get the absolute path to the resources directory.

    Returns:
        Absolute path string with forward slashes for QSS compatibility.
    """
    resources_dir = Path(__file__).parent / "resources"
    # Use forward slashes for QSS URL compatibility on Windows
    return resources_dir.as_posix()


def get_main_stylesheet() -> str:
    """
    Generate the complete QSS stylesheet for the application.

    Returns:
        Complete QSS stylesheet string.
    """
    res_path = _get_resource_path()
    return f"""
    /* ========== Main Window ========== */
    QMainWindow {{
        background-color: {Theme.BG_PRIMARY};
    }}

    QWidget {{
        background-color: transparent;
        color: {Theme.TEXT_PRIMARY};
        font-family: {Theme.FONT_FAMILY};
        font-size: {Theme.FONT_SIZE_NORMAL}px;
    }}

    /* ========== Group Box (Cards) ========== */
    QGroupBox {{
        background-color: {Theme.BG_CARD};
        border: 1px solid {Theme.BORDER};
        border-radius: {Theme.RADIUS_CARD}px;
        margin-top: 10px;
        padding: 12px;
        padding-top: 28px;
        font-weight: bold;
    }}

    QGroupBox::title {{
        subcontrol-origin: margin;
        subcontrol-position: top left;
        left: 10px;
        top: 2px;
        padding: 3px 8px;
        background-color: {Theme.BG_SECONDARY};
        border-radius: 4px;
        color: {Theme.ACCENT};
        font-size: {Theme.FONT_SIZE_GROUP}px;
    }}

    /* ========== Labels ========== */
    QLabel {{
        background-color: transparent;
        color: {Theme.TEXT_SECONDARY};
        font-size: {Theme.FONT_SIZE_NORMAL}px;
        padding: 0px;
        min-height: 18px;
    }}

    QLabel#titleLabel {{
        color: {Theme.TEXT_PRIMARY};
        font-size: {Theme.FONT_SIZE_TITLE}px;
        font-weight: bold;
    }}

    QLabel#hintLabel {{
        color: {Theme.TEXT_SECONDARY};
        font-size: {Theme.FONT_SIZE_SMALL}px;
        padding: 8px 10px;
        background-color: {Theme.BG_INPUT};
        border-radius: {Theme.RADIUS_INPUT}px;
        min-height: auto;
        line-height: 1.4;
    }}

    /* ========== Push Buttons ========== */
    QPushButton {{
        background-color: {Theme.BG_SECONDARY};
        color: {Theme.TEXT_PRIMARY};
        border: 1px solid {Theme.BORDER};
        border-radius: {Theme.RADIUS_BTN}px;
        padding: 4px 12px;
        min-height: 21px;
        font-weight: 500;
    }}

    QPushButton:hover {{
        background-color: {Theme.BG_CARD};
        border-color: {Theme.ACCENT};
        color: {Theme.ACCENT};
    }}

    QPushButton:pressed {{
        background-color: {Theme.BG_INPUT};
        border-color: {Theme.ACCENT_HOVER};
    }}

    QPushButton:disabled {{
        background-color: {Theme.BG_INPUT};
        color: {Theme.TEXT_DISABLED};
        border-color: {Theme.BG_SECONDARY};
    }}

    /* Primary Action Button */
    QPushButton#analyzeBtn {{
        background-color: {Theme.ACCENT};
        color: {Theme.TEXT_DARK};
        border: none;
        font-weight: bold;
        font-size: 12px;
    }}

    QPushButton#analyzeBtn:hover {{
        background-color: {Theme.ACCENT_HOVER};
        color: {Theme.TEXT_DARK};
    }}

    QPushButton#analyzeBtn:pressed {{
        background-color: {Theme.ACCENT_SECONDARY};
    }}

    QPushButton#analyzeBtn:disabled {{
        background-color: {Theme.BG_SECONDARY};
        color: {Theme.TEXT_DISABLED};
    }}

    /* Secondary Action Buttons */
    QPushButton#loadBtn {{
        background-color: {Theme.ACCENT_SECONDARY};
        color: {Theme.TEXT_DARK};
        border: none;
    }}

    QPushButton#loadBtn:hover {{
        background-color: #6dd4ff;
        color: {Theme.TEXT_DARK};
    }}

    QPushButton#loadBtn:disabled {{
        background-color: {Theme.BG_SECONDARY};
        color: {Theme.TEXT_DISABLED};
    }}

    /* Browse button - smaller */
    QPushButton#browseBtn {{
        padding: 2px 8px;
        min-height: 18px;
    }}

    /* ========== Line Edit ========== */
    QLineEdit {{
        background-color: {Theme.BG_INPUT};
        color: {Theme.TEXT_PRIMARY};
        border: 1px solid {Theme.BORDER};
        border-radius: {Theme.RADIUS_INPUT}px;
        padding: 2px 8px;
        min-height: 18px;
        max-height: 18px;
        selection-background-color: {Theme.ACCENT};
        selection-color: {Theme.TEXT_DARK};
    }}

    QLineEdit:focus {{
        border-color: {Theme.ACCENT};
    }}

    QLineEdit:disabled {{
        background-color: {Theme.BG_SECONDARY};
        color: {Theme.TEXT_DISABLED};
    }}

    QLineEdit::placeholder {{
        color: {Theme.TEXT_DISABLED};
    }}

    /* ========== Spin Boxes ========== */
    QSpinBox, QDoubleSpinBox {{
        background-color: {Theme.BG_INPUT};
        color: {Theme.TEXT_PRIMARY};
        border: 1px solid {Theme.BORDER};
        border-radius: {Theme.RADIUS_INPUT}px;
        padding: 2px 6px;
        min-height: 18px;
        max-height: 18px;
        min-width: 80px;
        max-width: 100px;
        font-family: {Theme.FONT_MONO};
    }}

    QSpinBox:focus, QDoubleSpinBox:focus {{
        border-color: {Theme.ACCENT};
    }}

    QSpinBox::up-button, QDoubleSpinBox::up-button {{
        subcontrol-origin: border;
        subcontrol-position: top right;
        width: 16px;
        border-left: 1px solid {Theme.BORDER};
        border-top-right-radius: {Theme.RADIUS_INPUT}px;
        background-color: {Theme.BG_SECONDARY};
    }}

    QSpinBox::up-button:hover, QDoubleSpinBox::up-button:hover {{
        background-color: {Theme.BG_CARD};
    }}

    QSpinBox::down-button, QDoubleSpinBox::down-button {{
        subcontrol-origin: border;
        subcontrol-position: bottom right;
        width: 16px;
        border-left: 1px solid {Theme.BORDER};
        border-bottom-right-radius: {Theme.RADIUS_INPUT}px;
        background-color: {Theme.BG_SECONDARY};
    }}

    QSpinBox::down-button:hover, QDoubleSpinBox::down-button:hover {{
        background-color: {Theme.BG_CARD};
    }}

    QSpinBox::up-arrow, QDoubleSpinBox::up-arrow {{
        image: url({res_path}/arrow_up.svg);
        width: 8px;
        height: 5px;
    }}

    QSpinBox::down-arrow, QDoubleSpinBox::down-arrow {{
        image: url({res_path}/arrow_down.svg);
        width: 8px;
        height: 5px;
    }}

    /* ========== Status Bar ========== */
    QStatusBar {{
        background-color: {Theme.BG_SECONDARY};
        color: {Theme.TEXT_SECONDARY};
        border-top: 1px solid {Theme.BORDER};
        font-size: {Theme.FONT_SIZE_SMALL}px;
        padding: 4px;
    }}

    QStatusBar::item {{
        border: none;
    }}

    /* ========== Splitter ========== */
    QSplitter::handle {{
        background-color: {Theme.BORDER};
        width: 2px;
    }}

    QSplitter::handle:hover {{
        background-color: {Theme.ACCENT};
    }}

    /* ========== Scroll Bars ========== */
    QScrollBar:vertical {{
        background-color: {Theme.BG_INPUT};
        width: 12px;
        border-radius: 6px;
        margin: 2px;
    }}

    QScrollBar::handle:vertical {{
        background-color: {Theme.BORDER};
        border-radius: 5px;
        min-height: 30px;
    }}

    QScrollBar::handle:vertical:hover {{
        background-color: {Theme.ACCENT};
    }}

    QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{
        height: 0px;
    }}

    QScrollBar:horizontal {{
        background-color: {Theme.BG_INPUT};
        height: 12px;
        border-radius: 6px;
        margin: 2px;
    }}

    QScrollBar::handle:horizontal {{
        background-color: {Theme.BORDER};
        border-radius: 5px;
        min-width: 30px;
    }}

    QScrollBar::handle:horizontal:hover {{
        background-color: {Theme.ACCENT};
    }}

    QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {{
        width: 0px;
    }}

    /* ========== Message Box ========== */
    QMessageBox {{
        background-color: {Theme.BG_CARD};
        min-width: 400px;
        min-height: 160px;
    }}

    QMessageBox QLabel {{
        color: {Theme.TEXT_PRIMARY};
        font-size: {Theme.FONT_SIZE_NORMAL}px;
        min-height: 80px;
        padding: 8px;
        line-height: 1.5;
    }}

    QMessageBox QPushButton {{
        min-width: 80px;
        margin: 6px;
    }}

    /* ========== Progress Dialog ========== */
    QProgressDialog {{
        background-color: {Theme.BG_CARD};
        min-width: 350px;
        min-height: 120px;
    }}

    QProgressDialog QLabel {{
        color: {Theme.TEXT_PRIMARY};
        padding: 8px;
        min-height: 30px;
    }}

    QProgressBar {{
        background-color: {Theme.BG_INPUT};
        border: 1px solid {Theme.BORDER};
        border-radius: {Theme.RADIUS_INPUT}px;
        text-align: center;
        color: {Theme.TEXT_PRIMARY};
        min-height: 20px;
    }}

    QProgressBar::chunk {{
        background-color: {Theme.ACCENT};
        border-radius: {Theme.RADIUS_INPUT}px;
    }}

    /* ========== Tool Tips ========== */
    QToolTip {{
        background-color: {Theme.BG_CARD};
        color: {Theme.TEXT_PRIMARY};
        border: 1px solid {Theme.BORDER};
        border-radius: {Theme.RADIUS_INPUT}px;
        padding: 6px;
        font-size: {Theme.FONT_SIZE_SMALL}px;
    }}
    """
