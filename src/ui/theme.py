"""
Theme configuration module for Micro Post application.

This module defines color constants, font settings, and size
specifications for the dark tech theme.
"""


class Theme:
    """
    Dark tech theme color and style constants.

    A modern dark theme with cyan/teal accent colors,
    designed for scientific data analysis applications.
    """

    # Background colors
    BG_PRIMARY = "#1e1e2e"      # Main window background
    BG_SECONDARY = "#2d2d44"    # Secondary panels
    BG_CARD = "#363654"         # Card/GroupBox background
    BG_INPUT = "#252538"        # Input field background

    # Accent colors
    ACCENT = "#00d4aa"          # Primary accent (cyan/teal)
    ACCENT_HOVER = "#00f5c4"    # Accent hover state
    ACCENT_SECONDARY = "#4fc3f7"  # Secondary accent (light blue)

    # Status colors
    SUCCESS = "#69db7c"         # Success/positive
    DANGER = "#ff6b6b"          # Error/danger
    WARNING = "#ffd43b"         # Warning

    # Text colors
    TEXT_PRIMARY = "#e0e0e0"    # Primary text
    TEXT_SECONDARY = "#a0a0a0"  # Secondary/muted text
    TEXT_DISABLED = "#666666"   # Disabled text
    TEXT_DARK = "#1e1e2e"       # Dark text (on accent bg)

    # Border colors
    BORDER = "#4a4a6a"          # Default border
    BORDER_FOCUS = "#00d4aa"    # Focused input border

    # Font families
    FONT_FAMILY = "Arial, Segoe UI, sans-serif"
    FONT_MONO = "JetBrains Mono, Consolas, monospace"

    # Font sizes (in pixels)
    FONT_SIZE_TITLE = 14
    FONT_SIZE_GROUP = 12
    FONT_SIZE_NORMAL = 11
    FONT_SIZE_SMALL = 10

    # Border radius (in pixels)
    RADIUS_CARD = 8
    RADIUS_BTN = 6
    RADIUS_INPUT = 4

    # Padding and margins (in pixels)
    PADDING_CARD = 16
    PADDING_BTN = 8
    SPACING = 12

    # Sizes
    BTN_MIN_HEIGHT = 32
    INPUT_MIN_HEIGHT = 28
    SPINBOX_WIDTH = 80
