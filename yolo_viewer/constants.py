"""Shared constants for the YOLO Dual Model Viewer."""

from PyQt5.QtGui import QColor

SUPPORTED_IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}

# Colors per source (RGBA)
SOURCE_COLORS = {
    "modelA": QColor(0, 220, 0),       # green
    "modelB": QColor(30, 144, 255),    # dodger blue
    "manual": QColor(255, 50, 50),     # red
}
SOURCE_LABELS = {
    "modelA": "Model A",
    "modelB": "Model B",
    "manual": "Manual",
}
