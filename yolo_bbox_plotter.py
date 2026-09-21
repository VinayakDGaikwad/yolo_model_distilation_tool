#!/usr/bin/env python3
"""
YOLO Bounding Box Plotter - Display images with YOLO-format bounding boxes overlaid.
Each image has a matching .txt file with lines: class x_center y_center width height (normalized).

Usage:
    python yolo_bbox_plotter.py [optional_folder_path]
"""
import sys
import os
from pathlib import Path

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QFileDialog, QListWidget, QListWidgetItem,
    QGroupBox, QLineEdit, QSplitter, QComboBox, QSpinBox, QCheckBox,
    QMessageBox, QSlider, QDoubleSpinBox
)
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QPixmap, QImage, QPainter, QPen, QBrush, QColor, QFont
import numpy as np

SUPPORTED_IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp", ".gif"}

# Default colors per class (cycles)
CLASS_COLORS = [
    (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0),
    (255, 0, 255), (0, 255, 255), (255, 128, 0), (128, 0, 255),
    (0, 128, 255), (255, 0, 128), (0, 255, 128), (128, 255, 0),
]


def yolo_boxes_to_abs(boxes, img_w, img_h):
    """Convert YOLO normalized (cls, xc, yc, w, h) -> absolute (x1, y1, x2, y2, cls)."""
    result = []
    for b in boxes:
        if len(b) < 5:
            continue
        cls = int(b[0])
        xc, yc, w, h = b[1], b[2], b[3], b[4]
        x1 = int((xc - w / 2) * img_w)
        y1 = int((yc - h / 2) * img_h)
        x2 = int((xc + w / 2) * img_w)
        y2 = int((yc + h / 2) * img_h)
        x1 = max(0, min(x1, img_w - 1))
        y1 = max(0, min(y1, img_h - 1))
        x2 = max(0, min(x2, img_w - 1))
        y2 = max(0, min(y2, img_h - 1))
        result.append((cls, x1, y1, x2, y2))
    return result


def load_yolo_txt(txt_path):
    """Load YOLO format txt. Returns list of [cls, xc, yc, w, h]."""
    boxes = []
    if not os.path.isfile(txt_path):
        return boxes
    with open(txt_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) >= 5:
                try:
                    cls = float(parts[0])
                    vals = [float(x) for x in parts[1:5]]
                    boxes.append([cls] + vals)
                except ValueError:
                    continue
    return boxes


def draw_boxes_on_image(img_path, boxes, img_w_override=None, img_h_override=None):
    """Draw YOLO boxes on image, return QPixmap."""
    pixmap = QPixmap(img_path)
    if pixmap.isNull():
        return None

    img_w = pixmap.width()
    img_h = pixmap.height()

    abs_boxes = yolo_boxes_to_abs(boxes, img_w, img_h)

    painter = QPainter(pixmap)
    painter.setRenderHint(QPainter.Antialiasing)

    for cls, x1, y1, x2, y2 in abs_boxes:
        color = CLASS_COLORS[cls % len(CLASS_COLORS)]
        pen = QPen(QColor(*color), 2)
        painter.setPen(pen)
        painter.setBrush(QBrush(QColor(*color, 40)))
        painter.drawRect(x1, y1, x2 - x1, y2 - y1)

        # Label background
        label = str(cls)
        font = QFont("Arial", 10, QFont.Bold)
        painter.setFont(font)
        fm = painter.fontMetrics()
        tw = fm.horizontalAdvance(label) + 6
        th = fm.height() + 2
        lx = x1
        ly = y1 - th if y1 - th > 0 else y1
        painter.fillRect(lx, ly, tw, th, QColor(*color, 200))
        painter.setPen(QColor(255, 255, 255))
        painter.drawText(lx + 3, ly + fm.ascent() + 1, label)

    painter.end()
    return pixmap


class YOLOBBoxPlotter(QMainWindow):
    def __init__(self, folder_path=None):
        super().__init__()
        self.setWindowTitle("YOLO Bounding Box Plotter")
        self.resize(1200, 800)
        self.folder_path = folder_path
        self.image_files = []
        self.current_idx = -1
        self.image_cache = {}  # path -> (pixmap, box_count)

        self._init_ui()
        if folder_path and os.path.isdir(folder_path):
            self.edit_folder.setText(folder_path)
            self.load_folder()

    def _init_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QHBoxLayout(central)
        main_layout.setContentsMargins(6, 6, 6, 6)

        splitter = QSplitter(Qt.Horizontal)
        main_layout.addWidget(splitter)

        # --- Left panel: file list & controls ---
        left = QWidget()
        left.setFixedWidth(300)
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(6, 6, 6, 6)

        # Folder
        grp_folder = QGroupBox("Image Folder")
        fb = QHBoxLayout(grp_folder)
        self.edit_folder = QLineEdit()
        self.edit_folder.setPlaceholderText("Folder with images + .txt")
        self.btn_browse = QPushButton("Browse")
        self.btn_browse.setFixedWidth(70)
        self.btn_browse.clicked.connect(self.browse_folder)
        self.btn_load = QPushButton("Load")
        self.btn_load.setFixedWidth(55)
        self.btn_load.clicked.connect(self.load_folder)
        fb.addWidget(self.edit_folder, 1)
        fb.addWidget(self.btn_browse)
        fb.addWidget(self.btn_load)
        left_layout.addWidget(grp_folder)

        # File list
        grp_files = QGroupBox("Images")
        fl = QVBoxLayout(grp_files)
        self.list_files = QListWidget()
        self.list_files.currentRowChanged.connect(self.on_select_image)
        fl.addWidget(self.list_files)

        nav = QHBoxLayout()
        self.btn_prev = QPushButton("< Prev")
        self.btn_prev.clicked.connect(self.prev_image)
        self.btn_next = QPushButton("Next >")
        self.btn_next.clicked.connect(self.next_image)
        nav.addWidget(self.btn_prev)
        nav.addWidget(self.btn_next)
        fl.addLayout(nav)
        left_layout.addWidget(grp_files, 1)

        # Display options
        grp_opts = QGroupBox("Display Options")
        ol = QVBoxLayout(grp_opts)

        h1 = QHBoxLayout()
        h1.addWidget(QLabel("Box width:"))
        self.spin_lw = QSpinBox()
        self.spin_lw.setRange(1, 8)
        self.spin_lw.setValue(2)
        self.spin_lw.valueChanged.connect(self.refresh_image)
        h1.addWidget(self.spin_lw)
        self.chk_labels = QCheckBox("Labels")
        self.chk_labels.setChecked(True)
        self.chk_labels.stateChanged.connect(self.refresh_image)
        h1.addWidget(self.chk_labels)
        self.chk_fill = QCheckBox("Fill")
        self.chk_fill.setChecked(True)
        self.chk_fill.stateChanged.connect(self.refresh_image)
        h1.addWidget(self.chk_fill)
        ol.addLayout(h1)

        h2 = QHBoxLayout()
        h2.addWidget(QLabel("Scale:"))
        self.slider_scale = QSlider(Qt.Horizontal)
        self.slider_scale.setRange(25, 300)
        self.slider_scale.setValue(100)
        self.slider_scale.valueChanged.connect(self.refresh_image)
        h2.addWidget(self.slider_scale)
        self.lbl_scale = QLabel("100%")
        self.lbl_scale.setFixedWidth(40)
        self.slider_scale.valueChanged.connect(lambda v: self.lbl_scale.setText(f"{v}%"))
        h2.addWidget(self.lbl_scale)
        ol.addLayout(h2)

        left_layout.addWidget(grp_opts)

        # Info
        self.lbl_info = QLabel("Load a folder to begin.")
        self.lbl_info.setWordWrap(True)
        self.lbl_info.setStyleSheet("font-size: 11px; color: #666;")
        left_layout.addWidget(self.lbl_info)

        splitter.addWidget(left)

        # --- Right: image view ---
        right = QWidget()
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(0, 0, 0, 0)

        self.lbl_filename = QLabel("No image loaded")
        self.lbl_filename.setStyleSheet("font-weight: bold; font-size: 13px; padding: 4px;")
        right_layout.addWidget(self.lbl_filename)

        self.lbl_counts = QLabel("")
        self.lbl_counts.setStyleSheet("font-size: 11px; color: #444; padding: 2px 4px;")
        right_layout.addWidget(self.lbl_counts)

        self.lbl_image = QLabel()
        self.lbl_image.setAlignment(Qt.AlignCenter)
        self.lbl_image.setStyleSheet("background: #2d2d2d;")
        self.lbl_image.setMinimumSize(400, 300)
        right_layout.addWidget(self.lbl_image, 1)

        # Navigation slider
        self.slider_nav = QSlider(Qt.Horizontal)
        self.slider_nav.setMinimum(0)
        self.slider_nav.setMaximum(0)
        self.slider_nav.valueChanged.connect(self.goto_image)
        right_layout.addWidget(self.slider_nav)

        splitter.addWidget(right)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)

    def browse_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Folder", self.edit_folder.text())
        if folder:
            self.edit_folder.setText(folder)
            self.load_folder()

    def load_folder(self):
        folder = self.edit_folder.text().strip()
        if not folder or not os.path.isdir(folder):
            QMessageBox.warning(self, "Invalid Folder", f"Folder not found:\n{folder}")
            return

        self.folder_path = folder
        self.image_files = []
        self.image_cache.clear()

        for f in sorted(os.listdir(folder)):
            if Path(f).suffix.lower() in SUPPORTED_IMG_EXTS:
                self.image_files.append(f)

        self.list_files.clear()
        for i, fname in enumerate(self.image_files):
            txt_path = os.path.join(folder, Path(fname).stem + ".txt")
            has_txt = os.path.isfile(txt_path)
            if has_txt:
                boxes = load_yolo_txt(txt_path)
                item = QListWidgetItem(f"{fname}  [{len(boxes)} boxes]")
            else:
                item = QListWidgetItem(f"{fname}  [no txt]")
                item.setForeground(Qt.gray)
            item.setData(Qt.UserRole, i)
            self.list_files.addItem(item)

        self.slider_nav.setMaximum(max(0, len(self.image_files) - 1))
        self.slider_nav.setValue(0)

        self.lbl_info.setText(
            f"Folder: {folder}\n"
            f"Images: {len(self.image_files)}\n"
            f"Paired .txt files: {sum(1 for f in self.image_files if os.path.isfile(os.path.join(folder, Path(f).stem + '.txt')))}"
        )

        if self.image_files:
            self.list_files.setCurrentRow(0)
        else:
            self.lbl_filename.setText("No images found")
            self.lbl_image.clear()

    def on_select_image(self, row):
        if row < 0 or row >= len(self.image_files):
            return
        self.current_idx = row
        self.slider_nav.blockSignals(True)
        self.slider_nav.setValue(row)
        self.slider_nav.blockSignals(False)
        self.refresh_image()

    def goto_image(self, idx):
        if idx != self.current_idx and 0 <= idx < len(self.image_files):
            self.list_files.setCurrentRow(idx)

    def prev_image(self):
        if self.current_idx > 0:
            self.list_files.setCurrentRow(self.current_idx - 1)

    def next_image(self):
        if self.current_idx < len(self.image_files) - 1:
            self.list_files.setCurrentRow(self.current_idx + 1)

    def refresh_image(self):
        if self.current_idx < 0 or self.current_idx >= len(self.image_files):
            return

        fname = self.image_files[self.current_idx]
        img_path = os.path.join(self.folder_path, fname)
        txt_path = os.path.join(self.folder_path, Path(fname).stem + ".txt")

        boxes = load_yolo_txt(txt_path)
        pixmap = draw_boxes_on_image(img_path, boxes)

        if pixmap is None:
            self.lbl_image.setText(f"Failed to load: {fname}")
            self.lbl_filename.setText(fname)
            return

        # Scale
        scale = self.slider_scale.value() / 100.0
        if abs(scale - 1.0) > 0.01:
            pw = int(pixmap.width() * scale)
            ph = int(pixmap.height() * scale)
            pixmap = pixmap.scaled(pw, ph, Qt.KeepAspectRatio, Qt.SmoothTransformation)

        self.lbl_image.setPixmap(pixmap)
        self.lbl_filename.setText(f"{fname}  ({Path(txt_path).name})")

        # Box count info
        class_counts = {}
        for b in boxes:
            c = int(b[0])
            class_counts[c] = class_counts.get(c, 0) + 1
        parts = [f"class {c}: {n}" for c, n in sorted(class_counts.items())]
        summary = f"Total: {len(boxes)} boxes"
        if parts:
            summary += "  |  " + ", ".join(parts)
        self.lbl_counts.setText(summary)

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Left:
            self.prev_image()
        elif event.key() == Qt.Key_Right:
            self.next_image()
        else:
            super().keyPressEvent(event)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    folder = sys.argv[1] if len(sys.argv) > 1 else None
    win = YOLOBBoxPlotter(folder)
    win.show()
    sys.exit(app.exec_())
