#!/usr/bin/env python3
"""
Simple Folder Plotter - Plot numeric data from .txt files in a folder.
GUI: PyQt5 + matplotlib

Usage:
    python folder_plotter.py [optional_folder_path]
"""
import sys
import os
from pathlib import Path

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QFileDialog, QListWidget, QListWidgetItem,
    QComboBox, QCheckBox, QGroupBox, QLineEdit, QSplitter, QMessageBox
)
from PyQt5.QtCore import Qt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import numpy as np


class FolderPlotter(QMainWindow):
    def __init__(self, folder_path=None):
        super().__init__()
        self.setWindowTitle("Folder & TXT File Plotter")
        self.resize(1100, 700)
        self.folder_path = folder_path
        self.txt_files = []
        self.file_data = {}  # {filename: np.array}

        self._init_ui()
        if folder_path and os.path.isdir(folder_path):
            self.edit_folder.setText(folder_path)
            self.load_folder()

    def _init_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QHBoxLayout(central)

        splitter = QSplitter(Qt.Horizontal)
        main_layout.addWidget(splitter)

        # --- Left panel: file list & controls ---
        left = QWidget()
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(8, 8, 8, 8)

        # Folder selection
        grp_folder = QGroupBox("Folder")
        f_layout = QHBoxLayout(grp_folder)
        self.edit_folder = QLineEdit()
        self.edit_folder.setPlaceholderText("Select folder with .txt files...")
        self.btn_browse = QPushButton("Browse")
        self.btn_browse.setFixedWidth(80)
        self.btn_browse.clicked.connect(self.browse_folder)
        self.btn_load = QPushButton("Load")
        self.btn_load.setFixedWidth(60)
        self.btn_load.clicked.connect(self.load_folder)
        f_layout.addWidget(self.edit_folder, 1)
        f_layout.addWidget(self.btn_browse)
        f_layout.addWidget(self.btn_load)
        left_layout.addWidget(grp_folder)

        # File list
        grp_files = QGroupBox("TXT Files (click to plot)")
        fl_layout = QVBoxLayout(grp_files)
        self.list_files = QListWidget()
        self.list_files.setSelectionMode(QListWidget.ExtendedSelection)
        self.list_files.itemSelectionChanged.connect(self.update_plot)
        fl_layout.addWidget(self.list_files)

        h_sel = QHBoxLayout()
        self.btn_select_all = QPushButton("Select All")
        self.btn_select_all.clicked.connect(self.list_files.selectAll)
        self.btn_clear_sel = QPushButton("Clear")
        self.btn_clear_sel.clicked.connect(self.list_files.clearSelection)
        h_sel.addWidget(self.btn_select_all)
        h_sel.addWidget(self.btn_clear_sel)
        fl_layout.addLayout(h_sel)
        left_layout.addWidget(grp_files, 1)

        # Plot options
        grp_opts = QGroupBox("Plot Options")
        o_layout = QVBoxLayout(grp_opts)

        h1 = QHBoxLayout()
        h1.addWidget(QLabel("Delimiter:"))
        self.combo_delim = QComboBox()
        self.combo_delim.addItems(["Whitespace", "Tab", "Comma", "Semicolon"])
        self.combo_delim.setFixedWidth(100)
        h1.addWidget(self.combo_delim)
        h1.addWidget(QLabel("Comment:"))
        self.edit_comment = QLineEdit("#")
        self.edit_comment.setFixedWidth(50)
        h1.addWidget(self.edit_comment)
        o_layout.addLayout(h1)

        h2 = QHBoxLayout()
        h2.addWidget(QLabel("X column:"))
        self.combo_xcol = QComboBox()
        self.combo_xcol.addItems(["Row index (auto)", "Column 0", "Column 1", "Column 2", "Column 3", "Column 4"])
        self.combo_xcol.setFixedWidth(130)
        h2.addWidget(self.combo_xcol)
        h2.addWidget(QLabel("Y columns:"))
        self.combo_ycol = QComboBox()
        self.combo_ycol.addItems(["All other columns", "Column 0", "Column 1", "Column 2", "Column 3", "Column 4"])
        self.combo_ycol.setFixedWidth(130)
        h2.addWidget(self.combo_ycol)
        o_layout.addLayout(h2)

        h3 = QHBoxLayout()
        self.chk_grid = QCheckBox("Grid")
        self.chk_grid.setChecked(True)
        self.chk_grid.stateChanged.connect(self.update_plot)
        self.chk_legend = QCheckBox("Legend")
        self.chk_legend.setChecked(True)
        self.chk_legend.stateChanged.connect(self.update_plot)
        self.chk_markers = QCheckBox("Markers")
        self.chk_markers.setChecked(False)
        self.chk_markers.stateChanged.connect(self.update_plot)
        h3.addWidget(self.chk_grid)
        h3.addWidget(self.chk_legend)
        h3.addWidget(self.chk_markers)
        o_layout.addLayout(h3)

        h4 = QHBoxLayout()
        self.edit_title = QLineEdit()
        self.edit_title.setPlaceholderText("Plot title")
        self.edit_title.textChanged.connect(self.update_plot)
        h4.addWidget(QLabel("Title:"))
        h4.addWidget(self.edit_title, 1)
        o_layout.addLayout(h4)

        h5 = QHBoxLayout()
        self.edit_xlabel = QLineEdit()
        self.edit_xlabel.setPlaceholderText("X label")
        self.edit_xlabel.textChanged.connect(self.update_plot)
        self.edit_ylabel = QLineEdit()
        self.edit_ylabel.setPlaceholderText("Y label")
        self.edit_ylabel.textChanged.connect(self.update_plot)
        h5.addWidget(QLabel("X:"))
        h5.addWidget(self.edit_xlabel, 1)
        h5.addWidget(QLabel("Y:"))
        h5.addWidget(self.edit_ylabel, 1)
        o_layout.addLayout(h5)

        left_layout.addWidget(grp_opts)
        left_layout.addStretch()

        # Info
        self.lbl_info = QLabel("Load a folder to begin.")
        self.lbl_info.setWordWrap(True)
        self.lbl_info.setStyleSheet("font-size: 11px; color: #666;")
        left_layout.addWidget(self.lbl_info)

        splitter.addWidget(left)

        # --- Right panel: matplotlib canvas ---
        self.figure = Figure(figsize=(7, 5), dpi=100)
        self.canvas = FigureCanvas(self.figure)
        self.canvas.setMinimumSize(500, 400)
        splitter.addWidget(self.canvas)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 3)

        self._update_info()

    def browse_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Folder", self.edit_folder.text())
        if folder:
            self.edit_folder.setText(folder)
            self.load_folder()

    def get_delimiter(self):
        d = self.combo_delim.currentText()
        if d == "Tab":
            return "\t"
        elif d == "Comma":
            return ","
        elif d == "Semicolon":
            return ";"
        return None  # whitespace

    def load_folder(self):
        folder = self.edit_folder.text().strip()
        if not folder or not os.path.isdir(folder):
            QMessageBox.warning(self, "Invalid Folder", f"Folder not found:\n{folder}")
            return
        self.folder_path = folder
        self.txt_files = sorted([
            f for f in os.listdir(folder)
            if f.lower().endswith(".txt") and os.path.isfile(os.path.join(folder, f))
        ])
        self.file_data.clear()
        self.list_files.clear()

        comment_char = self.edit_comment.text().strip() or "#"
        delim = self.get_delimiter()

        for fname in self.txt_files:
            fpath = os.path.join(folder, fname)
            try:
                data = np.loadtxt(fpath, delimiter=delim, comments=comment_char)
                if data.ndim == 1:
                    data = data.reshape(-1, 1)
                self.file_data[fname] = data
                cols = data.shape[1]
                rows = data.shape[0]
                item = QListWidgetItem(f"{fname}  ({rows}x{cols})")
                item.setData(Qt.UserRole, fname)
                self.list_files.addItem(item)
            except Exception as e:
                item = QListWidgetItem(f"{fname}  [ERROR: {e}]")
                item.setData(Qt.UserRole, fname)
                item.setForeground(Qt.red)
                self.list_files.addItem(item)

        self._update_info()
        self.update_plot()

    def _update_info(self):
        if self.txt_files:
            loaded = sum(1 for f in self.file_data.values())
            self.lbl_info.setText(
                f"Folder: {self.folder_path}\n"
                f"Files: {len(self.txt_files)} total, {loaded} loaded successfully"
            )
        else:
            self.lbl_info.setText("No .txt files found. Select a folder.")

    def update_plot(self):
        self.figure.clear()
        ax = self.figure.add_subplot(111)

        selected = self.list_files.selectedItems()
        if not selected:
            ax.text(0.5, 0.5, "Select file(s) to plot", ha="center", va="center",
                    fontsize=14, color="gray", transform=ax.transAxes)
            self.canvas.draw()
            return

        # Determine columns
        xcol_text = self.combo_xcol.currentText()
        ycol_text = self.combo_ycol.currentText()
        use_auto_x = "auto" in xcol_text.lower()

        markers = self.chk_markers.isChecked()

        colors = plt_color_cycle(len(selected))

        for i, item in enumerate(selected):
            fname = item.data(Qt.UserRole)
            if fname not in self.file_data:
                continue
            data = self.file_data[fname]
            color = colors[i % len(colors)]

            if data.shape[1] == 1:
                y = data[:, 0]
                x = np.arange(len(y))
                ax.plot(x, y, label=fname, color=color, marker="." if markers else None)
                continue

            if use_auto_x:
                # Use first column as X, rest as Y
                x = data[:, 0]
                y_cols = data[:, 1:]
            else:
                # Map selection to column index
                x_idx = int(xcol_text.split()[-1])
                x_idx = min(x_idx, data.shape[1] - 1)
                x = data[:, x_idx]

                if "All other" in ycol_text:
                    y_cols = np.delete(data, x_idx, axis=1)
                else:
                    y_idx = int(ycol_text.split()[-1])
                    y_idx = min(y_idx, data.shape[1] - 1)
                    y_cols = data[:, y_idx:y_idx+1]

            for col_idx in range(y_cols.shape[1]):
                label = fname if y_cols.shape[1] == 1 else f"{fname} [col{col_idx}]"
                ax.plot(x, y_cols[:, col_idx], label=label, color=color,
                        marker="." if markers else None, alpha=0.85)

        ax.set_title(self.edit_title.text() or "Plot")
        ax.set_xlabel(self.edit_xlabel.text() or "X")
        ax.set_ylabel(self.edit_ylabel.text() or "Y")
        if self.chk_grid.isChecked():
            ax.grid(True, alpha=0.3)
        if self.chk_legend.isChecked() and ax.get_legend_handles_labels()[1]:
            ax.legend(fontsize=8, loc="best")

        self.figure.tight_layout()
        self.canvas.draw()


def plt_color_cycle(n):
    """Generate n distinct colors."""
    base = [
        "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
        "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
    ]
    if n <= len(base):
        return base[:n]
    # Extend with HSV spread
    import colorsys
    extra = []
    for i in range(n - len(base)):
        h = i / (n - len(base))
        r, g, b = colorsys.hsv_to_rgb(h, 0.7, 0.9)
        extra.append(f"#{int(r*255):02x}{int(g*255):02x}{int(b*255):02x}")
    return base + extra


if __name__ == "__main__":
    app = QApplication(sys.argv)
    folder = sys.argv[1] if len(sys.argv) > 1 else None
    win = FolderPlotter(folder)
    win.show()
    sys.exit(app.exec_())
