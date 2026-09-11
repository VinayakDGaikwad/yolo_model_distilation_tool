#!/usr/bin/env python3
"""
YOLO Dual Model Viewer & Annotator - PyQt5 GUI
------------------------------------------------
Features:
- Load 2 YOLO models (ultralytics YOLO .pt/.onnx)
- Browse image folder, navigate images (Prev/Next, slider, list)
- Per-model confidence slider (0.05 - 0.95) + MaxDet spinbox (1..1000, default 300+)
- Per-model NMS IoU slider
- Run inference, overlay boxes with colors per model
- Filter/sort boxes, toggle visibility per source
- Draw new boxes (click-drag), assign class, move/delete boxes
- Box list table with selection sync, delete, edit class
- Zoom/Pan (Ctrl+Wheel, Middle-drag, Fit)
- Save YOLO format txt (normalized xywh) - current or all images

Requirements:
    pip install PyQt5 Pillow ultralytics

    # ultralytics brings opencv, torch etc.
    # If ultralytics not installed, GUI still runs in "manual annotate only" mock mode.

Usage:
    python yolo_dual_viewer.py
    python yolo_dual_viewer.py --modelA /path/to/best.pt --modelB /path/to/best2.pt --images /path/to/imgs

Author: Muse Spark
"""
import sys
import os
import json
import glob
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel, QPushButton, QSlider, QSpinBox, QDoubleSpinBox,
    QVBoxLayout, QHBoxLayout, QGridLayout, QFileDialog, QGroupBox, QCheckBox,
    QListWidget, QListWidgetItem, QGraphicsView, QGraphicsScene, QGraphicsRectItem,
    QGraphicsPixmapItem, QGraphicsTextItem, QGraphicsItem, QComboBox, QLineEdit,
    QTableWidget, QTableWidgetItem, QHeaderView, QMessageBox, QSplitter, QProgressBar,
    QShortcut, QFrame, QScrollArea, QMenu, QAction, QInputDialog, QToolButton
)
from PyQt5.QtGui import (
    QPixmap, QImage, QPen, QBrush, QColor, QFont, QPainter, QKeySequence, QCursor
)
from PyQt5.QtCore import Qt, QRectF, QPointF, pyqtSignal, QTimer, QSize, QItemSelectionModel

try:
    from PIL import Image
except ImportError:
    Image = None

# Try ultralytics
try:
    from ultralytics import YOLO
    ULTRALYTICS_AVAILABLE = True
except ImportError:
    YOLO = None
    ULTRALYTICS_AVAILABLE = False

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
    "manual": "Manual"
}

@dataclass
class Box:
    x1: float
    y1: float
    x2: float
    y2: float
    cls: int
    conf: float
    source: str  # modelA, modelB, manual
    label: str = ""

    def to_yolo(self, img_w, img_h):
        x_center = ((self.x1 + self.x2) / 2.0) / img_w
        y_center = ((self.y1 + self.y2) / 2.0) / img_h
        w = (self.x2 - self.x1) / img_w
        h = (self.y2 - self.y1) / img_h
        # clamp
        x_center = min(max(x_center, 0), 1)
        y_center = min(max(y_center, 0), 1)
        w = min(max(w, 0), 1)
        h = min(max(h, 0), 1)
        return self.cls, x_center, y_center, w, h


class YoloWrapper:
    """Thin wrapper around ultralytics YOLO. Handles load & predict."""
    def __init__(self, model_path: str = ""):
        self.model_path = model_path
        self.model = None
        self.class_names: Dict[int, str] = {}
        self.load_error = ""

    def load(self, path: str) -> bool:
        self.model_path = path
        self.model = None
        self.class_names = {}
        self.load_error = ""
        if not path or not os.path.exists(path):
            self.load_error = f"File not found: {path}"
            return False
        if not ULTRALYTICS_AVAILABLE:
            self.load_error = "ultralytics not installed (pip install ultralytics)"
            return False
        try:
            self.model = YOLO(path)
            # get names
            try:
                names = self.model.names  # dict
                if isinstance(names, dict):
                    self.class_names = {int(k): str(v) for k, v in names.items()}
                elif isinstance(names, list):
                    self.class_names = {i: str(n) for i, n in enumerate(names)}
                else:
                    self.class_names = {0: "object"}
            except Exception:
                self.class_names = {0: "object"}
            return True
        except Exception as e:
            self.load_error = str(e)
            self.model = None
            return False

    def predict(self, image_path: str, conf: float = 0.25, iou: float = 0.45, max_det: int = 300) -> List[Box]:
        if self.model is None:
            return []
        try:
            # ultralytics predict: verbose False to suppress logs
            results = self.model.predict(source=image_path, conf=conf, iou=iou, max_det=max_det, verbose=False)
            boxes: List[Box] = []
            for r in results:
                if r.boxes is None:
                    continue
                # r.boxes.xyxy, .cls, .conf
                xyxy = r.boxes.xyxy.cpu().numpy() if hasattr(r.boxes.xyxy, "cpu") else r.boxes.xyxy
                cls = r.boxes.cls.cpu().numpy() if hasattr(r.boxes.cls, "cpu") else r.boxes.cls
                confs = r.boxes.conf.cpu().numpy() if hasattr(r.boxes.conf, "cpu") else r.boxes.conf
                for (x1, y1, x2, y2), c, cf in zip(xyxy, cls, confs):
                    c = int(c)
                    label = self.class_names.get(c, str(c))
                    boxes.append(Box(float(x1), float(y1), float(x2), float(y2), c, float(cf), source="tmp", label=label))
            return boxes
        except Exception as e:
            print(f"[YoloWrapper] predict error: {e}")
            return []

    def get_names_list(self) -> List[str]:
        if not self.class_names:
            return []
        max_id = max(self.class_names.keys())
        lst = [self.class_names.get(i, f"class_{i}") for i in range(max_id+1)]
        return lst


# ---------- Graphics ---------

class BoxItem(QGraphicsRectItem):
    """Selectable/movable rect with label. Handles resizing via handles? MVP: movable, selectable, delete."""
    def __init__(self, box: Box, img_w: int, img_h: int, pen_width: float = 1.4):
        # rect in scene coords
        super().__init__(QRectF(box.x1, box.y1, box.x2 - box.x1, box.y2 - box.y1))
        self.box = box
        self.img_w = img_w
        self.img_h = img_h
        color = SOURCE_COLORS.get(box.source, QColor(255, 0, 0))
        pen = QPen(color, pen_width)
        pen.setCosmetic(True)  # keep constant thickness when zooming
        self.setPen(pen)
        # semi-transparent fill for manual, lighter for preds
        fill = QColor(color)
        fill.setAlpha(35 if box.source == "manual" else 20)
        self.setBrush(QBrush(fill))
        self.setFlags(QGraphicsItem.ItemIsSelectable | QGraphicsItem.ItemIsMovable)
        self.setAcceptHoverEvents(True)
        # label - disabled per user request: don't display labels at all in implot
        # Keep item for compatibility but hidden; no text/background drawn to avoid clutter (>300 boxes)
        self.label_item = QGraphicsTextItem(self)
        txt = f"{box.label or box.cls} {box.conf:.2f}" if box.source != "manual" else f"{box.label or box.cls} (manual)"
        self.label_item.setPlainText(txt)
        self.label_item.setDefaultTextColor(QColor(255, 255, 255))
        font = QFont("Arial", 6, QFont.Bold)
        font.setStyleStrategy(QFont.PreferAntialias)
        self.label_item.setFont(font)
        self.label_item.setPos(self.rect().topLeft() - QPointF(0, 10) if self.rect().top() > 10 else self.rect().topLeft())
        self.label_item.setVisible(False)  # hidden - no implot labels
        self.label_item.hide()
        self._hover = False

    def paint(self, painter: QPainter, option, widget=None):
        # Draw rect only - labels hidden
        super().paint(painter, option, widget)
        # Label background/text intentionally not drawn (disabled)

    def update_label(self):
        # No-op - labels hidden; keep text updated silently if ever re-enabled
        try:
            txt = f"{self.box.label or self.box.cls} {self.box.conf:.2f}" if self.box.source != "manual" else f"{self.box.label or self.box.cls} (manual)"
            self.label_item.setPlainText(txt)
        except Exception:
            pass

    def set_selected_appearance(self, selected: bool):
        if selected:
            pen = self.pen()
            pen.setWidthF(2.2)
            pen.setColor(QColor(255, 255, 0))
            self.setPen(pen)
            c = pen.color()
            fill = QColor(c)
            fill.setAlpha(60)
            # keep brush?
        else:
            color = SOURCE_COLORS.get(self.box.source, QColor(255, 0, 0))
            pen = QPen(color, 1.4)
            pen.setCosmetic(True)
            self.setPen(pen)

    def get_box_coords(self) -> Tuple[float, float, float, float]:
        # current rect + item position (movable). QGraphicsRectItem with IsMovable moves the item's pos, not rect.
        # So effective rect = rect.translated(pos)
        r = self.rect()
        p = self.pos()
        x1 = r.x() + p.x()
        y1 = r.y() + p.y()
        x2 = x1 + r.width()
        y2 = y1 + r.height()
        # clamp to image bounds
        x1 = max(0, min(x1, self.img_w))
        y1 = max(0, min(y1, self.img_h))
        x2 = max(0, min(x2, self.img_w))
        y2 = max(0, min(y2, self.img_h))
        return x1, y1, x2, y2

    def sync_box(self):
        x1, y1, x2, y2 = self.get_box_coords()
        self.box.x1, self.box.y1, self.box.x2, self.box.y2 = x1, y1, x2, y2

    def contextMenuEvent(self, event):
        # Right-click on a single box: offer Delete / Edit Class / Duplicate
        menu = QMenu()
        del_act = menu.addAction("🗑️ Delete this box (Del)")
        edit_act = menu.addAction("✏️ Edit class")
        dup_act = menu.addAction("⎘ Duplicate")
        # Color hint
        del_act.setIconText("Delete")
        chosen = menu.exec_(event.screenPos())
        scene = self.scene()
        # scene is AnnotScene, parent MainWindow reachable via scene.parent() or view
        # Find MainWindow via widget hierarchy
        if chosen == del_act:
            # Need to find MainWindow to push undo
            # walk up via scene -> view -> window
            # scene.views() gives list
            views = scene.views()
            if views:
                win = views[0].window()
                if hasattr(win, "delete_single_item"):
                    win.delete_single_item(self)
                else:
                    scene.remove_box_item = getattr(scene, "remove_box_item", None)
                    if scene.remove_box_item:
                        scene.removeItem(self)
            else:
                scene.removeItem(self)
        elif chosen == edit_act:
            views = scene.views()
            if views:
                win = views[0].window()
                if hasattr(win, "_edit_class_for_item"):
                    win._edit_class_for_item(self)
        elif chosen == dup_act:
            views = scene.views()
            if views:
                win = views[0].window()
                if hasattr(win, "duplicate_item"):
                    win.duplicate_item(self)


class AnnotScene(QGraphicsScene):
    sig_box_created = pyqtSignal(float, float, float, float)  # x1,y1,x2,y2 in image pix
    sig_click_empty = pyqtSignal()
    sig_selection_changed = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.image_item: Optional[QGraphicsPixmapItem] = None
        self.img_w = 0
        self.img_h = 0
        self.draw_mode = False
        self.drawing = False
        self.draw_start: QPointF = QPointF()
        self.temp_rect: Optional[QGraphicsRectItem] = None
        self.box_items: List[BoxItem] = []
        self._pixmap: Optional[QPixmap] = None

    def set_image(self, pixmap: QPixmap):
        self.clear()
        self.box_items.clear()
        self._pixmap = pixmap
        self.img_w = pixmap.width()
        self.img_h = pixmap.height()
        self.setSceneRect(0, 0, self.img_w, self.img_h)
        self.image_item = QGraphicsPixmapItem(pixmap)
        self.image_item.setZValue(-100)
        self.addItem(self.image_item)
        self.temp_rect = None
        self.drawing = False

    def set_draw_mode(self, enabled: bool):
        self.draw_mode = enabled
        if enabled:
            # make existing boxes not movable while drawing? keep selectable
            pass

    def add_box_item(self, box: Box) -> BoxItem:
        item = BoxItem(box, self.img_w, self.img_h)
        self.addItem(item)
        self.box_items.append(item)
        return item

    def remove_box_item(self, item: BoxItem):
        if item in self.box_items:
            self.box_items.remove(item)
        self.removeItem(item)

    def clear_boxes(self, source_filter: Optional[str] = None):
        to_remove = []
        for it in self.box_items:
            if source_filter is None or it.box.source == source_filter:
                to_remove.append(it)
        for it in to_remove:
            self.remove_box_item(it)

    def get_all_boxes(self) -> List[Box]:
        out = []
        for it in self.box_items:
            it.sync_box()
            out.append(it.box)
        return out

    # Mouse handling for drawing
    def mousePressEvent(self, event):
        if self.draw_mode and event.button() == Qt.LeftButton:
            pos = event.scenePos()
            # clamp
            pos.setX(max(0, min(pos.x(), self.img_w)))
            pos.setY(max(0, min(pos.y(), self.img_h)))
            self.drawing = True
            self.draw_start = pos
            if self.temp_rect:
                self.removeItem(self.temp_rect)
                self.temp_rect = None
            self.temp_rect = QGraphicsRectItem(QRectF(pos, pos))
            pen = QPen(SOURCE_COLORS["manual"], 2, Qt.DashLine)
            pen.setCosmetic(True)
            self.temp_rect.setPen(pen)
            self.temp_rect.setBrush(QBrush(QColor(255, 50, 50, 40)))
            self.addItem(self.temp_rect)
            event.accept()
            return
        # allow selection
        super().mousePressEvent(event)
        # if clicked empty, deselect
        items = self.items(event.scenePos())
        # items includes temp_rect etc. Check if only image
        has_box = any(isinstance(i, BoxItem) for i in items)
        if not has_box and not self.drawing:
            self.sig_click_empty.emit()

    def mouseMoveEvent(self, event):
        if self.drawing and self.draw_mode and self.temp_rect:
            pos = event.scenePos()
            pos.setX(max(0, min(pos.x(), self.img_w)))
            pos.setY(max(0, min(pos.y(), self.img_h)))
            rect = QRectF(self.draw_start, pos).normalized()
            self.temp_rect.setRect(rect)
            event.accept()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if self.drawing and self.draw_mode and event.button() == Qt.LeftButton and self.temp_rect:
            pos = event.scenePos()
            pos.setX(max(0, min(pos.x(), self.img_w)))
            pos.setY(max(0, min(pos.y(), self.img_h)))
            rect = QRectF(self.draw_start, pos).normalized()
            self.removeItem(self.temp_rect)
            self.temp_rect = None
            self.drawing = False
            # validate size
            if rect.width() >= 5 and rect.height() >= 5:
                self.sig_box_created.emit(rect.x(), rect.y(), rect.x() + rect.width(), rect.y() + rect.height())
            else:
                # too small ignore
                pass
            event.accept()
            return
        super().mouseReleaseEvent(event)
        # sync positions after move
        for it in self.box_items:
            if it.isSelected():
                it.sync_box()
        self.sig_selection_changed.emit()


class AnnotView(QGraphicsView):
    sig_zoom_changed = pyqtSignal(float)

    def __init__(self, scene: AnnotScene, parent=None):
        super().__init__(scene, parent)
        self.setRenderHints(QPainter.Antialiasing | QPainter.SmoothPixmapTransform)
        self.setDragMode(QGraphicsView.RubberBandDrag)
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.AnchorUnderMouse)
        self.setViewportUpdateMode(QGraphicsView.FullViewportUpdate)
        self.setBackgroundBrush(QBrush(QColor(45, 45, 45)))
        self._zoom = 1.0
        self._panning = False
        self._pan_start = QPointF()
        self._space_pressed = False

    def fit_image(self):
        if self.scene():
            self.fitInView(self.scene().sceneRect(), Qt.KeepAspectRatio)
            # compute zoom factor approx
            self._zoom = self.transform().m11()

    def wheelEvent(self, event):
        if event.modifiers() & Qt.ControlModifier:
            # zoom
            angle = event.angleDelta().y()
            factor = 1.15 if angle > 0 else 1/1.15
            self.scale(factor, factor)
            self._zoom *= factor
            self.sig_zoom_changed.emit(self._zoom)
            event.accept()
        else:
            super().wheelEvent(event)

    def mousePressEvent(self, event):
        scene = self.scene()
        if isinstance(scene, AnnotScene) and scene.draw_mode:
            # in draw mode, don't pan, forward to scene
            super().mousePressEvent(event)
            return
        if event.button() == Qt.MiddleButton or (event.button() == Qt.LeftButton and self._space_pressed):
            self._panning = True
            self._pan_start = event.pos()
            self.setCursor(Qt.ClosedHandCursor)
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self._panning:
            delta = event.pos() - self._pan_start
            self._pan_start = event.pos()
            self.horizontalScrollBar().setValue(self.horizontalScrollBar().value() - delta.x())
            self.verticalScrollBar().setValue(self.verticalScrollBar().value() - delta.y())
            event.accept()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if self._panning and event.button() in (Qt.MiddleButton, Qt.LeftButton):
            self._panning = False
            self.setCursor(Qt.ArrowCursor)
            event.accept()
            return
        super().mouseReleaseEvent(event)

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Space:
            self._space_pressed = True
            self.setCursor(Qt.OpenHandCursor)
        super().keyPressEvent(event)

    def keyReleaseEvent(self, event):
        if event.key() == Qt.Key_Space:
            self._space_pressed = False
            self.setCursor(Qt.ArrowCursor)
        super().keyReleaseEvent(event)


# ---------- Main Window ----------

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("YOLO Dual Model Viewer - Multi-box Annotator (Conf Slider | MaxDet >300 | Draw | YOLO Export)")
        # --- Responsive window sizing: fit available screen, avoid out-of-screen ---
        # Use 92% of available geometry, centered, with sensible minimum
        try:
            scr = QApplication.primaryScreen().availableGeometry() if QApplication.instance() and QApplication.primaryScreen() else None
        except Exception:
            scr = None
        if scr is None:
            # Fallback for offscreen tests
            sw, sh = 1920, 1080
        else:
            sw, sh = scr.width(), scr.height()
        # Target 1600x980 but clamp to screen
        target_w, target_h = 1600, 980
        w = min(target_w, int(sw * 0.93))
        h = min(target_h, int(sh * 0.90))
        # Ensure not too small
        w = max(w, 1100)
        h = max(h, 650)
        self.resize(w, h)
        self.setMinimumSize(1050, 620)
        # Center on screen
        try:
            if scr:
                self.move(scr.center() - self.rect().center())
        except Exception:
            pass
        # Auto-maximize on small screens (<1400x900)
        self._should_maximize = (sw < 1400 or sh < 860)

        self.wrapperA = YoloWrapper()
        self.wrapperB = YoloWrapper()

        self.image_paths: List[str] = []
        self.current_idx: int = -1
        # current image dimensions
        self.current_img_w = 0
        self.current_img_h = 0
        self.current_pixmap: Optional[QPixmap] = None
        self.current_image_path: str = ""

        # store manual boxes per image (so switching images preserves manual draws until saved)
        self.manual_boxes_cache: Dict[str, List[Box]] = {}
        # last predictions per model (unfiltered raw would be re-run, but we store last filtered)
        self.last_boxes_A: List[Box] = []
        self.last_boxes_B: List[Box] = []

        # class handling
        self.class_names_combined: List[str] = ["object"]
        self.next_class_id = 0
        # undo stack for deletions: list of {"boxes": List[Box], "desc": str}
        self.undo_stack: List[Dict] = []
        self._context_scene_pos: Optional[QPointF] = None
        # per-image save tracking: image_path -> bool saved (combined export)
        self.saved_status: Dict[str, bool] = {}

        self._init_ui()
        self._connect()
        self._update_ui_state()
        self.statusBar().showMessage("Ready. Load models and image folder. " + ("Ultralytics ready." if ULTRALYTICS_AVAILABLE else "Ultralytics NOT installed - manual mode only. pip install ultralytics"))

        # auto-load test image folder if exists? Try multiple known locations
        for test_dir in [
            "/home/trendzlink/Downloads/tz_batch_test_03_9_2025_020139/tz_batch_test_03_9_2025",
            "/home/trendzlink/yolo_tool/tz_batch_test_03_9_2025_020139/tz_batch_test_03_9_2025",
            "/home/trendzlink/yolo_tool/images",
            "./images",
        ]:
            if os.path.isdir(test_dir):
                self.image_dir_edit.setText(test_dir)
                self.load_image_dir(test_dir)
                break
        # Ensure window fits screen after UI built; maximize on small screens
        if getattr(self, "_should_maximize", False):
            QTimer.singleShot(100, self._apply_initial_maximize)

    def _apply_initial_maximize(self):
        try:
            # If still larger than available, maximize; also ensure centered
            scr = QApplication.primaryScreen().availableGeometry() if QApplication.primaryScreen() else None
            if scr and (self.width() > scr.width() * 0.95 or self.height() > scr.height() * 0.95):
                self.showMaximized()
            else:
                # Ensure fully visible (workaround for window manager placing partially off-screen)
                self.move(max(scr.x(), min(self.x(), scr.x() + scr.width() - self.width())),
                          max(scr.y(), min(self.y(), scr.y() + scr.height() - self.height())))
        except Exception:
            pass

    def fit_window_to_screen(self):
        """Manual action: resize and center to 92% of available screen, or toggle maximize."""
        try:
            scr = QApplication.primaryScreen().availableGeometry()
            if self.isMaximized():
                self.showNormal()
                # Shrink to 85% and center
                w = int(scr.width() * 0.85)
                h = int(scr.height() * 0.85)
                self.resize(w, h)
                self.move(scr.center() - self.rect().center())
            else:
                # If already normal but still large, maximize; else fit
                if self.width() >= scr.width() * 0.9 or self.height() >= scr.height() * 0.9:
                    self.showMaximized()
                else:
                    w = min(1600, int(scr.width() * 0.92))
                    h = min(980, int(scr.height() * 0.90))
                    self.resize(w, h)
                    self.move(scr.center() - self.rect().center())
            self.statusBar().showMessage(f"Window {self.width()}x{self.height()} fitted to screen {scr.width()}x{scr.height()}", 3000)
        except Exception as e:
            self.log(f"fit_window_to_screen failed: {e}")

    def _init_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        main_h = QHBoxLayout(central)
        main_h.setContentsMargins(6, 6, 6, 6)
        main_h.setSpacing(6)

        # Left control panel - scrollable & responsive (was FixedWidth 380 → out of screen on small displays)
        left_panel = QWidget()
        left_panel.setMinimumWidth(300)
        left_panel.setMaximumWidth(420)
        # Allow vertical scrolling when window height small
        left_panel.setSizePolicy(left_panel.sizePolicy().horizontalPolicy(), left_panel.sizePolicy().verticalPolicy())
        left_layout = QVBoxLayout(left_panel)
        left_layout.setSpacing(8)
        left_layout.setContentsMargins(4, 4, 4, 4)

        # --- Model A group ---
        gA = QGroupBox("Model A")
        gA.setStyleSheet("QGroupBox { font-weight: bold; }")
        vA = QVBoxLayout(gA)
        # path row
        hA1 = QHBoxLayout()
        self.modelA_edit = QLineEdit()
        self.modelA_edit.setPlaceholderText("Path to modelA .pt / .onnx")
        self.btn_browseA = QPushButton("Browse")
        self.btn_browseA.setFixedWidth(70)
        hA1.addWidget(self.modelA_edit, 1)
        hA1.addWidget(self.btn_browseA)
        vA.addLayout(hA1)
        hA2 = QHBoxLayout()
        self.btn_loadA = QPushButton("Load A")
        self.lbl_statusA = QLabel("Not loaded")
        self.lbl_statusA.setStyleSheet("color: #888; font-size: 11px;")
        self.chk_showA = QCheckBox("Show")
        self.chk_showA.setChecked(True)
        hA2.addWidget(self.btn_loadA)
        hA2.addWidget(self.chk_showA)
        hA2.addWidget(self.lbl_statusA, 1)
        vA.addLayout(hA2)
        # conf slider A
        hA3 = QHBoxLayout()
        hA3.addWidget(QLabel("Conf:"))
        self.slider_confA = QSlider(Qt.Horizontal)
        self.slider_confA.setRange(1, 95)
        self.slider_confA.setValue(25)
        self.spin_confA = QDoubleSpinBox()
        self.spin_confA.setRange(0.01, 0.95)
        self.spin_confA.setSingleStep(0.05)
        self.spin_confA.setValue(0.25)
        self.spin_confA.setFixedWidth(65)
        hA3.addWidget(self.slider_confA, 1)
        hA3.addWidget(self.spin_confA)
        vA.addLayout(hA3)
        # IoU + MaxDet
        hA4 = QHBoxLayout()
        hA4.addWidget(QLabel("IoU:"))
        self.spin_iouA = QDoubleSpinBox()
        self.spin_iouA.setRange(0.1, 0.95)
        self.spin_iouA.setSingleStep(0.05)
        self.spin_iouA.setValue(0.45)
        self.spin_iouA.setFixedWidth(65)
        hA4.addWidget(self.spin_iouA)
        hA4.addWidget(QLabel("MaxDet:"))
        self.spin_maxA = QSpinBox()
        self.spin_maxA.setRange(1, 2000)
        self.spin_maxA.setValue(400)
        self.spin_maxA.setFixedWidth(70)
        self.spin_maxA.setToolTip("Max predictions per image (supports >300)")
        hA4.addWidget(self.spin_maxA)
        hA4.addStretch()
        vA.addLayout(hA4)
        # color indicator
        hA5 = QHBoxLayout()
        hA5.addWidget(QLabel("Color:"))
        lblColorA = QLabel("   ")
        lblColorA.setFixedSize(18, 18)
        lblColorA.setStyleSheet(f"background: {SOURCE_COLORS['modelA'].name()}; border: 1px solid #333;")
        hA5.addWidget(lblColorA)
        hA5.addStretch()
        hA5.addWidget(QLabel("Class names:"))
        self.lbl_classesA = QLabel("-")
        self.lbl_classesA.setStyleSheet("font-size: 10px; color: #555;")
        hA5.addWidget(self.lbl_classesA, 1)
        vA.addLayout(hA5)
        left_layout.addWidget(gA)

        # Model B group
        gB = QGroupBox("Model B")
        gB.setStyleSheet("QGroupBox { font-weight: bold; }")
        vB = QVBoxLayout(gB)
        hB1 = QHBoxLayout()
        self.modelB_edit = QLineEdit()
        self.modelB_edit.setPlaceholderText("Path to modelB .pt / .onnx (optional)")
        self.btn_browseB = QPushButton("Browse")
        self.btn_browseB.setFixedWidth(70)
        hB1.addWidget(self.modelB_edit, 1)
        hB1.addWidget(self.btn_browseB)
        vB.addLayout(hB1)
        hB2 = QHBoxLayout()
        self.btn_loadB = QPushButton("Load B")
        self.lbl_statusB = QLabel("Not loaded")
        self.lbl_statusB.setStyleSheet("color: #888; font-size: 11px;")
        self.chk_showB = QCheckBox("Show")
        self.chk_showB.setChecked(True)
        hB2.addWidget(self.btn_loadB)
        hB2.addWidget(self.chk_showB)
        hB2.addWidget(self.lbl_statusB, 1)
        vB.addLayout(hB2)
        hB3 = QHBoxLayout()
        hB3.addWidget(QLabel("Conf:"))
        self.slider_confB = QSlider(Qt.Horizontal)
        self.slider_confB.setRange(1, 95)
        self.slider_confB.setValue(25)
        self.spin_confB = QDoubleSpinBox()
        self.spin_confB.setRange(0.01, 0.95)
        self.spin_confB.setSingleStep(0.05)
        self.spin_confB.setValue(0.25)
        self.spin_confB.setFixedWidth(65)
        hB3.addWidget(self.slider_confB, 1)
        hB3.addWidget(self.spin_confB)
        vB.addLayout(hB3)
        hB4 = QHBoxLayout()
        hB4.addWidget(QLabel("IoU:"))
        self.spin_iouB = QDoubleSpinBox()
        self.spin_iouB.setRange(0.1, 0.95)
        self.spin_iouB.setSingleStep(0.05)
        self.spin_iouB.setValue(0.45)
        self.spin_iouB.setFixedWidth(65)
        hB4.addWidget(self.spin_iouB)
        hB4.addWidget(QLabel("MaxDet:"))
        self.spin_maxB = QSpinBox()
        self.spin_maxB.setRange(1, 2000)
        self.spin_maxB.setValue(400)
        self.spin_maxB.setFixedWidth(70)
        self.spin_maxB.setToolTip("Max predictions per image (supports >300)")
        hB4.addWidget(self.spin_maxB)
        hB4.addStretch()
        vB.addLayout(hB4)
        hB5 = QHBoxLayout()
        hB5.addWidget(QLabel("Color:"))
        lblColorB = QLabel("   ")
        lblColorB.setFixedSize(18, 18)
        lblColorB.setStyleSheet(f"background: {SOURCE_COLORS['modelB'].name()}; border: 1px solid #333;")
        hB5.addWidget(lblColorB)
        hB5.addStretch()
        hB5.addWidget(QLabel("Classes:"))
        self.lbl_classesB = QLabel("-")
        self.lbl_classesB.setStyleSheet("font-size: 10px; color: #555;")
        hB5.addWidget(self.lbl_classesB, 1)
        vB.addLayout(hB5)
        left_layout.addWidget(gB)

        # Images group
        gImg = QGroupBox("Images")
        vImg = QVBoxLayout(gImg)
        hImg1 = QHBoxLayout()
        self.image_dir_edit = QLineEdit()
        self.image_dir_edit.setPlaceholderText("Image folder")
        self.btn_browseImg = QPushButton("Browse")
        self.btn_browseImg.setFixedWidth(70)
        hImg1.addWidget(self.image_dir_edit, 1)
        hImg1.addWidget(self.btn_browseImg)
        vImg.addLayout(hImg1)
        hImg2 = QHBoxLayout()
        self.btn_prev = QPushButton("◀ Prev")
        self.btn_next = QPushButton("Next ▶")
        self.btn_rerun = QPushButton("⟳ Re-run")
        self.btn_rerun.setToolTip("Re-run inference with current sliders")
        hImg2.addWidget(self.btn_prev)
        hImg2.addWidget(self.btn_next)
        hImg2.addWidget(self.btn_rerun)
        vImg.addLayout(hImg2)
        hImg3 = QHBoxLayout()
        self.lbl_imgInfo = QLabel("No images")
        self.lbl_imgInfo.setStyleSheet("font-size: 11px; color: #333;")
        self.lbl_zoom = QLabel("100%")
        self.lbl_zoom.setFixedWidth(55)
        self.lbl_zoom.setAlignment(Qt.AlignRight)
        hImg3.addWidget(self.lbl_imgInfo, 1)
        hImg3.addWidget(self.lbl_zoom)
        vImg.addLayout(hImg3)
        # image list
        self.list_images = QListWidget()
        self.list_images.setFixedHeight(120)
        vImg.addWidget(self.list_images)
        # nav slider
        self.slider_img = QSlider(Qt.Horizontal)
        self.slider_img.setEnabled(False)
        vImg.addWidget(self.slider_img)
        left_layout.addWidget(gImg)

        # Draw / Edit group
        gDraw = QGroupBox("Draw & Edit (Manual)")
        vDraw = QVBoxLayout(gDraw)
        # class selection for manual
        hD1 = QHBoxLayout()
        hD1.addWidget(QLabel("Class ID:"))
        self.spin_manual_cls = QSpinBox()
        self.spin_manual_cls.setRange(0, 80)
        self.spin_manual_cls.setValue(0)
        self.spin_manual_cls.setFixedWidth(60)
        hD1.addWidget(self.spin_manual_cls)
        hD1.addWidget(QLabel("Name:"))
        self.edit_manual_name = QLineEdit("object")
        self.edit_manual_name.setFixedWidth(90)
        hD1.addWidget(self.edit_manual_name)
        # class combo from model
        self.combo_classes = QComboBox()
        self.combo_classes.setEditable(False)
        self.combo_classes.addItem("object (0)")
        self.combo_classes.setFixedWidth(110)
        hD1.addWidget(self.combo_classes)
        vDraw.addLayout(hD1)
        hD2 = QHBoxLayout()
        self.btn_draw = QPushButton("✏️ Draw Box (D)")
        self.btn_draw.setCheckable(True)
        self.btn_draw.setStyleSheet("QPushButton:checked { background: #ffcccc; font-weight: bold; }")
        self.btn_draw.setToolTip("Toggle draw mode - click-drag on image to draw. Cursor will be crosshair.")
        self.chk_showManual = QCheckBox("Show Manual")
        self.chk_showManual.setChecked(True)
        hD2.addWidget(self.btn_draw, 1)
        hD2.addWidget(self.chk_showManual)
        vDraw.addLayout(hD2)
        # Row: Delete Selected (prominent) + Delete Filtered + Undo
        hD3 = QHBoxLayout()
        self.btn_delete_sel = QPushButton("🗑️ Delete Selected")
        self.btn_delete_sel.setToolTip("Delete selected box(es) - also Del/Backspace, Right-click → Delete")
        self.btn_delete_sel.setStyleSheet("background: #f8d7da; font-weight: bold;")
        self.btn_delete_sel.setFixedHeight(28)
        self.btn_delete_filtered = QPushButton("Delete Filtered")
        self.btn_delete_filtered.setToolTip("Delete all boxes matching current filter (Src/Class) in table")
        self.btn_delete_filtered.setFixedHeight(28)
        self.btn_undo = QPushButton("↩ Undo")
        self.btn_undo.setToolTip("Undo last deletion (Ctrl+Z)")
        self.btn_undo.setFixedHeight(28)
        self.btn_undo.setEnabled(False)
        hD3.addWidget(self.btn_delete_sel, 2)
        hD3.addWidget(self.btn_delete_filtered, 1)
        hD3.addWidget(self.btn_undo, 1)
        vDraw.addLayout(hD3)
        # Row: Clear by source
        hD3b = QHBoxLayout()
        self.btn_clear_manual = QPushButton("Clear Manual")
        self.btn_clear_manual.setToolTip("Remove ALL manual boxes for current image")
        self.btn_clear_manual.setStyleSheet("background: #ffe0e0;")
        self.btn_clear_A = QPushButton("Clear A")
        self.btn_clear_A.setToolTip("Remove all Model A predictions for current image")
        self.btn_clear_A.setStyleSheet("background: #e0ffe0;")
        self.btn_clear_B = QPushButton("Clear B")
        self.btn_clear_B.setToolTip("Remove all Model B predictions for current image")
        self.btn_clear_B.setStyleSheet("background: #e0eaff;")
        hD3b.addWidget(self.btn_clear_manual)
        hD3b.addWidget(self.btn_clear_A)
        hD3b.addWidget(self.btn_clear_B)
        vDraw.addLayout(hD3b)
        # Row: Clear combined / All / low-conf
        hD3c = QHBoxLayout()
        self.btn_clear_preds = QPushButton("Clear Preds (A+B)")
        self.btn_clear_preds.setToolTip("Remove all prediction boxes (A+B) for current image (Re-run to restore)")
        self.btn_clear_all = QPushButton("Clear ALL")
        self.btn_clear_all.setToolTip("Remove every box (manual + preds) for current image")
        self.btn_clear_all.setStyleSheet("background: #ffcccc; font-weight: bold; border: 1px solid #c00;")
        self.btn_delete_lowconf = QPushButton("Del <Conf")
        self.btn_delete_lowconf.setToolTip("Delete boxes with confidence below threshold (prompts)")
        hD3c.addWidget(self.btn_clear_preds, 1)
        hD3c.addWidget(self.btn_clear_all, 1)
        hD3c.addWidget(self.btn_delete_lowconf, 1)
        vDraw.addLayout(hD3c)
        hD4 = QHBoxLayout()
        hD4.addWidget(QLabel("Info:"))
        self.lbl_drawInfo = QLabel("Draw: OFF")
        self.lbl_drawInfo.setStyleSheet("font-size: 11px; color: #666;")
        hD4.addWidget(self.lbl_drawInfo, 1)
        vDraw.addLayout(hD4)
        # delete hints
        lblDelHints = QLabel(
            "<b>Delete options:</b> Click box → <b>Del</b> / <b>Backspace</b> | "
            "Right-click box → <i>Delete this box</i> | "
            "Right-click image → menu (Delete Selected/Filtered/All) | "
            "Table <i>✕</i> column or multi-select → Delete Selected | "
            "<b>Ctrl+Z</b> Undo"
        )
        lblDelHints.setWordWrap(True)
        lblDelHints.setStyleSheet("font-size: 9px; color: #444; background: #fff8dc; padding: 4px; border: 1px solid #eee;")
        vDraw.addWidget(lblDelHints)
        # general hints
        lblHints = QLabel("Tips: Select box → move with mouse. Ctrl+Wheel zoom. Middle-drag pan. Space+Drag pan. Ctrl+A select all.")
        lblHints.setWordWrap(True)
        lblHints.setStyleSheet("font-size: 9px; color: #777;")
        vDraw.addWidget(lblHints)
        left_layout.addWidget(gDraw)

        # Save group - Combined export, per-image options
        gSave = QGroupBox("Export Combined YOLO (A+B+Manual) — Per-Image")
        gSave.setStyleSheet("QGroupBox { font-weight: bold; color: #1a5c1a; }")
        vSave = QVBoxLayout(gSave)
        # Combined mode selector
        hS0 = QHBoxLayout()
        hS0.addWidget(QLabel("Export Mode:"))
        self.cmb_save_mode = QComboBox()
        self.cmb_save_mode.addItems([
            "Combined (A+B+Manual)",
            "Visible only (respects Show toggles)",
            "Model A + Manual",
            "Model B + Manual",
            "Manual only",
            "Model A only",
            "Model B only",
        ])
        self.cmb_save_mode.setCurrentIndex(0)
        self.cmb_save_mode.setToolTip("Combined = merges ALL models' predictions + manual boxes for selected image(s). \nPer-image files: <image_name>.txt in save dir.")
        hS0.addWidget(self.cmb_save_mode, 1)
        vSave.addLayout(hS0)
        # Dedup / NMS across models
        hS0b = QHBoxLayout()
        self.chk_dedup = QCheckBox("Merge duplicates (NMS across models)")
        self.chk_dedup.setToolTip("When Combined, run NMS across Model A+B boxes to suppress duplicate overlapping detections. Keeps higher-conf box.")
        self.chk_dedup.setChecked(False)
        hS0b.addWidget(self.chk_dedup, 1)
        hS0b.addWidget(QLabel("IoU:"))
        self.spin_dedup_iou = QDoubleSpinBox()
        self.spin_dedup_iou.setRange(0.1, 0.95)
        self.spin_dedup_iou.setSingleStep(0.05)
        self.spin_dedup_iou.setValue(0.5)
        self.spin_dedup_iou.setFixedWidth(60)
        self.spin_dedup_iou.setEnabled(False)
        self.chk_dedup.toggled.connect(self.spin_dedup_iou.setEnabled)
        hS0b.addWidget(self.spin_dedup_iou)
        vSave.addLayout(hS0b)
        # save dir
        hS1 = QHBoxLayout()
        self.save_dir_edit = QLineEdit()
        self.save_dir_edit.setPlaceholderText("Save dir (default: image folder — one .txt per image)")
        self.btn_browseSave = QPushButton("Browse")
        self.btn_browseSave.setFixedWidth(70)
        hS1.addWidget(self.save_dir_edit, 1)
        hS1.addWidget(self.btn_browseSave)
        vSave.addLayout(hS1)
        hS2 = QHBoxLayout()
        self.chk_save_include_preds = QCheckBox("Include predictions")
        self.chk_save_include_preds.setChecked(True)
        self.chk_save_include_preds.setToolTip("If off, only manual boxes saved (still Combined mode respects selection)")
        self.chk_save_normalize = QCheckBox("YOLO normalized")
        self.chk_save_normalize.setChecked(True)
        self.chk_save_normalize.setEnabled(False)
        self.chk_hide_on_export = QCheckBox("Ignore hidden (Show toggle)")
        self.chk_hide_on_export.setChecked(False)
        self.chk_hide_on_export.setToolTip("If checked, respects Show A/B/Manual toggles even in Combined mode. Unchecked = export ALL boxes regardless of Show.")
        hS2.addWidget(self.chk_save_include_preds)
        hS2.addWidget(self.chk_save_normalize)
        vSave.addLayout(hS2)
        hS2b = QHBoxLayout()
        hS2b.addWidget(self.chk_hide_on_export)
        hS2b.addStretch()
        vSave.addLayout(hS2b)
        # primary save buttons - per-image
        hS3 = QHBoxLayout()
        self.btn_save = QPushButton("💾 Save Current (Combined)")
        self.btn_save.setStyleSheet("font-weight: bold; background: #d4edda; border: 1px solid #28a745;")
        self.btn_save.setToolTip("Save COMBINED (A+B+Manual) YOLO for CURRENT image only → <save_dir>/<image>.txt (one txt per image)")
        self.btn_save.setFixedHeight(32)
        self.btn_saveAs = QPushButton("Save As...")
        self.btn_saveAs.setToolTip("Choose exact file/path for CURRENT image's combined YOLO (Save As dialog per image)")
        self.btn_saveAs.setFixedHeight(32)
        hS3.addWidget(self.btn_save, 3)
        hS3.addWidget(self.btn_saveAs, 1)
        vSave.addLayout(hS3)
        # bulk per-image
        hS3b = QHBoxLayout()
        self.btn_saveAll = QPushButton("💾 Save ALL (Combined, Per-Image Files)")
        self.btn_saveAll.setStyleSheet("font-weight: bold; background: #cce5ff; border: 1px solid #007bff;")
        self.btn_saveAll.setToolTip("Save COMBINED YOLO for EVERY image → one .txt per image in save dir (re-runs inference with current sliders, merges A+B+Manual, per-file)")
        self.btn_saveAll.setFixedHeight(32)
        self.btn_export_per_image = QPushButton("Per-Image Picker...")
        self.btn_export_per_image.setToolTip("Open dialog to select which images to export (checklist) — combined per-file saves")
        self.btn_export_per_image.setFixedHeight(32)
        hS3b.addWidget(self.btn_saveAll, 3)
        hS3b.addWidget(self.btn_export_per_image, 1)
        vSave.addLayout(hS3b)
        # info label
        lblExportInfo = QLabel("Mode <b>Combined</b> merges Model A (green) + Model B (blue) + Manual (red) into <b>one YOLO .txt per image</b> (<image_name>.txt). Each image gets its own file — no single merged file.")
        lblExportInfo.setWordWrap(True)
        lblExportInfo.setStyleSheet("font-size: 9px; color: #444; background: #e8f5e9; padding: 4px; border: 1px solid #c8e6c9;")
        vSave.addWidget(lblExportInfo)
        self.lbl_saveStatus = QLabel("")
        self.lbl_saveStatus.setStyleSheet("font-size: 11px; color: #2e7d32;")
        self.lbl_saveStatus.setWordWrap(True)
        vSave.addWidget(self.lbl_saveStatus)
        # per-image status + classes
        hS4 = QHBoxLayout()
        self.btn_export_classes = QPushButton("Export classes.txt")
        self.btn_export_classes.setFixedHeight(24)
        self.lbl_per_image_status = QLabel("0/0 saved")
        self.lbl_per_image_status.setStyleSheet("font-size: 10px; color: #666;")
        self.lbl_per_image_status.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        hS4.addWidget(self.btn_export_classes)
        hS4.addWidget(self.lbl_per_image_status, 1)
        vSave.addLayout(hS4)

        left_layout.addWidget(gSave)
        left_layout.addStretch()

        # status for ultralytics
        if not ULTRALYTICS_AVAILABLE:
            lblWarn = QLabel("⚠️ ultralytics not installed\npip install ultralytics\nManual mode only.")
            lblWarn.setStyleSheet("color: #b85c00; background: #fff3cd; padding: 6px; border: 1px solid #ffeeba; font-size: 11px;")
            lblWarn.setWordWrap(True)
            left_layout.addWidget(lblWarn)

        # Center view + Right table
        center_widget = QWidget()
        center_layout = QVBoxLayout(center_widget)
        center_layout.setContentsMargins(0, 0, 0, 0)
        center_layout.setSpacing(4)

        # Top info bar
        topBar = QHBoxLayout()
        self.lbl_filename = QLabel("No image loaded")
        self.lbl_filename.setStyleSheet("font-weight: bold; font-size: 13px;")
        topBar.addWidget(self.lbl_filename, 1)
        self.btn_fit = QPushButton("Fit")
        self.btn_fit.setToolTip("Fit image to view (F)")
        self.btn_fit.setFixedWidth(50)
        self.btn_zoomIn = QPushButton("Zoom +")
        self.btn_zoomIn.setFixedWidth(70)
        self.btn_zoomOut = QPushButton("Zoom -")
        self.btn_zoomOut.setFixedWidth(70)
        self.btn_fitWindow = QPushButton("Fit Window")
        self.btn_fitWindow.setToolTip("Fit window to screen (Ctrl+F / F11) — fixes out-of-screen / oversized window")
        self.btn_fitWindow.setFixedWidth(85)
        self.btn_fitWindow.setStyleSheet("background: #fff3cd; border: 1px solid #ffeeba;")
        topBar.addWidget(self.btn_fit)
        topBar.addWidget(self.btn_zoomIn)
        topBar.addWidget(self.btn_zoomOut)
        topBar.addWidget(self.btn_fitWindow)
        center_layout.addLayout(topBar)

        # Graphics view - reduced minimum to allow window to shrink
        self.scene = AnnotScene()
        self.view = AnnotView(self.scene)
        self.view.setMinimumSize(420, 320)
        self.view.setSizePolicy(self.view.sizePolicy().horizontalPolicy(), self.view.sizePolicy().verticalPolicy())
        center_layout.addWidget(self.view, 1)

        # Bottom status: counts
        bottomBar = QHBoxLayout()
        self.lbl_counts = QLabel("ModelA: 0 | ModelB: 0 | Manual: 0 | Total: 0")
        self.lbl_counts.setStyleSheet("font-size: 11px; color: #444;")
        bottomBar.addWidget(self.lbl_counts, 1)
        self.progress = QProgressBar()
        self.progress.setFixedWidth(180)
        self.progress.setVisible(False)
        bottomBar.addWidget(self.progress)
        center_layout.addLayout(bottomBar)

        # Right table panel - responsive (was FixedWidth 360)
        right_panel = QWidget()
        right_panel.setMinimumWidth(300)
        right_panel.setMaximumWidth(420)
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(4, 4, 4, 4)
        right_layout.setSpacing(6)
        gTable = QGroupBox("Detections (click to select)")
        vTable = QVBoxLayout(gTable)
        # filter row
        hF = QHBoxLayout()
        hF.addWidget(QLabel("Filter:"))
        self.combo_filter_source = QComboBox()
        self.combo_filter_source.addItems(["All", "Model A", "Model B", "Manual"])
        self.combo_filter_source.setFixedWidth(100)
        hF.addWidget(self.combo_filter_source)
        self.edit_filter_cls = QLineEdit()
        self.edit_filter_cls.setPlaceholderText("class filter")
        self.edit_filter_cls.setFixedWidth(90)
        hF.addWidget(self.edit_filter_cls)
        self.btn_filter = QPushButton("Apply")
        self.btn_filter.setFixedWidth(60)
        hF.addWidget(self.btn_filter)
        vTable.addLayout(hF)
        # table
        self.table = QTableWidget(0, 6)
        self.table.setHorizontalHeaderLabels(["Src", "Class", "Conf", "xyxy", "Area", "Del"])
        header = self.table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.Fixed)
        header.setSectionResizeMode(1, QHeaderView.Stretch)
        header.setSectionResizeMode(2, QHeaderView.Fixed)
        header.setSectionResizeMode(3, QHeaderView.Stretch)
        header.setSectionResizeMode(4, QHeaderView.Fixed)
        header.setSectionResizeMode(5, QHeaderView.Fixed)
        self.table.setColumnWidth(0, 65)
        self.table.setColumnWidth(2, 55)
        self.table.setColumnWidth(4, 55)
        self.table.setColumnWidth(5, 40)
        self.table.setSelectionBehavior(QTableWidget.SelectRows)
        self.table.setSelectionMode(QTableWidget.ExtendedSelection)
        self.table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.table.verticalHeader().setVisible(False)
        self.table.setAlternatingRowColors(True)
        self.table.setFixedHeight(360)
        self.table.setContextMenuPolicy(Qt.CustomContextMenu)
        self.table.setToolTip("Click row to select box (Ctrl+Click multi-select, Shift+Click range). Click ✕ to delete. Right-click for menu (Delete, Edit Class). Double-click Class to edit.")
        vTable.addWidget(self.table)
        # table actions
        hTA = QHBoxLayout()
        self.btn_select_all = QPushButton("Select All")
        self.btn_select_all.setToolTip("Ctrl+A")
        self.btn_deselect = QPushButton("Deselect")
        self.btn_delete_table_sel = QPushButton("🗑️ Delete Selected Rows")
        self.btn_delete_table_sel.setToolTip("Delete all selected rows in table (Del)")
        self.btn_delete_table_sel.setStyleSheet("background: #f8d7da; font-weight: bold;")
        hTA.addWidget(self.btn_select_all)
        hTA.addWidget(self.btn_deselect)
        hTA.addWidget(self.btn_delete_table_sel, 1)
        vTable.addLayout(hTA)
        # extra table hint
        lblTableHint = QLabel("Right-click table row → Delete / Edit Class. Multi-select with Ctrl/Shift, then Delete. Double-click Class to edit.")
        lblTableHint.setWordWrap(True)
        lblTableHint.setStyleSheet("font-size: 9px; color: #666;")
        vTable.addWidget(lblTableHint)
        right_layout.addWidget(gTable)

        # Stats / logs
        gLog = QGroupBox("Log")
        vLog = QVBoxLayout(gLog)
        self.log_list = QListWidget()
        self.log_list.setFixedHeight(180)
        vLog.addWidget(self.log_list)
        right_layout.addWidget(gLog)

        # Help box
        gHelp = QGroupBox("Shortcuts")
        vHelp = QVBoxLayout(gHelp)
        helpText = QLabel(
            "D: Draw mode | Right-click: Delete menu\n"
            "Del/Backspace: Delete selected (multi-select OK)\n"
            "Ctrl+Z: Undo delete | Ctrl+A: Select all\n"
            "Table ✕ or Right-click → Delete\n"
            "Clear Manual/A/B/All | Delete Filtered/Low-conf\n"
            "←/→ Prev/Next | S / Ctrl+S Save\n"
            "Ctrl+R Re-run | F Fit | Ctrl+Wheel Zoom"
        )
        helpText.setStyleSheet("font-size: 11px; color: #555;")
        vHelp.addWidget(helpText)
        right_layout.addWidget(gHelp)
        right_layout.addStretch()

        # --- Responsive layout: side panels become scrollable so window fits small screens ---
        # Wrap side panels in QScrollArea (vertical scroll when height < content)
        left_scroll = QScrollArea()
        left_scroll.setWidgetResizable(True)
        left_scroll.setWidget(left_panel)
        left_scroll.setMinimumWidth(320)
        left_scroll.setMaximumWidth(440)
        left_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        left_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        left_scroll.setFrameShape(QFrame.NoFrame)

        right_scroll = QScrollArea()
        right_scroll.setWidgetResizable(True)
        right_scroll.setWidget(right_panel)
        right_scroll.setMinimumWidth(320)
        right_scroll.setMaximumWidth(440)
        right_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        right_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        right_scroll.setFrameShape(QFrame.NoFrame)

        # Splitter for center+right (now with right_scroll)
        splitter_mid = QSplitter(Qt.Horizontal)
        splitter_mid.addWidget(center_widget)
        splitter_mid.addWidget(right_scroll)
        # Give center more stretch
        splitter_mid.setStretchFactor(0, 3)
        splitter_mid.setStretchFactor(1, 1)
        splitter_mid.setSizes([900, 360])

        # Outer splitter: left_scroll vs (center+right)
        outer_splitter = QSplitter(Qt.Horizontal)
        outer_splitter.addWidget(left_scroll)
        outer_splitter.addWidget(splitter_mid)
        outer_splitter.setStretchFactor(0, 0)
        outer_splitter.setStretchFactor(1, 1)
        # Initial sizes adapt to current window width (which is already clamped to screen)
        # Use 28% left, 72% rest
        try:
            tw = self.width()
            outer_splitter.setSizes([int(tw * 0.28), int(tw * 0.72)])
        except Exception:
            outer_splitter.setSizes([380, 1100])

        main_h.addWidget(outer_splitter, 1)

        # --- Menu bar for window fix (discoverable) ---
        try:
            menubar = self.menuBar()
            mWin = menubar.addMenu("&Window")
            actFitWin = QAction("Fit Window to Screen (Ctrl+F)", self)
            actFitWin.setShortcut(QKeySequence("Ctrl+F"))
            actFitWin.triggered.connect(self.fit_window_to_screen)
            mWin.addAction(actFitWin)
            actMax = QAction("Toggle Maximize (F11)", self)
            actMax.setShortcut(QKeySequence("F11"))
            actMax.triggered.connect(self.fit_window_to_screen)
            mWin.addAction(actMax)
            mView = menubar.addMenu("View")
            actFitImg = QAction("Fit Image to View (F)", self)
            actFitImg.setShortcut(QKeySequence("F"))
            actFitImg.triggered.connect(self.view.fit_image)
            mView.addAction(actFitImg)
        except Exception:
            pass

    def _connect(self):
        # browse
        self.btn_browseA.clicked.connect(lambda: self.browse_model("A"))
        self.btn_browseB.clicked.connect(lambda: self.browse_model("B"))
        self.btn_browseImg.clicked.connect(self.browse_image_dir)
        self.btn_browseSave.clicked.connect(self.browse_save_dir)
        self.btn_loadA.clicked.connect(lambda: self.load_model("A"))
        self.btn_loadB.clicked.connect(lambda: self.load_model("B"))
        # model edits -> update label?
        self.modelA_edit.textChanged.connect(lambda: self._update_load_btn_state())
        self.modelB_edit.textChanged.connect(lambda: self._update_load_btn_state())
        # sliders sync
        self.slider_confA.valueChanged.connect(self._on_confA_slider)
        self.spin_confA.valueChanged.connect(self._on_confA_spin)
        self.slider_confB.valueChanged.connect(self._on_confB_slider)
        self.spin_confB.valueChanged.connect(self._on_confB_spin)
        # iou / max det -> rerun auto with debounce
        self.spin_iouA.valueChanged.connect(self._schedule_rerun)
        self.spin_iouB.valueChanged.connect(self._schedule_rerun)
        self.spin_maxA.valueChanged.connect(self._schedule_rerun)
        self.spin_maxB.valueChanged.connect(self._schedule_rerun)
        # debounce timer
        self.rerun_timer = QTimer()
        self.rerun_timer.setSingleShot(True)
        self.rerun_timer.setInterval(400)
        self.rerun_timer.timeout.connect(self.rerun_inference)

        # navigation
        self.btn_prev.clicked.connect(self.prev_image)
        self.btn_next.clicked.connect(self.next_image)
        self.btn_rerun.clicked.connect(self.rerun_inference)
        self.list_images.currentRowChanged.connect(self._on_image_list_change)
        self.slider_img.valueChanged.connect(self._on_img_slider)

        # view
        self.scene.sig_box_created.connect(self._on_box_drawn)
        self.scene.sig_click_empty.connect(self._on_click_empty)
        self.scene.sig_selection_changed.connect(self._on_scene_selection)
        self.scene.selectionChanged.connect(self._on_scene_selection)
        self.view.sig_zoom_changed.connect(lambda z: self.lbl_zoom.setText(f"{int(z*100)}%"))
        self.btn_fit.clicked.connect(self.view.fit_image)
        self.btn_zoomIn.clicked.connect(lambda: self.view.scale(1.2, 1.2))
        self.btn_zoomOut.clicked.connect(lambda: self.view.scale(1/1.2, 1/1.2))
        self.btn_fitWindow.clicked.connect(self.fit_window_to_screen)

        # draw mode
        self.btn_draw.toggled.connect(self._on_draw_toggle)
        self.spin_manual_cls.valueChanged.connect(self._on_manual_cls_change)
        self.edit_manual_name.textChanged.connect(self._on_manual_name_change)
        self.combo_classes.currentIndexChanged.connect(self._on_combo_cls_change)
        self.chk_showA.toggled.connect(self._on_visibility_toggle)
        self.chk_showB.toggled.connect(self._on_visibility_toggle)
        self.chk_showManual.toggled.connect(self._on_visibility_toggle)
        self.btn_delete_sel.clicked.connect(self.delete_selected)
        self.btn_delete_filtered.clicked.connect(self.delete_filtered)
        self.btn_undo.clicked.connect(self.undo_last)
        self.btn_clear_manual.clicked.connect(lambda: self.clear_boxes("manual"))
        self.btn_clear_A.clicked.connect(lambda: self.clear_boxes("modelA"))
        self.btn_clear_B.clicked.connect(lambda: self.clear_boxes("modelB"))
        self.btn_clear_preds.clicked.connect(lambda: self.clear_boxes("preds"))
        self.btn_clear_all.clicked.connect(lambda: self.clear_boxes("all"))
        self.btn_delete_lowconf.clicked.connect(self.delete_low_conf)
        self.btn_delete_table_sel.clicked.connect(self.delete_table_selected_rows)
        # view context menu (right-click on image)
        self.view.setContextMenuPolicy(Qt.CustomContextMenu)
        self.view.customContextMenuRequested.connect(self.show_view_context_menu)
        # table context menu
        self.table.customContextMenuRequested.connect(self.show_table_context_menu)

        # filter
        self.btn_filter.clicked.connect(self.refresh_table)
        self.combo_filter_source.currentIndexChanged.connect(self.refresh_table)
        self.edit_filter_cls.textChanged.connect(self.refresh_table)
        self.table.itemSelectionChanged.connect(self._on_table_select)
        self.btn_select_all.clicked.connect(lambda: self.table.selectAll())
        self.btn_deselect.clicked.connect(lambda: self.table.clearSelection())

        # image list context menu for per-image save
        self.list_images.setContextMenuPolicy(Qt.CustomContextMenu)
        self.list_images.customContextMenuRequested.connect(self.show_image_list_context_menu)

        # save
        self.btn_save.clicked.connect(self.save_current)
        self.btn_saveAs.clicked.connect(self.save_current_as)
        self.btn_saveAll.clicked.connect(self.save_all)
        self.btn_export_per_image.clicked.connect(self.export_per_image_picker)
        self.btn_export_classes.clicked.connect(self.export_classes)
        self.cmb_save_mode.currentIndexChanged.connect(self._on_save_mode_changed)
        self.chk_save_include_preds.toggled.connect(self._on_save_mode_changed)

        # shortcuts
        QShortcut(QKeySequence("D"), self, activated=lambda: self.btn_draw.toggle())
        QShortcut(QKeySequence("Delete"), self, activated=self.delete_selected)
        QShortcut(QKeySequence("Backspace"), self, activated=self.delete_selected)
        QShortcut(QKeySequence("Ctrl+Z"), self, activated=self.undo_last)
        QShortcut(QKeySequence("Ctrl+A"), self, activated=lambda: self.table.selectAll())
        QShortcut(QKeySequence("Left"), self, activated=self.prev_image)
        QShortcut(QKeySequence("A"), self, activated=self.prev_image)
        QShortcut(QKeySequence("Right"), self, activated=self.next_image)
        # removed duplicate D->next to avoid conflict with draw toggle; use E for next alternative
        QShortcut(QKeySequence("E"), self, activated=self.next_image)
        QShortcut(QKeySequence("S"), self, activated=self.save_current)
        QShortcut(QKeySequence("Ctrl+S"), self, activated=self.save_current)
        QShortcut(QKeySequence("Ctrl+R"), self, activated=self.rerun_inference)
        QShortcut(QKeySequence("F"), self, activated=self.view.fit_image)
        QShortcut(QKeySequence("Ctrl+F"), self, activated=self.fit_window_to_screen)
        QShortcut(QKeySequence("F11"), self, activated=self.fit_window_to_screen)

        # make Right arrow alternative (D conflicts with draw) keep separate
        # Actually D is draw toggle, also next image would conflict. So override: use E for next etc.
        # Let's keep as is but draw toggle has priority via button?

    def _update_load_btn_state(self):
        pass

    def log(self, msg: str):
        from datetime import datetime
        ts = datetime.now().strftime("%H:%M:%S")
        self.log_list.addItem(f"[{ts}] {msg}")
        self.log_list.scrollToBottom()
        print(msg)
        self.statusBar().showMessage(msg, 4000)

    # ---------- Browse / Load ----------
    def browse_model(self, which: str):
        path, _ = QFileDialog.getOpenFileName(self, f"Select Model {which}", "/home/trendzlink/Downloads", "Model Files (*.pt *.onnx *.engine);;All Files (*)")
        if path:
            if which == "A":
                self.modelA_edit.setText(path)
                self.load_model("A")
            else:
                self.modelB_edit.setText(path)
                self.load_model("B")

    def browse_image_dir(self):
        dir_path = QFileDialog.getExistingDirectory(self, "Select Image Folder", self.image_dir_edit.text() or "/home/trendzlink")
        if dir_path:
            self.image_dir_edit.setText(dir_path)
            self.load_image_dir(dir_path)

    def browse_save_dir(self):
        dir_path = QFileDialog.getExistingDirectory(self, "Select Save Folder", self.save_dir_edit.text() or self.image_dir_edit.text() or "/home/trendzlink")
        if dir_path:
            self.save_dir_edit.setText(dir_path)

    def load_model(self, which: str):
        path = self.modelA_edit.text().strip() if which == "A" else self.modelB_edit.text().strip()
        if not path:
            QMessageBox.warning(self, "No path", f"Please set Model {which} path first.")
            return
        wrapper = self.wrapperA if which == "A" else self.wrapperB
        lbl = self.lbl_statusA if which == "A" else self.lbl_statusB
        self.log(f"Loading Model {which}: {path} ...")
        lbl.setText("Loading...")
        lbl.setStyleSheet("color: #b08800;")
        QApplication.processEvents()
        ok = wrapper.load(path)
        if ok:
            n = len(wrapper.class_names)
            names = ", ".join(list(wrapper.class_names.values())[:10])
            if n > 10:
                names += " ..."
            lbl.setText(f"Loaded ({n} classes)")
            lbl.setStyleSheet("color: #1b7a1b; font-weight: bold;")
            if which == "A":
                self.lbl_classesA.setText(names if names else str(wrapper.class_names))
            else:
                self.lbl_classesB.setText(names if names else str(wrapper.class_names))
            self.log(f"Model {which} loaded: {n} classes: {names}")
            self._update_combined_classes()
            if self.current_image_path:
                self.rerun_inference()
        else:
            lbl.setText(f"Failed: {wrapper.load_error[:60]}")
            lbl.setStyleSheet("color: #c00000;")
            self.log(f"Model {which} failed: {wrapper.load_error}")
            QMessageBox.critical(self, f"Model {which} Load Failed", wrapper.load_error)

    def _update_combined_classes(self):
        # merge class names from both models
        names_set = {}
        for w in [self.wrapperA, self.wrapperB]:
            for cid, name in w.class_names.items():
                if cid not in names_set or names_set[cid] != name:
                    # keep first, but ensure coverage for max id
                    if cid not in names_set:
                        names_set[cid] = name
        if not names_set:
            self.class_names_combined = ["object"]
        else:
            max_id = max(names_set.keys())
            self.class_names_combined = [names_set.get(i, f"class_{i}") for i in range(max_id+1)]
        # update combo
        self.combo_classes.blockSignals(True)
        self.combo_classes.clear()
        for i, n in enumerate(self.class_names_combined):
            self.combo_classes.addItem(f"{n} ({i})", i)
        self.combo_classes.blockSignals(False)
        # update manual name edit to match current spin value
        self._sync_manual_name_from_id()

    def _sync_manual_name_from_id(self):
        cid = self.spin_manual_cls.value()
        if 0 <= cid < len(self.class_names_combined):
            self.edit_manual_name.setText(self.class_names_combined[cid])
        # also set combo
        if 0 <= cid < self.combo_classes.count():
            self.combo_classes.blockSignals(True)
            self.combo_classes.setCurrentIndex(cid)
            self.combo_classes.blockSignals(False)

    def load_image_dir(self, dir_path: str):
        if not os.path.isdir(dir_path):
            self.log(f"Image dir not found: {dir_path}")
            return
        exts = SUPPORTED_IMG_EXTS
        files = []
        for ext in exts:
            files.extend(glob.glob(os.path.join(dir_path, f"*{ext}")))
            files.extend(glob.glob(os.path.join(dir_path, f"*{ext.upper()}")))
        files = sorted(set(files))
        if not files:
            QMessageBox.warning(self, "No Images", f"No images found in:\n{dir_path}\nSupported: {exts}")
            self.log(f"No images in {dir_path}")
            return
        self.image_paths = files
        self.list_images.blockSignals(True)
        self.list_images.clear()
        for p in files:
            self.list_images.addItem(os.path.basename(p))
        self.list_images.blockSignals(False)
        self.slider_img.blockSignals(True)
        self.slider_img.setRange(0, len(files)-1)
        self.slider_img.setEnabled(len(files) > 1)
        self.slider_img.setValue(0)
        self.slider_img.blockSignals(False)
        self.log(f"Loaded {len(files)} images from {dir_path}")
        # default save dir to same folder/labels?
        if not self.save_dir_edit.text().strip():
            # suggest .../labels sibling? but use same folder for now
            self.save_dir_edit.setText(dir_path)
        # reset per-image saved status, check existing txts
        self.saved_status = {}
        save_dir = self.save_dir_edit.text().strip() or dir_path
        for p in files:
            base = os.path.splitext(os.path.basename(p))[0]
            txt = os.path.join(save_dir, base + ".txt")
            if os.path.exists(txt):
                self.saved_status[p] = True
        self._update_per_image_status()
        self._refresh_image_list_icons()
        self.show_image(0)

    # ---------- Image navigation ----------
    def show_image(self, idx: int):
        if not (0 <= idx < len(self.image_paths)):
            return
        # save manual cache for previous image
        if self.current_image_path and self.current_img_w > 0:
            # collect manual boxes from current scene before switching
            manual = [b for b in self.scene.get_all_boxes() if b.source == "manual"]
            self.manual_boxes_cache[self.current_image_path] = manual

        self.current_idx = idx
        path = self.image_paths[idx]
        self.current_image_path = path
        self.list_images.blockSignals(True)
        self.list_images.setCurrentRow(idx)
        self.list_images.blockSignals(False)
        self.slider_img.blockSignals(True)
        self.slider_img.setValue(idx)
        self.slider_img.blockSignals(False)

        # load pixmap
        pixmap = QPixmap(path)
        if pixmap.isNull():
            # try PIL
            if Image is not None:
                try:
                    pil = Image.open(path).convert("RGB")
                    w, h = pil.size
                    data = pil.tobytes("raw", "RGB")
                    qimg = QImage(data, w, h, w*3, QImage.Format_RGB888)
                    pixmap = QPixmap.fromImage(qimg)
                except Exception as e:
                    self.log(f"Failed to load image {path}: {e}")
                    return
            else:
                self.log(f"Failed to load pixmap: {path}")
                return
        self.current_pixmap = pixmap
        self.current_img_w = pixmap.width()
        self.current_img_h = pixmap.height()
        self.scene.set_image(pixmap)
        self.view.fit_image()
        # slight delay to fit correctly after show
        QTimer.singleShot(50, self.view.fit_image)

        self.lbl_filename.setText(f"{os.path.basename(path)}  ({self.current_img_w} x {self.current_img_h})  [{idx+1}/{len(self.image_paths)}]")
        self.lbl_imgInfo.setText(f"{idx+1}/{len(self.image_paths)}  {os.path.basename(path)}")
        # reset boxes
        self.last_boxes_A = []
        self.last_boxes_B = []

        # inference
        self.rerun_inference()
        # restore manual boxes if any
        if path in self.manual_boxes_cache:
            for b in self.manual_boxes_cache[path]:
                # b already is Box with manual source, but need to re-add
                # ensure coords within new image bounds? same image, so ok
                self.scene.add_box_item(Box(b.x1, b.y1, b.x2, b.y2, b.cls, 1.0, "manual", b.label))
            self.refresh_table()

        self._update_ui_state()

    def prev_image(self):
        if self.current_idx > 0:
            self.show_image(self.current_idx - 1)

    def next_image(self):
        if self.current_idx < len(self.image_paths) - 1:
            self.show_image(self.current_idx + 1)

    def _on_image_list_change(self, row: int):
        if row >= 0 and row != self.current_idx:
            self.show_image(row)

    def _on_img_slider(self, val: int):
        if val != self.current_idx:
            self.show_image(val)

    def _on_confA_slider(self, val: int):
        # map 1..95 -> 0.01..0.95
        v = val / 100.0
        self.spin_confA.blockSignals(True)
        self.spin_confA.setValue(v)
        self.spin_confA.blockSignals(False)
        self._schedule_rerun()

    def _on_confA_spin(self, val: float):
        self.slider_confA.blockSignals(True)
        self.slider_confA.setValue(int(val*100))
        self.slider_confA.blockSignals(False)
        self._schedule_rerun()

    def _on_confB_slider(self, val: int):
        v = val / 100.0
        self.spin_confB.blockSignals(True)
        self.spin_confB.setValue(v)
        self.spin_confB.blockSignals(False)
        self._schedule_rerun()

    def _on_confB_spin(self, val: float):
        self.slider_confB.blockSignals(True)
        self.slider_confB.setValue(int(val*100))
        self.slider_confB.blockSignals(False)
        self._schedule_rerun()

    def _schedule_rerun(self):
        self.rerun_timer.start()

    def _update_ui_state(self):
        has_images = len(self.image_paths) > 0
        has_current = self.current_idx >= 0
        self.btn_prev.setEnabled(has_current and self.current_idx > 0)
        self.btn_next.setEnabled(has_current and self.current_idx < len(self.image_paths)-1)
        self.btn_save.setEnabled(has_current)
        self.btn_saveAll.setEnabled(has_images)
        self.btn_rerun.setEnabled(has_current)

    # ---------- Inference ----------
    def rerun_inference(self):
        if not self.current_image_path or self.current_img_w == 0:
            return
        # preserve manual boxes
        manual_boxes = [b for b in self.scene.get_all_boxes() if b.source == "manual"] if self.scene.box_items else []
        # also cached? we already have manual_boxes list

        # clear only pred boxes
        self.scene.clear_boxes("modelA")
        self.scene.clear_boxes("modelB")
        # also need to keep manual - already not cleared above but we cleared preds only
        # Alternative: clear all preds explicitly

        counts = {"A": 0, "B": 0}
        # Model A
        if self.wrapperA.model is not None and self.chk_showA.isChecked():
            conf = self.spin_confA.value()
            iou = self.spin_iouA.value()
            maxd = self.spin_maxA.value()
            # log
            self.log(f"Infer A: conf={conf:.2f} iou={iou:.2f} max_det={maxd} on {os.path.basename(self.current_image_path)}")
            boxes = self.wrapperA.predict(self.current_image_path, conf=conf, iou=iou, max_det=maxd)
            # tag source
            for b in boxes:
                b.source = "modelA"
                # ensure label
                if not b.label:
                    b.label = self.wrapperA.class_names.get(b.cls, str(b.cls))
            self.last_boxes_A = boxes
            counts["A"] = len(boxes)
            for b in boxes:
                self.scene.add_box_item(b)
        else:
            self.last_boxes_A = []
            if self.wrapperA.model is None and self.modelA_edit.text().strip():
                self.log("Model A not loaded - skipping inference")
        # Model B
        if self.wrapperB.model is not None and self.chk_showB.isChecked():
            conf = self.spin_confB.value()
            iou = self.spin_iouB.value()
            maxd = self.spin_maxB.value()
            self.log(f"Infer B: conf={conf:.2f} iou={iou:.2f} max_det={maxd} on {os.path.basename(self.current_image_path)}")
            boxes = self.wrapperB.predict(self.current_image_path, conf=conf, iou=iou, max_det=maxd)
            for b in boxes:
                b.source = "modelB"
                if not b.label:
                    b.label = self.wrapperB.class_names.get(b.cls, str(b.cls))
            self.last_boxes_B = boxes
            counts["B"] = len(boxes)
            for b in boxes:
                self.scene.add_box_item(b)
        else:
            self.last_boxes_B = []

        # re-add manual boxes (they were preserved but we cleared only preds, so still there)
        # Ensure manual visibility
        if not self.chk_showManual.isChecked():
            # hide manual items? we didn't hide, we need to set visible
            for it in self.scene.box_items:
                if it.box.source == "manual":
                    it.setVisible(False)
        else:
            for it in self.scene.box_items:
                if it.box.source == "manual":
                    it.setVisible(True)

        self._apply_visibility()
        self.refresh_table()
        self._update_counts()
        self.log(f"Displayed: A={counts['A']} B={counts['B']} Manual={len(manual_boxes)}")

    def _apply_visibility(self):
        showA = self.chk_showA.isChecked()
        showB = self.chk_showB.isChecked()
        showM = self.chk_showManual.isChecked()
        for it in self.scene.box_items:
            if it.box.source == "modelA":
                it.setVisible(showA)
            elif it.box.source == "modelB":
                it.setVisible(showB)
            elif it.box.source == "manual":
                it.setVisible(showM)

    def _on_visibility_toggle(self):
        self._apply_visibility()
        self.refresh_table()
        self._update_counts()

    def _update_counts(self):
        a = len([it for it in self.scene.box_items if it.box.source == "modelA" and it.isVisible()])
        b = len([it for it in self.scene.box_items if it.box.source == "modelB" and it.isVisible()])
        m = len([it for it in self.scene.box_items if it.box.source == "manual" and it.isVisible()])
        # also count all including hidden?
        total_visible = a + b + m
        total_all = len(self.scene.box_items)
        self.lbl_counts.setText(f"ModelA: {a} | ModelB: {b} | Manual: {m} | Visible: {total_visible} / Total: {total_all}  (img: {self.current_img_w}x{self.current_img_h})")

    # ---------- Drawing ----------
    def _on_draw_toggle(self, checked: bool):
        self.scene.set_draw_mode(checked)
        if checked:
            self.view.setCursor(Qt.CrossCursor)
            self.view.setDragMode(QGraphicsView.NoDrag)
            self.lbl_drawInfo.setText("Draw: ON - drag on image")
            self.lbl_drawInfo.setStyleSheet("color: #c00000; font-weight: bold;")
            self.log("Draw mode ON - drag to create box")
        else:
            self.view.setCursor(Qt.ArrowCursor)
            self.view.setDragMode(QGraphicsView.RubberBandDrag)
            self.lbl_drawInfo.setText("Draw: OFF")
            self.lbl_drawInfo.setStyleSheet("color: #666;")
            self.log("Draw mode OFF")

    def _on_manual_cls_change(self, val: int):
        self._sync_manual_name_from_id()
        # also update combo

    def _on_manual_name_change(self, txt: str):
        # update combined list entry? keep
        cid = self.spin_manual_cls.value()
        if 0 <= cid < len(self.class_names_combined):
            self.class_names_combined[cid] = txt

    def _on_combo_cls_change(self, idx: int):
        if idx >= 0:
            cid = self.combo_classes.currentData()
            if cid is not None:
                self.spin_manual_cls.blockSignals(True)
                self.spin_manual_cls.setValue(cid)
                self.spin_manual_cls.blockSignals(False)
                self.edit_manual_name.setText(self.class_names_combined[cid] if cid < len(self.class_names_combined) else str(cid))

    def _on_box_drawn(self, x1, y1, x2, y2):
        cid = self.spin_manual_cls.value()
        name = self.edit_manual_name.text().strip() or f"class_{cid}"
        # ensure class list extended if cid beyond
        if cid >= len(self.class_names_combined):
            # extend
            needed = cid - len(self.class_names_combined) + 1
            self.class_names_combined.extend([f"class_{i}" for i in range(len(self.class_names_combined), cid+1)])
            self.class_names_combined[cid] = name
            self._update_combined_classes()
            self.spin_manual_cls.setValue(cid)
        box = Box(x1, y1, x2, y2, cid, 1.0, "manual", name)
        item = self.scene.add_box_item(box)
        # select new item
        self.scene.clearSelection()
        item.setSelected(True)
        self.log(f"Manual box drawn: cls={cid} ({name}) xyxy={x1:.0f},{y1:.0f},{x2:.0f},{y2:.0f}")
        # keep manual cache
        if self.current_image_path:
            if self.current_image_path not in self.manual_boxes_cache:
                self.manual_boxes_cache[self.current_image_path] = []
            # will refresh on save; but also update cache now for navigation memory? we collect on switch
            # add to cache immediately for robustness
            # note: we will resync on image switch from scene, so not needed here
            pass
        self.refresh_table()
        self._update_counts()
        # optionally stay in draw mode for next box

    def _on_click_empty(self):
        self.scene.clearSelection()
        self.table.clearSelection()

    def _on_scene_selection(self):
        # sync table selection to scene - handle multi-select correctly (selectRow clears, so use selectionModel)
        selected = [it for it in self.scene.box_items if it.isSelected()]
        self.table.blockSignals(True)
        try:
            if len(selected) == 0:
                self.table.clearSelection()
            else:
                filtered = self._get_filtered_boxes_with_items()
                item_to_row = {item: idx for idx, (_, item) in enumerate(filtered)}
                self.table.clearSelection()
                model = self.table.selectionModel()
                for sel_item in selected:
                    if sel_item in item_to_row:
                        r = item_to_row[sel_item]
                        idx = self.table.model().index(r, 0)
                        # Select row with Ctrl-like behavior (add to selection)
                        model.select(idx, QItemSelectionModel.Select | QItemSelectionModel.Rows)
        finally:
            self.table.blockSignals(False)
        # update appearance
        for it in self.scene.box_items:
            it.set_selected_appearance(it.isSelected())
        self.view.viewport().update()

    def _on_table_select(self):
        # sync scene selection from table - supports multi-select
        rows = self.table.selectionModel().selectedRows() if self.table.selectionModel() else []
        if not rows:
            # if table deselected but scene has selection? keep? Actually clear scene selection if none in table?
            # optional: clear scene if no table rows? but scene->table sync already handles.
            return
        boxes = self._get_filtered_boxes_with_items()
        self.scene.blockSignals(True)
        try:
            self.scene.clearSelection()
            for r_idx in rows:
                r = r_idx.row()
                if 0 <= r < len(boxes):
                    _, item = boxes[r]
                    item.setSelected(True)
            # center on last selected
            if boxes and 0 <= rows[-1].row() < len(boxes):
                _, last_item = boxes[rows[-1].row()]
                self.view.centerOn(last_item)
        finally:
            self.scene.blockSignals(False)
        # update appearance
        for it in self.scene.box_items:
            it.set_selected_appearance(it.isSelected())
        self.view.viewport().update()

    # ---------- Delete helpers + Undo ----------
    def _push_undo(self, boxes: List[Box], desc: str):
        if not boxes:
            return
        # store deep copy of boxes
        import copy
        self.undo_stack.append({"boxes": copy.deepcopy(boxes), "desc": desc})
        # limit stack 50
        if len(self.undo_stack) > 50:
            self.undo_stack.pop(0)
        self.btn_undo.setEnabled(True)
        self.btn_undo.setToolTip(f"Undo: {desc} ({len(boxes)} boxes) - Ctrl+Z")
        self.log(f"Undo push: {desc} ({len(boxes)} boxes)")

    def undo_last(self):
        if not self.undo_stack:
            self.log("Nothing to undo")
            return
        entry = self.undo_stack.pop()
        boxes: List[Box] = entry["boxes"]
        desc = entry["desc"]
        for b in boxes:
            self.scene.add_box_item(b)
        self.refresh_table()
        self._update_counts()
        self.log(f"Undo: restored {len(boxes)} boxes from '{desc}'")
        self.btn_undo.setEnabled(len(self.undo_stack) > 0)
        if self.undo_stack:
            self.btn_undo.setToolTip(f"Undo: {self.undo_stack[-1]['desc']} ({len(self.undo_stack[-1]['boxes'])} boxes)")
        else:
            self.btn_undo.setToolTip("Undo last deletion (Ctrl+Z)")

    def delete_single_item(self, item: "BoxItem"):
        item.sync_box()
        self._push_undo([Box(item.box.x1, item.box.y1, item.box.x2, item.box.y2, item.box.cls, item.box.conf, item.box.source, item.box.label)], f"delete single {item.box.source}")
        self.scene.remove_box_item(item)
        self.log(f"Deleted single: {item.box.source} cls={item.box.cls} {item.box.x1:.0f},{item.box.y1:.0f},{item.box.x2:.0f},{item.box.y2:.0f}")
        self.refresh_table()
        self._update_counts()

    def duplicate_item(self, item: "BoxItem"):
        item.sync_box()
        # offset slightly
        off = 10
        b = Box(item.box.x1 + off, item.box.y1 + off, item.box.x2 + off, item.box.y2 + off, item.box.cls, item.box.conf, "manual", item.box.label)
        # clamp to img
        b.x1 = max(0, min(b.x1, self.current_img_w - 5))
        b.y1 = max(0, min(b.y1, self.current_img_h - 5))
        b.x2 = max(b.x1 + 5, min(b.x2, self.current_img_w))
        b.y2 = max(b.y1 + 5, min(b.y2, self.current_img_h))
        new_item = self.scene.add_box_item(b)
        self.scene.clearSelection()
        new_item.setSelected(True)
        self.refresh_table()
        self._update_counts()
        self.log(f"Duplicated box cls={b.cls} -> manual")

    def _edit_class_for_item(self, item: "BoxItem"):
        new_cls, ok = QInputDialog.getInt(self, "Edit Class", f"Class ID for box (current {item.box.cls}):", item.box.cls, 0, 100, 1)
        if ok:
            item.box.cls = new_cls
            if 0 <= new_cls < len(self.class_names_combined):
                item.box.label = self.class_names_combined[new_cls]
            else:
                item.box.label = f"class_{new_cls}"
            item.update_label()
            self.refresh_table()
            self.log(f"Edited class for box -> {new_cls}")

    def delete_selected(self):
        # Prefer table selection if table has focus/selected rows, else scene selection
        # Check table selected rows first (supports multi-select)
        table_rows = []
        if self.table.selectionModel():
            table_rows = self.table.selectionModel().selectedRows()
        if table_rows:
            # delete via table filtered mapping
            filtered = self._get_filtered_boxes_with_items()
            # collect items for selected rows (row indices relate to filtered order)
            # Note: selectedRows returns rows in table order = filtered order
            items_to_del = []
            for r in sorted(table_rows, key=lambda x: x.row(), reverse=True):
                row = r.row()
                if 0 <= row < len(filtered):
                    _, item = filtered[row]
                    items_to_del.append(item)
            if not items_to_del:
                self.log("No table rows to delete")
                return
            boxes = []
            for it in items_to_del:
                it.sync_box()
                boxes.append(Box(it.box.x1, it.box.y1, it.box.x2, it.box.y2, it.box.cls, it.box.conf, it.box.source, it.box.label))
            self._push_undo(boxes, f"delete selected rows ({len(items_to_del)})")
            for it in items_to_del:
                self.scene.remove_box_item(it)
                self.log(f"Deleted via table: {it.box.source} cls={it.box.cls}")
            self.refresh_table()
            self._update_counts()
            return
        # fallback: scene selection
        selected = [it for it in self.scene.box_items if it.isSelected()]
        if not selected:
            self.log("No selection to delete (select box on image or row in table, or use Right-click)")
            return
        # support multi-select via item selection
        boxes = []
        for it in selected:
            it.sync_box()
            boxes.append(Box(it.box.x1, it.box.y1, it.box.x2, it.box.y2, it.box.cls, it.box.conf, it.box.source, it.box.label))
        self._push_undo(boxes, f"delete selected ({len(selected)})")
        for it in selected:
            self.scene.remove_box_item(it)
            self.log(f"Deleted box: cls={it.box.cls} {it.box.x1:.0f},{it.box.y1:.0f},{it.box.x2:.0f},{it.box.y2:.0f}")
        self.refresh_table()
        self._update_counts()

    def delete_table_selected_rows(self):
        # explicit button for table
        self.delete_selected()

    def delete_filtered(self):
        filtered = self._get_filtered_boxes_with_items()
        if not filtered:
            self.log("No filtered boxes to delete")
            return
        filt_src = self.combo_filter_source.currentText()
        filt_cls = self.edit_filter_cls.text().strip()
        desc = f"filtered {filt_src} {filt_cls}".strip()
        if QMessageBox.question(self, "Delete Filtered", f"Delete {len(filtered)} boxes matching filter?\nFilter: {desc or 'All visible'}\nThis can be undone.", QMessageBox.Yes | QMessageBox.No) != QMessageBox.Yes:
            return
        boxes = []
        items = []
        for b, it in filtered:
            it.sync_box()
            boxes.append(Box(it.box.x1, it.box.y1, it.box.x2, it.box.y2, it.box.cls, it.box.conf, it.box.source, it.box.label))
            items.append(it)
        self._push_undo(boxes, f"delete filtered ({len(items)}) {desc}")
        for it in items:
            self.scene.remove_box_item(it)
        self.log(f"Deleted filtered: {len(items)} boxes")
        self.refresh_table()
        self._update_counts()

    def delete_low_conf(self):
        # prompt for threshold
        thresh, ok = QInputDialog.getDouble(self, "Delete Low Confidence", "Delete boxes with confidence < threshold (0.01-1.0):", 0.30, 0.01, 1.0, 2)
        if not ok:
            return
        # consider only non-manual (manual has conf 1.0) but include if below? manual always 1, so ignore
        to_del = []
        for it in self.scene.box_items:
            if not it.isVisible():
                continue
            it.sync_box()
            if it.box.source == "manual":
                continue
            if it.box.conf < thresh:
                to_del.append(it)
        if not to_del:
            self.log(f"No boxes with conf < {thresh:.2f}")
            return
        if QMessageBox.question(self, "Delete Low Conf", f"Delete {len(to_del)} boxes with conf < {thresh:.2f}?\nUndo available.", QMessageBox.Yes | QMessageBox.No) != QMessageBox.Yes:
            return
        boxes = [Box(it.box.x1, it.box.y1, it.box.x2, it.box.y2, it.box.cls, it.box.conf, it.box.source, it.box.label) for it in to_del]
        self._push_undo(boxes, f"delete low-conf <{thresh:.2f} ({len(to_del)})")
        for it in to_del:
            self.scene.remove_box_item(it)
        self.log(f"Deleted low-conf <{thresh:.2f}: {len(to_del)} boxes")
        self.refresh_table()
        self._update_counts()

    def clear_boxes(self, which: str):
        # which: manual, modelA, modelB, preds (A+B), all
        if which == "manual":
            items = [it for it in self.scene.box_items if it.box.source == "manual"]
            if not items:
                self.log("No manual boxes to clear")
                return
            if QMessageBox.question(self, "Clear Manual", f"Remove all {len(items)} MANUAL boxes for current image?", QMessageBox.Yes | QMessageBox.No) != QMessageBox.Yes:
                return
            boxes = []
            for it in items:
                it.sync_box()
                boxes.append(Box(it.box.x1, it.box.y1, it.box.x2, it.box.y2, it.box.cls, it.box.conf, it.box.source, it.box.label))
            self._push_undo(boxes, f"clear manual ({len(items)})")
            self.scene.clear_boxes("manual")
            if self.current_image_path in self.manual_boxes_cache:
                del self.manual_boxes_cache[self.current_image_path]
            self.log("Cleared manual boxes")
        elif which == "modelA":
            items = [it for it in self.scene.box_items if it.box.source == "modelA"]
            if not items:
                self.log("No Model A boxes to clear")
                return
            if QMessageBox.question(self, "Clear Model A", f"Remove all {len(items)} Model A boxes?\n(Re-run to restore)", QMessageBox.Yes | QMessageBox.No) != QMessageBox.Yes:
                return
            boxes = [Box(it.box.x1, it.box.y1, it.box.x2, it.box.y2, it.box.cls, it.box.conf, it.box.source, it.box.label) for it in items]
            self._push_undo(boxes, f"clear modelA ({len(items)})")
            self.scene.clear_boxes("modelA")
            self.log("Cleared Model A boxes")
        elif which == "modelB":
            items = [it for it in self.scene.box_items if it.box.source == "modelB"]
            if not items:
                self.log("No Model B boxes to clear")
                return
            if QMessageBox.question(self, "Clear Model B", f"Remove all {len(items)} Model B boxes?\n(Re-run to restore)", QMessageBox.Yes | QMessageBox.No) != QMessageBox.Yes:
                return
            boxes = [Box(it.box.x1, it.box.y1, it.box.x2, it.box.y2, it.box.cls, it.box.conf, it.box.source, it.box.label) for it in items]
            self._push_undo(boxes, f"clear modelB ({len(items)})")
            self.scene.clear_boxes("modelB")
            self.log("Cleared Model B boxes")
        elif which == "preds":
            items = [it for it in self.scene.box_items if it.box.source in ("modelA", "modelB")]
            if not items:
                self.log("No prediction boxes to clear")
                return
            if QMessageBox.question(self, "Clear Preds", f"Remove all {len(items)} prediction boxes (A+B) for current image? (Re-run to restore)", QMessageBox.Yes | QMessageBox.No) != QMessageBox.Yes:
                return
            boxes = [Box(it.box.x1, it.box.y1, it.box.x2, it.box.y2, it.box.cls, it.box.conf, it.box.source, it.box.label) for it in items]
            self._push_undo(boxes, f"clear preds ({len(items)})")
            self.scene.clear_boxes("modelA")
            self.scene.clear_boxes("modelB")
            self.log("Cleared prediction boxes")
        elif which == "all":
            items = list(self.scene.box_items)
            if not items:
                self.log("No boxes to clear")
                return
            if QMessageBox.question(self, "Clear ALL", f"Remove ALL {len(items)} boxes (manual + preds) for current image?", QMessageBox.Yes | QMessageBox.No) != QMessageBox.Yes:
                return
            boxes = []
            for it in items:
                it.sync_box()
                boxes.append(Box(it.box.x1, it.box.y1, it.box.x2, it.box.y2, it.box.cls, it.box.conf, it.box.source, it.box.label))
            self._push_undo(boxes, f"clear all ({len(items)})")
            self.scene.clear_boxes(None)
            if self.current_image_path in self.manual_boxes_cache:
                del self.manual_boxes_cache[self.current_image_path]
            self.log("Cleared all boxes")
        else:
            self.scene.clear_boxes(None)
            self.log("Cleared all boxes")
        self.refresh_table()
        self._update_counts()

    # Context menus
    def show_view_context_menu(self, pos):
        global_pos = self.view.mapToGlobal(pos)
        scene_pos = self.view.mapToScene(pos)
        self._context_scene_pos = scene_pos
        # find item under cursor
        items_under = self.scene.items(scene_pos)
        box_under = None
        for it in items_under:
            if isinstance(it, BoxItem):
                box_under = it
                break
            # child label? check parent
            if isinstance(it, QGraphicsTextItem) and isinstance(it.parentItem(), BoxItem):
                box_under = it.parentItem()
                break
        menu = QMenu(self)
        # delete under cursor
        if box_under:
            act_del_under = menu.addAction(f"🗑️ Delete box under cursor ({box_under.box.source} cls={box_under.box.cls} {box_under.box.conf:.2f})")
            act_del_under.triggered.connect(lambda: self.delete_single_item(box_under))
            act_edit_under = menu.addAction("✏️ Edit class of this box")
            act_edit_under.triggered.connect(lambda: self._edit_class_for_item(box_under))
            act_dup = menu.addAction("⎘ Duplicate this box")
            act_dup.triggered.connect(lambda: self.duplicate_item(box_under))
            menu.addSeparator()
        # general
        has_sel = any(it.isSelected() for it in self.scene.box_items)
        act_del_sel = menu.addAction("🗑️ Delete Selected" + (f" ({len([it for it in self.scene.box_items if it.isSelected()])})" if has_sel else ""))
        act_del_sel.setEnabled(has_sel)
        act_del_sel.triggered.connect(self.delete_selected)
        act_del_filtered = menu.addAction(f"Delete Filtered ({len(self._get_filtered_boxes_with_items())})")
        act_del_filtered.triggered.connect(self.delete_filtered)
        menu.addSeparator()
        # clear sub-menu
        clear_menu = menu.addMenu("Clear ...")
        act_c_manual = clear_menu.addAction(f"Clear Manual ({len([it for it in self.scene.box_items if it.box.source=='manual'])})")
        act_c_manual.triggered.connect(lambda: self.clear_boxes("manual"))
        act_c_a = clear_menu.addAction(f"Clear Model A ({len([it for it in self.scene.box_items if it.box.source=='modelA'])})")
        act_c_a.triggered.connect(lambda: self.clear_boxes("modelA"))
        act_c_b = clear_menu.addAction(f"Clear Model B ({len([it for it in self.scene.box_items if it.box.source=='modelB'])})")
        act_c_b.triggered.connect(lambda: self.clear_boxes("modelB"))
        act_c_preds = clear_menu.addAction(f"Clear Preds A+B ({len([it for it in self.scene.box_items if it.box.source in ('modelA','modelB')])})")
        act_c_preds.triggered.connect(lambda: self.clear_boxes("preds"))
        act_c_all = clear_menu.addAction(f"Clear ALL ({len(self.scene.box_items)})")
        act_c_all.triggered.connect(lambda: self.clear_boxes("all"))
        menu.addSeparator()
        act_low = menu.addAction("Delete Low Confidence (< thresh)...")
        act_low.triggered.connect(self.delete_low_conf)
        if self.undo_stack:
            act_undo = menu.addAction(f"↩ Undo {self.undo_stack[-1]['desc']}")
            act_undo.triggered.connect(self.undo_last)
        else:
            act_undo = menu.addAction("↩ Undo (nothing)")
            act_undo.setEnabled(False)
        menu.addSeparator()
        act_sel_all = menu.addAction("Select All Boxes (Ctrl+A)")
        act_sel_all.triggered.connect(lambda: [it.setSelected(True) for it in self.scene.box_items])
        act_desel = menu.addAction("Deselect All")
        act_desel.triggered.connect(lambda: self.scene.clearSelection())
        menu.exec_(global_pos)

    def show_table_context_menu(self, pos):
        global_pos = self.table.mapToGlobal(pos)
        item = self.table.itemAt(pos)
        menu = QMenu(self)
        has_sel = len(self.table.selectionModel().selectedRows()) > 0 if self.table.selectionModel() else False
        # find row
        if item:
            row = item.row()
            filtered = self._get_filtered_boxes_with_items()
            if 0 <= row < len(filtered):
                box, box_item = filtered[row]
                act_del_one = menu.addAction(f"🗑️ Delete this row ({box.source} cls={box.cls})")
                act_del_one.triggered.connect(lambda r=row: self._delete_table_row(r))
                act_edit = menu.addAction(f"✏️ Edit class (current {box.cls})")
                act_edit.triggered.connect(lambda r=row: self._edit_class_for_row(r))
                act_select = menu.addAction("Select this box on image")
                act_select.triggered.connect(lambda r=row: self._select_table_row_on_scene(r))
                menu.addSeparator()
        if has_sel:
            act_del_sel = menu.addAction(f"🗑️ Delete Selected Rows ({len(self.table.selectionModel().selectedRows())})")
            act_del_sel.triggered.connect(self.delete_table_selected_rows)
        else:
            act_del_sel = menu.addAction("🗑️ Delete Selected Rows")
            act_del_sel.setEnabled(False)
        act_del_filtered = menu.addAction(f"Delete All Filtered ({len(self._get_filtered_boxes_with_items())})")
        act_del_filtered.triggered.connect(self.delete_filtered)
        menu.addSeparator()
        act_clear_manual = menu.addAction("Clear Manual")
        act_clear_manual.triggered.connect(lambda: self.clear_boxes("manual"))
        act_clear_preds = menu.addAction("Clear Preds")
        act_clear_preds.triggered.connect(lambda: self.clear_boxes("preds"))
        act_clear_all = menu.addAction("Clear ALL")
        act_clear_all.triggered.connect(lambda: self.clear_boxes("all"))
        if self.undo_stack:
            act_undo = menu.addAction(f"↩ Undo {self.undo_stack[-1]['desc']}")
            act_undo.triggered.connect(self.undo_last)
        menu.exec_(global_pos)

    def _delete_table_row(self, row: int):
        filtered = self._get_filtered_boxes_with_items()
        if not (0 <= row < len(filtered)):
            return
        _, item = filtered[row]
        self.delete_single_item(item)

    def _select_table_row_on_scene(self, row: int):
        filtered = self._get_filtered_boxes_with_items()
        if not (0 <= row < len(filtered)):
            return
        _, item = filtered[row]
        self.scene.clearSelection()
        item.setSelected(True)
        self.view.centerOn(item)
        self.refresh_table()


    # ---------- Table ----------
    def _get_filtered_boxes_with_items(self) -> List[Tuple[Box, BoxItem]]:
        filt_src = self.combo_filter_source.currentText()
        filt_cls = self.edit_filter_cls.text().strip().lower()
        out = []
        for it in self.scene.box_items:
            if not it.isVisible():
                # visibility already reflects show toggles, but filter should hide further?
                # For table we want to show only visible? Or respect visibility toggles + source filter.
                # If its source hidden, skip
                pass
            # source filter
            src_map = {"Model A": "modelA", "Model B": "modelB", "Manual": "manual"}
            if filt_src != "All":
                need = src_map.get(filt_src, "")
                if it.box.source != need:
                    continue
            # class filter
            if filt_cls:
                label_match = filt_cls in str(it.box.cls).lower() or filt_cls in it.box.label.lower()
                if not label_match:
                    continue
            # respect global visibility toggles? if item is hidden due to show toggle, we already filter? Let's include only visible
            if not it.isVisible():
                continue
            it.sync_box()
            out.append((it.box, it))
        # sort by conf descending then source
        out.sort(key=lambda x: (-x[0].conf, x[0].source))
        return out

    def refresh_table(self):
        filtered = self._get_filtered_boxes_with_items()
        self.table.blockSignals(True)
        self.table.setRowCount(len(filtered))
        for r, (box, item) in enumerate(filtered):
            # Src
            src_name = SOURCE_LABELS.get(box.source, box.source)
            src_item = QTableWidgetItem(src_name)
            color = SOURCE_COLORS.get(box.source, QColor(0, 0, 0))
            src_item.setBackground(QBrush(QColor(color.red(), color.green(), color.blue(), 40)))
            src_item.setTextAlignment(Qt.AlignCenter)
            self.table.setItem(r, 0, src_item)
            # Class
            cls_txt = f"{box.cls}: {box.label}" if box.label else str(box.cls)
            cls_item = QTableWidgetItem(cls_txt)
            # editable? allow double click to edit class? we keep read-only but enable editing via click?
            self.table.setItem(r, 1, cls_item)
            # Conf
            conf_txt = f"{box.conf:.2f}" if box.source != "manual" else "1.00"
            conf_item = QTableWidgetItem(conf_txt)
            conf_item.setTextAlignment(Qt.AlignCenter)
            self.table.setItem(r, 2, conf_item)
            # xyxy
            xyxy_txt = f"{box.x1:.0f},{box.y1:.0f},{box.x2:.0f},{box.y2:.0f}"
            xy_item = QTableWidgetItem(xyxy_txt)
            xy_item.setToolTip(f"x1={box.x1:.1f} y1={box.y1:.1f} x2={box.x2:.1f} y2={box.y2:.1f}  W={box.x2-box.x1:.0f} H={box.y2-box.y1:.0f}")
            self.table.setItem(r, 3, xy_item)
            # area
            area = (box.x2 - box.x1) * (box.y2 - box.y1)
            area_item = QTableWidgetItem(f"{int(area)}")
            area_item.setTextAlignment(Qt.AlignRight)
            self.table.setItem(r, 4, area_item)
            # Del button text
            del_item = QTableWidgetItem("✕")
            del_item.setTextAlignment(Qt.AlignCenter)
            del_item.setForeground(QBrush(QColor(200, 0, 0)))
            self.table.setItem(r, 5, del_item)
        self.table.blockSignals(False)
        # also update counts already
        # double click handler for del column?
        self.table.viewport().update()

    # Override to handle double click on del column
    def eventFilter(self, source, event):
        return super().eventFilter(source, event)

    # intercept table clicks for delete
    def init_table_click_handler(self):
        self.table.cellClicked.connect(self._on_table_cell_clicked)

    def _on_table_cell_clicked(self, row, col):
        if col == 5:  # del
            filtered = self._get_filtered_boxes_with_items()
            if 0 <= row < len(filtered):
                box, item = filtered[row]
                self.scene.remove_box_item(item)
                self.log(f"Deleted via table: {box.source} cls={box.cls}")
                self.refresh_table()
                self._update_counts()
        elif col == 1:  # class edit on double click? use single click prompt?
            pass

    # Enable editing class via double click
    def _edit_class_for_row(self, row):
        filtered = self._get_filtered_boxes_with_items()
        if not (0 <= row < len(filtered)):
            return
        box, item = filtered[row]
        # prompt
        from PyQt5.QtWidgets import QInputDialog
        new_cls, ok = QInputDialog.getInt(self, "Edit Class", f"Class ID for box (current {box.cls}):", box.cls, 0, 100, 1)
        if ok:
            box.cls = new_cls
            # update label if known
            if 0 <= new_cls < len(self.class_names_combined):
                box.label = self.class_names_combined[new_cls]
            else:
                box.label = f"class_{new_cls}"
            item.box = box
            item.update_label()
            self.refresh_table()

    # ---------- Save (Combined + Per-Image) ----------
    def _iou(self, a: Box, b: Box) -> float:
        # IoU for dedup across models
        x1 = max(a.x1, b.x1)
        y1 = max(a.y1, b.y1)
        x2 = min(a.x2, b.x2)
        y2 = min(a.y2, b.y2)
        if x2 <= x1 or y2 <= y1:
            return 0.0
        inter = (x2 - x1) * (y2 - y1)
        area_a = max(0, (a.x2 - a.x1)) * max(0, (a.y2 - a.y1))
        area_b = max(0, (b.x2 - b.x1)) * max(0, (b.y2 - b.y1))
        union = area_a + area_b - inter
        return inter / union if union > 0 else 0

    def _nms_merge(self, boxes: List[Box], iou_thresh: float = 0.5) -> List[Box]:
        # Simple NMS: sort by conf desc, keep if IoU < thresh vs kept (same class only)
        if not boxes or iou_thresh >= 1:
            return boxes
        # sort by conf descending (manual boxes have 1.0)
        sorted_boxes = sorted(boxes, key=lambda b: b.conf, reverse=True)
        kept: List[Box] = []
        for b in sorted_boxes:
            keep = True
            for k in kept:
                if b.cls != k.cls:
                    continue
                if self._iou(b, k) > iou_thresh:
                    keep = False
                    break
            if keep:
                kept.append(b)
        return kept

    def _get_combined_boxes_for_current(self, respect_visibility: bool = None) -> List[Box]:
        # Determine mode-based filtering for CURRENT view's boxes
        mode = self.cmb_save_mode.currentText() if hasattr(self, 'cmb_save_mode') else "Combined (A+B+Manual)"
        include_preds = self.chk_save_include_preds.isChecked()
        if respect_visibility is None:
            respect_visibility = self.chk_hide_on_export.isChecked()
            # In "Visible only" mode force visibility respect
            if "Visible only" in mode:
                respect_visibility = True

        boxes: List[Box] = []
        for it in self.scene.box_items:
            it.sync_box()
            b = it.box
            # include filter
            if not include_preds and b.source != "manual":
                continue
            if respect_visibility and not it.isVisible():
                continue
            # mode filtering
            if mode == "Manual only":
                if b.source != "manual":
                    continue
            elif mode == "Model A only":
                if b.source != "modelA":
                    continue
            elif mode == "Model B only":
                if b.source != "modelB":
                    continue
            elif mode == "Model A + Manual":
                if b.source not in ("modelA", "manual"):
                    continue
            elif mode == "Model B + Manual":
                if b.source not in ("modelB", "manual"):
                    continue
            elif "Visible only" in mode:
                # already filtered by visibility above
                pass
            else:  # Combined
                pass  # include all that passed include_preds
            boxes.append(Box(b.x1, b.y1, b.x2, b.y2, b.cls, b.conf, b.source, b.label))

        # dedup across models if checked and mode combined
        if self.chk_dedup.isChecked() and "Combined" in mode and len(boxes) > 1:
            iou = self.spin_dedup_iou.value()
            # Only dedup preds, keep manual separate? We'll dedup all preds together, but keep manual untouched (add after)
            preds = [b for b in boxes if b.source != "manual"]
            manuals = [b for b in boxes if b.source == "manual"]
            preds = self._nms_merge(preds, iou)
            boxes = preds + manuals

        return boxes

    def _on_save_mode_changed(self):
        mode = self.cmb_save_mode.currentText()
        # update button texts to reflect mode
        self.btn_save.setText(f"💾 Save Current ({mode.split('(')[0].strip()})")
        self.btn_saveAll.setText(f"💾 Save ALL ({mode.split('(')[0].strip()}, Per-Image)")
        # hint
        self.statusBar().showMessage(f"Export mode: {mode} | Dedup: {self.chk_dedup.isChecked()} IoU={self.spin_dedup_iou.value():.2f}" , 3000)

    def _mark_saved(self, img_path: str, saved: bool = True):
        self.saved_status[img_path] = saved
        self._update_per_image_status()
        self._refresh_image_list_icons()

    def _refresh_image_list_icons(self):
        # Add ✓ marker for saved files in list
        for i in range(self.list_images.count()):
            item = self.list_images.item(i)
            if i < len(self.image_paths):
                path = self.image_paths[i]
                base = os.path.basename(path)
                if self.saved_status.get(path):
                    item.setText(f"✓ {base}")
                    item.setForeground(QColor(0, 128, 0))
                else:
                    item.setText(base)
                    item.setForeground(QColor(0, 0, 0))

    def _update_per_image_status(self):
        total = len(self.image_paths)
        saved = sum(1 for p in self.image_paths if self.saved_status.get(p))
        self.lbl_per_image_status.setText(f"{saved}/{total} images saved (combined)")

    def _get_headless_boxes_for_image(self, img_path: str) -> Tuple[List[Box], int, int]:
        # Headless inference for any image path, returns (boxes, w, h) per current export mode
        w, h = self._get_image_size(img_path)
        if w == 0 or h == 0:
            return [], 0, 0
        mode = self.cmb_save_mode.currentText() if hasattr(self, 'cmb_save_mode') else "Combined (A+B+Manual)"
        include_preds = self.chk_save_include_preds.isChecked()
        respect_vis = self.chk_hide_on_export.isChecked()
        if "Visible only" in mode:
            respect_vis = True
            # visibility in headless: respect Show toggles
            showA = self.chk_showA.isChecked()
            showB = self.chk_showB.isChecked()
            showM = self.chk_showManual.isChecked()
        else:
            showA = showB = showM = True

        boxes: List[Box] = []
        # Model A headless
        if self.wrapperA.model is not None and (mode not in ("Manual only", "Model B only", "Model B + Manual") and include_preds):
            if not respect_vis or showA:
                if "Model B only" not in mode and "Model B +" not in mode:  # not B-only
                    preds = self.wrapperA.predict(img_path, conf=self.spin_confA.value(), iou=self.spin_iouA.value(), max_det=self.spin_maxA.value())
                    for b in preds:
                        b.source = "modelA"
                        if not b.label:
                            b.label = self.wrapperA.class_names.get(b.cls, str(b.cls))
                        # mode filter
                        if mode == "Model B only" or mode == "Model B + Manual":
                            continue
                        boxes.append(b)
        if self.wrapperB.model is not None and (mode not in ("Manual only", "Model A only", "Model A + Manual") and include_preds):
            if not respect_vis or showB:
                if "Model A only" not in mode and "Model A +" not in mode:
                    preds = self.wrapperB.predict(img_path, conf=self.spin_confB.value(), iou=self.spin_iouB.value(), max_det=self.spin_maxB.value())
                    for b in preds:
                        b.source = "modelB"
                        if not b.label:
                            b.label = self.wrapperB.class_names.get(b.cls, str(b.cls))
                        if mode == "Model A only" or mode == "Model A + Manual":
                            continue
                        boxes.append(b)

        # Manual from cache (respect mode)
        if mode not in ("Model A only", "Model B only"):
            if not respect_vis or showM:
                if img_path in self.manual_boxes_cache:
                    for mb in self.manual_boxes_cache[img_path]:
                        # filter by mode
                        if mode == "Model A only" or mode == "Model B only":
                            continue
                        boxes.append(Box(mb.x1, mb.y1, mb.x2, mb.y2, mb.cls, mb.conf, "manual", mb.label))
                # also if current image is this path, include live manual boxes (not yet cached)
                if img_path == self.current_image_path:
                    for it in self.scene.box_items:
                        if it.box.source == "manual":
                            it.sync_box()
                            b = it.box
                            # check duplicate vs cache? Keep both but dedup later
                            # Only add if not already in cache (compare coords)
                            already = any(abs(b.x1 - c.x1) < 1 and abs(b.y1 - c.y1) < 1 for c in boxes if c.source == "manual")
                            if not already:
                                boxes.append(Box(b.x1, b.y1, b.x2, b.y2, b.cls, b.conf, "manual", b.label))
        # also if mode is manual-only, ensure only manual
        if mode == "Manual only":
            boxes = [b for b in boxes if b.source == "manual"]
        elif mode == "Model A only":
            boxes = [b for b in boxes if b.source == "modelA"]
        elif mode == "Model B only":
            boxes = [b for b in boxes if b.source == "modelB"]

        # dedup if needed
        if self.chk_dedup.isChecked() and "Combined" in mode and len(boxes) > 1:
            iou = self.spin_dedup_iou.value()
            preds = [b for b in boxes if b.source != "manual"]
            manuals = [b for b in boxes if b.source == "manual"]
            preds = self._nms_merge(preds, iou)
            boxes = preds + manuals

        return boxes, w, h

    def save_current(self):
        if not self.current_image_path or self.current_img_w == 0:
            QMessageBox.warning(self, "No Image", "No image loaded.")
            return
        # Ensure cache up to date for current
        self.manual_boxes_cache[self.current_image_path] = [b for b in self.scene.get_all_boxes() if b.source == "manual"]

        # Use combined logic for current view OR headless re-run? For current we use live scene boxes (includes drawn/moved/deleted)
        # This respects deletions and manual edits on current image.
        mode = self.cmb_save_mode.currentText()
        boxes = self._get_combined_boxes_for_current()
        # For current image, if mode is Combined and we have models loaded but scene may have no preds due to visibility filter? _get_combined respects mode, not visibility unless checked.
        # Count
        cntA = len([b for b in boxes if b.source == "modelA"])
        cntB = len([b for b in boxes if b.source == "modelB"])
        cntM = len([b for b in boxes if b.source == "manual"])
        self.log(f"Save Current: mode={mode} combined A={cntA} B={cntB} Manual={cntM} Total={len(boxes)} (dedup={self.chk_dedup.isChecked()})")

        if not boxes:
            if QMessageBox.question(self, "No Boxes", f"No boxes to save for mode '{mode}'. Save empty .txt (negative sample)?", QMessageBox.Yes | QMessageBox.No) != QMessageBox.Yes:
                return

        save_dir = self.save_dir_edit.text().strip() or os.path.dirname(self.current_image_path)
        if not os.path.isdir(save_dir):
            try:
                os.makedirs(save_dir, exist_ok=True)
            except Exception as e:
                QMessageBox.critical(self, "Save Failed", f"Cannot create save dir:\n{e}")
                return

        base = os.path.splitext(os.path.basename(self.current_image_path))[0]
        out_path = os.path.join(save_dir, base + ".txt")
        try:
            with open(out_path, "w") as f:
                for b in boxes:
                    cls, xc, yc, ww, hh = b.to_yolo(self.current_img_w, self.current_img_h)
                    f.write(f"{cls} {xc:.6f} {yc:.6f} {ww:.6f} {hh:.6f}\n")
            self.log(f"Saved COMBINED YOLO: {out_path} ({len(boxes)} boxes A={cntA} B={cntB} M={cntM}) [W={self.current_img_w} H={self.current_img_h}]")
            self.lbl_saveStatus.setText(f"Saved {len(boxes)} boxes (A:{cntA} B:{cntB} M:{cntM}) → {out_path}")
            self._mark_saved(self.current_image_path, True)
            QMessageBox.information(self, "Saved Combined", f"Saved {len(boxes)} boxes (A={cntA} B={cntB} M={cntM}) to:\n{out_path}\n\nMode: {mode}\nDedup: {self.chk_dedup.isChecked()} IoU={self.spin_dedup_iou.value():.2f}\nFormat: <cls> <x_center> <y_center> <w> <h> normalized\nEach image gets its own .txt (per-image).")
        except Exception as e:
            QMessageBox.critical(self, "Save Failed", str(e))
            self.log(f"Save failed: {e}")

    def save_current_as(self):
        if not self.current_image_path or self.current_img_w == 0:
            QMessageBox.warning(self, "No Image", "No image loaded.")
            return
        self.manual_boxes_cache[self.current_image_path] = [b for b in self.scene.get_all_boxes() if b.source == "manual"]
        mode = self.cmb_save_mode.currentText()
        boxes = self._get_combined_boxes_for_current()
        cntA = len([b for b in boxes if b.source == "modelA"])
        cntB = len([b for b in boxes if b.source == "modelB"])
        cntM = len([b for b in boxes if b.source == "manual"])
        default_dir = self.save_dir_edit.text().strip() or os.path.dirname(self.current_image_path)
        base = os.path.splitext(os.path.basename(self.current_image_path))[0]
        default_path = os.path.join(default_dir, base + ".txt")
        out_path, _ = QFileDialog.getSaveFileName(self, f"Save Combined YOLO for {base} ({mode})", default_path, "YOLO txt (*.txt);;All Files (*)")
        if not out_path:
            return
        # ensure dir exists
        out_dir = os.path.dirname(out_path)
        if out_dir and not os.path.isdir(out_dir):
            try:
                os.makedirs(out_dir, exist_ok=True)
            except Exception as e:
                QMessageBox.critical(self, "Save Failed", str(e))
                return
        try:
            with open(out_path, "w") as f:
                for b in boxes:
                    cls, xc, yc, ww, hh = b.to_yolo(self.current_img_w, self.current_img_h)
                    f.write(f"{cls} {xc:.6f} {yc:.6f} {ww:.6f} {hh:.6f}\n")
            self.log(f"Saved As: {out_path} ({len(boxes)} boxes A={cntA} B={cntB} M={cntM})")
            self.lbl_saveStatus.setText(f"Saved As {len(boxes)} boxes → {out_path}")
            self._mark_saved(self.current_image_path, True)
            QMessageBox.information(self, "Saved As", f"Saved {len(boxes)} boxes (A={cntA} B={cntB} M={cntM}) to:\n{out_path}")
        except Exception as e:
            QMessageBox.critical(self, "Save Failed", str(e))

    def save_all(self):
        if not self.image_paths:
            QMessageBox.warning(self, "No Images", "No image folder loaded.")
            return
        save_dir = self.save_dir_edit.text().strip() or os.path.dirname(self.image_paths[0])
        if not os.path.isdir(save_dir):
            try:
                os.makedirs(save_dir, exist_ok=True)
            except Exception as e:
                QMessageBox.critical(self, "Save Dir", str(e))
                return
        mode = self.cmb_save_mode.currentText()
        dedup_str = f" Dedup IoU={self.spin_dedup_iou.value():.2f}" if self.chk_dedup.isChecked() else ""
        msg = f"Save COMBINED YOLO for ALL {len(self.image_paths)} images to:\n{save_dir}\n\nMode: {mode}{dedup_str}\nOne .txt per image: <image_name>.txt\nThis will re-run inference per image (sliders A/B, maxDet) and merge A+B+Manual (per-image).\nInclude predictions: {self.chk_save_include_preds.isChecked()}\nManual boxes per image (drawn) will be included.\nProceed?"
        if QMessageBox.question(self, "Save ALL Combined (Per-Image)", msg, QMessageBox.Yes | QMessageBox.No) != QMessageBox.Yes:
            return
        self.progress.setVisible(True)
        self.progress.setRange(0, len(self.image_paths))
        self.progress.setValue(0)
        total_boxes = 0
        saved = 0
        # Save current scene's manual cache first
        if self.current_image_path:
            self.manual_boxes_cache[self.current_image_path] = [b for b in self.scene.get_all_boxes() if b.source == "manual"]

        orig_idx = self.current_idx

        for idx, img_path in enumerate(self.image_paths):
            self.progress.setValue(idx)
            QApplication.processEvents()
            boxes, w, h = self._get_headless_boxes_for_image(img_path)
            if w == 0 or h == 0:
                self.log(f"Skip {img_path}: cannot get size")
                continue
            base = os.path.splitext(os.path.basename(img_path))[0]
            out_path = os.path.join(save_dir, base + ".txt")
            try:
                with open(out_path, "w") as f:
                    for b in boxes:
                        cls, xc, yc, w_n, h_n = b.to_yolo(w, h)
                        f.write(f"{cls} {xc:.6f} {yc:.6f} {w_n:.6f} {h_n:.6f}\n")
                saved += 1
                total_boxes += len(boxes)
                self.saved_status[img_path] = True
                cntA = len([b for b in boxes if b.source == "modelA"])
                cntB = len([b for b in boxes if b.source == "modelB"])
                cntM = len([b for b in boxes if b.source == "manual"])
                self.log(f"Saved {idx+1}/{len(self.image_paths)}: {base}.txt A={cntA} B={cntB} M={cntM} total={len(boxes)}")
            except Exception as e:
                self.log(f"Save failed for {img_path}: {e}")
        self.progress.setValue(len(self.image_paths))
        QTimer.singleShot(1200, lambda: self.progress.setVisible(False))
        self._update_per_image_status()
        self._refresh_image_list_icons()
        if orig_idx >= 0:
            self.show_image(orig_idx)
        self.log(f"Save ALL Combined done: {saved}/{len(self.image_paths)} files, {total_boxes} total boxes -> {save_dir} (one txt per image)")
        self.lbl_saveStatus.setText(f"Saved ALL Combined: {saved} files, {total_boxes} boxes → {save_dir}")
        QMessageBox.information(self, "Save ALL Combined Done", f"Saved {saved}/{len(self.image_paths)} COMBINED YOLO txt files (one per image).\nMode: {mode}{dedup_str}\nTotal boxes: {total_boxes}\nFolder: {save_dir}\nEach image → <image>.txt")

    def export_per_image_picker(self):
        if not self.image_paths:
            QMessageBox.warning(self, "No Images", "No image folder loaded.")
            return
        # Dialog with checklist per image
        from PyQt5.QtWidgets import QDialog, QDialogButtonBox, QCheckBox, QScrollArea, QVBoxLayout, QLabel
        dlg = QDialog(self)
        dlg.setWindowTitle("Export Per-Image Picker — Combined YOLO")
        dlg.resize(520, 500)
        v = QVBoxLayout(dlg)
        v.addWidget(QLabel(f"Select images to export (Combined: {self.cmb_save_mode.currentText()}):\nOne .txt per image will be written to save dir."))
        # checks
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        inner = QWidget()
        iv = QVBoxLayout(inner)
        checks = []
        for p in self.image_paths:
            cb = QCheckBox(os.path.basename(p))
            cb.setChecked(True)
            # show saved status
            if self.saved_status.get(p):
                cb.setText(f"{os.path.basename(p)}  ✓ saved")
                cb.setStyleSheet("color: #0a0;")
            # store path
            cb.setProperty("img_path", p)
            iv.addWidget(cb)
            checks.append(cb)
        iv.addStretch()
        scroll.setWidget(inner)
        v.addWidget(scroll, 1)
        # buttons: Select All / None
        hBtns = QHBoxLayout()
        btn_all = QPushButton("Select All")
        btn_none = QPushButton("Select None")
        btn_all.clicked.connect(lambda: [c.setChecked(True) for c in checks])
        btn_none.clicked.connect(lambda: [c.setChecked(False) for c in checks])
        hBtns.addWidget(btn_all)
        hBtns.addWidget(btn_none)
        hBtns.addStretch()
        v.addLayout(hBtns)
        # save dir info
        save_dir = self.save_dir_edit.text().strip() or os.path.dirname(self.image_paths[0])
        lblDir = QLabel(f"Save dir: {save_dir}")
        lblDir.setStyleSheet("font-size: 10px; color: #555;")
        v.addWidget(lblDir)
        # dialog buttons
        bbox = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        bbox.accepted.connect(dlg.accept)
        bbox.rejected.connect(dlg.reject)
        v.addWidget(bbox)
        if dlg.exec_() != QDialog.Accepted:
            return
        selected = [c.property("img_path") for c in checks if c.isChecked()]
        if not selected:
            QMessageBox.warning(self, "No Selection", "No images selected.")
            return
        # proceed to save selected only
        save_dir = self.save_dir_edit.text().strip() or os.path.dirname(self.image_paths[0])
        if not os.path.isdir(save_dir):
            try:
                os.makedirs(save_dir, exist_ok=True)
            except Exception as e:
                QMessageBox.critical(self, "Save Dir", str(e))
                return
        self.progress.setVisible(True)
        self.progress.setRange(0, len(selected))
        self.progress.setValue(0)
        total_boxes = 0
        saved = 0
        if self.current_image_path:
            self.manual_boxes_cache[self.current_image_path] = [b for b in self.scene.get_all_boxes() if b.source == "manual"]
        orig_idx = self.current_idx
        for idx, img_path in enumerate(selected):
            self.progress.setValue(idx)
            QApplication.processEvents()
            boxes, w, h = self._get_headless_boxes_for_image(img_path)
            if w == 0 or h == 0:
                self.log(f"Skip {img_path}: cannot get size")
                continue
            base = os.path.splitext(os.path.basename(img_path))[0]
            out_path = os.path.join(save_dir, base + ".txt")
            try:
                with open(out_path, "w") as f:
                    for b in boxes:
                        cls, xc, yc, w_n, h_n = b.to_yolo(w, h)
                        f.write(f"{cls} {xc:.6f} {yc:.6f} {w_n:.6f} {h_n:.6f}\n")
                saved += 1
                total_boxes += len(boxes)
                self.saved_status[img_path] = True
                self.log(f"Picker Export {idx+1}/{len(selected)}: {base}.txt {len(boxes)} boxes")
            except Exception as e:
                self.log(f"Save failed for {img_path}: {e}")
        self.progress.setValue(len(selected))
        QTimer.singleShot(1200, lambda: self.progress.setVisible(False))
        self._update_per_image_status()
        self._refresh_image_list_icons()
        if orig_idx >= 0:
            self.show_image(orig_idx)
        self.log(f"Picker Export done: {saved}/{len(selected)} files, {total_boxes} boxes -> {save_dir}")
        self.lbl_saveStatus.setText(f"Picker Export: {saved} files → {save_dir}")
        QMessageBox.information(self, "Picker Export Done", f"Saved {saved}/{len(selected)} combined YOLO files (one per image).\nFolder: {save_dir}")

    def show_image_list_context_menu(self, pos):
        if not self.image_paths:
            return
        global_pos = self.list_images.mapToGlobal(pos)
        item = self.list_images.itemAt(pos)
        if not item:
            return
        row = self.list_images.row(item)
        if not (0 <= row < len(self.image_paths)):
            return
        img_path = self.image_paths[row]
        base = os.path.basename(img_path)
        menu = QMenu(self)
        act_open = menu.addAction(f"Go to {base}")
        act_open.triggered.connect(lambda: self.show_image(row))
        menu.addSeparator()
        act_save_this = menu.addAction(f"💾 Save Combined for THIS image (→ {base}.txt)")
        act_save_this.setToolTip("Save combined (A+B+Manual) for this single image to save_dir/<name>.txt (per-image)")
        act_save_this.triggered.connect(lambda: self.save_single_image_via_path(img_path))
        act_save_as_this = menu.addAction(f"Save As... for {base}")
        act_save_as_this.triggered.connect(lambda: self.save_single_image_as_via_path(img_path))
        act_reveal = menu.addAction("Reveal in save dir")
        # check if txt exists
        save_dir = self.save_dir_edit.text().strip() or os.path.dirname(img_path)
        txt_path = os.path.join(save_dir, os.path.splitext(base)[0] + ".txt")
        exists = os.path.exists(txt_path)
        act_reveal.setEnabled(exists)
        if exists:
            act_reveal.triggered.connect(lambda: self._reveal_in_folder(txt_path))
        menu.addSeparator()
        saved = self.saved_status.get(img_path, False)
        lbl = QLabel(f"  {'✓ Saved' if saved else '○ Not saved'} — Right-click any image to save that one image's combined YOLO individually (per-image).  ")
        lbl.setStyleSheet("font-size: 9px; color: #666; padding: 4px;")
        wa = QWidgetAction(menu)
        wa.setDefaultWidget(lbl)
        menu.addAction(wa)
        menu.exec_(global_pos)

    def save_single_image_via_path(self, img_path: str):
        # Save combined for arbitrary image path (not necessarily current) to save_dir
        save_dir = self.save_dir_edit.text().strip() or os.path.dirname(img_path)
        if not os.path.isdir(save_dir):
            try:
                os.makedirs(save_dir, exist_ok=True)
            except Exception as e:
                QMessageBox.critical(self, "Save Dir", str(e))
                return
        # If it's current image, use live scene boxes (so deletions/drawn apply)
        if img_path == self.current_image_path:
            self.save_current()
            return
        # Else headless
        boxes, w, h = self._get_headless_boxes_for_image(img_path)
        if w == 0 or h == 0:
            QMessageBox.warning(self, "Save Failed", f"Cannot get size for {img_path}")
            return
        base = os.path.splitext(os.path.basename(img_path))[0]
        out_path = os.path.join(save_dir, base + ".txt")
        try:
            with open(out_path, "w") as f:
                for b in boxes:
                    cls, xc, yc, w_n, h_n = b.to_yolo(w, h)
                    f.write(f"{cls} {xc:.6f} {yc:.6f} {w_n:.6f} {h_n:.6f}\n")
            cntA = len([b for b in boxes if b.source == "modelA"])
            cntB = len([b for b in boxes if b.source == "modelB"])
            cntM = len([b for b in boxes if b.source == "manual"])
            self.log(f"Saved single (per-image): {out_path} A={cntA} B={cntB} M={cntM}")
            self._mark_saved(img_path, True)
            QMessageBox.information(self, "Saved Single", f"Saved {len(boxes)} boxes (A={cntA} B={cntB} M={cntM}) for\n{base}\n→ {out_path}\nMode: {self.cmb_save_mode.currentText()}")
        except Exception as e:
            QMessageBox.critical(self, "Save Failed", str(e))

    def save_single_image_as_via_path(self, img_path: str):
        boxes, w, h = self._get_headless_boxes_for_image(img_path)
        if img_path == self.current_image_path:
            # for current, use live combined
            boxes = self._get_combined_boxes_for_current()
            w, h = self.current_img_w, self.current_img_h
        if w == 0 or h == 0:
            QMessageBox.warning(self, "Save Failed", f"Cannot get size for {img_path}")
            return
        base = os.path.splitext(os.path.basename(img_path))[0]
        default_dir = self.save_dir_edit.text().strip() or os.path.dirname(img_path)
        default_path = os.path.join(default_dir, base + ".txt")
        out_path, _ = QFileDialog.getSaveFileName(self, f"Save Combined YOLO for {base}", default_path, "YOLO txt (*.txt);;All Files (*)")
        if not out_path:
            return
        out_dir = os.path.dirname(out_path)
        if out_dir and not os.path.isdir(out_dir):
            try:
                os.makedirs(out_dir, exist_ok=True)
            except Exception as e:
                QMessageBox.critical(self, "Save Failed", str(e))
                return
        try:
            with open(out_path, "w") as f:
                for b in boxes:
                    cls, xc, yc, w_n, h_n = b.to_yolo(w, h)
                    f.write(f"{cls} {xc:.6f} {yc:.6f} {w_n:.6f} {h_n:.6f}\n")
            self.log(f"Saved As single: {out_path} {len(boxes)} boxes")
            self._mark_saved(img_path, True)
            QMessageBox.information(self, "Saved As", f"Saved {len(boxes)} boxes for {base} →\n{out_path}")
        except Exception as e:
            QMessageBox.critical(self, "Save Failed", str(e))

    def _reveal_in_folder(self, path: str):
        # Try to open folder (linux xdg-open)
        folder = os.path.dirname(path)
        try:
            import subprocess
            subprocess.Popen(["xdg-open", folder])
        except Exception:
            self.log(f"Reveal folder: {folder}")



    def _get_image_size(self, path: str) -> Tuple[int, int]:
        # fast via QImage or PIL
        qimg = QImage(path)
        if not qimg.isNull():
            return qimg.width(), qimg.height()
        if Image is not None:
            try:
                with Image.open(path) as im:
                    return im.size  # w,h
            except Exception:
                pass
        return 0, 0

    def export_classes(self):
        save_dir = self.save_dir_edit.text().strip() or (os.path.dirname(self.image_paths[0]) if self.image_paths else "/tmp")
        path, _ = QFileDialog.getSaveFileName(self, "Export classes.txt", os.path.join(save_dir, "classes.txt"), "Text Files (*.txt);;All (*)")
        if not path:
            return
        try:
            with open(path, "w") as f:
                for i, name in enumerate(self.class_names_combined):
                    f.write(f"{name}\n")
            self.log(f"Exported classes.txt ({len(self.class_names_combined)} names) -> {path}")
            QMessageBox.information(self, "Exported", f"Exported {len(self.class_names_combined)} class names to:\n{path}")
        except Exception as e:
            QMessageBox.critical(self, "Export Failed", str(e))

    # ---------- Misc (window fitting) ----------
    def resizeEvent(self, event):
        super().resizeEvent(event)
        # Clamp to available geometry if user resizes too large (keeps window on-screen)
        try:
            scr = QApplication.primaryScreen().availableGeometry() if QApplication.primaryScreen() else None
            if scr and not self.isMaximized():
                if self.width() > scr.width() or self.height() > scr.height():
                    # Shrink to fit with 20px margin
                    w = min(self.width(), scr.width() - 20)
                    h = min(self.height(), scr.height() - 40)
                    QTimer.singleShot(0, lambda: self.resize(w, h))
        except Exception:
            pass

    def moveEvent(self, event):
        super().moveEvent(event)
        # Keep window fully visible when moved
        try:
            scr = QApplication.primaryScreen().availableGeometry() if QApplication.primaryScreen() else None
            if scr and not self.isMaximized():
                x, y = self.x(), self.y()
                # Clamp
                nx = max(scr.x(), min(x, scr.x() + scr.width() - 100))
                ny = max(scr.y(), min(y, scr.y() + scr.height() - 100))
                if nx != x or ny != y:
                    QTimer.singleShot(0, lambda: self.move(nx, ny))
        except Exception:
            pass

    def closeEvent(self, event):
        # check unsaved manual boxes?
        # Save cache?
        super().closeEvent(event)


def parse_args():
    import argparse
    p = argparse.ArgumentParser(description="YOLO Dual Viewer")
    p.add_argument("--modelA", type=str, default="", help="Path to model A")
    p.add_argument("--modelB", type=str, default="", help="Path to model B")
    p.add_argument("--images", type=str, default="", help="Image folder")
    p.add_argument("--save", type=str, default="", help="Save folder")
    return p.parse_args()


def main():
    args = parse_args()
    # High-DPI must be set before QApplication
    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)
    app = QApplication(sys.argv)

    win = MainWindow()
    win.show()
    # apply extra handlers after show (need table click)
    win.init_table_click_handler()
    # double click to edit class
    win.table.cellDoubleClicked.connect(win._edit_class_for_row)

    # auto-load from args
    if args.modelA:
        win.modelA_edit.setText(args.modelA)
        win.load_model("A")
    else:
        for cand in [
            "/home/trendzlink/Downloads/best_07-25_122104/best.pt",
            "/home/trendzlink/yolo_tool/best_07-25_122104/best.pt",
            "./best.pt",
        ]:
            if os.path.exists(cand):
                win.modelA_edit.setText(cand)
                break
        # don't auto load to avoid heavy, but could:
        # win.load_model("A")
    if args.modelB:
        win.modelB_edit.setText(args.modelB)
        win.load_model("B")
    if args.images and os.path.isdir(args.images):
        win.image_dir_edit.setText(args.images)
        win.load_image_dir(args.images)
    if args.save:
        win.save_dir_edit.setText(args.save)

    # quick style
    app.setStyle("Fusion")
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
