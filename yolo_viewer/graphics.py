"""Graphics scene / view types for the YOLO Dual Model Viewer."""

from typing import List, Optional, Tuple

from PyQt5.QtWidgets import (
    QGraphicsItem, QGraphicsPixmapItem, QGraphicsRectItem, QGraphicsScene,
    QGraphicsTextItem, QGraphicsView, QMenu,
)
from PyQt5.QtGui import QBrush, QColor, QFont, QPainter, QPen, QPixmap
from PyQt5.QtCore import QPointF, QRectF, Qt, pyqtSignal

from yolo_viewer.constants import SOURCE_COLORS
from yolo_viewer.models import Box

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
                    scene.remove_box_item(self)
            else:
                scene.remove_box_item(self)
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
        self.box_items.clear()
        self.clear()
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


