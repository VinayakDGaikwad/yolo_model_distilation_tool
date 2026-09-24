"""Qt-independent data model and YOLO wrapper used by the GUI.

Everything here has no dependency on PyQt5, so it can be reused from
headless scripts (see ``test_headless.py``).
"""

import os
from typing import Dict, List, Optional

try:
    from ultralytics import YOLO
    ULTRALYTICS_AVAILABLE = True
except Exception:
    YOLO = None
    ULTRALYTICS_AVAILABLE = False


class Box:
    """A detection / annotation box in absolute image pixel coordinates."""

    __slots__ = ("x1", "y1", "x2", "y2", "cls", "conf", "source", "label")

    def __init__(self, x1, y1, x2, y2, cls, conf, source="", label: Optional[str] = None):
        self.x1 = float(x1)
        self.y1 = float(y1)
        self.x2 = float(x2)
        self.y2 = float(y2)
        self.cls = int(cls)
        self.conf = float(conf)
        self.source = source
        self.label = label

    def to_yolo(self, img_w: float, img_h: float) -> tuple:
        """Return (cls, x_center, y_center, width, height) normalized to 0-1."""
        xc = (self.x1 + self.x2) / 2.0 / img_w
        yc = (self.y1 + self.y2) / 2.0 / img_h
        w = (self.x2 - self.x1) / img_w
        h = (self.y2 - self.y1) / img_h
        xc = min(max(xc, 0), 1)
        yc = min(max(yc, 0), 1)
        w = min(max(w, 0), 1)
        h = min(max(h, 0), 1)
        return self.cls, xc, yc, w, h

    def __repr__(self):
        return (f"Box(x1={self.x1:.0f},y1={self.y1:.0f},x2={self.x2:.0f},"
                f"y2={self.y2:.0f},cls={self.cls},conf={self.conf:.2f},"
                f"source={self.source},label={self.label})")


class YoloWrapper:
    """Thin wrapper around ultralytics YOLO so the GUI never touches torch directly."""

    def __init__(self, model_path: str = ""):
        self.model_path = model_path
        self.model = None
        self.load_error = ""
        self.class_names: Dict[int, str] = {}

    def load(self, path: str) -> bool:
        self.model_path = path
        self.model = None
        self.class_names = {}
        self.load_error = ""
        if not path or not os.path.exists(path):
            self.load_error = f"File not found: {path}"
            return False
        if not ULTRALYTICS_AVAILABLE or YOLO is None:
            self.load_error = "ultralytics not installed (pip install ultralytics)"
            return False
        try:
            self.model = YOLO(path)
            names = getattr(self.model, "names", None)
            if isinstance(names, dict):
                self.class_names = {int(k): str(v) for k, v in names.items()}
            elif isinstance(names, list):
                self.class_names = {i: str(n) for i, n in enumerate(names)}
            else:
                self.class_names = {0: "object"}
            return True
        except Exception as e:
            self.load_error = str(e)
            self.model = None
            return False

    def predict(self, image_path: str, conf: float = 0.25, iou: float = 0.45,
                max_det: int = 400) -> List[Box]:
        if self.model is None or not ULTRALYTICS_AVAILABLE:
            return []
        try:
            results = self.model.predict(source=image_path, conf=conf, iou=iou,
                                         max_det=max_det, verbose=False)
        except Exception as e:
            print(f"[YoloWrapper] predict error: {e}")
            return []
        boxes: List[Box] = []
        for r in results:
            if getattr(r, "boxes", None) is None:
                continue
            for b in r.boxes:
                x1, y1, x2, y2 = [float(v) for v in b.xyxy[0]]
                cls = int(b.cls[0])
                c = float(b.conf[0])
                boxes.append(Box(x1, y1, x2, y2, cls, c, "",
                                 self.class_names.get(cls, str(cls))))
        return boxes