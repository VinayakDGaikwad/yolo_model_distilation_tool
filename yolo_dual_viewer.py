#!/usr/bin/env python3
"""
YOLO Dual Model Viewer & Annotator - PyQt5 GUI
-----------------------------------------------
Backwards-compatible launcher for the `yolo_viewer` package.

Features:
- Load 2 YOLO models (ultralytics YOLO .pt/.onnx)
- Browse image folder, navigate images (Prev/Next, slider, list)
- Per-model confidence interval [low, high] (0.01–0.95) — two sliders/spins BETWEEN two values instead of single threshold, both ways slider↔spin synced (0.01 granularity, auto-clamp low≤high)
- Per-model NMS IoU slider + MaxDet spinbox (1..2000, default 400, supports >300)
- Live cross-model overlap slider keeps only the higher-confidence box for duplicate A/B detections (0% disables)
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
    python -m yolo_viewer

Author: Muse Spark
"""
import os
import sys

# Allow running from anywhere: make the repo root (where yolo_viewer/ lives)
# importable no matter the current working directory.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from yolo_viewer.app import main

if __name__ == "__main__":
    main()