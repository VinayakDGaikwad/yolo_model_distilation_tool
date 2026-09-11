#!/usr/bin/env python3
"""Headless test: inference + YOLO save without GUI (uses venv python)."""
import os, glob, sys
sys.path.insert(0, os.path.dirname(__file__))
from yolo_dual_viewer import YoloWrapper, Box

def test_wrapper():
    model_path = "/home/trendzlink/Downloads/best_07-25_122104/best.pt"
    img_dir = "/home/trendzlink/Downloads/tz_batch_test_03_9_2025_020139/tz_batch_test_03_9_2025"
    if not os.path.exists(model_path):
        print(f"SKIP: model not found {model_path}")
        return
    imgs = sorted(glob.glob(os.path.join(img_dir, "*.jpg")))
    if not imgs:
        print(f"SKIP: no images in {img_dir}")
        return
    w = YoloWrapper()
    assert w.load(model_path), f"load failed: {w.load_error}"
    print(f"Loaded {model_path}: names={w.class_names}")
    # test max_det >300
    for max_det in [300, 400, 600]:
        boxes = w.predict(imgs[0], conf=0.25, iou=0.45, max_det=max_det)
        print(f"max_det={max_det} -> {len(boxes)} boxes (conf 0.25)")
        # ensure we can get >300 when max_det 400 (model gives 336)
        if max_det == 400:
            assert len(boxes) >= 300, "expected >=300 boxes for dense image at max_det 400"
    # test YOLO conversion and save
    out_dir = "/tmp/yolo_tool_test"
    os.makedirs(out_dir, exist_ok=True)
    boxes = w.predict(imgs[0], conf=0.25, max_det=400)
    # simulate manual box
    manual = Box(100, 100, 300, 300, 0, 1.0, "manual", "object")
    boxes.append(manual)
    # need image size
    from PyQt5.QtGui import QImage
    q = QImage(imgs[0])
    W, H = q.width(), q.height()
    print(f"Image {os.path.basename(imgs[0])} size {W}x{H}")
    base = os.path.splitext(os.path.basename(imgs[0]))[0]
    out_path = os.path.join(out_dir, base + ".txt")
    with open(out_path, "w") as f:
        for b in boxes:
            cls, xc, yc, ww, hh = b.to_yolo(W, H)
            assert 0 <= xc <= 1 and 0 <= yc <= 1, "normalized out of range"
            f.write(f"{cls} {xc:.6f} {yc:.6f} {ww:.6f} {hh:.6f}\n")
    print(f"Saved {len(boxes)} boxes to {out_path}")
    # verify file
    with open(out_path) as f:
        lines = f.readlines()
    print(f"File has {len(lines)} lines, first 3: {lines[:3]}")
    assert len(lines) == len(boxes)
    print("TEST PASSED")

if __name__ == "__main__":
    test_wrapper()
