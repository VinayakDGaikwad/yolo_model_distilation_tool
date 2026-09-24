# YOLO Dual Model Viewer & Annotator

PyQt5 GUI to compare **two YOLO models** on same images, tune confidence / `max_det` (>300), manually fix missed detections by drawing boxes, and export **YOLO format**.

![Python](https://img.shields.io/badge/python-3.8%2B-blue) ![PyQt5](https://img.shields.io/badge/PyQt5-5.14%2B-green) ![YOLO](https://img.shields.io/badge/YOLO-ultralytics%208.x-orange)

## Features
- Load **Model A + Model B** (`.pt` / `.onnx` via `ultralytics.YOLO`). Either can be empty → manual-only mode.
- Load every `.pt`, `.onnx`, or `.engine` file from a **Model Folder** and combine their detections in one view. The legacy Model A/B controls remain available when no folder is selected.
- **Per-model controls**: Confidence slider (0.01–0.95, synced QSlider + QDoubleSpinBox), IoU (NMS), **MaxDet 1–2000 (default 400, supports >300)**.
- Folder mode uses one **Universal conf** slider for all loaded models.
- **Live multi-model overlap control**: Overlap A/B slider suppresses the lower-confidence same-class box when any two folder models overlap; set to 0% to disable.
- **Show/Hide** per source with distinct colors (green = A, blue = B, red = manual).
- Image folder browser (jpg/png/bmp/tiff/webp), `Prev/Next`, list + slider navigation.
- **Interactive canvas** (QGraphicsView/Scene):
  - Zoom: `Ctrl+Wheel`, Buttons `+/-`, `Fit`
  - Pan: `Middle-drag` or `Space+Drag`
  - Select/move boxes (drag), Delete (`Del`/`Backspace` or table ✕), clear per source.
- **Draw new boxes**: Toggle `Draw Box (D)` → crosshair → click-drag on image. Assign `Class ID` + `Name` (combo synced to model `names`). New boxes are `manual` (conf 1.0, red).
- **Box table**: filtered view (`All`/`A`/`B`/`Manual` + class filter), columns `Src | Class | Conf | xyxy | Area | Del`. Click row → selects & centers box. Double-click class → edit ID. Delete via ✕.
- **Counts** bar: `ModelA: x | ModelB: y | Manual: z | Visible/Total`.
- **Save YOLO**: `Save Current` writes `<image>.txt` with `cls x_center y_center w h` normalized. Option to include/exclude predictions. `Save ALL` iterates folder (re-runs inference with current sliders) + manual cache. Custom save dir, `Export classes.txt`.

## Install
```bash
# system python (has PyQt5 5.14)
pip install ultralytics  # pulls torch, opencv

# OR use provided venv (already has ultralytics 8.1.0 + torch + PyQt5)
 /home/trendzlink/venv/envision/bin/pip install ultralytics  # if needed
```

System check:
```bash
/home/trendzlink/venv/envision/bin/python -c "import ultralytics, PyQt5, cv2; print('ok')"
```

## Run
```bash
# From a checkout (works without installing the package)
python -m yolo_viewer

# via venv (recommended, has ultralytics)
 /home/trendzlink/venv/envision/bin/python yolo_dual_viewer.py
 /home/trendzlink/venv/envision/bin/python yolo_dual_viewer.py --models-dir /path/to/model_folder --images /path/to/images
 /home/trendzlink/venv/envision/bin/python yolo_dual_viewer.py --modelA /path/to/best.pt --images /path/to/images --modelB /path/to/second.pt

# via system (manual-only if ultralytics missing)
 python3 yolo_dual_viewer.py
 bash run.sh
```

Launch scripts:
- `run.sh` → venv python
- `run_system.sh` → system `python3`

## Usage Workflow
1. **Browse** Model A (`.pt` from `Downloads/best_07-25_122104/best.pt`) → `Load A`. Optionally Model B → `Load B`. Check `Show` boxes.
2. **Browse** image folder (`tz_batch_test.../tz_batch_test_03_9_2025`) → auto runs inference.
3. Tune **Conf sliders** (25% default) + **MaxDet** (400 default, bump to 600 if image has >300 objects). IoU 0.45 default. `Re-run` auto on change (400ms debounce) or manual `⟳`.
4. Inspect counts, table, visual boxes. Zoom/Pan.
5. **Draw missing**: set `Class ID`/`Name` (combo matches model names), click `Draw Box` (D) → drag rectangle → release → new red manual box appears + table updates. Toggle visibility to compare.
6. **Edit**: click box → yellow highlight → drag to move; table row mirrors selection; delete via `Del` or table `✕`; double-click class cell to change ID.
7. **Save**: choose save dir (default image folder), check `Include predictions` → `💾 Save Current` writes YOLO txt for current image. `Save ALL` batches all images. `Export classes.txt` dumps `class_names_combined`.

## YOLO Format
```
<class_id> <x_center> <y_center> <width> <height>  # normalized 0-1
```
Example `image.txt`:
```
0 0.501953 0.416667 0.156250 0.277778
0 0.312500 0.700000 0.080000 0.120000
```

Image `1280x720` box `100,200,300,400` → `0 0.156250 0.416667 0.156250 0.277778`.

## File Structure
```
yolo_tool/
  yolo_dual_viewer.py   # backwards-compatible launcher (python yolo_dual_viewer.py)
  yolo_viewer/
    __init__.py         # public package exports
    __main__.py         # python -m yolo_viewer entrypoint
    app.py              # package entrypoint (parse_args + main)
    models.py           # Box and YoloWrapper, independent of Qt
    constants.py        # shared constants (colors, image extensions)
    graphics.py         # BoxItem / AnnotScene / AnnotView view classes
    main_window.py      # MainWindow GUI
  requirements.txt
  README.md
  run.sh
  run_system.sh
```

## Tips
- If `max_det` < actual objects, YOLO truncates top-conf boxes. Raise to 500-1000 for dense scenes (verified `best.pt` gives 336 boxes on sample).
- Confidence slider low (0.10) shows more low-conf preds to catch misses; high (0.6) hides noise.
- Manual boxes persist per-image in session cache (`manual_boxes_cache`) until saved or cleared — switching images won’t lose them.
- Headless test: `QT_QPA_PLATFORM=offscreen python -m py_compile yolo_dual_viewer.py`

## Troubleshooting
- `ultralytics not installed` → GUI still runs manual mode, shows warning. `pip install ultralytics`.
- `QPixmap null` → image path with `()`/`space` needs quoting; PIL fallback handles it.
- Slow inference → reduce `max_det` or use GPU (`torch` auto). `Save ALL` shows progress bar.
- PyQt5 vs PyQt6 → code imports `PyQt5`; if only PyQt6 present, `pip install PyQt5`.

## License / Author
Muse Spark — PyQt5 desktop tool. Modify `SOURCE_COLORS`, default sliders, or `SUPPORTED_IMG_EXTS` as needed.
# yolo_model_distilation_tool
