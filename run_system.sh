#!/bin/bash
set -e
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
echo "Using system python3"
python3 -c "import PyQt5.QtCore; print('PyQt5', PyQt5.QtCore.PYQT_VERSION_STR)" 2>&1 || echo "PyQt5 missing"
python3 -c "import ultralytics; print('ultralytics', ultralytics.__version__)" 2>&1 || echo "ultralytics missing - manual mode only, pip install ultralytics"
exec python3 "$SCRIPT_DIR/yolo_dual_viewer.py" "$@"
