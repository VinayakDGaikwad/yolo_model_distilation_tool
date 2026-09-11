#!/bin/bash
# Launch with venv that has ultralytics + PyQt5 + torch
set -e
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
VENV_PY="/home/trendzlink/venv/envision/bin/python"
if [ ! -x "$VENV_PY" ]; then
  echo "VENV python not found at $VENV_PY, falling back to python3"
  VENV_PY="python3"
fi
echo "Using python: $VENV_PY"
echo "Versions:"
"$VENV_PY" -c "import sys; print(sys.version)"
"$VENV_PY" -c "import PyQt5.QtCore; print('PyQt5', PyQt5.QtCore.PYQT_VERSION_STR)" 2>&1 || echo "PyQt5 missing"
"$VENV_PY" -c "import ultralytics; print('ultralytics', ultralytics.__version__)" 2>&1 || echo "ultralytics missing (manual mode only)"
# optional args passthrough
exec "$VENV_PY" "$SCRIPT_DIR/yolo_dual_viewer.py" "$@"
