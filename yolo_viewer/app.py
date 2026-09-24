"""Entry point (parse args + run the YOLO Dual Model Viewer GUI)."""

import os
import sys

from PyQt5.QtWidgets import QApplication
from PyQt5.QtCore import Qt

from yolo_viewer.main_window import MainWindow

def parse_args():
    import argparse
    p = argparse.ArgumentParser(description="YOLO Dual Viewer")
    p.add_argument("--modelA", type=str, default="", help="Path to model A")
    p.add_argument("--modelB", type=str, default="", help="Path to model B")
    p.add_argument("--models-dir", type=str, default="", help="Folder containing multiple .pt/.onnx/.engine models")
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
    if args.models_dir and os.path.isdir(args.models_dir):
        win.model_dir_edit.setText(args.models_dir)
        win.load_models_dir()
    elif args.modelA:
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
