#!/usr/bin/env python3
"""
YOLO Training Script — trains YOLOv8-L from scratch on YOLO txt labels
----------------------------------------------------------------------
By default this script builds a YOLOv8-L (yolov8l) architecture from its
YAML config with RANDOMLY INITIALIZED weights and trains it from scratch
on a folder containing images + co-located `<image>.txt` YOLO labels
(as exported by yolo_dual_viewer.py).

Pass a .pt checkpoint via --model instead if you want to fine-tune /
continue training an existing model (pretrained weights are then used).

Features:
- Auto-prepares a proper YOLO dataset structure:
      dataset_root/
          images/train/*.jpg  labels/train/*.txt
          images/val/*.jpg    labels/val/*.txt
          dataset.yaml
  from a flat folder where images and txts are co-located (handles spaces,
  parentheses in filenames).

- Auto-detects nc / class names from the .pt (fine-tune mode). When training
  from scratch (yaml model), nc / class names are scanned from the labels.

- Exposes ALL common Ultralytics training + augmentation knobs as CLI args
  so you can later pass any augmentation overrides, e.g.:

      python train.py --model yolov8l.yaml \
          --data-dir tz_batch_test_03_9_2025_020139/tz_batch_test_03_9_2025 \
          --epochs 50 --imgsz 640 --batch 8 \
          --hsv-h 0.015 --hsv-s 0.7 --hsv-v 0.4 \
          --degrees 10 --translate 0.1 --scale 0.5 --shear 2.0 \
          --flipud 0.0 --fliplr 0.5 --mosaic 1.0 --mixup 0.2 --copy-paste 0.3

- Pass-through any training param: lr, optimizer, patience, device, etc.
- Dry-run mode to only prepare dataset and show the effective config.
- Matplotlib preview: plot a few samples with GT boxes BEFORE training (see --plot).

Ultralytics version tested: 8.x (ultralytics>=8.0). Requires torch + ultralytics.

Examples:
    # 1. Train YOLOv8-L FROM SCRATCH (default) — auto-prepare dataset, 80/20 split:
    python train.py

    # 2. Same, explicitly + longer schedule (scratch training likes more epochs):
    python train.py --model yolov8l.yaml --epochs 300

    # 3. Plot 6 random train samples with GT boxes before training (sanity check):
    python train.py --plot --plot-n 6
    # only plot, no training:
    python train.py --plot-only --plot-n 9 --plot-split train --plot-save ./preview.png

    # 4. Custom data dir + explicit augment overrides:
    python train.py --model yolov8l.yaml \\
        --data-dir tz_batch_test_03_9_2025_020139/tz_batch_test_03_9_2025 \\
        --dataset-root ./datasets/my_run --val-split 0.2 --epochs 300 \\
        --hsv-h 0.02 --hsv-s 0.5 --degrees 15 --mosaic 0.8 --erasing 0.4 --plot

    # 5. Just prepare dataset, inspect, don't train:
    python train.py --dry-run --data-dir tz_batch_test_03_9_2025_020139/tz_batch_test_03_9_2025

    # 6. Re-use an already prepared dataset.yaml (skip auto-prepare):
    python train.py --data-yaml ./datasets/my_run/dataset.yaml --epochs 100

    # 7. Fine-tune an existing checkpoint instead of training from scratch:
    python train.py --model best_07-25_122104/best.pt --data-yaml ./datasets/my_run/dataset.yaml --epochs 50
"""
import argparse
import random
import shutil
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
SUPPORTED_IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}
DEFAULT_DATA_DIR = "/home/trendzlink/yolo_model_distilation_tool/only_boxes_with_lower_confidance_dataset/tz_batch_test_03_9_2025"
# yolov8l.yaml = YOLOv8-L architecture, random init -> trains FROM SCRATCH.
# (nc/names come from the dataset, not from the model yaml.)
DEFAULT_MODEL = "yolov8l.yaml"
DEFAULT_DATASET_ROOT = "/home/trendzlink/yolo_model_distilation_tool/only_boxes_with_lower_confidance_dataset/tz_batch_test_03_9_2025"


def is_scratch_model(spec: str) -> bool:
    """
    True if `spec` names an architecture config (random init / from scratch)
    rather than a .pt/.pth checkpoint with trained weights.

    Examples (True):  yolov8l.yaml, yolov8.yaml, yolo11l.yaml, yolov8l
    Examples (False): best.pt, runs/train/exp/weights/best.pt, /path/to/weights.pt
    """
    s = str(spec).strip()
    if s.lower().endswith((".yaml", ".yml")):
        return True
    p = Path(s)
    if p.exists():
        return False
    # bare model names like "yolov8l" (no dir, no suffix) resolve to an ultralytics cfg yaml
    return p.suffix == "" and "/" not in s and "\\" not in s

# Ultralytics defaults for augment (shown in --help); actual defaults come
# from ultralytics/cfg/default.yaml if you DON'T pass the flag.
AUG_DEFAULTS_HELP = {
    "hsv_h": 0.015, "hsv_s": 0.7, "hsv_v": 0.4,
    "degrees": 0.0, "translate": 0.1, "scale": 0.5, "shear": 0.0,
    "perspective": 0.0, "flipud": 0.0, "fliplr": 0.5, "bgr": 0.0,
    "mosaic": 1.0, "mixup": 0.0, "cutmix": 0.0, "copy_paste": 0.0,
    "copy_paste_mode": "flip", "auto_augment": "randaugment", "erasing": 0.4,
}

# ---------------------------------------------------------------------------
# Helpers — dataset
# ---------------------------------------------------------------------------

def find_images(data_dir: Path) -> List[Path]:
    """Find images in data_dir (non-recursive)."""
    imgs = []
    for ext in SUPPORTED_IMG_EXTS:
        imgs.extend(data_dir.glob(f"*{ext}"))
        imgs.extend(data_dir.glob(f"*{ext.upper()}"))
    imgs = sorted(set(imgs))
    return imgs


def scan_and_validate_labels(images: List[Path]) -> Tuple[List[Path], List[Path], int]:
    """
    Return (valid_images, skipped_images, max_class_id).
    """
    valid = []
    skipped = []
    max_cls = -1
    for img in images:
        txt = img.with_suffix(".txt")
        if not txt.exists():
            skipped.append(img)
            continue
        try:
            content = txt.read_text(encoding="utf-8", errors="ignore").strip()
            if content:
                for line in content.splitlines():
                    line = line.strip()
                    if not line:
                        continue
                    parts = line.split()
                    c = int(float(parts[0]))
                    max_cls = max(max_cls, c)
                    if len(parts) != 5:
                        print(f"[WARN] {txt.name}: expected 5 cols, got {len(parts)} — '{line}'")
                    else:
                        try:
                            _, x, y, w, h = map(float, parts)
                            if not all(0 <= v <= 1 for v in (x, y, w, h)):
                                print(f"[WARN] {txt.name}: values outside [0,1] — '{line}'")
                        except ValueError:
                            print(f"[WARN] {txt.name}: non-float values — '{line}'")
        except Exception as e:
            print(f"[WARN] Failed reading {txt}: {e}")
        valid.append(img)
    return valid, skipped, max_cls


def get_model_info(model_path: str) -> Tuple[int, List[str]]:
    """Load .pt via ultralytics to get nc and names. Fallback to nc=1 if fails."""
    try:
        from ultralytics import YOLO
        m = YOLO(model_path)
        names = m.names
        if isinstance(names, dict):
            max_id = max(names.keys()) if names else -1
            nc = max_id + 1
            lst = [str(names.get(i, f"class_{i}")) for i in range(nc)]
        elif isinstance(names, list):
            nc = len(names)
            lst = [str(n) for n in names]
        else:
            nc = 1
            lst = ["object"]
        print(f"[INFO] Model {model_path}: nc={nc} names={lst[:10]}{' ...' if len(lst)>10 else ''}")
        return nc, lst
    except Exception as e:
        print(f"[WARN] Could not read model names from {model_path}: {e}. Falling back to label scan.")
        return -1, []


def prepare_yolo_dataset(
    data_dir: Path,
    dataset_root: Path,
    val_split: float = 0.2,
    seed: int = 0,
    use_symlink: bool = False,
    clean: bool = False,
    copy_labels: bool = True,
) -> Path:
    """
    Prepare YOLO dataset structure from flat folder.
    Returns dataset_root.
    """
    data_dir = data_dir.resolve()
    dataset_root = dataset_root.resolve()

    if not data_dir.exists():
        raise FileNotFoundError(f"data_dir not found: {data_dir}")
    if not data_dir.is_dir():
        raise NotADirectoryError(f"data_dir is not a dir: {data_dir}")

    images = find_images(data_dir)
    print(f"[INFO] Found {len(images)} images in {data_dir}")
    if not images:
        raise RuntimeError(f"No images found in {data_dir} (exts={SUPPORTED_IMG_EXTS})")

    valid, skipped, max_cls = scan_and_validate_labels(images)
    if skipped:
        print(f"[WARN] {len(skipped)} images have NO paired .txt and will be SKIPPED:")
        for p in skipped[:10]:
            print(f"       - {p.name}")
        if len(skipped) > 10:
            print(f"       ... and {len(skipped)-10} more")
    print(f"[INFO] Using {len(valid)} paired images (max class id={max_cls if max_cls>=0 else 'none'})")

    if not valid:
        raise RuntimeError("No valid image+txt pairs found.")

    rng = random.Random(seed)
    valid_shuffled = valid.copy()
    rng.shuffle(valid_shuffled)

    n_val = max(1, int(round(len(valid_shuffled) * val_split))) if val_split > 0 else 0
    if val_split >= 1.0:
        raise ValueError("val_split must be <1.0")
    if val_split == 0:
        train_imgs = valid_shuffled
        val_imgs = []
    else:
        if len(valid_shuffled) - n_val < 1:
            n_val = len(valid_shuffled) - 1
        val_imgs = valid_shuffled[:n_val]
        train_imgs = valid_shuffled[n_val:]

    print(f"[INFO] Split (seed={seed}): train={len(train_imgs)} val={len(val_imgs)} (val_split={val_split})")

    if clean and dataset_root.exists():
        print(f"[INFO] Cleaning existing dataset_root: {dataset_root}")
        shutil.rmtree(dataset_root)

    for sub in ["images/train", "images/val", "labels/train", "labels/val"]:
        (dataset_root / sub).mkdir(parents=True, exist_ok=True)

    def _link_or_copy(src: Path, dst: Path):
        dst.parent.mkdir(parents=True, exist_ok=True)
        if dst.exists() or dst.is_symlink():
            dst.unlink()
        if use_symlink:
            try:
                dst.symlink_to(src.resolve())
                return
            except Exception as e:
                print(f"[WARN] symlink failed {src} -> {dst}: {e}, falling back to copy")
        shutil.copy2(src, dst)

    def _populate(img_list: List[Path], split: str):
        img_out_dir = dataset_root / f"images/{split}"
        lbl_out_dir = dataset_root / f"labels/{split}"
        for img in img_list:
            txt = img.with_suffix(".txt")
            _link_or_copy(img, img_out_dir / img.name)
            if txt.exists():
                _link_or_copy(txt, lbl_out_dir / txt.name)
            else:
                (lbl_out_dir / (img.stem + ".txt")).touch()
        print(f"[INFO] Populated {split}: {len(img_list)} pairs")

    _populate(train_imgs, "train")
    _populate(val_imgs, "val")
    return dataset_root


def build_dataset_yaml(
    dataset_root: Path,
    nc: int,
    names: List[str],
    yaml_path: Optional[Path] = None,
) -> Path:
    """Write dataset.yaml for YOLO."""
    dataset_root = dataset_root.resolve()
    if yaml_path is None:
        yaml_path = dataset_root / "dataset.yaml"
    if not names or len(names) != nc:
        if nc <= 0:
            nc = 1
        names = names or [f"class_{i}" for i in range(nc)]
        if len(names) < nc:
            names = names + [f"class_{i}" for i in range(len(names), nc)]
        elif len(names) > nc:
            names = names[:nc]
    yaml_content = f"""# Auto-generated by train.py
path: {dataset_root.as_posix()}
train: images/train
val: images/val
nc: {nc}
names: {names}
"""
    yaml_path.parent.mkdir(parents=True, exist_ok=True)
    yaml_path.write_text(yaml_content, encoding="utf-8")
    print(f"[INFO] Wrote dataset YAML: {yaml_path}")
    print("--- dataset.yaml ---")
    print(yaml_content.rstrip())
    print("---")
    return yaml_path


# ---------------------------------------------------------------------------
# Helpers — matplotlib preview BEFORE training  (NEW STEP)
# ---------------------------------------------------------------------------

def _resolve_image_dirs_from_yaml(yaml_path: Path) -> Tuple[Path, Path, List[str], int]:
    """
    Parse dataset.yaml to get train/val image dirs and class info.
    Returns (train_dir, val_dir, names_list, nc).
    Falls back to yaml.parent/images/train etc if yaml missing or unparsable.
    """
    yaml_path = yaml_path.resolve()
    base = yaml_path.parent
    train_dir = base / "images" / "train"
    val_dir = base / "images" / "val"
    names: List[str] = []
    nc = -1

    if not yaml_path.exists():
        return train_dir, val_dir, names, nc

    # Try PyYAML first
    try:
        import yaml as pyyaml  # type: ignore
        data = pyyaml.safe_load(yaml_path.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            # path may be absolute
            yaml_base = Path(data.get("path", base))
            if not yaml_base.is_absolute():
                yaml_base = (base / yaml_base).resolve()
            else:
                yaml_base = yaml_base.resolve()
            t = data.get("train", "images/train")
            v = data.get("val", "images/val")
            # train/val can be absolute or relative to path
            train_dir = Path(t) if Path(t).is_absolute() else (yaml_base / t)
            val_dir = Path(v) if Path(v).is_absolute() else (yaml_base / v)
            nc = int(data.get("nc", -1)) if data.get("nc") is not None else -1
            raw_names = data.get("names", [])
            if isinstance(raw_names, dict):
                # ultralytics allows dict {0: 'cls'}
                max_id = max(int(k) for k in raw_names.keys()) if raw_names else -1
                names = [str(raw_names.get(i, raw_names.get(str(i), f"class_{i}"))) for i in range(max_id + 1)]
            elif isinstance(raw_names, list):
                names = [str(n) for n in raw_names]
            elif isinstance(raw_names, str):
                names = [raw_names]
    except Exception:
        # Manual fallback — parse lines naively
        try:
            txt = yaml_path.read_text(encoding="utf-8")
            for line in txt.splitlines():
                line=line.strip()
                if line.startswith("path:"):
                    base = Path(line.split(":",1)[1].strip())
                    if not base.is_absolute():
                        base = (yaml_path.parent / base).resolve()
                elif line.startswith("train:"):
                    t = line.split(":",1)[1].strip()
                    train_dir = Path(t) if Path(t).is_absolute() else (base / t)
                elif line.startswith("val:"):
                    v = line.split(":",1)[1].strip()
                    val_dir = Path(v) if Path(v).is_absolute() else (base / v)
                elif line.startswith("nc:"):
                    try: nc = int(line.split(":",1)[1].strip())
                    except: pass
                elif line.startswith("names:"):
                    # e.g. names: ['0'] or names: ["cat","dog"]
                    val = line.split(":",1)[1].strip()
                    # crude eval
                    try:
                        import ast
                        parsed = ast.literal_eval(val)
                        if isinstance(parsed, list):
                            names = [str(x) for x in parsed]
                    except: pass
        except Exception:
            pass
    return train_dir.resolve() if train_dir.exists() else base / "images" / "train", \
           val_dir.resolve() if val_dir.exists() else base / "images" / "val", \
           names, nc


def _collect_images_for_preview(
    yaml_path: Path,
    dataset_root_hint: Optional[Path],
    split: str
) -> List[Path]:
    """Return list of image Paths to sample from, given yaml + split."""
    yaml_path = yaml_path.resolve() if yaml_path and yaml_path.exists() else None

    candidates: List[Path] = []

    if yaml_path is not None:
        train_dir, val_dir, _, _ = _resolve_image_dirs_from_yaml(yaml_path)
        if split == "train":
            dirs = [train_dir]
        elif split == "val":
            dirs = [val_dir]
        else:  # all
            dirs = [train_dir, val_dir]
        for d in dirs:
            if d.exists():
                for ext in SUPPORTED_IMG_EXTS:
                    candidates.extend(d.glob(f"*{ext}"))
                    candidates.extend(d.glob(f"*{ext.upper()}"))
    # Fallback to dataset_root_hint if still empty (e.g. yaml not parsable)
    if not candidates and dataset_root_hint is not None and dataset_root_hint.exists():
        for s in (["train","val"] if split=="all" else [split]):
            d = dataset_root_hint / "images" / s
            if d.exists():
                for ext in SUPPORTED_IMG_EXTS:
                    candidates.extend(d.glob(f"*{ext}"))
                    candidates.extend(d.glob(f"*{ext.upper()}"))
    # De-dup
    candidates = sorted(set(p for p in candidates if p.is_file()))
    return candidates


def _build_label_index(label_dirs: List[Path]) -> Dict[str, Path]:
    """Map stem -> txt path by scanning label dirs (handles spaces/parentheses)."""
    idx: Dict[str, Path] = {}
    for d in label_dirs:
        if not d.exists():
            continue
        for txt in d.rglob("*.txt"):
            # keep first occurrence; if duplicates, prefer shallowest
            if txt.stem not in idx:
                idx[txt.stem] = txt
    return idx


def _find_label_for_image(
    img_path: Path,
    label_index: Dict[str, Path],
) -> Optional[Path]:
    """Find txt for image: try co-located, then index by stem."""
    # 1. co-located (flat dataset case)
    co = img_path.with_suffix(".txt")
    if co.exists():
        return co
    # 2. index by stem (prepared dataset: images/train/... -> labels/train/...)
    #    stems can duplicate across splits but train/val split separation keeps it ok
    #    For uniqueness, the flat filename including spaces is the key
    if img_path.stem in label_index:
        return label_index[img_path.stem]
    # 3. try labels/train vs labels/val sibling
    #    e.g. /.../images/train/a.jpg -> /.../labels/train/a.txt
    try:
        parts = img_path.parts
        if "images" in parts:
            idx = parts.index("images")
            # construct labels path: .../labels/<split>/a.txt
            # need dataset root = Path(*parts[:idx])
            root = Path(*parts[:idx])
            split = parts[idx+1] if len(parts) > idx+1 else ""
            cand = root / "labels" / split / (img_path.stem + ".txt")
            if cand.exists():
                return cand
            cand2 = root / "labels" / (img_path.stem + ".txt")
            if cand2.exists():
                return cand2
    except Exception:
        pass
    return None


def plot_yolo_samples_matplotlib(
    yaml_path: Path,
    dataset_root_hint: Optional[Path] = None,
    num: int = 6,
    split: str = "train",
    save_path: Optional[Path] = None,
    show: bool = False,
    seed: int = 0,
    max_boxes: Optional[int] = None,
    dpi: int = 150,
) -> Optional[Path]:
    """
    STEP: Plot a few YOLO images with GT boxes using matplotlib BEFORE training.

    - Uses matplotlib (Agg backend for headless). No window required.
    - Draws YOLO normalized xywh boxes as rectangles (no label spam for >300 boxes
      dense scenes — boxes are thin, semi-transparent).
    - Saves preview PNG and optionally shows interactively.

    Returns saved path or None on failure.

    This is the "implot matplotlib" preview step requested: call with --plot.
    """
    # --- lazy imports, handle missing deps gracefully ---
    try:
        import matplotlib
        # Use Agg unless user wants interactive show
        if not show:
            matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.patches as patches
    except ImportError as e:
        print(f"[PLOT][WARN] matplotlib not installed, skipping preview: {e}")
        print("             pip install matplotlib")
        return None

    # Try image loaders: prefer cv2 (faster), fallback to PIL
    has_cv2 = False
    has_pil = False
    try:
        import cv2  # type: ignore
        has_cv2 = True
    except ImportError:
        pass
    try:
        from PIL import Image  # type: ignore
        has_pil = True
    except ImportError:
        pass
    if not has_cv2 and not has_pil:
        print("[PLOT][ERROR] Need cv2 or Pillow to load images. pip install opencv-python pillow")
        return None

    yaml_path = Path(yaml_path) if yaml_path else None
    if yaml_path is not None and not yaml_path.exists():
        print(f"[PLOT][WARN] yaml not found: {yaml_path}, trying dataset_root_hint")
        yaml_path = None

    # Resolve image candidates
    candidates = _collect_images_for_preview(yaml_path, dataset_root_hint, split)
    if not candidates:
        print(f"[PLOT][WARN] No images found for split='{split}' (yaml={yaml_path}, hint={dataset_root_hint})")
        return None

    print(f"[PLOT] Found {len(candidates)} images for split='{split}' — sampling {min(num,len(candidates))} (seed={seed})")

    rng = random.Random(seed)
    sampled = candidates.copy()
    rng.shuffle(sampled)
    sampled = sampled[:num]
    if not sampled:
        print("[PLOT][WARN] No samples selected")
        return None

    # Build label index (scan labels dirs once)
    label_dirs: List[Path] = []
    if yaml_path is not None:
        tr_dir, va_dir, names_hint, nc_hint = _resolve_image_dirs_from_yaml(yaml_path)
        # infer label dirs: replace images -> labels
        for img_dir in [tr_dir, va_dir]:
            # img_dir is like .../images/train ; label_dir is .../labels/train
            try:
                # robust: replace first "images" with "labels"
                parts = img_dir.parts
                if "images" in parts:
                    idx = parts.index("images")
                    lbl = Path(*parts[:idx]) / "labels" / Path(*parts[idx+1]) if len(parts) > idx+1 else Path(*parts[:idx]) / "labels"
                    label_dirs.append(lbl)
            except Exception:
                pass
        # also add generic <yaml.parent>/labels/train|val
        base = yaml_path.parent
        label_dirs.extend([base / "labels" / "train", base / "labels" / "val", base / "labels"])
    if dataset_root_hint is not None:
        label_dirs.extend([dataset_root_hint / "labels" / "train", dataset_root_hint / "labels" / "val", dataset_root_hint / "labels"])
    # flat data-dir case: same dir as images (txt co-located)
    label_dirs = list(dict.fromkeys([p.resolve() for p in label_dirs if p is not None]))
    label_index = _build_label_index(label_dirs)

    # Resolve class names for legend
    class_names: List[str] = []
    nc = 0
    if yaml_path is not None:
        _, _, names_hint, nc_hint = _resolve_image_dirs_from_yaml(yaml_path)
        if names_hint:
            class_names = names_hint
            nc = len(class_names)
        elif nc_hint and nc_hint > 0:
            class_names = [f"class_{i}" for i in range(nc_hint)]
            nc = nc_hint
    if not class_names:
        class_names = ["object"]
        nc = 1

    # Color map per class — tab10 / husl; single class = vivid red/lime
    cmap = plt.cm.get_cmap("tab10", max(nc, 10))
    def _color_for_cls(c: int):
        if nc == 1:
            return "#00ff00"  # lime for single class (visible on many backgrounds)
        return cmap(c % 10)

    # Prepare figure grid: auto cols/rows
    n = len(sampled)
    cols = min(3, n) if n <= 6 else 3
    if n == 1:
        cols, rows = 1, 1
    elif n == 2:
        cols, rows = 2, 1
    elif n <= 4:
        cols, rows = 2, 2 if n > 2 else 1
    elif n <= 6:
        cols, rows = 3, 2
    elif n <= 9:
        cols, rows = 3, 3
    else:
        cols = 4
        rows = (n + cols - 1)//cols

    figsize_w = cols * 5.0
    figsize_h = rows * 5.0
    fig, axes = plt.subplots(rows, cols, figsize=(figsize_w, figsize_h), dpi=dpi, squeeze=False)
    axes_flat = axes.flatten()

    # Default save path
    if save_path is None:
        if yaml_path is not None:
            save_path = yaml_path.parent / f"preview_{split}_{n}samples.png"
        elif dataset_root_hint is not None:
            save_path = dataset_root_hint / f"preview_{split}_{n}samples.png"
        else:
            save_path = Path(f"./preview_{split}_{n}samples.png")
    else:
        save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    ok_count = 0
    for idx, ax in enumerate(axes_flat):
        if idx >= n:
            ax.axis("off")
            continue
        img_path = sampled[idx]
        # --- load image ---
        img_rgb = None
        W = H = 0
        try:
            if has_cv2:
                import cv2
                bgr = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
                if bgr is not None:
                    H, W = bgr.shape[:2]
                    img_rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            if img_rgb is None and has_pil:
                from PIL import Image
                pil = Image.open(img_path).convert("RGB")
                W, H = pil.size  # PIL is (W,H)
                # Convert to numpy for imshow
                import numpy as np
                img_rgb = np.array(pil)
            if img_rgb is None:
                raise RuntimeError("Failed to load image with both cv2 and PIL")
        except Exception as e:
            print(f"[PLOT][WARN] Failed to load {img_path.name}: {e}")
            ax.text(0.5, 0.5, f"load fail\n{img_path.name}", ha="center", va="center", transform=ax.transAxes, color="red")
            ax.axis("off")
            continue

        ax.imshow(img_rgb)
        ax.axis("off")

        # --- find + parse labels ---
        txt_path = _find_label_for_image(img_path, label_index)
        boxes = []  # list of (cls, x1,y1,x2,y2)
        if txt_path is None or not txt_path.exists():
            # try co-located as last resort
            co = img_path.with_suffix(".txt")
            if co.exists():
                txt_path = co
            else:
                print(f"[PLOT][WARN] No label found for {img_path.name} (looked in {label_dirs})")
                ax.set_title(f"{img_path.name}\n(no txt, 0 boxes)", fontsize=8, color="#c00")
                continue
        try:
            lines = txt_path.read_text(encoding="utf-8", errors="ignore").strip().splitlines()
            for line in lines:
                line=line.strip()
                if not line:
                    continue
                parts = line.split()
                if len(parts) < 5:
                    continue
                try:
                    cls = int(float(parts[0]))
                    cx, cy, bw, bh = map(float, parts[1:5])
                except ValueError:
                    continue
                # clamp norms (handle slightly out-of-range)
                cx = min(max(cx, 0), 1); cy = min(max(cy, 0), 1)
                bw = min(max(bw, 0), 1); bh = min(max(bh, 0), 1)
                # xywh normalized -> xyxy absolute
                x1 = (cx - bw/2) * W
                y1 = (cy - bh/2) * H
                x2 = (cx + bw/2) * W
                y2 = (cy + bh/2) * H
                # clamp to image
                x1 = max(0, min(x1, W)); y1 = max(0, min(y1, H))
                x2 = max(0, min(x2, W)); y2 = max(0, min(y2, H))
                if x2 <= x1 or y2 <= y1:
                    continue
                boxes.append((cls, x1, y1, x2, y2))
        except Exception as e:
            print(f"[PLOT][WARN] Failed parsing {txt_path}: {e}")

        # Optionally limit for dense (>300) to avoid matplotlib slowdown
        total_boxes = len(boxes)
        if max_boxes is not None and len(boxes) > max_boxes:
            # keep random subset for visibility but note truncation in title
            rng2 = random.Random(seed + idx)
            rng2.shuffle(boxes)
            boxes = boxes[:max_boxes]

        # --- draw boxes ---
        # For dense scenes (>100), use very thin lines and lower alpha for visibility
        lw = 1.2 if total_boxes < 100 else (0.8 if total_boxes < 300 else 0.6)
        alpha = 0.95 if total_boxes < 50 else (0.7 if total_boxes < 200 else 0.5)
        for cls, x1, y1, x2, y2 in boxes:
            w = x2 - x1; h = y2 - y1
            col = _color_for_cls(cls)
            rect = patches.Rectangle((x1, y1), w, h, linewidth=lw, edgecolor=col, facecolor="none", alpha=alpha)
            ax.add_patch(rect)
            # Only add text if sparse (<50) to avoid clutter — mirrors yolo_dual_viewer hide labels
            if total_boxes < 40:
                label = class_names[cls] if 0 <= cls < len(class_names) else str(cls)
                ax.text(x1, max(0, y1-2), label, fontsize=5, color="white",
                        bbox=dict(boxstyle="round,pad=0.2", fc=col, ec="none", alpha=0.85),
                        va="bottom", ha="left")

        title = f"{img_path.name}\n{total_boxes} box{'es' if total_boxes!=1 else ''}"
        if max_boxes is not None and total_boxes > max_boxes:
            title += f" (showing {len(boxes)})"
        title += f"  {W}x{H}"
        ax.set_title(title, fontsize=8, pad=4)

        ok_count += 1

    # Global title
    fig.suptitle(f"YOLO Preview — {split} split — {ok_count}/{n} images — {yaml_path.name if yaml_path else dataset_root_hint}",
                 fontsize=12, fontweight="bold", y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    # Adjust spacing for dense grids
    plt.subplots_adjust(wspace=0.08, hspace=0.25)

    try:
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
        print(f"[PLOT] Saved preview to: {save_path.resolve()}  ({save_path.stat().st_size/1024:.1f} KB)")
    except Exception as e:
        print(f"[PLOT][ERROR] Failed to save preview to {save_path}: {e}")
        save_path = None

    if show:
        try:
            plt.show()
        except Exception as e:
            print(f"[PLOT][WARN] plt.show() failed (headless?): {e}")

    plt.close(fig)
    return save_path


# ---------------------------------------------------------------------------
# Argparse
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="YOLO training: train YOLOv8-L from scratch (default) or fine-tune a .pt on YOLO txt labels (with full augmentation control + matplotlib preview)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        epilog=(
            "Augmentation overrides examples:\n"
            "  --hsv-h 0.02 --hsv-s 0.6 --degrees 10 --translate 0.2 --scale 0.9\n"
            "  --shear 2 --perspective 0.001 --flipud 0.5 --fliplr 0.5\n"
            "  --mosaic 0.8 --mixup 0.2 --copy-paste 0.3 --erasing 0.4 --auto-augment randaugment\n"
            "\n"
            "Preview example:\n"
            "  python train.py --plot --plot-n 6 --plot-split train   # plot 6 train images BEFORE training\n"
            "  python train.py --plot-only --plot-n 9 --plot-save ./my_preview.png\n"
            "\n"
            "All augmentation args are optional; if omitted Ultralytics defaults are used.\n"
            "See `ultralytics/cfg/default.yaml` for full defaults."
        )
    )

    # --- Data / Model ---
    g_data = p.add_argument_group("Data / Model")
    g_data.add_argument("--model", type=str, default=DEFAULT_MODEL,
                        help="Model to train. Default 'yolov8l.yaml' = YOLOv8-L trained FROM SCRATCH (random weights). "
                             "Pass a .pt (e.g. best.pt / yolov8l.pt) to fine-tune instead.")
    g_data.add_argument("--data-dir", type=str, default=DEFAULT_DATA_DIR,
                        help="Flat folder containing images + paired .txt YOLO labels (spaces in names OK). Ignored if --data-yaml is given and --no-prepare.")
    g_data.add_argument("--data-yaml", type=str, default=None,
                        help="Path to existing dataset.yaml. If given, dataset preparation is skipped unless --force-prepare.")
    g_data.add_argument("--dataset-root", type=str, default=DEFAULT_DATASET_ROOT,
                        help="Output root for prepared dataset (images/train, labels/train, dataset.yaml).")
    g_data.add_argument("--val-split", type=float, default=0.2,
                        help="Fraction for validation split (0.0 = no val, 0.2 = 20%% val).")
    g_data.add_argument("--seed", type=int, default=0, help="Random seed for shuffle/split.")
    g_data.add_argument("--single-cls", action="store_true",
                        help="Treat all classes as one (single_cls=True). Useful if your labels are all class 0.")
    g_data.add_argument("--force-prepare", action="store_true",
                        help="Re-prepare dataset even if --data-yaml exists.")
    g_data.add_argument("--clean", action="store_true",
                        help="Remove existing dataset-root before preparing (fresh copy).")
    g_data.add_argument("--symlink", action="store_true",
                        help="Use symlinks instead of copying images/labels (faster, but not portable).")
    g_data.add_argument("--no-prepare", action="store_true",
                        help="Skip dataset preparation step entirely (assumes dataset already exists at --dataset-root or --data-yaml).")

    # --- Training core ---
    g_train = p.add_argument_group("Training")
    g_train.add_argument("--epochs", type=int, default=100, help="Number of epochs.")
    g_train.add_argument("--imgsz", type=int, default=640, help="Input image size (imgsz).")
    g_train.add_argument("--batch", type=int, default=8, help="Batch size (use -1 for auto-batch).")
    g_train.add_argument("--workers", type=int, default=8, help="Dataloader workers.")
    g_train.add_argument("--device", type=str, default=None,
                         help="Device e.g. '0', '0,1', 'cpu', 'mps'. Auto if not set.")
    g_train.add_argument("--project", type=str, default="runs/train",
                         help="Project dir for saving runs (project).")
    g_train.add_argument("--name", type=str, default=None,
                         help="Run name (name). Auto (increment) if not set.")
    g_train.add_argument("--exist-ok", action="store_true", help="Allow overwriting existing project/name.")
    g_train.add_argument("--patience", type=int, default=50, help="EarlyStopping patience (epochs).")
    g_train.add_argument("--save-period", type=int, default=-1, help="Save checkpoint every x epochs (disabled if -1).")
    g_train.add_argument("--cache", type=str, default="False",
                         help="Cache images: False / True / ram / disk")
    g_train.add_argument("--rect", action="store_true", help="Rectangular training (less padding, faster).")
    g_train.add_argument("--resume", type=str, default=None,
                         help="Resume training from checkpoint (path to last.pt). Or set True to auto-resume.")
    g_train.add_argument("--freeze", type=str, default=None,
                         help="Freeze layers: e.g. '10' or None. See ultralytics freeze arg.")
    g_train.add_argument("--amp", dest="amp", action="store_true", default=True,
                         help="Enable Automatic Mixed Precision (AMP).")
    g_train.add_argument("--no-amp", dest="amp", action="store_false",
                         help="Disable AMP (use full precision).")
    g_train.add_argument("--optimizer", type=str, default="auto",
                         help="Optimizer: auto, SGD, Adam, AdamW, NAdam, RAdam, RMSProp, etc.")
    g_train.add_argument("--lr0", type=float, default=None, help="Initial learning rate (lr0, default 0.01).")
    g_train.add_argument("--lrf", type=float, default=None, help="Final learning rate factor (lrf, default 0.01).")
    g_train.add_argument("--momentum", type=float, default=None, help="SGD momentum.")
    g_train.add_argument("--weight-decay", type=float, default=None, help="Weight decay (weight_decay).")
    g_train.add_argument("--warmup-epochs", type=float, default=None, help="Warmup epochs.")
    g_train.add_argument("--close-mosaic", type=int, default=None,
                         help="Disable mosaic augmentation for last N epochs (close_mosaic, default 10). Set 0 to keep mosaic till end.")
    g_train.add_argument("--multi-scale", type=float, default=None,
                         help="Multi-scale training factor (0.0=off, 0.5 typical).")
    g_train.add_argument("--cos-lr", action="store_true", help="Use cosine LR scheduler (cos_lr=True).")
    g_train.add_argument("--dropout", type=float, default=None, help="Dropout (detect dropout).")
    g_train.add_argument("--val", dest="do_val", action="store_true", default=True, help="Validate during training.")
    g_train.add_argument("--no-val", dest="do_val", action="store_false", help="Disable validation during training.")
    g_train.add_argument("--plots", action="store_true", default=True, help="Save plots.")
    g_train.add_argument("--no-plots", dest="plots", action="store_false", help="Disable plots.")
    g_train.add_argument("--fraction", type=float, default=None, help="Fraction of dataset to use (0.0-1.0).")
    g_train.add_argument("--profile", action="store_true", help="Profile training (profile=True).")
    g_train.add_argument("--deterministic", dest="deterministic", action="store_true", default=True, help="Deterministic training.")
    g_train.add_argument("--nondeterministic", dest="deterministic", action="store_false", help="Disable deterministic (faster, less reproducible).")

    # --- Augmentation (you will give values later — all optional) ---
    g_aug = p.add_argument_group(
        "Augmentation (all OPTIONAL — omitted = Ultralytics default; set explicitly to override)")
    g_aug.add_argument("--hsv-h", type=float, default=None, help=f"HSV Hue augmentation (default {AUG_DEFAULTS_HELP['hsv_h']}).")
    g_aug.add_argument("--hsv-s", type=float, default=None, help=f"HSV Saturation (default {AUG_DEFAULTS_HELP['hsv_s']}).")
    g_aug.add_argument("--hsv-v", type=float, default=None, help=f"HSV Value/Brightness (default {AUG_DEFAULTS_HELP['hsv_v']}).")
    g_aug.add_argument("--degrees", type=float, default=None, help=f"Rotation degrees (-degrees..+degrees) (default {AUG_DEFAULTS_HELP['degrees']}).")
    g_aug.add_argument("--translate", type=float, default=None, help=f"Translation fraction (default {AUG_DEFAULTS_HELP['translate']}).")
    g_aug.add_argument("--scale", type=float, default=None, help=f"Scale gain (default {AUG_DEFAULTS_HELP['scale']}).")
    g_aug.add_argument("--shear", type=float, default=None, help=f"Shear degrees (default {AUG_DEFAULTS_HELP['shear']}).")
    g_aug.add_argument("--perspective", type=float, default=None, help=f"Perspective (default {AUG_DEFAULTS_HELP['perspective']}).")
    g_aug.add_argument("--flipud", type=float, default=None, help=f"Vertical flip prob (default {AUG_DEFAULTS_HELP['flipud']}).")
    g_aug.add_argument("--fliplr", type=float, default=None, help=f"Horizontal flip prob (default {AUG_DEFAULTS_HELP['fliplr']}).")
    g_aug.add_argument("--bgr", type=float, default=None, help=f"BGR channel shuffle prob (default {AUG_DEFAULTS_HELP['bgr']}).")
    g_aug.add_argument("--mosaic", type=float, default=None, help=f"Mosaic prob (default {AUG_DEFAULTS_HELP['mosaic']}). 0=off.")
    g_aug.add_argument("--mixup", type=float, default=None, help=f"Mixup prob (default {AUG_DEFAULTS_HELP['mixup']}).")
    g_aug.add_argument("--cutmix", type=float, default=None, help=f"CutMix prob (default {AUG_DEFAULTS_HELP['cutmix']}).")
    g_aug.add_argument("--copy-paste", type=float, dest="copy_paste", default=None, help=f"Copy-paste (seg) prob (default {AUG_DEFAULTS_HELP['copy_paste']}).")
    g_aug.add_argument("--copy-paste-mode", type=str, dest="copy_paste_mode", default=None,
                       choices=["flip", "mixup"],
                       help=f"Copy-paste mode (default {AUG_DEFAULTS_HELP['copy_paste_mode']}).")
    g_aug.add_argument("--auto-augment", type=str, dest="auto_augment", default=None,
                       help=f"Auto augment policy (randaugment, autoaugment, augmix or '' to disable) (default {AUG_DEFAULTS_HELP['auto_augment']}).")
    g_aug.add_argument("--erasing", type=float, default=None, help=f"Random erasing prob (default {AUG_DEFAULTS_HELP['erasing']}). 0=off.")
    g_aug.add_argument("--aug-preset", type=str, default=None, choices=["light", "medium", "heavy", "none"],
                       help="Quick preset that sets many augment values at once (can still be overridden by explicit flags AFTER). "
                            "none=disable most augment (mosaic 0 etc) for debugging.")

    # --- Preview / Plot (NEW STEP — matplotlib before training) ---
    g_plot = p.add_argument_group("Preview — Plot samples BEFORE training (matplotlib)")
    g_plot.add_argument("--plot", action="store_true",
                        help="Plot a few training images with GT YOLO boxes using matplotlib BEFORE training starts (sanity-check). Saves PNG.")
    g_plot.add_argument("--plot-only", action="store_true",
                        help="Only plot samples and exit (no training). Implies --plot.")
    g_plot.add_argument("--plot-n", type=int, default=6,
                        help="Number of sample images to plot in grid (e.g. 4=2x2, 6=2x3, 9=3x3).")
    g_plot.add_argument("--plot-split", type=str, default="train", choices=["train", "val", "all"],
                        help="Which split to sample from for preview.")
    g_plot.add_argument("--plot-save", type=str, default=None,
                        help="Path to save preview PNG. Default: <dataset-root>/preview_<split>_<n>samples.png")
    g_plot.add_argument("--plot-show", action="store_true",
                        help="Also show plot interactively with plt.show() (needs display; headless will just save).")
    g_plot.add_argument("--plot-seed", type=int, default=0,
                        help="Random seed for sampling images to plot.")
    g_plot.add_argument("--plot-max-boxes", type=int, default=None,
                        help="If set, limit boxes drawn per image (for dense >300 boxes scenes) to avoid clutter/slowdown. E.g. 100 or 200.")
    g_plot.add_argument("--plot-dpi", type=int, default=150,
                        help="DPI for saved preview PNG (higher = sharper).")

    # --- Misc ---
    g_misc = p.add_argument_group("Misc")
    g_misc.add_argument("--dry-run", action="store_true",
                        help="Only prepare dataset + print the YOLO train() kwargs, do NOT start training.")
    g_misc.add_argument("--check-only", action="store_true",
                        help="Validate dataset/model and exit (no training, no preparation).")
    g_misc.add_argument("--verbose", action="store_true", default=True, help="Verbose ultralytics logging.")
    g_misc.add_argument("--no-verbose", dest="verbose", action="store_false", help="Quiet logging.")

    return p


def resolve_augmentation_args(args, parser) -> Dict:
    """
    Build dict of augmentation kwargs to pass to YOLO.train().
    Only include keys where the user explicitly passed a value (to preserve
    ultralytics defaults if not mentioned). For preset handling, we inject preset
    values first, then let explicit flags override.
    """
    explicit = set()
    for token in sys.argv[1:]:
        if token.startswith("--"):
            k = token.split("=")[0].lstrip("-").replace("-", "_")
            explicit.add(k)

    aug_keys = ["hsv_h", "hsv_s", "hsv_v", "degrees", "translate", "scale", "shear",
                "perspective", "flipud", "fliplr", "bgr", "mosaic", "mixup", "cutmix",
                "copy_paste", "copy_paste_mode", "auto_augment", "erasing"]

    aug = {}

    if args.aug_preset is not None:
        if args.aug_preset == "none":
            preset = dict(degrees=0, translate=0, scale=0, shear=0, perspective=0,
                          flipud=0, fliplr=0, hsv_h=0, hsv_s=0, hsv_v=0,
                          mosaic=0, mixup=0, copy_paste=0, erasing=0, auto_augment="")
        elif args.aug_preset == "light":
            preset = dict(hsv_h=0.015, hsv_s=0.3, hsv_v=0.2,
                          degrees=5, translate=0.05, scale=0.3, shear=1,
                          mosaic=0.5, mixup=0.0, copy_paste=0.0, erasing=0.2, fliplr=0.5, flipud=0.0)
        elif args.aug_preset == "medium":
            preset = dict(hsv_h=0.015, hsv_s=0.5, hsv_v=0.3,
                          degrees=10, translate=0.1, scale=0.5, shear=1.5,
                          mosaic=0.8, mixup=0.1, copy_paste=0.1, erasing=0.3, fliplr=0.5)
        elif args.aug_preset == "heavy":
            preset = dict(hsv_h=0.03, hsv_s=0.7, hsv_v=0.5,
                          degrees=15, translate=0.2, scale=0.7, shear=3,
                          mosaic=1.0, mixup=0.3, copy_paste=0.3, erasing=0.5, fliplr=0.5, flipud=0.1)
        else:
            preset = {}
        for k, v in preset.items():
            if k not in explicit:
                aug[k] = v
        print(f"[INFO] Applied aug-preset='{args.aug_preset}': {preset}")

    for k in aug_keys:
        v = getattr(args, k, None)
        if v is not None:
            aug[k] = v
    return aug


def normalize_cache_arg(val: str):
    """Ultralytics expects bool or str 'ram'/'disk'. CLI gives string."""
    if isinstance(val, bool):
        return val
    low = str(val).lower()
    if low in ("false", "0", "no", "none", ""):
        return False
    if low in ("true", "1", "yes"):
        return True
    if low in ("ram", "disk"):
        return low
    return val


def main():
    parser = build_parser()
    args = parser.parse_args()

    print("=" * 78)
    print(" YOLO Train — yolov8l from scratch (pass a .pt to fine-tune instead)")
    print("=" * 78)
    print(f"  Model      : {args.model}")
    print(f"  Mode       : {'FROM SCRATCH (random init, architecture from yaml)' if is_scratch_model(args.model) else 'fine-tune / continue from checkpoint'}")
    print(f"  Data dir   : {args.data_dir}")
    print(f"  DatasetRoot: {args.dataset_root}")
    print(f"  Data YAML  : {args.data_yaml if args.data_yaml else '(auto-generated)'}")
    print()

    # ---- checks ----
    model_path = Path(args.model)
    from_scratch = is_scratch_model(args.model)
    if not from_scratch and not model_path.exists():
        print(f"[ERROR] Model not found: {model_path} (cwd={Path.cwd()})")
        alt = Path(DEFAULT_MODEL)
        if alt.exists():
            print(f"        Hint: default model exists at {alt.resolve()}")
        sys.exit(2)
    if from_scratch:
        print(f"[INFO] Training FROM SCRATCH: {args.model} (random init, no pretrained weights)")

    data_yaml_path: Optional[Path] = None
    if args.data_yaml:
        p = Path(args.data_yaml)
        if p.exists() and not args.force_prepare and not args.no_prepare:
            print(f"[INFO] Using provided --data-yaml: {p.resolve()} (skipping auto-prepare)")
            data_yaml_path = p.resolve()
        else:
            if not p.exists():
                print(f"[WARN] --data-yaml {p} does not exist, will auto-prepare and generate one.")
            elif args.force_prepare:
                print(f"[INFO] --force-prepare set, will regenerate dataset despite --data-yaml")

    # ---- dataset preparation ----
    if data_yaml_path is None and not args.no_prepare:
        try:
            data_dir = Path(args.data_dir)
            dataset_root = Path(args.dataset_root)
            if from_scratch:
                # A yaml-built model has no trained class names (defaults to nc=80),
                # so derive nc/names from the labels instead.
                nc, names = -1, []
            else:
                nc, names = get_model_info(str(model_path))
            if nc < 0:
                imgs = find_images(data_dir)
                _, _, max_c = scan_and_validate_labels(imgs)
                nc = (max_c + 1) if max_c >= 0 else 1
                names = [f"class_{i}" for i in range(nc)]
                print(f"[INFO] Inferred from labels: nc={nc}")

            prepared_root = prepare_yolo_dataset(
                data_dir=data_dir,
                dataset_root=dataset_root,
                val_split=args.val_split,
                seed=args.seed,
                use_symlink=args.symlink,
                clean=args.clean,
            )
            max_cls_prepared = -1
            for lbl_dir in [prepared_root / "labels/train", prepared_root / "labels/val"]:
                for txt in lbl_dir.glob("*.txt"):
                    try:
                        for line in txt.read_text(errors="ignore").splitlines():
                            line=line.strip()
                            if line:
                                max_cls_prepared = max(max_cls_prepared, int(float(line.split()[0])))
                    except Exception:
                        pass
            if max_cls_prepared >= 0 and max_cls_prepared >= nc:
                print(f"[WARN] Label max class id {max_cls_prepared} >= model nc {nc}. "
                      f"Forcing yaml nc to {max_cls_prepared+1}.")
                nc = max_cls_prepared + 1
                if len(names) < nc:
                    names = names + [f"class_{i}" for i in range(len(names), nc)]

            data_yaml_path = build_dataset_yaml(prepared_root, nc=nc, names=names)
            ntrain = len(list((prepared_root / "images/train").glob("*.*")))
            nval = len(list((prepared_root / "images/val").glob("*.*")))
            print(f"[INFO] Prepared dataset: train={ntrain} val={nval} -> {prepared_root.resolve()}")
        except Exception as e:
            print(f"[ERROR] Dataset preparation failed: {e}", file=sys.stderr)
            import traceback; traceback.print_exc()
            sys.exit(3)
    elif args.no_prepare:
        if args.data_yaml and Path(args.data_yaml).exists():
            data_yaml_path = Path(args.data_yaml).resolve()
        else:
            cand = Path(args.dataset_root) / "dataset.yaml"
            if cand.exists():
                data_yaml_path = cand.resolve()
                print(f"[INFO] --no-prepare: using existing {data_yaml_path}")
            else:
                print(f"[ERROR] --no-prepare set but no dataset.yaml found at {cand.resolve()} nor --data-yaml. "
                      f"Provide --data-yaml or remove --no-prepare.", file=sys.stderr)
                sys.exit(2)
    else:
        pass

    if args.check_only:
        print("[INFO] --check-only: dataset and model validated, exiting.")
        print(f"       dataset yaml = {data_yaml_path}")
        print(f"       model        = {args.model}{' (from scratch)' if from_scratch else ''}")
        sys.exit(0)

    if data_yaml_path is None or not data_yaml_path.exists():
        print(f"[ERROR] No dataset YAML available. Expected {data_yaml_path}", file=sys.stderr)
        sys.exit(3)

    # -------------------------------------------------------------------
    # NEW STEP: Plot a few images with GT boxes BEFORE training (matplotlib)
    # -------------------------------------------------------------------
    # This is the requested "implot matplotlib" sanity check — verifies
    # that YOLO txt -> image alignment is correct before spending GPU hours.
    do_plot = args.plot or args.plot_only
    if do_plot:
        print()
        print("=" * 78)
        print(" Preview — plotting samples BEFORE training (matplotlib)")
        print("=" * 78)
        try:
            dataset_root_hint = Path(args.dataset_root).resolve() if args.dataset_root else None
            # If dataset was prepared just now, prefer that root; else fallback to hint
            # Try to infer hint from yaml parent if yaml exists
            if data_yaml_path and data_yaml_path.exists():
                # prefer yaml parent as hint too
                pass
            save_hint = Path(args.plot_save) if args.plot_save else None
            out = plot_yolo_samples_matplotlib(
                yaml_path=data_yaml_path,
                dataset_root_hint=dataset_root_hint,
                num=args.plot_n,
                split=args.plot_split,
                save_path=save_hint,
                show=args.plot_show,
                seed=args.plot_seed,
                max_boxes=args.plot_max_boxes,
                dpi=args.plot_dpi,
            )
            if out is not None:
                print(f"[PLOT] Preview ready: {out}")
                print(f"[PLOT] Open with:  xdg-open \"{out}\"  or  python -m http.server")
            else:
                print("[PLOT][WARN] Preview not created (see warnings above)")
        except Exception as e:
            print(f"[PLOT][ERROR] Preview failed: {e}", file=sys.stderr)
            import traceback; traceback.print_exc()
            # don't abort training just because plot failed
        print("=" * 78)
        print()
        if args.plot_only:
            print("[PLOT-ONLY] Exiting after preview (no training started). Remove --plot-only to train.")
            sys.exit(0)

    # ---- build training kwargs ----
    aug_kwargs = resolve_augmentation_args(args, parser)

    train_kwargs: Dict = dict(
        data=str(data_yaml_path),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        workers=args.workers,
        device=args.device,
        project=args.project,
        name=args.name,
        exist_ok=args.exist_ok,
        patience=args.patience,
        save_period=args.save_period,
        cache=normalize_cache_arg(args.cache),
        rect=args.rect,
        amp=args.amp,
        optimizer=args.optimizer,
        pretrained=not from_scratch,
        verbose=args.verbose,
        seed=args.seed,
        deterministic=args.deterministic,
        single_cls=args.single_cls,
        plots=args.plots,
        val=args.do_val,
        cos_lr=args.cos_lr,
        profile=args.profile,
    )
    optional_map = {
        "lr0": args.lr0, "lrf": args.lrf, "momentum": args.momentum,
        "weight_decay": args.weight_decay, "warmup_epochs": args.warmup_epochs,
        "close_mosaic": args.close_mosaic, "multi_scale": args.multi_scale,
        "dropout": args.dropout, "fraction": args.fraction, "freeze": args.freeze,
    }
    for k, v in optional_map.items():
        if v is not None:
            train_kwargs[k] = v

    train_kwargs.update(aug_kwargs)

    if args.resume is not None:
        if isinstance(args.resume, str):
            low = args.resume.lower()
            if low in ("true", "1", "yes"):
                train_kwargs["resume"] = True
            elif low in ("false", "0", "no"):
                pass
            else:
                train_kwargs["resume"] = args.resume
        else:
            train_kwargs["resume"] = bool(args.resume)

    train_kwargs = {k: v for k, v in train_kwargs.items() if v is not None}
    if train_kwargs.get("name") is None:
        train_kwargs.pop("name", None)
    if train_kwargs.get("device") is None:
        train_kwargs.pop("device", None)
    if train_kwargs.get("freeze") is None:
        train_kwargs.pop("freeze", None)

    print()
    print("=" * 78)
    print(" Effective YOLO.train() kwargs")
    print("=" * 78)
    for k in sorted(train_kwargs.keys()):
        print(f"  {k:18s}= {train_kwargs[k]!r}")
    if aug_kwargs:
        print()
        print("  -> augmentation overrides active")
    else:
        print()
        print("  -> no augmentation overrides; Ultralytics defaults will be used")

    print()
    print(f"  Dataset YAML : {data_yaml_path}")
    print(f"  Model        : {args.model}{' (from scratch, random init)' if from_scratch else f' -> {model_path.resolve()}'}")
    print("=" * 78)

    if args.dry_run:
        print("[DRY-RUN] Prepared dataset and printed config. NOT starting training (remove --dry-run to train).")
        print(f"[DRY-RUN] To train, run:")
        print(f"  python {Path(__file__).name} --model {model_path} --data-yaml {data_yaml_path} --epochs {args.epochs} --imgsz {args.imgsz} --batch {args.batch}")
        if not do_plot:
            print(f"  Tip: add --plot --plot-n 6  to preview samples before training")
        sys.exit(0)

    # ---- start training ----
    try:
        from ultralytics import YOLO
    except ImportError:
        print("[ERROR] ultralytics not installed. Run: pip install ultralytics torch", file=sys.stderr)
        sys.exit(4)

    print()
    print(f"[INFO] Loading model: {args.model}"
          + ("  (architecture only — weights randomly initialized, training FROM SCRATCH)"
             if from_scratch else ""))
    model = YOLO(str(model_path))

    print(f"[INFO] Starting training for {args.epochs} epochs (imgsz={args.imgsz}, batch={args.batch}) ...")
    print(f"[INFO] Project={args.project}  Name={args.name or '(auto)'}  Device={args.device or 'auto'}")
    print(f"[INFO] Logs / checkpoints -> {args.project}/")
    print()

    try:
        results = model.train(**train_kwargs)
        print()
        print("=" * 78)
        print(" Training finished")
        print("=" * 78)
        if results is not None:
            print(f"  Results: {results}")
        proj = Path(args.project or "runs/train")
        try:
            save_dir = getattr(getattr(model, "trainer", None), "save_dir", None)
            if save_dir:
                print(f"  Save dir        : {save_dir}")
                best = Path(save_dir) / "weights" / "best.pt"
                if best.exists():
                    print(f"  Best checkpoint : {best.resolve()}")
                else:
                    print(f"  (expected best at {best})")
            else:
                candidates = sorted(proj.glob("train*/weights/best.pt"), key=lambda p: p.stat().st_mtime, reverse=True)
                if candidates:
                    print(f"  Best checkpoint (guess): {candidates[0].resolve()}")
        except Exception:
            pass
        print("=" * 78)
    except Exception as e:
        print(f"[ERROR] Training failed: {e}", file=sys.stderr)
        import traceback; traceback.print_exc()
        sys.exit(5)


if __name__ == "__main__":
    main()
