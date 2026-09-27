#!/usr/bin/env python3
"""
train_gui.py — Tkinter GUI front-end for train.py (YOLO training script)
--------------------------------------------------------------------------
Exposes every train.py CLI option as a proper widget (entry / dropdown /
checkbox / file-picker), lets you save & reload parameter presets, shows
the exact command that will be run, and streams train.py's live stdout
into a log console — with a Stop button to kill a running job.

Run it with:
    python train_gui.py

Requires only the Python standard library (tkinter). train.py itself still
needs `ultralytics` + `torch` installed, but the GUI will open fine even
without them — you'll just get an error in the log when you hit Run.

Place this file in the same folder as train.py (or point "Model" /
"Data dir" fields wherever you like — the folder is only used to locate
train.py and to load its defaults).
"""
import json
import os
import queue
import subprocess
import sys
import threading
import tkinter as tk
from pathlib import Path
from tkinter import ttk, filedialog, messagebox, scrolledtext

SCRIPT_DIR = Path(__file__).resolve().parent
TRAIN_SCRIPT = SCRIPT_DIR / "train.py"
CONFIG_FILE = SCRIPT_DIR / "train_gui_config.json"

# Try to import train.py to reuse its defaults (safe: no heavy deps at
# module import time — ultralytics is only imported inside functions).
try:
    sys.path.insert(0, str(SCRIPT_DIR))
    import train as train_mod  # type: ignore
    DEFAULT_MODEL = train_mod.DEFAULT_MODEL
    DEFAULT_DATA_DIR = train_mod.DEFAULT_DATA_DIR
    DEFAULT_DATASET_ROOT = train_mod.DEFAULT_DATASET_ROOT
except Exception:
    DEFAULT_MODEL = "yolov8l.yaml"
    DEFAULT_DATA_DIR = ""
    DEFAULT_DATASET_ROOT = ""


# ---------------------------------------------------------------------------
# Field specification: one row per train.py CLI argument.
#   key       - internal id / config key
#   label     - shown in the GUI
#   kind      - 'str' | 'int' | 'float' | 'bool_toggle' | 'bool_flag' | 'choice'
#   default   - prefilled value (non-empty defaults are ALWAYS sent as a flag;
#               empty string / None means "optional, only sent if filled in")
#   flag      - CLI flag to emit
#   neg_flag  - only for bool_toggle: flag emitted when unchecked
#   choices   - only for 'choice'
#   browse    - None | 'file' | 'dir' | 'save'
#   help      - short tooltip text
# ---------------------------------------------------------------------------
FIELDS = [
    # ---------------- Data / Model ----------------
    dict(tab="Data / Model", key="model", label="Model", kind="str",
         default=DEFAULT_MODEL, flag="--model", browse="file",
         help="Architecture .yaml (train from scratch, random weights) or a .pt checkpoint (fine-tune)."),
    dict(tab="Data / Model", key="data_dir", label="Data dir", kind="str",
         default=DEFAULT_DATA_DIR, flag="--data-dir", browse="dir",
         help="Flat folder of images + co-located <image>.txt YOLO labels."),
    dict(tab="Data / Model", key="data_yaml", label="Existing data.yaml (optional)", kind="str",
         default="", flag="--data-yaml", browse="file",
         help="If set, dataset preparation is skipped (unless Force-prepare is checked)."),
    dict(tab="Data / Model", key="dataset_root", label="Dataset root (output)", kind="str",
         default=DEFAULT_DATASET_ROOT, flag="--dataset-root", browse="dir",
         help="Where the prepared images/train, labels/train, dataset.yaml are written."),
    dict(tab="Data / Model", key="val_split", label="Val split", kind="float",
         default="0.2", flag="--val-split",
         help="Fraction of images used for validation (0.0-1.0)."),
    dict(tab="Data / Model", key="seed", label="Seed", kind="int",
         default="0", flag="--seed", help="Random seed for the train/val shuffle."),
    dict(tab="Data / Model", key="single_cls", label="Single class (single_cls)", kind="bool_flag",
         default=False, flag="--single-cls", help="Treat every label as one class."),
    dict(tab="Data / Model", key="force_prepare", label="Force re-prepare dataset", kind="bool_flag",
         default=False, flag="--force-prepare", help="Regenerate dataset even if data.yaml already exists."),
    dict(tab="Data / Model", key="clean", label="Clean dataset root first", kind="bool_flag",
         default=False, flag="--clean", help="Delete dataset-root before preparing a fresh copy."),
    dict(tab="Data / Model", key="symlink", label="Use symlinks (not copies)", kind="bool_flag",
         default=False, flag="--symlink", help="Faster, but not portable across machines."),
    dict(tab="Data / Model", key="no_prepare", label="Skip dataset preparation", kind="bool_flag",
         default=False, flag="--no-prepare", help="Assumes dataset already exists at dataset-root / data-yaml."),

    # ---------------- Training ----------------
    dict(tab="Training", key="epochs", label="Epochs", kind="int",
         default="100", flag="--epochs"),
    dict(tab="Training", key="imgsz", label="Image size", kind="int",
         default="640", flag="--imgsz"),
    dict(tab="Training", key="batch", label="Batch size", kind="int",
         default="8", flag="--batch", help="-1 = auto-batch."),
    dict(tab="Training", key="workers", label="Dataloader workers", kind="int",
         default="8", flag="--workers"),
    dict(tab="Training", key="device", label="Device (optional)", kind="str",
         default="", flag="--device", help="e.g. '0', '0,1', 'cpu', 'mps'. Blank = auto."),
    dict(tab="Training", key="project", label="Project dir", kind="str",
         default="runs/train", flag="--project", browse="dir"),
    dict(tab="Training", key="name", label="Run name (optional)", kind="str",
         default="", flag="--name", help="Blank = auto-incremented name."),
    dict(tab="Training", key="exist_ok", label="Allow overwrite (exist_ok)", kind="bool_flag",
         default=False, flag="--exist-ok"),
    dict(tab="Training", key="patience", label="Early-stop patience", kind="int",
         default="50", flag="--patience"),
    dict(tab="Training", key="save_period", label="Save checkpoint every N epochs", kind="int",
         default="-1", flag="--save-period", help="-1 disables periodic checkpoints."),
    dict(tab="Training", key="cache", label="Cache images", kind="choice",
         default="False", flag="--cache", choices=["False", "True", "ram", "disk"]),
    dict(tab="Training", key="rect", label="Rectangular training (rect)", kind="bool_flag",
         default=False, flag="--rect"),
    dict(tab="Training", key="resume", label="Resume from (optional)", kind="str",
         default="", flag="--resume", browse="file",
         help="Path to last.pt, or leave blank. Type 'true' to auto-resume."),
    dict(tab="Training", key="freeze", label="Freeze layers (optional)", kind="str",
         default="", flag="--freeze", help="e.g. '10'."),
    dict(tab="Training", key="amp", label="AMP (mixed precision)", kind="bool_toggle",
         default=True, flag="--amp", neg_flag="--no-amp"),
    dict(tab="Training", key="optimizer", label="Optimizer", kind="choice",
         default="auto", flag="--optimizer",
         choices=["auto", "SGD", "Adam", "AdamW", "NAdam", "RAdam", "RMSProp"]),
    dict(tab="Training", key="lr0", label="Initial LR (lr0, optional)", kind="float",
         default="", flag="--lr0"),
    dict(tab="Training", key="lrf", label="Final LR factor (lrf, optional)", kind="float",
         default="", flag="--lrf"),
    dict(tab="Training", key="momentum", label="Momentum (optional)", kind="float",
         default="", flag="--momentum"),
    dict(tab="Training", key="weight_decay", label="Weight decay (optional)", kind="float",
         default="", flag="--weight-decay"),
    dict(tab="Training", key="warmup_epochs", label="Warmup epochs (optional)", kind="float",
         default="", flag="--warmup-epochs"),
    dict(tab="Training", key="close_mosaic", label="Close mosaic last N epochs (optional)", kind="int",
         default="", flag="--close-mosaic"),
    dict(tab="Training", key="multi_scale", label="Multi-scale factor (optional)", kind="float",
         default="", flag="--multi-scale"),
    dict(tab="Training", key="cos_lr", label="Cosine LR schedule", kind="bool_flag",
         default=False, flag="--cos-lr"),
    dict(tab="Training", key="dropout", label="Dropout (optional)", kind="float",
         default="", flag="--dropout"),
    dict(tab="Training", key="do_val", label="Validate during training", kind="bool_toggle",
         default=True, flag="--val", neg_flag="--no-val"),
    dict(tab="Training", key="plots", label="Save training plots", kind="bool_toggle",
         default=True, flag="--plots", neg_flag="--no-plots"),
    dict(tab="Training", key="fraction", label="Dataset fraction to use (optional)", kind="float",
         default="", flag="--fraction"),
    dict(tab="Training", key="profile", label="Profile training", kind="bool_flag",
         default=False, flag="--profile"),
    dict(tab="Training", key="deterministic", label="Deterministic", kind="bool_toggle",
         default=True, flag="--deterministic", neg_flag="--nondeterministic"),

    # ---------------- Augmentation ----------------
    dict(tab="Augmentation", key="aug_preset", label="Preset (optional)", kind="choice",
         default="", flag="--aug-preset", choices=["", "light", "medium", "heavy", "none"],
         help="Applies a bundle of augment values; individual fields below still override it."),
    dict(tab="Augmentation", key="hsv_h", label="HSV Hue (optional)", kind="float", default="", flag="--hsv-h"),
    dict(tab="Augmentation", key="hsv_s", label="HSV Saturation (optional)", kind="float", default="", flag="--hsv-s"),
    dict(tab="Augmentation", key="hsv_v", label="HSV Value (optional)", kind="float", default="", flag="--hsv-v"),
    dict(tab="Augmentation", key="degrees", label="Rotation degrees (optional)", kind="float", default="", flag="--degrees"),
    dict(tab="Augmentation", key="translate", label="Translate fraction (optional)", kind="float", default="", flag="--translate"),
    dict(tab="Augmentation", key="scale", label="Scale gain (optional)", kind="float", default="", flag="--scale"),
    dict(tab="Augmentation", key="shear", label="Shear degrees (optional)", kind="float", default="", flag="--shear"),
    dict(tab="Augmentation", key="perspective", label="Perspective (optional)", kind="float", default="", flag="--perspective"),
    dict(tab="Augmentation", key="flipud", label="Flip-up-down prob (optional)", kind="float", default="", flag="--flipud"),
    dict(tab="Augmentation", key="fliplr", label="Flip-left-right prob (optional)", kind="float", default="", flag="--fliplr"),
    dict(tab="Augmentation", key="bgr", label="BGR channel-shuffle prob (optional)", kind="float", default="", flag="--bgr"),
    dict(tab="Augmentation", key="mosaic", label="Mosaic prob (optional)", kind="float", default="", flag="--mosaic"),
    dict(tab="Augmentation", key="mixup", label="Mixup prob (optional)", kind="float", default="", flag="--mixup"),
    dict(tab="Augmentation", key="cutmix", label="CutMix prob (optional)", kind="float", default="", flag="--cutmix"),
    dict(tab="Augmentation", key="copy_paste", label="Copy-paste prob (optional)", kind="float", default="", flag="--copy-paste"),
    dict(tab="Augmentation", key="copy_paste_mode", label="Copy-paste mode (optional)", kind="choice",
         default="", flag="--copy-paste-mode", choices=["", "flip", "mixup"]),
    dict(tab="Augmentation", key="auto_augment", label="Auto-augment policy (optional)", kind="str",
         default="", flag="--auto-augment", help="randaugment / autoaugment / augmix / '' to disable."),
    dict(tab="Augmentation", key="erasing", label="Random erasing prob (optional)", kind="float", default="", flag="--erasing"),

    # ---------------- Preview ----------------
    dict(tab="Preview", key="plot", label="Plot samples before training", kind="bool_flag",
         default=False, flag="--plot", help="Sanity-check a few images with GT boxes before training starts."),
    dict(tab="Preview", key="plot_only", label="Plot only (no training)", kind="bool_flag",
         default=False, flag="--plot-only"),
    dict(tab="Preview", key="plot_n", label="Number of samples", kind="int",
         default="6", flag="--plot-n"),
    dict(tab="Preview", key="plot_split", label="Split to sample from", kind="choice",
         default="train", flag="--plot-split", choices=["train", "val", "all"]),
    dict(tab="Preview", key="plot_save", label="Save preview PNG to (optional)", kind="str",
         default="", flag="--plot-save", browse="save"),
    dict(tab="Preview", key="plot_show", label="Also show interactively (plt.show)", kind="bool_flag",
         default=False, flag="--plot-show", help="Needs a display; ignored if headless."),
    dict(tab="Preview", key="plot_seed", label="Sampling seed", kind="int",
         default="0", flag="--plot-seed"),
    dict(tab="Preview", key="plot_max_boxes", label="Max boxes drawn per image (optional)", kind="int",
         default="", flag="--plot-max-boxes", help="Useful for very dense (>300 box) scenes."),
    dict(tab="Preview", key="plot_dpi", label="Preview PNG DPI", kind="int",
         default="150", flag="--plot-dpi"),

    # ---------------- Misc ----------------
    dict(tab="Misc", key="dry_run", label="Dry run (prepare + show config only)", kind="bool_flag",
         default=False, flag="--dry-run"),
    dict(tab="Misc", key="check_only", label="Check only (validate + exit)", kind="bool_flag",
         default=False, flag="--check-only"),
    dict(tab="Misc", key="verbose", label="Verbose logging", kind="bool_toggle",
         default=True, flag="--verbose", neg_flag="--no-verbose"),
]

TAB_ORDER = ["Data / Model", "Training", "Augmentation", "Preview", "Misc"]


# ---------------------------------------------------------------------------
# Small tooltip helper
# ---------------------------------------------------------------------------
class Tooltip:
    def __init__(self, widget, text):
        self.widget = widget
        self.text = text
        self.tip = None
        widget.bind("<Enter>", self._show)
        widget.bind("<Leave>", self._hide)

    def _show(self, _event=None):
        if self.tip or not self.text:
            return
        x = self.widget.winfo_rootx() + 12
        y = self.widget.winfo_rooty() + self.widget.winfo_height() + 6
        self.tip = tw = tk.Toplevel(self.widget)
        tw.wm_overrideredirect(True)
        tw.wm_geometry(f"+{x}+{y}")
        tk.Label(tw, text=self.text, justify="left", background="#ffffe0",
                 relief="solid", borderwidth=1, wraplength=360,
                 font=("TkDefaultFont", 9)).pack(ipadx=4, ipady=2)

    def _hide(self, _event=None):
        if self.tip:
            self.tip.destroy()
            self.tip = None


class ScrollableFrame(ttk.Frame):
    """A vertically-scrollable frame (mouse wheel enabled)."""
    def __init__(self, parent):
        super().__init__(parent)
        canvas = tk.Canvas(self, borderwidth=0, highlightthickness=0)
        vscroll = ttk.Scrollbar(self, orient="vertical", command=canvas.yview)
        self.inner = ttk.Frame(canvas)

        self.inner.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        window_id = canvas.create_window((0, 0), window=self.inner, anchor="nw")
        canvas.bind("<Configure>", lambda e: canvas.itemconfig(window_id, width=e.width))
        canvas.configure(yscrollcommand=vscroll.set)

        canvas.pack(side="left", fill="both", expand=True)
        vscroll.pack(side="right", fill="y")

        def _wheel(event):
            delta = -1 if event.num == 5 or event.delta < 0 else 1
            canvas.yview_scroll(-delta, "units")
        canvas.bind_all("<MouseWheel>", lambda e: canvas.yview_scroll(-1 if e.delta > 0 else 1, "units"))
        canvas.bind_all("<Button-4>", _wheel)
        canvas.bind_all("<Button-5>", _wheel)


# ---------------------------------------------------------------------------
# Main application
# ---------------------------------------------------------------------------
class TrainGUI(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("YOLO Trainer")
        self.geometry("980x760")
        self.minsize(820, 600)

        self.vars = {}
        self.proc = None
        self.log_queue = queue.Queue()
        self.run_thread = None

        self._build_menu()
        self._build_layout()
        self._load_config(silent=True)
        self.after(150, self._poll_log_queue)
        self.protocol("WM_DELETE_WINDOW", self._on_close)

    # ---------------- menu ----------------
    def _build_menu(self):
        menubar = tk.Menu(self)
        filemenu = tk.Menu(menubar, tearoff=0)
        filemenu.add_command(label="Save config...", command=self._save_config_as)
        filemenu.add_command(label="Load config...", command=self._load_config_dialog)
        filemenu.add_command(label="Reset to defaults", command=self._reset_defaults)
        filemenu.add_separator()
        filemenu.add_command(label="Exit", command=self._on_close)
        menubar.add_cascade(label="File", menu=filemenu)

        helpmenu = tk.Menu(menubar, tearoff=0)
        helpmenu.add_command(label="About", command=lambda: messagebox.showinfo(
            "About", "GUI front-end for train.py\nRuns train.py as a subprocess with the "
                     "parameters set below and streams its output live."))
        menubar.add_cascade(label="Help", menu=helpmenu)
        self.config(menu=menubar)

    # ---------------- layout ----------------
    def _build_layout(self):
        top = ttk.Frame(self, padding=(8, 6))
        top.pack(fill="x")
        ttk.Label(top, text=f"train.py: {TRAIN_SCRIPT}",
                  foreground=("#0a0" if TRAIN_SCRIPT.exists() else "#a00")).pack(side="left")
        if not TRAIN_SCRIPT.exists():
            ttk.Label(top, text="  (not found next to this GUI — place train.py alongside train_gui.py)",
                      foreground="#a00").pack(side="left")

        notebook = ttk.Notebook(self)
        notebook.pack(fill="both", expand=True, padx=8, pady=(0, 4))

        self.tab_frames = {}
        for tab_name in TAB_ORDER:
            scroll = ScrollableFrame(notebook)
            notebook.add(scroll, text=tab_name)
            self.tab_frames[tab_name] = scroll.inner

        for field in FIELDS:
            self._add_field_row(self.tab_frames[field["tab"]], field)

        for tab_name in TAB_ORDER:
            self.tab_frames[tab_name].columnconfigure(1, weight=1)

        # ---- action bar ----
        actions = ttk.Frame(self, padding=(8, 4))
        actions.pack(fill="x")
        self.run_btn = ttk.Button(actions, text="▶ Run", command=self._on_run)
        self.run_btn.pack(side="left", padx=(0, 6))
        self.stop_btn = ttk.Button(actions, text="■ Stop", command=self._on_stop, state="disabled")
        self.stop_btn.pack(side="left", padx=(0, 6))
        ttk.Button(actions, text="Preview command", command=self._on_preview).pack(side="left", padx=(0, 6))
        ttk.Button(actions, text="Save config", command=self._save_config_as).pack(side="left", padx=(0, 6))
        ttk.Button(actions, text="Load config", command=self._load_config_dialog).pack(side="left", padx=(0, 6))
        ttk.Button(actions, text="Clear log", command=self._clear_log).pack(side="left", padx=(0, 6))
        self.status_var = tk.StringVar(value="Idle")
        ttk.Label(actions, textvariable=self.status_var, foreground="#555").pack(side="right")

        # ---- command preview ----
        cmd_frame = ttk.LabelFrame(self, text="Command", padding=4)
        cmd_frame.pack(fill="x", padx=8, pady=(0, 4))
        self.cmd_text = tk.Text(cmd_frame, height=3, wrap="word")
        self.cmd_text.pack(fill="x")
        self.cmd_text.configure(state="disabled")

        # ---- log ----
        log_frame = ttk.LabelFrame(self, text="Log", padding=4)
        log_frame.pack(fill="both", expand=True, padx=8, pady=(0, 8))
        self.log_text = scrolledtext.ScrolledText(log_frame, wrap="word", height=14,
                                                    background="#111", foreground="#ddd",
                                                    insertbackground="#ddd")
        self.log_text.pack(fill="both", expand=True)
        self.log_text.configure(state="disabled")

    def _add_field_row(self, parent, field):
        row = len(parent.grid_slaves())  # rough row index; fine since we add sequentially
        row = getattr(parent, "_row_count", 0)
        parent._row_count = row + 1

        lbl = ttk.Label(parent, text=field["label"])
        lbl.grid(row=row, column=0, sticky="w", padx=(8, 6), pady=4)
        if field.get("help"):
            Tooltip(lbl, field["help"])

        kind = field["kind"]
        key = field["key"]

        if kind in ("bool_toggle", "bool_flag"):
            var = tk.BooleanVar(value=bool(field["default"]))
            widget = ttk.Checkbutton(parent, variable=var)
            widget.grid(row=row, column=1, sticky="w", padx=4, pady=4)
        elif kind == "choice":
            var = tk.StringVar(value=str(field["default"]))
            widget = ttk.Combobox(parent, textvariable=var, values=field["choices"],
                                   state="readonly", width=20)
            widget.grid(row=row, column=1, sticky="w", padx=4, pady=4)
        else:
            var = tk.StringVar(value=str(field["default"]))
            entry_frame = ttk.Frame(parent)
            entry_frame.grid(row=row, column=1, sticky="ew", padx=4, pady=4)
            entry_frame.columnconfigure(0, weight=1)
            widget = ttk.Entry(entry_frame, textvariable=var)
            widget.grid(row=0, column=0, sticky="ew")
            if field.get("browse"):
                btn = ttk.Button(entry_frame, text="Browse...", width=10,
                                  command=lambda v=var, m=field["browse"]: self._browse(v, m))
                btn.grid(row=0, column=1, padx=(4, 0))

        if field.get("help"):
            Tooltip(widget, field["help"])

        self.vars[key] = var

    def _browse(self, var, mode):
        initial = var.get() or str(SCRIPT_DIR)
        if mode == "dir":
            path = filedialog.askdirectory(initialdir=initial or ".")
        elif mode == "save":
            path = filedialog.asksaveasfilename(initialdir=str(Path(initial).parent) if initial else ".")
        else:
            path = filedialog.askopenfilename(initialdir=str(Path(initial).parent) if initial else ".")
        if path:
            var.set(path)

    # ---------------- command building ----------------
    def _build_command(self):
        if not TRAIN_SCRIPT.exists():
            raise FileNotFoundError(f"train.py not found at {TRAIN_SCRIPT}")
        cmd = [sys.executable, "-u", str(TRAIN_SCRIPT)]
        for field in FIELDS:
            key = field["key"]
            kind = field["kind"]
            var = self.vars[key]

            if kind == "bool_toggle":
                cmd.append(field["flag"] if var.get() else field["neg_flag"])
            elif kind == "bool_flag":
                if var.get():
                    cmd.append(field["flag"])
            else:
                val = var.get().strip()
                if val != "":
                    cmd += [field["flag"], val]
        return cmd

    def _on_preview(self):
        try:
            cmd = self._build_command()
        except Exception as e:
            messagebox.showerror("Error", str(e))
            return
        self._show_command(cmd)

    def _show_command(self, cmd):
        pretty = " ".join(_quote(c) for c in cmd)
        self.cmd_text.configure(state="normal")
        self.cmd_text.delete("1.0", "end")
        self.cmd_text.insert("1.0", pretty)
        self.cmd_text.configure(state="disabled")

    # ---------------- run / stop ----------------
    def _on_run(self):
        if self.proc is not None:
            messagebox.showwarning("Already running", "A training job is already running.")
            return
        try:
            cmd = self._build_command()
        except Exception as e:
            messagebox.showerror("Error", str(e))
            return
        self._show_command(cmd)
        self._append_log(f"$ {' '.join(_quote(c) for c in cmd)}\n\n")

        self.run_btn.configure(state="disabled")
        self.stop_btn.configure(state="normal")
        self.status_var.set("Running...")

        self.run_thread = threading.Thread(target=self._run_worker, args=(cmd,), daemon=True)
        self.run_thread.start()

    def _run_worker(self, cmd):
        try:
            env = os.environ.copy()
            # Force the child to write UTF-8 to stdout regardless of the
            # system console codepage (fixes 'charmap' codec crashes on
            # Windows when ultralytics prints unicode/color characters).
            env["PYTHONIOENCODING"] = "utf-8"
            env["PYTHONUTF8"] = "1"
            self.proc = subprocess.Popen(
                cmd, cwd=str(SCRIPT_DIR), stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, bufsize=1, encoding="utf-8", errors="replace", env=env,
            )
            for line in self.proc.stdout:
                self.log_queue.put(line)
            self.proc.wait()
            code = self.proc.returncode
        except Exception as e:
            self.log_queue.put(f"\n[GUI ERROR] Failed to launch: {e}\n")
            code = -1
        finally:
            self.log_queue.put(("__DONE__", code))
            self.proc = None

    def _on_stop(self):
        if self.proc is None:
            return
        if not messagebox.askyesno("Stop training", "Terminate the running process?"):
            return
        try:
            self.proc.terminate()
            self._append_log("\n[GUI] Sent terminate signal...\n")
        except Exception as e:
            self._append_log(f"\n[GUI] Failed to terminate: {e}\n")

    def _poll_log_queue(self):
        try:
            while True:
                item = self.log_queue.get_nowait()
                if isinstance(item, tuple) and item and item[0] == "__DONE__":
                    code = item[1]
                    self._append_log(f"\n[GUI] Process finished (exit code {code}).\n")
                    self.status_var.set(f"Finished (exit {code})" if code == 0 else f"Failed (exit {code})")
                    self.run_btn.configure(state="normal")
                    self.stop_btn.configure(state="disabled")
                else:
                    self._append_log(item)
        except queue.Empty:
            pass
        self.after(150, self._poll_log_queue)

    def _append_log(self, text):
        self.log_text.configure(state="normal")
        self.log_text.insert("end", text)
        self.log_text.see("end")
        self.log_text.configure(state="disabled")

    def _clear_log(self):
        self.log_text.configure(state="normal")
        self.log_text.delete("1.0", "end")
        self.log_text.configure(state="disabled")

    # ---------------- config save/load ----------------
    def _collect_config(self):
        cfg = {}
        for field in FIELDS:
            var = self.vars[field["key"]]
            cfg[field["key"]] = var.get()
        return cfg

    def _apply_config(self, cfg):
        for field in FIELDS:
            key = field["key"]
            if key in cfg:
                try:
                    self.vars[key].set(cfg[key])
                except Exception:
                    pass

    def _save_config_as(self):
        path = filedialog.asksaveasfilename(
            initialdir=str(SCRIPT_DIR), initialfile="train_gui_config.json",
            defaultextension=".json", filetypes=[("JSON", "*.json")])
        if not path:
            return
        try:
            Path(path).write_text(json.dumps(self._collect_config(), indent=2), encoding="utf-8")
            self.status_var.set(f"Saved config -> {path}")
        except Exception as e:
            messagebox.showerror("Error", f"Could not save config: {e}")

    def _load_config_dialog(self):
        path = filedialog.askopenfilename(initialdir=str(SCRIPT_DIR), filetypes=[("JSON", "*.json")])
        if not path:
            return
        self._load_config(path=path)

    def _load_config(self, path=None, silent=False):
        p = Path(path) if path else CONFIG_FILE
        if not p.exists():
            return
        try:
            cfg = json.loads(p.read_text(encoding="utf-8"))
            self._apply_config(cfg)
            if not silent:
                self.status_var.set(f"Loaded config <- {p}")
        except Exception as e:
            if not silent:
                messagebox.showerror("Error", f"Could not load config: {e}")

    def _reset_defaults(self):
        if not messagebox.askyesno("Reset", "Reset all fields to their defaults?"):
            return
        for field in FIELDS:
            self.vars[field["key"]].set(field["default"])

    def _on_close(self):
        if self.proc is not None:
            if not messagebox.askyesno("Quit", "A training job is running. Quit anyway (it keeps running)?"):
                return
        try:
            CONFIG_FILE.write_text(json.dumps(self._collect_config(), indent=2), encoding="utf-8")
        except Exception:
            pass
        self.destroy()


def _quote(s: str) -> str:
    if s == "" or any(c.isspace() for c in s):
        return f'"{s}"'
    return s


if __name__ == "__main__":
    app = TrainGUI()
    app.mainloop()