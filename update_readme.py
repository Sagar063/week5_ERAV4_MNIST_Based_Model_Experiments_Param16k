#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
update_readme.py

Regenerates README.md for the MNIST experiments repo.

Assumptions:
- results/results.csv already uses the latest 21-column schema (NO mixed/old rows handling).

What this script does:
- Loads results/results.csv (21 columns in the exact order below).
- Builds a rich README.md that includes:
  * Static intro (cleaned sentence + code-fenced folder structure).
  * Objective.
  * Model: TinyMNISTNet (detailed architecture, shape evolution, parameter table; dynamic last_channels).
  * Experiment Design with explanations (LR, OneCycleLR, StepLR, ReduceLROnPlateau;
    how optimizers update weights; activation functions overview).
  * Best Result (So Far).
  * Full Results table ONLY (sorted by val_acc desc, then fewer params, then lower val_loss, then train_time_sec).
  * NEW: Combined Learning Curves (All Experiments): generates and embeds single plots that compare
         all experiments' validation accuracy and validation loss (and training metrics if available).
  * Learning Curves & Diagnostics for ALL experiments (per-run images if present).
    Also links per-epoch CSV if present.

Notes:
- This script now uses matplotlib (Agg backend) to generate combined plots.
- POSIX-normalizes paths in README so GitHub renders images correctly.

Requirements: pandas, matplotlib (and Python stdlib only)
Windows-friendly: no GUI ops; file I/O only.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import List, Optional, Tuple, Dict

import pandas as pd

# Use non-GUI backend for matplotlib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SHOW_PER_RUN_ACCURACY = False
SHOW_PER_RUN_LOSS = False
SHOW_PER_RUN_CM = True
SHOW_PER_RUN_MISCLS = True
# ---------- Paths & Expected Schema ----------
REPO_ROOT = Path(__file__).resolve().parent
RESULTS_DIR = REPO_ROOT / "results"
RESULTS_CSV = RESULTS_DIR / "results.csv"
PLOTS_DIR = RESULTS_DIR / "plots"
README_MD = REPO_ROOT / "README.md"

# Exact, latest schema (21 columns), left-to-right:
EXPECTED_COLUMNS: List[str] = [
    "exp_name", "model_variant", "use_bn", "dropout_p", "activation",
    "optimizer", "scheduler", "lr", "weight_decay", "batch_size", "epochs",
    "params", "train_time_sec", "best_epoch", "val_acc", "val_loss",
    "epoch_csv", "acc_plot", "loss_plot", "cm_plot", "miscls_plot"
]

SECTION_DIVIDER = "\n---\n"


# ---------- Helpers ----------
def err(msg: str) -> None:
    print(f"[update_readme] {msg}", file=sys.stderr)


def _to_posix(rel_path: str) -> str:
    """
    Convert Windows-style paths (with backslashes) to POSIX-style forward slashes
    so GitHub Markdown can render images/links. Also trims a leading './'.
    """
    if not rel_path:
        return ""
    s = Path(rel_path).as_posix()
    if s.startswith("./"):
        s = s[2:]
    return s


def file_exists(rel_path: str) -> bool:
    if not rel_path:
        return False
    return (REPO_ROOT / rel_path).exists()


def format_float(val: Optional[float], ndigits: int) -> str:
    if pd.isna(val):
        return ""
    try:
        return f"{float(val):.{ndigits}f}"
    except Exception:
        return ""


def format_time(val: Optional[float]) -> str:
    if pd.isna(val):
        return ""
    try:
        return f"{float(val):.1f}"
    except Exception:
        return ""


def best_overall(df: pd.DataFrame) -> pd.Series:
    d = df.copy()
    d["val_acc"] = pd.to_numeric(d["val_acc"], errors="coerce")
    d["params"] = pd.to_numeric(d["params"], errors="coerce")
    d["val_loss"] = pd.to_numeric(d["val_loss"], errors="coerce")
    d["train_time_sec"] = pd.to_numeric(d["train_time_sec"], errors="coerce")
    d = d.sort_values(
        by=["val_acc", "params", "val_loss", "train_time_sec"],
        ascending=[False, True, True, True],
        kind="mergesort"
    )
    return d.iloc[0]

def _best_by_sort(df: pd.DataFrame) -> Optional[pd.Series]:
    """Return best row using the same sort rule as best_overall (val_acc desc,
    then params asc, val_loss asc, train_time asc)."""
    if df is None or df.empty:
        return None
    d = df.copy()
    d["val_acc"] = pd.to_numeric(d["val_acc"], errors="coerce")
    d["params"] = pd.to_numeric(d["params"], errors="coerce")
    d["val_loss"] = pd.to_numeric(d["val_loss"], errors="coerce")
    d["train_time_sec"] = pd.to_numeric(d["train_time_sec"], errors="coerce")
    d = d.sort_values(
        by=["val_acc", "params", "val_loss", "train_time_sec"],
        ascending=[False, True, True, True],
        kind="mergesort"
    )
    return d.iloc[0] if len(d) else None


def typical_param_count(df: pd.DataFrame) -> Optional[int]:
    vc = df["params"].dropna().astype(int).value_counts()
    if vc.empty:
        return None
    return int(vc.idxmax())


def config_summary(r: pd.Series) -> str:
    parts = []
    if "use_bn" in r:
        parts.append(f"BN: {str(r['use_bn']).strip()}")
    if "dropout_p" in r and not pd.isna(r["dropout_p"]):
        parts.append(f"Dropout: {format_float(r['dropout_p'], 3)}")
    if "activation" in r and str(r["activation"]).strip():
        parts.append(f"Activation: {str(r['activation']).strip()}")
    # Optimizer + Scheduler
    opt = str(r.get("optimizer", "")).strip()
    sch = str(r.get("scheduler", "")).strip()
    if opt and sch:
        parts.append(f"Optimizer+Scheduler: {opt} + {sch}")
    elif opt:
        parts.append(f"Optimizer: {opt}")
    if "lr" in r and not pd.isna(r["lr"]):
        parts.append(f"LR: {format_float(r['lr'], 5)}")
    if "batch_size" in r and not pd.isna(r["batch_size"]):
        parts.append(f"BatchSize: {int(r['batch_size'])}")
    if "epochs" in r and not pd.isna(r["epochs"]):
        parts.append(f"Epochs: {int(r['epochs'])}")
    return " | ".join(parts)


def md_image_or_note(title: str, rel_path: str) -> str:
    posix_path = _to_posix(rel_path)
    if rel_path and file_exists(rel_path):
        return f"**{title}:**\n\n![]({posix_path})\n"
    missing_path = posix_path if posix_path else "(not provided)"
    return f"**{title}:** _Missing at '{missing_path}'_\n"


def make_table(df: pd.DataFrame, columns: List[str]) -> str:
    header = "| " + " | ".join(columns) + " |"
    sep = "| " + " | ".join(["---"] * len(columns)) + " |"
    lines = [header, sep]
    for _, row in df.iterrows():
        vals = []
        for c in columns:
            v = row.get(c, "")
            if pd.isna(v):
                v = ""
            vals.append(str(v))
        lines.append("| " + " | ".join(vals) + " |")     # keep alignment
    return "\n".join(lines)


def format_for_display(d: pd.DataFrame) -> pd.DataFrame:
    out = d.copy()
    if "val_acc" in out:
        out["val_acc"] = out["val_acc"].apply(lambda x: format_float(x, 2))
    if "val_loss" in out:
        out["val_loss"] = out["val_loss"].apply(lambda x: format_float(x, 4))
    if "train_time_sec" in out:
        out["train_time_sec"] = out["train_time_sec"].apply(format_time)
    for c in ["batch_size", "epochs", "params", "best_epoch"]:
        if c in out:
            out[c] = out[c].apply(lambda x: "" if pd.isna(x) else str(int(x)))
    if "dropout_p" in out:
        out["dropout_p"] = out["dropout_p"].apply(lambda x: "" if pd.isna(x) else format_float(x, 3))
    if "lr" in out:
        out["lr"] = out["lr"].apply(lambda x: "" if pd.isna(x) else format_float(x, 5))
    return out


# ---------- Combined Plotting ----------
# def _read_epoch_csv(path_str: str) -> Optional[pd.DataFrame]:
#     """
#     Read a per-epoch CSV robustly. Returns a DataFrame or None.
#     Tries to coerce common column names for epochs, accuracy, loss (train/val).
#     """
#     if not path_str:
#         return None
#     p = REPO_ROOT / path_str
#     if not p.exists():
#         return None
#     try:
#         df_ep = pd.read_csv(p)
#     except Exception:
#         return None

#     # Standardize column names (lowercase, strip)
#     df_ep.columns = [c.strip().lower() for c in df_ep.columns]

#     # Ensure an epoch column exists; if not, create a 1..N index
#     if "epoch" not in df_ep.columns:
#         df_ep["epoch"] = range(1, len(df_ep) + 1)

#     return df_ep

def _read_epoch_csv(path_str: str, exp_name: str) -> Optional[pd.DataFrame]:
    """
    Read a per-epoch CSV robustly. Returns a DataFrame or None.
    Tries multiple candidate paths:
      1) epoch_csv as given
      2) epoch_csv with backslashes -> slashes
      3) results/plots/<exp_name>_metrics.csv
    Standardizes column names and ensures an 'epoch' column exists.
    """
    candidates = []
    if path_str:
        candidates.append(path_str)
        candidates.append(Path(path_str).as_posix())

    # common fallback from your repo layout
    candidates.append((PLOTS_DIR / f"{exp_name}_metrics.csv").relative_to(REPO_ROOT).as_posix())

    for cand in candidates:
        p = REPO_ROOT / cand
        if p.exists():
            try:
                df_ep = pd.read_csv(p)
                df_ep.columns = [c.strip().lower() for c in df_ep.columns]
                if "epoch" not in df_ep.columns:
                    df_ep["epoch"] = range(1, len(df_ep) + 1)
                return df_ep
            except Exception:
                continue
    return None


def _pick_col(df_ep: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    for c in candidates:
        if c in df_ep.columns:
            return c
    return None

def prettify_label(exp: str) -> str:
    """
    Shorten long experiment names for cleaner legends in combined plots.
    Adjust the replacements as needed for your naming conventions.
    """
    s = exp

    # Collapse block prefixes
    s = s.replace("A_noBN_noDO_", "A/")
    s = s.replace("B_bn_do_", "B/")
    s = s.replace("C_bs_sweep_", "C/")
    s = s.replace("D_activation_", "D/")

    # Optimizers + schedulers
    s = s.replace("_onecycle", "+1cyc")
    s = s.replace("_step", "+step")
    s = s.replace("_plateau", "+plat")

    # Common optimizers
    s = s.replace("adamw", "AdamW")
    s = s.replace("adam", "Adam")
    s = s.replace("rmsprop", "RMSprop")
    s = s.replace("sgd", "SGD")

    # Batch sizes
    s = s.replace("_bs32", "/32")
    s = s.replace("_bs64", "/64")
    s = s.replace("_bs128", "/128")

    # Dropout shorthand
    s = s.replace("_drop", " d")

    return s

def build_dynamic_conclusions(df: pd.DataFrame) -> str:
    """
    Build a narrative Conclusions section directly from the CSV.
    Mirrors the human-written summary:
      - BN/Dropout importance (A vs others)
      - Optimizer+scheduler highlights
      - Batch-size sweep takeaways
      - Activations block
      - Best overall run (name + stats)
      - Final takeaway with typical params
    """
    if df is None or df.empty:
        return "## Conclusions\n\n_Results not available._\n"

    d = df.copy()
    # normalize types
    for col in ("val_acc", "val_loss", "params", "epochs", "best_epoch", "train_time_sec", "batch_size"):
        if col in d.columns:
            d[col] = pd.to_numeric(d[col], errors="coerce")

    # Overall best
    best = _best_by_sort(d)
    best_name = str(best.get("exp_name", "")) if best is not None else ""
    best_acc = format_float(best.get("val_acc"), 2) if best is not None else ""
    best_loss = format_float(best.get("val_loss"), 4) if best is not None else ""
    best_params = f"{int(best['params']):,}" if (best is not None and not pd.isna(best.get('params'))) else ""
    best_ep = str(int(best["best_epoch"])) if (best is not None and not pd.isna(best.get("best_epoch"))) else ""
    best_epochs = str(int(best["epochs"])) if (best is not None and not pd.isna(best.get("epochs"))) else ""

    # Typical params for the final line
    tp = typical_param_count(d)
    tp_str = f"≈{tp:,} parameters" if tp else "<20k parameters"

    # ≥99.40% count (within ≤20 epochs if you want to enforce)
    reached = d[(d["val_acc"] >= 99.40)]
    n_reached = int(reached.shape[0]) if not reached.empty else 0

    # Block splits by exp_name prefix
    A = d[d["exp_name"].astype(str).str.startswith("A_")]
    B = d[d["exp_name"].astype(str).str.startswith("B_")]
    C = d[d["exp_name"].astype(str).str.startswith("C_")]
    D = d[d["exp_name"].astype(str).str.startswith("D_")]

    # A (baseline)
    a_best = _best_by_sort(A) if not A.empty else None
    a_acc = format_float(a_best.get("val_acc"), 2) if a_best is not None else None

    # B (optimizers)
    def _best_for(opt: str, sch: str) -> Optional[pd.Series]:
        dd = B[(B["optimizer"].astype(str)==opt) & (B["scheduler"].astype(str)==sch)]
        return _best_by_sort(dd) if not dd.empty else None

    b_sgd_1cyc = _best_for("sgd", "onecycle")
    b_adam_1cyc = _best_for("adam", "onecycle")
    b_adamw_step = _best_for("adamw", "step")
    b_rms_plateau = _best_for("rmsprop", "plateau")

    # C (batch sizes)
    c_best = _best_by_sort(C) if not C.empty else None
    # best batch size by peak val_acc (tie break by same rule)
    best_bs = None
    if not C.empty:
        # take best per batch_size
        per_bs = []
        for bs, grp in C.groupby("batch_size"):
            row = _best_by_sort(grp)
            if row is not None:
                per_bs.append(row)
        if per_bs:
            per_bs_df = pd.DataFrame(per_bs)
            row_bs = _best_by_sort(per_bs_df)
            if row_bs is not None and not pd.isna(row_bs.get("batch_size")):
                best_bs = int(row_bs["batch_size"])

    # D (activations)
    d_best = _best_by_sort(D) if not D.empty else None
    # highlight best activation under D_
    d_act = str(d_best.get("activation")) if d_best is not None else None
    d_acc = format_float(d_best.get("val_acc"), 2) if d_best is not None else None

    # Compose narrative bullets
    bullets = []

    # BN + Dropout importance
    if a_acc is not None:
        bullets.append(f"- **BN + Dropout are critical:** the baseline (A) peaked at **{a_acc}%**, while BN/Dropout runs exceeded **99.4%** ({n_reached} runs).")
    else:
        bullets.append("- **BN + Dropout are critical:** BN/Dropout runs exceeded **99.4%** in multiple cases; the no-BN/Dropout baseline was not competitive.")

    # Optimizers highlight
    opt_lines = []
    if b_adam_1cyc is not None:
        opt_lines.append(f"**Adam + OneCycleLR** up to **{format_float(b_adam_1cyc['val_acc'], 2)}%**")
    if b_adamw_step is not None:
        opt_lines.append(f"**AdamW + StepLR** up to **{format_float(b_adamw_step['val_acc'], 2)}%**")
    if b_sgd_1cyc is not None:
        opt_lines.append(f"**SGD + OneCycleLR** up to **{format_float(b_sgd_1cyc['val_acc'], 2)}%**")
    if b_rms_plateau is not None:
        opt_lines.append(f"**RMSprop + ReduceLROnPlateau** around **{format_float(b_rms_plateau['val_acc'], 2)}%**")

    if opt_lines:
        bullets.append("- **Optimizers:** " + "; ".join(opt_lines) + ".")
    else:
        bullets.append("- **Optimizers:** Adaptive methods (Adam/AdamW) and SGD+OneCycleLR were generally strong; RMSprop lagged.")

    # Batch-size sweep takeaway
    if best_bs is not None and c_best is not None:
        bullets.append(f"- **Batch-size sweep (C):** best results clustered at **bs={best_bs}**, with the top C-run reaching **{format_float(c_best['val_acc'], 2)}%**.")
    elif c_best is not None:
        bullets.append(f"- **Batch-size sweep (C):** top C-run reached **{format_float(c_best['val_acc'], 2)}%**.")
    else:
        bullets.append("- **Batch-size sweep (C):** results not available.")

    # Activations
    if d_act is not None and d_acc is not None:
        bullets.append(f"- **Activations (D):** differences were modest on MNIST; best was **{d_act}** at **{d_acc}%** (under AdamW+StepLR).")
    else:
        bullets.append("- **Activations (D):** differences were modest on MNIST.")

    # Final takeaway / best run
    if best is not None:
        bullets.append(
            f"- **Best overall:** `{best_name}` → **{best_acc}%** (val loss **{best_loss}**) "
            f"by epoch **{best_ep}** / {best_epochs}, **{best_params}** params."
        )
        bullets.append(
            f"- **Final takeaway:** With BN + Dropout, thoughtful scheduling (e.g., OneCycleLR/StepLR), and a good batch size, "
            f"TinyMNISTNet ({tp_str}) reliably reaches **≥99.4% within 20 epochs**; the best run achieved **{best_acc}%**."
        )
    conclusion_md = [
    "## Conclusions\n",
    "### A. Baseline (no BN/Dropout, vanilla SGD)\n",
    f"- `{a_best['exp_name']}` peaked at **{a_acc}%** with {int(a_best['params'])} params.\n"
    "  → Clear gap vs. BN+Dropout variants, confirms normalization/regularization are essential.\n\n",
    "### B. BN + Dropout + Optimizers\n",
    f"- Adam OneCycle best: `{b_adam_1cyc['exp_name']}` → **{format_float(b_adam_1cyc['val_acc'], 2)}%** (val loss {format_float(b_adam_1cyc['val_loss'], 4)}).\n"
    f"- AdamW StepLR strong: `{b_adamw_step['exp_name']}` → **{format_float(b_adamw_step['val_acc'], 2)}%**, faster convergence (best epoch {int(b_adamw_step['best_epoch'])}).\n"
    f"- SGD OneCycle: `{b_sgd_1cyc['exp_name']}` → **{format_float(b_sgd_1cyc['val_acc'], 2)}%**.\n"
    f"- RMSprop Plateau weaker, around {format_float(b_rms_plateau['val_acc'], 2)}%.\n\n",
    "### C. BN + Dropout + Batch-Size Sweep\n",
    f"- Best overall run: `{best_name}` → **{best_acc}%** (val loss {best_loss}) at epoch {best_ep}/{best_epochs}, {best_params} params.\n"
    f"- Batch size sweet spot at **{best_bs}**: stable convergence and top accuracy.\n\n",
    "### D. BN + Dropout + Activations\n",
    f"- Best activation: {d_act} → **{d_acc}%** (under AdamW+StepLR).\n"
    "- Differences modest on MNIST (<0.1%).\n\n",
    "### 🏆 Collective Insights\n",
    "- BN + Dropout mandatory — baseline A lagged by ~0.6–0.7% absolute accuracy.\n"
    "- Optimizers: Adam OneCycle and AdamW StepLR were most reliable; RMSprop lagged.\n"
    "- Batch size: sweet spot at 64. Too small/large showed minor trade-offs.\n"
    "- Activations: SiLU/GELU did not significantly outperform ReLU.\n"
    f"- **Best overall:** `{best_name}` → {best_acc}% @ epoch {best_ep}, {best_params} params.\n"
]
    return "".join(conclusion_md)

    #return "## Conclusions\n\n" + "\n".join(bullets) + "\n"


def generate_combined_plots(df: pd.DataFrame) -> Dict[str, str]:
    """
    Generate combined plots for:
      - validation accuracy vs. epoch (all experiments)
      - validation loss vs. epoch (all experiments)
    If training metrics exist, also generate:
      - training accuracy vs. epoch
      - training loss vs. epoch

    Returns dict of generated image relative paths.
    """
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    # Paths to save combined images
    # out_paths: Dict[str, str] = {
    #     "val_acc": _to_posix((PLOTS_DIR / "combined_val_accuracy.png").relative_to(REPO_ROOT).as_posix()),
    #     "val_loss": _to_posix((PLOTS_DIR / "combined_val_loss.png").relative_to(REPO_ROOT).as_posix()),
    #     "train_acc": _to_posix((PLOTS_DIR / "combined_train_accuracy.png").relative_to(REPO_ROOT).as_posix()),
    #     "train_loss": _to_posix((PLOTS_DIR / "combined_train_loss.png").relative_to(REPO_ROOT).as_posix()),
    # }

    out_paths_top5: Dict[str, str] = {
    "val_acc": _to_posix((PLOTS_DIR / "combined_val_accuracy_top5.png").relative_to(REPO_ROOT).as_posix()),
    "val_loss": _to_posix((PLOTS_DIR / "combined_val_loss_top5.png").relative_to(REPO_ROOT).as_posix()),
    "train_acc": _to_posix((PLOTS_DIR / "combined_train_accuracy_top5.png").relative_to(REPO_ROOT).as_posix()),
    "train_loss": _to_posix((PLOTS_DIR / "combined_train_loss_top5.png").relative_to(REPO_ROOT).as_posix()),}
    out_paths_all: Dict[str, str] = {
        "val_acc": _to_posix((PLOTS_DIR / "combined_val_accuracy_all.png").relative_to(REPO_ROOT).as_posix()),
        "val_loss": _to_posix((PLOTS_DIR / "combined_val_loss_all.png").relative_to(REPO_ROOT).as_posix()),
        "train_acc": _to_posix((PLOTS_DIR / "combined_train_accuracy_all.png").relative_to(REPO_ROOT).as_posix()),
        "train_loss": _to_posix((PLOTS_DIR / "combined_train_loss_all.png").relative_to(REPO_ROOT).as_posix()),}

    # # Top-5 series lists Accumulate series
    series_val_acc: List[Tuple[str, pd.Series, pd.Series]] = []   # (label, epoch, val_acc)
    series_val_loss: List[Tuple[str, pd.Series, pd.Series]] = []  # (label, epoch, val_loss)
    series_tr_acc: List[Tuple[str, pd.Series, pd.Series]] = []
    series_tr_loss: List[Tuple[str, pd.Series, pd.Series]] = []

    # NEW: All-experiments series lists
    series_val_acc_all: List[Tuple[str, pd.Series, pd.Series]] = []
    series_val_loss_all: List[Tuple[str, pd.Series, pd.Series]] = []
    series_tr_acc_all: List[Tuple[str, pd.Series, pd.Series]] = []
    series_tr_loss_all: List[Tuple[str, pd.Series, pd.Series]] = []

    ## plot all experiments in train and val accuaracy and loss plots
    # for _, r in df.iterrows():
    #     label = str(r.get("exp_name", ""))
    #     epoch_csv = str(r.get("epoch_csv", ""))
    #     ep_df = _read_epoch_csv(epoch_csv)
    #     if ep_df is None:
    #         continue

    #     # Identify columns
    #     epoch_col = "epoch"
    #     val_acc_col = _pick_col(ep_df, ["val_acc", "val_accuracy", "valid_acc", "valid_accuracy"])
    #     val_loss_col = _pick_col(ep_df, ["val_loss", "valid_loss"])
    #     tr_acc_col = _pick_col(ep_df, ["train_acc", "accuracy", "acc"])  # beware: 'accuracy' might be train in some logs
    #     tr_loss_col = _pick_col(ep_df, ["train_loss", "loss"])           # 'loss' alone likely train loss

    #     # Collect series if present
    #     if val_acc_col and pd.api.types.is_numeric_dtype(ep_df[val_acc_col]):
    #         series_val_acc.append((label, ep_df[epoch_col], ep_df[val_acc_col]))
    #     if val_loss_col and pd.api.types.is_numeric_dtype(ep_df[val_loss_col]):
    #         series_val_loss.append((label, ep_df[epoch_col], ep_df[val_loss_col]))
    #     if tr_acc_col and pd.api.types.is_numeric_dtype(ep_df[tr_acc_col]):
    #         series_tr_acc.append((label, ep_df[epoch_col], ep_df[tr_acc_col]))
    #     if tr_loss_col and pd.api.types.is_numeric_dtype(ep_df[tr_loss_col]):
    #         series_tr_loss.append((label, ep_df[epoch_col], ep_df[tr_loss_col]))

    ## plot all experiments in train and val accuaracy and loss plots
    # --- Build series for ALL experiments ---
    for _, r in df.iterrows():
        raw_name = str(r.get("exp_name", ""))
        label_all = prettify_label(raw_name)
        epoch_csv = str(r.get("epoch_csv", ""))
        ep_df = _read_epoch_csv(epoch_csv, raw_name)
        if ep_df is None:
            continue

        epoch_col = "epoch"
        val_acc_col = _pick_col(ep_df, ["val_acc", "val_accuracy", "valid_acc", "valid_accuracy"])
        val_loss_col = _pick_col(ep_df, ["val_loss", "valid_loss"])
        tr_acc_col = _pick_col(ep_df, ["train_acc", "accuracy", "acc"])
        tr_loss_col = _pick_col(ep_df, ["train_loss", "loss"])

        if val_acc_col and pd.api.types.is_numeric_dtype(ep_df[val_acc_col]):
            series_val_acc_all.append((label_all, ep_df[epoch_col], ep_df[val_acc_col]))
        if val_loss_col and pd.api.types.is_numeric_dtype(ep_df[val_loss_col]):
            series_val_loss_all.append((label_all, ep_df[epoch_col], ep_df[val_loss_col]))
        if tr_acc_col and pd.api.types.is_numeric_dtype(ep_df[tr_acc_col]):
            series_tr_acc_all.append((label_all, ep_df[epoch_col], ep_df[tr_acc_col]))
        if tr_loss_col and pd.api.types.is_numeric_dtype(ep_df[tr_loss_col]):
            series_tr_loss_all.append((label_all, ep_df[epoch_col], ep_df[tr_loss_col]))

    # --- Pick Top-5 experiments by val_acc desc, then params asc, val_loss asc, train_time asc ---
    df_sorted_for_top = df.copy()
    df_sorted_for_top["val_acc"] = pd.to_numeric(df_sorted_for_top["val_acc"], errors="coerce")
    df_sorted_for_top["params"] = pd.to_numeric(df_sorted_for_top["params"], errors="coerce")
    df_sorted_for_top["val_loss"] = pd.to_numeric(df_sorted_for_top["val_loss"], errors="coerce")
    df_sorted_for_top["train_time_sec"] = pd.to_numeric(df_sorted_for_top["train_time_sec"], errors="coerce")
    df_sorted_for_top = df_sorted_for_top.sort_values(
        by=["val_acc", "params", "val_loss", "train_time_sec"],
        ascending=[False, True, True, True],
        kind="mergesort"
    ).reset_index(drop=True)

    top5 = df_sorted_for_top.head(5)
    picked_names = [str(r.get("exp_name", "")) for _, r in top5.iterrows()]

    for _, r in top5.iterrows():
        # label = str(r.get("exp_name", ""))
        # epoch_csv = str(r.get("epoch_csv", ""))
        # ep_df = _read_epoch_csv(epoch_csv)
        label = prettify_label(str(r.get("exp_name", "")))
        epoch_csv = str(r.get("epoch_csv", ""))
        ep_df = _read_epoch_csv(epoch_csv, str(r.get("exp_name", "")))

        if ep_df is None:
            continue

        # Identify columns
        epoch_col = "epoch"
        val_acc_col = _pick_col(ep_df, ["val_acc", "val_accuracy", "valid_acc", "valid_accuracy"])
        val_loss_col = _pick_col(ep_df, ["val_loss", "valid_loss"])
        tr_acc_col = _pick_col(ep_df, ["train_acc", "accuracy", "acc"])
        tr_loss_col = _pick_col(ep_df, ["train_loss", "loss"])

        # Collect series if present
        if val_acc_col and pd.api.types.is_numeric_dtype(ep_df[val_acc_col]):
            series_val_acc.append((label, ep_df[epoch_col], ep_df[val_acc_col]))
        if val_loss_col and pd.api.types.is_numeric_dtype(ep_df[val_loss_col]):
            series_val_loss.append((label, ep_df[epoch_col], ep_df[val_loss_col]))
        if tr_acc_col and pd.api.types.is_numeric_dtype(ep_df[tr_acc_col]):
            series_tr_acc.append((label, ep_df[epoch_col], ep_df[tr_acc_col]))
        if tr_loss_col and pd.api.types.is_numeric_dtype(ep_df[tr_loss_col]):
            series_tr_loss.append((label, ep_df[epoch_col], ep_df[tr_loss_col]))

    # Helper to plot lists
    # def _plot_series(series_list: List[Tuple[str, pd.Series, pd.Series]], title: str, xlabel: str, ylabel: str, save_as: str) -> bool:
    #     if not series_list:
    #         return False
    #     plt.figure()
    #     for label, x, y in series_list:
    #         try:
    #             plt.plot(x.values, y.values, label=label)
    #         except Exception:
    #             continue
    #     plt.title(title)
    #     plt.xlabel(xlabel)
    #     plt.ylabel(ylabel)
    #     plt.legend(loc="best", fontsize="small", ncol=1)
    #     plt.grid(True, alpha=0.3)
    #     # Save
    #     out_file = REPO_ROOT / save_as
    #     out_file.parent.mkdir(parents=True, exist_ok=True)
    #     plt.tight_layout()
    #     plt.savefig(out_file, dpi=150)
    #     plt.close()
    #     return True
    def _plot_series(series_list: List[Tuple[str, pd.Series, pd.Series]], title: str, xlabel: str, ylabel: str, save_as: str) -> bool:
        if not series_list:
            return False

        fig, ax = plt.subplots(figsize=(10, 6))  # bigger canvas
        for label, x, y in series_list:
            try:
                ax.plot(x.values, y.values, label=label, linewidth=1.5)
            except Exception:
                continue

        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)

        # Legend outside, auto columns
        ncol = 2 if len(series_list) > 4 else 1
        ax.legend(
            loc="center left",
            bbox_to_anchor=(1.02, 0.5),
            fontsize="small",
            frameon=False,
            ncol=ncol,
            borderaxespad=0.0
        )

        fig.tight_layout(rect=[0, 0, 0.80, 1])  # leave room on right for legend

        out_file = REPO_ROOT / save_as
        out_file.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_file, dpi=150)
        plt.close(fig)
        return True


    # # Generate plots
    # has_val_acc = _plot_series(series_val_acc, "Validation Accuracy vs Epoch (All Experiments)", "Epoch", "Val Accuracy", out_paths["val_acc"])
    # has_val_loss = _plot_series(series_val_loss, "Validation Loss vs Epoch (All Experiments)", "Epoch", "Val Loss", out_paths["val_loss"])
    # has_tr_acc  = _plot_series(series_tr_acc,  "Training Accuracy vs Epoch (All Experiments)",  "Epoch", "Train Accuracy", out_paths["train_acc"])
    # has_tr_loss = _plot_series(series_tr_loss, "Training Loss vs Epoch (All Experiments)",      "Epoch", "Train Loss", out_paths["train_loss"])

    # First: ALL experiments
    has_val_acc_all = _plot_series(series_val_acc_all, "Validation Accuracy vs Epoch (All Experiments)", "Epoch", "Val Accuracy", out_paths_all["val_acc"])
    has_val_loss_all = _plot_series(series_val_loss_all, "Validation Loss vs Epoch (All Experiments)", "Epoch", "Val Loss", out_paths_all["val_loss"])
    has_tr_acc_all  = _plot_series(series_tr_acc_all,  "Training Accuracy vs Epoch (All Experiments)",  "Epoch", "Train Accuracy", out_paths_all["train_acc"])
    has_tr_loss_all = _plot_series(series_tr_loss_all, "Training Loss vs Epoch (All Experiments)",      "Epoch", "Train Loss", out_paths_all["train_loss"])

    # Then: TOP-5 only
    has_val_acc = _plot_series(series_val_acc, "Validation Accuracy vs Epoch (Top 5 Experiments)", "Epoch", "Val Accuracy", out_paths_top5["val_acc"])
    has_val_loss = _plot_series(series_val_loss, "Validation Loss vs Epoch (Top 5 Experiments)", "Epoch", "Val Loss", out_paths_top5["val_loss"])
    has_tr_acc  = _plot_series(series_tr_acc,  "Training Accuracy vs Epoch (Top 5 Experiments)",  "Epoch", "Train Accuracy", out_paths_top5["train_acc"])
    has_tr_loss = _plot_series(series_tr_loss, "Training Loss vs Epoch (Top 5 Experiments)",      "Epoch", "Train Loss", out_paths_top5["train_loss"])



    # Only return those that were actually generated
    generated = {}

    # TOP-5: these are the ones the README will embed
    if has_val_acc: generated["val_acc"] = out_paths_top5["val_acc"]
    if has_val_loss: generated["val_loss"] = out_paths_top5["val_loss"]
    if has_tr_acc: generated["train_acc"] = out_paths_top5["train_acc"]
    if has_tr_loss: generated["train_loss"] = out_paths_top5["train_loss"]

    # ALL: returned under separate keys (README won’t use them, but files are created)
    if has_val_acc_all: generated["val_acc_all"] = out_paths_all["val_acc"]
    if has_val_loss_all: generated["val_loss_all"] = out_paths_all["val_loss"]
    if has_tr_acc_all: generated["train_acc_all"] = out_paths_all["train_acc"]
    if has_tr_loss_all: generated["train_loss_all"] = out_paths_all["train_loss"]

    generated["picked_list"] = picked_names
    return generated

# ---------- README Builder ----------
def build_readme(df: pd.DataFrame) -> str:
    # Intro (clean sentence) + fenced folder structure (renders reliably)
    intro = """# MNIST Experiments (TinyMNISTNet)

This repo runs a compact set of **reproducible MNIST experiments** with a parameter-budget model.
It compares:
1. **No BN/Dropout** + vanilla gradient descent
2. **With BN/Dropout** + multiple optimizers
3. **With BN/Dropout** + batch-size sweep using the optimizers from (2)
4. **Activation variants** (ReLU, SiLU, GELU) with BN/Dropout

Results are logged to `results/results.csv`. Use `update_readme.py` to generate the README with tables and plots.
Per-experiment epoch CSVs and plots are saved in `results/` and `results/plots/`.

## Folder structure
```text
mnist_experiments/
├─ models/
│  └─ model.py
├─ results/
│  ├─ results.csv
│  └─ plots/
├─ train.py
├─ update_readme.py
├─ requirements.txt
└─ README.md

Quickstart
pip install -r requirements.txt
python train.py --mode grid          # run all experiments
python update_readme.py              # generate README.md with tables + plots

# Single run example
python train.py --mode single --use_bn 1 --dropout_p 0.05 --activation relu \
  --optimizer adamw --scheduler step --lr 0.0025 --weight_decay 1e-4 \
  --batch_size 128 --epochs 15 --augment 1
python update_readme.py
```"""

    # Objective
    objective = (
        "## Objective\n\n"
        "- **Constraints:** `< 20,000` parameters, `≤ 20` epochs, target `≥ 99.4%` validation accuracy "
        "(MNIST 10k test set used as validation; training split is 50k).\n"
    )

    # ----- Model section (dynamic param table, dynamic last_channels) -----
    typical_params = typical_param_count(df)
    typical_line = f"- **Typical total parameters (most common across runs):** ~{typical_params:,}\n" if typical_params else ""

    # Estimate last_channels dynamically (default 32)
    last_ch = 32
    try:
        if not df.empty:
            if "model_variant" in df.columns:
                mv = df["model_variant"].dropna().astype(str)
                if len(mv) > 0 and mv.str.contains("last=").any():
                    last_ch = int(mv.str.extract(r"last=(\d+)").dropna().iloc[0, 0])
    except Exception:
        last_ch = 32

    # compute param counts dynamically
    def conv_params(cin, cout, k, use_bias=True):
        return (cin * cout * (k * k)) + (cout if use_bias else 0)

    conv1_p = conv_params(1, 8, 3)
    conv2_p = conv_params(8, 12, 3)
    conv3_p = conv_params(12, 16, 3)
    conv4_p = conv_params(16, 16, 3)
    conv5_p = conv_params(16, 24, 3)
    conv6_p = conv_params(24, last_ch, 3)
    conv1x1_p = conv_params(last_ch, 10, 1)
    total_p = conv1_p + conv2_p + conv3_p + conv4_p + conv5_p + conv6_p + conv1x1_p

    param_table = f"""| Layer       | In→Out Channels | Kernel | Params |
|-------------|-----------------|--------|--------|
| Conv1       | 1 → 8           | 3×3    | {conv1_p:,} |
| Conv2       | 8 → 12          | 3×3    | {conv2_p:,} |
| Conv3       | 12 → 16         | 3×3    | {conv3_p:,} |
| Conv4       | 16 → 16         | 3×3    | {conv4_p:,} |
| Conv5       | 16 → 24         | 3×3    | {conv5_p:,} |
| Conv6       | 24 → {last_ch}  | 3×3    | {conv6_p:,} |
| Conv1×1     | {last_ch} → 10  | 1×1    | {conv1x1_p:,} |
| **Total**   |                 |        | **{total_p:,}** |
"""

    model = (
        "## Model: TinyMNISTNet\n\n"
        "TinyMNISTNet is a deliberately compact CNN designed for MNIST digits.  \n"
        "It enforces three constraints: **<20k parameters**, **≤20 epochs**, and **≥99.4% accuracy**.\n\n"
        "---\n\n"
        "### Architecture\n\n"
        "```text\n"
        "Input  : [B, 1, 28, 28]\n\n"
        "Conv1  : 1  →  8   (3×3, pad=1)   → [B, 8, 28, 28]\n"
        "Conv2  : 8  → 12   (3×3, pad=1)   → [B, 12, 28, 28]\n"
        "Pool   : 2×2                         [B, 12, 14, 14]\n\n"
        "Conv3  : 12 → 16  (3×3, pad=1)   → [B, 16, 14, 14]\n"
        "Conv4  : 16 → 16  (3×3, pad=1)   → [B, 16, 14, 14]\n"
        "Pool   : 2×2                         [B, 16,  7,  7]\n\n"
        "Conv5  : 16 → 24  (3×3, pad=1)   → [B, 24,  7,  7]\n"
        f"Conv6  : 24 → {last_ch}  (3×3, pad=1)   → [B, {last_ch},  7,  7]\n\n"
        f"Conv1×1: {last_ch} → 10   (1×1)          → [B, 10,  7,  7]\n"
        "GAP    : 7×7 → 1×1                → [B, 10,  1,  1]\n"
        "Flatten → [B, 10]\n"
        "Softmax → class probabilities\n"
        "```\n\n"
        "---\n\n"
        "### Shape Evolution\n\n"
        "- Start: `1×28×28`\n"
        f"- After Conv/Pool blocks: `{last_ch}×7×7`\n"
        f"- 1×1 Conv: `{last_ch}→10`, output `10×7×7`\n"
        "- GAP: average each map → `[10]`\n"
        "- Softmax: probabilities over 10 digits\n\n"
        "---\n\n"
        "### Why 1×1 Conv + GAP?\n\n"
        "- Flattening features with a dense layer would require ~15k+ parameters.\n"
        "- Instead: **1×1 conv** needs only hundreds of weights.\n"
        "- GAP has no parameters, just averages.\n"
        "- Result: <20k params total, less overfitting, faster convergence.\n\n"
        "---\n\n"
        "### Parameter Count\n\n"
        f"{param_table}\n"
        f"{typical_line}"
    )

    # Experiment Design (explanations)
    design = (
        "## Experiment Design\n\n"
        "**What’s varied and why**\n\n"
        "- **Learning Rate (LR):** step size for weight updates each iteration.\n"
        "- **Schedulers:**\n"
        "  - **OneCycleLR:** Increases LR up to a peak then decreases it within a single run; encourages fast convergence and regularization.\n"
        "  - **StepLR:** Multiplies LR by a factor (e.g., 0.1) every fixed number of epochs; a simple decay schedule.\n"
        "  - **ReduceLROnPlateau:** Lowers LR when a monitored metric (e.g., val loss) stops improving; adapts LR to training plateaus.\n"
        "- **Optimizers (how they update weights):**\n"
        "  - **SGD (vanilla):** `w ← w − lr * grad` (no momentum here in baseline A). Simple, stable with proper schedules.\n"
        "  - **SGD + OneCycleLR (B/C):** Same rule but LR follows OneCycle; typically reaches good accuracy quickly.\n"
        "  - **AdamW + StepLR (B/D):** Adam-style adaptive moments with **decoupled weight decay** (better regularization) + StepLR decay.\n"
        "  - **RMSprop + ReduceLROnPlateau (B):** Scales updates by running average of squared gradients; LR reduced when progress stalls.\n"
        "  - **Adam + OneCycleLR (B/C):** Adam’s adaptive moments combined with OneCycle schedule.\n"
        "- **Activations:**\n"
        "  - **ReLU:** max(0, x); cheap, strong baseline.\n"
        "  - **SiLU (Swish):** x * sigmoid(x); smooth, can improve convergence.\n"
        "  - **GELU:** Gaussian-error linear unit; smooth, often strong in transformers/CNNs.\n\n"
        "**Blocks we run**\n\n"
        "- **A. Baseline:** no BN/Dropout, **SGD (no momentum)**.\n"
        "- **B. BN + Dropout + Optimizers:** SGD+OneCycleLR, AdamW+StepLR, RMSprop+ReduceLROnPlateau, Adam+OneCycleLR.\n"
        "- **C. BN + Dropout + Batch sizes:** {32, 64, 128} across optimizers from (B).\n"
        "- **D. BN + Dropout + Activations:** {ReLU, SiLU, GELU} using **AdamW + StepLR**.\n"
    )

    # Best Result
    best = best_overall(df)
    best_md = (
        "## Best Result (So Far)\n\n"
        f"- **Experiment:** `{best.get('exp_name', '')}`\n"
        f"- **Val Acc:** {format_float(best.get('val_acc'), 2)}%\n"
        f"- **Val Loss:** {format_float(best.get('val_loss'), 4)}\n"
        f"- **Params:** {'' if pd.isna(best.get('params')) else f'{int(best.get('params')):,}'}\n"
        f"- **Epochs:** {'' if pd.isna(best.get('epochs')) else int(best.get('epochs'))}\n"
        f"- **Best Epoch:** {'' if pd.isna(best.get('best_epoch')) else int(best.get('best_epoch'))}\n"
        f"- **Config:** {config_summary(best)}\n"
    )

    # Full Results only (sorted with explicit rules)
    df_sorted = df.copy()
    df_sorted["val_acc"] = pd.to_numeric(df_sorted["val_acc"], errors="coerce")
    df_sorted["params"] = pd.to_numeric(df_sorted["params"], errors="coerce")
    df_sorted["val_loss"] = pd.to_numeric(df_sorted["val_loss"], errors="coerce")
    df_sorted["train_time_sec"] = pd.to_numeric(df_sorted["train_time_sec"], errors="coerce")
    df_sorted = df_sorted.sort_values(
        by=["val_acc", "params", "val_loss", "train_time_sec"],
        ascending=[False, True, True, True],
        kind="mergesort"
    ).reset_index(drop=True)

    display_cols = [
        "exp_name", "use_bn", "dropout_p", "activation", "optimizer", "scheduler",
        "lr", "batch_size", "epochs", "params", "val_acc", "val_loss", "best_epoch", "train_time_sec"
    ]
    full_table = make_table(format_for_display(df_sorted)[display_cols], display_cols)

    results_md = (
        "## Full Results\n\n"
        "_Sorted by **Val Acc (desc)**, then **Params (asc)**, **Val Loss (asc)**, **Train Time (asc)**._\n\n"
        f"{full_table}\n"
    )

    # ---------- NEW: Combined Learning Curves section ----------
    # Generate combined plots from per-epoch CSVs (if present)
    combined = generate_combined_plots(df_sorted)

    #combined_md_parts = ["## Combined Learning Curves (All Experiments)\n"]
    combined_md_parts = ["## Combined Learning Curves (Top 5 Experiments)\n"]
        # List the five runs that were plotted
    picked = combined.get("picked_list", [])
    # if picked:
    #     combined_md_parts.append("**Included runs:** " + ", ".join(f"`{name}`" for name in picked) + "\n")
    if picked:
        combined_md_parts.append("**Included runs (Top-5 by Val Acc → Params → Loss → Time):**\n")

        # Keep order same as in picked_list (so table rows match legend colors)
        df_top5 = df_sorted[df_sorted["exp_name"].isin(picked)].copy()
        df_top5["order"] = pd.Categorical(df_top5["exp_name"], categories=picked, ordered=True)
        df_top5 = df_top5.sort_values("order").drop(columns=["order"])

        display_cols = ["exp_name", "val_acc", "val_loss", "params", "epochs", "best_epoch"]
        df_top5_fmt = df_top5.copy()
        df_top5_fmt["val_acc"] = df_top5_fmt["val_acc"].apply(lambda x: format_float(x, 2))
        df_top5_fmt["val_loss"] = df_top5_fmt["val_loss"].apply(lambda x: format_float(x, 4))
        df_top5_fmt["params"] = df_top5_fmt["params"].apply(lambda x: f"{int(x):,}" if not pd.isna(x) else "")
        df_top5_fmt["epochs"] = df_top5_fmt["epochs"].apply(lambda x: str(int(x)) if not pd.isna(x) else "")
        df_top5_fmt["best_epoch"] = df_top5_fmt["best_epoch"].apply(lambda x: str(int(x)) if not pd.isna(x) else "")

        header = "| " + " | ".join(display_cols) + " |"
        sep = "| " + " | ".join(["---"] * len(display_cols)) + " |"
        lines = [header, sep]
        for _, row in df_top5_fmt.iterrows():
            vals = [str(row[c]) for c in display_cols]
            lines.append("| " + " | ".join(vals) + " |")
        combined_md_parts.append("\n".join(lines) + "\n")
        combined_md_parts.append(
            "_Note: Only Top-5 runs are shown below. "
            "Full combined plots for **all experiments** are saved in "
            "`results/plots/combined_*_all.png`._\n"
        )


    if "val_acc" in combined:
        combined_md_parts.append(md_image_or_note("Validation Accuracy (Top 5 Experiments)", combined["val_acc"]))
    else:
        combined_md_parts.append("_Validation accuracy curves could not be generated (no per-epoch CSVs with a recognized `val_acc`)._\n")

    if "val_loss" in combined:
        combined_md_parts.append(md_image_or_note("Validation Loss (Top 5 Experiments)", combined["val_loss"]))
    else:
        combined_md_parts.append("_Validation loss curves could not be generated (no per-epoch CSVs with a recognized `val_loss`)._\n")

    # If training metrics were also found, include them
    if "train_acc" in combined:
        combined_md_parts.append(md_image_or_note("Training Accuracy (Top 5 Experiments)", combined["train_acc"]))
    if "train_loss" in combined:
        combined_md_parts.append(md_image_or_note("Training Loss (Top 5 Experiments)", combined["train_loss"]))

    combined_md = "\n".join(combined_md_parts)

    # Learning Curves & Diagnostics for ALL experiments (per-run images if present)
    diags = ["## Learning Curves & Diagnostics (Per Experiment)\n"]
    for _, r in df_sorted.iterrows():
        exp = str(r.get("exp_name", ""))
        acc_plot = str(r.get("acc_plot", ""))
        loss_plot = str(r.get("loss_plot", ""))
        cm_plot = str(r.get("cm_plot", ""))
        mis_plot = str(r.get("miscls_plot", ""))
        epoch_csv = str(r.get("epoch_csv", ""))

        block = [f"### `{exp}`\n"]
        if SHOW_PER_RUN_ACCURACY:
            block.append(md_image_or_note("Accuracy", acc_plot))
        if SHOW_PER_RUN_LOSS:
            block.append(md_image_or_note("Loss", loss_plot))
        if SHOW_PER_RUN_CM:
            block.append(md_image_or_note("Confusion Matrix", cm_plot))
        if SHOW_PER_RUN_MISCLS:
            block.append(md_image_or_note("Misclassified Samples", mis_plot))

        if epoch_csv and file_exists(epoch_csv):
            block.append(f"- Per-epoch CSV: `{_to_posix(epoch_csv)}`\n")
        else:
            missing = _to_posix(epoch_csv) if epoch_csv else "(not provided)"
            block.append(f"- Per-epoch CSV: _Missing at '{missing}'_\n")

        diags.append("\n".join(block))

    diagnostics_md = "\n\n".join(diags)

    # diags = ["## Learning Curves & Diagnostics (Per Experiment)\n"]
    # for _, r in df_sorted.iterrows():
    #     exp = str(r.get("exp_name", ""))
    #     acc_plot = str(r.get("acc_plot", ""))
    #     loss_plot = str(r.get("loss_plot", ""))
    #     cm_plot = str(r.get("cm_plot", ""))
    #     mis_plot = str(r.get("miscls_plot", ""))
    #     epoch_csv = str(r.get("epoch_csv", ""))

    #     block = [f"### `{exp}`\n"]
    #     block.append(md_image_or_note("Accuracy", acc_plot))
    #     block.append(md_image_or_note("Loss", loss_plot))
    #     block.append(md_image_or_note("Confusion Matrix", cm_plot))
    #     block.append(md_image_or_note("Misclassified Samples", mis_plot))

    #     if epoch_csv and file_exists(epoch_csv):
    #         block.append(f"- Per-epoch CSV: `{_to_posix(epoch_csv)}`\n")
    #     else:
    #         missing = _to_posix(epoch_csv) if epoch_csv else "(not provided)"
    #         block.append(f"- Per-epoch CSV: _Missing at '{missing}'_\n")

    #     diags.append("\n".join(block))

    # diagnostics_md = "\n\n".join(diags)

    # Use the most common param count for dynamic reporting
    tp = typical_param_count(df)
    tp_str = f"**≈{tp:,} parameters total**" if tp else "**<20k parameters total**"

    # conclusions = (
    #     "## Conclusions\n\n"
    #     "- **BN + Dropout** are critical under a tight parameter budget.\n"
    #     "- **AdamW + StepLR** and **SGD + OneCycleLR** typically converge to strong accuracy within few epochs.\n"
    #     "- **Batch size trade-offs:** 32/64 often edge out 128 in this budget on MNIST.\n"
    #     "- **SiLU/GELU vs ReLU:** Differences are modest on MNIST; small gains are possible.\n"
    #     "- With proper scheduling + light augmentation, **≥ 99.4% within ≤ 20 epochs** is consistently achievable.\n"
    #     f"- The full model stays within {tp_str}, thanks to the **1×1 Conv + GAP** head that replaces a large fully connected layer.\n"
    # )
    conclusions = build_dynamic_conclusions(df_sorted)


    reproduce = (
        "## Reproduce\n\n"
        "Use the same commands as in **Quickstart**.\n\n"
        "```bash\n"
        "pip install -r requirements.txt\n"
        "python train.py --mode grid\n"
        "python update_readme.py\n"
        "```\n"
    )

    # parts = [
    #     intro,
    #     SECTION_DIVIDER,
    #     objective,
    #     SECTION_DIVIDER,
    #     model,
    #     SECTION_DIVIDER,
    #     design,
    #     SECTION_DIVIDER,
    #     best_md,
    #     SECTION_DIVIDER,
    #     results_md,
    #     SECTION_DIVIDER,
    #     combined_md,
    #     SECTION_DIVIDER,
    #     diagnostics_md,
    #     SECTION_DIVIDER,
    #     conclusions,
    #     SECTION_DIVIDER,
    #     reproduce,
    # ]
    parts = [
    intro,
    SECTION_DIVIDER,
    objective,
    SECTION_DIVIDER,
    model,
    SECTION_DIVIDER,
    design,
    SECTION_DIVIDER,
    best_md,
    SECTION_DIVIDER,
    results_md,
    SECTION_DIVIDER,
    combined_md,
    SECTION_DIVIDER,
    conclusions,        # <--- move conclusions up
    SECTION_DIVIDER,
    diagnostics_md,     # diagnostics after conclusions
    SECTION_DIVIDER,
    reproduce,]

    return "\n".join(parts).rstrip() + "\n"


# ---------- Main ----------
def main() -> None:
    if not RESULTS_CSV.exists():
        err(f"ERROR: '{RESULTS_CSV.as_posix()}' not found. Run training first (e.g., 'python train.py').")
        sys.exit(1)

    # Read CSV with the expected header (21 columns)
    try:
        df = pd.read_csv(RESULTS_CSV)
    except Exception as e:
        err(f"ERROR: Failed to read '{RESULTS_CSV.as_posix()}': {e}")
        sys.exit(1)

    # Validate schema length (strict)
    if list(df.columns) != EXPECTED_COLUMNS:
        err("ERROR: results.csv does not match the expected 21-column schema.\n"
            f"Expected columns:\n{EXPECTED_COLUMNS}\n"
            f"Found:\n{list(df.columns)}")
        sys.exit(1)

    if df.empty:
        err(f"ERROR: '{RESULTS_CSV.as_posix()}' has no data rows. Run training first.")
        sys.exit(1)

    readme_text = build_readme(df)

    # Write README (atomic-ish)
    tmp = README_MD.with_suffix(".md.tmp")
    tmp.write_text(readme_text, encoding="utf-8", newline="\n")
    tmp.replace(README_MD)

    print("[update_readme] README.md generated successfully.")


if __name__ == "__main__":
    main()
