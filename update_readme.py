#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
update_readme.py

Regenerates README.md for the MNIST experiments repo and normalizes results/results.csv
to the latest schema (21 columns). Handles mixed old/new rows (19 → 21 by padding).

Requirements: pandas (and Python stdlib only)
Platform: Windows-friendly (no GUI ops; file I/O only)
"""

from __future__ import annotations

import sys
import csv
import shutil
from pathlib import Path
from typing import List, Tuple, Optional, Dict

import pandas as pd


# ======== Constants ========

REPO_ROOT = Path(__file__).resolve().parent
RESULTS_DIR = REPO_ROOT / "results"
PLOTS_DIR = RESULTS_DIR / "plots"
RESULTS_CSV = RESULTS_DIR / "results.csv"
README_MD = REPO_ROOT / "README.md"

# Latest schema (21 columns) — exact order required
EXPECTED_COLUMNS: List[str] = [
    "exp_name", "model_variant", "use_bn", "dropout_p", "activation",
    "optimizer", "scheduler", "lr", "weight_decay", "batch_size", "epochs",
    "params", "train_time_sec", "best_epoch", "val_acc", "val_loss",
    "epoch_csv", "acc_plot", "loss_plot", "cm_plot", "miscls_plot"
]

# Older schema (19 columns) — missing the last two plot fields
OLD_19_COLUMNS: List[str] = [
    "exp_name", "model_variant", "use_bn", "dropout_p", "activation",
    "optimizer", "scheduler", "lr", "weight_decay", "batch_size", "epochs",
    "params", "train_time_sec", "best_epoch", "val_acc", "val_loss",
    "epoch_csv", "acc_plot", "loss_plot"
]

SECTION_DIVIDER = "\n---\n"


# ======== Utilities ========

def print_err(msg: str) -> None:
    print(f"[update_readme] {msg}", file=sys.stderr)


def safe_replace(src: Path, dst: Path) -> None:
    """
    Replace file atomically-ish: write to temp, then replace destination.
    """
    tmp = dst.with_suffix(dst.suffix + ".tmp")
    shutil.copy2(src, tmp) if src.exists() else None
    tmp.replace(dst)


def write_text_atomic(path: Path, text: str) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8", newline="\n")
    tmp.replace(path)


def file_exists(rel_path: str) -> bool:
    """
    Check if a path (likely relative from repo root) exists.
    """
    if not rel_path:
        return False
    p = REPO_ROOT / rel_path
    return p.exists()


def load_and_normalize_results(csv_path: Path) -> pd.DataFrame:
    """
    Load results.csv and normalize all rows to EXPECTED_COLUMNS (21).
    Handles:
      - file missing / empty
      - mixed schemas (19 & 21)
      - header/no header scenarios
    Writes back a normalized CSV safely (temp then replace).
    Returns a pandas DataFrame with EXPECTED_COLUMNS, dtypes best-effort coerced.
    """
    if not csv_path.exists():
        print_err(f"ERROR: '{csv_path.as_posix()}' is missing. Run training first (e.g., 'python train.py').")
        sys.exit(1)

    # Read raw file to detect if empty
    raw = csv_path.read_text(encoding="utf-8", errors="ignore")
    if not raw.strip():
        print_err(f"ERROR: '{csv_path.as_posix()}' is empty. Run training first (e.g., 'python train.py').")
        sys.exit(1)

    # Try to read with header, fall back to no header
    try:
        df_try = pd.read_csv(csv_path)
        cols = list(df_try.columns)
        has_header = "exp_name" in cols or set(cols) == set(EXPECTED_COLUMNS) or set(cols) == set(OLD_19_COLUMNS)
    except Exception:
        has_header = False

    if has_header:
        df = pd.read_csv(csv_path)
    else:
        df = pd.read_csv(csv_path, header=None)

    # Normalize columns
    if list(df.columns) == EXPECTED_COLUMNS:
        # Already normalized
        normalized_df = df.copy()
    elif list(df.columns) == OLD_19_COLUMNS:
        # Old header present (19)
        normalized_df = df.copy()
        normalized_df["cm_plot"] = ""
        normalized_df["miscls_plot"] = ""
        normalized_df = normalized_df[EXPECTED_COLUMNS]
    elif not has_header:
        # No header; infer by column count row-wise, then rebuild a normalized DF
        # We will read as raw rows and map each to 21 columns.
        rows = []
        for _, row in df.iterrows():
            values = list(row.dropna().values)  # drop trailing NaNs that pandas inserted
            if len(values) == 21:
                rows.append(values)
            elif len(values) == 19:
                rows.append(values + ["", ""])
            else:
                # If row length unexpected, try to pad/truncate
                if len(values) > 21:
                    rows.append(values[:21])
                else:
                    rows.append(values + [""] * (21 - len(values)))
        normalized_df = pd.DataFrame(rows, columns=EXPECTED_COLUMNS)
    else:
        # Header present but unknown/partial: try to align by count
        current_cols = list(df.columns)
        if len(current_cols) == 21:
            # Just rename to expected, assume order is correct
            df.columns = EXPECTED_COLUMNS
            normalized_df = df.copy()
        elif len(current_cols) == 19:
            df.columns = OLD_19_COLUMNS
            df["cm_plot"] = ""
            df["miscls_plot"] = ""
            normalized_df = df[EXPECTED_COLUMNS].copy()
        else:
            # Try to coerce by position
            values = df.values.tolist()
            fixed_rows = []
            for vals in values:
                vals = [v for v in vals]
                if len(vals) >= 21:
                    fixed_rows.append(vals[:21])
                else:
                    fixed_rows.append(vals + [""] * (21 - len(vals)))
            normalized_df = pd.DataFrame(fixed_rows, columns=EXPECTED_COLUMNS)

    # Coerce dtypes where sensible
    numeric_cols = ["dropout_p", "lr", "weight_decay", "batch_size", "epochs", "params",
                    "train_time_sec", "best_epoch", "val_acc", "val_loss"]
    for c in numeric_cols:
        if c in normalized_df.columns:
            normalized_df[c] = pd.to_numeric(normalized_df[c], errors="coerce")

    # Convert categorical to str (avoid NaN -> 'nan' later by handling when formatting)
    str_cols = ["exp_name", "model_variant", "use_bn", "activation",
                "optimizer", "scheduler", "epoch_csv", "acc_plot", "loss_plot", "cm_plot", "miscls_plot"]
    for c in str_cols:
        if c in normalized_df.columns:
            normalized_df[c] = normalized_df[c].astype(str).fillna("")

    # Write back normalized CSV safely
    normalized_path_tmp = csv_path.with_suffix(".csv.normalized.tmp")
    normalized_df.to_csv(normalized_path_tmp, index=False, quoting=csv.QUOTE_MINIMAL)
    normalized_path_tmp.replace(csv_path)

    return normalized_df


def format_float(val: Optional[float], ndigits: int) -> str:
    if pd.isna(val):
        return ""
    try:
        return f"{float(val):.{ndigits}f}"
    except Exception:
        return ""


def format_time_seconds(val: Optional[float]) -> str:
    if pd.isna(val):
        return ""
    try:
        return f"{float(val):.1f}"
    except Exception:
        return ""


def best_overall(df: pd.DataFrame) -> pd.Series:
    """
    Best overall by highest val_acc; tiebreaker: fewer params, then lower val_loss, then shorter train_time_sec.
    """
    d = df.copy()
    d["val_acc"] = pd.to_numeric(d["val_acc"], errors="coerce")
    d["params"] = pd.to_numeric(d["params"], errors="coerce")
    d["val_loss"] = pd.to_numeric(d["val_loss"], errors="coerce")
    d["train_time_sec"] = pd.to_numeric(d["train_time_sec"], errors="coerce")

    # Sort with desired priority
    d = d.sort_values(
        by=["val_acc", "params", "val_loss", "train_time_sec"],
        ascending=[False, True, True, True],
        kind="mergesort"  # stable
    )
    return d.iloc[0]


def best_by_prefix(df: pd.DataFrame, prefix: str) -> Tuple[int, Optional[pd.Series]]:
    sub = df[df["exp_name"].astype(str).str.startswith(prefix)]
    if sub.empty:
        return 0, None
    row = best_overall(sub)
    return len(sub), row


def typical_param_count(df: pd.DataFrame) -> Optional[int]:
    if "params" not in df.columns:
        return None
    vc = df["params"].dropna().astype(int).value_counts()
    if vc.empty:
        return None
    return int(vc.idxmax())


def row_to_config_summary(r: pd.Series) -> str:
    use_bn = str(r.get("use_bn", "")).strip()
    dropout_p = r.get("dropout_p", None)
    activation = str(r.get("activation", "")).strip()
    optimizer = str(r.get("optimizer", "")).strip()
    scheduler = str(r.get("scheduler", "")).strip()
    lr = r.get("lr", None)
    batch_size = r.get("batch_size", None)
    epochs = r.get("epochs", None)

    # Compose "Optimizer+Scheduler"
    opt_sched = optimizer
    if scheduler:
        opt_sched = f"{optimizer} + {scheduler}"

    parts = []
    if use_bn != "":
        parts.append(f"BN: {use_bn}")
    if not pd.isna(dropout_p):
        parts.append(f"Dropout: {format_float(dropout_p, 3)}")
    if activation:
        parts.append(f"Activation: {activation}")
    if opt_sched:
        parts.append(f"Optimizer+Scheduler: {opt_sched}")
    if not pd.isna(lr):
        parts.append(f"LR: {format_float(lr, 5)}")
    if not pd.isna(batch_size):
        parts.append(f"BatchSize: {int(batch_size)}")
    if not pd.isna(epochs):
        parts.append(f"Epochs: {int(epochs)}")
    return " | ".join(parts)


def markdown_image_or_note(title: str, rel_path: str) -> str:
    """
    Return a Markdown image embed if the file exists, otherwise a small italicized note.
    """
    if rel_path and file_exists(rel_path):
        # Use relative path as-is in README
        return f"**{title}:**\n\n![]({rel_path})\n"
    else:
        missing_path = rel_path if rel_path else "(not provided)"
        return f"**{title}:** _Missing at '{missing_path}'_\n"


def make_table(df: pd.DataFrame, columns: List[str]) -> str:
    """
    Build a GitHub-flavored Markdown table from the dataframe subset.
    """
    # Header
    header = "| " + " | ".join(columns) + " |"
    sep = "| " + " | ".join(["---"] * len(columns)) + " |"
    lines = [header, sep]

    for _, row in df.iterrows():
        cells = []
        for col in columns:
            val = row.get(col, "")
            if pd.isna(val):
                val = ""
            # Ensure strings for markdown
            cells.append(str(val))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def format_results_for_tables(df: pd.DataFrame) -> pd.DataFrame:
    """
    Returns a copy of df with human-formatted numeric columns for printing tables.
    """
    d = df.copy()

    # Formatting columns as specified
    if "val_acc" in d.columns:
        d["val_acc"] = d["val_acc"].apply(lambda x: format_float(x, 2))
    if "val_loss" in d.columns:
        d["val_loss"] = d["val_loss"].apply(lambda x: format_float(x, 4))
    if "train_time_sec" in d.columns:
        d["train_time_sec"] = d["train_time_sec"].apply(format_time_seconds)

    # Ensure integers display cleanly
    for c in ["batch_size", "epochs", "params", "best_epoch"]:
        if c in d.columns:
            d[c] = d[c].apply(lambda x: "" if pd.isna(x) else str(int(x)))

    # Drop excessive precision in dropout and lr for display (keep raw in CSV)
    if "dropout_p" in d.columns:
        d["dropout_p"] = d["dropout_p"].apply(lambda x: "" if pd.isna(x) else format_float(x, 3))
    if "lr" in d.columns:
        d["lr"] = d["lr"].apply(lambda x: "" if pd.isna(x) else format_float(x, 5))

    return d


def build_readme(df: pd.DataFrame) -> str:
    """
    Construct the final README.md content based on the provided DataFrame.
    """

    # ---------- Static Intro (KEEP VERBATIM) ----------
    intro = """# MNIST Experiments (TinyMNISTNet)

This repo runs a compact set of **reproducible MNIST experiments** with a parameter-budget model.
It compares:
1. **No BN/Dropout** + vanilla gradient descent
2. **With BN/Dropout** + multiple optimizers
3. **With BN/Dropout** + batch-size sweep using the optimizers from (2)
4. **Activation variants** (ReLU, SiLU, GELU) with BN/Dropout

Results are logged to `results/results.csv`. Generate the README with tables **and curves** using `update_readme.py`.
Per-experiment epoch CSVs and plots are saved in `results/` and `results/plots/`.

## Folder structure


mnist_experiments/
├─ models/
│ └─ model.py
├─ results/
│ ├─ results.csv
│ └─ plots/
├─ train.py
├─ update_readme.py
├─ requirements.txt
└─ README.md


## Quickstart
```bash
pip install -r requirements.txt
python train.py --mode grid          # run all experiments
python update_readme.py              # generate README.md with tables + plots

Single run examples
python train.py --mode single --use_bn 1 --dropout_p 0.05 --activation relu \
  --optimizer adamw --scheduler step --lr 0.0025 --weight_decay 1e-4 \
  --batch_size 128 --epochs 15 --augment 1
python update_readme.py
```"""

    # ---------- Objective ----------
    objective = (
        "## Objective\n\n"
        "- **Constraints:** `< 20,000` parameters, `≤ 20` epochs, target `≥ 99.4%` validation accuracy "
        "(MNIST 10k test set used as validation; training split is 50k).\n"
    )

    # ---------- Model: TinyMNISTNet ----------
    typical_params = typical_param_count(df)
    typical_params_line = f"- **Typical parameter count (most common across runs):** ~{typical_params:,} params\n" if typical_params else ""

    model = (
        "## Model: TinyMNISTNet\n\n"
        "- Compact CNN using only **3×3 convs**, two **MaxPools** (spatial: `28→14→7`).\n"
        "- A **1×1 conv** + **Global Average Pooling (GAP)** head replaces large fully-connected layers.\n"
        "- **BatchNorm**/**Dropout** optional; activations tried: **ReLU**, **SiLU**, **GELU**.\n"
        f"{typical_params_line}"
        "- **Why GAP?** It eliminates big FC layers, reduces parameters, and improves generalization under tight budgets.\n"
    )

    # ---------- Experiment Design (A/B/C/D) ----------
    # Compute summaries by prefix
    blocks = []
    for prefix, title, desc in [
        ("A_", "A. Baseline", "No BN/Dropout, vanilla SGD (no momentum)"),
        ("B_", "B. BN+Dropout + Optimizers", "SGD+OneCycleLR, AdamW+StepLR, RMSprop+ReduceLROnPlateau, Adam+OneCycleLR"),
        ("C_", "C. BN+Dropout + Batch-size sweep", "Batch sizes: 32, 64, 128 across the optimizers from (B)"),
        ("D_", "D. BN+Dropout + Activation variants", "Activations: ReLU, SiLU, GELU using AdamW + StepLR"),
    ]:
        n, best = best_by_prefix(df, prefix)
        if best is None:
            line = f"**{title}** — {desc}\n\n- Runs: 0 | Best acc: N/A | Best exp: N/A\n"
        else:
            line = (
                f"**{title}** — {desc}\n\n"
                f"- Runs: {n} | Best acc: {format_float(best.get('val_acc'), 2)}% | Best exp: `{best.get('exp_name', '')}`\n"
            )
        blocks.append(line)
    design = "## Experiment Design\n\n" + "\n".join(blocks)

    # ---------- Best Result (So Far) ----------
    best = best_overall(df)
    best_line = (
        "## Best Result (So Far)\n\n"
        f"- **Experiment:** `{best.get('exp_name', '')}`\n"
        f"- **Val Acc:** {format_float(best.get('val_acc'), 2)}%\n"
        f"- **Val Loss:** {format_float(best.get('val_loss'), 4)}\n"
        f"- **Params:** {'' if pd.isna(best.get('params')) else f'{int(best.get('params')):,}'}\n"
        f"- **Epochs:** {'' if pd.isna(best.get('epochs')) else int(best.get('epochs'))}\n"
        f"- **Best Epoch:** {'' if pd.isna(best.get('best_epoch')) else int(best.get('best_epoch'))}\n"
        f"- **Config:** {row_to_config_summary(best)}\n"
    )

    # ---------- Top Results (Top 10) & Full Results ----------
    # Sort master DF for ranking
    df_sorted = df.sort_values(
        by=["val_acc", "params", "val_loss", "train_time_sec"],
        ascending=[False, True, True, True],
        kind="mergesort"
    ).reset_index(drop=True)

    top_k = 10
    df_top = df_sorted.head(top_k).copy()

    # Build display tables
    display_cols = [
        "exp_name", "use_bn", "dropout_p", "activation", "optimizer", "scheduler",
        "lr", "batch_size", "epochs", "params", "val_acc", "val_loss", "best_epoch", "train_time_sec"
    ]
    # Prepare formatted copies
    top_for_table = format_results_for_tables(df_top)[display_cols]
    full_for_table = format_results_for_tables(df_sorted)[display_cols]

    top_table_md = make_table(top_for_table, display_cols)
    full_table_md = make_table(full_for_table, display_cols)

    tables = (
        "## Top Results (Top 10)\n\n"
        f"{top_table_md}\n\n"
        "## Full Results\n\n"
        f"{full_table_md}\n"
    )

    # ---------- Learning Curves & Diagnostics (Top Results) ----------
    diag_blocks = ["## Learning Curves & Diagnostics (Top Results)\n"]
    for _, r in df_top.iterrows():
        exp = str(r.get("exp_name", ""))
        acc_plot = str(r.get("acc_plot", ""))
        loss_plot = str(r.get("loss_plot", ""))
        cm_plot = str(r.get("cm_plot", ""))
        mis_plot = str(r.get("miscls_plot", ""))
        epoch_csv = str(r.get("epoch_csv", ""))

        section = [f"### `{exp}`\n"]
        section.append(markdown_image_or_note("Accuracy", acc_plot))
        section.append(markdown_image_or_note("Loss", loss_plot))
        section.append(markdown_image_or_note("Confusion Matrix", cm_plot))
        section.append(markdown_image_or_note("Misclassified Samples", mis_plot))

        if epoch_csv and file_exists(epoch_csv):
            section.append(f"- Per-epoch CSV: `{epoch_csv}`\n")
        else:
            missing_path = epoch_csv if epoch_csv else "(not provided)"
            section.append(f"- Per-epoch CSV: _Missing at '{missing_path}'_\n")

        diag_blocks.append("\n".join(section))

    diagnostics = "\n\n".join(diag_blocks)

    # ---------- Conclusions ----------
    conclusions = (
        "## Conclusions\n\n"
        "- **BN + Dropout** are critical under a strict parameter budget.\n"
        "- **AdamW + StepLR** and **SGD + OneCycleLR** converge strongly within few epochs.\n"
        "- **Batch size trade-offs:** 32/64 can edge out 128 on accuracy for MNIST in this budget.\n"
        "- **SiLU/GELU vs ReLU:** Differences are small on MNIST; minor gains are possible.\n"
        "- With proper scheduling + light augmentation, **≥ 99.4%** within **≤ 20 epochs** is reliable.\n"
    )

    # ---------- Reproduce ----------
    reproduce = (
        "## Reproduce\n\n"
        "Use the same commands as in **Quickstart**.\n\n"
        "```bash\n"
        "pip install -r requirements.txt\n"
        "python train.py --mode grid\n"
        "python update_readme.py\n"
        "```\n"
    )

    # Assemble full README
    parts = [
        intro,
        SECTION_DIVIDER,
        objective,
        SECTION_DIVIDER,
        model,
        SECTION_DIVIDER,
        design,
        SECTION_DIVIDER,
        best_line,
        SECTION_DIVIDER,
        tables,
        SECTION_DIVIDER,
        diagnostics,
        SECTION_DIVIDER,
        conclusions,
        SECTION_DIVIDER,
        reproduce
    ]
    return "\n".join(parts).rstrip() + "\n"


def main() -> None:
    # 1) Normalize CSV to latest schema
    df = load_and_normalize_results(RESULTS_CSV)

    # Guard against empty DF after normalization (e.g., CSV had headers only)
    if df is None or df.empty:
        print_err(f"ERROR: '{RESULTS_CSV.as_posix()}' has no data rows. Run training first (e.g., 'python train.py').")
        sys.exit(1)

    # 2) Build README content
    readme_text = build_readme(df)

    # 3) Write README.md safely
    write_text_atomic(README_MD, readme_text)

    print("[update_readme] README.md generated successfully.")
    print(f"[update_readme] Normalized CSV saved at: {RESULTS_CSV.as_posix()}")


if __name__ == "__main__":
    main()
