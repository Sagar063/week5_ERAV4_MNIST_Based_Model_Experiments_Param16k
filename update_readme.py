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
  * Model: TinyMNISTNet (architecture block + notes; shows typical total params from CSV).
  * Experiment Design with explanations (LR, OneCycleLR, StepLR, ReduceLROnPlateau;
    how optimizers update weights; activation functions overview).
  * Best Result (So Far).
  * Full Results table ONLY (sorted by val_acc desc, then fewer params, then lower val_loss, then train_time_sec).
    The sorting basis is explicitly stated.
  * Learning Curves & Diagnostics for ALL experiments (not only top runs).
    For each run, it tries to embed Accuracy/Loss/CM/Misclassified images; otherwise prints a small missing note.
    Also links per-epoch CSV if present.

Requirements: pandas (and Python stdlib only)
Windows-friendly: no GUI ops; file I/O only.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import List, Optional

import pandas as pd

# ---------- Paths & Expected Schema ----------
REPO_ROOT = Path(__file__).resolve().parent
RESULTS_DIR = REPO_ROOT / "results"
RESULTS_CSV = RESULTS_DIR / "results.csv"
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
        lines.append("| " + " | ".join(vals) + " |")
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

    # Model section (architecture block + typical params)
    typical_params = typical_param_count(df)
    typical_line = f"- **Typical total parameters (most common across runs):** ~{typical_params:,}\n" if typical_params else ""

    # Estimate last_channels dynamically
    # Rough heuristic: from params count, deduce likely last_channels (default 32)
    # But we can safely assume 32 if CSV doesn’t vary
    last_ch = 32
    try:
        if not df.empty:
            # read model_variant or last conv channels from params if available
            if "model_variant" in df.columns:
                mv = df["model_variant"].dropna().astype(str)
                if len(mv) > 0 and mv.str.contains("last=").any():
                    last_ch = int(mv.str.extract(r"last=(\d+)").dropna().iloc[0,0])
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

    # typical_params = typical_param_count(df)
    # typical_line = f"- **Typical total parameters (most common across runs):** ~{typical_params:,}\n" if typical_params else ""

    # model = (
    #     "## Model: TinyMNISTNet\n\n"
    #     "TinyMNISTNet is a deliberately compact CNN designed for MNIST digits.  \n"
    #     "It enforces three constraints: **<20k parameters**, **≤20 epochs**, and **≥99.4% accuracy**.\n\n"
    #     "---\n\n"
    #     "### Architecture\n\n"
    #     "```text\n"
    #     "Input  : [B, 1, 28, 28]\n\n"
    #     "Conv1  : 1  →  8   (3×3, pad=1)   → [B, 8, 28, 28]\n"
    #     "Conv2  : 8  → 12   (3×3, pad=1)   → [B, 12, 28, 28]\n"
    #     "Pool   : 2×2                         [B, 12, 14, 14]\n\n"
    #     "Conv3  : 12 → 16  (3×3, pad=1)   → [B, 16, 14, 14]\n"
    #     "Conv4  : 16 → 16  (3×3, pad=1)   → [B, 16, 14, 14]\n"
    #     "Pool   : 2×2                         [B, 16,  7,  7]\n\n"
    #     "Conv5  : 16 → 24  (3×3, pad=1)   → [B, 24,  7,  7]\n"
    #     "Conv6  : 24 → 32  (3×3, pad=1)   → [B, 32,  7,  7]\n\n"
    #     "Conv1×1: 32 → 10   (1×1)          → [B, 10,  7,  7]\n"
    #     "GAP    : 7×7 → 1×1                → [B, 10,  1,  1]\n"
    #     "Flatten → [B, 10]\n"
    #     "Softmax → class probabilities\n"
    #     "```\n\n"
    #     "---\n\n"
    #     "### Shape Evolution\n\n"
    #     "- Start: `1×28×28`\n"
    #     "- After Conv/Pool blocks: `32×7×7`\n"
    #     "- 1×1 Conv: `32→10`, output `10×7×7`\n"
    #     "- GAP: average each map → `[10]`\n"
    #     "- Softmax: probabilities over 10 digits\n\n"
    #     "---\n\n"
    #     "### Why 1×1 Conv + GAP?\n\n"
    #     "- Flattening `32×7×7 = 1568` features with a dense layer → ~15k params.\n"
    #     "- Instead: **1×1 conv (32→10)** needs only 320 weights + biases.\n"
    #     "- GAP has no parameters, just averages.\n"
    #     "- Result: <20k params total, less overfitting, faster convergence.\n\n"
    #     "---\n\n"
    #     "### Parameter Count (default last_channels=32)\n\n"
    #     "| Layer       | In→Out Channels | Kernel | Params |\n"
    #     "|-------------|-----------------|--------|--------|\n"
    #     "| Conv1       | 1 → 8           | 3×3    |   80   |\n"
    #     "| Conv2       | 8 → 12          | 3×3    |  876   |\n"
    #     "| Conv3       | 12 → 16         | 3×3    | 1,744  |\n"
    #     "| Conv4       | 16 → 16         | 3×3    | 2,320  |\n"
    #     "| Conv5       | 16 → 24         | 3×3    | 3,480  |\n"
    #     "| Conv6       | 24 → 32         | 3×3    | 6,944  |\n"
    #     "| Conv1×1     | 32 → 10         | 1×1    |   330  |\n"
    #     "| **Total**   |                 |        | **15,774** |\n\n"
    #     f"{typical_line}"
    # )

#     typical_params = typical_param_count(df)
#     typical_line = f"- **Typical total parameters (most common across runs):** ~{typical_params:,}\n" if typical_params else ""
#     # Architecture block (TinyMNISTNet design)
#     architecture_block = """```text
# Input  : 1×28×28

# Conv   : 1 → C1, 3×3, pad=1     (Act)
# Conv   : C1 → C2, 3×3, pad=1    (Act)
# Pool   : 2×2                     (28→14)

# Conv   : C2 → C3, 3×3, pad=1    (Act)
# Conv   : C3 → C4, 3×3, pad=1    (Act)
# Pool   : 2×2                     (14→7)

# Conv1×1: C4 → 10
# GAP    : 7×7 → 1×1
# Softmax: 10
# ```"""

#     model = (
#         "## Model: TinyMNISTNet\n\n"
#         "- Compact CNN using only **3×3 convs**, two **MaxPools** (spatial: `28→14→7`).\n"
#         "- A **1×1 conv** + **Global Average Pooling (GAP)** head replaces large fully-connected layers.\n"
#         "- **BatchNorm**/**Dropout** optional; activations tried: **ReLU**, **SiLU**, **GELU**.\n"
#         f"{typical_line}"
#         "- **Why GAP?** It eliminates big FC layers, reduces parameters, and improves generalization under tight budgets.\n\n"
#         "### Architecture\n\n"
#         f"{architecture_block}\n"
#     )

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

    # Learning Curves & Diagnostics for ALL experiments
    diags = ["## Learning Curves & Diagnostics (All Experiments)\n"]
    for _, r in df_sorted.iterrows():
        exp = str(r.get("exp_name", ""))
        acc_plot = str(r.get("acc_plot", ""))
        loss_plot = str(r.get("loss_plot", ""))
        cm_plot = str(r.get("cm_plot", ""))
        mis_plot = str(r.get("miscls_plot", ""))
        epoch_csv = str(r.get("epoch_csv", ""))

        block = [f"### `{exp}`\n"]
        block.append(md_image_or_note("Accuracy", acc_plot))
        block.append(md_image_or_note("Loss", loss_plot))
        block.append(md_image_or_note("Confusion Matrix", cm_plot))
        block.append(md_image_or_note("Misclassified Samples", mis_plot))

        if epoch_csv and file_exists(epoch_csv):
            block.append(f"- Per-epoch CSV: `{_to_posix(epoch_csv)}`\n")
        else:
            missing = _to_posix(epoch_csv) if epoch_csv else "(not provided)"
            block.append(f"- Per-epoch CSV: _Missing at '{missing}'_\n")

        diags.append("\n".join(block))

    diagnostics_md = "\n\n".join(diags)


    # Use the most common param count for dynamic reporting
    tp = typical_param_count(df)
    tp_str = f"**≈{tp:,} parameters total**" if tp else "**<20k parameters total**"

    conclusions = (
        "## Conclusions\n\n"
        "- **BN + Dropout** are critical under a tight parameter budget.\n"
        "- **AdamW + StepLR** and **SGD + OneCycleLR** typically converge to strong accuracy within few epochs.\n"
        "- **Batch size trade-offs:** 32/64 often edge out 128 in this budget on MNIST.\n"
        "- **SiLU/GELU vs ReLU:** Differences are modest on MNIST; small gains are possible.\n"
        "- With proper scheduling + light augmentation, **≥ 99.4% within ≤ 20 epochs** is consistently achievable.\n"
        f"- The full model stays within {tp_str}, thanks to the **1×1 Conv + GAP** head that replaces a large fully connected layer.\n"
    )



    # conclusions = (
    #     "## Conclusions\n\n"
    #     "- **BN + Dropout** are critical under a tight parameter budget.\n"
    #     "- **AdamW + StepLR** and **SGD + OneCycleLR** typically converge to strong accuracy within few epochs.\n"
    #     "- **Batch size trade-offs:** 32/64 often edge out 128 in this budget on MNIST.\n"
    #     "- **SiLU/GELU vs ReLU:** Differences are modest on MNIST; small gains are possible.\n"
    #     "- With proper scheduling + light augmentation, **≥ 99.4% within ≤ 20 epochs** is consistently achievable.\n"
    # )

    reproduce = (
        "## Reproduce\n\n"
        "Use the same commands as in **Quickstart**.\n\n"
        "```bash\n"
        "pip install -r requirements.txt\n"
        "python train.py --mode grid\n"
        "python update_readme.py\n"
        "```\n"
    )

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
        diagnostics_md,
        SECTION_DIVIDER,
        conclusions,
        SECTION_DIVIDER,
        reproduce,
    ]
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
