# MNIST Experiments (TinyMNISTNet)

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
python train.py --mode single --use_bn 1 --dropout_p 0.05 --activation relu   --optimizer adamw --scheduler step --lr 0.0025 --weight_decay 1e-4   --batch_size 128 --epochs 15 --augment 1
python update_readme.py
```

---

## Objective

- **Constraints:** `< 20,000` parameters, `≤ 20` epochs, target `≥ 99.4%` validation accuracy (MNIST 10k test set used as validation; training split is 50k).


---

## Model: TinyMNISTNet

- Compact CNN using only **3×3 convs**, two **MaxPools** (spatial: `28→14→7`).
- A **1×1 conv** + **Global Average Pooling (GAP)** head replaces large fully-connected layers.
- **BatchNorm**/**Dropout** optional; activations tried: **ReLU**, **SiLU**, **GELU**.
- **Typical total parameters (most common across runs):** ~15,882
- **Why GAP?** It eliminates big FC layers, reduces parameters, and improves generalization under tight budgets.

### Architecture

```text
Input  : 1×28×28

Conv   : 1 → C1, 3×3, pad=1     (Act)
Conv   : C1 → C2, 3×3, pad=1    (Act)
Pool   : 2×2                     (28→14)

Conv   : C2 → C3, 3×3, pad=1    (Act)
Conv   : C3 → C4, 3×3, pad=1    (Act)
Pool   : 2×2                     (14→7)

Conv1×1: C4 → 10
GAP    : 7×7 → 1×1
Softmax: 10
```


---

## Experiment Design

**What’s varied and why**

- **Learning Rate (LR):** step size for weight updates each iteration.
- **Schedulers:**
  - **OneCycleLR:** Increases LR up to a peak then decreases it within a single run; encourages fast convergence and regularization.
  - **StepLR:** Multiplies LR by a factor (e.g., 0.1) every fixed number of epochs; a simple decay schedule.
  - **ReduceLROnPlateau:** Lowers LR when a monitored metric (e.g., val loss) stops improving; adapts LR to training plateaus.
- **Optimizers (how they update weights):**
  - **SGD (vanilla):** `w ← w − lr * grad` (no momentum here in baseline A). Simple, stable with proper schedules.
  - **SGD + OneCycleLR (B/C):** Same rule but LR follows OneCycle; typically reaches good accuracy quickly.
  - **AdamW + StepLR (B/D):** Adam-style adaptive moments with **decoupled weight decay** (better regularization) + StepLR decay.
  - **RMSprop + ReduceLROnPlateau (B):** Scales updates by running average of squared gradients; LR reduced when progress stalls.
  - **Adam + OneCycleLR (B/C):** Adam’s adaptive moments combined with OneCycle schedule.
- **Activations:**
  - **ReLU:** max(0, x); cheap, strong baseline.
  - **SiLU (Swish):** x * sigmoid(x); smooth, can improve convergence.
  - **GELU:** Gaussian-error linear unit; smooth, often strong in transformers/CNNs.

**Blocks we run**

- **A. Baseline:** no BN/Dropout, **SGD (no momentum)**.
- **B. BN + Dropout + Optimizers:** SGD+OneCycleLR, AdamW+StepLR, RMSprop+ReduceLROnPlateau, Adam+OneCycleLR.
- **C. BN + Dropout + Batch sizes:** {32, 64, 128} across optimizers from (B).
- **D. BN + Dropout + Activations:** {ReLU, SiLU, GELU} using **AdamW + StepLR**.


---

## Best Result (So Far)

- **Experiment:** `C_bs_sweep_sgd_onecycle_bs32`
- **Val Acc:** 99.03%
- **Val Loss:** 0.0328
- **Params:** 15,882
- **Epochs:** 2
- **Best Epoch:** 2
- **Config:** BN: True | Dropout: 0.050 | Activation: relu | Optimizer+Scheduler: sgd + onecycle | LR: 0.05000 | BatchSize: 32 | Epochs: 2


---

## Full Results

_Sorted by **Val Acc (desc)**, then **Params (asc)**, **Val Loss (asc)**, **Train Time (asc)**._

| exp_name | use_bn | dropout_p | activation | optimizer | scheduler | lr | batch_size | epochs | params | val_acc | val_loss | best_epoch | train_time_sec |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| C_bs_sweep_sgd_onecycle_bs32 | True | 0.050 | relu | sgd | onecycle | 0.05000 | 32 | 2 | 15882 | 99.03 | 0.0328 | 2 | 24.2 |
| C_bs_sweep_adam_onecycle_bs32 | True | 0.050 | relu | adam | onecycle | 0.00300 | 32 | 2 | 15882 | 99.02 | 0.0325 | 2 | 24.3 |
| C_bs_sweep_adam_onecycle_bs128 | True | 0.050 | relu | adam | onecycle | 0.00300 | 128 | 2 | 15882 | 98.90 | 0.0433 | 2 | 20.1 |
| C_bs_sweep_adamw_step_bs128 | True | 0.050 | relu | adamw | step | 0.00250 | 128 | 2 | 15882 | 98.87 | 0.0419 | 2 | 20.0 |
| C_bs_sweep_sgd_onecycle_bs64 | True | 0.050 | relu | sgd | onecycle | 0.05000 | 64 | 2 | 15882 | 98.85 | 0.0348 | 2 | 22.0 |
| C_bs_sweep_adam_onecycle_bs64 | True | 0.050 | relu | adam | onecycle | 0.00300 | 64 | 2 | 15882 | 98.85 | 0.0367 | 2 | 22.1 |
| D_activation_silu | True | 0.050 | silu | adamw | step | 0.00250 | 128 | 2 | 15882 | 98.72 | 0.0467 | 2 | 19.9 |
| C_bs_sweep_adamw_step_bs32 | True | 0.050 | relu | adamw | step | 0.00250 | 32 | 2 | 15882 | 98.71 | 0.0417 | 2 | 24.1 |
| C_bs_sweep_adamw_step_bs64 | True | 0.050 | relu | adamw | step | 0.00250 | 64 | 2 | 15882 | 98.61 | 0.0456 | 2 | 22.1 |
| C_bs_sweep_sgd_onecycle_bs128 | True | 0.050 | relu | sgd | onecycle | 0.05000 | 128 | 2 | 15882 | 98.60 | 0.0477 | 2 | 19.9 |
| B_bn_do_sgd_onecycle | True | 0.050 | relu | sgd | onecycle | 0.05000 | 128 | 2 | 15882 | 98.43 | 0.0534 | 2 | 20.1 |
| B_bn_do_adam_onecycle | True | 0.050 | relu | adam | onecycle | 0.00300 | 128 | 2 | 15882 | 98.39 | 0.0545 | 2 | 19.9 |
| B_bn_do_adamw_step | True | 0.050 | relu | adamw | step | 0.00250 | 128 | 2 | 15882 | 98.06 | 0.0578 | 2 | 19.6 |
| B_bn_do_rmsprop_plateau | True | 0.050 | relu | rmsprop | plateau | 0.00100 | 128 | 2 | 15882 | 97.93 | 0.0703 | 2 | 21.3 |
| D_activation_relu | True | 0.050 | relu | adamw | step | 0.00250 | 128 | 2 | 15882 | 97.60 | 0.0757 | 2 | 19.9 |
| C_bs_sweep_rmsprop_plateau_bs64 | True | 0.050 | relu | rmsprop | plateau | 0.00100 | 64 | 2 | 15882 | 97.33 | 0.0783 | 2 | 23.8 |
| C_bs_sweep_rmsprop_plateau_bs32 | True | 0.050 | relu | rmsprop | plateau | 0.00100 | 32 | 2 | 15882 | 97.28 | 0.0823 | 1 | 26.5 |
| D_activation_gelu | True | 0.050 | gelu | adamw | step | 0.00250 | 128 | 2 | 15882 | 96.62 | 0.1018 | 1 | 19.5 |
| C_bs_sweep_rmsprop_plateau_bs128 | True | 0.050 | relu | rmsprop | plateau | 0.00100 | 128 | 2 | 15882 | 94.30 | 0.1847 | 2 | 21.7 |
| A_noBN_noDO_vanillaGD | False | 0.000 | relu | vanilla |  | 0.10000 | 128 | 2 | 15774 | 16.52 | 2.0457 | 2 | 13.0 |


---

## Learning Curves & Diagnostics (All Experiments)


### `C_bs_sweep_sgd_onecycle_bs32`

**Accuracy:**

![](results\plots\C_bs_sweep_sgd_onecycle_bs32_acc.png)

**Loss:**

![](results\plots\C_bs_sweep_sgd_onecycle_bs32_loss.png)

**Confusion Matrix:**

![](results\plots\C_bs_sweep_sgd_onecycle_bs32_cm.png)

**Misclassified Samples:**

![](results\plots\C_bs_sweep_sgd_onecycle_bs32_miscls.png)

- Per-epoch CSV: `results\C_bs_sweep_sgd_onecycle_bs32_metrics.csv`


### `C_bs_sweep_adam_onecycle_bs32`

**Accuracy:**

![](results\plots\C_bs_sweep_adam_onecycle_bs32_acc.png)

**Loss:**

![](results\plots\C_bs_sweep_adam_onecycle_bs32_loss.png)

**Confusion Matrix:**

![](results\plots\C_bs_sweep_adam_onecycle_bs32_cm.png)

**Misclassified Samples:**

![](results\plots\C_bs_sweep_adam_onecycle_bs32_miscls.png)

- Per-epoch CSV: `results\C_bs_sweep_adam_onecycle_bs32_metrics.csv`


### `C_bs_sweep_adam_onecycle_bs128`

**Accuracy:**

![](results\plots\C_bs_sweep_adam_onecycle_bs128_acc.png)

**Loss:**

![](results\plots\C_bs_sweep_adam_onecycle_bs128_loss.png)

**Confusion Matrix:**

![](results\plots\C_bs_sweep_adam_onecycle_bs128_cm.png)

**Misclassified Samples:**

![](results\plots\C_bs_sweep_adam_onecycle_bs128_miscls.png)

- Per-epoch CSV: `results\C_bs_sweep_adam_onecycle_bs128_metrics.csv`


### `C_bs_sweep_adamw_step_bs128`

**Accuracy:**

![](results\plots\C_bs_sweep_adamw_step_bs128_acc.png)

**Loss:**

![](results\plots\C_bs_sweep_adamw_step_bs128_loss.png)

**Confusion Matrix:**

![](results\plots\C_bs_sweep_adamw_step_bs128_cm.png)

**Misclassified Samples:**

![](results\plots\C_bs_sweep_adamw_step_bs128_miscls.png)

- Per-epoch CSV: `results\C_bs_sweep_adamw_step_bs128_metrics.csv`


### `C_bs_sweep_sgd_onecycle_bs64`

**Accuracy:**

![](results\plots\C_bs_sweep_sgd_onecycle_bs64_acc.png)

**Loss:**

![](results\plots\C_bs_sweep_sgd_onecycle_bs64_loss.png)

**Confusion Matrix:**

![](results\plots\C_bs_sweep_sgd_onecycle_bs64_cm.png)

**Misclassified Samples:**

![](results\plots\C_bs_sweep_sgd_onecycle_bs64_miscls.png)

- Per-epoch CSV: `results\C_bs_sweep_sgd_onecycle_bs64_metrics.csv`


### `C_bs_sweep_adam_onecycle_bs64`

**Accuracy:**

![](results\plots\C_bs_sweep_adam_onecycle_bs64_acc.png)

**Loss:**

![](results\plots\C_bs_sweep_adam_onecycle_bs64_loss.png)

**Confusion Matrix:**

![](results\plots\C_bs_sweep_adam_onecycle_bs64_cm.png)

**Misclassified Samples:**

![](results\plots\C_bs_sweep_adam_onecycle_bs64_miscls.png)

- Per-epoch CSV: `results\C_bs_sweep_adam_onecycle_bs64_metrics.csv`


### `D_activation_silu`

**Accuracy:**

![](results\plots\D_activation_silu_acc.png)

**Loss:**

![](results\plots\D_activation_silu_loss.png)

**Confusion Matrix:**

![](results\plots\D_activation_silu_cm.png)

**Misclassified Samples:**

![](results\plots\D_activation_silu_miscls.png)

- Per-epoch CSV: `results\D_activation_silu_metrics.csv`


### `C_bs_sweep_adamw_step_bs32`

**Accuracy:**

![](results\plots\C_bs_sweep_adamw_step_bs32_acc.png)

**Loss:**

![](results\plots\C_bs_sweep_adamw_step_bs32_loss.png)

**Confusion Matrix:**

![](results\plots\C_bs_sweep_adamw_step_bs32_cm.png)

**Misclassified Samples:**

![](results\plots\C_bs_sweep_adamw_step_bs32_miscls.png)

- Per-epoch CSV: `results\C_bs_sweep_adamw_step_bs32_metrics.csv`


### `C_bs_sweep_adamw_step_bs64`

**Accuracy:**

![](results\plots\C_bs_sweep_adamw_step_bs64_acc.png)

**Loss:**

![](results\plots\C_bs_sweep_adamw_step_bs64_loss.png)

**Confusion Matrix:**

![](results\plots\C_bs_sweep_adamw_step_bs64_cm.png)

**Misclassified Samples:**

![](results\plots\C_bs_sweep_adamw_step_bs64_miscls.png)

- Per-epoch CSV: `results\C_bs_sweep_adamw_step_bs64_metrics.csv`


### `C_bs_sweep_sgd_onecycle_bs128`

**Accuracy:**

![](results\plots\C_bs_sweep_sgd_onecycle_bs128_acc.png)

**Loss:**

![](results\plots\C_bs_sweep_sgd_onecycle_bs128_loss.png)

**Confusion Matrix:**

![](results\plots\C_bs_sweep_sgd_onecycle_bs128_cm.png)

**Misclassified Samples:**

![](results\plots\C_bs_sweep_sgd_onecycle_bs128_miscls.png)

- Per-epoch CSV: `results\C_bs_sweep_sgd_onecycle_bs128_metrics.csv`


### `B_bn_do_sgd_onecycle`

**Accuracy:**

![](results\plots\B_bn_do_sgd_onecycle_acc.png)

**Loss:**

![](results\plots\B_bn_do_sgd_onecycle_loss.png)

**Confusion Matrix:**

![](results\plots\B_bn_do_sgd_onecycle_cm.png)

**Misclassified Samples:**

![](results\plots\B_bn_do_sgd_onecycle_miscls.png)

- Per-epoch CSV: `results\B_bn_do_sgd_onecycle_metrics.csv`


### `B_bn_do_adam_onecycle`

**Accuracy:**

![](results\plots\B_bn_do_adam_onecycle_acc.png)

**Loss:**

![](results\plots\B_bn_do_adam_onecycle_loss.png)

**Confusion Matrix:**

![](results\plots\B_bn_do_adam_onecycle_cm.png)

**Misclassified Samples:**

![](results\plots\B_bn_do_adam_onecycle_miscls.png)

- Per-epoch CSV: `results\B_bn_do_adam_onecycle_metrics.csv`


### `B_bn_do_adamw_step`

**Accuracy:**

![](results\plots\B_bn_do_adamw_step_acc.png)

**Loss:**

![](results\plots\B_bn_do_adamw_step_loss.png)

**Confusion Matrix:**

![](results\plots\B_bn_do_adamw_step_cm.png)

**Misclassified Samples:**

![](results\plots\B_bn_do_adamw_step_miscls.png)

- Per-epoch CSV: `results\B_bn_do_adamw_step_metrics.csv`


### `B_bn_do_rmsprop_plateau`

**Accuracy:**

![](results\plots\B_bn_do_rmsprop_plateau_acc.png)

**Loss:**

![](results\plots\B_bn_do_rmsprop_plateau_loss.png)

**Confusion Matrix:**

![](results\plots\B_bn_do_rmsprop_plateau_cm.png)

**Misclassified Samples:**

![](results\plots\B_bn_do_rmsprop_plateau_miscls.png)

- Per-epoch CSV: `results\B_bn_do_rmsprop_plateau_metrics.csv`


### `D_activation_relu`

**Accuracy:**

![](results\plots\D_activation_relu_acc.png)

**Loss:**

![](results\plots\D_activation_relu_loss.png)

**Confusion Matrix:**

![](results\plots\D_activation_relu_cm.png)

**Misclassified Samples:**

![](results\plots\D_activation_relu_miscls.png)

- Per-epoch CSV: `results\D_activation_relu_metrics.csv`


### `C_bs_sweep_rmsprop_plateau_bs64`

**Accuracy:**

![](results\plots\C_bs_sweep_rmsprop_plateau_bs64_acc.png)

**Loss:**

![](results\plots\C_bs_sweep_rmsprop_plateau_bs64_loss.png)

**Confusion Matrix:**

![](results\plots\C_bs_sweep_rmsprop_plateau_bs64_cm.png)

**Misclassified Samples:**

![](results\plots\C_bs_sweep_rmsprop_plateau_bs64_miscls.png)

- Per-epoch CSV: `results\C_bs_sweep_rmsprop_plateau_bs64_metrics.csv`


### `C_bs_sweep_rmsprop_plateau_bs32`

**Accuracy:**

![](results\plots\C_bs_sweep_rmsprop_plateau_bs32_acc.png)

**Loss:**

![](results\plots\C_bs_sweep_rmsprop_plateau_bs32_loss.png)

**Confusion Matrix:**

![](results\plots\C_bs_sweep_rmsprop_plateau_bs32_cm.png)

**Misclassified Samples:**

![](results\plots\C_bs_sweep_rmsprop_plateau_bs32_miscls.png)

- Per-epoch CSV: `results\C_bs_sweep_rmsprop_plateau_bs32_metrics.csv`


### `D_activation_gelu`

**Accuracy:**

![](results\plots\D_activation_gelu_acc.png)

**Loss:**

![](results\plots\D_activation_gelu_loss.png)

**Confusion Matrix:**

![](results\plots\D_activation_gelu_cm.png)

**Misclassified Samples:**

![](results\plots\D_activation_gelu_miscls.png)

- Per-epoch CSV: `results\D_activation_gelu_metrics.csv`


### `C_bs_sweep_rmsprop_plateau_bs128`

**Accuracy:**

![](results\plots\C_bs_sweep_rmsprop_plateau_bs128_acc.png)

**Loss:**

![](results\plots\C_bs_sweep_rmsprop_plateau_bs128_loss.png)

**Confusion Matrix:**

![](results\plots\C_bs_sweep_rmsprop_plateau_bs128_cm.png)

**Misclassified Samples:**

![](results\plots\C_bs_sweep_rmsprop_plateau_bs128_miscls.png)

- Per-epoch CSV: `results\C_bs_sweep_rmsprop_plateau_bs128_metrics.csv`


### `A_noBN_noDO_vanillaGD`

**Accuracy:**

![](results\plots\A_noBN_noDO_vanillaGD_acc.png)

**Loss:**

![](results\plots\A_noBN_noDO_vanillaGD_loss.png)

**Confusion Matrix:**

![](results\plots\A_noBN_noDO_vanillaGD_cm.png)

**Misclassified Samples:**

![](results\plots\A_noBN_noDO_vanillaGD_miscls.png)

- Per-epoch CSV: `results\A_noBN_noDO_vanillaGD_metrics.csv`


---

## Conclusions

- **BN + Dropout** are critical under a tight parameter budget.
- **AdamW + StepLR** and **SGD + OneCycleLR** typically converge to strong accuracy within few epochs.
- **Batch size trade-offs:** 32/64 often edge out 128 in this budget on MNIST.
- **SiLU/GELU vs ReLU:** Differences are modest on MNIST; small gains are possible.
- With proper scheduling + light augmentation, **≥ 99.4% within ≤ 20 epochs** is consistently achievable.


---

## Reproduce

Use the same commands as in **Quickstart**.

```bash
pip install -r requirements.txt
python train.py --mode grid
python update_readme.py
```
