# MNIST Experiment Results

Sorted by **validation accuracy** (10k MNIST test set) and then by parameter count.


## Top 10

| exp_name | model_variant | use_bn | dropout_p | activation | optimizer | scheduler | lr | batch_size | epochs | params | val_acc | val_loss | best_epoch | train_time_sec |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| C_bs_sweep_sgd_onecycle_bs32 | TinyMNISTNet | True | 0.05 | relu | sgd | onecycle | 0.05 | 32 | 2 | 15882 | 99.03 | 0.032809 | 2 | 24.21 |
| C_bs_sweep_adam_onecycle_bs32 | TinyMNISTNet | True | 0.05 | relu | adam | onecycle | 0.003 | 32 | 2 | 15882 | 99.02 | 0.032512 | 2 | 24.27 |
| C_bs_sweep_adam_onecycle_bs128 | TinyMNISTNet | True | 0.05 | relu | adam | onecycle | 0.003 | 128 | 2 | 15882 | 98.90 | 0.043326 | 2 | 20.15 |
| C_bs_sweep_adamw_step_bs128 | TinyMNISTNet | True | 0.05 | relu | adamw | step | 0.0025 | 128 | 2 | 15882 | 98.87 | 0.041911 | 2 | 20.04 |
| C_bs_sweep_sgd_onecycle_bs64 | TinyMNISTNet | True | 0.05 | relu | sgd | onecycle | 0.05 | 64 | 2 | 15882 | 98.85 | 0.034786 | 2 | 22.04 |
| C_bs_sweep_adam_onecycle_bs64 | TinyMNISTNet | True | 0.05 | relu | adam | onecycle | 0.003 | 64 | 2 | 15882 | 98.85 | 0.036745 | 2 | 22.06 |
| D_activation_silu | TinyMNISTNet | True | 0.05 | silu | adamw | step | 0.0025 | 128 | 2 | 15882 | 98.72 | 0.046674 | 2 | 19.88 |
| C_bs_sweep_adamw_step_bs32 | TinyMNISTNet | True | 0.05 | relu | adamw | step | 0.0025 | 32 | 2 | 15882 | 98.71 | 0.041743 | 2 | 24.1 |
| C_bs_sweep_adamw_step_bs64 | TinyMNISTNet | True | 0.05 | relu | adamw | step | 0.0025 | 64 | 2 | 15882 | 98.61 | 0.045591 | 2 | 22.05 |
| C_bs_sweep_sgd_onecycle_bs128 | TinyMNISTNet | True | 0.05 | relu | sgd | onecycle | 0.05 | 128 | 2 | 15882 | 98.60 | 0.047694 | 2 | 19.91 |


## Learning Curves & Diagnostics (Top Results)


### 1. C_bs_sweep_sgd_onecycle_bs32

**Accuracy:**

![](results\plots\C_bs_sweep_sgd_onecycle_bs32_acc.png)

**Loss:**

![](results\plots\C_bs_sweep_sgd_onecycle_bs32_loss.png)

**Confusion Matrix:**

![](results\plots\C_bs_sweep_sgd_onecycle_bs32_cm.png)

**Misclassified Samples:**

![](results\plots\C_bs_sweep_sgd_onecycle_bs32_miscls.png)

Epoch CSV: `results\C_bs_sweep_sgd_onecycle_bs32_metrics.csv`


### 2. C_bs_sweep_adam_onecycle_bs32

**Accuracy:**

![](results\plots\C_bs_sweep_adam_onecycle_bs32_acc.png)

**Loss:**

![](results\plots\C_bs_sweep_adam_onecycle_bs32_loss.png)

**Confusion Matrix:**

![](results\plots\C_bs_sweep_adam_onecycle_bs32_cm.png)

**Misclassified Samples:**

![](results\plots\C_bs_sweep_adam_onecycle_bs32_miscls.png)

Epoch CSV: `results\C_bs_sweep_adam_onecycle_bs32_metrics.csv`


### 3. C_bs_sweep_adam_onecycle_bs128

**Accuracy:**

![](results\plots\C_bs_sweep_adam_onecycle_bs128_acc.png)

**Loss:**

![](results\plots\C_bs_sweep_adam_onecycle_bs128_loss.png)

**Confusion Matrix:**

![](results\plots\C_bs_sweep_adam_onecycle_bs128_cm.png)

**Misclassified Samples:**

![](results\plots\C_bs_sweep_adam_onecycle_bs128_miscls.png)

Epoch CSV: `results\C_bs_sweep_adam_onecycle_bs128_metrics.csv`


### 4. C_bs_sweep_adamw_step_bs128

**Accuracy:**

![](results\plots\C_bs_sweep_adamw_step_bs128_acc.png)

**Loss:**

![](results\plots\C_bs_sweep_adamw_step_bs128_loss.png)

**Confusion Matrix:**

![](results\plots\C_bs_sweep_adamw_step_bs128_cm.png)

**Misclassified Samples:**

![](results\plots\C_bs_sweep_adamw_step_bs128_miscls.png)

Epoch CSV: `results\C_bs_sweep_adamw_step_bs128_metrics.csv`


### 5. C_bs_sweep_sgd_onecycle_bs64

**Accuracy:**

![](results\plots\C_bs_sweep_sgd_onecycle_bs64_acc.png)

**Loss:**

![](results\plots\C_bs_sweep_sgd_onecycle_bs64_loss.png)

**Confusion Matrix:**

![](results\plots\C_bs_sweep_sgd_onecycle_bs64_cm.png)

**Misclassified Samples:**

![](results\plots\C_bs_sweep_sgd_onecycle_bs64_miscls.png)

Epoch CSV: `results\C_bs_sweep_sgd_onecycle_bs64_metrics.csv`


### 6. C_bs_sweep_adam_onecycle_bs64

**Accuracy:**

![](results\plots\C_bs_sweep_adam_onecycle_bs64_acc.png)

**Loss:**

![](results\plots\C_bs_sweep_adam_onecycle_bs64_loss.png)

**Confusion Matrix:**

![](results\plots\C_bs_sweep_adam_onecycle_bs64_cm.png)

**Misclassified Samples:**

![](results\plots\C_bs_sweep_adam_onecycle_bs64_miscls.png)

Epoch CSV: `results\C_bs_sweep_adam_onecycle_bs64_metrics.csv`


### 7. D_activation_silu

**Accuracy:**

![](results\plots\D_activation_silu_acc.png)

**Loss:**

![](results\plots\D_activation_silu_loss.png)

**Confusion Matrix:**

![](results\plots\D_activation_silu_cm.png)

**Misclassified Samples:**

![](results\plots\D_activation_silu_miscls.png)

Epoch CSV: `results\D_activation_silu_metrics.csv`


### 8. C_bs_sweep_adamw_step_bs32

**Accuracy:**

![](results\plots\C_bs_sweep_adamw_step_bs32_acc.png)

**Loss:**

![](results\plots\C_bs_sweep_adamw_step_bs32_loss.png)

**Confusion Matrix:**

![](results\plots\C_bs_sweep_adamw_step_bs32_cm.png)

**Misclassified Samples:**

![](results\plots\C_bs_sweep_adamw_step_bs32_miscls.png)

Epoch CSV: `results\C_bs_sweep_adamw_step_bs32_metrics.csv`


### 9. C_bs_sweep_adamw_step_bs64

**Accuracy:**

![](results\plots\C_bs_sweep_adamw_step_bs64_acc.png)

**Loss:**

![](results\plots\C_bs_sweep_adamw_step_bs64_loss.png)

**Confusion Matrix:**

![](results\plots\C_bs_sweep_adamw_step_bs64_cm.png)

**Misclassified Samples:**

![](results\plots\C_bs_sweep_adamw_step_bs64_miscls.png)

Epoch CSV: `results\C_bs_sweep_adamw_step_bs64_metrics.csv`


### 10. C_bs_sweep_sgd_onecycle_bs128

**Accuracy:**

![](results\plots\C_bs_sweep_sgd_onecycle_bs128_acc.png)

**Loss:**

![](results\plots\C_bs_sweep_sgd_onecycle_bs128_loss.png)

**Confusion Matrix:**

![](results\plots\C_bs_sweep_sgd_onecycle_bs128_cm.png)

**Misclassified Samples:**

![](results\plots\C_bs_sweep_sgd_onecycle_bs128_miscls.png)

Epoch CSV: `results\C_bs_sweep_sgd_onecycle_bs128_metrics.csv`



## Full Results

| exp_name | model_variant | use_bn | dropout_p | activation | optimizer | scheduler | lr | batch_size | epochs | params | val_acc | val_loss | best_epoch | train_time_sec |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| C_bs_sweep_sgd_onecycle_bs32 | TinyMNISTNet | True | 0.05 | relu | sgd | onecycle | 0.05 | 32 | 2 | 15882 | 99.03 | 0.032809 | 2 | 24.21 |
| C_bs_sweep_adam_onecycle_bs32 | TinyMNISTNet | True | 0.05 | relu | adam | onecycle | 0.003 | 32 | 2 | 15882 | 99.02 | 0.032512 | 2 | 24.27 |
| C_bs_sweep_adam_onecycle_bs128 | TinyMNISTNet | True | 0.05 | relu | adam | onecycle | 0.003 | 128 | 2 | 15882 | 98.90 | 0.043326 | 2 | 20.15 |
| C_bs_sweep_adamw_step_bs128 | TinyMNISTNet | True | 0.05 | relu | adamw | step | 0.0025 | 128 | 2 | 15882 | 98.87 | 0.041911 | 2 | 20.04 |
| C_bs_sweep_sgd_onecycle_bs64 | TinyMNISTNet | True | 0.05 | relu | sgd | onecycle | 0.05 | 64 | 2 | 15882 | 98.85 | 0.034786 | 2 | 22.04 |
| C_bs_sweep_adam_onecycle_bs64 | TinyMNISTNet | True | 0.05 | relu | adam | onecycle | 0.003 | 64 | 2 | 15882 | 98.85 | 0.036745 | 2 | 22.06 |
| D_activation_silu | TinyMNISTNet | True | 0.05 | silu | adamw | step | 0.0025 | 128 | 2 | 15882 | 98.72 | 0.046674 | 2 | 19.88 |
| C_bs_sweep_adamw_step_bs32 | TinyMNISTNet | True | 0.05 | relu | adamw | step | 0.0025 | 32 | 2 | 15882 | 98.71 | 0.041743 | 2 | 24.1 |
| C_bs_sweep_adamw_step_bs64 | TinyMNISTNet | True | 0.05 | relu | adamw | step | 0.0025 | 64 | 2 | 15882 | 98.61 | 0.045591 | 2 | 22.05 |
| C_bs_sweep_sgd_onecycle_bs128 | TinyMNISTNet | True | 0.05 | relu | sgd | onecycle | 0.05 | 128 | 2 | 15882 | 98.60 | 0.047694 | 2 | 19.91 |
| B_bn_do_sgd_onecycle | TinyMNISTNet | True | 0.05 | relu | sgd | onecycle | 0.05 | 128 | 2 | 15882 | 98.43 | 0.053441 | 2 | 20.14 |
| B_bn_do_adam_onecycle | TinyMNISTNet | True | 0.05 | relu | adam | onecycle | 0.003 | 128 | 2 | 15882 | 98.39 | 0.054544 | 2 | 19.9 |
| B_bn_do_adamw_step | TinyMNISTNet | True | 0.05 | relu | adamw | step | 0.0025 | 128 | 2 | 15882 | 98.06 | 0.05782 | 2 | 19.59 |
| B_bn_do_rmsprop_plateau | TinyMNISTNet | True | 0.05 | relu | rmsprop | plateau | 0.001 | 128 | 2 | 15882 | 97.93 | 0.070333 | 2 | 21.28 |
| D_activation_relu | TinyMNISTNet | True | 0.05 | relu | adamw | step | 0.0025 | 128 | 2 | 15882 | 97.60 | 0.075655 | 2 | 19.89 |
| C_bs_sweep_rmsprop_plateau_bs64 | TinyMNISTNet | True | 0.05 | relu | rmsprop | plateau | 0.001 | 64 | 2 | 15882 | 97.33 | 0.078304 | 2 | 23.77 |
| C_bs_sweep_rmsprop_plateau_bs32 | TinyMNISTNet | True | 0.05 | relu | rmsprop | plateau | 0.001 | 32 | 2 | 15882 | 97.28 | 0.082284 | 1 | 26.49 |
| D_activation_gelu | TinyMNISTNet | True | 0.05 | gelu | adamw | step | 0.0025 | 128 | 2 | 15882 | 96.62 | 0.101842 | 1 | 19.52 |
| C_bs_sweep_rmsprop_plateau_bs128 | TinyMNISTNet | True | 0.05 | relu | rmsprop | plateau | 0.001 | 128 | 2 | 15882 | 94.30 | 0.184738 | 2 | 21.71 |
| A_noBN_noDO_vanillaGD | TinyMNISTNet | False | 0.0 | relu | vanilla | nan | 0.1 | 128 | 2 | 15774 | 16.52 | 2.045735 | 2 | 12.98 |