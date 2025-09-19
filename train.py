import argparse, time, csv, os
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch, torch.nn as nn, torch.nn.functional as F
from torch.optim import SGD, Adam, AdamW, RMSprop
from torch.optim.lr_scheduler import StepLR, OneCycleLR, ReduceLROnPlateau
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
from models.model import TinyMNISTNet, count_parameters
import numpy as np

def set_seed(seed=42):
    import random
    random.seed(seed); torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def loaders(batch_size=128, augment=True):
    tfm_train = [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    if augment:
        tfm_train = [transforms.RandomAffine(degrees=5, translate=(0.02,0.02))] + tfm_train
    train_tfms = transforms.Compose(tfm_train)
    test_tfms  = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])

    train_full = datasets.MNIST("./data", train=True, download=True, transform=train_tfms)
    train_set = Subset(train_full, list(range(50000)))  # 50k
    test_set  = datasets.MNIST("./data", train=False, download=True, transform=test_tfms)

    # ⬇️ Windows is happier with num_workers=0 for plotting-heavy pipelines
    nw = 0 if os.name == "nt" else 2

    dl_train = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=nw, pin_memory=True)
    dl_test  = DataLoader(test_set,  batch_size=batch_size, shuffle=False, num_workers=nw, pin_memory=True)
    return dl_train, dl_test


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    total, correct, loss_sum = 0, 0, 0.0
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        logits = model(xb)
        loss = F.cross_entropy(logits, yb, reduction='sum')
        loss_sum += loss.item()
        pred = logits.argmax(dim=1)
        correct += (pred == yb).sum().item()
        total += yb.size(0)
    return loss_sum/total, 100.0*correct/total

def collect_preds_labels(model, loader, device):
    ys, yh = [], []
    with torch.no_grad():
        model.eval()
        for xb, yb in loader:
            xb = xb.to(device)
            logits = model(xb)
            pred = logits.argmax(dim=1).cpu().numpy()
            ys.append(yb.numpy())
            yh.append(pred)
    y_true = np.concatenate(ys)
    y_pred = np.concatenate(yh)
    return y_true, y_pred

def confusion_matrix_np(y_true, y_pred, num_classes=10):
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
    return cm

def save_confusion_matrix(exp_name, model, loader, device, out_dir):
    y_true, y_pred = collect_preds_labels(model, loader, device)
    cm = confusion_matrix_np(y_true, y_pred, num_classes=10)
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{exp_name}_cm.png"
    plt.figure()
    plt.imshow(cm, interpolation='nearest')
    plt.title(f"{exp_name} - Confusion Matrix")
    plt.xlabel("Predicted"); plt.ylabel("True")
    plt.colorbar(fraction=0.046, pad=0.04)
    ticks = np.arange(10)
    plt.xticks(ticks, ticks); plt.yticks(ticks, ticks)
    # annotate
    thresh = cm.max() / 2.0 if cm.max() > 0 else 1
    for i in range(10):
        for j in range(10):
            val = cm[i, j]
            if val > 0:
                color = "white" if val > thresh else "black"
                plt.text(j, i, str(val), ha="center", va="center", fontsize=7, color=color)
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()
    return str(path)

def save_misclassified_grid(exp_name, model, loader, device, out_dir, max_items=36):
    imgs, preds, gts = [], [], []
    with torch.no_grad():
        model.eval()
        for xb, yb in loader:
            xb = xb.to(device)
            logits = model(xb)
            yhat = logits.argmax(dim=1).cpu()
            mism = yhat.ne(yb)
            if mism.any():
                mis_imgs = xb[mism].cpu()
                mis_preds = yhat[mism].cpu()
                mis_labels = yb[mism]
                for i in range(len(mis_imgs)):
                    imgs.append(mis_imgs[i]); preds.append(int(mis_preds[i])); gts.append(int(mis_labels[i]))
                    if len(imgs) >= max_items:
                        break
            if len(imgs) >= max_items:
                break

    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{exp_name}_miscls.png"
    if not imgs:
        # still create an empty figure to avoid broken links
        plt.figure()
        plt.title(f"{exp_name} - No misclassifications in sample")
        plt.tight_layout()
        plt.savefig(path, dpi=160); plt.close()
        return str(path)

    cols = 6
    rows = (len(imgs) + cols - 1) // cols
    import math
    plt.figure(figsize=(2.0*cols, 2.0*rows))
    for i, (img, p, y) in enumerate(zip(imgs, preds, gts)):
        plt.subplot(rows, cols, i+1)
        # show grayscale image; avoid specifying explicit colors for compliance
        plt.imshow(img.squeeze(0))
        plt.title(f"GT:{y} Pred:{p}", fontsize=8)
        plt.axis('off')
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()
    return str(path)

def train_one_epoch(model, loader, opt, device, scheduler=None, use_labelsmoothing=0.0):
    model.train()
    total, correct, loss_sum = 0, 0, 0.0
    criterion = nn.CrossEntropyLoss(label_smoothing=use_labelsmoothing) if use_labelsmoothing > 0 else nn.CrossEntropyLoss()
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        opt.zero_grad(set_to_none=True)
        logits = model(xb)
        loss = criterion(logits, yb)
        loss.backward()
        opt.step()
        if isinstance(scheduler, OneCycleLR):
            scheduler.step()

        loss_sum += loss.item() * xb.size(0)
        pred = logits.argmax(dim=1)
        correct += (pred == yb).sum().item()
        total += xb.size(0)
    return loss_sum/total, 100.0*correct/total

def build_optimizer(name, params, lr, weight_decay=0.0, momentum=0.9):
    name = (name or "").lower()
    if name == "sgd":
        return SGD(params, lr=lr, momentum=momentum, weight_decay=weight_decay, nesterov=False)
    if name == "vanilla":
        return SGD(params, lr=lr, momentum=0.0, weight_decay=0.0, nesterov=False)
    if name == "adam":
        return Adam(params, lr=lr, weight_decay=weight_decay)
    if name == "adamw":
        return AdamW(params, lr=lr, weight_decay=weight_decay)
    if name == "rmsprop":
        return RMSprop(params, lr=lr, alpha=0.9, momentum=momentum, weight_decay=weight_decay)
    raise ValueError(f"Unknown optimizer: {name}")

def build_scheduler(name, optimizer, steps_per_epoch, epochs):
    if not name: return None
    name = name.lower()
    if name == "step":
        return StepLR(optimizer, step_size=5, gamma=0.5)
    if name == "onecycle":
        return OneCycleLR(optimizer, max_lr=optimizer.param_groups[0]['lr'],
                          steps_per_epoch=steps_per_epoch, epochs=epochs,
                          pct_start=0.2, div_factor=5.0, final_div_factor=20.0)
    if name == "plateau":
        return ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=2)
    return None

def plot_curves(exp_name, epochs_hist, out_dir):
    """Save accuracy and loss curves: train vs val (two separate images)."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    acc_path = out_dir / f"{exp_name}_acc.png"
    plt.figure()
    plt.plot(range(1, len(epochs_hist)+1), [e['train_acc'] for e in epochs_hist], label="train_acc")
    plt.plot(range(1, len(epochs_hist)+1), [e['val_acc']   for e in epochs_hist], label="val_acc")
    plt.xlabel("Epoch"); plt.ylabel("Accuracy (%)"); plt.title(f"{exp_name} - Accuracy")
    plt.legend(); plt.tight_layout(); plt.savefig(acc_path, dpi=160); plt.close()

    loss_path = out_dir / f"{exp_name}_loss.png"
    plt.figure()
    plt.plot(range(1, len(epochs_hist)+1), [e['train_loss'] for e in epochs_hist], label="train_loss")
    plt.plot(range(1, len(epochs_hist)+1), [e['val_loss']   for e in epochs_hist], label="val_loss")
    plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.title(f"{exp_name} - Loss")
    plt.legend(); plt.tight_layout(); plt.savefig(loss_path, dpi=160); plt.close()
    return str(acc_path), str(loss_path)

def save_epoch_csv(exp_name, epochs_hist, out_dir):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ep_csv = out_dir / f"{exp_name}_metrics.csv"
    import csv
    with open(ep_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["epoch","train_loss","train_acc","val_loss","val_acc"])
        for i, e in enumerate(epochs_hist, start=1):
            w.writerow([i, round(e['train_loss'],6), round(e['train_acc'],4), round(e['val_loss'],6), round(e['val_acc'],4)])
    return str(ep_csv)

def run_experiment(exp_name, cfg, device, results_csv):
    model = TinyMNISTNet(
        use_bn=cfg.get("use_bn", True),
        dropout_p=cfg.get("dropout_p", 0.05),
        activation=cfg.get("activation", "relu"),
        last_channels=cfg.get("last_channels", 32)
    ).to(device)

    params = count_parameters(model)
    dl_train, dl_test = loaders(batch_size=cfg.get("batch_size", 128), augment=cfg.get("augment", True))

    optimizer = build_optimizer(cfg["optimizer"], model.parameters(), lr=cfg.get("lr", 0.0025),
                                weight_decay=cfg.get("weight_decay", 0.0), momentum=cfg.get("momentum", 0.9))
    scheduler = build_scheduler(cfg.get("scheduler"), optimizer, len(dl_train), cfg["epochs"])

    best_acc, best_epoch = 0.0, 0
    t0 = time.time()
    epochs_hist = []

    for epoch in range(1, cfg["epochs"]+1):
        tloss, tacc = train_one_epoch(model, dl_train, optimizer, device, scheduler, use_labelsmoothing=cfg.get("label_smoothing", 0.0))
        if isinstance(scheduler, StepLR): scheduler.step()
        if isinstance(scheduler, ReduceLROnPlateau):
            vloss_tmp, vacc_tmp = evaluate(model, dl_test, device); scheduler.step(vacc_tmp)
        vloss, vacc = evaluate(model, dl_test, device)
        epochs_hist.append({"train_loss":tloss,"train_acc":tacc,"val_loss":vloss,"val_acc":vacc})
        if vacc > best_acc: best_acc, best_epoch = vacc, epoch
        print(f"[{exp_name}] Epoch {epoch:02d}/{cfg['epochs']} | Train: loss {tloss:.4f}, acc {tacc:.2f}% | Val: loss {vloss:.4f}, acc {vacc:.2f}% | Best: {best_acc:.2f}%")

    dur = time.time() - t0
    vloss, vacc = evaluate(model, dl_test, device)

    # Save per-epoch CSV and plots
    ep_csv = save_epoch_csv(exp_name, epochs_hist, "results")
    acc_png, loss_png = plot_curves(exp_name, epochs_hist, "results/plots")
    # Save qualitative diagnostics
    cm_png = save_confusion_matrix(exp_name, model, dl_test, device, "results/plots")
    mis_png = save_misclassified_grid(exp_name, model, dl_test, device, "results/plots", max_items=36)

    # Log summary
    Path(results_csv).parent.mkdir(parents=True, exist_ok=True)
    write_header = not Path(results_csv).exists()
    with open(results_csv, "a", newline="") as f:
        w = csv.writer(f)
        if write_header:
            w.writerow(["exp_name","model_variant","use_bn","dropout_p","activation","optimizer","scheduler","lr","weight_decay",
                        "batch_size","epochs","params","train_time_sec","best_epoch","val_acc","val_loss",
                        "epoch_csv","acc_plot","loss_plot","cm_plot","miscls_plot"])
        w.writerow([
            exp_name, cfg.get("model_variant","TinyMNISTNet"), cfg.get("use_bn", True), cfg.get("dropout_p", 0.0),
            cfg.get("activation","relu"), cfg["optimizer"], cfg.get("scheduler"), cfg.get("lr", 0.001), cfg.get("weight_decay", 0.0),
            cfg.get("batch_size", 128), cfg["epochs"], params, round(dur,2), best_epoch, round(vacc,4), round(vloss,6),
            ep_csv, acc_png, loss_png, cm_png, mis_png
        ])

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, default="grid", choices=["grid","single"])
    parser.add_argument("--results_csv", type=str, default="results/results.csv")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--use_bn", type=int, default=1)
    parser.add_argument("--dropout_p", type=float, default=0.05)
    parser.add_argument("--activation", type=str, default="relu")
    parser.add_argument("--optimizer", type=str, default="adamw")
    parser.add_argument("--scheduler", type=str, default="step")
    parser.add_argument("--lr", type=float, default=0.0025)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--label_smoothing", type=float, default=0.0)
    parser.add_argument("--augment", type=int, default=1)
    parser.add_argument("--last_channels", type=int, default=32)
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    if args.mode == "single":
        cfg = {
            "model_variant":"TinyMNISTNet",
            "use_bn": bool(args.use_bn),
            "dropout_p": args.dropout_p,
            "activation": args.activation,
            "optimizer": args.optimizer,
            "scheduler": args.scheduler if args.scheduler.lower()!="none" else None,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "momentum": args.momentum,
            "batch_size": args.batch_size,
            "epochs": args.epochs,
            "label_smoothing": args.label_smoothing,
            "augment": bool(args.augment),
            "last_channels": args.last_channels
        }
        run_experiment("single_run", cfg, device, args.results_csv)
        return

    # GRID experiments
    grid = []
    grid.append({
        "exp_name": "A_noBN_noDO_vanillaGD",
        "model_variant": "TinyMNISTNet",
        "use_bn": False, "dropout_p": 0.0, "activation": "relu",
        "optimizer": "vanilla", "scheduler": None, "lr": 0.1, "weight_decay": 0.0, "momentum": 0.0,
        "batch_size": 128, "epochs": 15, "label_smoothing": 0.0, "augment": True, "last_channels": 32
    })
    opt_configs = [
        ("sgd_onecycle",   {"optimizer":"sgd",   "scheduler":"onecycle", "lr":0.05,  "weight_decay":0.0}),
        ("adamw_step",     {"optimizer":"adamw", "scheduler":"step",      "lr":0.0025,"weight_decay":1e-4}),
        ("rmsprop_plateau",{"optimizer":"rmsprop","scheduler":"plateau",  "lr":0.001, "weight_decay":1e-4}),
        ("adam_onecycle",  {"optimizer":"adam",  "scheduler":"onecycle",  "lr":0.003, "weight_decay":1e-4}),
    ]
    for name, cfg in opt_configs:
        grid.append({
            "exp_name": f"B_bn_do_{name}",
            "model_variant":"TinyMNISTNet",
            "use_bn": True, "dropout_p": 0.05, "activation": "relu",
            "batch_size": 128, "epochs": 15, "label_smoothing": 0.0, "augment": True, "last_channels": 32, **cfg
        })
    batch_sizes = [32, 64, 128]
    for name, cfg in opt_configs:
        for bs in batch_sizes:
            grid.append({
                "exp_name": f"C_bs_sweep_{name}_bs{bs}",
                "model_variant":"TinyMNISTNet",
                "use_bn": True, "dropout_p": 0.05, "activation": "relu",
                "batch_size": bs, "epochs": 15, "label_smoothing": 0.0, "augment": True, "last_channels": 32, **cfg
            })
    for act in ["relu", "silu", "gelu"]:
        grid.append({
            "exp_name": f"D_activation_{act}",
            "model_variant":"TinyMNISTNet",
            "use_bn": True, "dropout_p": 0.05, "activation": act,
            "optimizer": "adamw", "scheduler": "step", "lr": 0.0025, "weight_decay": 1e-4, "momentum": 0.9,
            "batch_size": 128, "epochs": 15, "label_smoothing": 0.0, "augment": True, "last_channels": 32
        })

    results_csv = args.results_csv
    for cfg in grid:
        cfg["epochs"] = 2
    for cfg in grid:
        exp_name = cfg.pop("exp_name")
        print(f"\n=== Running {exp_name} ===")
        run_experiment(exp_name, cfg, device, results_csv)

if __name__ == "__main__":
    main()
