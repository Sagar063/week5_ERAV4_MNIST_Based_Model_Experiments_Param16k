
import pandas as pd
from pathlib import Path

def to_md_table(df):
    out = []
    out.append("| " + " | ".join(df.columns) + " |")
    out.append("| " + " | ".join(["---"]*len(df.columns)) + " |")
    for _, row in df.iterrows():
        out.append("| " + " | ".join(str(v) for v in row.values) + " |")
    return "\n".join(out)

def main(results_csv="results/results.csv", out_readme="README.cmd", top_k=10):
    p = Path(results_csv)
    if not p.exists():
        print("No results.csv found. Run train.py first.")
        return
    df = pd.read_csv(p)
    df_sorted = df.sort_values(by=["val_acc","params"], ascending=[False, True]).reset_index(drop=True)

    show_cols = ["exp_name","model_variant","use_bn","dropout_p","activation","optimizer","scheduler",
                 "lr","batch_size","epochs","params","val_acc","val_loss","best_epoch","train_time_sec"]
    for c in show_cols:
        if c not in df_sorted.columns: df_sorted[c] = ""
    slim = df_sorted[show_cols].copy()
    slim["val_acc"] = slim["val_acc"].map(lambda x: f"{x:.2f}")

    md = []
    md.append("# MNIST Experiment Results\n")
    md.append("Sorted by **validation accuracy** (10k MNIST test set) and then by parameter count.\n")
    md.append("\n## Top 10\n")
    md.append(to_md_table(slim.head(top_k)))

    # Embed plots for top_k
    md.append("\n\n## Learning Curves (Top Results)\n")
    for i in range(min(top_k, len(df_sorted))):
        row = df_sorted.iloc[i]
        exp = row["exp_name"]
        acc_plot = row.get("acc_plot", "")
        loss_plot = row.get("loss_plot", "")
        ep_csv = row.get("epoch_csv", "")
        md.append(f"\n### {i+1}. {exp}\n")
        if acc_plot and Path(acc_plot).exists():
            md.append(f"**Accuracy:**\n\n![]({acc_plot})\n")
        if loss_plot and Path(loss_plot).exists():
            md.append(f"**Loss:**\n\n![]({loss_plot})\n")
        if ep_csv and Path(ep_csv).exists():
            md.append(f"Epoch CSV: `{ep_csv}`\n")

    md.append("\n\n## Full Results\n")
    md.append(to_md_table(slim))

    Path(out_readme).write_text("\n".join(md), encoding="utf-8")
    print(f"Wrote {out_readme}")

if __name__ == "__main__":
    main()
