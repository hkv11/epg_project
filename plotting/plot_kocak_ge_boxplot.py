import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data_loader import load_all

# ─────────────────────────────────────────────
# Boxplot — KOCAK top ADRN and MES genes (GE)
# Run from project root: python plotting/plot_kocak_ge_boxplot.py
# ─────────────────────────────────────────────

os.makedirs("results/plots", exist_ok=True)

ADRN_COLOR = "#E84B4B"
MES_COLOR  = "#4B9BE8"

ge_df    = pd.read_csv("results/kocak_GE_differential.csv")
datasets = load_all()
kocak_ge = datasets["kocak_ge"]

sig = ge_df[ge_df["significant"] == "Yes"]

top_adrn = sig[sig["difference"] > 0].nlargest(1, "abs_difference")["gene"].values[0]
q_adrn   = sig[sig["gene"] == top_adrn]["q_value"].values[0]

top_mes  = sig[sig["difference"] < 0].nlargest(1, "abs_difference")["gene"].values[0]
q_mes    = sig[sig["gene"] == top_mes]["q_value"].values[0]

print(f"Top ADRN gene: {top_adrn} (q = {q_adrn:.2e})")
print(f"Top MES gene:  {top_mes}  (q = {q_mes:.2e})")

fig, axes = plt.subplots(1, 2, figsize=(10, 6), sharey=False)

def draw_box(ax, gene_name, q_val, title):
    plot_df = kocak_ge[["condition", "AM_class_stringent", gene_name]].dropna()
    plot_df = plot_df[plot_df["AM_class_stringent"].isin(["ADRN", "MES"])]

    sns.boxplot(data=plot_df, x="AM_class_stringent", y=gene_name,
                palette={"ADRN": ADRN_COLOR, "MES": MES_COLOR},
                order=["ADRN", "MES"], ax=ax, width=0.5, linewidth=1.2)
    sns.stripplot(data=plot_df, x="AM_class_stringent", y=gene_name,
                  color="black", size=4, alpha=0.6,
                  order=["ADRN", "MES"], ax=ax, jitter=True)

    ax.axhline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.5)

    y_min   = plot_df[gene_name].min()
    y_max   = plot_df[gene_name].max()
    y_range = y_max - y_min
    ax.set_ylim(y_min - y_range * 0.05, y_max + y_range * 0.08)
    ax.text(0.98, 0.97, f"q = {q_val:.2e}",
            ha="right", va="top", fontsize=10,
            transform=ax.transAxes,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                      edgecolor="#cccccc", alpha=0.8))

    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_ylabel("Gene Expression Score (z-score)", fontsize=11)
    ax.set_xlabel("Tumour Class", fontsize=11)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

draw_box(axes[0], top_adrn, q_adrn, f"ADRN-associated: {top_adrn}")
draw_box(axes[1], top_mes,  q_mes,  f"MES-associated: {top_mes}")

plt.tight_layout()
plt.savefig("results/plots/kocak_boxplot_GE.png", dpi=300, bbox_inches="tight")
plt.close()
print("Saved: results/plots/kocak_boxplot_GE.png")