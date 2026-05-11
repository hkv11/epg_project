import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data_loader import load_all

# ─────────────────────────────────────────────
# Boxplot — DepMap top ADRN and MES TFs side by side
# Run from project root: python plotting/plot_depmap_boxplot.py
# ─────────────────────────────────────────────

os.makedirs("results/plots", exist_ok=True)

ADRN_COLOR = "#E84B4B"
MES_COLOR  = "#4B9BE8"

tf_df     = pd.read_csv("results/depmap_TF_differential.csv")
datasets  = load_all()
depmap_tf = datasets["depmap_tf"]

sig = tf_df[tf_df["significant"] == "Yes"]

# Top ADRN-associated: highest positive difference
top_adrn = sig[sig["difference"] > 0].nlargest(1, "abs_difference")["TF"].values[0]
q_adrn   = sig[sig["TF"] == top_adrn]["q_value"].values[0]

# Top MES-associated: highest negative difference
top_mes  = sig[sig["difference"] < 0].nlargest(1, "abs_difference")["TF"].values[0]
q_mes    = sig[sig["TF"] == top_mes]["q_value"].values[0]

print(f"Top ADRN TF: {top_adrn} (q = {q_adrn:.2e})")
print(f"Top MES TF:  {top_mes}  (q = {q_mes:.2e})")

fig, axes = plt.subplots(1, 2, figsize=(10, 6), sharey=False)

def draw_box(ax, tf_name, q_val, title):
    plot_df = depmap_tf[["condition", "AM_class_stringent", tf_name]].dropna()
    plot_df = plot_df[plot_df["AM_class_stringent"].isin(["ADRN", "MES"])]

    sns.boxplot(data=plot_df, x="AM_class_stringent", y=tf_name,
                palette={"ADRN": ADRN_COLOR, "MES": MES_COLOR},
                order=["ADRN", "MES"], ax=ax, width=0.5, linewidth=1.2)
    sns.stripplot(data=plot_df, x="AM_class_stringent", y=tf_name,
                  color="black", size=4, alpha=0.6,
                  order=["ADRN", "MES"], ax=ax, jitter=True)

    ax.axhline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.5)

    y_min   = plot_df[tf_name].min()
    y_max   = plot_df[tf_name].max()
    y_range = y_max - y_min
    ax.set_ylim(y_min - y_range * 0.05, y_max + y_range * 0.08)
    ax.text(0.98, 0.97, f"q = {q_val:.2e}",
            ha="right", va="top", fontsize=10,
            transform=ax.transAxes,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                      edgecolor="#cccccc", alpha=0.8))

    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_ylabel("TF Activity Score (z-score)", fontsize=11)
    ax.set_xlabel("Cell Line Class", fontsize=11)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

draw_box(axes[0], top_adrn, q_adrn,
         f"ADRN-associated: {top_adrn}")
draw_box(axes[1], top_mes,  q_mes,
         f"MES-associated: {top_mes}")

fig.suptitle("DepMap — Top Differentially Active TFs (ADRN vs MES)",
             fontsize=13, fontweight="bold", y=1.01)

plt.tight_layout()
plt.savefig("results/plots/depmap_boxplot_TF.png", dpi=300, bbox_inches="tight")
plt.close()
print("Saved: results/plots/depmap_boxplot_TF.png")