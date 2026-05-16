import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from adjustText import adjust_text
import os

# ─────────────────────────────────────────────
# Volcano plot — KOCAK GE differential
# Run from project root: python plotting/plot_kocak_ge_volcano.py
# ─────────────────────────────────────────────

os.makedirs("results/plots", exist_ok=True)

SIG_COLOR   = "#E84B4B"
INSIG_COLOR = "#CCCCCC"
Q_THRESH    = 0.05

df = pd.read_csv("results/kocak_GE_differential.csv")
df["neg_log10_q"] = -np.log10(df["q_value"].replace(0, 1e-12))
df["color"] = df["significant"].map({"Yes": SIG_COLOR, "No": INSIG_COLOR})

top_adrn = df[df["difference"] > 0].nlargest(1, "abs_difference")
top_mes  = df[df["difference"] < 0].nlargest(1, "abs_difference")
to_label = pd.concat([top_adrn, top_mes])

fig, ax = plt.subplots(figsize=(9, 6))
ax.scatter(df["difference"], df["neg_log10_q"],
           c=df["color"], alpha=0.6, s=18, linewidths=0)
ax.axhline(-np.log10(Q_THRESH), color="black", linewidth=0.8,
           linestyle="--", alpha=0.5, label=f"q = {Q_THRESH}")
ax.axvline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.3)

texts = []
for _, row in to_label.iterrows():
    texts.append(ax.text(row["difference"], row["neg_log10_q"],
                         row["gene"], fontsize=8, fontweight="bold"))
adjust_text(texts, arrowprops=dict(arrowstyle="-", color="black", lw=0.5))

ax.set_xlabel("Mean Difference (ADRN − MES)", fontsize=11)
ax.set_ylabel("−log10(q-value)", fontsize=11)
ax.set_title("KOCAK — Differential Gene Expression (ADRN vs MES)",
             fontsize=13, fontweight="bold")

sig_patch   = mpatches.Patch(color=SIG_COLOR,   label=f"Significant (q<{Q_THRESH})")
insig_patch = mpatches.Patch(color=INSIG_COLOR, label="Not significant")
ax.legend(handles=[sig_patch, insig_patch], fontsize=9)

plt.tight_layout()
plt.savefig("results/plots/kocak_volcano_GE.png", dpi=300, bbox_inches="tight")
plt.close()
print("Saved: results/plots/kocak_volcano_GE.png")