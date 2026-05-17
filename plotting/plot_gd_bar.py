import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os

os.makedirs("results/plots", exist_ok=True)

TOP_N      = 20
ADRN_COLOR = "#E84B4B"
MES_COLOR  = "#4B9BE8"

gd  = pd.read_csv("results/depmap_GD_differential.csv")
top = gd.head(TOP_N).copy()

top["color"] = top["difference"].apply(
    lambda x: ADRN_COLOR if x < 0 else MES_COLOR)

fig, ax = plt.subplots(figsize=(9, 7))

ax.barh(top["gene"], top["difference"],
        color=top["color"], height=0.7, alpha=0.85)

ax.axvline(0, color="black", linewidth=0.8, linestyle="-", alpha=0.4)

x_min = top["difference"].min()
x_max = top["difference"].max()
ax.set_xlim(x_min * 1.25, x_max * 1.25)

for _, row in top.iterrows():
    diff = row["difference"]
    x_offset = 0.02 if diff > 0 else -0.02
    ha = "left" if diff > 0 else "right"
    ax.text(diff + x_offset, row["gene"],
            f"{diff:.2f}",
            va="center", ha=ha, fontsize=8, color="#444444")

ax.invert_yaxis()
ax.set_xlabel("Mean Difference in Dependency Score (ADRN − MES)", fontsize=11)
ax.set_ylabel("Transcription Factor", fontsize=11)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

adrn_patch = mpatches.Patch(color=ADRN_COLOR, label="More essential in ADRN")
mes_patch  = mpatches.Patch(color=MES_COLOR,  label="More essential in MES")
ax.legend(handles=[adrn_patch, mes_patch], fontsize=9,
          loc="lower right", frameon=False)

plt.tight_layout()
plt.savefig("results/plots/depmap_barplot_GD.png", dpi=300, bbox_inches="tight")
plt.close()
print("Saved: results/plots/depmap_barplot_GD.png")