import pandas as pd
import matplotlib.pyplot as plt
from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data_loader import load_all

# ─────────────────────────────────────────────
# Kaplan-Meier — top survival TF
# EFS and OS side by side
# Run from project root: python plotting/plot_survival_km.py
# ─────────────────────────────────────────────

os.makedirs("results/plots", exist_ok=True)

HIGH_COLOR = "#E84B4B"
LOW_COLOR  = "#4B9BE8"

# Load survival results to find top TF
surv_results = pd.read_csv("results/kocak_TF_survival.csv")
efs_poor     = surv_results[
    (surv_results["EFS_significant"] == "Yes") &
    (surv_results["EFS_HR"] > 1)
]
top_tf = efs_poor.nlargest(1, "EFS_HR")["TF"].values[0]
print(f"Top survival TF: {top_tf}")

# Load raw data
datasets       = load_all()
kocak_tf       = datasets["kocak_tf"]
kocak_survival = datasets["kocak_survival"]

merged = kocak_tf.merge(
    kocak_survival[["condition", "EFS_d", "EFS_bin", "OS_d", "OS_bin"]],
    on="condition", how="inner"
)

# Split by median
median     = merged[top_tf].median()
merged["group"] = (merged[top_tf] > median).map({True: "High", False: "Low"})
high = merged[merged["group"] == "High"]
low  = merged[merged["group"] == "Low"]

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

for ax, duration_col, event_col, label in [
    (axes[0], "EFS_d", "EFS_bin", "Event-Free Survival"),
    (axes[1], "OS_d",  "OS_bin",  "Overall Survival")
]:
    kmf_high = KaplanMeierFitter()
    kmf_low  = KaplanMeierFitter()

    kmf_high.fit(high[duration_col], high[event_col], label=f"High {top_tf}")
    kmf_low.fit(low[duration_col],  low[event_col],  label=f"Low {top_tf}")

    kmf_high.plot_survival_function(ax=ax, color=HIGH_COLOR, linewidth=2, ci_show=False)
    kmf_low.plot_survival_function(ax=ax,  color=LOW_COLOR,  linewidth=2, ci_show=False)

    # Log-rank p-value
    result = logrank_test(
        high[duration_col], low[duration_col],
        event_observed_A=high[event_col],
        event_observed_B=low[event_col]
    )
    ax.text(0.05, 0.08, f"log-rank p = {result.p_value:.2e}",
            transform=ax.transAxes, fontsize=9,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.7))

    ax.set_title(f"{label} — {top_tf}", fontsize=12, fontweight="bold")
    ax.set_xlabel("Days", fontsize=11)
    ax.set_ylabel("Survival Probability", fontsize=11)
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=9)

plt.suptitle(f"KOCAK — Survival Analysis: {top_tf} (High vs Low Activity)",
             fontsize=13, fontweight="bold", y=1.02)
plt.tight_layout()
plt.savefig("results/plots/kocak_survival_KM.png", dpi=300, bbox_inches="tight")
plt.close()
print("Saved: results/plots/kocak_survival_KM.png")