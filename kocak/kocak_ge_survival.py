import pandas as pd
import numpy as np
from lifelines.statistics import logrank_test
from lifelines import CoxPHFitter
from statsmodels.stats.multitest import multipletests
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data_loader import load_all

# ─────────────────────────────────────────────
# LOAD DATA
# ─────────────────────────────────────────────

datasets       = load_all()
kocak_ge       = datasets["kocak_ge"]
kocak_survival = datasets["kocak_survival"]

# ─────────────────────────────────────────────
# MERGE GE WITH SURVIVAL DATA
# ─────────────────────────────────────────────

merged = kocak_ge.merge(
    kocak_survival[["condition", "EFS_d", "EFS_bin", "OS_d", "OS_bin"]],
    on="condition", how="inner"
)

print(f"Patients after merge: {merged.shape[0]}")

# ─────────────────────────────────────────────
# IDENTIFY GENE COLUMNS
# ─────────────────────────────────────────────

meta_cols = ["condition", "MYCN_status", "MYCN_bin", "INSS",
             "INSS_bin_4vall", "Age_d", "Age_bin_18m", "Cell_Name_M",
             "ADRN.score", "MES.score", "AM.score",
             "AM_class", "AM_class_stringent",
             "EFS_d", "EFS_bin", "OS_d", "OS_bin"]

gene_cols = [col for col in merged.columns if col not in meta_cols]
print(f"Number of TF genes to analyse: {len(gene_cols)}")

# ─────────────────────────────────────────────
# SURVIVAL ANALYSIS
# For each gene:
#   - Split patients into High/Low by median GE score
#   - Log-rank test for EFS and OS (p-value)
#   - Cox model for EFS and OS (hazard ratio)
#
# HR > 1 = high expression = worse survival (poor outcome)
# HR < 1 = high expression = better survival (good outcome)
# ─────────────────────────────────────────────

results = []

for gene in gene_cols:
    scores = merged[gene].dropna()
    if len(scores) < 10:
        continue

    median = scores.median()
    merged["group"] = (merged[gene] > median).astype(int)  # 1 = High, 0 = Low

    high = merged[merged["group"] == 1]
    low  = merged[merged["group"] == 0]

    if len(high) < 5 or len(low) < 5:
        continue

    # ── EFS ──
    efs_logrank = logrank_test(
        high["EFS_d"], low["EFS_d"],
        event_observed_A=high["EFS_bin"],
        event_observed_B=low["EFS_bin"]
    )

    try:
        cph_efs = CoxPHFitter()
        cph_efs.fit(merged[["EFS_d", "EFS_bin", "group"]].dropna(),
                    duration_col="EFS_d", event_col="EFS_bin")
        efs_hr = cph_efs.hazard_ratios_["group"]
    except Exception:
        efs_hr = np.nan

    # ── OS ──
    os_logrank = logrank_test(
        high["OS_d"], low["OS_d"],
        event_observed_A=high["OS_bin"],
        event_observed_B=low["OS_bin"]
    )

    try:
        cph_os = CoxPHFitter()
        cph_os.fit(merged[["OS_d", "OS_bin", "group"]].dropna(),
                   duration_col="OS_d", event_col="OS_bin")
        os_hr = cph_os.hazard_ratios_["group"]
    except Exception:
        os_hr = np.nan

    results.append({
        "gene":         gene,
        "median_score": round(median, 4),
        "EFS_HR":       round(efs_hr, 4) if not np.isnan(efs_hr) else np.nan,
        "EFS_p_value":  efs_logrank.p_value,
        "OS_HR":        round(os_hr,  4) if not np.isnan(os_hr)  else np.nan,
        "OS_p_value":   os_logrank.p_value,
    })

# ─────────────────────────────────────────────
# APPLY FDR CORRECTION
# ─────────────────────────────────────────────

results_df = pd.DataFrame(results)

_, efs_q, _, _ = multipletests(results_df["EFS_p_value"], alpha=0.05, method="fdr_bh")
_, os_q,  _, _ = multipletests(results_df["OS_p_value"],  alpha=0.05, method="fdr_bh")

results_df["EFS_q_value"]     = efs_q
results_df["OS_q_value"]      = os_q
results_df["EFS_significant"] = ["Yes" if q < 0.05 else "No" for q in efs_q]
results_df["OS_significant"]  = ["Yes" if q < 0.05 else "No" for q in os_q]
results_df["both_significant"] = [
    "Yes" if e == "Yes" and o == "Yes" else "No"
    for e, o in zip(results_df["EFS_significant"], results_df["OS_significant"])
]

# Format p and q values
for col in ["EFS_p_value", "OS_p_value", "EFS_q_value", "OS_q_value"]:
    results_df[col] = results_df[col].apply(lambda x: float(f"{x:.2e}"))

results_df = results_df.sort_values("EFS_q_value", ascending=True).reset_index(drop=True)

# ─────────────────────────────────────────────
# SUMMARY
# ─────────────────────────────────────────────

efs_sig  = (results_df["EFS_significant"]  == "Yes").sum()
os_sig   = (results_df["OS_significant"]   == "Yes").sum()
both_sig = (results_df["both_significant"] == "Yes").sum()

efs_poor = ((results_df["EFS_significant"] == "Yes") & (results_df["EFS_HR"] > 1)).sum()
efs_good = ((results_df["EFS_significant"] == "Yes") & (results_df["EFS_HR"] < 1)).sum()

print(f"\nTotal TF genes analysed:        {len(results_df)}")
print(f"Significant EFS (q<0.05):       {efs_sig}")
print(f"  → Poor outcome (HR>1):        {efs_poor}")
print(f"  → Good outcome (HR<1):        {efs_good}")
print(f"Significant OS  (q<0.05):       {os_sig}")
print(f"Significant in both:            {both_sig}")
print(f"\nTop 10 TF genes by EFS q-value:")
print(results_df[["gene", "EFS_HR", "EFS_q_value", "EFS_significant",
                   "OS_HR", "OS_q_value", "OS_significant"]].head(10).to_string(index=False))

# ─────────────────────────────────────────────
# SAVE RESULTS
# ─────────────────────────────────────────────

results_df.to_csv("results/kocak_GE_survival.csv", index=False)
print("\nResults saved to results/kocak_GE_survival.csv")