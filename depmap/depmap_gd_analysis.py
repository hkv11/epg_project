import pandas as pd
import numpy as np
from scipy.stats import mannwhitneyu
from statsmodels.stats.multitest import multipletests
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data_loader import load_all

datasets  = load_all()
depmap_gd = datasets["depmap_gd"]

adrn = depmap_gd[depmap_gd["AM_class_stringent"] == "ADRN"]
mes  = depmap_gd[depmap_gd["AM_class_stringent"] == "MES"]

print(f"ADRN cell lines: {adrn.shape[0]}")
print(f"MES cell lines:  {mes.shape[0]}")
print(f"HYBRID excluded: {(depmap_gd['AM_class_stringent'] == 'HYBRID').sum()}")

meta_cols = ["condition", "ModelID", "MYCN_cn", "MYCN_status",
             "ADRN.score", "MES.score", "AM.score",
             "AM_class", "AM_class_stringent"]

gd_cols = [col for col in depmap_gd.columns if col not in meta_cols]
print(f"\nNumber of TF dependency genes to analyse: {len(gd_cols)}")

results = []

for gene in gd_cols:
    adrn_scores = adrn[gene].dropna()
    mes_scores  = mes[gene].dropna()

    if len(adrn_scores) < 2 or len(mes_scores) < 2:
        continue

    mean_adrn  = adrn_scores.mean()
    mean_mes   = mes_scores.mean()
    difference = round(mean_adrn - mean_mes, 4)

    stat, p_value = mannwhitneyu(adrn_scores, mes_scores,
                                  alternative="two-sided")

    results.append({
        "gene":         gene,
        "mean_ADRN":    round(mean_adrn, 4),
        "mean_MES":     round(mean_mes,  4),
        "difference":   difference,
        "p_value":      p_value,
    })

results_df = pd.DataFrame(results)

reject, q_values, _, _ = multipletests(
    results_df["p_value"], alpha=0.05, method="fdr_bh")

results_df["q_value"]        = q_values.round(6)
results_df["p_value"]        = results_df["p_value"].round(6)
results_df["significant"]    = ["Yes" if r else "No" for r in reject]
results_df["abs_difference"] = results_df["difference"].abs()

# Sort by abs_difference descending
results_df = results_df.sort_values(
    "abs_difference", ascending=False).reset_index(drop=True)


total     = len(results_df)
sig       = (results_df["significant"] == "Yes").sum()
adrn_deps = ((results_df["significant"] == "Yes") &
             (results_df["difference"] < 0)).sum()
mes_deps  = ((results_df["significant"] == "Yes") &
             (results_df["difference"] > 0)).sum()

print(f"\nTotal TFs analysed:                    {total}")
print(f"Significant (q<0.05):                  {sig}")
print(f"  → More essential in ADRN (diff < 0): {adrn_deps}")
print(f"  → More essential in MES  (diff > 0): {mes_deps}")

print(f"\nTop 10 most differentially essential TFs:")
print(results_df[["gene", "mean_ADRN", "mean_MES", "difference",
                   "q_value", "significant"]].head(10).to_string(index=False))

results_df.to_csv("results/depmap_GD_differential.csv", index=False)
print("\nResults saved to results/depmap_GD_differential.csv")