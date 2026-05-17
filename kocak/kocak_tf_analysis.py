import pandas as pd
import numpy as np
from scipy.stats import mannwhitneyu
from statsmodels.stats.multitest import multipletests
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data_loader import load_all

datasets  = load_all()
kocak_tf  = datasets["kocak_tf"]

adrn = kocak_tf[kocak_tf["AM_class_stringent"] == "ADRN"]
mes  = kocak_tf[kocak_tf["AM_class_stringent"] == "MES"]

print(f"ADRN tumors: {adrn.shape[0]}")
print(f"MES tumors:  {mes.shape[0]}")
print(f"HYBRID excluded: {(kocak_tf['AM_class_stringent'] == 'HYBRID').sum()}")

meta_cols = ["condition", "MYCN_status", "MYCN_bin", "INSS",
             "INSS_bin_4vall", "Age_d", "Age_bin_18m", "Cell_Name_M",
             "ADRN.score", "MES.score", "AM.score",
             "AM_class", "AM_class_stringent"]

tf_cols = [col for col in kocak_tf.columns if col not in meta_cols]
print(f"\nNumber of TFs to analyse: {len(tf_cols)}")

results = []

for tf in tf_cols:
    adrn_scores = adrn[tf].dropna()
    mes_scores  = mes[tf].dropna()

    if len(adrn_scores) < 2 or len(mes_scores) < 2:
        continue

    mean_adrn  = adrn_scores.mean()
    mean_mes   = mes_scores.mean()
    difference = mean_adrn - mean_mes

    stat, p_value = mannwhitneyu(adrn_scores, mes_scores, alternative="two-sided")

    results.append({
        "TF":         tf,
        "mean_ADRN":  round(mean_adrn,  4),
        "mean_MES":   round(mean_mes,   4),
        "difference": round(difference, 4),
        "p_value":    p_value,
    })

results_df = pd.DataFrame(results)

reject, q_values, _, _ = multipletests(
    results_df["p_value"], alpha=0.05, method="fdr_bh")

results_df["q_value"]        = q_values.round(6)
results_df["p_value"]        = results_df["p_value"].round(6)
results_df["significant"]    = ["Yes" if r else "No" for r in reject]
results_df["abs_difference"] = results_df["difference"].abs()
results_df = results_df.sort_values("abs_difference", ascending=False).reset_index(drop=True)

total       = len(results_df)
significant = (results_df["significant"] == "Yes").sum()
adrn_tfs    = ((results_df["significant"] == "Yes") & (results_df["difference"] > 0)).sum()
mes_tfs     = ((results_df["significant"] == "Yes") & (results_df["difference"] < 0)).sum()

print(f"\nTotal TFs analysed:        {total}")
print(f"Significant TFs (q<0.05):  {significant}")
print(f"  → Higher in ADRN:        {adrn_tfs}")
print(f"  → Higher in MES:         {mes_tfs}")
print(f"\nTop 10 most differential TFs:")
print(results_df[["TF", "mean_ADRN", "mean_MES", "difference",
                   "p_value", "q_value", "significant"]].head(10).to_string(index=False))

results_df.to_csv("results/kocak_TF_differential.csv", index=False)
print("\nResults saved to results/kocak_TF_differential.csv")