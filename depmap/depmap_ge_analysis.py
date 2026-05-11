import pandas as pd
import numpy as np
from scipy.stats import mannwhitneyu
from statsmodels.stats.multitest import multipletests
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data_loader import load_all

# ─────────────────────────────────────────────
# LOAD DATA
# ─────────────────────────────────────────────

datasets  = load_all()
depmap_ge = datasets["depmap_ge"]

# ─────────────────────────────────────────────
# SPLIT INTO ADRN AND MES GROUPS
# Using AM_class_stringent to exclude HYBRID samples
# ─────────────────────────────────────────────

adrn = depmap_ge[depmap_ge["AM_class_stringent"] == "ADRN"]
mes  = depmap_ge[depmap_ge["AM_class_stringent"] == "MES"]

print(f"ADRN cell lines: {adrn.shape[0]}")
print(f"MES cell lines:  {mes.shape[0]}")
print(f"HYBRID excluded: {(depmap_ge['AM_class_stringent'] == 'HYBRID').sum()}")

# ─────────────────────────────────────────────
# IDENTIFY GENE COLUMNS
# ─────────────────────────────────────────────

meta_cols = ["condition", "ModelID", "MYCN_cn", "MYCN_status",
             "ADRN.score", "MES.score", "AM.score",
             "AM_class", "AM_class_stringent"]

gene_cols = [col for col in depmap_ge.columns if col not in meta_cols]
print(f"\nNumber of TF genes to analyse: {len(gene_cols)}")

# ─────────────────────────────────────────────
# DIFFERENTIAL GENE EXPRESSION ANALYSIS
# Mann-Whitney U test for each gene
# ─────────────────────────────────────────────

results = []

for gene in gene_cols:
    adrn_scores = adrn[gene].dropna()
    mes_scores  = mes[gene].dropna()

    if len(adrn_scores) < 2 or len(mes_scores) < 2:
        continue

    mean_adrn  = adrn_scores.mean()
    mean_mes   = mes_scores.mean()
    difference = mean_adrn - mean_mes

    stat, p_value = mannwhitneyu(adrn_scores, mes_scores, alternative="two-sided")

    results.append({
        "gene":       gene,
        "mean_ADRN":  round(mean_adrn,  4),
        "mean_MES":   round(mean_mes,   4),
        "difference": round(difference, 4),
        "p_value":    p_value,
    })

# ─────────────────────────────────────────────
# APPLY FDR CORRECTION (Benjamini-Hochberg)
# ─────────────────────────────────────────────

results_df = pd.DataFrame(results)

reject, q_values, _, _ = multipletests(
    results_df["p_value"], alpha=0.05, method="fdr_bh")

results_df["q_value"]     = q_values.round(6)
results_df["p_value"]     = results_df["p_value"].round(6)
results_df["significant"] = ["Yes" if r else "No" for r in reject]
results_df["abs_difference"] = results_df["difference"].abs()
results_df = results_df.sort_values("abs_difference", ascending=False).reset_index(drop=True)

# ─────────────────────────────────────────────
# SUMMARY
# ─────────────────────────────────────────────

total       = len(results_df)
significant = (results_df["significant"] == "Yes").sum()
adrn_genes  = ((results_df["significant"] == "Yes") & (results_df["difference"] > 0)).sum()
mes_genes   = ((results_df["significant"] == "Yes") & (results_df["difference"] < 0)).sum()

print(f"\nTotal TF genes analysed:       {total}")
print(f"Significant genes (q<0.05):    {significant}")
print(f"  → Higher expressed in ADRN:  {adrn_genes}")
print(f"  → Higher expressed in MES:   {mes_genes}")
print(f"\nTop 10 most differential TF genes:")
print(results_df[["gene", "mean_ADRN", "mean_MES", "difference",
                   "p_value", "q_value", "significant"]].head(10).to_string(index=False))

# ─────────────────────────────────────────────
# SAVE RESULTS
# ─────────────────────────────────────────────

results_df.to_csv("results/depmap_GE_differential.csv", index=False)
print("\nResults saved to results/depmap_GE_differential.csv")