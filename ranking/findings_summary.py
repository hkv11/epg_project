import pandas as pd
import os

# ─────────────────────────────────────────────
# FINDINGS SUMMARY
# Run from project root: python ranking/findings_summary.py
# Extracts all [X] values needed for the findings section
# ─────────────────────────────────────────────

RESULTS_DIR = "results"
Q_THRESH    = 0.05

print("=" * 70)
print("FINDINGS SUMMARY — [X] VALUES FOR REPORT")
print("=" * 70)

# ─────────────────────────────────────────────
# 4.1 DEPMAP TF DIFFERENTIAL
# ─────────────────────────────────────────────

dm_tf = pd.read_csv(os.path.join(RESULTS_DIR, "depmap_TF_differential.csv"))

dm_tf_sig      = dm_tf[dm_tf["significant"] == "Yes"]
dm_tf_sig_n    = len(dm_tf_sig)
dm_tf_adrn_n   = (dm_tf_sig["difference"] > 0).sum()
dm_tf_mes_n    = (dm_tf_sig["difference"] < 0).sum()
dm_tf_top_adrn = dm_tf_sig[dm_tf_sig["difference"] > 0].nlargest(3, "abs_difference")[["TF", "difference", "q_value"]].values.tolist()
dm_tf_top_mes  = dm_tf_sig[dm_tf_sig["difference"] < 0].nlargest(3, "abs_difference")[["TF", "difference", "q_value"]].values.tolist()

print("\n── 4.1 DepMap TF Differential ─────────────────────────────────────")
print(f"  Total TFs tested:                            {len(dm_tf)}")
print(f"  Significant (q<0.05):                        {dm_tf_sig_n}")
print(f"  Higher in ADRN:                              {dm_tf_adrn_n}")
print(f"  Higher in MES:                               {dm_tf_mes_n}")
print(f"  Top 3 ADRN TFs:")
for tf, diff, q in dm_tf_top_adrn:
    print(f"    {tf:<12}  diff = {diff:+.3f},  q = {q:.2e}")
print(f"  Top 3 MES TFs:")
for tf, diff, q in dm_tf_top_mes:
    print(f"    {tf:<12}  diff = {diff:+.3f},  q = {q:.2e}")

# ─────────────────────────────────────────────
# 4.1 KOCAK TF DIFFERENTIAL
# ─────────────────────────────────────────────

ko_tf = pd.read_csv(os.path.join(RESULTS_DIR, "kocak_TF_differential.csv"))

ko_tf_sig      = ko_tf[ko_tf["significant"] == "Yes"]
ko_tf_sig_n    = len(ko_tf_sig)
ko_tf_adrn_n   = (ko_tf_sig["difference"] > 0).sum()
ko_tf_mes_n    = (ko_tf_sig["difference"] < 0).sum()
ko_tf_top_adrn = ko_tf_sig[ko_tf_sig["difference"] > 0].nlargest(3, "abs_difference")[["TF", "difference", "q_value"]].values.tolist()
ko_tf_top_mes  = ko_tf_sig[ko_tf_sig["difference"] < 0].nlargest(3, "abs_difference")[["TF", "difference", "q_value"]].values.tolist()

print("\n── 4.1 KOCAK TF Differential ───────────────────────────────────────")
print(f"  Total TFs tested:                            {len(ko_tf)}")
print(f"  Significant (q<0.05):                        {ko_tf_sig_n}")
print(f"  Higher in ADRN:                              {ko_tf_adrn_n}")
print(f"  Higher in MES:                               {ko_tf_mes_n}")
print(f"  Top 3 ADRN TFs:")
for tf, diff, q in ko_tf_top_adrn:
    print(f"    {tf:<12}  diff = {diff:+.3f},  q = {q:.2e}")
print(f"  Top 3 MES TFs:")
for tf, diff, q in ko_tf_top_mes:
    print(f"    {tf:<12}  diff = {diff:+.3f},  q = {q:.2e}")

# ─────────────────────────────────────────────
# 4.2 DEPMAP GE DIFFERENTIAL
# ─────────────────────────────────────────────

dm_ge = pd.read_csv(os.path.join(RESULTS_DIR, "depmap_GE_differential.csv"))

dm_ge_sig      = dm_ge[dm_ge["significant"] == "Yes"]
dm_ge_sig_n    = len(dm_ge_sig)
dm_ge_adrn_n   = (dm_ge_sig["difference"] > 0).sum()
dm_ge_mes_n    = (dm_ge_sig["difference"] < 0).sum()
dm_ge_top_adrn = dm_ge_sig[dm_ge_sig["difference"] > 0].nlargest(3, "abs_difference")[["gene", "difference", "q_value"]].values.tolist()
dm_ge_top_mes  = dm_ge_sig[dm_ge_sig["difference"] < 0].nlargest(3, "abs_difference")[["gene", "difference", "q_value"]].values.tolist()

print("\n── 4.2 DepMap GE Differential ──────────────────────────────────────")
print(f"  Total genes tested:                          {len(dm_ge)}")
print(f"  Significant (q<0.05):                        {dm_ge_sig_n}")
print(f"  Higher in ADRN:                              {dm_ge_adrn_n}")
print(f"  Higher in MES:                               {dm_ge_mes_n}")
print(f"  Top 3 ADRN genes:")
for gene, diff, q in dm_ge_top_adrn:
    print(f"    {gene:<12}  diff = {diff:+.3f},  q = {q:.2e}")
print(f"  Top 3 MES genes:")
for gene, diff, q in dm_ge_top_mes:
    print(f"    {gene:<12}  diff = {diff:+.3f},  q = {q:.2e}")

# ─────────────────────────────────────────────
# 4.2 KOCAK GE DIFFERENTIAL
# ─────────────────────────────────────────────

ko_ge = pd.read_csv(os.path.join(RESULTS_DIR, "kocak_GE_differential.csv"))

ko_ge_sig      = ko_ge[ko_ge["significant"] == "Yes"]
ko_ge_sig_n    = len(ko_ge_sig)
ko_ge_adrn_n   = (ko_ge_sig["difference"] > 0).sum()
ko_ge_mes_n    = (ko_ge_sig["difference"] < 0).sum()
ko_ge_top_adrn = ko_ge_sig[ko_ge_sig["difference"] > 0].nlargest(3, "abs_difference")[["gene", "difference", "q_value"]].values.tolist()
ko_ge_top_mes  = ko_ge_sig[ko_ge_sig["difference"] < 0].nlargest(3, "abs_difference")[["gene", "difference", "q_value"]].values.tolist()

print("\n── 4.2 KOCAK GE Differential ───────────────────────────────────────")
print(f"  Total genes tested:                          {len(ko_ge)}")
print(f"  Significant (q<0.05):                        {ko_ge_sig_n}")
print(f"  Higher in ADRN:                              {ko_ge_adrn_n}")
print(f"  Higher in MES:                               {ko_ge_mes_n}")
print(f"  Top 3 ADRN genes:")
for gene, diff, q in ko_ge_top_adrn:
    print(f"    {gene:<12}  diff = {diff:+.3f},  q = {q:.2e}")
print(f"  Top 3 MES genes:")
for gene, diff, q in ko_ge_top_mes:
    print(f"    {gene:<12}  diff = {diff:+.3f},  q = {q:.2e}")

# ─────────────────────────────────────────────
# 4.3 GENE DEPENDENCY
# ─────────────────────────────────────────────

gd = pd.read_csv(os.path.join(RESULTS_DIR, "depmap_GD_differential.csv"))

gd_sig_n      = (gd["significant"] == "Yes").sum()
gd_adrn_sig   = gd[(gd["significant"] == "Yes") & (gd["difference"] < 0)]
gd_mes_sig    = gd[(gd["significant"] == "Yes") & (gd["difference"] > 0)]
gd_top_adrn   = gd[gd["difference"] < 0].nlargest(3, "abs_difference")[["gene", "mean_ADRN", "mean_MES", "difference", "q_value"]].values.tolist()
gd_top_mes    = gd[gd["difference"] > 0].nlargest(3, "abs_difference")[["gene", "mean_ADRN", "mean_MES", "difference", "q_value"]].values.tolist()

# MYCN specifically
mycn = gd[gd["gene"] == "MYCN"]

print("\n── 4.3 Gene Dependency ─────────────────────────────────────────────")
print(f"  Total TFs tested:                            {len(gd)}")
print(f"  Significant (q<0.05):                        {gd_sig_n}")
print(f"  Note: low n due to underpowered test")
print(f"  ADRN-selective (diff < 0, significant):      {len(gd_adrn_sig)}")
print(f"  MES-selective  (diff > 0, significant):      {len(gd_mes_sig)}")
print(f"  Top 3 ADRN-essential (by abs_difference, all TFs):")
for gene, m_a, m_m, diff, q in gd_top_adrn:
    print(f"    {gene:<12}  mean_ADRN = {m_a:.3f},  mean_MES = {m_m:.3f},  diff = {diff:+.3f},  q = {q:.2e}")
print(f"  Top 3 MES-essential (by abs_difference, all TFs):")
for gene, m_a, m_m, diff, q in gd_top_mes:
    print(f"    {gene:<12}  mean_ADRN = {m_a:.3f},  mean_MES = {m_m:.3f},  diff = {diff:+.3f},  q = {q:.2e}")
if len(mycn) > 0:
    print(f"  MYCN: mean_ADRN = {mycn['mean_ADRN'].values[0]:.3f}, mean_MES = {mycn['mean_MES'].values[0]:.3f}, diff = {mycn['difference'].values[0]:+.3f}, q = {mycn['q_value'].values[0]:.2e}")

# ─────────────────────────────────────────────
# 4.4 SURVIVAL
# ─────────────────────────────────────────────

surv = pd.read_csv(os.path.join(RESULTS_DIR, "kocak_TF_survival.csv"))

efs_sig      = surv[surv["EFS_significant"] == "Yes"]
os_sig       = surv[surv["OS_significant"]  == "Yes"]
both_sig     = surv[surv["both_significant"] == "Yes"]
efs_poor     = efs_sig[efs_sig["EFS_HR"] > 1]
efs_good     = efs_sig[efs_sig["EFS_HR"] < 1]
top_efs      = efs_poor.nlargest(5, "EFS_HR")[["TF", "EFS_HR", "EFS_q_value", "OS_HR", "OS_q_value"]].values.tolist()
top_surv_tf  = efs_poor.nlargest(1, "EFS_HR")["TF"].values[0]

print("\n── 4.4 Survival Analysis ───────────────────────────────────────────")
print(f"  Total TFs tested:                            {len(surv)}")
print(f"  Significant EFS (q<0.05):                    {len(efs_sig)}")
print(f"  Significant OS  (q<0.05):                    {len(os_sig)}")
print(f"  Significant both:                            {len(both_sig)}")
print(f"  Poor outcome EFS (HR>1):                     {len(efs_poor)}")
print(f"  Good outcome EFS (HR<1):                     {len(efs_good)}")
print(f"  Top 5 by EFS HR (poor prognosis only):")
for tf, efs_hr, efs_q, os_hr, os_q in top_efs:
    print(f"    {tf:<12}  EFS HR = {efs_hr:.2f} (q={efs_q:.2e}),  OS HR = {os_hr:.2f} (q={os_q:.2e})")
print(f"  Top TF for KM plot:                          {top_surv_tf}")

# ─────────────────────────────────────────────
# 4.5 INTEGRATIVE RANKING
# ─────────────────────────────────────────────

adrn_rank = pd.read_csv(os.path.join(RESULTS_DIR, "ranking_ADRN.csv"))
mes_rank  = pd.read_csv(os.path.join(RESULTS_DIR, "ranking_MES.csv"))

adrn_top10 = adrn_rank.head(10)[["TF", "score", "g1_dm_diff", "g2_gd", "g3_ko_diff", "g4_survival"]].values.tolist()
mes_top10  = mes_rank.head(10)[["TF", "score", "g1_dm_diff", "g2_gd", "g3_ko_diff", "g4_survival"]].values.tolist()


print("\n── 4.5 Integrative Ranking ─────────────────────────────────────────")
print(f"  Top 10 ADRN candidates:")
for tf, score, g1, g2, g3, g4 in adrn_top10:
    print(f"    {tf:<12}  score={score:.2f}  g1={g1:.2f}  g2={g2:.2f}  g3={g3:.2f}  g4={g4:.2f}")
print(f"\n  Top 10 MES candidates:")
for tf, score, g1, g2, g3, g4 in mes_top10:
    print(f"    {tf:<12}  score={score:.2f}  g1={g1:.2f}  g2={g2:.2f}  g3={g3:.2f}  g4={g4:.2f}")


print("\n" + "=" * 70)
print("END OF FINDINGS SUMMARY")
