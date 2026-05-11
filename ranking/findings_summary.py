import pandas as pd
import os

# ─────────────────────────────────────────────
# FINDINGS SUMMARY
# Run from project root: python ranking/findings_summary.py
# Extracts all [X] values needed for the findings section
# Each print statement is labelled with the exact [X] it fills
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
dm_tf_top_adrn = dm_tf_sig[dm_tf_sig["difference"] > 0].nlargest(3, "abs_difference")["TF"].tolist()
dm_tf_top_mes  = dm_tf_sig[dm_tf_sig["difference"] < 0].nlargest(3, "abs_difference")["TF"].tolist()

print("\n── 4.1 DepMap TF Differential ─────────────────────────────────────")
print(f"  [X] = total TFs tested:                     {len(dm_tf)}")
print(f"  [X] = significant (q<0.05):                 {dm_tf_sig_n}")
print(f"        → used in: 'DepMap TF differential: [X] significant TFs'")
print(f"  [X] = higher in ADRN:                       {dm_tf_adrn_n}")
print(f"  [X] = higher in MES:                        {dm_tf_mes_n}")
print(f"  Top ADRN TFs (replace if different):        {', '.join(dm_tf_top_adrn)}")
print(f"  Top MES TFs  (replace if different):        {', '.join(dm_tf_top_mes)}")

# ─────────────────────────────────────────────
# 4.1 KOCAK TF DIFFERENTIAL
# ─────────────────────────────────────────────

ko_tf = pd.read_csv(os.path.join(RESULTS_DIR, "kocak_TF_differential.csv"))

ko_tf_sig      = ko_tf[ko_tf["significant"] == "Yes"]
ko_tf_sig_n    = len(ko_tf_sig)
ko_tf_adrn_n   = (ko_tf_sig["difference"] > 0).sum()
ko_tf_mes_n    = (ko_tf_sig["difference"] < 0).sum()
ko_tf_top_adrn = ko_tf_sig[ko_tf_sig["difference"] > 0].nlargest(3, "abs_difference")["TF"].tolist()
ko_tf_top_mes  = ko_tf_sig[ko_tf_sig["difference"] < 0].nlargest(3, "abs_difference")["TF"].tolist()

print("\n── 4.1 KOCAK TF Differential ───────────────────────────────────────")
print(f"  [X] = total TFs tested:                     {len(ko_tf)}")
print(f"  [X] = significant (q<0.05):                 {ko_tf_sig_n}")
print(f"        → used in: 'KOCAK TF differential: [X] significant TFs'")
print(f"  [X] = higher in ADRN:                       {ko_tf_adrn_n}")
print(f"  [X] = higher in MES:                        {ko_tf_mes_n}")
print(f"  Top ADRN TFs (replace if different):        {', '.join(ko_tf_top_adrn)}")
print(f"  Top MES TFs  (replace if different):        {', '.join(ko_tf_top_mes)}")

# ─────────────────────────────────────────────
# 4.2 DEPMAP GE DIFFERENTIAL
# ─────────────────────────────────────────────

dm_ge = pd.read_csv(os.path.join(RESULTS_DIR, "depmap_GE_differential.csv"))

dm_ge_sig    = dm_ge[dm_ge["significant"] == "Yes"]
dm_ge_sig_n  = len(dm_ge_sig)
dm_ge_adrn_n = (dm_ge_sig["difference"] > 0).sum()
dm_ge_mes_n  = (dm_ge_sig["difference"] < 0).sum()

print("\n── 4.2 DepMap GE Differential ──────────────────────────────────────")
print(f"  [X] = total genes tested:                   {len(dm_ge)}")
print(f"  [X] = significant (q<0.05):                 {dm_ge_sig_n}")
print(f"        → used in: 'DepMap GE differential: [X] significant genes'")
print(f"  [X] = higher in ADRN:                       {dm_ge_adrn_n}")
print(f"  [X] = higher in MES:                        {dm_ge_mes_n}")

# ─────────────────────────────────────────────
# 4.2 KOCAK GE DIFFERENTIAL
# ─────────────────────────────────────────────

ko_ge = pd.read_csv(os.path.join(RESULTS_DIR, "kocak_GE_differential.csv"))

ko_ge_sig    = ko_ge[ko_ge["significant"] == "Yes"]
ko_ge_sig_n  = len(ko_ge_sig)
ko_ge_adrn_n = (ko_ge_sig["difference"] > 0).sum()
ko_ge_mes_n  = (ko_ge_sig["difference"] < 0).sum()

print("\n── 4.2 KOCAK GE Differential ───────────────────────────────────────")
print(f"  [X] = total genes tested:                   {len(ko_ge)}")
print(f"  [X] = significant (q<0.05):                 {ko_ge_sig_n}")
print(f"        → used in: 'KOCAK GE differential: [X] significant genes'")
print(f"  [X] = higher in ADRN:                       {ko_ge_adrn_n}")
print(f"  [X] = higher in MES:                        {ko_ge_mes_n}")

# ─────────────────────────────────────────────
# 4.3 GENE DEPENDENCY
# ─────────────────────────────────────────────

gd_adrn = pd.read_csv(os.path.join(RESULTS_DIR, "depmap_GD_ADRN.csv"))
gd_mes  = pd.read_csv(os.path.join(RESULTS_DIR, "depmap_GD_MES.csv"))

gd_adrn_sig = gd_adrn[(gd_adrn["q_value"] < Q_THRESH) & (gd_adrn["mean_ADRN"] < 0)]
gd_mes_sig  = gd_mes[(gd_mes["q_value"]  < Q_THRESH) & (gd_mes["mean_MES"]   < 0)]
gd_adrn_str = gd_adrn[gd_adrn["mean_ADRN"] < -1]
gd_mes_str  = gd_mes[gd_mes["mean_MES"]   < -1]

print("\n── 4.3 Gene Dependency ─────────────────────────────────────────────")
print(f"  [X] = ADRN significantly essential:         {len(gd_adrn_sig)}")
print(f"        → used in: 'Gene dependency ADRN: [X] significantly essential TFs'")
print(f"  [X] = ADRN strongly essential (mean < -1):  {len(gd_adrn_str)}")
print(f"  ADRN top 3:                                 {gd_adrn_sig.head(3)['gene'].tolist()}")
print(f"  [X] = MES significantly essential:          {len(gd_mes_sig)}")
print(f"        → used in: 'Gene dependency MES: [X] significantly essential TFs'")
print(f"  [X] = MES strongly essential (mean < -1):   {len(gd_mes_str)}")
print(f"  MES top 3:                                  {gd_mes_sig.head(3)['gene'].tolist()}")

mycn_adrn = gd_adrn[gd_adrn["gene"] == "MYCN"]
if len(mycn_adrn) > 0:
    print(f"  MYCN ADRN mean dependency:                  {mycn_adrn['mean_ADRN'].values[0]:.3f}")
    print(f"        → used in: 'MYCN (mean = [X])'")
    print(f"  MYCN ADRN q-value:                          {mycn_adrn['q_value'].values[0]:.6f}")

# ─────────────────────────────────────────────
# 4.4 SURVIVAL
# ─────────────────────────────────────────────

surv = pd.read_csv(os.path.join(RESULTS_DIR, "kocak_TF_survival.csv"))

efs_sig  = surv[surv["EFS_significant"] == "Yes"]
os_sig   = surv[surv["OS_significant"]  == "Yes"]
both_sig = surv[surv["both_significant"] == "Yes"]
efs_poor = efs_sig[efs_sig["EFS_HR"] > 1]
efs_good = efs_sig[efs_sig["EFS_HR"] < 1]
top_efs  = efs_poor.nlargest(5, "EFS_HR")[["TF", "EFS_HR", "OS_HR"]].values.tolist()

top_surv_tf = efs_poor.nlargest(1, "EFS_HR")["TF"].values[0]

print("\n── 4.4 Survival Analysis ───────────────────────────────────────────")
print(f"  [X] = significant EFS (q<0.05):             {len(efs_sig)}")
print(f"        → used in: 'Survival: [X] TFs significant for EFS'")
print(f"  [X] = significant OS  (q<0.05):             {len(os_sig)}")
print(f"        → used in: 'Survival: [X] with OS'")
print(f"  [X] = significant both:                     {len(both_sig)}")
print(f"        → used in: 'Survival: [X] for both'")
print(f"  [X] = poor outcome EFS (HR>1):              {len(efs_poor)}")
print(f"  [X] = good outcome EFS (HR<1):              {len(efs_good)}")
print(f"  Top 5 by EFS HR:")
for tf, efs_hr, os_hr in top_efs:
    print(f"    {tf:<12}  EFS HR = {efs_hr:.2f},  OS HR = {os_hr:.2f}")
print(f"  Top TF for KM plot:                         {top_surv_tf}")
print(f"        → used in: '[Insert Figure 7: KM — {top_surv_tf}]'")

