import pandas as pd
import numpy as np
import os

# ─────────────────────────────────────────────
# CONFIGURATION
# Run from project root: python ranking/tf_scoring.py
# ─────────────────────────────────────────────

RESULTS_DIR = "results"
OUTPUT_DIR  = "results"

# ─────────────────────────────────────────────
# LOAD RESULT FILES
# ─────────────────────────────────────────────

print("Loading result files...")

depmap_tf  = pd.read_csv(os.path.join(RESULTS_DIR, "depmap_TF_differential.csv"))
depmap_ge  = pd.read_csv(os.path.join(RESULTS_DIR, "depmap_GE_differential.csv"))
depmap_gd  = pd.read_csv(os.path.join(RESULTS_DIR, "depmap_GD_differential.csv"))
kocak_tf   = pd.read_csv(os.path.join(RESULTS_DIR, "kocak_TF_differential.csv"))
kocak_ge   = pd.read_csv(os.path.join(RESULTS_DIR, "kocak_GE_differential.csv"))
kocak_surv = pd.read_csv(os.path.join(RESULTS_DIR, "kocak_TF_survival.csv"))

depmap_ge = depmap_ge.rename(columns={"gene": "TF"})
depmap_gd = depmap_gd.rename(columns={"gene": "TF"})
kocak_ge  = kocak_ge.rename(columns={"gene": "TF"})

print(f"  depmap_TF_differential : {len(depmap_tf):>5} TFs")
print(f"  depmap_GE_differential : {len(depmap_ge):>5} TFs")
print(f"  depmap_GD_differential : {len(depmap_gd):>5} TFs")
print(f"  kocak_TF_differential  : {len(kocak_tf):>5} TFs")
print(f"  kocak_GE_differential  : {len(kocak_ge):>5} TFs")
print(f"  kocak_TF_survival      : {len(kocak_surv):>5} TFs")

# ─────────────────────────────────────────────
# BUILD UNION TF SET
# ─────────────────────────────────────────────

all_tfs = set()
for df in [depmap_tf, depmap_ge, depmap_gd, kocak_tf, kocak_ge, kocak_surv]:
    all_tfs.update(df["TF"].tolist())
all_tfs = sorted(all_tfs)
print(f"\nTotal unique TFs across all files: {len(all_tfs)}")

# ─────────────────────────────────────────────
# HELPER: MIN-MAX NORMALISATION (NaN-aware)
# NaN values are excluded from min/max calculation
# and filled with 0.5 (neutral) after normalisation
# ─────────────────────────────────────────────

def minmax(series):
    mn = series.min(skipna=True)
    mx = series.max(skipna=True)
    if mx == mn:
        return pd.Series(0.5, index=series.index)
    normalised = (series - mn) / (mx - mn)
    return normalised  # NaNs preserved here, filled separately per group

# ─────────────────────────────────────────────
# INDEX ALL FILES BY TF
# ─────────────────────────────────────────────

dm_tf_idx   = depmap_tf.set_index("TF")
dm_ge_idx   = depmap_ge.set_index("TF")
dm_gd_idx   = depmap_gd.set_index("TF")
ko_tf_idx   = kocak_tf.set_index("TF")
ko_ge_idx   = kocak_ge.set_index("TF")
ko_surv_idx = kocak_surv.set_index("TF")

# ─────────────────────────────────────────────
# SCORING FRAMEWORK — 4 GROUPS, NO SIGNIFICANCE GATES
#
# All metrics use raw signed values — no q < 0.05 filtering.
# Penalisation is built into directionality via sign encoding:
#   strong evidence for target state  → near 1.0 after minmax
#   no evidence either way            → near 0.5
#   evidence against target state     → near 0.0 (penalised)
#
# Group 1: DepMap TF + GE differential (averaged)
# Group 2: DepMap GD differential
# Group 3: KOCAK TF + GE differential (averaged)
# Group 4: Survival HR (EFS + OS averaged)
#
# Survival logic (same for ADRN and MES):
#   HR > 1 → poor prognosis → therapeutically relevant → high score
#   HR < 1 → good prognosis → not a target → low score
#   HR ≈ 1 → no effect → neutral → mid score (~0.5)
#   NaN    → no data → neutral → filled with 0.5 after normalisation
#
# Sign encoding per state:
#   ADRN differential metrics: pass raw difference (positive = ADRN > MES)
#   MES differential metrics:  negate difference  (positive = MES > ADRN)
#   ADRN GD:  negate gd difference (negative diff = ADRN essential → high)
#   MES GD:   keep gd difference   (positive diff = MES essential → high)
#   Survival: same raw HR for both states (HR > 1 = poor prognosis = target)
# ─────────────────────────────────────────────

records = []

for tf in all_tfs:

    # ── Group 1: DepMap TF + GE ───────────────────────────────────────
    dm_tf_diff = dm_tf_idx.loc[tf, "difference"] if tf in dm_tf_idx.index else 0.0
    dm_ge_diff = dm_ge_idx.loc[tf, "difference"] if tf in dm_ge_idx.index else 0.0

    # ── Group 2: DepMap GD ───────────────────────────────────────────
    # difference = mean_ADRN - mean_MES
    # negative = ADRN more essential; positive = MES more essential
    dm_gd_diff = dm_gd_idx.loc[tf, "difference"] if tf in dm_gd_idx.index else 0.0

    # ── Group 3: KOCAK TF + GE ───────────────────────────────────────
    ko_tf_diff = ko_tf_idx.loc[tf, "difference"] if tf in ko_tf_idx.index else 0.0
    ko_ge_diff = ko_ge_idx.loc[tf, "difference"] if tf in ko_ge_idx.index else 0.0

    # ── Group 4: Survival HR ─────────────────────────────────────────
    # HR > 1 = poor prognosis = therapeutically relevant
    # HR < 1 = good prognosis = not a target
    # Missing = NaN → filled with 0.5 (neutral) after normalisation
    if tf in ko_surv_idx.index:
        s      = ko_surv_idx.loc[tf]
        efs_hr = s["EFS_HR"] if pd.notna(s["EFS_HR"]) else np.nan
        os_hr  = s["OS_HR"]  if pd.notna(s["OS_HR"])  else np.nan
    else:
        efs_hr = np.nan
        os_hr  = np.nan

    records.append({
        "TF": tf,
        # ADRN raw inputs
        "ADRN_dm_tf":    dm_tf_diff,
        "ADRN_dm_ge":    dm_ge_diff,
        "ADRN_dm_gd":   -dm_gd_diff,   # flip: ADRN-essential = negative diff
        "ADRN_ko_tf":    ko_tf_diff,
        "ADRN_ko_ge":    ko_ge_diff,
        "ADRN_surv_efs": efs_hr,
        "ADRN_surv_os":  os_hr,
        # MES raw inputs
        "MES_dm_tf":    -dm_tf_diff,   # flip: MES-higher = negative diff
        "MES_dm_ge":    -dm_ge_diff,
        "MES_dm_gd":     dm_gd_diff,   # MES-essential = positive diff
        "MES_ko_tf":    -ko_tf_diff,
        "MES_ko_ge":    -ko_ge_diff,
        "MES_surv_efs":  efs_hr,        # same HR for both — poor prognosis = target
        "MES_surv_os":   os_hr,
    })

raw_df = pd.DataFrame(records).set_index("TF")

# ─────────────────────────────────────────────
# NORMALISE: min-max per column across all TFs
# NaN values excluded from min/max, then filled
# with 0.5 (neutral) after normalisation
# ─────────────────────────────────────────────

print("\nNormalising metrics...")

norm_df = pd.DataFrame(index=raw_df.index)

for col in raw_df.columns:
    normalised = minmax(raw_df[col])
    # Fill NaN with 0.5 only for survival columns (others have 0.0 defaults)
    if "surv" in col:
        normalised = normalised.fillna(0.5)
    norm_df[col + "_norm"] = normalised

# ─────────────────────────────────────────────
# COMPUTE GROUP SCORES
# ─────────────────────────────────────────────

for state in ["ADRN", "MES"]:

    # Group 1: DepMap TF + GE averaged
    norm_df[f"{state}_g1"] = norm_df[[
        f"{state}_dm_tf_norm",
        f"{state}_dm_ge_norm"
    ]].mean(axis=1)

    # Group 2: GD directly
    norm_df[f"{state}_g2"] = norm_df[f"{state}_dm_gd_norm"]

    # Group 3: KOCAK TF + GE averaged
    norm_df[f"{state}_g3"] = norm_df[[
        f"{state}_ko_tf_norm",
        f"{state}_ko_ge_norm"
    ]].mean(axis=1)

    # Group 4: EFS + OS HR averaged
    # Both already NaN-filled with 0.5 so mean is safe
    norm_df[f"{state}_g4"] = norm_df[[
        f"{state}_surv_efs_norm",
        f"{state}_surv_os_norm"
    ]].mean(axis=1)

    # Final score (max = 4.0)
    norm_df[f"{state}_score"] = (
        norm_df[f"{state}_g1"] +
        norm_df[f"{state}_g2"] +
        norm_df[f"{state}_g3"] +
        norm_df[f"{state}_g4"]
    ).round(4)

    # Breadth: count raw components with positive evidence
    raw_cols = [c for c in raw_df.columns if c.startswith(state)]
    norm_df[f"{state}_n_metrics"] = (raw_df[raw_cols] > 0).sum(axis=1)

# ─────────────────────────────────────────────
# BUILD OUTPUT TABLES
# ─────────────────────────────────────────────

def build_output(state, norm_df):
    out = norm_df[[
        f"{state}_score", f"{state}_n_metrics",
        f"{state}_g1", f"{state}_g2",
        f"{state}_g3", f"{state}_g4"
    ]].copy()
    out.columns = ["score", "n_metrics", "g1_dm_diff", "g2_gd", "g3_ko_diff", "g4_survival"]
    out = out.sort_values(["score", "n_metrics"], ascending=False)
    out.index.name = "TF"
    return out.reset_index()

adrn_out = build_output("ADRN", norm_df)
mes_out  = build_output("MES",  norm_df)

# ─────────────────────────────────────────────
# PRINT SUMMARIES
# ─────────────────────────────────────────────

print("\n── TOP 20 ADRN CANDIDATES ──────────────────────────────")
print(adrn_out[["TF", "score", "n_metrics",
                 "g1_dm_diff", "g2_gd", "g3_ko_diff", "g4_survival"]
               ].head(20).to_string(index=False))

print("\n── TOP 20 MES CANDIDATES ───────────────────────────────")
print(mes_out[["TF", "score", "n_metrics",
               "g1_dm_diff", "g2_gd", "g3_ko_diff", "g4_survival"]
              ].head(20).to_string(index=False))

adrn_top = set(adrn_out.head(20)["TF"])
mes_top  = set(mes_out.head(20)["TF"])
overlap  = adrn_top & mes_top
if overlap:
    print(f"\nTFs in BOTH top-20 lists: {sorted(overlap)}")
else:
    print("\nNo overlap between top-20 ADRN and MES lists.")

# ─────────────────────────────────────────────
# SAVE
# ─────────────────────────────────────────────

adrn_path = os.path.join(OUTPUT_DIR, "ranking_ADRN.csv")
mes_path  = os.path.join(OUTPUT_DIR, "ranking_MES.csv")

adrn_out.to_csv(adrn_path, index=False)
mes_out.to_csv(mes_path,   index=False)

print(f"\nResults saved to:")
print(f"  {adrn_path}")
print(f"  {mes_path}")