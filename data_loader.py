import pandas as pd

# ─────────────────────────────────────────────
# HELPER
# ─────────────────────────────────────────────

def rename_index_col(df, new_name="sample_id"):
    """Rename the unnamed first column (row index) to a meaningful name."""
    df = df.rename(columns={"Unnamed: 0": new_name})
    return df


# ─────────────────────────────────────────────
# LOAD RAW FILES
# ─────────────────────────────────────────────

def load_raw():
    depmap_meta     = pd.read_csv("data/depmap_meta.txt",          sep="\t")
    depmap_ge       = pd.read_csv("data/depmap_GE_scaled.txt",     sep="\t")
    depmap_tf       = pd.read_csv("data/depmap_TF Activity.txt",   sep="\t")
    depmap_gd       = pd.read_csv("data/depmap_GD.txt",            sep="\t")

    kocak_meta      = pd.read_csv("data/KOCAK_meta.txt",           sep="\t")
    kocak_ge        = pd.read_csv("data/KOCAK_GE_scaled.txt",      sep="\t")
    kocak_tf        = pd.read_csv("data/KOCAK_TF Activity.txt",    sep="\t")
    kocak_survival  = pd.read_csv("data/KOCAK_survival.txt",       sep="\t")

    # Rename unnamed index columns
    depmap_ge      = rename_index_col(depmap_ge,      "condition")
    depmap_tf      = rename_index_col(depmap_tf,      "condition")
    depmap_gd      = rename_index_col(depmap_gd,      "condition")
    kocak_ge       = rename_index_col(kocak_ge,       "condition")
    kocak_tf       = rename_index_col(kocak_tf,       "condition")

    return (depmap_meta, depmap_ge, depmap_tf, depmap_gd,
            kocak_meta, kocak_ge, kocak_tf, kocak_survival)


# ─────────────────────────────────────────────
# FILTER GE AND GD TO TF COLUMNS ONLY
# ─────────────────────────────────────────────

def filter_to_tf_columns(depmap_ge, depmap_tf, depmap_gd, kocak_ge, kocak_tf):
    """
    Filter GE and GD files to only keep columns that are also TFs.
    TF names are taken from the TF activity files (depmap and KOCAK).
    This reduces GE/GD from ~18000 genes down to only TF-relevant genes.
    """

    # Get TF names from both TF activity files (excluding condition column)
    depmap_tf_names = set(depmap_tf.columns) - {"condition"}
    kocak_tf_names  = set(kocak_tf.columns)  - {"condition"}

    # Union of both — all known TF names across datasets
    all_tf_names = depmap_tf_names | kocak_tf_names

    # Filter GE and GD — keep condition + any column that is a TF
    depmap_ge_tf_cols = ["condition"] + [c for c in depmap_ge.columns if c in all_tf_names]
    depmap_gd_tf_cols = ["condition"] + [c for c in depmap_gd.columns if c in all_tf_names]
    kocak_ge_tf_cols  = ["condition"] + [c for c in kocak_ge.columns  if c in all_tf_names]

    depmap_ge_filtered = depmap_ge[depmap_ge_tf_cols]
    depmap_gd_filtered = depmap_gd[depmap_gd_tf_cols]
    kocak_ge_filtered  = kocak_ge[kocak_ge_tf_cols]

    print(f"[Filter] depmap_GE: {len(depmap_ge.columns)-1} genes → {len(depmap_ge_tf_cols)-1} TFs retained")
    print(f"[Filter] depmap_GD: {len(depmap_gd.columns)-1} genes → {len(depmap_gd_tf_cols)-1} TFs retained")
    print(f"[Filter] kocak_GE:  {len(kocak_ge.columns)-1} genes → {len(kocak_ge_tf_cols)-1} TFs retained")

    return depmap_ge_filtered, depmap_gd_filtered, kocak_ge_filtered


# ─────────────────────────────────────────────
# CLEAN DEPMAP
# ─────────────────────────────────────────────

def clean_depmap(depmap_meta, depmap_ge, depmap_tf, depmap_gd):
    """
    1. Drop samples with no AM_class (unclassified, useless for ADRN/MES analysis)
    2. Use cleaned meta as base
    3. Inner join GE, TF, and GD onto meta
    """

    meta_clean = depmap_meta.dropna(subset=["AM_class"]).copy()
    print(f"\n[DepMap] Meta after dropping NaN AM_class: {meta_clean.shape[0]} samples")

    depmap_merged_ge = meta_clean.merge(depmap_ge, on="condition", how="inner")
    print(f"[DepMap] After merging GE: {depmap_merged_ge.shape[0]} samples, {depmap_merged_ge.shape[1]} columns")

    depmap_merged_tf = meta_clean.merge(depmap_tf, on="condition", how="inner")
    print(f"[DepMap] After merging TF: {depmap_merged_tf.shape[0]} samples, {depmap_merged_tf.shape[1]} columns")

    depmap_merged_gd = meta_clean.merge(depmap_gd, on="condition", how="inner")
    print(f"[DepMap] After merging GD: {depmap_merged_gd.shape[0]} samples, {depmap_merged_gd.shape[1]} columns")

    return depmap_merged_ge, depmap_merged_tf, depmap_merged_gd


# ─────────────────────────────────────────────
# CLEAN KOCAK
# ─────────────────────────────────────────────

def clean_kocak(kocak_meta, kocak_ge, kocak_tf, kocak_survival):
    """
    1. Drop samples with no AM_class
    2. Use cleaned meta as base
    3. Inner join GE, TF, and survival onto meta
    """

    meta_clean = kocak_meta.dropna(subset=["AM_class"]).copy()
    print(f"\n[KOCAK] Meta after dropping NaN AM_class: {meta_clean.shape[0]} samples")

    kocak_merged_ge = meta_clean.merge(kocak_ge, on="condition", how="inner")
    print(f"[KOCAK] After merging GE: {kocak_merged_ge.shape[0]} samples, {kocak_merged_ge.shape[1]} columns")

    kocak_merged_tf = meta_clean.merge(kocak_tf, on="condition", how="inner")
    print(f"[KOCAK] After merging TF: {kocak_merged_tf.shape[0]} samples, {kocak_merged_tf.shape[1]} columns")

    kocak_merged_survival = meta_clean.merge(kocak_survival, on="condition", how="inner")
    print(f"[KOCAK] After merging survival: {kocak_merged_survival.shape[0]} samples, {kocak_merged_survival.shape[1]} columns")

    return kocak_merged_ge, kocak_merged_tf, kocak_merged_survival


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def load_all():
    print("Loading raw files...")
    (depmap_meta, depmap_ge, depmap_tf, depmap_gd,
     kocak_meta, kocak_ge, kocak_tf, kocak_survival) = load_raw()

    print("\nFiltering GE and GD to TF columns only...")
    depmap_ge, depmap_gd, kocak_ge = filter_to_tf_columns(
        depmap_ge, depmap_tf, depmap_gd, kocak_ge, kocak_tf)

    print("\nCleaning DepMap...")
    depmap_merged_ge, depmap_merged_tf, depmap_merged_gd = clean_depmap(
        depmap_meta, depmap_ge, depmap_tf, depmap_gd)

    print("\nCleaning KOCAK...")
    kocak_merged_ge, kocak_merged_tf, kocak_merged_survival = clean_kocak(
        kocak_meta, kocak_ge, kocak_tf, kocak_survival)

    print("\nAll datasets loaded and cleaned successfully.")

    return {
        "depmap_ge":       depmap_merged_ge,
        "depmap_tf":       depmap_merged_tf,
        "depmap_gd":       depmap_merged_gd,
        "kocak_ge":        kocak_merged_ge,
        "kocak_tf":        kocak_merged_tf,
        "kocak_survival":  kocak_merged_survival,
    }


if __name__ == "__main__":
    datasets = load_all()