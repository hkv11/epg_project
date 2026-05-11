import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# CONFIGURATION
# Run from project root: python plotting/tf_heatmap.py
# ─────────────────────────────────────────────

RESULTS_DIR  = "results"
OUTPUT_DIR   = "results/plots"
TOP_N        = 25

import os
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ─────────────────────────────────────────────
# LOAD RANKING FILES
# ─────────────────────────────────────────────

print("Loading ranking files...")

adrn_rank = pd.read_csv(os.path.join(RESULTS_DIR, "ranking_ADRN.csv"))
mes_rank  = pd.read_csv(os.path.join(RESULTS_DIR, "ranking_MES.csv"))

METRIC_COLS = ["g1_dm_diff", "g2_gd", "g3_ko_diff", "g4_survival"]

METRIC_LABELS = {
    "g1_dm_diff":  "DepMap\nTF+GE",
    "g2_gd":       "DepMap\nEssent.",      # abbreviated
    "g3_ko_diff":  "KOCAK\nTF+GE",
    "g4_survival": "Survival\nEFS+OS",
}

# ─────────────────────────────────────────────
# PREPARE MATRICES
# ─────────────────────────────────────────────

def prepare_matrix(rank_df, top_n):
    top    = rank_df.head(top_n).copy()
    matrix = top[METRIC_COLS].values.astype(float)
    tfs    = top["TF"].tolist()
    scores = top["score"].tolist()
    return matrix, tfs, scores

adrn_mat, adrn_tfs, adrn_scores = prepare_matrix(adrn_rank, TOP_N)
mes_mat,  mes_tfs,  mes_scores  = prepare_matrix(mes_rank,  TOP_N)

# ─────────────────────────────────────────────
# COLOUR MAPS
# ─────────────────────────────────────────────

cmap_adrn = LinearSegmentedColormap.from_list(
    "adrn", ["#ffffff", "#c6dbef", "#4292c6", "#084594"], N=256)
cmap_mes  = LinearSegmentedColormap.from_list(
    "mes",  ["#ffffff", "#fcbba1", "#cb181d", "#67000d"], N=256)

def apply_zero_mask(ax, matrix, cmap, vmin=0, vmax=1):
    n_rows, n_cols = matrix.shape
    ax.imshow(matrix, aspect="auto", cmap=cmap,
              vmin=vmin, vmax=vmax, interpolation="nearest")
    zero_mask = np.zeros((*matrix.shape, 4))
    grey_rgba = np.array([0.91, 0.91, 0.91, 1.0])
    for r in range(n_rows):
        for c in range(n_cols):
            if matrix[r, c] == 0.0:
                zero_mask[r, c] = grey_rgba
    ax.imshow(zero_mask, aspect="auto", interpolation="nearest")

# ─────────────────────────────────────────────
# FIGURE LAYOUT
# ─────────────────────────────────────────────

FIG_W = 13
FIG_H = max(TOP_N * 0.38 + 3, 12)

fig = plt.figure(figsize=(FIG_W, FIG_H), facecolor="white")

# 5 columns per side: label | heatmap | score bar | spacer | (repeat)
gs = gridspec.GridSpec(
    1, 7,
    figure=fig,
    left=0.01, right=0.95,
    top=0.94,  bottom=0.04,   # was top=0.89
    wspace=0.04,
    width_ratios=[0.15, 1, 0.12, 0.08, 0.15, 1, 0.12]
)

ax_adrn_lbl   = fig.add_subplot(gs[0])
ax_adrn_heat  = fig.add_subplot(gs[1])
ax_adrn_score = fig.add_subplot(gs[2])

ax_mes_lbl    = fig.add_subplot(gs[4])
ax_mes_heat   = fig.add_subplot(gs[5])
ax_mes_score  = fig.add_subplot(gs[6])

# ─────────────────────────────────────────────
# DRAW HEATMAPS
# ─────────────────────────────────────────────

def draw_heatmap(ax_lbl, ax_heat, ax_score,
                 matrix, tfs, scores,
                 cmap, score_colour, title, title_colour,
                 show_col_labels=True):
    n_rows, n_cols = matrix.shape

    # ── Heatmap ──
    apply_zero_mask(ax_heat, matrix, cmap)

    for r in range(n_rows):
        for c in range(n_cols):
            val = matrix[r, c]
            if val > 0.05:
                txt_col = "white" if val > 0.6 else "#333333"
                ax_heat.text(c, r, f"{val:.2f}", ha="center", va="center",
                             fontsize=6.2, color=txt_col, fontweight="normal")

    for x in np.arange(-0.5, n_cols, 1):
        ax_heat.axvline(x, color="white", linewidth=0.8)
    for y in np.arange(-0.5, n_rows, 1):
        ax_heat.axhline(y, color="white", linewidth=0.8)

    ax_heat.set_xlim(-0.5, n_cols - 0.5)
    ax_heat.set_ylim(n_rows - 0.5, -0.5)
    ax_heat.set_xticks([])
    ax_heat.set_yticks([])

    if show_col_labels:
        col_labels = [METRIC_LABELS[m] for m in METRIC_COLS]
        ax_heat.set_xticks(range(n_cols))
        ax_heat.set_xticklabels(col_labels, fontsize=7.5, color="#333333",
                                ha="center", va="bottom")
        ax_heat.xaxis.set_ticks_position("top")
        ax_heat.xaxis.set_label_position("top")
        ax_heat.tick_params(axis="x", which="both", length=0, pad=4)

    # ── TF name labels ──
    ax_lbl.set_xlim(0, 1)
    ax_lbl.set_ylim(n_rows - 0.5, -0.5)
    ax_lbl.axis("off")

    for r, tf in enumerate(tfs):
        ax_lbl.text(0.95, r, tf, ha="right", va="center",
                    fontsize=8.5, color="#111111", fontweight="bold")

    # ── Score bar ──
    max_score = 4.0
    ax_score.set_xlim(0, max_score)
    ax_score.set_ylim(n_rows - 0.5, -0.5)
    ax_score.axis("off")

    for r, sc in enumerate(scores):
        # filled bar
        ax_score.barh(r, sc, height=0.65, left=0,
                      color=score_colour, alpha=0.75)
        # score label
        ax_score.text(sc + 0.05, r, f"{sc:.2f}", ha="left", va="center",
                      fontsize=6.5, color="#333333")

    # column header for score bar
    if show_col_labels:
        ax_score.text(max_score / 2, -1.1, "Score", ha="center", va="bottom",
                      fontsize=7.5, color="#333333")

    # ── Panel title (on heatmap ax) ──
    ax_heat.set_title(title, fontsize=13, fontweight="bold",
                      color=title_colour, pad=36 if show_col_labels else 8)


# Draw panels
draw_heatmap(
    ax_adrn_lbl, ax_adrn_heat, ax_adrn_score,
    adrn_mat, adrn_tfs, adrn_scores,
    cmap=cmap_adrn,
    score_colour="#4292c6",
    title="ADRN Candidates",
    title_colour="#084594",
    show_col_labels=True
)

draw_heatmap(
    ax_mes_lbl, ax_mes_heat, ax_mes_score,
    mes_mat, mes_tfs, mes_scores,
    cmap=cmap_mes,
    score_colour="#cb181d",
    title="MES Candidates",
    title_colour="#a50f15",
    show_col_labels=True
)

# ─────────────────────────────────────────────
# SAVE
# ─────────────────────────────────────────────

out_path = os.path.join(OUTPUT_DIR, "tf_evidence_heatmap.png")
fig.savefig(out_path, dpi=180, bbox_inches="tight", facecolor="white")
print(f"Heatmap saved to {out_path}")
plt.close()