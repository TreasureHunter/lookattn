#!/usr/bin/env python
"""plot_heads_qwen25_banded.py -- Qwen2.5-VL-7B layer-20 head figures, with the
alpha top-25% eligibility band marked by hatching.

Two figures, one per metric, each a dual-axis combo like obs_heads_plot.py:
  bars (left axis)  = alpha_h   or   rho_h      for all 28 heads of layer 20
  line (right axis) = per-head token-selection accuracy (top-20% budget)

Head selection rule (obs_heads.py:389):
  band       = the 7 heads with the largest alpha_h   (H // 4 = 25% of 28 heads)
  h_alpha    = argmax alpha_h            (unconstrained)                  -> head 21
  h_rho_band = argmax rho_h within band  (the head we actually use)        -> head 8
  h_rho      = argmax rho_h unconstrained -> head 0, NOT selected because it
               falls outside the band (its alpha_h is only 0.128) and its own
               accuracy is 69.5% (rank 26/28).
Heads outside the band are drawn hatched in BOTH figures so the reader can see at a
glance which heads were ineligible for the rho-band pick.

All numbers below are hard-coded copies of
/ossfs/workspace/visionselector_artifacts/obs_qwen25/heads/heads.json
(model qwen25, layer 20, n_samples 600, n_heads 28, strict_q_fallbacks 0).
ACC is rounded to 6 decimals; ALPHA/RHO are verbatim.

Usage: plot_heads_qwen25_banded.py [out_dir]   (default: the heads dir above)
Writes fig_heads_qwen25_banded.{pdf,png} and fig_heads_rho_qwen25_banded.{pdf,png}
"""
import sys
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np

plt.rcParams.update({'font.family': 'sans-serif', 'font.sans-serif': ['DejaVu Sans'],
                     'font.size': 13, 'svg.fonttype': 'none', 'pdf.fonttype': 42})

OUT_DIR = sys.argv[1] if len(sys.argv) > 1 else \
    "/ossfs/workspace/visionselector_artifacts/obs_qwen25/heads"

LAYER = 20
N_SAMPLES = 600

# alpha_h -- per-head visual attention mass onto image tokens (last-token query row)
ALPHA = [0.12844045856967568, 0.026761309377810297, 0.06164279628389825,
         0.046149841338434874, 0.13497540945963313, 0.25501614600856554,
         0.16958391134316722, 0.14736441529045502, 0.45123245571856385,
         0.17912504756939598, 0.36942333715346953, 0.07470364501466975,
         0.08358916682346414, 0.03460468829104987, 0.2933942689994971,
         0.20467571506897608, 0.1819301365308153, 0.40577905611474613,
         0.04677144865163086, 0.27075579862420757, 0.05887709475743274,
         0.5576063798740506, 0.1533898997005114, 0.2571567538442711,
         0.31508638101940356, 0.17105241197627039, 0.49444003274664283,
         0.45504602395308513]

# rho_h -- generation-to-question attention ratio (the head-selection constraint)
RHO = [7.177777905528347, 2.9556863337596795, 1.5620512119217607,
       3.0130075480534275, 3.180721106537332, 4.673331363094874,
       2.3325727881587337, 5.771337080279677, 6.840788100394626,
       6.121764694736191, 3.578508902620478, 5.498973475389165,
       3.3441321631944096, 1.754427461283451, 3.1085596571062375,
       1.8705260258563088, 2.3965129415807103, 1.7754016762015417,
       1.3001055532396897, 1.852642826750319, 1.1611826588893421,
       2.4017750951524386, 4.83485521644276, 1.9716330024471647,
       2.581274044913791, 2.8625861866404714, 3.5242295896543423,
       3.6397698922067403]

# per-head accuracy (%) when the top-20% of visual tokens are selected by that head,
# 600 samples, answers scored with the shared relaxed matcher
ACC = [69.5, 73.5, 81.666667, 80.166667, 72.166667, 81.666667, 82.666667,
       85.5, 82.8, 80.333333, 83.333333, 77.166667, 81.5, 82.833333,
       81.666667, 82.666667, 84.0, 82.333333, 82.333333, 83.833333, 80.5,
       84.5, 82.166667, 83.333333, 81.5, 67.666667, 80.0, 82.833333]
# two-head teacher accuracies from the same run (for reference only):
PAIR_ACC = {"pair:21,0": 84.33333333333334, "pair:21,8": 84.5}

alpha = np.array(ALPHA, float)
rho = np.array(RHO, float)
acc = np.array(ACC, float)
H = len(alpha)

# --- reproduce the selection rule exactly -------------------------------------
band = sorted(range(H), key=lambda h: -alpha[h])[:H // 4]      # alpha top-25%
in_band = np.zeros(H, bool)
in_band[band] = True
h_alpha = int(np.argmax(alpha))
h_rho = int(np.argmax(rho))
h_rho_band = max(band, key=lambda h: rho[h])


def spearman(x, y):
    def ranks(v):
        order = np.argsort(v)
        r = np.empty(len(v), float)
        r[order] = np.arange(len(v))
        return r
    rx, ry = ranks(x), ranks(y)
    return float(np.corrcoef(rx, ry)[0, 1])


SP_A = spearman(alpha, acc)
SP_R = spearman(rho, acc)

# --- palette (shared with obs_heads_plot.py) ----------------------------------
BAR, BAR_HI, LINE = "#9eb9d9", "#375b89", "#d1495b"   # blues follow figure03
INEL_FILL, INEL_EDGE = "#e9eef6", "#8d97a3"
HATCH = "///"
ARG_FILL, ARG_EDGE, ARG_HATCH = "#f7e4e6", "#b23a48", "xxx"
GRID, SPINE, INK, MUTED = "#e4e4e2", "#9a9a96", "#111111", "#5c5c58"
XLABEL = f"Head index (all {H} heads)"
SYM = {"alpha": (r"$\alpha_h$", "Visual attention mass"),
       "rho": (r"$\rho_h$", "Generation-to-question ratio")}


def draw(metric, out_stem, argmax_marker=None):
    """argmax_marker: head index of the unconstrained argmax of THIS metric, drawn with
    its own hatch to show it lost to the banded pick (used for rho -> head 0)."""
    vals = alpha if metric == "alpha" else rho
    sym, name = SYM[metric]
    sel = h_alpha if metric == "alpha" else h_rho_band

    fig, ax = plt.subplots(figsize=(7.0, 3.1), dpi=300)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    x = np.arange(1, H + 1)                           # heads numbered 1..28

    for h in range(H):
        if argmax_marker is not None and h == argmax_marker:
            bar = ax.bar(h + 1, vals[h], width=0.72, color=ARG_FILL, edgecolor=ARG_EDGE,
                         hatch=ARG_HATCH, linewidth=0.7, zorder=2)
        elif metric == "rho" and not in_band[h]:
            bar = ax.bar(h + 1, vals[h], width=0.72, color=INEL_FILL, edgecolor=INEL_EDGE,
                         hatch=HATCH, linewidth=0.6, zorder=2)
        else:
            bar = ax.bar(h + 1, vals[h], width=0.72, color=BAR, edgecolor="none",
                         linewidth=0.0, zorder=2)
        if h == sel:                                  # the head we actually use
            rect = bar[0]
            rect.set_color(BAR_HI)
            rect.set_edgecolor("#0d366b")
            rect.set_hatch(None)
            rect.set_linewidth(0.9)

    headroom = 1.30 if metric == "alpha" else 1.45
    ax.set_ylim(0, float(vals.max()) * headroom)
    ax.set_ylabel(f"{name} {sym}", fontsize=9, color=INK)
    ax.set_xlabel(XLABEL, fontsize=9, color=INK)
    ax.set_xticks(x[::2])
    ax.tick_params(labelsize=8, colors=INK, length=2.4, width=0.7)
    ax.grid(axis="y", color=GRID, linewidth=0.6, zorder=0)
    ax.set_axisbelow(True)

    ax2 = ax.twinx()
    ax2.plot(x, acc, color=LINE, linewidth=1.1, marker="o", markersize=3.0,
             markerfacecolor=LINE, markeredgecolor="white", markeredgewidth=0.4,
             zorder=4)
    alo, ahi = acc.min(), acc.max()
    pad = (ahi - alo) * 0.30
    ax2.set_ylim(alo - pad, ahi + pad)
    ax2.set_ylabel("Accuracy (%)", fontsize=9, color=INK)
    ax2.tick_params(labelsize=8, colors=INK, length=2.4, width=0.7)
    ax2.grid(False)

    for axx in (ax, ax2):
        for side, sp in axx.spines.items():
            sp.set_color(SPINE)
            sp.set_linewidth(0.7)
            sp.set_visible(side in ("left", "right", "bottom"))

    line_handle = plt.Line2D([], [], color=LINE, linewidth=1.1, marker="o", markersize=3.0,
                             markerfacecolor=LINE, markeredgecolor="white", label="Accuracy")
    if metric == "alpha":
        handles = [Patch(facecolor=BAR, edgecolor="none",
                         label=r"Visual attention mass $\alpha_h$"),
                   line_handle]
    else:
        handles = [Patch(facecolor=BAR, edgecolor="none",
                         label=r"$\rho_h$ of top 25% $\alpha_h$ heads"),
                   Patch(facecolor=INEL_FILL, edgecolor=INEL_EDGE, hatch=HATCH,
                         label=r"$\rho_h$ of the remaining heads"),
                   Patch(facecolor=ARG_FILL, edgecolor=ARG_EDGE, hatch=ARG_HATCH,
                         label=r"Head with max $\rho_h$, rejected"),
                   line_handle]
    ax.legend(handles=handles, fontsize=7.0, frameon=False, loc="upper left",
              handlelength=2.0, borderpad=0.25, labelspacing=0.28)

    fig.tight_layout()
    fig.savefig(f"{out_stem}.pdf", bbox_inches="tight", pad_inches=.03)
    fig.savefig(f"{out_stem}.png", dpi=300, bbox_inches="tight", pad_inches=.03)
    plt.close(fig)
    print(f"wrote {out_stem}.pdf / .png   (metric={metric}, selected head {sel})")


os.makedirs(OUT_DIR, exist_ok=True)

# figure 1: alpha_h (the band-defining metric)
draw("alpha", f"{OUT_DIR}/fig_heads_qwen25_banded")

# figure 2: rho_h -- flag that the unconstrained argmax (head 0) is outside the band
draw("rho", f"{OUT_DIR}/fig_heads_rho_qwen25_banded", argmax_marker=h_rho)

print()
print(f"alpha top-25% band (eligible)      : {band}")
print(f"h_alpha  = argmax alpha_h          : {h_alpha}  (alpha {alpha[h_alpha]:.4f}, acc {acc[h_alpha]:.2f})")
print(f"h_rho    = argmax rho_h            : {h_rho}  (rho {rho[h_rho]:.3f}, acc {acc[h_rho]:.2f}, NOT eligible)")
print(f"h_rho_band = argmax rho within band: {h_rho_band}  (rho {rho[h_rho_band]:.3f}, "
      f"alpha {alpha[h_rho_band]:.4f}, acc {acc[h_rho_band]:.2f})")
print(f"Spearman(alpha, acc) = {SP_A:+.4f}   Spearman(rho, acc) = {SP_R:+.4f}")
print(f"teacher pair accuracies: {PAIR_ACC}")
