"""Generates every figure for the DUTS beamer deck as a standalone vector PDF.

Style follows the dataviz skill's reference palette (categorical order fixed,
status colors reserved for outcome states, sequential blue for pure magnitude,
thin marks, light recessive gridlines, direct labels instead of dense legends
where there are <=4 series).
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

HERE = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(HERE, "figures")
os.makedirs(FIG, exist_ok=True)
REPO = "/u6/bkassaie/DUTS"

# ---------------------------------------------------------------- palette ---
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
BASE = "#c3c2b7"
SURFACE = "#fcfcfb"

CAT1 = "#2a78d6"   # blue    -- slot 1
CAT2 = "#eb6834"   # orange  -- slot 2
CAT3 = "#1baf7a"   # aqua    -- slot 3
CAT4 = "#eda100"   # yellow  -- slot 4

GOOD = "#0ca30c"
WARNING = "#fab219"
SERIOUS = "#ec835a"
CRITICAL = "#d03b3b"

SEQ = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95"]

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
    "font.size": 11,
    "axes.edgecolor": BASE,
    "axes.labelcolor": INK2,
    "axes.titlecolor": INK,
    "text.color": INK,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
    "axes.grid": False,
    "svg.fonttype": "none",
})


def style_ax(ax, ygrid=True):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_color(BASE)
    ax.spines["bottom"].set_linewidth(0.8)
    ax.tick_params(length=0)
    if ygrid:
        ax.yaxis.grid(True, color=GRID, linewidth=0.8, zorder=0)
        ax.set_axisbelow(True)


def save(fig, name):
    fig.savefig(os.path.join(FIG, name + ".pdf"), bbox_inches="tight", pad_inches=0.06)
    plt.close(fig)
    print("wrote", name)


# =============================================================================
# SANTOS3 FOCUSED (solvable-only, alpha=2, k in {3,5})
# =============================================================================
s3 = pd.read_csv(os.path.join(REPO, "experiments/results/santos3_focused_k3_k5_alpha2.csv"))

# ---- Fig S1: cohort funnel (why we restrict) ----
funnel = {
    3: dict(total=48, insufficient=5, unreachable=0, retained=43),
    5: dict(total=48, insufficient=5, unreachable=0, retained=43),
}
fig, ax = plt.subplots(figsize=(6.6, 2.9))
ks = [3, 5]
y = np.arange(len(ks))
h = 0.5
retained = [funnel[k]["retained"] for k in ks]
insuff = [funnel[k]["insufficient"] for k in ks]
unreach = [funnel[k]["unreachable"] for k in ks]
ax.barh(y, retained, height=h, color=CAT1, label="Retained -- $\\tau$ reachable", zorder=3)
ax.barh(y, unreach, left=retained, height=h, color=WARNING, label="Excluded -- $\\tau$ unreachable", zorder=3)
left2 = [r + u for r, u in zip(retained, unreach)]
ax.barh(y, insuff, left=left2, height=h, color=MUTED, alpha=0.55, label="Excluded -- $|D|<k$ (retrieval)", zorder=3)
for i, k in enumerate(ks):
    ax.text(retained[i] / 2, i, str(retained[i]), ha="center", va="center",
            color="white", fontweight="bold", fontsize=12)
    ax.text(49.5, i, f"of 48", ha="left", va="center", color=MUTED, fontsize=9.5)
ax.set_yticks(y)
ax.set_yticklabels([f"$k={k}$" for k in ks], fontsize=12)
ax.set_xlim(0, 54)
ax.set_xlabel("santos3 queries")
style_ax(ax, ygrid=False)
ax.xaxis.grid(True, color=GRID, linewidth=0.8, zorder=0)
ax.set_axisbelow(True)
ax.legend(loc="lower right", frameon=False, fontsize=9, ncol=1,
          bbox_to_anchor=(1.0, -0.55))
save(fig, "s3_cohort_funnel")

# ---- Fig S2: quality -- F_P/F_R vs F* vs each stage's OWN attainable range ----
# F_P's ceiling is [F_min_pool, F_max_pool]: the range over subsets sized
# alpha*k (or |D| when C5-clamped) drawn from D -- Stage 1's actual selection
# cardinality and superset. F_R's ceiling is [F_min_P, F_max_P]: the range
# over k-subsets of P, NOT of D -- Stage 2 only ever sees P (R subseteq P),
# so a k-subset-of-D range (as brute-forced by fraction_reachability.py) is
# the wrong reference for it. Both computed by experiments/scoped_reachability.py.
scoped = pd.read_csv(os.path.join(REPO, "experiments/results/santos3_scoped_reachability_k3_k5_alpha2.csv"))
fig, ax = plt.subplots(figsize=(6.4, 3.4))
x = np.arange(len(ks))
w = 0.30
FP = [s3[s3.k == k].F_P.mean() for k in ks]
FR = [s3[s3.k == k].F_R.mean() for k in ks]
F_min_pool = [scoped[scoped.k == k].F_min_pool.mean() for k in ks]
F_max_pool = [scoped[scoped.k == k].F_max_pool.mean() for k in ks]
F_min_P = [scoped[scoped.k == k].F_min_P.mean() for k in ks]
F_max_P = [scoped[scoped.k == k].F_max_P.mean() for k in ks]
Fstar = 0.30
bandw = w + 0.08
band_patch = None
for i in range(len(ks)):
    band_patch = ax.add_patch(plt.Rectangle(
        (x[i] - w/2 - bandw/2, F_min_pool[i]), bandw, F_max_pool[i] - F_min_pool[i],
        facecolor=CRITICAL, alpha=0.14, zorder=1, linewidth=0))
    ax.add_patch(plt.Rectangle(
        (x[i] + w/2 - bandw/2, F_min_P[i]), bandw, F_max_P[i] - F_min_P[i],
        facecolor=CRITICAL, alpha=0.14, zorder=1, linewidth=0))
ax.bar(x - w/2, FP, width=w, color=CAT3, label="$F_P$ (Stage 1 pool)", zorder=3)
ax.bar(x + w/2, FR, width=w, color=CAT1, label="$F_R$ (result)", zorder=3)
ax.axhline(Fstar, color=INK, linewidth=1.3, linestyle="--", zorder=4, label="$F^\\star=0.30$ (target)")
for i, k in enumerate(ks):
    ax.text(x[i] - w/2, FP[i] + 0.018, f"{FP[i]:.2f}", ha="center", fontsize=9.5, color=INK2)
    ax.text(x[i] + w/2, FR[i] + 0.018, f"{FR[i]:.2f}", ha="center", fontsize=9.5, color=INK2)
band_patch.set_label("attainable range, scoped to each bar's own selection")
ax.set_xticks(x)
ax.set_xticklabels([f"$k={k}$" for k in ks], fontsize=12.5)
ax.set_xlim(-0.62, len(ks) - 0.38)
ax.set_ylim(0, 0.63)
ax.set_ylabel("proportion $F$")
style_ax(ax)
ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=3, frameon=False, fontsize=8.6,
          handletextpad=0.5, columnspacing=1.2)
save(fig, "s3_quality")

# ---- Fig S3: per-stage timing (stacked, ms) ----
fig, ax = plt.subplots(figsize=(6.4, 3.2))
stages = ["retrieval_time_s", "stage1_time_s", "scoring_time_s", "stage2_time_s"]
labels = ["Retrieval\n(index probe)", "Stage 1\n(Dinkelbach)", "Scoring\n(unionability $U$)", "Stage 2\n(ILP)"]
colors = [CAT1, CAT4, CAT3, CAT2]
means = {st: [1000 * s3[s3.k == k][st].mean() for k in ks] for st in stages}
bottoms = np.zeros(len(ks))
x = np.arange(len(ks))
for st, lab, col in zip(stages, labels, colors):
    vals = np.array(means[st])
    ax.bar(x, vals, bottom=bottoms, width=0.5, color=col, label=lab, zorder=3)
    for i in range(len(ks)):
        if vals[i] > 0.35:
            ax.text(x[i], bottoms[i] + vals[i] / 2, f"{vals[i]:.2f}", ha="center", va="center",
                    fontsize=8.3, color="white" if col != CAT4 else INK)
    bottoms += vals
for i, k in enumerate(ks):
    ax.text(x[i], bottoms[i] + 0.15, f"{bottoms[i]:.1f} ms", ha="center", va="bottom",
            fontsize=10, fontweight="bold", color=INK)
ax.set_xticks(x)
ax.set_xticklabels([f"$k={k}$" for k in ks], fontsize=12)
ax.set_ylabel("mean end-to-end time (ms)")
ax.set_ylim(0, max(bottoms) * 1.22)
style_ax(ax)
ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=4, frameon=False, fontsize=8.6,
          handletextpad=0.4, columnspacing=1.1)
save(fig, "s3_stage_timing")

# ---- Fig S4: Dinkelbach iterations + LP decisiveness (two small panels) ----
fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.7))
ax = axes[0]
ran = s3[s3.dinkelbach_ran == True]
counts = ran.dinkelbach_iterations.value_counts().sort_index()
ax.bar(counts.index.astype(str), counts.values, color=CAT1, width=0.55, zorder=3)
for xv, yv in zip(counts.index.astype(str), counts.values):
    ax.text(xv, yv + 0.3, str(yv), ha="center", fontsize=10, color=INK2)
ax.set_xlabel("Dinkelbach iterations")
ax.set_ylabel("invocations (of 39)")
ax.set_title("Stage 1 convergence", fontsize=11, color=INK, pad=8)
style_ax(ax)

ax = axes[1]
lp_ran = int(s3.lp_ran.sum())
lp_dec = int(s3.lp_decisive.sum())
bars = ax.bar(["LP\npre-check ran", "LP was\ndecisive"], [lp_ran, lp_dec],
              color=[CAT3, CRITICAL], width=0.55, zorder=3)
for b, v in zip(bars, [lp_ran, lp_dec]):
    ax.text(b.get_x() + b.get_width() / 2, v + 1.5, str(v), ha="center", fontsize=12,
            fontweight="bold", color=INK)
ax.set_ylim(0, 84)
ax.set_ylabel("instances (of 76)")
ax.set_title("LP feasibility pre-check", fontsize=11, color=INK, pad=8)
style_ax(ax)
fig.tight_layout(w_pad=3.0)
save(fig, "s3_dinkelbach_lp")

# ---- Fig S5: Stage-1-skip -- time & quality tradeoff ----
fig, axes = plt.subplots(1, 2, figsize=(6.8, 3.0))
ax = axes[0]
e2e_2s = [1000 * s3[s3.k == k].end_to_end_s.mean() for k in ks]
e2e_sk = [1000 * s3[s3.k == k].skip_end_to_end_s.mean() for k in ks]
x = np.arange(len(ks))
w = 0.32
ax.bar(x - w/2, e2e_2s, width=w, color=CAT1, label="two_stage", zorder=3)
ax.bar(x + w/2, e2e_sk, width=w, color=CAT2, label="skip_stage1", zorder=3)
for i in range(len(ks)):
    ax.text(x[i] - w/2, e2e_2s[i] + 0.15, f"{e2e_2s[i]:.1f}", ha="center", fontsize=8.6, color=INK2)
    ax.text(x[i] + w/2, e2e_sk[i] + 0.15, f"{e2e_sk[i]:.1f}", ha="center", fontsize=8.6, color=INK2)
ax.set_xticks(x); ax.set_xticklabels([f"$k={k}$" for k in ks], fontsize=11)
ax.set_ylabel("end-to-end (ms)")
ax.set_ylim(0, max(e2e_sk) * 1.28)
style_ax(ax)
ax.legend(loc="upper left", frameon=False, fontsize=8.6)

ax = axes[1]
FR_2s = [s3[s3.k == k].F_R.mean() for k in ks]
FR_sk = [s3[s3.k == k].skip_F_R.mean() for k in ks]
ax.bar(x - w/2, FR_2s, width=w, color=CAT1, zorder=3)
ax.bar(x + w/2, FR_sk, width=w, color=CAT2, zorder=3)
ax.axhline(0.30, color=INK, linewidth=1.1, linestyle="--", zorder=4)
ax.text(len(ks) - 0.5 + 0.03, 0.30, "$F^\\star$", va="bottom", fontsize=9.5, color=INK)
for i in range(len(ks)):
    ax.text(x[i] - w/2, FR_2s[i] + 0.012, f"{FR_2s[i]:.2f}", ha="center", fontsize=8.6, color=INK2)
    ax.text(x[i] + w/2, FR_sk[i] + 0.012, f"{FR_sk[i]:.2f}", ha="center", fontsize=8.6, color=INK2)
ax.set_xticks(x); ax.set_xticklabels([f"$k={k}$" for k in ks], fontsize=11)
ax.set_ylabel("$F_R$ achieved")
ax.set_ylim(0, 0.58)
style_ax(ax)
fig.tight_layout(w_pad=3.4)
save(fig, "s3_skip_stage1")

# =============================================================================
# SANTOSLARGE (top_n=5000, |D|>=50 cohort)  -- the first configuration where
# alpha actually bites
# =============================================================================
sl = pd.read_csv(os.path.join(REPO, "experiments/results/santoslarge_maxd_topn5000_D60.csv"))
configs = [(10, 2.0), (10, 3.0), (10, 4.0), (15, 2.0), (15, 3.0), (15, 4.0)]
clabels = [f"$k={k}$\n$\\alpha={int(a)}$" for k, a in configs]
SL_FSTAR = 0.122

# ---- Fig L0a: maxD selection -- before (random) vs after (chosen) |D| ----
maxd_report_path = os.path.join(REPO, "experiments/results/santoslarge_maxd_selection_report.csv")
if os.path.exists(maxd_report_path):
    md = pd.read_csv(maxd_report_path).sort_values("new_n_D", ascending=False).reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(7.0, 3.1))
    x = np.arange(len(md))
    ax.bar(x, md.new_n_D, color=CAT1, width=0.85, zorder=3, label="new (maxD-selected)")
    ax.bar(x, md.orig_n_D, color=MUTED, width=0.5, zorder=4, label="old (random)")
    ax.axhline(60, color=CRITICAL, linewidth=1.3, linestyle="--", zorder=5)
    ax.text(len(md) - 1, 60, "  $|D|\\geq 60$ cohort cutoff", va="bottom", ha="right",
            color=CRITICAL, fontsize=9.5)
    n_ge60 = int((md.new_n_D >= 60).sum())
    ax.text(0.5, md.new_n_D.max() * 0.95, f"{n_ge60} of {len(md)} clear the new cutoff", fontsize=11,
            color=INK, fontweight="bold", va="top")
    ax.set_xlabel("santosLarge queries, ranked by new $|D|$")
    ax.set_ylabel("$|D|$ (candidate tables)")
    style_ax(ax)
    ax.set_xticks([])
    ax.legend(loc="upper right", frameon=False, fontsize=9)
    save(fig, "sl_maxd_before_after")

    # ---- Fig L0b: threshold-selection curve -- old vs new, marking the chosen cutoff ----
    thresholds = [0, 10, 20, 30, 40, 50, 60, 80, 100, 150, 200, 250, 300]
    old_counts = [(md.orig_n_D >= t).sum() for t in thresholds]
    new_counts = [(md.new_n_D >= t).sum() for t in thresholds]
    fig, ax = plt.subplots(figsize=(6.6, 3.0))
    ax.plot(thresholds, old_counts, marker="o", color=MUTED, linewidth=1.6, label="old (random)", zorder=3)
    ax.plot(thresholds, new_counts, marker="o", color=CAT1, linewidth=1.6, label="new (maxD-selected)", zorder=3)
    ax.axvline(60, color=CRITICAL, linewidth=1.2, linestyle="--", zorder=2)
    ax.text(60, 5, " chosen\n cutoff", color=CRITICAL, fontsize=9, ha="left")
    ax.set_xlabel("$|D|$ threshold")
    ax.set_ylabel("queries clearing it (of 78)")
    style_ax(ax)
    ax.legend(loc="upper right", frameon=False, fontsize=9)
    save(fig, "sl_maxd_threshold_curve")
else:
    print("!! santoslarge_maxd_selection_report.csv not found -- skipping Fig L0a/L0b")

# ---- Fig L0c: tau/F* sweep -- reach rate vs F*, delta=0.15 fixed ----
tau_sweep_path = os.path.join(REPO, "experiments/results/santoslarge_tau_sweep_by_config.csv")
if os.path.exists(tau_sweep_path):
    ts = pd.read_csv(tau_sweep_path)
    ts_all = ts[ts.k.isna()].sort_values("F_star")
    ts_cells = ts[ts.k.notna()]
    fig, ax = plt.subplots(figsize=(6.6, 3.0))
    for (k, a), g in ts_cells.groupby(["k", "alpha"]):
        g = g.sort_values("F_star")
        ax.plot(g.F_star, 100 * g.reach_rate, color=BASE, linewidth=0.9, zorder=2)
    ax.plot(ts_all.F_star, 100 * ts_all.reach_rate, color=CAT1, linewidth=2.2, zorder=4, label="all 6 configs (aggregate)")
    ax.axvline(0.202, color=CRITICAL, linewidth=1.3, linestyle="--", zorder=3)
    ax.text(0.202, 30, "  chosen $\\tau{=}0.052$\n  (shown here at $\\delta{=}0.15$;\n  final: $\\delta{=}0.07,F^\\star{=}0.122$,\n  same $\\tau$)", color=CRITICAL, fontsize=8.6, ha="left")
    ax.set_xlabel("$F^\\star$ (with $\\delta=0.15$ fixed, so $\\tau=F^\\star-0.15$)")
    ax.set_ylabel("queries reaching feasibility (%)")
    ax.set_ylim(0, 105)
    style_ax(ax)
    ax.legend(loc="upper right", frameon=False, fontsize=9)
    save(fig, "sl_tau_sweep_curve")
else:
    print("!! santoslarge_tau_sweep_by_config.csv not found -- skipping Fig L0c")

# ---- Fig L1: index build breakdown ----
fig, ax = plt.subplots(figsize=(6.8, 2.6))
segs = [("Load vectors", 0.33, CAT3), ("HNSW build", 224.65, CAT1), ("Inverted index (posting lists)", 109.65, CAT2)]
left = 0
callout_y = [0.62, 0.90]   # alternate callout heights for the two thin segments
ci = 0
for name, dur, col in segs:
    ax.barh([0], [dur], left=left, height=0.55, color=col, zorder=3)
    mid = left + dur / 2
    if dur > 15:
        ax.text(mid, 0, f"{name}\n{dur:.1f}s", ha="center", va="center",
                color="white", fontsize=10, linespacing=1.15)
    else:
        y = callout_y[ci % len(callout_y)]
        ci += 1
        ax.annotate(f"{name}, {dur:.2g}s", xy=(mid, 0.28), xytext=(mid, y),
                    ha="center", va="bottom", fontsize=8.6, color=INK2,
                    arrowprops=dict(arrowstyle="-", color=BASE, linewidth=0.9))
    left += dur
ax.set_xlim(-left * 0.01, left * 1.02)
ax.set_ylim(-0.45, 1.15)
ax.set_yticks([])
ax.set_xlabel("seconds (one-off, amortized over all queries)")
total = left
ax.text(left + total * 0.015, 0, f"= {total:.0f}s total", va="center", ha="left", fontsize=11,
        color=INK, fontweight="bold")
style_ax(ax, ygrid=False)
ax.spines["bottom"].set_visible(True)
save(fig, "sl_index_build")

# ---- Fig L2: |P| tracks alpha x k exactly ----
fig, ax = plt.subplots(figsize=(6.6, 3.0))
ks_sl = [10, 15]
alphas_sl = [2.0, 3.0, 4.0]
x = np.arange(len(alphas_sl))
w = 0.32
for i, k in enumerate(ks_sl):
    vals = [sl[(sl.k == k) & (sl.alpha == a)].n_P.mean() for a in alphas_sl]
    off = (-w/2 if i == 0 else w/2)
    col = CAT1 if i == 0 else CAT2
    bars = ax.bar(x + off, vals, width=w, color=col, label=f"$k={k}$", zorder=3)
    for xi, v in zip(x + off, vals):
        ax.text(xi, v + 1.3, f"{v:.0f}", ha="center", fontsize=9.5, color=INK2)
ax.set_xticks(x)
ax.set_xticklabels([f"$\\alpha={a:.0f}$" for a in alphas_sl], fontsize=12)
ax.set_ylabel("realized pool size $|P|$")
ax.set_ylim(0, max(k * a for k in ks_sl for a in alphas_sl) * 1.15)
style_ax(ax)
ax.legend(loc="upper left", frameon=False, fontsize=10)
save(fig, "sl_pool_scaling")

# ---- Fig L3: per-stage timing scaling (the money plot) ----
fig, ax = plt.subplots(figsize=(8.4, 3.6))
stages = ["retrieval_time_s", "stage1_time_s", "scoring_time_s", "stage2_time_s"]
labels = ["Retrieval", "Stage 1", "Scoring ($U$)", "Stage 2 (ILP)"]
colors = [CAT1, CAT4, CAT3, CAT2]
x = np.arange(len(configs))
bottoms = np.zeros(len(configs))
means = {}
for st in stages:
    means[st] = [1000 * sl[(sl.k == k) & (sl.alpha == a)][st].mean() for k, a in configs]
for st, lab, col in zip(stages, labels, colors):
    vals = np.array(means[st])
    ax.bar(x, vals, bottom=bottoms, width=0.55, color=col, label=lab, zorder=3)
    for i in range(len(configs)):
        if vals[i] > 3:
            ax.text(x[i], bottoms[i] + vals[i] / 2, f"{vals[i]:.0f}", ha="center", va="center",
                    fontsize=8.4, color="white" if col != CAT4 else INK)
    bottoms += vals
for i in range(len(configs)):
    ax.text(x[i], bottoms[i] + 1.5, f"{bottoms[i]:.0f} ms", ha="center", va="bottom",
            fontsize=10, fontweight="bold", color=INK)
ax.set_xticks(x)
ax.set_xticklabels(clabels, fontsize=10.5)
ax.set_ylabel("mean end-to-end time (ms)")
ax.set_ylim(0, max(bottoms) * 1.2)
style_ax(ax)
ax.legend(loc="upper left", frameon=False, fontsize=9, ncol=2)
save(fig, "sl_stage_timing_scaling")

# ---- Fig L4: outcome (tau reached) + Dinkelbach iterations ----
fig, axes = plt.subplots(1, 2, figsize=(6.8, 2.8))
ax = axes[0]
n_feas = int(sl[(sl.k == 10) & (sl.alpha == 2.0)].feasible.sum())
n_total = int((sl.k == 10).sum() // 1)
n_total = int(len(sl[(sl.k == 10) & (sl.alpha == 2.0)]))
n_infeas = n_total - n_feas
ax.barh([0], [n_feas], color=GOOD, height=0.5, zorder=3, label="$\\tau$ reached")
ax.barh([0], [n_infeas], left=[n_feas], color=CRITICAL, height=0.5, zorder=3, label="infeasible")
ax.text(n_feas / 2, 0, f"{n_feas}", color="white", ha="center", va="center", fontsize=12, fontweight="bold")
ax.text(n_feas + n_infeas / 2, 0, f"{n_infeas}", color="white", ha="center", va="center", fontsize=12, fontweight="bold")
ax.set_xlim(0, n_total)
ax.set_yticks([])
ax.set_xlabel(f"$n={n_total}$ queries")
ax.set_title("Outcome", fontsize=11, color=INK, pad=10)
ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.42), ncol=2, frameon=False, fontsize=9)
style_ax(ax, ygrid=False)

ax = axes[1]
ran = sl[sl.dinkelbach_ran == True]
counts = ran.dinkelbach_iterations.value_counts().sort_index()
ax.bar(counts.index.astype(str), counts.values, color=CAT1, width=0.55, zorder=3)
for xv, yv in zip(counts.index.astype(str), counts.values):
    ax.text(xv, yv + 0.6, str(yv), ha="center", fontsize=10, color=INK2)
ax.set_xlabel("Dinkelbach iterations")
ax.set_ylabel(f"invocations (of {len(ran)})")
ax.set_title("Stage 1 convergence", fontsize=11, color=INK, pad=10)
style_ax(ax)
fig.tight_layout(w_pad=3.4)
save(fig, "sl_outcome_dinkelbach")

# ---- Fig L5: LP decisiveness -- santos3 vs santosLarge (the reversal) ----
fig, ax = plt.subplots(figsize=(6.2, 3.0))
benches = [f"santos3\n({len(s3)} instances)", f"santosLarge\n({len(sl)} instances)"]
ran_pct = [100 * s3.lp_ran.mean(), 100 * sl.lp_ran.mean()]
dec_pct = [100 * s3.lp_decisive.mean(), 100 * sl.lp_decisive.mean()]
x = np.arange(len(benches))
w = 0.32
ax.bar(x - w/2, ran_pct, width=w, color=CAT3, label="LP pre-check ran", zorder=3)
ax.bar(x + w/2, dec_pct, width=w, color=CRITICAL, label="LP was decisive", zorder=3)
for i in range(len(benches)):
    ax.text(x[i] - w/2, ran_pct[i] + 2, f"{ran_pct[i]:.0f}%", ha="center", fontsize=10, color=INK2)
    ax.text(x[i] + w/2, dec_pct[i] + 2, f"{dec_pct[i]:.0f}%", ha="center", fontsize=10,
            color=INK, fontweight="bold")
ax.set_xticks(x)
ax.set_xticklabels(benches, fontsize=11)
ax.set_ylabel("share of instances")
ax.set_ylim(0, 112)
ax.yaxis.set_major_formatter(mticker.PercentFormatter())
style_ax(ax)
ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=2, frameon=False, fontsize=9.5)
save(fig, "lp_decisiveness_comparison")

# ---- Fig L6: Stage-1-skip -- speedup and quality tradeoff ----
fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.4))
ax = axes[0]
speedup = [sl[(sl.k == k) & (sl.alpha == a)].skip_end_to_end_s.mean() /
           sl[(sl.k == k) & (sl.alpha == a)].end_to_end_s.mean() for k, a in configs]
x = np.arange(len(configs))
bars = ax.bar(x, speedup, width=0.55, color=CAT1, zorder=3)
ax.axhline(1.0, color=INK, linewidth=1.0, linestyle="--", zorder=4)
for xi, v in zip(x, speedup):
    ax.text(xi, v + 0.05, f"{v:.2f}$\\times$", ha="center", fontsize=10, fontweight="bold", color=INK)
ax.set_xticks(x)
ax.set_xticklabels(clabels, fontsize=9.5, rotation=0)
ax.set_ylabel("two-stage speedup vs. skip_stage1")
ax.set_ylim(0, max(speedup) * 1.28)
style_ax(ax)

ax = axes[1]
FR_2s = [sl[(sl.k == k) & (sl.alpha == a)].F_R.mean() for k, a in configs]
FR_sk = [sl[(sl.k == k) & (sl.alpha == a)].skip_F_R.mean() for k, a in configs]
w = 0.32
ax.bar(x - w/2, FR_2s, width=w, color=CAT1, label="two_stage", zorder=3)
ax.bar(x + w/2, FR_sk, width=w, color=CAT2, label="skip_stage1", zorder=3)
ax.axhline(SL_FSTAR, color=INK, linewidth=1.0, linestyle="--", zorder=4)
ax.text(len(configs) - 0.5 + 0.05, SL_FSTAR, "$F^\\star$", va="bottom", fontsize=9.5, color=INK)
ax.set_xticks(x)
ax.set_xticklabels(clabels, fontsize=9.5)
ax.set_ylabel("$F_R$ achieved")
ax.set_ylim(0, max(max(FR_2s), max(FR_sk), SL_FSTAR) * 1.15)
style_ax(ax)
ax.legend(loc="upper right", frameon=False, fontsize=9)
fig.tight_layout(w_pad=3.6)
save(fig, "sl_skip_stage1")


# =============================================================================
# SANTOS3 PRECISION / RECALL -- same solvable-only cohort as the focused study
# =============================================================================
pr_rows = []
ideal_recall = {}   # k -> mean_i min(k, gt_size_i) / gt_size_i  (starmie_fair checkPrecisionRecall.py)
for k in ks:  # ks = [3, 5], defined above from the santos3 section
    keep = set(s3[s3.k == k].q_table)
    g = pd.read_csv(os.path.join(REPO, f"experiments/results/santos3_groundtruth_eval_k{k}.csv"))
    r = g[g.q_table.isin(keep) & g.gt_size.notna()]
    for stage, lab in (("d", "D"), ("p", "P"), ("r", "R")):
        pr_rows.append((k, lab, r["precision_" + stage].mean(), r["recall_" + stage].mean(), len(r)))
    ideal_recall[k] = (r.gt_size.apply(lambda gt: min(k, gt) / gt)).mean()
pr = pd.DataFrame(pr_rows, columns=["k", "stage", "precision", "recall", "n"])

fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.2))
stages_pr = ["D", "P", "R"]
stage_colors = [CAT1, CAT3, CAT2]

ax = axes[0]
x = np.arange(len(stages_pr))
w = 0.32
for i, k in enumerate(ks):
    vals = [pr[(pr.k == k) & (pr.stage == s)].precision.values[0] for s in stages_pr]
    off = (-w/2 if i == 0 else w/2)
    col = CAT1 if i == 0 else CAT2
    bars = ax.bar(x + off, vals, width=w, color=col, label=f"$k={k}$", zorder=3)
    for xi, v in zip(x + off, vals):
        ax.text(xi, v + 0.02, f"{v:.2f}", ha="center", fontsize=8.6, color=INK2)
ax.set_xticks(x)
ax.set_xticklabels([f"stage {s}" for s in stages_pr], fontsize=10.5)
ax.set_ylabel("precision")
ax.set_ylim(0, 1.18)
style_ax(ax)
ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=2, frameon=False, fontsize=9)

ax = axes[1]
r_idx = stages_pr.index("R")
for i, k in enumerate(ks):
    vals = [pr[(pr.k == k) & (pr.stage == s)].recall.values[0] for s in stages_pr]
    off = (-w/2 if i == 0 else w/2)
    col = CAT1 if i == 0 else CAT2
    ax.bar(x + off, vals, width=w, color=col, zorder=3)
    for xi, v in zip(x + off, vals):
        ax.text(xi, v + 0.012, f"{v:.2f}", ha="center", fontsize=8.6, color=INK2)
    # ideal-recall ceiling marker at stage R: mean_i min(k, gt_size_i)/gt_size_i
    ax.plot([x[r_idx] + off], [ideal_recall[k]], marker="D", markersize=8,
            markerfacecolor="none", markeredgecolor=INK, markeredgewidth=1.6, zorder=5,
            label="ideal recall ceiling" if i == 0 else None)
ax.set_xticks(x)
ax.set_xticklabels([f"stage {s}" for s in stages_pr], fontsize=10.5)
ax.set_ylabel("recall")
ax.set_ylim(0, 0.46)
style_ax(ax)
ax.legend(loc="upper left", frameon=False, fontsize=8.6)
fig.tight_layout(w_pad=3.6)
save(fig, "s3_precision_recall")

print("ideal_recall(k=3)={:.4f}  ideal_recall(k=5)={:.4f}".format(ideal_recall[3], ideal_recall[5]))

# =============================================================================
# WDC tier_10k + tier_100k -- real-data scalability results (50.8M-table
# corpus, sampled tiers; tier_1m not yet run)
# =============================================================================
import json

wdc = pd.read_csv(os.path.join(REPO, "experiments/results/wdc_tier_10k.csv"))
with open(os.path.join(REPO, "experiments/results/wdc_tier_10k_selection_report.json")) as f:
    wdc_sel = json.load(f)
wdc100 = pd.read_csv(os.path.join(REPO, "experiments/results/wdc_tier_100k.csv"))
with open(os.path.join(REPO, "experiments/results/wdc_tier_100k_selection_report.json")) as f:
    wdc100_sel = json.load(f)

# ---- Fig W1: index build breakdown, tier_10k vs tier_100k (new: embedding_
# extraction, a cost santos-family never paid since embeddings were prebuilt) ----
fig, axes = plt.subplots(2, 1, figsize=(7.6, 4.0), sharex=False)
tier_segs = {
    "tier_10k (10,000 tables)": [
        ("CSV conv.", 5.73, CAT3), ("Embed (GPU)", 70.2, CAT4),
        ("Load", 0.15, MUTED), ("HNSW", 5.71, CAT1), ("Inverted idx", 25.01, CAT2),
    ],
    "tier_100k (100,000 tables)": [
        ("CSV conv.", 62.49, CAT3), ("Embed (GPU)", 622.9, CAT4),
        ("Load", 1.43, MUTED), ("HNSW", 83.97, CAT1), ("Inverted idx", 314.15, CAT2),
    ],
}
for ax, (label, segs) in zip(axes, tier_segs.items()):
    left = 0
    for name, dur, col in segs:
        ax.barh([0], [dur], left=left, height=0.55, color=col, zorder=3)
        mid = left + dur / 2
        if dur / sum(d for _, d, _ in segs) > 0.08:
            ax.text(mid, 0, f"{name}\n{dur:.1f}s", ha="center", va="center",
                    color="white", fontsize=8, linespacing=1.05)
        left += dur
    ax.set_xlim(0, left * 1.02)
    ax.set_ylim(-0.45, 0.45)
    ax.set_yticks([0])
    ax.set_yticklabels([label], fontsize=9.5, color=INK2)
    ax.text(left * 1.03, 0, f"{left:.0f}s", va="center", ha="left", fontsize=11,
            color=INK, fontweight="bold")
    style_ax(ax, ygrid=False)
    ax.spines["bottom"].set_visible(True)
axes[-1].set_xlabel("seconds (one-off per tier; excludes offline GPU embedding step from the total below)")
fig.text(0.99, 0.02, "total (excl. embed): 30.9s → 399.6s = 12.9×", ha="right",
          fontsize=9, color=INK2, style="italic")
fig.tight_layout(h_pad=1.6, rect=(0, 0.04, 1, 1))
save(fig, "wdc_index_build")

# ---- Fig W2: categorical column coverage -- WDC vs santos, with row-count
# context (the ratio alone doesn't make the point -- see plan's correction) ----
fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.0))
ax = axes[0]
names = ["santos\n(~5,000 rows/tbl)", "tier_10k\n(~6 rows/tbl)", "tier_100k\n(~6 rows/tbl)"]
ratios = [82.0, 94.2, 94.1]
colors_b = [CAT1, CAT2, CAT4]
bars = ax.bar(names, ratios, color=colors_b, width=0.55, zorder=3)
for i, r in enumerate(ratios):
    ax.text(i, r + 2, f"{r:.1f}%", ha="center", fontsize=11, fontweight="bold", color=INK)
ax.set_ylabel("columns classified categorical\n(theta_cat=50)")
ax.set_ylim(0, 108)
style_ax(ax)
ax.set_title("Coverage ratio alone", fontsize=10.5, color=INK2, pad=8)

ax = axes[1]
ax.axis("off")
ax.text(0.02, 0.90, "Stable 94.2%→94.1%", fontsize=11, color=INK, fontweight="bold")
ax.text(0.02, 0.73, "across 10× the tables:", fontsize=11, color=INK, fontweight="bold")
ax.text(0.02, 0.50, "santos: data is genuinely\nlow-cardinality (18% of\ncolumns correctly excluded).",
        fontsize=9.3, color=INK2, linespacing=1.35)
ax.text(0.02, 0.14, "WDC: $nunique \\leq n\\_rows$\nitself ≤ theta\\_cat regardless\nof tier size -- structural,\nnot data-driven.",
        fontsize=9.3, color=INK2, linespacing=1.35)
fig.tight_layout(w_pad=2.0)
save(fig, "wdc_column_coverage")

# ---- Fig W3: retrieval-yield funnel + paired biased-vs-unbiased |D| ----
fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.0))
ax = axes[0]
n_selected = wdc_sel["n_queries_selected"]
n_workable = wdc.q_table.nunique()
n_unworkable = n_selected - n_workable
ax.barh([0], [n_workable], color=GOOD, height=0.5, zorder=3, label="$|D|\\geq 20$ (workable)")
ax.barh([0], [n_unworkable], left=[n_workable], color=CRITICAL, height=0.5, zorder=3,
        label="$|D|<20$ (too small to attempt)")
ax.text(n_workable / 2, 0, f"{n_workable}", color="white", ha="center", va="center",
        fontsize=12, fontweight="bold")
ax.text(n_workable + n_unworkable / 2, 0, f"{n_unworkable}", color="white", ha="center",
        va="center", fontsize=12, fontweight="bold")
ax.set_xlim(0, n_selected)
ax.set_yticks([])
ax.set_xlabel(f"$n={n_selected}$ auto-selected queries")
ax.set_title("tier_10k retrieval yield", fontsize=10.5, color=INK, pad=10)
ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.42), ncol=1, frameon=False, fontsize=8.6)
style_ax(ax, ygrid=False)

ax = axes[1]
diffs = [wdc_sel["paired_biased_minus_unbiased_mean"], wdc100_sel["paired_biased_minus_unbiased_mean"]]
x = np.arange(2)
bars = ax.bar(x, diffs, color=[CAT1, CAT2], width=0.5, zorder=3)
for xi, v in zip(x, diffs):
    ax.text(xi, v + 0.05, f"{v:+.2f}", ha="center", fontsize=12, fontweight="bold", color=INK)
ax.set_xticks(x)
ax.set_xticklabels(["tier_10k", "tier_100k"], fontsize=10.5)
ax.set_ylabel("paired mean diff\n(biased $-$ unbiased $|D|$)")
ax.set_ylim(0, max(diffs) * 1.35)
ax.set_title("biasing helps MORE at scale", fontsize=10.5, color=INK2, pad=10)
style_ax(ax)
fig.tight_layout(w_pad=3.4)
save(fig, "wdc_retrieval_yield")

# ---- Fig W4: per-stage timing, averaged over the 5 workable queries x 4 configs ----
fig, ax = plt.subplots(figsize=(7.2, 3.4))
wdc_configs = [(10, 2.0), (10, 3.0), (20, 2.0), (20, 3.0)]
wdc_clabels = [f"$k{{=}}{k},\\alpha{{=}}{a:.0f}$" for k, a in wdc_configs]
stages = ["retrieval_time_s", "stage1_time_s", "scoring_time_s", "stage2_time_s"]
labels = ["Retrieval", "Stage 1", "Scoring ($U$)", "Stage 2 (ILP)"]
colors = [CAT1, CAT4, CAT3, CAT2]
x = np.arange(len(wdc_configs))
bottoms = np.zeros(len(wdc_configs))
means = {}
for st in stages:
    means[st] = [1000 * wdc[(wdc.k == k) & (wdc.alpha == a)][st].mean() for k, a in wdc_configs]
for st, lab, col in zip(stages, labels, colors):
    vals = np.array(means[st])
    ax.bar(x, vals, bottom=bottoms, width=0.55, color=col, label=lab, zorder=3)
    for i in range(len(wdc_configs)):
        if vals[i] > 0.3:
            ax.text(x[i], bottoms[i] + vals[i] / 2, f"{vals[i]:.1f}", ha="center", va="center",
                    fontsize=8.0, color="white" if col != CAT4 else INK)
    bottoms += vals
for i in range(len(wdc_configs)):
    ax.text(x[i], bottoms[i] + max(bottoms) * 0.02, f"{bottoms[i]:.1f} ms", ha="center", va="bottom",
            fontsize=9.5, fontweight="bold", color=INK)
ax.set_xticks(x)
ax.set_xticklabels(wdc_clabels, fontsize=10)
ax.set_ylabel("mean end-to-end time (ms)")
ax.set_ylim(0, max(bottoms) * 1.25)
style_ax(ax)
ax.legend(loc="upper left", frameon=False, fontsize=9, ncol=2)
save(fig, "wdc_stage_timing")

# ---- Fig W5: cross-tier trend -- feasibility rate and workable-query yield,
# the two findings only visible once a second tier exists ----
fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0))
ax = axes[0]
feas = [100.0, 100 * 16 / 24]
x = np.arange(2)
bars = ax.bar(x, feas, color=[CAT1, CRITICAL], width=0.5, zorder=3)
for xi, v in zip(x, feas):
    ax.text(xi, v + 2, f"{v:.1f}%", ha="center", fontsize=11, fontweight="bold", color=INK)
ax.set_xticks(x)
ax.set_xticklabels(["tier_10k\n(5 workable q)", "tier_100k\n(6 workable q)"], fontsize=9.5)
ax.set_ylabel("feasibility rate")
ax.set_ylim(0, 112)
ax.yaxis.set_major_formatter(mticker.PercentFormatter())
ax.set_title("Feasibility dropped", fontsize=10.5, color=INK, pad=8)
style_ax(ax)

ax = axes[1]
yields = [100 * 5 / 30, 100 * 6 / 30]
bars = ax.bar(x, yields, color=[CAT1, CAT2], width=0.5, zorder=3)
for xi, v in zip(x, yields):
    ax.text(xi, v + 0.5, f"{v:.1f}%", ha="center", fontsize=11, fontweight="bold", color=INK)
ax.set_xticks(x)
ax.set_xticklabels(["tier_10k", "tier_100k"], fontsize=9.5)
ax.set_ylabel("workable queries (of 30)")
ax.set_ylim(0, 26)
ax.yaxis.set_major_formatter(mticker.PercentFormatter())
ax.set_title("Yield rose slightly", fontsize=10.5, color=INK, pad=8)
style_ax(ax)
fig.tight_layout(w_pad=3.4)
save(fig, "wdc_cross_tier_trend")

print("\nWDC figures written -- tier_10k n_workable={}/{} queries ({} rows); "
      "tier_100k n_workable=6/30 queries ({} rows)".format(
          n_workable, n_selected, len(wdc), len(wdc100)))

# =============================================================================
# WDC UNION-POOL EXPERIMENT -- D = D_sem ∪ D_ovl instead of D_sem ∩ D_ovl
# (experiments/wdc/union_retrieval.py; a WDC-scalability-only deviation from
# §7.3, kept out of dutsx/ entirely). Same 30 auto-selected queries per tier
# as the intersection study above (same seed/params), so pool sizes are
# directly paired per query. Exists to answer: once the pool is big enough,
# what does Stage 1 / scoring / Stage 2 actually cost?
# =============================================================================
wu10 = pd.read_csv(os.path.join(REPO, "experiments/results/wdc_union_tier_10k.csv"))
wu100 = pd.read_csv(os.path.join(REPO, "experiments/results/wdc_union_tier_100k.csv"))

# ---- Fig U1: pool-size growth -- intersection vs union, both tiers (log scale) ----
fig, ax = plt.subplots(figsize=(7.0, 3.4))
tiers_u = ["tier_10k", "tier_100k"]
med_inter = [wu10.drop_duplicates("q_table").n_D_intersection.median(),
             wu100.drop_duplicates("q_table").n_D_intersection.median()]
med_union = [wu10.drop_duplicates("q_table").n_D_union.median(),
             wu100.drop_duplicates("q_table").n_D_union.median()]
x = np.arange(len(tiers_u))
w = 0.32
ax.bar(x - w/2, med_inter, width=w, color=CAT2, label="intersection ($D_{sem} \\cap D_{ovl}$, §7.3)", zorder=3)
ax.bar(x + w/2, med_union, width=w, color=CAT1, label="union ($D_{sem} \\cup D_{ovl}$, this experiment)", zorder=3)
for xi, v in zip(x - w/2, med_inter):
    ax.text(xi, v * 1.15, f"{v:.0f}", ha="center", fontsize=10.5, fontweight="bold", color=INK)
for xi, v in zip(x + w/2, med_union):
    ax.text(xi, v * 1.15, f"{v:.0f}", ha="center", fontsize=10.5, fontweight="bold", color=INK)
ax.set_yscale("log")
ax.set_xticks(x)
ax.set_xticklabels(tiers_u, fontsize=11.5)
ax.set_ylabel("median $|D|$ (log scale)")
ax.set_title("Same queries, same $M$ -- only the combine rule differs", fontsize=10.5, color=INK2, pad=8)
style_ax(ax)
ax.legend(loc="upper left", frameon=False, fontsize=9.2)
save(fig, "wdc_union_pool_growth")

# ---- Fig U2: per-stage timing under union pools, both tiers ----
fig, axes = plt.subplots(1, 2, figsize=(9.4, 3.6), sharey=False)
stages = ["retrieval_time_s", "stage1_time_s", "scoring_time_s", "stage2_time_s"]
labels = ["Retrieval", "Stage 1", "Scoring ($U$)", "Stage 2 (ILP)"]
colors = [CAT1, CAT4, CAT3, CAT2]
wdc_configs_u = [(10, 2.0), (10, 3.0), (20, 2.0), (20, 3.0)]
wdc_clabels_u = [f"$k{{=}}{k},\\alpha{{=}}{a:.0f}$" for k, a in wdc_configs_u]
for ax, (df, tlabel) in zip(axes, [(wu10, "tier_10k"), (wu100, "tier_100k")]):
    x = np.arange(len(wdc_configs_u))
    bottoms = np.zeros(len(wdc_configs_u))
    means = {st: [1000 * df[(df.k == k) & (df.alpha == a)][st].mean() for k, a in wdc_configs_u]
             for st in stages}
    for st, lab, col in zip(stages, labels, colors):
        vals = np.array(means[st])
        ax.bar(x, vals, bottom=bottoms, width=0.55, color=col, label=lab, zorder=3)
        for i in range(len(wdc_configs_u)):
            if vals[i] > max(bottoms + vals) * 0.03:
                ax.text(x[i], bottoms[i] + vals[i] / 2, f"{vals[i]:.1f}", ha="center", va="center",
                        fontsize=7.6, color="white" if col != CAT4 else INK)
        bottoms += vals
    for i in range(len(wdc_configs_u)):
        ax.text(x[i], bottoms[i] + max(bottoms) * 0.02, f"{bottoms[i]:.0f} ms", ha="center", va="bottom",
                fontsize=9, fontweight="bold", color=INK)
    ax.set_xticks(x)
    ax.set_xticklabels(wdc_clabels_u, fontsize=8.6)
    ax.set_ylabel("mean end-to-end time (ms)")
    ax.set_ylim(0, max(bottoms) * 1.28)
    ax.set_title(tlabel, fontsize=11, color=INK, pad=8)
    style_ax(ax)
axes[0].legend(loc="upper left", frameon=False, fontsize=8.2, ncol=1)
fig.tight_layout(w_pad=3.2)
save(fig, "wdc_union_stage_timing")

# ---- Fig U2.5: what retrieval time is actually spent on -- probe vs. combine
# vs. N_i/n_i lookup, per tier. Corrects an earlier measurement bug (an
# earlier version probed semantic+overlap TWICE -- once per combine-rule
# variant -- and reported their summed cost as "retrieval time"; fixed by
# probing once and timing each phase separately, see RESULTS-wdc.md §6.1) ----
fig, axes = plt.subplots(2, 1, figsize=(7.6, 3.6), sharex=False)
retrieval_segs = {
    "tier_10k": [("Probe", 0.74, CAT3), ("Union combine", 0.54, CAT4), ("$N_i/n_i$ lookup", 59.65, CAT1)],
    "tier_100k": [("Probe", 4.91, CAT3), ("Union combine", 6.47, CAT4), ("$N_i/n_i$ lookup", 602.32, CAT1)],
}
for ax, (label, segs) in zip(axes, retrieval_segs.items()):
    left = 0
    total = sum(d for _, d, _ in segs)
    callout_y = [0.62, 0.90]
    ci = 0
    for name, dur, col in segs:
        ax.barh([0], [dur], left=left, height=0.55, color=col, zorder=3)
        mid = left + dur / 2
        if dur / total > 0.10:
            ax.text(mid, 0, f"{name}\n{dur:.2f}ms", ha="center", va="center",
                    color="white", fontsize=8.6, linespacing=1.05)
        else:
            y = callout_y[ci % len(callout_y)]
            ci += 1
            ax.annotate(f"{name}, {dur:.2f}ms", xy=(mid, 0.28), xytext=(mid, y),
                        ha="center", va="bottom", fontsize=7.8, color=INK2,
                        arrowprops=dict(arrowstyle="-", color=BASE, linewidth=0.9))
        left += dur
    ax.set_xlim(-total * 0.01, total * 1.02)
    ax.set_ylim(-0.45, 1.15)
    ax.set_yticks([0])
    ax.set_yticklabels([label], fontsize=9.5, color=INK2)
    ax.text(total * 1.03, 0, f"{total:.1f}ms", va="center", ha="left", fontsize=10.5,
            color=INK, fontweight="bold")
    style_ax(ax, ygrid=False)
    ax.spines["bottom"].set_visible(True)
axes[-1].set_xlabel("mean retrieval time (ms), union path")
fig.tight_layout(h_pad=1.8)
save(fig, "wdc_union_retrieval_breakdown")

# ---- Fig U3: feasibility under union pools -- no longer retrieval-limited,
# but still not universally satisfiable (genuine Stage 2 infeasibility) ----
fig, ax = plt.subplots(figsize=(6.2, 3.0))
feas_u = [100 * wu10.feasible.mean(), 100 * wu100.feasible.mean()]
ax.bar(np.arange(2), feas_u, color=[CAT1, CAT2], width=0.5, zorder=3)
for xi, v in zip(np.arange(2), feas_u):
    ax.text(xi, v + 2, f"{v:.1f}%", ha="center", fontsize=12, fontweight="bold", color=INK)
ax.set_xticks(np.arange(2))
ax.set_xticklabels(tiers_u, fontsize=11)
ax.set_ylabel("feasible (of all $k,\\alpha$ configs \\& queries)")
ax.set_ylim(0, 100)
ax.yaxis.set_major_formatter(mticker.PercentFormatter())
ax.set_title("Union pools remove the retrieval bottleneck --\nremaining infeasibility is genuine ($\\tau$ unreachable)",
             fontsize=10, color=INK2, pad=8)
style_ax(ax)
save(fig, "wdc_union_feasibility")

print("\nWDC union-pool figures written -- tier_10k median |D| {:.0f}->{:.0f}, "
      "tier_100k median |D| {:.0f}->{:.0f}".format(
          med_inter[0], med_union[0], med_inter[1], med_union[1]))

# =============================================================================
# WDC attributes indexed / retrieved + queries tried (per-tier reporting slide)
# =============================================================================
tier_attr_stats = {
    "tier_10k": dict(
        n_cols_total=51854, n_cols_cat=48857,
        n_queries_selected=wdc_sel["n_queries_selected"],
        n_workable_union=int(wu10.q_table.nunique()),
        n_workable_intersection=int(wdc.q_table.nunique()),
        attrs_retrieved_union=float(wu10.drop_duplicates("q_table").n_D_union.sum()),
        attrs_retrieved_intersection=float(wu10.drop_duplicates("q_table").n_D_intersection.sum()),
    ),
    "tier_100k": dict(
        n_cols_total=518131, n_cols_cat=487728,
        n_queries_selected=wdc100_sel["n_queries_selected"],
        n_workable_union=int(wu100.q_table.nunique()),
        n_workable_intersection=int(wdc100.q_table.nunique()),
        attrs_retrieved_union=float(wu100.drop_duplicates("q_table").n_D_union.sum()),
        attrs_retrieved_intersection=float(wu100.drop_duplicates("q_table").n_D_intersection.sum()),
    ),
}

fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.2))
ax = axes[0]
names_t = list(tier_attr_stats)
indexed = [tier_attr_stats[t]["n_cols_cat"] for t in names_t]
retrieved = [tier_attr_stats[t]["attrs_retrieved_union"] for t in names_t]
x = np.arange(len(names_t))
w = 0.32
ax.bar(x - w/2, indexed, width=w, color=CAT3, label="indexed (categorical attrs)", zorder=3)
ax.bar(x + w/2, retrieved, width=w, color=CAT1, label="retrieved (union, 30 queries, summed)", zorder=3)
ax.set_yscale("log")
for xi, v in zip(x - w/2, indexed):
    ax.text(xi, v * 1.2, f"{v:,.0f}", ha="center", fontsize=8.6, color=INK2)
for xi, v in zip(x + w/2, retrieved):
    ax.text(xi, v * 1.2, f"{v:,.0f}", ha="center", fontsize=8.6, color=INK2)
ax.set_xticks(x)
ax.set_xticklabels(names_t, fontsize=10.5)
ax.set_ylabel("attribute-columns (log scale)")
ax.set_title("Indexed vs. retrieved", fontsize=10.5, color=INK2, pad=8)
style_ax(ax)
ax.legend(loc="upper left", frameon=False, fontsize=7.6)

ax = axes[1]
sel_q = [tier_attr_stats[t]["n_queries_selected"] for t in names_t]
work_i = [tier_attr_stats[t]["n_workable_intersection"] for t in names_t]
work_u = [tier_attr_stats[t]["n_workable_union"] for t in names_t]
w = 0.26
ax.bar(x - w, sel_q, width=w, color=MUTED, label="tried (auto-selected)", zorder=3)
ax.bar(x, work_i, width=w, color=CAT2, label="workable (intersection)", zorder=3)
ax.bar(x + w, work_u, width=w, color=CAT1, label="workable (union)", zorder=3)
for xs, vals in [(x - w, sel_q), (x, work_i), (x + w, work_u)]:
    for xi, v in zip(xs, vals):
        ax.text(xi, v + 0.6, f"{v}", ha="center", fontsize=9, color=INK2)
ax.set_xticks(x)
ax.set_xticklabels(names_t, fontsize=10.5)
ax.set_ylabel("queries")
ax.set_ylim(0, max(sel_q) * 1.25)
ax.set_title("Queries tried", fontsize=10.5, color=INK2, pad=8)
style_ax(ax)
ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=1, frameon=False, fontsize=7.6)
fig.tight_layout(w_pad=3.2)
save(fig, "wdc_attrs_queries")

print("attrs/queries stats:", json.dumps(tier_attr_stats, indent=2))

print("\nALL FIGURES WRITTEN to", FIG)
