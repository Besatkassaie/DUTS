"""Plots for the ``top_n`` x ``alpha`` x ``k`` sweep (sigma=0.30, union retrieval) --
``topn_alpha_k_sweep.py``'s companion visualization. Reads
``experiments/results/wdc_sweep_tier_{10k,100k}.csv``, writes PDFs to
``experiments/results/sweep_figures/``.

Style matches ``experiments/beamer/plot_all.py`` (same palette, same ``style_ax``/``save``
helpers) so these sit visually consistent with the rest of the WDC figure set, without importing
that module directly (it runs its own top-level script body on import).
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
RESULTS = os.path.join(REPO, "experiments", "results")
FIG = os.path.join(RESULTS, "sweep_figures")
os.makedirs(FIG, exist_ok=True)

# ---------------------------------------------------------------- palette --- (matches plot_all.py)
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
BASE = "#c3c2b7"
SURFACE = "#fcfcfb"

CAT1 = "#2a78d6"   # blue    -- slot 1 (tier_10k / alpha=5)
CAT2 = "#eb6834"   # orange  -- slot 2 (tier_100k / alpha=10)
CAT3 = "#1baf7a"   # aqua    -- slot 3 (alpha=15)
CAT4 = "#eda100"   # yellow  -- slot 4 (alpha=20)
ALPHA_COLORS = {5.0: CAT1, 10.0: CAT2, 15.0: CAT3, 20.0: CAT4}
TIER_COLORS = {"tier_10k": CAT1, "tier_100k": CAT2}
STAGE_COLORS = {
    "probe_time_s": "#cde2fb", "union_combine_time_s": "#9ec5f4",
    "union_lookup_time_s": "#3987e5", "stage1_time_s": CAT3,
    "scoring_time_s": CAT4, "stage2_time_s": CAT2,
}

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
    path = os.path.join(FIG, name + ".pdf")
    fig.savefig(path, bbox_inches="tight", pad_inches=0.06)
    plt.close(fig)
    print("wrote", path)


def load_tier(tier):
    path = os.path.join(RESULTS, "wdc_sweep_%s.csv" % tier)
    return pd.read_csv(path)


STAGE_FIELDS = [
    ("probe_time_s", "Probe (HNSW + overlap)"),
    ("union_combine_time_s", "Union combine"),
    ("union_lookup_time_s", "$N_i/n_i$ lookup"),
    ("stage1_time_s", "Stage 1 (Dinkelbach)"),
    ("scoring_time_s", "Unionability scoring"),
    ("stage2_time_s", "Stage 2 (ILP)"),
]


def plot_time_vs_topn_by_alpha(df, tier):
    g = df.groupby(["top_n", "alpha"])["end_to_end_s"].mean().unstack()
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    for alpha in sorted(g.columns):
        ax.plot(g.index, g[alpha], marker="o", ms=4, lw=1.6,
                color=ALPHA_COLORS.get(alpha, MUTED), label=r"$\alpha$=%g" % alpha)
    style_ax(ax)
    ax.set_xlabel("top_n (HNSW probe breadth)")
    ax.set_ylabel("mean end-to-end time (s)")
    ax.set_title("%s: time vs. top_n, by $\\alpha$ ($\\sigma$=0.30, union retrieval)" % tier)
    ax.legend(loc="upper left", frameon=False, fontsize=9)
    save(fig, "time_vs_topn_by_alpha_%s" % tier)


def plot_time_vs_k_by_alpha_both_tiers(dfs, top_n=5000):
    alphas = sorted(set(a for df in dfs.values() for a in df["alpha"].unique()))
    fig, axes = plt.subplots(1, len(alphas), figsize=(4.0 * len(alphas), 3.6), sharey=True)
    if len(alphas) == 1:
        axes = [axes]
    for ax, alpha in zip(axes, alphas):
        for tier, df in dfs.items():
            sub = df[(df.alpha == alpha) & (df.top_n == top_n)]
            g = sub.groupby("k")["end_to_end_s"].mean().sort_index()
            ax.plot(g.index, g.values, marker="o", ms=4, lw=1.6,
                    color=TIER_COLORS.get(tier, MUTED), label=tier)
        style_ax(ax)
        ax.set_xlabel("k")
        ax.set_title(r"$\alpha$=%g" % alpha, fontsize=10.5)
        if ax is axes[0]:
            ax.set_ylabel("mean end-to-end time (s)")
    axes[-1].legend(loc="upper left", frameon=False, fontsize=9)
    fig.suptitle("Time vs. k, by $\\alpha$, both tiers (top_n=%d, $\\sigma$=0.30)" % top_n,
                 fontsize=11.5, y=1.03)
    save(fig, "time_vs_k_by_alpha_both_tiers_topn%d" % top_n)


def plot_time_vs_k_by_alpha_both_tiers_all_topn(dfs, top_ns):
    """Same trend, faceted by top_n too -- one row per top_n, one column per alpha,
    so the k-trend's dependence on top_n is visible rather than fixed at one setting."""
    alphas = sorted(set(a for df in dfs.values() for a in df["alpha"].unique()))
    fig, axes = plt.subplots(len(top_ns), len(alphas),
                              figsize=(3.4 * len(alphas), 2.6 * len(top_ns)),
                              sharex=True, sharey="row")
    for i, top_n in enumerate(top_ns):
        for j, alpha in enumerate(alphas):
            ax = axes[i][j]
            for tier, df in dfs.items():
                sub = df[(df.alpha == alpha) & (df.top_n == top_n)]
                g = sub.groupby("k")["end_to_end_s"].mean().sort_index()
                ax.plot(g.index, g.values, marker="o", ms=3, lw=1.4,
                        color=TIER_COLORS.get(tier, MUTED), label=tier)
            style_ax(ax)
            if i == 0:
                ax.set_title(r"$\alpha$=%g" % alpha, fontsize=10)
            if j == 0:
                ax.set_ylabel("top_n=%d\ntime (s)" % top_n, fontsize=9)
            if i == len(top_ns) - 1:
                ax.set_xlabel("k")
    axes[0][-1].legend(loc="upper left", frameon=False, fontsize=8.5)
    fig.suptitle("Time vs. k, by $\\alpha$ and top_n, both tiers ($\\sigma$=0.30)",
                 fontsize=12, y=1.01)
    save(fig, "time_vs_k_by_alpha_both_tiers_all_topn")


def plot_columns_retrieved_vs_topn(df, tier):
    g = df.groupby("top_n")[["n_sem", "n_ovl", "n_D_union"]].mean()
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    ax.plot(g.index, g.n_sem, marker="o", ms=4, lw=1.6, color=CAT1, label="$n_{sem}$ (semantic hits)")
    ax.plot(g.index, g.n_ovl, marker="o", ms=4, lw=1.6, color=CAT2, label="$n_{ovl}$ (overlap hits)")
    ax.plot(g.index, g.n_D_union, marker="o", ms=4, lw=1.6, color=CAT3,
            label="$|D_{union}|$ (post per-table reduction)")
    style_ax(ax)
    ax.set_xlabel("top_n (HNSW probe breadth)")
    ax.set_ylabel("mean columns retrieved (per query)")
    ax.set_title("%s: retrieval yield vs. top_n ($\\sigma$=0.30)" % tier)
    ax.legend(loc="upper left", frameon=False, fontsize=9)
    save(fig, "columns_retrieved_vs_topn_%s" % tier)


def plot_stage_breakdown_vs_topn(df, tier):
    g = df.groupby("top_n")[[f for f, _ in STAGE_FIELDS]].mean()
    fig, ax = plt.subplots(figsize=(6.8, 4.4))
    x = g.index.astype(str)
    bottom = None
    for field, label in STAGE_FIELDS:
        vals = g[field].values
        ax.bar(x, vals, bottom=bottom, color=STAGE_COLORS[field], label=label,
               width=0.6, zorder=3)
        bottom = vals if bottom is None else bottom + vals
    style_ax(ax)
    ax.set_xlabel("top_n")
    ax.set_ylabel("mean time (s), stacked")
    ax.set_title("%s: stage-time breakdown vs. top_n ($\\sigma$=0.30, mean over $\\alpha$,k)" % tier)
    ax.legend(loc="upper left", frameon=False, fontsize=8.2, ncol=2,
              bbox_to_anchor=(0.0, -0.18))
    save(fig, "stage_breakdown_vs_topn_%s" % tier)


def main():
    tiers = ["tier_10k", "tier_100k"]
    dfs = {}
    for tier in tiers:
        path = os.path.join(RESULTS, "wdc_sweep_%s.csv" % tier)
        if not os.path.isfile(path):
            print("skip %s: %s not found" % (tier, path))
            continue
        dfs[tier] = load_tier(tier)

    for tier, df in dfs.items():
        plot_time_vs_topn_by_alpha(df, tier)
        plot_columns_retrieved_vs_topn(df, tier)
        plot_stage_breakdown_vs_topn(df, tier)

    if len(dfs) == 2:
        top_ns = sorted(set(dfs["tier_10k"]["top_n"].unique()) & set(dfs["tier_100k"]["top_n"].unique()))
        plot_time_vs_k_by_alpha_both_tiers(dfs, top_n=max(top_ns))
        plot_time_vs_k_by_alpha_both_tiers_all_topn(dfs, top_ns)


if __name__ == "__main__":
    main()
