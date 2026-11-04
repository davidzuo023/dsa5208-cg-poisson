#!/usr/bin/env python3
"""Turn benchmark / validation CSVs into the report's figures, LaTeX tables and number macros.

    python3 scripts/make_plots.py --results results/local     (default)
    python3 scripts/make_plots.py --results results/vanda

Inputs : <results>/bench.csv, <results>/stream.csv, <results>/env.txt,
         results/validation/*.csv, results/commvol.csv
Outputs: figures/*.pdf and report/generated/*.tex
Every number quoted in the report text comes from report/generated/macros.tex, so re-running
this script after a cluster run updates figures, tables and numbers together.
"""
import argparse
import math
import os
import re

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.ticker import FixedLocator, FuncFormatter, NullFormatter, NullLocator  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIG = os.path.join(ROOT, "figures")
GEN = os.path.join(ROOT, "report", "generated")

# ---------------------------------------------------------------- style
# report style: sans-serif, light full grid, framed legends, bold figure and panel titles
C = ["#2b6ca8", "#e3792b", "#3a8f63", "#8467b0", "#d2504c", "#7f7f7f", "#b8860b", "#17becf"]
INK, INK2, GRID = "#222222", "#444444", "#e3e3e3"
MARK = ["o", "s", "^", "D", "v", "P", "X", "*"]
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 8, "axes.titlesize": 8.5, "axes.titleweight": "bold",
    "axes.labelsize": 8, "legend.fontsize": 7, "xtick.labelsize": 7, "ytick.labelsize": 7,
    "text.color": INK, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
    "axes.edgecolor": "#b0b0b0", "axes.linewidth": 0.7, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.6, "grid.linestyle": "-", "axes.axisbelow": True, "axes.spines.top": True,
    "axes.spines.right": True, "lines.linewidth": 1.6, "lines.markersize": 4.5,
    "legend.frameon": True, "legend.framealpha": 1.0, "legend.edgecolor": "#cccccc",
    "legend.fancybox": False, "figure.dpi": 150, "savefig.bbox": "tight", "pdf.fonttype": 42,
    "xtick.major.size": 2.5, "ytick.major.size": 2.5,
})


def suptitle(fig, text, sub=None):
    """bold figure title (and an optional grey subtitle) above all panels; call after tight_layout"""
    h = fig.get_figheight()
    y = 1.0 + 0.04 / h
    if sub:
        fig.text(0.5, y, sub, ha="center", va="bottom", fontsize=6.8, color="#666666")
        y += 0.17 / h
    fig.text(0.5, y, text, ha="center", va="bottom", fontsize=9, fontweight="bold")


def save(fig, name):
    os.makedirs(FIG, exist_ok=True)
    fig.savefig(os.path.join(FIG, name + ".pdf"))
    plt.close(fig)


def log2_axis(ax, values):
    ax.set_xscale("log", base=2)
    vals = sorted(set(int(v) for v in values))
    ax.xaxis.set_major_locator(FixedLocator(vals))
    ax.xaxis.set_minor_locator(NullLocator())
    # walk down from the largest value and label a tick only if it is at most 1/1.4 of the
    # last labelled one: of two close values (32/36, 64/72) the larger one -- a full socket
    # or node -- keeps its label; the marks themselves all stay
    labels, last = [""] * len(vals), None
    for k in range(len(vals) - 1, -1, -1):
        if last is None or last / vals[k] >= 1.4:
            labels[k] = str(vals[k])
            last = vals[k]
    ax.set_xticklabels(labels)
    ax.xaxis.set_minor_formatter(NullFormatter())
    if len(vals) == 1:
        ax.set_xlim(vals[0] / 1.6, vals[0] * 1.6)


def plain_log_y(ax):
    """log y-axis with plain-number labels (0.2, 1, 10) and no minor labels"""
    ax.yaxis.set_major_locator(FixedLocator([m * 10.0 ** e for e in range(-3, 5) for m in (1, 2, 5)]))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    ax.yaxis.set_minor_locator(NullLocator())


def tex_escape(s):
    return str(s).replace("_", r"\_").replace("%", r"\%").replace("&", r"\&").replace("#", r"\#")


def fmt(x, nd=2):
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "--"
    return f"{x:.{nd}f}"


def sci(x, nd=1):
    """1.2e-05 -> 1.2\\times10^{-5} (math mode)"""
    if x == 0:
        return "0"
    e = int(math.floor(math.log10(abs(x))))
    m = round(x / 10 ** e, nd)
    if abs(m) >= 10:
        m, e = m / 10, e + 1
    return rf"{m:.{nd}f}\times10^{{{e}}}"


# ---------------------------------------------------------------- data
def load_bench(path):
    df = pd.read_csv(path)
    # fixed-N experiments must all use the strong-scaling grid of the current run (NS in
    # env.txt; otherwise the most recent strong row); rows from a run with another NS
    # (e.g. an earlier job appended with APPEND=1) would otherwise mix into the plots
    strong = df[df.exp == "strong"]
    if not strong.empty:
        ns = int(strong.N0.iloc[-1])
        env = os.path.join(os.path.dirname(path), "env.txt")
        if os.path.exists(env):
            m = re.search(r"\bNS=(\d+)", open(env).read())
            if m and (df.N0 == int(m.group(1))).any():
                ns = int(m.group(1))
        fixed = df.exp.isin(["serial", "strong", "decomp", "comm", "alg", "hybrid", "solve"])
        df = df[~fixed | (df.N0 == ns)]
    keys = ["exp", "code", "alg", "comm", "decomp", "P", "T", "N0"]
    # best (minimum) time per iteration over the repetitions; keep that run's breakdown
    idx = df.groupby(keys)["t_iter"].idxmin()
    best = df.loc[idx].copy()
    spread = df.groupby(keys)["t_iter"].agg(lambda s: (s.max() - s.min()) / s.min()).rename("spread")
    nreps = df.groupby(keys)["t_iter"].size().rename("nreps")
    best = best.merge(spread.reset_index(), on=keys).merge(nreps.reset_index(), on=keys)
    for c in ["t_spmv", "t_halo", "t_reduce", "t_vec"]:
        best[c + "_it"] = best[c] / best["iters"]
    return best.sort_values(["exp", "P", "T"]).reset_index(drop=True)


def machine_info(results):
    info = {"name": os.path.basename(os.path.normpath(results)), "cpu": "unknown CPU", "nodes": 1, "cpn": 1}
    p = os.path.join(results, "env.txt")
    if os.path.exists(p):
        txt = open(p).read()
        m = re.search(r"Model name:\s*(.+)", txt)
        if m:
            info["cpu"] = m.group(1).strip()
        m = re.search(r"nodes:\s*(\d+)\s+cores/node:\s*(\d+)", txt)
        if m:
            info["nodes"], info["cpn"] = int(m.group(1)), int(m.group(2))
        m = re.search(r"\(Open MPI\)\s*([\d.]+)", txt)
        if m:
            info["mpi"] = "Open MPI " + m.group(1)
        else:
            m = re.search(r"Intel\(R\) MPI Library.*?Version\s*([\w.]+)", txt)
            if m:
                info["mpi"] = "Intel MPI " + m.group(1)
            else:
                m = re.search(r"(?:mpich|hydra).*?version:?\s*([\d.]+)", txt, re.I | re.S)
                if m:
                    info["mpi"] = "MPICH " + m.group(1)
        m = re.search(r"gcc.*?(\d+\.\d+\.\d+)", txt)
        if m:
            info["gcc"] = m.group(1)
    return info


# ---------------------------------------------------------------- figures
def fig_strong(b, macros, tables):
    s = b[(b.exp == "strong")].sort_values("P")
    ser = b[b.exp == "serial"]
    if s.empty:
        return
    t_ser = float(ser.t_iter.iloc[0]) if not ser.empty else float(s[s.P == 1].t_iter.iloc[0])
    t_p1 = float(s[s.P == 1].t_iter.iloc[0]) if (s.P == 1).any() else t_ser
    s = s.assign(speedup=t_ser / s.t_iter, eff=t_ser / s.t_iter / s.P)
    Ps = s.P.values

    fig, ax = plt.subplots(1, 2, figsize=(6.6, 2.6))
    ax[0].plot(Ps, Ps, color=INK2, lw=1.0, ls="--", label="ideal")
    ax[0].plot(Ps, s.speedup, color=C[0], marker="o", label="MPI, 3D, overlap, CG")
    log2_axis(ax[0], Ps)
    ax[0].set_yscale("log", base=2)
    ticks = [2.0 ** e for e in range(-2, 13)]
    ax[0].yaxis.set_major_locator(FixedLocator(ticks))
    ax[0].yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    ax[0].yaxis.set_minor_locator(NullLocator())
    ax[0].set_ylim(min(0.5, s.speedup.min() / 1.3), max(Ps) * 1.3)
    ax[0].set_xlabel("MPI processes $P$")
    ax[0].set_ylabel("speedup vs. serial code")
    ax[0].set_title("(a) Speedup over the serial code")
    ax[0].legend(loc="upper left")
    ax[1].plot(Ps, 100 * s.eff, color=C[0], marker="o")
    ax[1].axhline(100, color=INK2, lw=1.0, ls="--")
    log2_axis(ax[1], Ps)
    ax[1].set_ylim(0, max(110, 100 * s.eff.max() * 1.1))
    ax[1].set_xlabel("MPI processes $P$")
    ax[1].set_ylabel("parallel efficiency (%)")
    ax[1].set_title("(b) Parallel efficiency $S(P)/P$")
    fig.tight_layout()
    suptitle(fig, "Strong scaling of the baseline solver", f"pure MPI, 3D blocks, overlapped halo, classic CG, "
             f"$N={int(s.N0.iloc[0])}^3$, {int(s.iters.iloc[0])} iterations")
    save(fig, "strong")

    # breakdown: share of the iteration spent in each phase (100% stacked)
    fig, ax = plt.subplots(figsize=(6.6, 2.35))
    comps = [("t_spmv_it", "stencil $A\\mathbf{p}$", C[0]), ("t_vec_it", "vector updates + local dots", C[2]),
             ("t_halo_it", "halo exchange", C[1]), ("t_reduce_it", "MPI_Allreduce (incl. waiting)", C[3])]
    tot = sum(s[c] for c, _, _ in comps)
    xs = range(len(s))
    bottom = [0.0] * len(s)
    for col, lab, colr in comps:
        vals = (100 * s[col] / tot).tolist()
        ax.bar(xs, vals, bottom=bottom, color=colr, width=0.62, label=lab, edgecolor="white", linewidth=1.0)
        bottom = [a + v for a, v in zip(bottom, vals)]
    ax.set_xticks(list(xs))
    ax.set_xticklabels([str(int(p)) for p in Ps])
    ax.set_xlabel("MPI processes $P$ (grid $%d^3$)" % int(s.N0.iloc[0]))
    ax.set_ylabel("share of iteration time (%)")
    ax.set_ylim(0, 100)
    ax.grid(axis="x", visible=False)
    ax.legend(ncol=4, loc="lower center", bbox_to_anchor=(0.5, 1.01), fontsize=7)
    fig.tight_layout()
    suptitle(fig, "Where a CG iteration spends its time")
    save(fig, "breakdown")

    rows = []
    for _, r in s.iterrows():
        comm = (r.t_halo_it + r.t_reduce_it) / r.t_iter * 100
        rows.append(f"{int(r.P)} & {int(r.px)}$\\times${int(r.py)}$\\times${int(r.pz)} & {1e3 * r.t_iter:.2f} & "
                    f"{r.speedup:.2f} & {100 * r.eff:.0f} & {comm:.1f} & {r.mlups:.0f} \\\\")
    tables["strong"] = rows
    pmax = s.iloc[-1]
    macros.update({
        "SerialTiter": fmt(1e3 * t_ser, 1), "PoneTiter": fmt(1e3 * t_p1, 1),
        "PoneComparison": (f"{100 * (t_p1 / t_ser - 1):.1f}\\% slower than" if t_p1 >= t_ser
                           else f"{100 * (1 - t_p1 / t_ser):.1f}\\% faster than"),
        "Pmax": str(int(pmax.P)), "SpeedupPmax": fmt(pmax.speedup, 2), "EffPmax": fmt(100 * pmax.eff, 0),
        "CommFracPmax": fmt((pmax.t_halo_it + pmax.t_reduce_it) / pmax.t_iter * 100, 1),
        "VecFracSerial": fmt(100 * float(s.iloc[0].t_vec_it / s.iloc[0].t_iter), 0),
        "NS": str(int(s.N0.iloc[0])), "NSdofs": fmt(int(s.N0.iloc[0]) ** 3 / 1e6, 1),
        "Iters": str(int(s.iters.iloc[0])),
        "MedSpread": fmt(100 * b[b.exp != "solve"].spread.median(), 1),
        "Reps": str(int(b.nreps.max())),
    })


def fig_variants(b, macros, tables):
    base = b[(b.exp == "strong")][["P", "t_iter"]].rename(columns={"t_iter": "t_base"})
    v = b[b.exp.isin(["decomp", "comm", "alg"])].merge(base, on="P")
    if v.empty:
        return

    def label(r):
        if r.exp == "decomp":
            return f"{r.decomp} decomposition"
        if r.exp == "comm":
            return "blocking halo"
        return {"cg1": "CG1 (1 Allreduce)", "pipecg": "pipelined CG"}[r.alg] + \
            (" + blocking halo" if r.comm == "blocking" else "")

    v = v.assign(label=v.apply(label, axis=1), rel=v.t_iter / v.t_base)
    Ps = sorted(v.P.unique())
    panels = [("(a) Data distribution and halo strategy",
               ["1D decomposition", "2D decomposition", "blocking halo"]),
              ("(b) CG formulation (3D blocks, overlapped halo)",
               ["CG1 (1 Allreduce)", "pipelined CG", "pipelined CG + blocking halo"])]
    fig, axs = plt.subplots(1, 2, figsize=(6.6, 2.45), sharey=True)
    for ax, (title, labs) in zip(axs, panels):
        w = 0.8 / len(labs)
        for k, lab in enumerate(labs):
            d = v[v.label == lab].set_index("P").reindex(Ps)
            xs = [i + (k - (len(labs) - 1) / 2) * w for i in range(len(Ps))]
            ax.bar(xs, d.rel, width=w * 0.92, color=C[k], label=lab, edgecolor="white", linewidth=0.8)
        ax.axhline(1.0, color=INK, lw=1.0)
        ax.set_xticks(range(len(Ps)))
        ax.set_xticklabels([str(int(p)) for p in Ps])
        ax.set_xlabel("MPI processes $P$")
        ax.set_title(title)
        ax.grid(axis="x", visible=False)
        ax.legend(loc="upper left", fontsize=7)
    axs[0].set_ylabel("time / baseline time")
    lo = min(0.8, v.rel.min() * 0.95)
    hi = max(1.25, v.rel.max() * 1.25)
    axs[0].set_ylim(lo, hi)
    fig.tight_layout()
    suptitle(fig, "One design decision at a time", "time per iteration relative to the baseline "
             "(3D blocks, overlapped halo, classic CG) at the same $P$; below 1 is faster")
    save(fig, "variants")

    order = ["3D, overlap, CG (baseline)", "1D decomposition", "2D decomposition", "blocking halo",
             "CG1 (1 Allreduce)", "pipelined CG", "pipelined CG + blocking halo"]
    b0 = b[b.exp == "strong"].set_index("P")
    # the table shows at most four process counts (evenly spread, smallest and largest
    # included) so that it fits the page; the figure shows all of them
    Pt = Ps if len(Ps) <= 4 else [Ps[round(i * (len(Ps) - 1) / 3)] for i in range(4)]
    rows = []
    for lab in order:
        cells = []
        for P in Pt:
            if lab.endswith("(baseline)"):
                cells.append(f"{1e3 * b0.loc[P, 't_iter']:.2f}")
            else:
                d = v[(v.label == lab) & (v.P == P)]
                cells.append("--" if d.empty else f"{1e3 * d.t_iter.iloc[0]:.2f} ({d.rel.iloc[0]:.2f})")
        rows.append(tex_escape(lab) + " & " + " & ".join(cells) + r" \\")
    tables["variants"] = rows
    tables["variants_cols"] = Pt
    pm = max(Ps)

    def rel(lab):
        d = v[(v.label == lab) & (v.P == pm)]
        return float(d.rel.iloc[0]) if not d.empty else float("nan")

    macros.update({"VarP": str(int(pm)), "RelOneD": fmt(rel("1D decomposition")),
                   "RelBlocking": fmt(rel("blocking halo")), "RelCGone": fmt(rel("CG1 (1 Allreduce)")),
                   "RelPipe": fmt(rel("pipelined CG")),
                   "RelPipeBlock": fmt(rel("pipelined CG + blocking halo"))})


def fig_commvol(path, macros):
    if not os.path.exists(path):
        return
    cv = pd.read_csv(path)
    N = int(cv.N0.iloc[0])
    fig, ax = plt.subplots(1, 2, figsize=(6.6, 2.6))
    for k, d in enumerate(["1D", "2D", "3D"]):
        s = cv[cv.decomp == d].sort_values("P")
        ax[0].plot(s.P, s.halo_bytes_iter_max / 2 ** 20, color=C[k], marker=MARK[k], label=d)
        ax[1].plot(s.P, s.halo_bytes_iter_tot / 2 ** 20, color=C[k], marker=MARK[k], label=d)
    for a in ax:
        log2_axis(a, cv.P)
        a.set_yscale("log")
        plain_log_y(a)
        a.set_xlabel("MPI processes $P$")
    ax[0].set_ylabel("MiB sent per iteration")
    ax[0].set_title("(a) Busiest process")
    ax[1].set_title("(b) All processes together")
    ax[1].legend(title="decomposition", loc="upper left")
    fig.tight_layout()
    suptitle(fig, "Halo-exchange volume per CG iteration", f"$N={N}^3$, measured by the solver's byte counters")
    save(fig, "commvol")
    s64 = cv[cv.P == cv.P.max()].set_index("decomp")
    macros.update({"CVP": str(int(cv.P.max())), "CVN": str(N),
                   "CVoneD": fmt(s64.loc["1D", "halo_bytes_iter_max"] / 2 ** 20, 2),
                   "CVthreeD": fmt(s64.loc["3D", "halo_bytes_iter_max"] / 2 ** 20, 2),
                   "CVratio": fmt(s64.loc["1D", "halo_bytes_iter_max"] / s64.loc["3D", "halo_bytes_iter_max"], 1),
                   "CVmsgsOneD": str(int(s64.loc["1D", "msgs_iter_max"])),
                   "CVmsgsThreeD": str(int(s64.loc["3D", "msgs_iter_max"]))})


def fig_hybrid(b, macros, tables):
    h = b[b.exp == "hybrid"].sort_values("T")
    if h.empty:
        return
    fig, ax = plt.subplots(figsize=(3.4, 2.2))
    labels = [f"{int(p)}$\\times${int(t)}" for p, t in zip(h.P, h['T'])]
    ax.bar(range(len(h)), 1e3 * h.t_iter, color=C[0], width=0.6)
    ax.set_xticks(range(len(h)))
    ax.set_xticklabels(labels, rotation=45 if len(h) > 6 else 0)
    ax.set_xlabel("MPI ranks $\\times$ OpenMP threads")
    ax.set_ylabel("ms per iteration")
    ax.set_title(f"Processes × threads on {int(h.P.iloc[0] * h['T'].iloc[0])} cores")
    ax.grid(axis="x", visible=False)
    fig.tight_layout()
    save(fig, "hybrid")
    best = h.loc[h.t_iter.idxmin()]
    pure = h[h["T"] == 1]
    tables['hybrid'] = [f"{int(r.P)} & {int(r['T'])} & {1e3 * r.t_iter:.2f} & {1e3 * r.t_halo_it:.2f} & "
                        f"{1e3 * r.t_reduce_it:.2f} & {r.halo_bytes_iter_tot / 2 ** 20:.2f} \\\\"
                        for _, r in h.iterrows()]
    worst = h.loc[h.t_iter.idxmax()]
    res = b[b.exp != "solve"].spread.median()
    d_all = float(worst.t_iter / best.t_iter - 1)
    d_pure = float(pure.t_iter.iloc[0] / best.t_iter - 1) if not pure.empty else float("nan")
    cores = int(best.P * best["T"])
    if d_all < res:
        sent = ("All splits run within %.1f\\%% of each other, inside the measurement resolution, so on "
                "%d cores neither effect dominates." % (100 * d_all, cores))
    elif int(best["T"]) == 1:
        sent = ("Pure MPI (%d$\\times$1) is the fastest split, and the slowest, %d$\\times$%d, takes %.0f\\%% "
                "longer." % (int(best.P), int(worst.P), int(worst["T"]), 100 * d_all))
    elif d_pure < res:
        sent = ("Pure MPI is as fast as the best split (%d$\\times$%d, %.1f\\%% faster, inside the measurement "
                "resolution), and the slowest split, %d$\\times$%d, takes %.0f\\%% longer." % (
                    int(best.P), int(best["T"]), 100 * d_pure, int(worst.P), int(worst["T"]), 100 * d_all))
    else:
        sent = ("The fastest split is %d$\\times$%d, %.2f$\\times$ faster than pure MPI." % (
            int(best.P), int(best["T"]), 1 + d_pure))
    macros["HybSentence"] = sent
    macros.update({"HybBest": f"{int(best.P)}$\\times${int(best['T'])}",
                   "HybCores": str(int(h.P.iloc[0] * h["T"].iloc[0])),
                   "HybBestVsPure": fmt(float(pure.t_iter.iloc[0] / best.t_iter), 2) if not pure.empty else "--"})


def fig_weak(b, macros):
    w = b[b.exp == "weak"].sort_values("P")
    if w.empty:
        return
    t1 = float(w[w.P == w.P.min()].t_iter.iloc[0])
    # normalise by points per rank: the global N is rounded, so per-rank work varies slightly
    tpp = w.t_iter / w.nloc_max
    eff = (t1 / float(w.nloc_max.iloc[0])) / tpp
    fig, ax = plt.subplots(figsize=(3.4, 2.5))
    ax.plot(w.P, 100 * eff, color=C[0], marker="o")
    ax.axhline(100, color=INK2, ls="--", lw=1.0)
    log2_axis(ax, w.P)
    ax.set_ylim(0, 110)
    ax.set_xlabel("MPI processes $P$")
    ax.set_ylabel("weak-scaling efficiency (%)")
    ax.set_title(f"Weak scaling, $\\approx{round(w.nloc_avg.iloc[0] ** (1 / 3))}^3$ points per process")
    fig.tight_layout()
    save(fig, "weak")
    macros.update({"WeakEffPmax": fmt(100 * float(eff.iloc[-1]), 0), "WeakPmax": str(int(w.P.max())),
                   "WeakLocal": str(round(w.nloc_avg.iloc[0] ** (1 / 3)))})


def fig_bandwidth(b, results, macros):
    p = os.path.join(results, "stream.csv")
    s = b[b.exp == "strong"].sort_values("P")
    if not os.path.exists(p) or s.empty:
        return
    st = pd.read_csv(p)
    cpn = machine_info(results)["cpn"]
    s = s[s.P <= cpn]
    if s.empty:
        return
    # actual DRAM traffic on both sides: classic CG moves 88 B/point of explicit loads and
    # stores plus 8 B write-allocate for the stencil output (a pure write); the STREAM triad
    # reports 24 B/element but moves 32 B (its store is a pure write too)
    BYTES = 96.0
    eff_bw = s.N0 ** 3 * BYTES / s.t_iter / 1e9
    st = st.assign(triad_GBs=st.triad_GBs * 32.0 / 24.0)
    fig, ax = plt.subplots(figsize=(3.4, 2.5))
    ax.plot(st.threads, st.triad_GBs, color=INK2, marker="s", ls="--", label="STREAM triad (32 B/element)")
    ax.plot(s.P, eff_bw, color=C[0], marker="o", label="CG solver (96 B/point)")
    log2_axis(ax, list(st.threads) + list(s.P))
    ax.set_ylim(0, max(st.triad_GBs.max(), eff_bw.max()) * 1.2)
    ax.set_xlabel("cores used on one node")
    ax.set_ylabel("memory bandwidth (GB/s)")
    ax.set_title("Memory bandwidth: solver vs. STREAM")
    ax.legend(loc="upper left")
    fig.tight_layout()
    save(fig, "bandwidth")
    macros.update({"StreamOne": fmt(float(st.triad_GBs.iloc[0]), 1), "StreamAll": fmt(float(st.triad_GBs.iloc[-1]), 1),
                   "StreamThreads": str(int(st.threads.iloc[-1])),
                   "StreamRatio": fmt(float(st.triad_GBs.iloc[-1] / st.triad_GBs.iloc[0]), 1),
                   "CGbwOne": fmt(float(eff_bw.iloc[0]), 1), "CGbwAll": fmt(float(eff_bw.iloc[-1]), 1)})


def fig_validation(vdir, macros, tables):
    conv = os.path.join(vdir, "convergence.csv")
    if os.path.exists(conv):
        cv = pd.read_csv(conv)
        fig, ax = plt.subplots(1, 2, figsize=(6.6, 2.6))
        groups = [("serial", "cg", "serial CG", C[0], "o", "-"), ("mpi", "cg", "MPI CG, $P=4$", C[1], "x", "none"),
                  ("mpi", "pipecg", "MPI pipelined CG, $P=4$", C[2], "+", "none")]
        for code, alg, lab, col, mk, ls in groups:
            d = cv[(cv.code == code) & (cv.alg == alg)].sort_values("N0")
            h = 1.0 / (d.N0 + 1)
            ax[0].plot(h, d.err_l2, color=col, marker=mk, ls=ls, label=lab, markersize=6 if mk != "o" else 4)
        d = cv[(cv.code == "serial")].sort_values("N0")
        h = 1.0 / (d.N0 + 1)
        ref = d.err_l2.iloc[-1] * (h / h.iloc[-1]) ** 2
        ax[0].plot(h, ref * 0.5, color=INK2, ls="--", lw=1.0, label="$O(h^2)$ reference")
        ax[0].set_xscale("log")
        ax[0].set_yscale("log")
        ax[0].xaxis.set_major_locator(FixedLocator([0.01, 0.02, 0.05]))
        ax[0].xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
        ax[0].xaxis.set_minor_formatter(NullFormatter())
        ax[0].set_xlabel("grid spacing $h$")
        ax[0].set_ylabel("$\\|u_h - u\\|_{L^2}$")
        ax[0].set_title("(a) Discretisation error")
        ax[0].legend(loc="upper left", fontsize=7)
        for k, alg in enumerate(["cg", "cg1", "pipecg"]):
            p = os.path.join(vdir, f"hist_{alg}.csv")
            if not os.path.exists(p):
                continue
            hh = pd.read_csv(p)
            name = {"cg": "CG", "cg1": "CG1", "pipecg": "pipelined CG"}[alg]
            lw = 2.6 if alg == "cg" else 1.3   # CG and CG1 coincide: draw CG wide underneath
            ax[1].semilogy(hh.iter, hh.relres_true, color=C[k], lw=lw, label=f"{name}: true", zorder=3 - k / 10)
            ax[1].semilogy(hh.iter, hh.relres_recursive, color=C[k], ls=":", lw=lw * 0.8,
                           label=f"{name}: recursive")
        ax[1].set_xlabel("iteration")
        ax[1].set_ylabel("$\\|r\\|/\\|b\\|$")
        ax[1].set_title("(b) Residual histories, $N=64^3$, $P=4$")
        ax[1].set_ylim(1e-16, 10)
        ax[1].legend(fontsize=6.3, ncol=1, loc="upper right")
        fig.tight_layout()
        suptitle(fig, "Correctness of the parallel solver")
        save(fig, "validation")

        ser = cv[cv.code == "serial"].set_index("N0")
        par = cv[(cv.code == "mpi") & (cv.alg == "cg")].set_index("N0")
        rows, prev = [], None
        for N in ser.index:
            e = ser.loc[N, "err_l2"]
            order = "--" if prev is None else f"{math.log(prev[1] / e) / math.log((prev[0] + 1) / (N + 1)) * -1:.3f}"
            pe = par.loc[N, "err_l2"] if N in par.index else float("nan")
            rows.append(f"{N} & ${sci(1 / (N + 1), 2)}$ & {int(ser.loc[N, 'iters'])} & ${sci(e, 3)}$ & "
                        f"${sci(pe, 3)}$ & {order} \\\\")
            prev = (N, e)
        tables["convergence"] = rows

    var = os.path.join(vdir, "variants.csv")
    if os.path.exists(var):
        v = pd.read_csv(var)
        ref = pd.read_csv(os.path.join(vdir, "serial_ref.csv")).iloc[0]
        rows = []
        for (alg, comm), d in v.groupby(["alg", "comm"], sort=False):
            rows.append(f"{ {'cg': 'CG', 'cg1': 'CG1', 'pipecg': 'pipelined CG'}[alg]} & {comm} & {len(d)} & "
                        f"{d.P.min()}--{d.P.max()} & {d['T'].min()}--{d['T'].max()} & "
                        f"{int(d.iters.min())}--{int(d.iters.max())} & ${sci(d.maxdiff_vs_serial.max(), 1)}$ & "
                        f"${sci(d.true_relres.max(), 2)}$ \\\\")
        tables["validation"] = rows
        edge = os.path.join(vdir, "edge.csv")
        ne = len(pd.read_csv(edge)) if os.path.exists(edge) else 0
        cg = v[v.alg != "pipecg"].maxdiff_vs_serial.max()
        pp = v[v.alg == "pipecg"].maxdiff_vs_serial.max()
        macros.update({"ValRuns": str(len(v)), "ValEdge": str(ne), "ValN": str(int(ref.N0)),
                       "ValIters": str(int(ref.iters)), "ValWorstCG": f"${sci(cg, 1)}$",
                       "ValWorstPipe": f"${sci(pp, 1)}$", "ValErrMax": f"${sci(ref.err_max, 2)}$"})
    for alg in ["cg", "pipecg"]:
        p = os.path.join(vdir, f"hist_{alg}.csv")
        if os.path.exists(p):
            hh = pd.read_csv(p)
            tag = "Pipe" if alg == "pipecg" else "CG"
            macros["Floor" + tag] = f"${sci(hh.relres_true.min(), 0)}$"
            gap = hh[hh.relres_true > 3 * hh.relres_recursive]
            if not gap.empty:
                macros["Split" + tag] = f"${sci(gap.relres_recursive.iloc[0], 0)}$"
            macros["Final" + tag] = f"${sci(hh.relres_true.iloc[-1], 0)}$"
            macros["Last" + tag] = str(int(hh.iter.iloc[-1]))
    if "FloorCG" in macros and "FloorPipe" in macros:
        a = pd.read_csv(os.path.join(vdir, "hist_cg.csv")).relres_true.min()
        b = pd.read_csv(os.path.join(vdir, "hist_pipecg.csv")).relres_true.min()
        macros['FloorRatio'] = f"{b / a:.0f}"


def tab_solve(b, macros, tables):
    s = b[b.exp == "solve"]
    if s.empty:
        return
    name = {"cg": "CG", "cg1": "CG1", "pipecg": "pipelined CG"}
    tables['solve'] = [f"{name[r.alg]} & {int(r.P)} & {int(r.iters)} & {r.t_solve:.2f} & {1e3 * r.t_iter:.2f} & "
                       f"${sci(r.true_relres, 2)}$ & ${sci(r.err_max, 2)}$ \\\\" for _, r in s.iterrows()]
    if (s.alg == "cg").any():
        macros["SolveIters"] = str(int(s[s.alg == "cg"].iters.iloc[0]))
    pp, cg = s[s.alg == "pipecg"], s[s.alg == "cg"]
    if not pp.empty and not cg.empty:
        macros['SolvePipeTrue'] = f"${sci(float(pp.true_relres.iloc[0]), 1)}$"
        macros['SolveCGTrue'] = f"${sci(float(cg.true_relres.iloc[0]), 1)}$"
        macros["SolvePipeRatio"] = fmt(float(pp.true_relres.iloc[0]) / 1e-8, 1)


# every macro the report uses defaults to "--" if its experiment is missing; the list is read
# from report.tex (capitalised control sequences minus LaTeX's own)
FIGURES = ["strong", "breakdown", "variants", "commvol", "hybrid", "weak", "bandwidth", "validation"]
LATEX_OWN = {"Cref", "Omega", "Comment", "State", "For", "EndFor", "While", "EndWhile", "If", "EndIf"}


def report_macros():
    path = os.path.join(ROOT, "report", "report.tex")
    if not os.path.exists(path):
        return []
    return sorted(set(re.findall(r"\\([A-Z][A-Za-z]+)", open(path).read())) - LATEX_OWN)


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=os.path.join(ROOT, "results", "local"))
    ap.add_argument("--validation", default=os.path.join(ROOT, "results", "validation"))
    a = ap.parse_args()
    os.makedirs(GEN, exist_ok=True)
    os.makedirs(FIG, exist_ok=True)
    # never let a figure from an earlier dataset survive into the report
    for f in FIGURES:
        path = os.path.join(FIG, f + ".pdf")
        if os.path.exists(path):
            os.remove(path)
    macros, tables = {}, {}
    info = machine_info(a.results)
    macros.update({"MachineName": tex_escape(info["name"]), "MachineCPU": tex_escape(info["cpu"]),
                   "MachineNodes": str(info["nodes"]), "MachineCPN": str(info["cpn"]),
                   "MachineCores": str(info["nodes"] * info["cpn"]),
                   "MachineMPI": tex_escape(info.get("mpi", "MPI")), "MachineGCC": info.get("gcc", "")})
    bench = os.path.join(a.results, "bench.csv")
    if os.path.exists(bench):
        b = load_bench(bench)
        b.to_csv(os.path.join(a.results, "bench_best.csv"), index=False)
        fig_strong(b, macros, tables)
        fig_variants(b, macros, tables)
        fig_hybrid(b, macros, tables)
        fig_weak(b, macros)
        fig_bandwidth(b, a.results, macros)
        tab_solve(b, macros, tables)
    fig_commvol(os.path.join(ROOT, "results", "commvol.csv"), macros)
    fig_validation(a.validation, macros, tables)

    # placeholders for experiments that were not run (keeps the report compilable)
    defaults = {k for k in report_macros() if k not in macros}
    for t in ["validation", "convergence", "strong", "variants", "hybrid", "solve"]:
        tables.setdefault(t, [])
    if "variants_cols" not in tables:
        tables["variants_cols"] = []
    for f in os.listdir(GEN):
        if f.endswith(".tex"):
            os.remove(os.path.join(GEN, f))
    with open(os.path.join(GEN, "macros.tex"), "w") as f:
        f.write("% generated by scripts/make_plots.py -- do not edit\n")
        f.write("\\newif\\iflocalrun\n\\localrun%s\n" % ("true" if info["name"] == "local" else "false"))
        for k, v in sorted(macros.items()):
            f.write(f"\\newcommand{{\\{k}}}{{{v}}}\n")
        # placeholders only for names LaTeX does not already know (\providecommand)
        for k in sorted(defaults):
            f.write(f"\\providecommand{{\\{k}}}{{--}}\n")
    for name, rows in tables.items():
        if name.endswith("_cols"):
            continue
        with open(os.path.join(GEN, f"tab_{name}.tex"), "w") as f:
            f.write("\n".join(rows) + "\n")
    for f in FIGURES:  # experiments without data get an explicit placeholder
        if not os.path.exists(os.path.join(FIG, f + ".pdf")):
            fig, ax = plt.subplots(figsize=(3.4, 1.2))
            ax.axis("off")
            ax.text(0.5, 0.5, f"{f}: no data in this results directory", ha="center", va="center", color=INK2)
            save(fig, f)
    with open(os.path.join(GEN, "tab_variants_head.tex"), "w") as f:
            Ps = tables["variants_cols"]
            f.write("\\begin{tabular}{l" + "r" * max(len(Ps), 1) + "}\n\\toprule\n")
            cols = " & ".join(f"$P={p}$" for p in Ps) if Ps else "not measured"
            f.write("variant & " + cols + " \\\\\n\\midrule\n")
    print("figures ->", FIG, "| tables/macros ->", GEN)
    for k in sorted(macros):
        print(f"  {k} = {macros[k]}")


if __name__ == "__main__":
    main()
