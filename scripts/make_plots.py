#!/usr/bin/env python3
"""Turn the Vanda measurements into the report's figures, LaTeX tables and number macros.

    python3 scripts/make_plots.py            (defaults below; `make report` calls this)

Inputs (all under results/):
  vanda256/     multi-node pure-MPI and hybrid runs, 8 nodes x 32 cores    (pbs/vanda256.pbs)
  vanda_node/   STREAM, serial baseline and quick validation on 32 cores     (pbs/vanda_node.pbs)
  vanda_sizes/  serial problem-size study                                    (pbs/vanda_sizes.pbs)
  vanda_gpu/    GPU size sweep and time to solution on one A40               (pbs/vanda_gpu.pbs)
  vanda_gpu_val/ GPU validation, P = 1, 2, 4, 8 ranks on one A40
  validation/   full correctness suite;  commvol.csv  halo volumes
Outputs: figures/*.pdf and report/generated/*.tex. Every number quoted in the report text comes
from report/generated/macros.tex, so re-running this script updates figures, tables and text.
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
RES = os.path.join(ROOT, "results")

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

BYTES_CPU = 96.0   # classic CG per point and iteration: 88 B of loads/stores + 8 B write-allocate
BYTES_GPU = 88.0   # GPUs write full sectors without reading them first: no write-allocate
A40_PEAK = 696.0   # GB/s, NVIDIA A40 data sheet


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


def log2_axis(ax, values, axis="x"):
    vals = sorted(set(int(v) for v in values))
    a = ax.xaxis if axis == "x" else ax.yaxis
    (ax.set_xscale if axis == "x" else ax.set_yscale)("log", base=2)
    a.set_major_locator(FixedLocator(vals))
    a.set_minor_locator(NullLocator())
    # label a tick only if it is at most 1/1.4 of the last labelled one (walking down), so
    # close values (192/224/256) do not collide; the marks themselves all stay
    labels, last = [""] * len(vals), None
    for k in range(len(vals) - 1, -1, -1):
        if last is None or last / vals[k] >= 1.4:
            labels[k] = str(vals[k])
            last = vals[k]
    (ax.set_xticklabels if axis == "x" else ax.set_yticklabels)(labels)
    a.set_minor_formatter(NullFormatter())


def plain_log(ax, axis="y", lo=-3, hi=5):
    a = ax.yaxis if axis == "y" else ax.xaxis
    a.set_major_locator(FixedLocator([m * 10.0 ** e for e in range(lo, hi) for m in (1, 2, 5)]))
    a.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    a.set_minor_locator(NullLocator())


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
def best_of(df, keys):
    """fastest repetition per configuration (with that run's phase breakdown), plus the spread"""
    idx = df.groupby(keys)["t_iter"].idxmin()
    best = df.loc[idx].copy()
    agg = df.groupby(keys)["t_iter"].agg(spread=lambda s: (s.max() - s.min()) / s.min(), nreps="size",
                                         t_med="median").reset_index()
    best = best.merge(agg, on=keys)
    for c in ["t_spmv", "t_halo", "t_reduce", "t_vec"]:
        best[c + "_it"] = best[c] / best["iters"]
    return best.reset_index(drop=True)


KEYS = ["exp", "code", "alg", "comm", "decomp", "P", "T", "N0"]


def env_info(d):
    info = {}
    p = os.path.join(d, "env.txt")
    if not os.path.exists(p):
        return info
    txt = open(p).read()
    for key, pat in [("cpu", r"Model name:\s*(.+)"), ("mpi", r"\(Open MPI\)\s*([\w.]+)"),
                     ("gcc", r"gcc \(GCC\)\s*([\d.]+)"), ("gpu", r"(NVIDIA [^,]+), (\d+) MiB, ([\d.]+)"),
                     ("nvc", r"nvc ([\d.]+-\d+)"), ("module", r"toolchain module:\s*(\S+)"),
                     ("cps", r"Core\(s\) per socket:\s*(\d+)"), ("sockets", r"Socket\(s\):\s*(\d+)")]:
        m = re.search(pat, txt)
        if m:
            info[key] = m.groups() if key == "gpu" else m.group(1).strip()
    m = re.search(r"nodes:\s*(\d+)\s+cores/node:\s*(\d+)", txt)
    if m:
        info["nodes"], info["cpn"] = int(m.group(1)), int(m.group(2))
    info["tmrsh"] = "pbs_tmrsh" in txt
    return info


# ---------------------------------------------------------------- serial problem size
def fig_sizes(d, macros, tables):
    p = os.path.join(d, "bench.csv")
    if not os.path.exists(p):
        return None
    s = pd.read_csv(p)
    it = s[s.exp == "sizes_iter"].groupby("N0").agg(t=("t_iter", "min"), t_med=("t_iter", "median")).reset_index()
    so = s[s.exp == "sizes_solve"].groupby("N0").agg(iters=("iters", "max"), ts=("t_solve", "min"),
                                                     err=("err_max", "min"), tr=("true_relres", "min")).reset_index()
    z = it.merge(so, on="N0", how="left")
    z["n"] = z.N0.astype(float) ** 3
    z["nnz"] = 7 * z.N0.astype(float) ** 3 - 6 * z.N0.astype(float) ** 2
    z["nspt"] = 1e9 * z.t / z.n
    z["bw"] = z.n * BYTES_CPU / z.t / 1e9
    fig, ax = plt.subplots(1, 3, figsize=(6.8, 2.35))
    ax[0].loglog(z.n / 1e6, 1e3 * z.t, color=C[0], marker="o", label="measured")
    ref = 1e3 * z.t.iloc[-1] * (z.n / z.n.iloc[-1])
    ax[0].loglog(z.n / 1e6, ref, color=INK2, ls="--", lw=1.0, label=r"$\propto N^3$")
    ax[0].set_xlabel("unknowns $N^3$ (millions)")
    ax[0].set_ylabel("ms per iteration")
    ax[0].set_title("(a) Cost of one iteration")
    plain_log(ax[0], "y")
    ax[0].xaxis.set_major_locator(FixedLocator([0.1, 1, 10, 100]))
    ax[0].xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    ax[0].xaxis.set_minor_formatter(NullFormatter())
    ax[0].legend(loc="upper left")
    ax[1].plot(z.N0, z.iters, color=C[1], marker="s", label="measured")
    k = float((z.iters / z.N0).iloc[-1])
    ax[1].plot(z.N0, k * z.N0, color=INK2, ls="--", lw=1.0, label=rf"${k:.2f}\,N$")
    ax[1].set_xlabel("grid points per direction $N$")
    ax[1].set_ylabel(r"iterations to $\varepsilon=10^{-8}$")
    ax[1].set_title("(b) Iterations")
    ax[1].legend(loc="upper left")
    ax[2].loglog(z.N0, z.ts, color=C[2], marker="^", label="measured")
    ax[2].loglog(z.N0, z.ts.iloc[-1] * (z.N0 / z.N0.iloc[-1]) ** 4, color=INK2, ls="--", lw=1.0,
                 label=r"$\propto N^4$")
    ax[2].set_xlabel("grid points per direction $N$")
    ax[2].set_ylabel("time to solution (s)")
    ax[2].set_title("(c) Time to solution")
    log2_axis(ax[2], [64, 128, 256, 512])
    plain_log(ax[2], "y", -2, 5)
    ax[2].legend(loc="upper left")
    fig.tight_layout()
    suptitle(fig, "Serial solver: cost against problem size", "one core; matrix dimension $N^3$, "
             "$7N^3-6N^2$ nonzeros; 20-iteration timing runs and complete solves")
    save(fig, "sizes")
    tables["sizes"] = [f"{int(r.N0)} & {r.n / 1e6:.2f} & {r.nnz / 1e6:.1f} & {5 * r.n * 8 / 2 ** 20:.0f} & "
                       f"{1e3 * r.t:.2f} & {r.nspt:.2f} & {r.bw:.1f} & {int(r.iters)} & {r.ts:.1f} & "
                       f"${sci(r.err, 2)}$ \\\\" for _, r in z.iterrows()]
    big = z[z.N0 >= 128]
    slope_t = math.log(big.t.iloc[-1] / big.t.iloc[0]) / math.log(big.N0.iloc[-1] / big.N0.iloc[0])
    slope_s = math.log(big.ts.iloc[-1] / big.ts.iloc[0]) / math.log(big.N0.iloc[-1] / big.N0.iloc[0])
    r64, r512 = z.iloc[0], z.iloc[-1]
    macros.update({"SzNmin": str(int(r64.N0)), "SzNmax": str(int(r512.N0)), "SzRuns": str(len(z)),
                   "SzNsSmall": fmt(r64.nspt, 2), "SzNsLarge": fmt(r512.nspt, 2),
                   "SzNsMin": fmt(z.nspt.min(), 2), "SzNsMax": fmt(z.nspt.max(), 2),
                   "SzBwLarge": fmt(r512.bw, 1), "SzSlopeIter": fmt(slope_t, 2), "SzSlopeSolve": fmt(slope_s, 2),
                   "SzItPerN": fmt(k, 2), "SzItMin": str(int(r64.iters)), "SzItMax": str(int(r512.iters)),
                   "SzSolveMin": fmt(r64.ts, 2), "SzSolveMax": fmt(r512.ts, 0),
                   "SzSolveMaxMin": fmt(r512.ts / 60, 1), "SzTiterMax": fmt(1e3 * r512.t, 0)})
    return z


# ---------------------------------------------------------------- multi-node strong scaling
def fig_strong(b, t_ser, cpn, macros, tables):
    s = b[b.exp == "strong"].sort_values("P").copy()
    s2 = b[b.exp == "strong2"].sort_values("P").copy()
    if s.empty:
        return
    s["S"] = t_ser / s.t_iter
    s["E"] = s.S / s.P
    t32 = float(s[s.P == cpn].t_iter.iloc[0])
    s["En"] = (t32 * cpn) / (s.t_iter * s.P)
    t32b = float(s2[s2.P == cpn].t_iter.iloc[0]) if not s2.empty else float("nan")
    s2["Sn"] = cpn * t32b / s2.t_iter
    s2["En"] = s2.Sn / s2.P
    Ps = s.P.values

    fig, ax = plt.subplots(1, 2, figsize=(6.6, 2.65))
    ax[0].plot(Ps, Ps, color=INK2, lw=1.0, ls="--", label="ideal")
    ax[0].plot(Ps, s.S, color=C[0], marker="o", label=r"$512^3$, vs. serial code")
    if not s2.empty:
        ax[0].plot(s2.P, s2.Sn, color=C[1], marker="s", label=r"$768^3$, vs. one node $\times$ 32")
    ax[0].axvline(cpn, color=C[5], lw=0.8, ls=":")
    ax[0].text(cpn * 1.06, 1.3, "one node", fontsize=6.5, color=C[5], rotation=90, va="bottom")
    log2_axis(ax[0], Ps)
    log2_axis(ax[0], [1, 2, 4, 8, 16, 32, 64, 128, 256], axis="y")
    ax[0].set_ylim(0.8, 330)
    ax[0].set_xlabel("MPI processes $P$")
    ax[0].set_ylabel("speedup")
    ax[0].set_title("(a) Speedup")
    ax[0].legend(loc="upper left", fontsize=6.5)
    ax[1].plot(Ps, 100 * s.E, color=C[0], marker="o", label=r"$512^3$, vs. serial code")
    sn = s[s.P >= cpn]
    ax[1].plot(sn.P, 100 * sn.En, color=C[0], marker="o", ls="--", mfc="white", label=r"$512^3$, vs. one node")
    if not s2.empty:
        ax[1].plot(s2.P, 100 * s2.En, color=C[1], marker="s", ls="--", mfc="white", label=r"$768^3$, vs. one node")
    ax[1].axhline(100, color=INK2, lw=1.0, ls="--")
    ax[1].axvline(cpn, color=C[5], lw=0.8, ls=":")
    log2_axis(ax[1], Ps)
    ax[1].set_ylim(0, 110)
    ax[1].set_xlabel("MPI processes $P$")
    ax[1].set_ylabel("parallel efficiency (%)")
    ax[1].set_title("(b) Efficiency")
    ax[1].legend(loc="lower left", fontsize=6.5)
    fig.tight_layout()
    suptitle(fig, "Strong scaling from one core to eight nodes", f"pure MPI, 3D blocks, overlapped halo, "
             f"classic CG; {cpn} processes per node, {int(s.iters.iloc[0])} iterations, best of "
             f"{int(s.nreps.max())} runs")
    save(fig, "strong")

    # where the time goes: share of each phase
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
    ax.axvline(list(Ps).index(cpn) + 0.5, color=INK2, lw=0.8, ls=":")
    ax.set_xlabel("MPI processes $P$ (grid $%d^3$); dotted line: one node" % int(s.N0.iloc[0]))
    ax.set_ylabel("share of iteration time (%)")
    ax.set_ylim(0, 100)
    ax.grid(axis="x", visible=False)
    ax.legend(ncol=4, loc="lower center", bbox_to_anchor=(0.5, 1.01), fontsize=7)
    fig.tight_layout()
    suptitle(fig, "Where a CG iteration spends its time")
    save(fig, "breakdown")

    s2i = s2.set_index("P")
    rows = []
    for _, r in s.iterrows():
        P = int(r.P)
        comm = (r.t_halo_it + r.t_reduce_it) / r.t_iter * 100
        nodes = max(1, P // cpn)
        en = f"{100 * r.En:.0f}" if P >= cpn else "--"
        if P in s2i.index:
            t2, e2 = f"{1e3 * s2i.loc[P, 't_iter']:.1f}", f"{100 * s2i.loc[P, 'En']:.0f}"
        else:
            t2, e2 = "--", "--"
        rows.append(f"{P} & {nodes} & {int(r.px)}$\\times${int(r.py)}$\\times${int(r.pz)} & {1e3 * r.t_iter:.2f} & "
                    f"{r.S:.1f} & {100 * r.E:.0f} & {en} & {comm:.0f} & {r.halo_bytes_iter_max / 2 ** 10:.0f} & "
                    f"{t2} & {e2} \\\\")
    tables["strong"] = rows
    last, n1 = s.iloc[-1], s[s.P == cpn].iloc[0]
    l2 = s2.iloc[-1] if not s2.empty else None
    pm = int(last.P)
    macros.update({
        "SerialTiter": fmt(1e3 * t_ser, 0), "PoneTiter": fmt(1e3 * float(s.iloc[0].t_iter), 0),
        "PoneVsSerial": fmt(100 * abs(1 - float(s.iloc[0].t_iter) / t_ser), 1),
        "PoneFaster": "faster" if float(s.iloc[0].t_iter) < t_ser else "slower",
        "Pmax": str(pm), "NodesMax": str(pm // cpn), "SpeedupPmax": fmt(last.S, 1), "EffPmax": fmt(100 * last.E, 0),
        "TiterPmax": fmt(1e3 * last.t_iter, 2), "TiterNode": fmt(1e3 * n1.t_iter, 1),
        "SpeedupNode": fmt(n1.S, 1), "EffNode": fmt(100 * n1.E, 0), "NodeEffPmax": fmt(100 * last.En, 0),
        "NodeSpeedPmax": fmt(t32 / float(last.t_iter), 2),
        "NodeEffSixtyFour": fmt(100 * float(s[s.P == 2 * cpn].En.iloc[0]), 0),
        "TiterSixtyFour": fmt(1e3 * float(s[s.P == 2 * cpn].t_iter.iloc[0]), 1),
        "SpeedupEight": fmt(float(s[s.P == 8].S.iloc[0]), 1),
        "SpeedupFour": fmt(float(s[s.P == 4].S.iloc[0]), 2), "EffFour": fmt(100 * float(s[s.P == 4].E.iloc[0]), 0),
        "CommFracPmax": fmt((last.t_halo_it + last.t_reduce_it) / last.t_iter * 100, 0),
        "CommFracNode": fmt((n1.t_halo_it + n1.t_reduce_it) / n1.t_iter * 100, 0),
        "RedFracPmax": fmt(100 * last.t_reduce_it / last.t_iter, 0), "RedMsPmax": fmt(1e3 * last.t_reduce_it, 2),
        "HaloFracPmax": fmt(100 * last.t_halo_it / last.t_iter, 0), "HaloMsPmax": fmt(1e3 * last.t_halo_it, 2),
        "RedMsNode": fmt(1e3 * n1.t_reduce_it, 2), "HaloKiBPmax": fmt(last.halo_bytes_iter_max / 2 ** 10, 0),
        "HaloKiBNode": fmt(n1.halo_bytes_iter_max / 2 ** 10, 0),
        "VecFracSerial": fmt(100 * float(s.iloc[0].t_vec_it / s.iloc[0].t_iter), 0),
        "NS": str(int(s.N0.iloc[0])), "NSdofs": fmt(int(s.N0.iloc[0]) ** 3 / 1e6, 0), "Iters": str(int(s.iters.iloc[0])),
        "Reps": str(int(b.nreps.max())), "MedSpread": fmt(100 * b[b.exp != "solve"].spread.median(), 1),
        "MaxSpread": fmt(100 * b[b.exp != "solve"].spread.max(), 0),
    })
    if l2 is not None:
        macros.update({"NStwo": str(int(s2.N0.iloc[0])), "NStwoDofs": fmt(int(s2.N0.iloc[0]) ** 3 / 1e6, 0),
                       "NodeEffTwoPmax": fmt(100 * l2.En, 0), "NodeSpeedTwoPmax": fmt(l2.Sn / cpn, 2),
                       "TiterTwoNode": fmt(1e3 * t32b, 1), "TiterTwoPmax": fmt(1e3 * l2.t_iter, 1)})
    return s


def fig_bandwidth(s, stream, cpn, macros):
    if s is None or stream is None:
        return
    st = stream.assign(gbs=stream.triad_GBs * 32.0 / 24.0)   # STREAM counts 24 B, moves 32 B
    bw = s.N0.astype(float) ** 3 * BYTES_CPU / s.t_iter / 1e9
    sock = float(st.gbs.iloc[-1])
    fig, ax = plt.subplots(figsize=(3.4, 2.6))
    nodes = sorted(set(int(p) for p in s.P if p >= cpn))
    ax.plot(nodes, [sock * p / cpn for p in nodes], color=INK2, ls=":", lw=1.0,
            label="STREAM of one 32-core socket $\\times$ nodes")
    ax.plot(st.threads, st.gbs, color=INK2, marker="s", ls="--", label="STREAM triad, one socket")
    ax.plot(s.P, bw, color=C[0], marker="o", label="CG solver, $512^3$")
    log2_axis(ax, s.P)
    ax.set_yscale("log")
    plain_log(ax, "y", 0, 4)
    ax.set_xlabel("cores ($%d$ per node)" % cpn)
    ax.set_ylabel("DRAM bandwidth (GB/s)")
    ax.set_title("Memory bandwidth: solver vs. STREAM")
    ax.legend(loc="upper left", fontsize=6.3)
    fig.tight_layout()
    save(fig, "bandwidth")
    i32 = list(s.P).index(cpn)
    macros.update({"StreamOne": fmt(float(st.gbs.iloc[0]), 1), "StreamAll": fmt(sock, 0),
                   "StreamThreads": str(int(st.threads.iloc[-1])), "StreamEight": fmt(float(st[st.threads == 8].gbs.iloc[0]), 0),
                   "CGbwOne": fmt(float(bw.iloc[0]), 1), "CGbwNode": fmt(float(bw.iloc[i32]), 0),
                   "CGbwFrac": fmt(100 * float(bw.iloc[i32]) / sock, 0), "CGbwAll": fmt(float(bw.iloc[-1]), 0),
                   "CGbwTB": fmt(float(bw.iloc[-1]) / 1e3, 2),
                   "CGbwPerNodeMax": fmt(float(bw.iloc[-1]) / (int(s.P.iloc[-1]) // cpn), 0)})


def fig_weak(b, cpn, macros):
    w = b[b.exp == "weak"].sort_values("P")
    if w.empty:
        return
    tpp = w.t_iter / w.nloc_max                 # time per point owned by the busiest process
    e1 = float(tpp.iloc[0]) / tpp
    t32 = float(tpp[w.P == cpn].iloc[0])
    wn = w[w.P >= cpn]
    en = t32 / (wn.t_iter / wn.nloc_max)
    fig, ax = plt.subplots(figsize=(3.4, 2.6))
    ax.plot(w.P, 100 * e1, color=C[0], marker="o", label="relative to one process")
    ax.plot(wn.P, 100 * en, color=C[1], marker="s", ls="--", mfc="white", label="relative to one node")
    ax.axhline(100, color=INK2, ls="--", lw=1.0)
    ax.axvline(cpn, color=C[5], lw=0.8, ls=":")
    log2_axis(ax, w.P)
    ax.set_ylim(0, 110)
    ax.set_xlabel("MPI processes $P$")
    ax.set_ylabel("weak-scaling efficiency (%)")
    ax.set_title(f"Weak scaling, $\\approx{round(w.nloc_avg.iloc[0] ** (1 / 3))}^3$ points per process")
    ax.legend(loc="lower left", fontsize=6.5)
    fig.tight_layout()
    save(fig, "weak")
    last = w.iloc[-1]
    macros.update({"WeakEffPmax": fmt(100 * float(e1.iloc[-1]), 0), "WeakPmax": str(int(last.P)),
                   "WeakNodeEff": fmt(100 * float(en.iloc[-1]), 0), "WeakEffNode": fmt(100 * float(e1[w.P == cpn].iloc[0]), 0),
                   "WeakLocal": str(round(w.nloc_avg.iloc[0] ** (1 / 3))),
                   "WeakTone": fmt(1e3 * float(w.t_iter.iloc[0]), 1),
                   "WeakTnode": fmt(1e3 * float(w[w.P == cpn].t_iter.iloc[0]), 1),
                   "WeakTmax": fmt(1e3 * float(last.t_iter), 1), "WeakNmax": str(int(last.N0)),
                   "WeakDofsMax": fmt(int(last.N0) ** 3 / 1e9, 2),
                   "WeakBwOne": fmt(float(w.nloc_max.iloc[0]) * BYTES_CPU / float(w.t_iter.iloc[0]) / 1e9, 1),
                   "WeakBwNode": fmt(float(w[w.P == cpn].nloc_max.iloc[0]) * BYTES_CPU /
                                     float(w[w.P == cpn].t_iter.iloc[0]) / 1e9, 1)})


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
    panels = [("(a) Data distribution and halo strategy", ["1D decomposition", "2D decomposition", "blocking halo"]),
              ("(b) CG formulation (3D blocks)", ["CG1 (1 Allreduce)", "pipelined CG", "pipelined CG + blocking halo"])]
    fig, axs = plt.subplots(1, 2, figsize=(6.6, 2.5), sharey=True)
    for ax, (title, labs) in zip(axs, panels):
        w = 0.8 / len(labs)
        for k, lab in enumerate(labs):
            d = v[v.label == lab].set_index("P").reindex(Ps)
            xs = [i + (k - (len(labs) - 1) / 2) * w for i in range(len(Ps))]
            ax.bar(xs, d.rel, width=w * 0.92, color=C[k], label=lab, edgecolor="white", linewidth=0.6)
        ax.axhline(1.0, color=INK, lw=1.0)
        ax.set_xticks(range(len(Ps)))
        ax.set_xticklabels([str(int(p)) for p in Ps])
        ax.set_xlabel("MPI processes $P$ (%s per node)" % 32)
        ax.set_title(title)
        ax.grid(axis="x", visible=False)
        ax.legend(loc="upper left", fontsize=6.5)
    axs[0].set_ylabel("time / baseline time")
    axs[0].set_ylim(0.8, max(1.25, v.rel.max() * 1.22))
    fig.tight_layout()
    suptitle(fig, "One design decision at a time, one to eight nodes", "time per iteration relative to the "
             "baseline (3D blocks, overlapped halo, classic CG) at the same $P$; below 1 is faster")
    save(fig, "variants")

    order = ["3D, overlap, CG (baseline)", "1D decomposition", "2D decomposition", "blocking halo",
             "CG1 (1 Allreduce)", "pipelined CG", "pipelined CG + blocking halo"]
    b0 = b[b.exp == "strong"].set_index("P")
    Pt = [p for p in (32, 64, 128, 256) if p in Ps] or Ps[:4]
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

    def rel(lab, P):
        d = v[(v.label == lab) & (v.P == P)]
        return float(d.rel.iloc[0]) if not d.empty else float("nan")

    def rng(lab):
        d = v[v.label == lab].rel
        return fmt(d.min(), 2) + "--" + fmt(d.max(), 2)

    pm, p0 = max(Ps), min(Ps)
    macros.update({"VarP": str(int(pm)), "VarPmin": str(int(p0)),
                   "RelOneD": fmt(rel("1D decomposition", pm)), "RelOneDmin": fmt(rel("1D decomposition", p0)),
                   "RelTwoD": fmt(rel("2D decomposition", pm)), "RelTwoDRange": rng("2D decomposition"),
                   "RelBlocking": fmt(rel("blocking halo", pm)), "RelBlockingRange": rng("blocking halo"),
                   "RelCGone": fmt(rel("CG1 (1 Allreduce)", pm)), "RelCGonemin": fmt(rel("CG1 (1 Allreduce)", p0)),
                   "RelPipe": fmt(rel("pipelined CG", pm)), "RelPipemin": fmt(rel("pipelined CG", p0)),
                   "RelPipeBlock": fmt(rel("pipelined CG + blocking halo", pm)),
                   "RelPipeTwo": fmt(float(b[b.exp == "strong2_pipe"].t_iter.iloc[-1] /
                                           b[b.exp == "strong2"].t_iter.iloc[-1]), 2)
                   if (b.exp == "strong2_pipe").any() else "--"})


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
        plain_log(a)
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
    fig, ax = plt.subplots(figsize=(3.3, 2.3))
    labels = [f"{int(p)}$\\times${int(t)}" for p, t in zip(h.P, h['T'])]
    ax.bar(range(len(h)), 1e3 * h.t_iter, color=C[0], width=0.6, label="best run")
    ax.plot(range(len(h)), 1e3 * h.t_med, color=C[1], marker="D", ls="none", label="median run")
    ax.set_xticks(range(len(h)))
    ax.set_xticklabels(labels)
    ax.set_xlabel("MPI processes $\\times$ OpenMP threads")
    ax.set_ylabel("ms per iteration")
    ax.set_ylim(0, 1e3 * h.t_med.max() * 1.25)
    ax.set_title(f"Processes × threads on {int(h.P.iloc[0] * h['T'].iloc[0])} cores")
    ax.grid(axis="x", visible=False)
    ax.legend(loc="upper left", fontsize=6.5, ncol=2)
    fig.tight_layout()
    save(fig, "hybrid")
    best, pure = h.loc[h.t_iter.idxmin()], h[h["T"] == 1].iloc[0]
    worst = h.loc[h.t_iter.idxmax()]
    tables['hybrid'] = [f"{int(r.P)} & {int(r['T'])} & {1e3 * r.t_iter:.2f} & {1e3 * r.t_med:.2f} & "
                        f"{1e3 * r.t_halo_it:.2f} & {1e3 * r.t_reduce_it:.2f} \\\\" for _, r in h.iterrows()]
    macros.update({"HybBest": f"{int(best.P)}$\\times${int(best['T'])}", "HybBestT": fmt(1e3 * best.t_iter, 2),
                   "HybPureT": fmt(1e3 * pure.t_iter, 2), "HybWorst": f"{int(worst.P)}$\\times${int(worst['T'])}",
                   "HybWorstT": fmt(1e3 * worst.t_iter, 2),
                   "HybRange": fmt(100 * (worst.t_iter / best.t_iter - 1), 0),
                   "HybCores": str(int(h.P.iloc[0] * h["T"].iloc[0])),
                   "HybGain": fmt(100 * (1 - best.t_iter / pure.t_iter), 0),
                   "HybRedBest": fmt(1e3 * best.t_reduce_it, 2), "HybRedPure": fmt(1e3 * pure.t_reduce_it, 2)})


# ---------------------------------------------------------------- GPU
def fig_gpu(gdir, s, t_ser, macros, tables):
    p = os.path.join(gdir, "gpu.csv")
    if not os.path.exists(p):
        return
    g = pd.read_csv(p)
    gs = best_of(g[g.exp == "gsize"], KEYS).sort_values(["alg", "N0"])
    cg, c1 = gs[gs.alg == "cg"].set_index("N0"), gs[gs.alg == "cg1"].set_index("N0")
    Ns = list(cg.index)
    bw = {a: d.N0.astype(float) ** 3 * BYTES_GPU / d.t_iter / 1e9 for a, d in [("cg", cg.reset_index()),
                                                                              ("cg1", c1.reset_index())]}
    fig, ax = plt.subplots(1, 2, figsize=(6.6, 2.6), gridspec_kw={"width_ratios": [1.1, 1]})
    ax[0].axhline(A40_PEAK, color=INK2, ls="--", lw=1.0, label="A40 peak (696 GB/s)")
    ax[0].plot(Ns, bw["cg"].values, color=C[0], marker="o", label="classic CG")
    ax[0].plot(Ns, bw["cg1"].values, color=C[1], marker="s", label="CG1")
    ax[0].set_xscale("log", base=2)
    ax[0].xaxis.set_major_locator(FixedLocator(Ns))
    ax[0].xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{int(v)}"))
    ax[0].xaxis.set_minor_locator(NullLocator())
    ax[0].set_ylim(0, 800)
    ax[0].set_xlabel("grid points per direction $N$ (one GPU)")
    ax[0].set_ylabel("DRAM bandwidth (GB/s, 88 B/point)")
    ax[0].set_title("(a) One A40 across problem sizes")
    ax[0].legend(loc="lower right", fontsize=6.5)
    n512 = 512
    plat = [("1 core", t_ser), ("1 socket\n(32 cores)", float(s[s.P == 32].t_iter.iloc[0])),
            ("1 A40", float(cg.loc[n512, "t_iter"])),
            ("8 nodes\n(256 cores)", float(s[s.P == s.P.max()].t_iter.iloc[0]))]
    cols = [C[5], C[0], C[2], C[3]]
    ax[1].bar(range(4), [1e3 * t for _, t in plat], color=cols, width=0.6)
    for i, (_, t) in enumerate(plat):
        ax[1].text(i, 1e3 * t * 1.12, f"{1e3 * t:.1f}", ha="center", va="bottom", fontsize=6.8)
    ax[1].set_yscale("log")
    plain_log(ax[1], "y", 0, 4)
    ax[1].set_ylim(5, 2000)
    ax[1].set_xticks(range(4))
    ax[1].set_xticklabels([n for n, _ in plat], fontsize=6.8)
    ax[1].set_ylabel("ms per iteration")
    ax[1].set_title("(b) $512^3$: GPU against CPU")
    ax[1].grid(axis="x", visible=False)
    fig.tight_layout()
    suptitle(fig, "MPI + OpenMP target offload on an NVIDIA A40", "classic CG, all vectors resident in GPU "
             "memory, 50 iterations, best of 3 runs")
    save(fig, "gpu")
    rows = []
    for N in Ns:
        r, q = cg.loc[N], c1.loc[N]
        sp = 100 * r.t_spmv_it / r.t_iter
        mem = 4 * N ** 3 * 8 / 2 ** 30
        rows.append(f"{N} & {N ** 3 / 1e6:.0f} & {mem:.1f} & {1e3 * r.t_iter:.2f} & {1e3 * q.t_iter:.2f} & "
                    f"{bw['cg'][Ns.index(N)]:.0f} & {sp:.0f} & {r.mlups / 1e3:.2f} \\\\")
    tables["gpu"] = rows
    r = cg.loc[n512]
    t_sock = float(s[s.P == 32].t_iter.iloc[0])
    macros.update({"GpuTfiveTwelve": fmt(1e3 * r.t_iter, 1), "GpuBwFiveTwelve": fmt(bw["cg"][Ns.index(n512)], 0),
                   "GpuBwFrac": fmt(100 * bw["cg"][Ns.index(n512)] / A40_PEAK, 0),
                   "GpuBwMin": fmt(bw["cg"].min(), 0), "GpuBwMax": fmt(bw["cg"].max(), 0),
                   "GpuNmin": str(Ns[0]), "GpuNmax": str(Ns[-1]), "GpuDofsMax": fmt(Ns[-1] ** 3 / 1e6, 0),
                   "GpuVsSocket": fmt(t_sock / r.t_iter, 1), "GpuVsCore": fmt(t_ser / r.t_iter, 0),
                   "GpuVsEight": fmt(r.t_iter / float(s[s.P == s.P.max()].t_iter.iloc[0]), 1),
                   "GpuSpmvFrac": fmt(100 * r.t_spmv_it / r.t_iter, 0),
                   "GpuCGoneRel": fmt(100 * (c1.loc[n512, "t_iter"] / r.t_iter - 1), 0),
                   "GpuSpmvBw": fmt(n512 ** 3 * 24 / r.t_spmv_it / 1e9, 0),
                   "GpuVecBw": fmt(n512 ** 3 * 64 / r.t_vec_it / 1e9, 0),
                   "GpuMLUPS": fmt(r.mlups / 1e3, 2)})
    sol = g[g.exp == "gsolve"]
    return sol


def gpu_validation(vdir, macros):
    p = os.path.join(vdir, "validation.txt")
    if not os.path.exists(p):
        return
    txt = open(p).read()
    vals = [float(x) for x in re.findall(r"PASS\s+max\S+\s*=\s*([\d.e+-]+)", txt)]
    ps = sorted(set(int(x) for x in re.findall(r"\(P=(\d+)", txt)))
    m = re.search(r"validation:\s*(\d+) failure", txt)
    macros.update({"GpuValRuns": str(len(vals)), "GpuValWorst": f"${sci(max(vals), 1)}$" if vals else "--",
                   "GpuValP": ", ".join(str(p) for p in ps), "GpuValFail": m.group(1) if m else "--"})


# ---------------------------------------------------------------- validation
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
            ax[1].semilogy(hh.iter, hh.relres_recursive, color=C[k], ls=":", lw=lw * 0.8, label=f"{name}: recursive")
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


def node_validation(path, macros):
    if not os.path.exists(path):
        return
    txt = open(path).read()
    w = {a: float(x) for a, x in re.findall(r"^\s*(\w+)\s+\d+ runs: worst \S+ = ([\d.e+-]+)", txt, re.M)}
    n = sum(int(x) for x in re.findall(r"^\s*\w+\s+(\d+) runs:", txt, re.M))
    e = re.search(r"edge cases:\s*(\d+) runs, worst diff ([\d.e+-]+)", txt)
    macros.update({"NodeValRuns": str(n), "NodeValEdge": e.group(1) if e else "--",
                   "NodeValCG": f"${sci(max(w.get('cg', 0), w.get('cg1', 0)), 1)}$",
                   "NodeValPipe": f"${sci(w.get('pipecg', 0), 1)}$",
                   "NodeValResult": "passes" if "RESULT: PASS" in txt else "fails"})


# ---------------------------------------------------------------- time to solution
def tab_solve(b, sizes, gsol, cpn, macros, tables):
    name = {"cg": "CG", "cg1": "CG1", "pipecg": "pipelined CG"}
    rows = []
    t_ser = None
    if sizes is not None and (sizes.N0 == 512).any():
        r = sizes[sizes.N0 == 512].iloc[0]
        t_ser = float(r.ts)
        rows.append(f"1 core & CG & {int(r.iters)} & {r.ts:.0f} & {1e3 * r.ts / r.iters:.0f} & "
                    f"${sci(r.tr, 2)}$ & ${sci(r.err, 2)}$ & 1 \\\\")
    s = b[b.exp == "solve"]
    nodes = int(s.P.max()) // cpn if not s.empty else 0
    for _, r in s.iterrows():
        sp = f"{t_ser / r.t_solve:.0f}" if t_ser else "--"
        rows.append(f"{int(r.P)} cores ({nodes} nodes) & {name[r.alg]} & {int(r.iters)} & {r.t_solve:.1f} & "
                    f"{1e3 * r.t_solve / r.iters:.2f} & ${sci(r.true_relres, 2)}$ & ${sci(r.err_max, 2)}$ & {sp} \\\\")
    if gsol is not None:
        for _, r in gsol.iterrows():
            sp = f"{t_ser / r.t_solve:.0f}" if t_ser else "--"
            rows.append(f"1 A40 GPU & {name[r.alg]} & {int(r.iters)} & {r.t_solve:.1f} & "
                        f"{1e3 * r.t_solve / r.iters:.2f} & ${sci(r.true_relres, 2)}$ & ${sci(r.err_max, 2)}$ & {sp} \\\\")
    tables["solve"] = rows
    cg, pp, c1 = s[s.alg == "cg"], s[s.alg == "pipecg"], s[s.alg == "cg1"]
    if not cg.empty:
        macros.update({"SolveIters": str(int(cg.iters.iloc[0])), "SolveCG": fmt(float(cg.t_solve.iloc[0]), 1),
                       "SolveCGTrue": f"${sci(float(cg.true_relres.iloc[0]), 1)}$"})
        if t_ser:
            macros["SolveSpeedup"] = fmt(t_ser / float(cg.t_solve.iloc[0]), 0)
    if not c1.empty:
        macros["SolveCGone"] = fmt(float(c1.t_solve.iloc[0]), 1)
    if not pp.empty:
        macros.update({"SolvePipe": fmt(float(pp.t_solve.iloc[0]), 1),
                       "SolvePipeTrue": f"${sci(float(pp.true_relres.iloc[0]), 1)}$",
                       "SolvePipeRatio": fmt(float(pp.true_relres.iloc[0]) / 1e-8, 0)})
    if gsol is not None and not gsol.empty:
        gc = gsol[gsol.alg == "cg"].iloc[0]
        macros.update({"SolveGpu": fmt(float(gc.t_solve), 1), "SolveGpuIters": str(int(gc.iters))})
        if t_ser:
            macros["SolveGpuSpeedup"] = fmt(t_ser / float(gc.t_solve), 0)


# every macro the report uses defaults to "--" if its experiment is missing; the list is read
# from report.tex (capitalised control sequences minus LaTeX's own)
FIGURES = ["sizes", "strong", "breakdown", "variants", "commvol", "hybrid", "weak", "bandwidth", "validation", "gpu"]
LATEX_OWN = {"LARGE", "Cref", "Omega", "Comment", "State", "For", "EndFor", "While", "EndWhile", "If", "EndIf"}


def report_macros():
    path = os.path.join(ROOT, "report", "report.tex")
    if not os.path.exists(path):
        return []
    return sorted(set(re.findall(r"\\([A-Z][A-Za-z]+)", open(path).read())) - LATEX_OWN)


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--main", default=os.path.join(RES, "vanda256"))
    ap.add_argument("--node", default=os.path.join(RES, "vanda_node"))
    ap.add_argument("--sizes", default=os.path.join(RES, "vanda_sizes"))
    ap.add_argument("--gpu", default=os.path.join(RES, "vanda_gpu"))
    ap.add_argument("--gpuval", default=os.path.join(RES, "vanda_gpu_val"))
    ap.add_argument("--validation", default=os.path.join(RES, "validation"))
    a = ap.parse_args()
    os.makedirs(GEN, exist_ok=True)
    for f in FIGURES:
        p = os.path.join(FIG, f + ".pdf")
        if os.path.exists(p):
            os.remove(p)
    macros, tables = {}, {}
    mi, ni, si, gi = env_info(a.main), env_info(a.node), env_info(a.sizes), env_info(a.gpu)
    cpn = mi.get("cpn", 32)
    macros.update({"MachineCPU": tex_escape(mi.get("cpu", "?")), "MachineNodes": str(mi.get("nodes", "?")),
                   "MachineCPN": str(cpn), "MachineCores": str(mi.get("nodes", 0) * cpn),
                   "MachineCPS": str(mi.get("cps", "?")), "MachineMPI": "Open MPI " + mi.get("mpi", "?"),
                   "MachineGCC": mi.get("gcc", "?"), "NodeCPU": tex_escape(ni.get("cpu", "?")),
                   "SizesCPU": tex_escape(si.get("cpu", "?"))})
    if "gpu" in gi:
        macros.update({"GpuName": gi["gpu"][0], "GpuMem": fmt(int(gi["gpu"][1]) / 1024, 0), "GpuDriver": gi["gpu"][2],
                       "GpuNVC": gi.get("nvc", "?"), "GpuModule": tex_escape(gi.get("module", "?"))})
    b = best_of(pd.read_csv(os.path.join(a.main, "bench.csv")), KEYS)
    b.to_csv(os.path.join(a.main, "bench_best.csv"), index=False)
    nb = os.path.join(a.node, "bench.csv")
    t_ser = float(best_of(pd.read_csv(nb), KEYS).query("exp == 'serial'").t_iter.iloc[0]) if os.path.exists(nb) \
        else float(b[(b.exp == "strong") & (b.P == 1)].t_iter.iloc[0])
    stp = os.path.join(a.node, "stream.csv")
    stream = pd.read_csv(stp) if os.path.exists(stp) else None
    sizes = fig_sizes(a.sizes, macros, tables)
    s = fig_strong(b, t_ser, cpn, macros, tables)
    fig_bandwidth(s, stream, cpn, macros)
    fig_weak(b, cpn, macros)
    fig_variants(b, macros, tables)
    fig_hybrid(b, macros, tables)
    gsol = fig_gpu(a.gpu, s, t_ser, macros, tables)
    gpu_validation(a.gpuval, macros)
    tab_solve(b, sizes, gsol, cpn, macros, tables)
    fig_commvol(os.path.join(RES, "commvol.csv"), macros)
    fig_validation(a.validation, macros, tables)
    node_validation(os.path.join(a.node, "validation", "summary.txt"), macros)

    defaults = {k for k in report_macros() if k not in macros}
    for f in os.listdir(GEN):
        if f.endswith(".tex"):
            os.remove(os.path.join(GEN, f))
    with open(os.path.join(GEN, "macros.tex"), "w") as f:
        f.write("% generated by scripts/make_plots.py -- do not edit\n")
        for k, v in sorted(macros.items()):
            f.write(f"\\newcommand{{\\{k}}}{{{v}}}\n")
        for k in sorted(defaults):
            f.write(f"\\providecommand{{\\{k}}}{{--}}\n")
    for t in ["validation", "convergence", "strong", "variants", "hybrid", "solve", "sizes", "gpu"]:
        with open(os.path.join(GEN, f"tab_{t}.tex"), "w") as f:
            f.write("\n".join(tables.get(t, [])) + "\n")
    Ps = tables.get("variants_cols", [])
    with open(os.path.join(GEN, "tab_variants_head.tex"), "w") as f:
        f.write("\\begin{tabular}{l" + "r" * max(len(Ps), 1) + "}\n\\toprule\n")
        f.write("variant & " + (" & ".join(f"$P={p}$" for p in Ps) if Ps else "not measured") + " \\\\\n\\midrule\n")
    for f in FIGURES:
        if not os.path.exists(os.path.join(FIG, f + ".pdf")):
            fig, ax = plt.subplots(figsize=(3.4, 1.2))
            ax.axis("off")
            ax.text(0.5, 0.5, f"{f}: no data", ha="center", va="center", color=INK2)
            save(fig, f)
    if defaults:
        print("WARNING: macros without data:", " ".join(sorted(defaults)))
    print("figures ->", FIG, "| tables/macros ->", GEN)
    for k in sorted(macros):
        print(f"  {k} = {macros[k]}")


if __name__ == "__main__":
    main()
