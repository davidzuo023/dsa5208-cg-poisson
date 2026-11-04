#!/usr/bin/env python3
"""PASS/FAIL checks on the CSV files written by scripts/validate.sh (standard library only).

Criteria
  * every parallel run must reproduce the serial solution point-wise:
        max|x_par - x_ser| / max|x_ser| <= 1e-10  (cg, cg1)   <= 1e-8 (pipecg)
  * iteration count within +-1 (cg, cg1) / +-2 (pipecg) of the serial run
  * true residual ||b - A x|| / ||b|| <= 1.5 * tol
  * grid study: observed order of the discretisation error in [1.9, 2.1] and parallel
    errors equal to the serial ones (relative difference < 1e-6)
"""
import csv
import math
import os
import sys

out = sys.argv[1] if len(sys.argv) > 1 else "results/validation"
fails = []
lines = []


def read(name):
    path = os.path.join(out, name)
    if not os.path.exists(path):
        return []
    with open(path) as f:
        return list(csv.DictReader(f))


def diff_tol(alg):
    return 1e-8 if alg == "pipecg" else 1e-10


def it_tol(alg):
    return 2 if alg == "pipecg" else 1


ref = read("serial_ref.csv")
variants = read("variants.csv")
if ref and variants:
    r = ref[0]
    it_ref = int(r["iters"])
    worst = {}
    for v in variants:
        alg = v["alg"]
        ok = (float(v["maxdiff_vs_serial"]) <= diff_tol(alg)
              and abs(int(v["iters"]) - it_ref) <= it_tol(alg)
              and float(v["true_relres"]) <= 1.5e-8
              and int(v["converged"]) == 1)
        if not ok:
            fails.append("variant %s/%s/%s P=%s T=%s: diff=%s iters=%s true_relres=%s" % (
                alg, v["comm"], v["decomp"], v["P"], v["T"], v["maxdiff_vs_serial"], v["iters"], v["true_relres"]))
        w = worst.setdefault(alg, [0.0, 0, 0])
        w[0] = max(w[0], float(v["maxdiff_vs_serial"]))
        w[1] = max(w[1], abs(int(v["iters"]) - it_ref))
        w[2] += 1
    lines.append("serial reference N=%s: iters=%d true_relres=%s err_max=%s" % (
        r["N0"], it_ref, r["true_relres"], r["err_max"]))
    for alg, (d, di, n) in sorted(worst.items()):
        lines.append("  %-7s %3d runs: worst max|x_par-x_ser|/max|x_ser| = %.2e, worst |iters-iters_ser| = %d"
                     % (alg, n, d, di))

edge = read("edge.csv")
for v in edge:
    alg = v["alg"]
    if float(v["maxdiff_vs_serial"]) > diff_tol(alg) or float(v["true_relres"]) > 1.5e-8:
        fails.append("edge N=%s P=%s %s/%s/%s diff=%s" % (v["N0"], v["P"], alg, v["comm"], v["decomp"],
                                                          v["maxdiff_vs_serial"]))
if edge:
    lines.append("edge cases: %d runs, worst diff %.2e" % (
        len(edge), max(float(v["maxdiff_vs_serial"]) for v in edge)))

conv = read("convergence.csv")
if conv:
    groups = {}
    for v in conv:
        key = (v["code"], v["alg"], v["P"])
        groups.setdefault(key, []).append((int(v["N0"]), float(v["err_l2"]), float(v["err_max"])))
    ser = {n: e for n, e, _ in groups.get(("serial", "cg", "1"), [])}
    for key, rows in sorted(groups.items()):
        rows.sort()
        hs = [1.0 / (n + 1) for n, _, _ in rows]
        es = [e for _, e, _ in rows]
        # least-squares slope of log(err) vs log(h) over the finest half of the grids
        k = max(2, len(rows) // 2)
        xs, ys = [math.log(h) for h in hs[-k:]], [math.log(e) for e in es[-k:]]
        mx, my = sum(xs) / k, sum(ys) / k
        slope = sum((a - mx) * (b - my) for a, b in zip(xs, ys)) / sum((a - mx) ** 2 for a in xs)
        lines.append("grid study %-6s %-6s P=%-2s: observed order %.3f" % (key[0], key[1], key[2], slope))
        if not 1.9 <= slope <= 2.1:
            fails.append("order of accuracy %.3f for %s" % (slope, key))
        for n, e, _ in rows:
            if n in ser and abs(e - ser[n]) / ser[n] > 1e-6:
                fails.append("parallel error differs from serial at N=%d for %s" % (n, key))

summary = "\n".join(lines + ["", "RESULT: " + ("PASS" if not fails else "FAIL (%d)" % len(fails))] + fails)
print(summary)
with open(os.path.join(out, "summary.txt"), "w") as f:
    f.write(summary + "\n")
sys.exit(1 if fails else 0)
