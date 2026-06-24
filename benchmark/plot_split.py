"""
Split-plot benchmark results: communication and compute are plotted on SEPARATE
figures (comm dominates by ~1000x on a WiFi cluster, so a shared axis hides the
compute-only scaling). Reads the same CSVs as run_campaign.

Produces, in --res dir:
  exp1_wall_vs_n.png       exp1_compute_vs_n.png
  exp2_comm_per_rank.png   exp2_compute_per_rank.png
  exp3_wall_vs_procs.png   exp3_compute_vs_procs.png   exp3_speedup.png
"""
import argparse, csv, re
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

def read_csv(p):
    with open(p) as f: return list(csv.DictReader(f))

def save(fig, p, title):
    fig.suptitle(title, fontsize=12, fontweight="bold")
    fig.tight_layout(); fig.savefig(p, dpi=150, bbox_inches="tight")
    print("saved", p); plt.close(fig)

def imb(a):
    return (a.max()-a.min())/a.max()*100 if len(a) and a.max() > 0 else 0.0

def exp1(csvp, outdir, plabel):
    rows = read_csv(csvp)
    if not rows: print("exp1 no data"); return
    ns = np.array([int(r["n_points"]) for r in rows])
    w  = np.array([float(r["wall_s"]) for r in rows])
    c  = np.array([float(r["compute_s"]) for r in rows])
    o = np.argsort(ns); ns, w, c = ns[o], w[o], c[o]

    fig, ax = plt.subplots(figsize=(8,5))
    ax.plot(ns, w, "o-", color="steelblue")
    ax.axhline(120, ls=":", color="gray", alpha=.7, label="2-min")
    ax.axhline(180, ls=":", color="gray", alpha=.4, label="3-min")
    ax.set_xlabel("Input size N (points)"); ax.set_ylabel("Wall time incl. comm (s)")
    ax.grid(alpha=.3); ax.legend()
    save(fig, Path(outdir)/"exp1_wall_vs_n.png", f"Exp1a: Wall time WITH comm vs N  ({plabel})")

    fig, ax = plt.subplots(figsize=(8,5))
    ax.plot(ns, c, "s-", color="darkorange")
    ax.set_xlabel("Input size N (points)"); ax.set_ylabel("Compute time only (s)")
    ax.grid(alpha=.3)
    save(fig, Path(outdir)/"exp1_compute_vs_n.png", f"Exp1b: Compute time WITHOUT comm vs N  ({plabel})")

def exp2(csvp, outdir):
    rows = read_csv(csvp)
    if not rows: print("exp2 no data"); return
    tf = rows[0]["timing_file"]; N = rows[0]["n_points"]
    txt = Path(tf).read_text()
    ranks, tc, tm = [], [], []; intab = False
    for line in txt.splitlines():
        if "rank" in line and "compute" in line: intab = True; continue
        if intab:
            m = re.match(r'\s*(\d+)\s+([\d.]+)\s+([\d.]+)', line)
            if m:
                ranks.append(int(m.group(1))); tc.append(float(m.group(2))); tm.append(float(m.group(3)))
    if not ranks: print("exp2 no per-rank data"); return
    ranks = np.array(ranks); tc = np.array(tc); tm = np.array(tm)
    o = np.argsort(ranks); ranks, tc, tm = ranks[o], tc[o], tm[o]
    x = np.arange(len(ranks)); lbl = [f"r{r}" for r in ranks]

    fig, ax = plt.subplots(figsize=(max(8,len(ranks)),5))
    ax.bar(x, tm, color="tomato"); ax.set_xticks(x); ax.set_xticklabels(lbl)
    ax.set_ylabel("Communication time (s)"); ax.grid(axis="y", alpha=.3)
    save(fig, Path(outdir)/"exp2_comm_per_rank.png",
         f"Exp2a: Comm time per rank  (N={N}, imbalance={imb(tm):.1f}%)")

    fig, ax = plt.subplots(figsize=(max(8,len(ranks)),5))
    ax.bar(x, tc, color="steelblue"); ax.set_xticks(x); ax.set_xticklabels(lbl)
    ax.set_ylabel("Compute time (s)"); ax.grid(axis="y", alpha=.3)
    save(fig, Path(outdir)/"exp2_compute_per_rank.png",
         f"Exp2b: Compute time per rank  (N={N}, imbalance={imb(tc):.1f}%)")

def exp3(csvp, outdir):
    rows = read_csv(csvp)
    if not rows: print("exp3 no data"); return
    nps = np.array([int(r["n_procs"]) for r in rows])
    w = np.array([float(r["wall_s"]) for r in rows]); c = np.array([float(r["compute_s"]) for r in rows])
    o = np.argsort(nps); nps, w, c = nps[o], w[o], c[o]; N = rows[0]["n_points"]

    fig, ax = plt.subplots(figsize=(8,5))
    ax.plot(nps, w, "o-", color="steelblue")
    ax.set_xlabel("Number of MPI processes"); ax.set_ylabel("Wall time incl. comm (s)")
    ax.set_xticks(nps); ax.grid(alpha=.3)
    save(fig, Path(outdir)/"exp3_wall_vs_procs.png", f"Exp3a: Wall time WITH comm vs procs  (N={N})")

    fig, ax = plt.subplots(figsize=(8,5))
    ax.plot(nps, c, "s-", color="darkorange")
    ax.set_xlabel("Number of MPI processes"); ax.set_ylabel("Compute time only (s)")
    ax.set_xticks(nps); ax.grid(alpha=.3)
    save(fig, Path(outdir)/"exp3_compute_vs_procs.png", f"Exp3b: Compute time WITHOUT comm vs procs  (N={N})")

    t1w = w[nps==1][0] if any(nps==1) else w[0]
    t1c = c[nps==1][0] if any(nps==1) else c[0]
    fig, ax = plt.subplots(figsize=(8,5))
    ax.plot(nps, t1w/w, "o-", color="steelblue", label="Speedup (wall, with comm)")
    ax.plot(nps, t1c/c, "s-", color="darkorange", label="Speedup (compute only)")
    ax.plot(nps, nps, "k:", alpha=.5, label="Ideal linear")
    ax.set_xlabel("Number of MPI processes"); ax.set_ylabel("Speedup")
    ax.set_xticks(nps); ax.grid(alpha=.3); ax.legend()
    save(fig, Path(outdir)/"exp3_speedup.png", f"Exp3c: Speedup vs procs  (N={N})")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--res", required=True)
    p.add_argument("--procs", default="np=8")
    a = p.parse_args()
    exp1(f"{a.res}/exp1_runtime_vs_n.csv", a.res, a.procs)
    exp2(f"{a.res}/exp2_granularity.csv", a.res)
    exp3(f"{a.res}/exp3_speedup.csv", a.res)
    print("all split plots done")
