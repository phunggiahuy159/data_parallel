"""
Plot benchmark results from the C++ PDSDBSCAN-D runs.
Reads CSV files produced by run_benchmark.sh.

Produces:
  exp1_runtime_vs_n.png  — wall time + compute time vs N (12 procs)
  exp2_granularity.png   — stacked bar per rank (compute vs comm)
  exp3_speedup.png       — runtime + speedup vs number of processes
"""

import argparse
import csv
import re
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path


def read_csv(path):
    with open(path) as f:
        return list(csv.DictReader(f))


def savefig(fig, path, title):
    fig.suptitle(title, fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"Saved: {path}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Experiment 1 — Runtime vs N
# ---------------------------------------------------------------------------
def plot_exp1(csv_path, outdir):
    rows = read_csv(csv_path)
    if not rows:
        print("exp1: no data"); return

    ns       = np.array([int(r["n_points"])  for r in rows])
    t_wall   = np.array([float(r["wall_s"])   for r in rows])
    t_comp   = np.array([float(r["compute_s"]) for r in rows])

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(ns, t_wall,  "o-", color="steelblue",  label="Wall time (incl. comm)")
    ax.plot(ns, t_comp,  "s--", color="darkorange", label="Compute time only")
    ax.axhline(120, color="gray", linestyle=":", alpha=0.7, label="2-min target")
    ax.axhline(180, color="gray", linestyle=":", alpha=0.4, label="3-min target")
    ax.set_xlabel("Input size N (points)")
    ax.set_ylabel("Time (seconds)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    savefig(fig, Path(outdir) / "exp1_runtime_vs_n.png",
            f"Experiment 1: Runtime vs N  ({rows[0]['n_procs']} MPI processes)")


# ---------------------------------------------------------------------------
# Experiment 2 — Granularity (per-rank timing)
# ---------------------------------------------------------------------------
def plot_exp2(csv_path, outdir):
    rows = read_csv(csv_path)
    if not rows:
        print("exp2: no data"); return

    timing_file = rows[0]["timing_file"]
    try:
        txt = Path(timing_file).read_text()
    except FileNotFoundError:
        print(f"exp2: timing file not found: {timing_file}"); return

    # Parse per-rank table from output:
    #  rank  compute(s)  comm(s)
    #     0     12.3450      2.1
    ranks, t_comp, t_comm = [], [], []
    in_table = False
    for line in txt.splitlines():
        if "rank" in line and "compute" in line:
            in_table = True; continue
        if in_table:
            m = re.match(r'\s*(\d+)\s+([\d.]+)\s+([\d.]+)', line)
            if m:
                ranks.append(int(m.group(1)))
                t_comp.append(float(m.group(2)))
                t_comm.append(float(m.group(3)))

    if not ranks:
        print("exp2: no per-rank data found in timing file"); return

    order = np.argsort(ranks)
    ranks  = np.array(ranks)[order]
    t_comp = np.array(t_comp)[order]
    t_comm = np.array(t_comm)[order]

    fig, ax = plt.subplots(figsize=(max(8, len(ranks)), 5))
    x = np.arange(len(ranks))
    ax.bar(x, t_comp, label="Compute time", color="steelblue")
    ax.bar(x, t_comm, bottom=t_comp, label="Comm time", color="tomato")
    ax.set_xticks(x)
    ax.set_xticklabels([f"rank {r}" for r in ranks], rotation=45, ha="right")
    ax.set_ylabel("Time (seconds)")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    totals = t_comp + t_comm
    imbal  = (totals.max() - totals.min()) / totals.max() * 100
    ax.set_title(f"Granularity Check  (N={rows[0]['n_points']}, "
                 f"imbalance={imbal:.1f}%)")
    savefig(fig, Path(outdir) / "exp2_granularity.png",
            "Experiment 2: Per-rank Timing (Granularity Check)")


# ---------------------------------------------------------------------------
# Experiment 3 — Speedup
# ---------------------------------------------------------------------------
def plot_exp3(csv_path, outdir):
    rows = read_csv(csv_path)
    if not rows:
        print("exp3: no data"); return

    nps    = np.array([int(r["n_procs"])    for r in rows])
    t_wall = np.array([float(r["wall_s"])   for r in rows])
    t_comp = np.array([float(r["compute_s"]) for r in rows])

    order  = np.argsort(nps)
    nps, t_wall, t_comp = nps[order], t_wall[order], t_comp[order]

    t1_wall = t_wall[nps == 1][0] if any(nps == 1) else t_wall[0]
    t1_comp = t_comp[nps == 1][0] if any(nps == 1) else t_comp[0]

    sp_wall = t1_wall / t_wall
    sp_comp = t1_comp / t_comp

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    ax1.plot(nps, t_wall,  "o-",  color="steelblue",  label="Wall time (incl. comm)")
    ax1.plot(nps, t_comp,  "s--", color="darkorange",  label="Compute only")
    ax1.set_xlabel("Number of MPI Processes")
    ax1.set_ylabel("Time (seconds)")
    ax1.set_title("Runtime vs Number of Processes")
    ax1.set_xticks(nps)
    ax1.legend(); ax1.grid(True, alpha=0.3)

    ax2.plot(nps, sp_wall, "o-",  color="steelblue",  label="Speedup (wall)")
    ax2.plot(nps, sp_comp, "s--", color="darkorange",  label="Speedup (compute)")
    ax2.plot(nps, nps,     "k:",  alpha=0.5,           label="Ideal linear")
    ax2.set_xlabel("Number of MPI Processes")
    ax2.set_ylabel("Speedup")
    ax2.set_title("Speedup vs Number of Processes")
    ax2.set_xticks(nps)
    ax2.legend(); ax2.grid(True, alpha=0.3)

    savefig(fig, Path(outdir) / "exp3_speedup.png",
            f"Experiment 3: Speedup  (N={rows[0]['n_points']})")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--exp1",   required=True)
    p.add_argument("--exp2",   required=True)
    p.add_argument("--exp3",   required=True)
    p.add_argument("--outdir", default="benchmark/results")
    args = p.parse_args()

    Path(args.outdir).mkdir(parents=True, exist_ok=True)
    plot_exp1(args.exp1, args.outdir)
    plot_exp2(args.exp2, args.outdir)
    plot_exp3(args.exp3, args.outdir)
    print("All plots generated.")


if __name__ == "__main__":
    main()
