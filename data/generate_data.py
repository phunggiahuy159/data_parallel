"""
Generate synthetic clustering datasets and save in the binary format
expected by the C++ PDSDBSCAN-D binary.

Binary format (.bin):
  Header : int32 N, int32 d
  Points : N * d  float64 (row-major, C order)

Usage:
  python3 data/generate_data.py --n 40000 --d 2 --clusters 10 \
                                 --seed 42 --output data/dataset.bin
"""

import argparse
import struct
import numpy as np
from pathlib import Path


def make_dataset(n: int, n_clusters: int = 10, noise_frac: float = 0.05,
                 dim: int = 2, seed: int = 42):
    rng = np.random.default_rng(seed)
    centres = rng.uniform(1, 9, size=(n_clusters, dim))
    std = 0.35
    n_noise   = int(n * noise_frac)
    n_cluster = n - n_noise

    pts_list, lbl_list = [], []
    ppc   = n_cluster // n_clusters
    extra = n_cluster - ppc * n_clusters
    for c in range(n_clusters):
        cnt = ppc + (1 if c < extra else 0)
        pts_list.append(rng.normal(centres[c], std, size=(cnt, dim)))
        lbl_list.append(np.full(cnt, c, dtype=np.int64))

    if n_noise > 0:
        pts_list.append(rng.uniform(0, 10, size=(n_noise, dim)))
        lbl_list.append(np.full(n_noise, -1, dtype=np.int64))

    points = np.vstack(pts_list).astype(np.float64)
    labels = np.concatenate(lbl_list)
    perm   = rng.permutation(n)
    return points[perm], labels[perm]


def save_bin(path: str, points: np.ndarray):
    """Save points array to the C++ binary format."""
    N, d = points.shape
    with open(path, 'wb') as f:
        f.write(struct.pack('ii', N, d))
        f.write(points.astype(np.float64).tobytes())


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n",        type=int,   default=10000)
    p.add_argument("--d",        type=int,   default=2)
    p.add_argument("--clusters", type=int,   default=10)
    p.add_argument("--noise",    type=float, default=0.05)
    p.add_argument("--seed",     type=int,   default=42)
    p.add_argument("--output",   type=str,   default="data/dataset.bin")
    args = p.parse_args()

    pts, lbls = make_dataset(args.n, args.clusters, args.noise, args.d, args.seed)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    save_bin(str(out), pts)
    # Also save labels for reference
    np.save(str(out).replace('.bin', '_true_labels.npy'), lbls)
    print(f"Saved {args.n} points (d={args.d}, {args.clusters} clusters, "
          f"{args.noise*100:.0f}% noise) → {out}")


if __name__ == "__main__":
    main()
