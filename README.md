# PDSDBSCAN-D  —  Parallel DBSCAN with Union-Find (C++ / MPI)

Implements the **PDSDBSCAN-D** algorithm from:
> Patwary, Palsetia, Agrawal, Liao, Manne, Choudhary.
> *"A New Scalable Parallel DBSCAN Algorithm Using the Disjoint-Set Data Structure."*
> SC'12, ACM/IEEE, Salt Lake City, 2012.

Cluster target: **3 physical machines × 4 cores = 12 MPI processes**.

> **🚀 New teammate setting up your node?** Clone this repo, open **Claude Code**
> in the folder, and paste the one-shot prompt in [`prompt.md`](prompt.md) — it
> builds your Ubuntu VM, installs MPI, builds this project, and verifies it
> end-to-end. Cluster bring-up details are in [`SETUP.md`](SETUP.md), and
> [`TEAM.md`](TEAM.md) explains how the team stays in sync (no version drift).

---

## Repository Layout

```
para/
├── src/
│   ├── union_find.hpp       — Dense UnionFind + sparse GlobalUF (Rem's algorithm)
│   ├── kd_tree.hpp          — KD-tree for O(N log N) range queries
│   ├── dbscan_seq.hpp/.cpp  — Sequential DSDBSCAN (Algorithm 2, reference)
│   ├── pdsdbscan.hpp/.cpp   — Parallel PDSDBSCAN-D  (Algorithm 5)
│   └── main.cpp             — MPI entry point + CLI
├── tools/
│   └── seq_main.cpp         — Standalone sequential binary (no MPI)
├── data/
│   └── generate_data.py     — Dataset generator → .bin format
├── benchmark/
│   ├── run_benchmark.sh     — Automated 3-experiment suite
│   └── plot_results.py      — Matplotlib charts (runtime, granularity, speedup)
├── cluster_setup/
│   ├── hosts.conf           — MPI hostfile template
│   ├── install_deps.sh      — Per-node dependency installer
│   └── setup_cluster.sh     — Full cluster bootstrap (from master node)
├── report/
│   └── report.md            — Comprehensive 18-page technical report
├── Makefile
└── requirements.txt         — Python deps (data gen + plotting only)
```

---

## 1. Cluster Setup

### 1.1 Prerequisites

- 3+ Ubuntu 20.04/22.04 machines, reachable by SSH from the master.
- Same user account and home directory on all nodes.
- Passwordless SSH from master to all workers.

### 1.2 Install Dependencies on Every Node

```bash
# Run on EACH node (or use setup_cluster.sh from master)
sudo apt-get update
sudo apt-get install -y openmpi-bin libopenmpi-dev g++ make python3-pip
pip3 install numpy matplotlib

# Verify
mpirun --version
mpicxx --version
```

### 1.3 Passwordless SSH (from master)

```bash
ssh-keygen -t rsa -b 4096 -N "" -f ~/.ssh/id_rsa
ssh-copy-id user@node2
ssh-copy-id user@node3
ssh node2 hostname && ssh node3 hostname   # should print hostnames
```

### 1.4 Configure Hostfile

Edit `cluster_setup/hosts.conf`:

```
node1 slots=4    # master + worker
node2 slots=4    # worker
node3 slots=4    # worker
```

Replace `node1/node2/node3` with actual IPs or hostnames.

### 1.5 Sync and Build on All Nodes

```bash
# From master — build locally first
make -j4

# Sync code and build on workers
make deploy      # uses rsync + ssh (see Makefile)

# Or manually:
rsync -avz --exclude='.git' --exclude='benchmark/results/' . node2:~/para/
rsync -avz --exclude='.git' --exclude='benchmark/results/' . node3:~/para/
ssh node2 "cd ~/para && make -j4"
ssh node3 "cd ~/para && make -j4"
```

### 1.6 Smoke Test

```bash
mpirun -np 12 --hostfile cluster_setup/hosts.conf \
    ./pdsdbscan --n 1000 --eps 0.5 --min-pts 5 --verify
```

Expected output:
```
[rank 0] Dataset: 1000 points, 2 dims
=== PARALLEL RESULTS (PDSDBSCAN-D) ===
  clusters    : 10
  noise pts   : 47
  wall time   : 0.0234 s
  ...
=== VERIFICATION ===
  ARI score   : 1.000000
```

---

## 2. Building

```bash
# Parallel binary (requires mpicxx)
make

# Sequential-only binary (no MPI, for baseline timing)
make seq

# Both, then smoke test
make check
```

**Compiler requirements:** `mpicxx` (wraps g++), C++17, `-O3`.

---

## 3. Running

### 3.1 Generate a Dataset

```bash
python3 data/generate_data.py \
    --n 100000 --d 2 --clusters 10 --seed 42 \
    --output data/dataset.bin
```

### 3.2 Single Run (12 processes, 3 machines)

```bash
mpirun -np 12 \
    --hostfile cluster_setup/hosts.conf \
    --map-by slot \
    ./pdsdbscan \
        --input  data/dataset.bin \
        --eps    0.5 \
        --min-pts 5 \
        --output results/labels.bin \
        --verify \
        --timing
```

### 3.3 Single Node (development, 4 processes)

```bash
mpirun -np 4 ./pdsdbscan --n 5000 --eps 0.5 --min-pts 5 --verify
```

### 3.4 CLI Flags

| Flag          | Default | Description                                     |
|---------------|---------|-------------------------------------------------|
| `--n N`       | 10000   | Points to generate (ignored if --input given)   |
| `--d D`       | 2       | Dimensionality of generated data                |
| `--eps E`     | 0.5     | DBSCAN epsilon radius                           |
| `--min-pts M` | 5       | DBSCAN minimum neighborhood size               |
| `--clusters K`| 10      | Clusters in generated data                      |
| `--seed S`    | 42      | Random seed                                     |
| `--input F`   | —       | Read points from .bin file (overrides --n)      |
| `--output F`  | —       | Write labels to .bin file                       |
| `--verify`    | off     | Compare against sequential, print ARI           |
| `--timing`    | off     | Print per-rank compute/comm breakdown           |

---

## 4. Full Benchmark Suite

```bash
mkdir -p benchmark/results
bash benchmark/run_benchmark.sh
```

Runs 3 experiments and generates all charts automatically:

| Experiment | Measures                                      | Output file               |
|------------|-----------------------------------------------|---------------------------|
| 1          | Wall time + compute time vs N  (12 procs)     | `exp1_runtime_vs_n.png`   |
| 2          | Per-rank compute vs comm  (granularity check) | `exp2_granularity.png`    |
| 3          | Speedup: vary procs 1→12, N=2*target          | `exp3_speedup.png`        |

Edit `TARGET_N` in `run_benchmark.sh` after seeing Experiment 1 results.

---

## 5. Algorithm Summary (PDSDBSCAN-D, Algorithm 5)

```
Phase 0  MPI_Bcast full dataset (sorted by x-coordinate)
Stage 1  Gather-Neighbors: each rank filters remote points inside extended bbox
Stage 2  Local DSDBSCAN on (local ∪ remote) points — ZERO communication
           local-local unions → immediate  (local Union-Find)
           local-remote unions → deferred  (UnionQuery pairs)
Stage 3  MPI_Gather deferred pairs + metadata → rank 0
           Global sparse Union-Find resolves all cross-boundary merges
           MPI_Bcast global remapping
Phase 4  MPI_Gather labels → rank 0, restore original order, compact IDs
```

See `report/report.md` for the full 18-page technical report with pseudocode,
complexity analysis, communication diagrams, and expected experimental results.

---

## 6. Data Format

**Input `.bin`:**
```
[int32 N] [int32 d] [N*d float64 row-major]
```

**Output `.bin`:**
```
[int32 N] [N int64 labels]   (-1 = noise)
```

Read output labels in Python:
```python
import numpy as np, struct
with open("results/labels.bin","rb") as f:
    N = struct.unpack("i", f.read(4))[0]
    labels = np.frombuffer(f.read(N*8), dtype=np.int64)
```

---

## 7. Troubleshooting

**`mpicxx: command not found`**
```bash
sudo apt-get install -y libopenmpi-dev openmpi-bin
```

**MPI can't reach workers**
```bash
# Test connectivity
mpirun -np 3 --hostfile cluster_setup/hosts.conf hostname
# If firewall is an issue:
sudo ufw allow from <worker_ip>
```

**Compilation errors (C++17)**
```bash
mpicxx --version   # need g++ >= 7
# If old: sudo apt-get install g++-9 && export CXX=g++-9 then rebuild
```

**ARI < 1.0**
- Check that all nodes run the *same compiled binary* (use `make deploy`).
- Increase `--eps` if clusters are too sparse.
- A tiny ARI drop (0.9998) near boundary points is acceptable.

---

## References

1. Patwary et al. (2012). SC'12. http://cucis.ece.northwestern.edu/publications/pdf/PatPal12.pdf
2. Ester et al. (1996). DBSCAN. KDD'96.
3. Dalcín et al. (2005). MPI for Python. JPDC.
