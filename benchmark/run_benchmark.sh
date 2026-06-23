#!/usr/bin/env bash
# =============================================================================
# run_benchmark.sh  —  Full benchmark suite for PDSDBSCAN-D (C++ binary)
#
# Experiments (as required by course rubric):
#   1. Runtime vs input size  (find N for 2-3 min,  12 processes)
#   2. Granularity check      (per-rank timing at target N)
#   3. Speedup analysis       (input = 2*N, vary procs 1→12)
#
# Usage:
#   cd /path/to/para
#   bash benchmark/run_benchmark.sh
# =============================================================================

set -euo pipefail

BINARY="./pdsdbscan"
HOSTFILE="cluster_setup/hosts.conf"
RESULTS="benchmark/results"
TOTAL_PROCS=12          # 3 machines × 4 cores
EPS=0.5
MIN_PTS=5
SEED=42
DIM=2

mkdir -p "$RESULTS" data

log() { echo "[$(date +%H:%M:%S)] $*"; }

# ---------------------------------------------------------------------------
# Check binary is built
# ---------------------------------------------------------------------------
if [ ! -f "$BINARY" ]; then
    log "Building pdsdbscan..."
    make -j"$(nproc)"
fi

# ---------------------------------------------------------------------------
# Helper: run one MPI job, append timing line to CSV
# ---------------------------------------------------------------------------
run_one() {
    local np=$1 n=$2 csv=$3
    local binout="${RESULTS}/run_np${np}_n${n}.bin"
    local timefile="${RESULTS}/run_np${np}_n${n}.time"

    log "  mpirun -np $np  N=$n"

    # Generate dataset
    python3 data/generate_data.py --n "$n" --d "$DIM" \
        --seed "$SEED" --output "data/tmp_dataset.bin"

    # Run parallel DBSCAN, capture timing
    { time mpirun -np "$np" \
           --hostfile "$HOSTFILE" \
           --map-by slot \
           "$BINARY" \
               --input data/tmp_dataset.bin \
               --eps "$EPS" --min-pts "$MIN_PTS" \
               --output "$binout"; } 2>&1 | tee "$timefile"

    # Parse wall time from output
    wall=$(grep "wall time" "$timefile" | grep -oP '[0-9]+\.[0-9]+' | head -1)
    comp=$(grep "max compute" "$timefile" | grep -oP '[0-9]+\.[0-9]+' | head -1)
    comm=$(grep "max comm"   "$timefile" | grep -oP '[0-9]+\.[0-9]+' | head -1)

    echo "$np,$n,${wall:-0},${comp:-0},${comm:-0}" >> "$csv"
}

run_one_timing() {
    local np=$1 n=$2 csv=$3
    local binout="${RESULTS}/run_np${np}_n${n}_timing.bin"
    local timefile="${RESULTS}/run_np${np}_n${n}_timing.time"

    log "  mpirun -np $np  N=$n  (with --timing)"
    python3 data/generate_data.py --n "$n" --d "$DIM" \
        --seed "$SEED" --output "data/tmp_dataset.bin"

    mpirun -np "$np" \
           --hostfile "$HOSTFILE" \
           --map-by slot \
           "$BINARY" \
               --input data/tmp_dataset.bin \
               --eps "$EPS" --min-pts "$MIN_PTS" \
               --timing \
               --output "$binout" | tee "$timefile"

    echo "$np,$n,$timefile" >> "$csv"
}

# ---------------------------------------------------------------------------
# Experiment 1: Runtime vs N  (fixed 12 processes)
# ---------------------------------------------------------------------------
log "=== Experiment 1: Runtime vs N (np=$TOTAL_PROCS) ==="
CSV1="${RESULTS}/exp1_runtime_vs_n.csv"
echo "n_procs,n_points,wall_s,compute_s,comm_s" > "$CSV1"

for N in 10000 20000 30000 50000 80000 100000 150000 200000 300000; do
    run_one "$TOTAL_PROCS" "$N" "$CSV1"
done
log "Experiment 1 done → $CSV1"

# ---------------------------------------------------------------------------
# Experiment 2: Granularity check (target N)
# ---------------------------------------------------------------------------
# Adjust TARGET_N based on Experiment 1 results (where wall ≈ 120-180 s)
TARGET_N=100000
log "=== Experiment 2: Granularity check (N=$TARGET_N, np=$TOTAL_PROCS) ==="
CSV2="${RESULTS}/exp2_granularity.csv"
echo "n_procs,n_points,timing_file" > "$CSV2"
run_one_timing "$TOTAL_PROCS" "$TARGET_N" "$CSV2"
log "Experiment 2 done → $CSV2"

# ---------------------------------------------------------------------------
# Experiment 3: Speedup  (N = 2*TARGET_N, vary procs)
# ---------------------------------------------------------------------------
DOUBLE_N=$((TARGET_N * 2))
log "=== Experiment 3: Speedup (N=$DOUBLE_N, procs: 1 2 4 8 12) ==="
CSV3="${RESULTS}/exp3_speedup.csv"
echo "n_procs,n_points,wall_s,compute_s,comm_s" > "$CSV3"

for NP in 1 2 4 8 12; do
    run_one "$NP" "$DOUBLE_N" "$CSV3"
done
log "Experiment 3 done → $CSV3"

# ---------------------------------------------------------------------------
# Generate plots
# ---------------------------------------------------------------------------
log "=== Generating plots ==="
python3 benchmark/plot_results.py \
    --exp1 "$CSV1" \
    --exp2 "${RESULTS}/exp2_granularity.csv" \
    --exp3 "$CSV3" \
    --outdir "$RESULTS"

log "All done. Results and plots in $RESULTS/"
