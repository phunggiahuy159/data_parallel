#!/usr/bin/env bash
# =============================================================================
# setup_cluster.sh — Bootstrap an OpenMPI cluster on 3+ Ubuntu/Debian nodes
#
# Run this script ONCE on the MASTER node (node1).
# It assumes passwordless SSH access to all worker nodes is already configured.
#
# What this script does:
#   1. Installs OpenMPI and Python dependencies on all nodes
#   2. Sets up SSH key-pair for passwordless inter-node communication
#   3. Configures /etc/hosts if needed
#   4. Syncs project code to all nodes via rsync
#   5. Smoke-tests MPI connectivity
#
# =============================================================================

set -euo pipefail

# ---- EDIT THESE -------------------------------------------------------
MASTER="node1"
WORKERS=("node2" "node3")
ALL_NODES=("$MASTER" "${WORKERS[@]}")

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
REMOTE_DIR="/home/$USER/para"          # path on worker nodes
PYTHON_VERSION="python3"
# -----------------------------------------------------------------------

log() { echo -e "\033[1;34m[$(date +%H:%M:%S)] $*\033[0m"; }
ok()  { echo -e "\033[1;32m  ✓ $*\033[0m"; }
err() { echo -e "\033[1;31m  ✗ $*\033[0m"; exit 1; }

# ---------------------------------------------------------------------------
# 1. Generate SSH key if not present
# ---------------------------------------------------------------------------
log "Step 1/5: SSH key setup"
if [ ! -f ~/.ssh/id_rsa ]; then
    ssh-keygen -t rsa -b 4096 -N "" -f ~/.ssh/id_rsa
    ok "Generated ~/.ssh/id_rsa"
else
    ok "SSH key already exists"
fi

for node in "${WORKERS[@]}"; do
    if ssh-copy-id -o StrictHostKeyChecking=no "$node" 2>/dev/null; then
        ok "Copied public key to $node"
    else
        echo "  (May already be authorized, or requires manual password — check $node)"
    fi
done

# ---------------------------------------------------------------------------
# 2. Install OpenMPI + Python deps on every node
# ---------------------------------------------------------------------------
log "Step 2/5: Installing dependencies on all nodes"

INSTALL_CMD="
    sudo apt-get update -qq && \
    sudo apt-get install -y --no-install-recommends \
        openmpi-bin libopenmpi-dev \
        python3-pip python3-dev \
        && \
    pip3 install --upgrade pip && \
    pip3 install mpi4py numpy scipy scikit-learn matplotlib
"

for node in "${ALL_NODES[@]}"; do
    log "  Installing on $node ..."
    ssh -o StrictHostKeyChecking=no "$node" "$INSTALL_CMD" && ok "$node done"
done

# ---------------------------------------------------------------------------
# 3. /etc/hosts sync (optional — skip if DNS is available)
# ---------------------------------------------------------------------------
log "Step 3/5: /etc/hosts (skipped if DNS resolves hostnames)"
# Uncomment and edit if your nodes don't have DNS:
# for node in "${WORKERS[@]}"; do
#     IP=$(ssh "$node" "hostname -I | awk '{print \$1}'")
#     echo "$IP $node" | sudo tee -a /etc/hosts
# done
ok "Skipped (DNS assumed)"

# ---------------------------------------------------------------------------
# 4. Rsync project to workers
# ---------------------------------------------------------------------------
log "Step 4/5: Syncing project files to worker nodes"
for node in "${WORKERS[@]}"; do
    rsync -avz --delete \
        --exclude="*.pyc" --exclude="__pycache__" \
        --exclude="benchmark/results/" \
        "$PROJECT_DIR/" "$node:$REMOTE_DIR/"
    ok "Synced to $node:$REMOTE_DIR"
done

# ---------------------------------------------------------------------------
# 5. Smoke test
# ---------------------------------------------------------------------------
log "Step 5/5: MPI smoke test"
cd "$PROJECT_DIR"
mpirun -np "${#ALL_NODES[@]}" \
       --hostfile cluster_setup/hosts.conf \
       --map-by slot \
       $PYTHON_VERSION -c "
from mpi4py import MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
import socket
print(f'  rank {rank}/{size} on {socket.gethostname()}')
" && ok "MPI test passed — ${#ALL_NODES[@]} processes across ${#ALL_NODES[@]} nodes"

log "Cluster setup complete!"
echo ""
echo "  To run the full benchmark:"
echo "    cd $PROJECT_DIR"
echo "    bash benchmark/run_benchmark.sh"
