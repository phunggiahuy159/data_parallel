#!/usr/bin/env bash
# Run this script on EACH node individually if setup_cluster.sh is not used.
set -euo pipefail

sudo apt-get update -qq
sudo apt-get install -y --no-install-recommends \
    openmpi-bin libopenmpi-dev \
    python3-pip python3-dev \
    build-essential

pip3 install --upgrade pip
pip3 install mpi4py numpy scipy scikit-learn matplotlib

echo "Done. Verify with: mpirun --version && python3 -c 'from mpi4py import MPI; print(MPI.Get_library_version())'"
