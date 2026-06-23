#pragma once
/*
 * PDSDBSCAN-D — Parallel Disjoint-Set DBSCAN for Distributed Memory
 *
 * Algorithm 5 from:
 *   Patwary, Palsetia, Agrawal, Liao, Manne, Choudhary. SC'12.
 *
 * Exports one function: pdsdbscan_d()
 */

#include <mpi.h>
#include <vector>
#include <array>
#include <cstdint>

struct ParResult {
    std::vector<int64_t> labels;     // only valid on rank 0; compact 0-based
    double t_wall;                   // rank 0: wall time from start to finish
    double t_compute;                // max compute time across all ranks
    double t_comm;                   // max comm time across all ranks
    // per_rank[i] = {rank, t_compute_i, t_comm_i}
    std::vector<std::array<double,3>> per_rank;
};

/*
 * Run PDSDBSCAN-D on 'pts' (flat row-major [N × d]).
 * 'pts' must be the full dataset and is only meaningful on rank 0 before Bcast.
 * Returns a filled ParResult on rank 0; other ranks return an empty struct.
 */
ParResult pdsdbscan_d(const std::vector<double>& pts,
                      int64_t N, int d,
                      double eps, int min_pts,
                      MPI_Comm comm = MPI_COMM_WORLD);
