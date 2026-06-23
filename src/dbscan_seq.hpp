#pragma once
/*
 * Sequential DSDBSCAN  (Algorithm 2, Patwary et al. SC'12).
 *
 * Used as:
 *   (a) ground-truth reference for ARI correctness checks, and
 *   (b) baseline T(1) for speedup measurements.
 */

#include <vector>
#include <cstdint>

struct SeqResult {
    std::vector<int64_t> labels;  // -1 = noise, >= 0 = compact cluster id
    double elapsed_sec;
};

/*
 * Run DSDBSCAN on 'pts' (flat row-major [N × d]).
 * Cluster IDs are remapped to 0-based compact integers.
 */
SeqResult dbscan_sequential(const double* pts, int N, int d,
                             double eps, int min_pts);
