/*
 * Sequential DSDBSCAN implementation (Algorithm 2, Patwary et al. SC'12).
 *
 * Replaces the BFS/DFS region-growing of classic DBSCAN with Union-Find
 * merging, enabling any processing order.  Produces identical clusters.
 */

#include "dbscan_seq.hpp"
#include "union_find.hpp"
#include "kd_tree.hpp"

#include <chrono>
#include <algorithm>
#include <numeric>
#include <unordered_map>
#include <vector>

using Clock = std::chrono::steady_clock;
using Dur   = std::chrono::duration<double>;

SeqResult dbscan_sequential(const double* pts, int N, int d,
                             double eps, int min_pts)
{
    auto t0 = Clock::now();

    // Build KD-tree on all N points
    KDTree tree(pts, N, d);

    // Find neighbours and mark core points
    std::vector<std::vector<int>> nbrs(N);
    std::vector<bool> is_core(N, false);

    for (int i = 0; i < N; ++i) {
        nbrs[i]   = tree.radius_search(pts + static_cast<size_t>(i) * d, eps);
        is_core[i] = (static_cast<int>(nbrs[i].size()) >= min_pts);
    }

    // -------------------------------------------------------------------
    // Algorithm 2: DSDBSCAN
    //   for each point x:
    //     if x is core:
    //       for each neighbour x':
    //         if x' is core        → UNION(x, x')
    //         elif x' not in cluster → mark, UNION(x, x')
    // -------------------------------------------------------------------
    UnionFind uf(N);
    std::vector<bool> in_cluster(N, false);

    for (int i = 0; i < N; ++i) {
        if (!is_core[i]) continue;
        in_cluster[i] = true;

        for (int j : nbrs[i]) {
            if (j == i) continue;
            if (is_core[j]) {
                uf.unite(i, j);
                in_cluster[j] = true;
            } else if (!in_cluster[j]) {
                in_cluster[j] = true;
                uf.unite(i, j);
            }
        }
    }

    // Assign labels: cluster root as representative, -1 for noise
    std::vector<int64_t> labels(N, -1);
    for (int i = 0; i < N; ++i) {
        if (in_cluster[i])
            labels[i] = static_cast<int64_t>(uf.find(i));
    }

    // Remap roots to compact 0-based cluster IDs
    std::unordered_map<int64_t, int64_t> remap;
    for (int i = 0; i < N; ++i) {
        if (labels[i] >= 0 && !remap.count(labels[i])) {
            int64_t new_id = static_cast<int64_t>(remap.size());
            remap[labels[i]] = new_id;
        }
    }
    for (int i = 0; i < N; ++i) {
        if (labels[i] >= 0) labels[i] = remap[labels[i]];
    }

    double elapsed = Dur(Clock::now() - t0).count();
    return {std::move(labels), elapsed};
}
