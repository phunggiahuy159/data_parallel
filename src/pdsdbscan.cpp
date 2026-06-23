/*
 * PDSDBSCAN-D  (Algorithm 5, Patwary et al. SC'12)
 * Parallel DBSCAN with Union-Find on distributed memory.
 *
 * Four-phase structure:
 *   Phase 0: Sort + MPI_Bcast full dataset.
 *   Stage 1: Gather-Neighbors — each rank filters its remote points
 *            from the already-broadcast dataset using extended bounding box.
 *   Stage 2: Local Computation — DSDBSCAN on X_t ∪ X_t' (no communication).
 *            Local-local unions: immediate.
 *            Local-remote unions: deferred as (x_gid, y'_gid) pairs.
 *   Stage 3: Merging — gather deferred pairs + per-point metadata to rank 0,
 *            run global sparse Union-Find, MPI_Bcast the remapping.
 *   Phase 4: MPI_Gather final labels to rank 0, restore original order.
 */

#include "pdsdbscan.hpp"
#include "union_find.hpp"
#include "kd_tree.hpp"

#include <algorithm>
#include <chrono>
#include <cstring>
#include <limits>
#include <numeric>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include <array>
#include <cassert>

using Clock = std::chrono::steady_clock;
using Dur   = std::chrono::duration<double>;

static double now_s() { return Dur(Clock::now().time_since_epoch()).count(); }

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

// 1-D block range for this rank
static void block_range(int64_t N, int rank, int size,
                        int64_t& start, int64_t& end) {
    int64_t base = N / size, rem = N % size;
    start = static_cast<int64_t>(rank) * base + std::min(static_cast<int64_t>(rank), rem);
    end   = start + base + (rank < rem ? 1 : 0);
}

// Gather a variable-length array of int64_t from all ranks to rank 0
// Returns the concatenated data (only valid on rank 0).
static std::vector<int64_t> gather_i64(
    const std::vector<int64_t>& local, int rank, int size, MPI_Comm comm)
{
    int local_cnt = static_cast<int>(local.size());
    std::vector<int> counts(size), displs(size);
    MPI_Gather(&local_cnt, 1, MPI_INT,
               rank == 0 ? counts.data() : nullptr, 1, MPI_INT,
               0, comm);

    std::vector<int64_t> recv;
    if (rank == 0) {
        displs[0] = 0;
        for (int i = 1; i < size; ++i) displs[i] = displs[i-1] + counts[i-1];
        recv.resize(static_cast<size_t>(displs[size-1] + counts[size-1]));
    }
    MPI_Gatherv(local.data(), local_cnt, MPI_INT64_T,
                rank == 0 ? recv.data()   : nullptr,
                rank == 0 ? counts.data() : nullptr,
                rank == 0 ? displs.data() : nullptr,
                MPI_INT64_T, 0, comm);
    return recv;
}

// ---------------------------------------------------------------------------
// Stage 1: Gather-Neighbors  (Section IV-B, paper)
// ---------------------------------------------------------------------------
// The full dataset is already available via the Phase-0 Bcast.
// We simply filter from all_pts for remote points inside our extended bbox.
static void gather_neighbors(
    const std::vector<double>& all_pts, int64_t N_global, int d,
    const std::vector<double>& local_pts, int64_t local_N,
    int64_t start, int64_t end,
    double eps,
    std::vector<double>&  remote_pts,    // output
    std::vector<int64_t>& remote_gids)   // output
{
    if (local_N == 0) return;

    // Compute extended bounding box of local partition
    std::vector<double> ext_min(d,  std::numeric_limits<double>::max());
    std::vector<double> ext_max(d, -std::numeric_limits<double>::max());
    for (int64_t i = 0; i < local_N; ++i)
        for (int k = 0; k < d; ++k) {
            ext_min[k] = std::min(ext_min[k], local_pts[i*d+k]);
            ext_max[k] = std::max(ext_max[k], local_pts[i*d+k]);
        }
    for (int k = 0; k < d; ++k) { ext_min[k] -= eps;  ext_max[k] += eps; }

    // Scan all_pts, collect remote points inside extended bbox
    for (int64_t gi = 0; gi < N_global; ++gi) {
        if (gi >= start && gi < end) continue;  // skip own points
        bool inside = true;
        for (int k = 0; k < d && inside; ++k) {
            double v = all_pts[gi*d+k];
            inside = (v >= ext_min[k]) && (v <= ext_max[k]);
        }
        if (inside) {
            for (int k = 0; k < d; ++k)
                remote_pts.push_back(all_pts[gi*d+k]);
            remote_gids.push_back(gi);
        }
    }
}

// ---------------------------------------------------------------------------
// Stage 2: Local Computation  (Algorithm 5, lines 2-16)
// ---------------------------------------------------------------------------
struct LocalResult {
    std::vector<int64_t> labels;     // for local points only (-1 = noise)
    std::vector<int>     is_core;    // [local_N] — 1 if core
    std::vector<int>     in_cluster; // [local_N] — 1 if assigned
    // Deferred UnionQuery pairs: (x_gid, y'_gid)
    std::vector<int64_t> deferred;   // flat: [x0, y0, x1, y1, ...]
    double t_compute;
};

static LocalResult local_computation(
    const std::vector<double>&  all_pts,   // local + remote, flat [total_N × d]
    const std::vector<int64_t>& all_gids,  // global ids for all_pts
    int64_t local_N,                        // first local_N entries are local
    int d, double eps, int min_pts)
{
    auto t0 = Clock::now();

    int64_t total_N = static_cast<int64_t>(all_gids.size());
    LocalResult res;
    res.labels.assign(local_N, -1);

    if (local_N == 0 || total_N == 0) {
        res.t_compute = 0.0;
        return res;
    }

    // Build KD-tree on (local ∪ remote)
    KDTree tree(all_pts.data(), static_cast<int>(total_N), d);

    // Find neighbours for ALL points (needed for core-point determination)
    std::vector<std::vector<int>> nbrs(total_N);
    std::vector<bool> is_core_v(total_N, false);
    for (int64_t i = 0; i < total_N; ++i) {
        nbrs[i]     = tree.radius_search(all_pts.data() + i*d, eps);
        is_core_v[i] = (static_cast<int>(nbrs[i].size()) >= min_pts);
    }

    // -------------------------------------------------------------------
    // DSDBSCAN loop (Algorithm 5, lines 4-16):
    //   iterate ONLY over X_t (local points)
    // -------------------------------------------------------------------
    UnionFind uf(static_cast<int>(total_N));
    std::vector<bool> in_cluster_v(total_N, false);

    for (int64_t i = 0; i < local_N; ++i) {
        if (!is_core_v[i]) continue;
        in_cluster_v[i] = true;

        for (int j : nbrs[i]) {
            if (static_cast<int64_t>(j) == i) continue;

            if (static_cast<int64_t>(j) < local_N) {
                // y ∈ N (local) — merge immediately
                if (is_core_v[j]) {
                    uf.unite(static_cast<int>(i), j);
                    in_cluster_v[j] = true;
                } else if (!in_cluster_v[j]) {
                    in_cluster_v[j] = true;
                    uf.unite(static_cast<int>(i), j);
                }
            } else {
                // y' ∈ N' (remote) — defer as UnionQuery
                res.deferred.push_back(all_gids[i]);
                res.deferred.push_back(all_gids[j]);
            }
        }
    }

    // Assign labels using global-index root
    for (int64_t i = 0; i < local_N; ++i) {
        if (in_cluster_v[i]) {
            int root = uf.find(static_cast<int>(i));
            res.labels[i] = all_gids[root];
        }
    }

    // Export per-point metadata for local points (for Stage 3)
    res.is_core.resize(local_N);
    res.in_cluster.resize(local_N);
    for (int64_t i = 0; i < local_N; ++i) {
        res.is_core[i]    = is_core_v[i]    ? 1 : 0;
        res.in_cluster[i] = in_cluster_v[i] ? 1 : 0;
    }

    res.t_compute = Dur(Clock::now() - t0).count();
    return res;
}

// ---------------------------------------------------------------------------
// Stage 3: Global Merge  (Algorithm 5, lines 17-28, simplified)
// ---------------------------------------------------------------------------
// Returns: flat [old_root, canonical_root, ...] — only valid on rank 0.
static std::vector<int64_t> global_merge(
    const std::vector<int64_t>& deferred,
    const std::vector<int64_t>& local_gids,
    const std::vector<int>&     is_core_local,
    const std::vector<int>&     in_cluster_local,
    const std::vector<int64_t>& labels_local,
    int64_t local_N,
    int rank, int size, MPI_Comm comm)
{
    // --- Gather deferred pairs ---
    std::vector<int64_t> all_deferred = gather_i64(deferred, rank, size, comm);

    // --- Gather per-point metadata: [gid, is_core, in_cluster, label] × local_N ---
    std::vector<int64_t> meta_local;
    meta_local.reserve(static_cast<size_t>(local_N) * 4);
    for (int64_t i = 0; i < local_N; ++i) {
        meta_local.push_back(local_gids[i]);
        meta_local.push_back(static_cast<int64_t>(is_core_local[i]));
        meta_local.push_back(static_cast<int64_t>(in_cluster_local[i]));
        meta_local.push_back(labels_local[i]);
    }
    std::vector<int64_t> all_meta = gather_i64(meta_local, rank, size, comm);

    // --- Rank 0: run global sparse Union-Find ---
    std::vector<int64_t> flat_remap;
    if (rank == 0) {
        // Parse metadata: gid → {is_core, in_cluster, cluster-root label}
        struct Meta { bool is_core; bool in_cluster; int64_t label; };
        std::unordered_map<int64_t, Meta> meta_map;
        meta_map.reserve(all_meta.size() / 4 + 1);
        for (size_t i = 0; i < all_meta.size(); i += 4) {
            int64_t gid = all_meta[i];
            if (!meta_map.count(gid)) {   // first entry wins (local > remote)
                meta_map[gid] = { all_meta[i+1] != 0, all_meta[i+2] != 0, all_meta[i+3] };
            }
        }

        // Global union-find operates on CLUSTER-ROOT LABELS, not raw point ids
        // (a cluster's root is rarely a boundary point). A cross-partition edge
        // (x core, y') merges two clusters ONLY when BOTH endpoints are core —
        // border points never merge clusters in DBSCAN. x is always a core point
        // by construction (deferred pairs are emitted only from core points).
        GlobalUF guf;
        for (size_t i = 0; i < all_deferred.size(); i += 2) {
            int64_t x_gid  = all_deferred[i];
            int64_t yp_gid = all_deferred[i+1];
            auto itx = meta_map.find(x_gid);
            auto ity = meta_map.find(yp_gid);
            if (itx == meta_map.end() || ity == meta_map.end()) continue;
            int64_t Lx = itx->second.label;          // x core ⇒ Lx ≥ 0
            if (Lx < 0) continue;
            if (ity->second.is_core) {
                int64_t Ly = ity->second.label;
                if (Ly >= 0) guf.unite(Lx, Ly);      // both core → merge clusters
            }
            // else: y' is a border point — it does not merge clusters.
        }

        // Build flat remap over distinct cluster-root labels: [label, canonical].
        std::unordered_set<int64_t> seen;
        for (auto& kv : meta_map) {
            int64_t L = kv.second.label;
            if (L < 0 || seen.count(L)) continue;
            seen.insert(L);
            flat_remap.push_back(L);
            flat_remap.push_back(guf.find(L));
        }
    }

    // --- Bcast remap size, then data ---
    int64_t remap_sz = static_cast<int64_t>(flat_remap.size());
    MPI_Bcast(&remap_sz, 1, MPI_INT64_T, 0, comm);
    if (rank != 0) flat_remap.resize(static_cast<size_t>(remap_sz));
    if (remap_sz > 0)
        MPI_Bcast(flat_remap.data(), static_cast<int>(remap_sz),
                  MPI_INT64_T, 0, comm);

    return flat_remap;
}

// ---------------------------------------------------------------------------
// Public entry point
// ---------------------------------------------------------------------------
ParResult pdsdbscan_d(const std::vector<double>& pts,
                      int64_t N, int d,
                      double eps, int min_pts,
                      MPI_Comm comm)
{
    int rank, size;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    double t_wall_start = MPI_Wtime();
    double t_comm  = 0.0;
    double t_comp  = 0.0;

    // ----------------------------------------------------------------
    // Phase 0: Sort on rank 0, Bcast full dataset
    // ----------------------------------------------------------------
    int64_t N_global = N;
    MPI_Bcast(&N_global, 1, MPI_INT64_T, 0, comm);
    MPI_Bcast(&d,        1, MPI_INT,     0, comm);

    // inv_order: only rank 0 needs this for final restoration
    std::vector<int64_t> inv_order;

    std::vector<double> all_pts(static_cast<size_t>(N_global) * d);
    if (rank == 0) {
        // Sort input by first coordinate (x-axis → 1-D block partition)
        std::vector<int64_t> order(N_global);
        std::iota(order.begin(), order.end(), 0LL);
        std::sort(order.begin(), order.end(), [&](int64_t a, int64_t b) {
            return pts[a*d] < pts[b*d];
        });
        for (int64_t i = 0; i < N_global; ++i)
            for (int k = 0; k < d; ++k)
                all_pts[i*d+k] = pts[order[i]*d+k];

        inv_order.resize(N_global);
        for (int64_t i = 0; i < N_global; ++i) inv_order[order[i]] = i;
    }

    double tc = MPI_Wtime();
    MPI_Bcast(all_pts.data(), static_cast<int>(N_global * d), MPI_DOUBLE, 0, comm);
    t_comm += MPI_Wtime() - tc;

    // 1-D block partition
    int64_t start, end;
    block_range(N_global, rank, size, start, end);
    int64_t local_N = end - start;

    std::vector<double>  local_pts(static_cast<size_t>(local_N) * d);
    std::vector<int64_t> local_gids(local_N);
    for (int64_t i = 0; i < local_N; ++i) {
        std::copy(all_pts.begin() + (start+i)*d,
                  all_pts.begin() + (start+i+1)*d,
                  local_pts.begin() + i*d);
        local_gids[i] = start + i;
    }

    // ----------------------------------------------------------------
    // Stage 1: Gather-Neighbors
    // ----------------------------------------------------------------
    std::vector<double>  remote_pts;
    std::vector<int64_t> remote_gids;

    // Allgather was done implicitly via Bcast above — filter locally.
    // Time the filtering as "comm" (it uses the Bcast-ed buffer).
    tc = MPI_Wtime();
    gather_neighbors(all_pts, N_global, d,
                     local_pts, local_N, start, end, eps,
                     remote_pts, remote_gids);
    // (filtering is O(N_global) memory reads — treated as communication overhead)
    t_comm += MPI_Wtime() - tc;

    // Build combined (local ∪ remote) point set for local computation
    int64_t n_remote = static_cast<int64_t>(remote_gids.size());
    int64_t total_N  = local_N + n_remote;

    std::vector<double>  all_local_pts(static_cast<size_t>(total_N) * d);
    std::vector<int64_t> all_local_gids(total_N);
    std::copy(local_pts.begin(),  local_pts.end(),  all_local_pts.begin());
    std::copy(local_gids.begin(), local_gids.end(), all_local_gids.begin());
    if (n_remote > 0) {
        std::copy(remote_pts.begin(),  remote_pts.end(),
                  all_local_pts.begin()  + local_N * d);
        std::copy(remote_gids.begin(), remote_gids.end(),
                  all_local_gids.begin() + local_N);
    }

    // ----------------------------------------------------------------
    // Stage 2: Local Computation  (zero communication)
    // ----------------------------------------------------------------
    LocalResult lr = local_computation(all_local_pts, all_local_gids,
                                       local_N, d, eps, min_pts);
    t_comp = lr.t_compute;

    // ----------------------------------------------------------------
    // Stage 3: Global Merge
    // ----------------------------------------------------------------
    tc = MPI_Wtime();
    std::vector<int64_t> flat_remap = global_merge(
        lr.deferred,
        local_gids, lr.is_core, lr.in_cluster, lr.labels,
        local_N, rank, size, comm);
    t_comm += MPI_Wtime() - tc;

    // Parse remap into a hash map for O(1) lookup
    std::unordered_map<int64_t, int64_t> remap;
    remap.reserve(flat_remap.size() / 2 + 1);
    for (size_t i = 0; i < flat_remap.size(); i += 2)
        remap[flat_remap[i]] = flat_remap[i+1];

    // Apply remapping to local labels (follow chain until stable)
    auto apply_remap = [&](int64_t lbl) -> int64_t {
        if (lbl < 0) return -1;
        int64_t root = lbl;
        for (int iter = 0; iter < 100; ++iter) {
            auto it = remap.find(root);
            if (it == remap.end() || it->second == root) break;
            root = it->second;
        }
        return root;
    };
    for (int64_t i = 0; i < local_N; ++i)
        lr.labels[i] = apply_remap(lr.labels[i]);

    // ----------------------------------------------------------------
    // Phase 4: Gather labels to rank 0
    // ----------------------------------------------------------------
    // Pack as [gid0, lbl0, gid1, lbl1, ...]
    std::vector<int64_t> flat_labels;
    flat_labels.reserve(static_cast<size_t>(local_N) * 2);
    for (int64_t i = 0; i < local_N; ++i) {
        flat_labels.push_back(local_gids[i]);
        flat_labels.push_back(lr.labels[i]);
    }

    tc = MPI_Wtime();
    std::vector<int64_t> all_labels = gather_i64(flat_labels, rank, size, comm);
    t_comm += MPI_Wtime() - tc;

    // ----------------------------------------------------------------
    // Gather per-rank timing
    // ----------------------------------------------------------------
    std::array<double,2> my_times = {t_comp, t_comm};
    std::vector<double> all_times(static_cast<size_t>(size) * 2);
    MPI_Gather(my_times.data(), 2, MPI_DOUBLE,
               rank == 0 ? all_times.data() : nullptr, 2, MPI_DOUBLE,
               0, comm);

    ParResult result;
    if (rank != 0) {
        result.t_wall = result.t_compute = result.t_comm = 0.0;
        return result;
    }

    // --- Assemble labels in sorted order, then restore original order ---
    std::vector<int64_t> labels_sorted(N_global, -1);
    for (size_t i = 0; i < all_labels.size(); i += 2) {
        int64_t gid = all_labels[i];
        int64_t lbl = all_labels[i+1];
        labels_sorted[gid] = lbl;
    }

    // labels_sorted is indexed by sorted position; inv_order[i] gives the
    // sorted position of the point whose ORIGINAL index is i.
    std::vector<int64_t> labels_orig(N_global);
    for (int64_t i = 0; i < N_global; ++i)
        labels_orig[i] = labels_sorted[inv_order[i]];

    // Compact to 0-based cluster IDs
    std::unordered_map<int64_t, int64_t> compact;
    for (int64_t i = 0; i < N_global; ++i) {
        if (labels_orig[i] >= 0 && !compact.count(labels_orig[i])) {
            int64_t new_id = static_cast<int64_t>(compact.size());
            compact[labels_orig[i]] = new_id;
        }
    }
    for (int64_t i = 0; i < N_global; ++i) {
        if (labels_orig[i] >= 0)
            labels_orig[i] = compact[labels_orig[i]];
    }

    // Build per-rank timing
    result.per_rank.resize(size);
    double max_comp = 0, max_comm = 0;
    for (int r = 0; r < size; ++r) {
        result.per_rank[r] = { static_cast<double>(r),
                               all_times[r*2],
                               all_times[r*2+1] };
        max_comp = std::max(max_comp, all_times[r*2]);
        max_comm = std::max(max_comm, all_times[r*2+1]);
    }

    result.labels    = std::move(labels_orig);
    result.t_wall    = MPI_Wtime() - t_wall_start;
    result.t_compute = max_comp;
    result.t_comm    = max_comm;
    return result;
}
