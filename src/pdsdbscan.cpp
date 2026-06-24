/*
 * PDSDBSCAN-D  (Algorithm 5, Patwary et al. SC'12)
 * Parallel DBSCAN with Union-Find on distributed memory.
 *
 * Communication architecture (matches the reference parallel_mpi design):
 *   Phase 0: Scatter — rank 0 Scatterv's DISJOINT contiguous chunks of the
 *            dataset (NOT a full broadcast). Each rank holds only ~N/p points.
 *   Phase 1: Geometric partition — recursive median bisection (tree topology,
 *            Comm_split + pairwise Sendrecv) so each rank owns a spatially
 *            compact region. Replaces the old "x-sort + Bcast-everything".
 *   Phase 2: Distributed halo ("extra points") — Allgather only the per-rank
 *            extended bounding boxes, then point-to-point Isend/Irecv of just
 *            the boundary points that fall inside a neighbour's region.
 *   Stage 3: Local Computation — DSDBSCAN on X_t ∪ X_t' (no communication).
 *            Local-local unions immediate; local-remote deferred as gid pairs.
 *   Stage 4: Merge — gather the SMALL boundary set (deferred pairs + per-point
 *            metadata) to rank 0, run global sparse Union-Find, Bcast remap.
 *   Phase 5: Gather (orig_id, label) to rank 0 and reconstruct.
 *
 * Versus the old version this removes the O(N·p) full broadcast and the
 * O(N·p)-style funnel; only O(boundary) crosses the wire in the hot path.
 */

#include "pdsdbscan.hpp"
#include "union_find.hpp"
#include "kd_tree.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
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

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

// 1-D block range for this rank (used only for the initial Scatterv chunks).
static void block_range(int64_t N, int rank, int size,
                        int64_t& start, int64_t& end) {
    int64_t base = N / size, rem = N % size;
    start = static_cast<int64_t>(rank) * base + std::min(static_cast<int64_t>(rank), rem);
    end   = start + base + (rank < rem ? 1 : 0);
}

// Is n a power of two (>=1)?
static bool is_pow2(int n) { return n > 0 && (n & (n - 1)) == 0; }

// Gather a variable-length array of int64_t from all ranks to rank 0.
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
// Phase 1: Geometric recursive-median partition (tree topology)
// ---------------------------------------------------------------------------
// Redistributes points so each rank owns a spatially compact region. Migrates
// coordinates AND their original ids together. Requires size = power of two;
// for non-power-of-two callers must fall back (see pdsdbscan_d).
static void geometric_partition(std::vector<double>& P,
                                std::vector<int64_t>& oid,
                                int d, MPI_Comm comm)
{
    int rank, size;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);
    if (size < 2) return;

    int L = 0; { int s = size; while (s > 1) { s >>= 1; ++L; } }

    for (int level = 0; level < L; ++level) {
        int group_size = size >> level;     // procs sharing this sub-region
        int half       = group_size / 2;
        int color      = rank / group_size; // which sub-region
        int base       = color * group_size;
        int subrank    = rank - base;

        MPI_Comm sub;
        MPI_Comm_split(comm, color, rank, &sub);

        int64_t n_local = static_cast<int64_t>(oid.size());

        // Region bounding box over this sub-communicator
        std::vector<double> lmin(d,  std::numeric_limits<double>::max());
        std::vector<double> lmax(d, -std::numeric_limits<double>::max());
        for (int64_t i = 0; i < n_local; ++i)
            for (int k = 0; k < d; ++k) {
                double v = P[i*d+k];
                lmin[k] = std::min(lmin[k], v);
                lmax[k] = std::max(lmax[k], v);
            }
        std::vector<double> gmin(d), gmax(d);
        MPI_Allreduce(lmin.data(), gmin.data(), d, MPI_DOUBLE, MPI_MIN, sub);
        MPI_Allreduce(lmax.data(), gmax.data(), d, MPI_DOUBLE, MPI_MAX, sub);

        // Split along the widest dimension
        int dd = 0; double best = gmax[0] - gmin[0];
        for (int k = 1; k < d; ++k) {
            double w = gmax[k] - gmin[k];
            if (w > best) { best = w; dd = k; }
        }

        // Approximate group median = median of per-proc local medians
        double lmed = std::numeric_limits<double>::quiet_NaN();
        if (n_local > 0) {
            std::vector<double> col(n_local);
            for (int64_t i = 0; i < n_local; ++i) col[i] = P[i*d+dd];
            std::nth_element(col.begin(), col.begin() + col.size()/2, col.end());
            lmed = col[col.size()/2];
        }
        std::vector<double> allmed(group_size);
        MPI_Allgather(&lmed, 1, MPI_DOUBLE, allmed.data(), 1, MPI_DOUBLE, sub);
        std::vector<double> valid;
        for (double m : allmed) if (!std::isnan(m)) valid.push_back(m);
        double median;
        if (valid.empty()) median = 0.5 * (gmin[dd] + gmax[dd]);
        else { std::nth_element(valid.begin(), valid.begin()+valid.size()/2, valid.end());
               median = valid[valid.size()/2]; }

        // Lower half keeps coord<=median; upper half keeps coord>median.
        int  partner = (subrank < half) ? base + subrank + half
                                        : base + subrank - half;
        bool lower   = subrank < half;

        std::vector<double>  keepP, sendP;
        std::vector<int64_t> keepO, sendO;
        for (int64_t i = 0; i < n_local; ++i) {
            bool mine = lower ? (P[i*d+dd] <= median) : (P[i*d+dd] > median);
            auto& dstP = mine ? keepP : sendP;
            auto& dstO = mine ? keepO : sendO;
            for (int k = 0; k < d; ++k) dstP.push_back(P[i*d+k]);
            dstO.push_back(oid[i]);
        }

        int scount = static_cast<int>(sendO.size()), rcount = 0;
        MPI_Sendrecv(&scount, 1, MPI_INT, partner, 10,
                     &rcount, 1, MPI_INT, partner, 10, comm, MPI_STATUS_IGNORE);
        std::vector<double>  recvP(static_cast<size_t>(rcount) * d);
        std::vector<int64_t> recvO(rcount);
        MPI_Sendrecv(sendP.data(), scount*d, MPI_DOUBLE, partner, 11,
                     recvP.data(), rcount*d, MPI_DOUBLE, partner, 11,
                     comm, MPI_STATUS_IGNORE);
        MPI_Sendrecv(sendO.data(), scount, MPI_INT64_T, partner, 12,
                     recvO.data(), rcount, MPI_INT64_T, partner, 12,
                     comm, MPI_STATUS_IGNORE);

        P = std::move(keepP);
        oid = std::move(keepO);
        for (int64_t i = 0; i < rcount; ++i) {
            for (int k = 0; k < d; ++k) P.push_back(recvP[i*d+k]);
            oid.push_back(recvO[i]);
        }

        MPI_Comm_free(&sub);
    }
}

// ---------------------------------------------------------------------------
// Phase 2: Distributed halo  ("get_extra_points", Section IV-B)
// ---------------------------------------------------------------------------
// Allgather only the extended bounding boxes, then point-to-point exchange of
// the boundary points each neighbour needs. Outputs remote points + their gids.
static void get_extra_points(const std::vector<double>& P,
                             const std::vector<int64_t>& oid,
                             int d, double eps, MPI_Comm comm,
                             std::vector<double>&  outP,
                             std::vector<int64_t>& outO)
{
    int rank, size;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);
    int64_t n = static_cast<int64_t>(oid.size());

    // Extended local bounding box
    std::vector<double> mn(d,  std::numeric_limits<double>::max());
    std::vector<double> mx(d, -std::numeric_limits<double>::max());
    for (int64_t i = 0; i < n; ++i)
        for (int k = 0; k < d; ++k) {
            double v = P[i*d+k];
            mn[k] = std::min(mn[k], v);
            mx[k] = std::max(mx[k], v);
        }
    for (int k = 0; k < d; ++k) { mn[k] -= eps; mx[k] += eps; }

    // Allgather boxes:  per rank [mn(d) , mx(d)]
    std::vector<double> mybb(2*d);
    for (int k = 0; k < d; ++k) { mybb[k] = mn[k]; mybb[d+k] = mx[k]; }
    std::vector<double> allbb(static_cast<size_t>(size) * 2 * d);
    MPI_Allgather(mybb.data(), 2*d, MPI_DOUBLE, allbb.data(), 2*d, MPI_DOUBLE, comm);

    // Collect my points that fall inside each neighbour's extended box
    std::vector<std::vector<double>>  sbufP(size);
    std::vector<std::vector<int64_t>> sbufO(size);
    for (int k = 0; k < size; ++k) {
        if (k == rank) continue;
        const double* kmn = &allbb[static_cast<size_t>(k)*2*d];
        const double* kmx = &allbb[static_cast<size_t>(k)*2*d + d];
        bool overlap = true;
        for (int j = 0; j < d; ++j)
            if (mx[j] < kmn[j] || mn[j] > kmx[j]) { overlap = false; break; }
        if (!overlap) continue;
        for (int64_t i = 0; i < n; ++i) {
            bool inside = true;
            for (int j = 0; j < d; ++j) {
                double v = P[i*d+j];
                if (v < kmn[j] || v > kmx[j]) { inside = false; break; }
            }
            if (inside) {
                for (int j = 0; j < d; ++j) sbufP[k].push_back(P[i*d+j]);
                sbufO[k].push_back(oid[i]);
            }
        }
    }

    // Exchange counts, then the points (non-blocking)
    std::vector<int> ssz(size, 0), rsz(size, 0);
    for (int k = 0; k < size; ++k) ssz[k] = static_cast<int>(sbufO[k].size());
    MPI_Alltoall(ssz.data(), 1, MPI_INT, rsz.data(), 1, MPI_INT, comm);

    std::vector<std::vector<double>>  rbufP(size);
    std::vector<std::vector<int64_t>> rbufO(size);
    std::vector<MPI_Request> reqs;
    for (int k = 0; k < size; ++k) {
        if (rsz[k] > 0) {
            rbufP[k].resize(static_cast<size_t>(rsz[k]) * d);
            rbufO[k].resize(rsz[k]);
            MPI_Request r1, r2;
            MPI_Irecv(rbufP[k].data(), rsz[k]*d, MPI_DOUBLE,  k, 21, comm, &r1);
            MPI_Irecv(rbufO[k].data(), rsz[k],   MPI_INT64_T, k, 22, comm, &r2);
            reqs.push_back(r1); reqs.push_back(r2);
        }
    }
    for (int k = 0; k < size; ++k) {
        if (ssz[k] > 0) {
            MPI_Request r1, r2;
            MPI_Isend(sbufP[k].data(), ssz[k]*d, MPI_DOUBLE,  k, 21, comm, &r1);
            MPI_Isend(sbufO[k].data(), ssz[k],   MPI_INT64_T, k, 22, comm, &r2);
            reqs.push_back(r1); reqs.push_back(r2);
        }
    }
    if (!reqs.empty())
        MPI_Waitall(static_cast<int>(reqs.size()), reqs.data(), MPI_STATUSES_IGNORE);

    for (int k = 0; k < size; ++k)
        for (int64_t i = 0; i < rsz[k]; ++i) {
            for (int j = 0; j < d; ++j) outP.push_back(rbufP[k][i*d+j]);
            outO.push_back(rbufO[k][i]);
        }
}

// ---------------------------------------------------------------------------
// Stage 3: Local Computation  (Algorithm 5, lines 2-16)  — UNCHANGED logic
// ---------------------------------------------------------------------------
struct LocalResult {
    std::vector<int64_t> labels;     // for local points only (-1 = noise)
    std::vector<int>     is_core;    // [local_N]
    std::vector<int>     in_cluster; // [local_N]
    std::vector<int64_t> deferred;   // flat: [x0, y0, x1, y1, ...]  (gid pairs)
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

    if (local_N == 0 || total_N == 0) { res.t_compute = 0.0; return res; }

    KDTree tree(all_pts.data(), static_cast<int>(total_N), d);

    std::vector<std::vector<int>> nbrs(total_N);
    std::vector<bool> is_core_v(total_N, false);
    for (int64_t i = 0; i < total_N; ++i) {
        nbrs[i]      = tree.radius_search(all_pts.data() + i*d, eps);
        is_core_v[i] = (static_cast<int>(nbrs[i].size()) >= min_pts);
    }

    UnionFind uf(static_cast<int>(total_N));
    std::vector<bool> in_cluster_v(total_N, false);

    for (int64_t i = 0; i < local_N; ++i) {
        if (!is_core_v[i]) continue;
        in_cluster_v[i] = true;
        for (int j : nbrs[i]) {
            if (static_cast<int64_t>(j) == i) continue;
            if (static_cast<int64_t>(j) < local_N) {
                if (is_core_v[j]) {
                    uf.unite(static_cast<int>(i), j);
                    in_cluster_v[j] = true;
                } else if (!in_cluster_v[j]) {
                    in_cluster_v[j] = true;
                    uf.unite(static_cast<int>(i), j);
                }
            } else {
                res.deferred.push_back(all_gids[i]);
                res.deferred.push_back(all_gids[j]);
            }
        }
    }

    for (int64_t i = 0; i < local_N; ++i)
        if (in_cluster_v[i]) {
            int root = uf.find(static_cast<int>(i));
            res.labels[i] = all_gids[root];
        }

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
// Stage 4: Global Merge of the boundary set  — UNCHANGED logic
// ---------------------------------------------------------------------------
static std::vector<int64_t> global_merge(
    const std::vector<int64_t>& deferred,
    const std::vector<int64_t>& local_gids,
    const std::vector<int>&     is_core_local,
    const std::vector<int>&     in_cluster_local,
    const std::vector<int64_t>& labels_local,
    int64_t local_N,
    int rank, int size, MPI_Comm comm)
{
    std::vector<int64_t> all_deferred = gather_i64(deferred, rank, size, comm);

    std::vector<int64_t> meta_local;
    meta_local.reserve(static_cast<size_t>(local_N) * 4);
    for (int64_t i = 0; i < local_N; ++i) {
        meta_local.push_back(local_gids[i]);
        meta_local.push_back(static_cast<int64_t>(is_core_local[i]));
        meta_local.push_back(static_cast<int64_t>(in_cluster_local[i]));
        meta_local.push_back(labels_local[i]);
    }
    std::vector<int64_t> all_meta = gather_i64(meta_local, rank, size, comm);

    std::vector<int64_t> flat_remap;
    if (rank == 0) {
        struct Meta { bool is_core; bool in_cluster; int64_t label; };
        std::unordered_map<int64_t, Meta> meta_map;
        meta_map.reserve(all_meta.size() / 4 + 1);
        for (size_t i = 0; i < all_meta.size(); i += 4) {
            int64_t gid = all_meta[i];
            if (!meta_map.count(gid))
                meta_map[gid] = { all_meta[i+1] != 0, all_meta[i+2] != 0, all_meta[i+3] };
        }

        GlobalUF guf;
        for (size_t i = 0; i < all_deferred.size(); i += 2) {
            int64_t x_gid  = all_deferred[i];
            int64_t yp_gid = all_deferred[i+1];
            auto itx = meta_map.find(x_gid);
            auto ity = meta_map.find(yp_gid);
            if (itx == meta_map.end() || ity == meta_map.end()) continue;
            int64_t Lx = itx->second.label;
            if (Lx < 0) continue;
            if (ity->second.is_core) {
                int64_t Ly = ity->second.label;
                if (Ly >= 0) guf.unite(Lx, Ly);
            }
        }

        std::unordered_set<int64_t> seen;
        for (auto& kv : meta_map) {
            int64_t L = kv.second.label;
            if (L < 0 || seen.count(L)) continue;
            seen.insert(L);
            flat_remap.push_back(L);
            flat_remap.push_back(guf.find(L));
        }
    }

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
    double t_comm = 0.0, t_comp = 0.0, tc;

    // ----------------------------------------------------------------
    // Phase 0: broadcast meta, then Scatterv disjoint chunks (no Bcast!)
    // ----------------------------------------------------------------
    int64_t N_global = N;
    MPI_Bcast(&N_global, 1, MPI_INT64_T, 0, comm);
    MPI_Bcast(&d,        1, MPI_INT,     0, comm);

    int64_t start, end;
    block_range(N_global, rank, size, start, end);
    int64_t local_N0 = end - start;

    std::vector<double>  local_pts(static_cast<size_t>(local_N0) * d);
    std::vector<int64_t> local_oid(local_N0);
    for (int64_t i = 0; i < local_N0; ++i) local_oid[i] = start + i;

    // Scatterv coordinates from rank 0
    std::vector<int> sc_cnt(size), sc_dis(size);
    for (int r = 0; r < size; ++r) {
        int64_t s, e; block_range(N_global, r, size, s, e);
        sc_cnt[r] = static_cast<int>((e - s) * d);
        sc_dis[r] = static_cast<int>(s * d);
    }
    tc = MPI_Wtime();
    MPI_Scatterv(rank == 0 ? pts.data() : nullptr,
                 sc_cnt.data(), sc_dis.data(), MPI_DOUBLE,
                 local_pts.data(), static_cast<int>(local_N0 * d), MPI_DOUBLE,
                 0, comm);
    t_comm += MPI_Wtime() - tc;

    // ----------------------------------------------------------------
    // Phase 1: geometric median partition (tree topology) — power-of-2 only
    // ----------------------------------------------------------------
    if (is_pow2(size) && size > 1) {
        tc = MPI_Wtime();
        geometric_partition(local_pts, local_oid, d, comm);
        t_comm += MPI_Wtime() - tc;
    } else if (size > 1) {
        // Fallback for non-power-of-two: globally x-sort then Scatterv compact
        // slabs so regions stay compact (correct either way; halo handles it).
        std::vector<double>  sorted;
        std::vector<int64_t> order_oid;
        if (rank == 0) {
            std::vector<int64_t> order(N_global);
            std::iota(order.begin(), order.end(), 0LL);
            std::sort(order.begin(), order.end(),
                      [&](int64_t a, int64_t b){ return pts[a*d] < pts[b*d]; });
            sorted.resize(static_cast<size_t>(N_global) * d);
            order_oid.resize(N_global);
            for (int64_t i = 0; i < N_global; ++i) {
                order_oid[i] = order[i];
                for (int k = 0; k < d; ++k) sorted[i*d+k] = pts[order[i]*d+k];
            }
        }
        tc = MPI_Wtime();
        MPI_Scatterv(rank == 0 ? sorted.data() : nullptr,
                     sc_cnt.data(), sc_dis.data(), MPI_DOUBLE,
                     local_pts.data(), static_cast<int>(local_N0 * d), MPI_DOUBLE,
                     0, comm);
        // ship the matching oids too
        std::vector<int> oc(size), od(size);
        for (int r = 0; r < size; ++r) { int64_t s,e; block_range(N_global,r,size,s,e);
            oc[r] = static_cast<int>(e - s); od[r] = static_cast<int>(s); }
        MPI_Scatterv(rank == 0 ? order_oid.data() : nullptr,
                     oc.data(), od.data(), MPI_INT64_T,
                     local_oid.data(), static_cast<int>(local_N0), MPI_INT64_T,
                     0, comm);
        t_comm += MPI_Wtime() - tc;
    }

    int64_t local_N = static_cast<int64_t>(local_oid.size());

    // ----------------------------------------------------------------
    // Phase 2: distributed halo (extra points)
    // ----------------------------------------------------------------
    std::vector<double>  remote_pts;
    std::vector<int64_t> remote_gids;
    if (size > 1) {
        tc = MPI_Wtime();
        get_extra_points(local_pts, local_oid, d, eps, comm, remote_pts, remote_gids);
        t_comm += MPI_Wtime() - tc;
    }

    // Combined (local ∪ remote) set for local computation
    int64_t n_remote = static_cast<int64_t>(remote_gids.size());
    int64_t total_N  = local_N + n_remote;
    std::vector<double>  all_local_pts(static_cast<size_t>(total_N) * d);
    std::vector<int64_t> all_local_gids(total_N);
    std::copy(local_pts.begin(),  local_pts.end(),  all_local_pts.begin());
    std::copy(local_oid.begin(),  local_oid.end(),  all_local_gids.begin());
    if (n_remote > 0) {
        std::copy(remote_pts.begin(),  remote_pts.end(),
                  all_local_pts.begin()  + local_N * d);
        std::copy(remote_gids.begin(), remote_gids.end(),
                  all_local_gids.begin() + local_N);
    }

    // ----------------------------------------------------------------
    // Stage 3: Local Computation
    // ----------------------------------------------------------------
    LocalResult lr = local_computation(all_local_pts, all_local_gids,
                                       local_N, d, eps, min_pts);
    t_comp = lr.t_compute;

    // ----------------------------------------------------------------
    // Stage 4: Merge boundary set on rank 0
    // ----------------------------------------------------------------
    tc = MPI_Wtime();
    std::vector<int64_t> flat_remap = global_merge(
        lr.deferred, local_oid, lr.is_core, lr.in_cluster, lr.labels,
        local_N, rank, size, comm);
    t_comm += MPI_Wtime() - tc;

    std::unordered_map<int64_t, int64_t> remap;
    remap.reserve(flat_remap.size() / 2 + 1);
    for (size_t i = 0; i < flat_remap.size(); i += 2)
        remap[flat_remap[i]] = flat_remap[i+1];
    auto apply_remap = [&](int64_t lbl) -> int64_t {
        if (lbl < 0) return -1;
        int64_t root = lbl;
        for (int it = 0; it < 100; ++it) {
            auto f = remap.find(root);
            if (f == remap.end() || f->second == root) break;
            root = f->second;
        }
        return root;
    };
    for (int64_t i = 0; i < local_N; ++i)
        lr.labels[i] = apply_remap(lr.labels[i]);

    // ----------------------------------------------------------------
    // Phase 5: Gather (orig_id, label) to rank 0
    // ----------------------------------------------------------------
    std::vector<int64_t> flat_labels;
    flat_labels.reserve(static_cast<size_t>(local_N) * 2);
    for (int64_t i = 0; i < local_N; ++i) {
        flat_labels.push_back(local_oid[i]);
        flat_labels.push_back(lr.labels[i]);
    }
    tc = MPI_Wtime();
    std::vector<int64_t> all_labels = gather_i64(flat_labels, rank, size, comm);
    t_comm += MPI_Wtime() - tc;

    // Per-rank timing
    std::array<double,2> my_times = {t_comp, t_comm};
    std::vector<double> all_times(static_cast<size_t>(size) * 2);
    MPI_Gather(my_times.data(), 2, MPI_DOUBLE,
               rank == 0 ? all_times.data() : nullptr, 2, MPI_DOUBLE, 0, comm);

    ParResult result;
    if (rank != 0) {
        result.t_wall = result.t_compute = result.t_comm = 0.0;
        return result;
    }

    // Reconstruct labels in original order (each point carries its orig id)
    std::vector<int64_t> labels_orig(N_global, -1);
    for (size_t i = 0; i < all_labels.size(); i += 2)
        labels_orig[all_labels[i]] = all_labels[i+1];

    std::unordered_map<int64_t, int64_t> compact;
    for (int64_t i = 0; i < N_global; ++i)
        if (labels_orig[i] >= 0 && !compact.count(labels_orig[i])) {
            int64_t new_id = static_cast<int64_t>(compact.size());
            compact[labels_orig[i]] = new_id;
        }
    for (int64_t i = 0; i < N_global; ++i)
        if (labels_orig[i] >= 0) labels_orig[i] = compact[labels_orig[i]];

    result.per_rank.resize(size);
    double max_comp = 0, max_comm = 0;
    for (int r = 0; r < size; ++r) {
        result.per_rank[r] = { static_cast<double>(r), all_times[r*2], all_times[r*2+1] };
        max_comp = std::max(max_comp, all_times[r*2]);
        max_comm = std::max(max_comm, all_times[r*2+1]);
    }

    result.labels    = std::move(labels_orig);
    result.t_wall    = MPI_Wtime() - t_wall_start;
    result.t_compute = max_comp;
    result.t_comm    = max_comm;
    return result;
}
