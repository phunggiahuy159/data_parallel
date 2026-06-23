#pragma once
/*
 * Union-Find (Disjoint-Set) data structures.
 *
 * UnionFind     — dense array, for local computation (indices 0..n-1).
 * GlobalUF      — sparse hash-map, for global merge on rank 0.
 *
 * Both use Rem's algorithm (path halving + union by rank), as described in:
 *   Patwary et al. SC'12 — "A New Scalable Parallel DBSCAN Algorithm
 *   Using the Disjoint-Set Data Structure."
 */

#include <vector>
#include <unordered_map>
#include <numeric>
#include <algorithm>
#include <cstdint>

// ---------------------------------------------------------------------------
// Dense Union-Find (local computation, indices 0 .. n-1)
// ---------------------------------------------------------------------------
class UnionFind {
    std::vector<int> parent_;
    std::vector<int> rank_;

public:
    explicit UnionFind(int n) : parent_(n), rank_(n, 0) {
        std::iota(parent_.begin(), parent_.end(), 0);
    }

    // Iterative find with path halving (Rem's algorithm approximation)
    int find(int x) {
        while (parent_[x] != x) {
            parent_[x] = parent_[parent_[x]];   // splice / path halving
            x = parent_[x];
        }
        return x;
    }

    // Union by rank; returns true if a merge happened
    bool unite(int x, int y) {
        x = find(x);  y = find(y);
        if (x == y) return false;
        if (rank_[x] < rank_[y]) std::swap(x, y);
        parent_[y] = x;
        if (rank_[x] == rank_[y]) ++rank_[x];
        return true;
    }

    bool connected(int x, int y) { return find(x) == find(y); }
    int  size()  const { return static_cast<int>(parent_.size()); }
};

// ---------------------------------------------------------------------------
// Sparse Union-Find (global merge, arbitrary int64 keys)
// ---------------------------------------------------------------------------
class GlobalUF {
    std::unordered_map<int64_t, int64_t> parent_;
    std::unordered_map<int64_t, int>     rank_;

    void ensure(int64_t x) {
        if (!parent_.count(x)) { parent_[x] = x;  rank_[x] = 0; }
    }

public:
    int64_t find(int64_t x) {
        ensure(x);
        while (parent_[x] != x) {
            int64_t pp = parent_[x];
            ensure(pp);
            parent_[x] = parent_[pp];   // path halving
            x = parent_[x];
        }
        return x;
    }

    void unite(int64_t x, int64_t y) {
        ensure(x);  ensure(y);
        x = find(x);  y = find(y);
        if (x == y) return;
        if (rank_[x] < rank_[y]) std::swap(x, y);
        parent_[y] = x;
        if (rank_[x] == rank_[y]) ++rank_[x];
    }
};
