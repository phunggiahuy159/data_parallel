#pragma once
/*
 * Simple KD-tree supporting radius (range) queries.
 *
 * Points are stored externally as a flat row-major array:
 *   pts[i * d + k]  = k-th coordinate of point i
 *
 * Builds in O(N log N), answers each radius query in O(log N + k)
 * where k = number of results.
 */

#include <vector>
#include <algorithm>
#include <numeric>
#include <cmath>
#include <cstdint>
#include <cassert>

class KDTree {
    struct Node {
        int point_idx;   // index into the pts array
        int left, right; // child indices (-1 = none)
        int split_dim;   // -1 for leaves (hi - lo == 1)
        double split_val;
    };

    std::vector<Node> nodes_;
    const double*     pts_;    // external pointer, not owned
    int               d_;

    // Recursive build: points covered by idx[lo..hi)
    int build(std::vector<int>& idx, int lo, int hi, int depth) {
        if (lo >= hi) return -1;

        int sdim = depth % d_;
        int mid  = (lo + hi) / 2;

        // Partial sort: idx[mid] gets the median along sdim
        std::nth_element(
            idx.begin() + lo, idx.begin() + mid, idx.begin() + hi,
            [&](int a, int b) {
                return pts_[static_cast<size_t>(a) * d_ + sdim] <
                       pts_[static_cast<size_t>(b) * d_ + sdim];
            });

        Node node;
        node.point_idx = idx[mid];
        node.split_dim = (hi - lo > 1) ? sdim : -1;
        node.split_val = pts_[static_cast<size_t>(idx[mid]) * d_ + sdim];
        node.left = node.right = -1;

        nodes_.push_back(node);            // push before recursing
        int nid = static_cast<int>(nodes_.size()) - 1;

        if (hi - lo > 1) {
            // nodes_ is pre-reserved → no reallocation → nid remains valid
            nodes_[nid].left  = build(idx, lo,      mid,  depth + 1);
            nodes_[nid].right = build(idx, mid + 1, hi,   depth + 1);
        }
        return nid;
    }

    void query_(int nid, const double* q, double eps2,
                std::vector<int>& out) const {
        if (nid < 0) return;
        const Node& nd = nodes_[nid];

        // Distance from q to this node's point
        double dist2 = 0.0;
        for (int k = 0; k < d_; ++k) {
            double diff = pts_[static_cast<size_t>(nd.point_idx) * d_ + k] - q[k];
            dist2 += diff * diff;
            if (dist2 > eps2) break;     // early termination
        }
        if (dist2 <= eps2) out.push_back(nd.point_idx);

        if (nd.split_dim < 0) return;    // leaf — no children

        double ds = q[nd.split_dim] - nd.split_val;
        int near_child = (ds <= 0) ? nd.left  : nd.right;
        int far_child  = (ds <= 0) ? nd.right : nd.left;

        query_(near_child, q, eps2, out);
        // Visit far child only if its half-space overlaps the ball
        if (ds * ds <= eps2)
            query_(far_child, q, eps2, out);
    }

public:
    // pts: flat row-major array [N × d], must outlive this object
    KDTree(const double* pts, int n, int d) : pts_(pts), d_(d) {
        if (n <= 0) return;
        nodes_.reserve(n);           // exactly n nodes → no reallocation
        std::vector<int> idx(n);
        std::iota(idx.begin(), idx.end(), 0);
        build(idx, 0, n, 0);
    }

    // Returns ALL indices i such that dist(q, pts[i]) <= eps
    std::vector<int> radius_search(const double* q, double eps) const {
        std::vector<int> res;
        query_(0, q, eps * eps, res);
        return res;
    }
};
