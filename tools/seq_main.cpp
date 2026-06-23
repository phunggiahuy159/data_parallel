/*
 * Minimal sequential wrapper: run DSDBSCAN without MPI.
 * Used only for baseline timing (T_seq) separate from the parallel binary.
 *
 * Build: g++ -O3 -std=c++17 -Isrc -o pdsdbscan_seq \
 *             src/dbscan_seq.cpp tools/seq_main.cpp
 */

#include "dbscan_seq.hpp"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <vector>
#include <random>
#include <numeric>
#include <algorithm>
#include <stdexcept>

static std::vector<double> generate_data(int64_t N, int d, int seed) {
    std::mt19937_64 rng(seed);
    std::normal_distribution<double>       gauss(0.0, 0.35);
    std::uniform_real_distribution<double> noise(0.0, 10.0);
    std::uniform_real_distribution<double> ctr(1.0, 9.0);
    int K = 10;
    std::vector<std::vector<double>> centres(K, std::vector<double>(d));
    for (int c = 0; c < K; ++c)
        for (int k = 0; k < d; ++k) centres[c][k] = ctr(rng);
    std::vector<double> pts(static_cast<size_t>(N)*d);
    int64_t noise_n = N / 20, cluster_n = N - noise_n;
    for (int64_t i = 0; i < cluster_n; ++i) {
        int c = (int)(i % K);
        for (int k = 0; k < d; ++k) pts[i*d+k] = centres[c][k] + gauss(rng);
    }
    for (int64_t i = cluster_n; i < N; ++i)
        for (int k = 0; k < d; ++k) pts[i*d+k] = noise(rng);
    return pts;
}

int main(int argc, char** argv) {
    int64_t N = 10000;
    int d = 2, min_pts = 5, seed = 42;
    double eps = 0.5;
    for (int i = 1; i < argc; ++i) {
        if (!strcmp(argv[i],"--n"))       N       = atoll(argv[++i]);
        else if(!strcmp(argv[i],"--eps")) eps     = atof(argv[++i]);
        else if(!strcmp(argv[i],"--min-pts")) min_pts = atoi(argv[++i]);
        else if(!strcmp(argv[i],"--seed"))    seed    = atoi(argv[++i]);
    }
    auto pts = generate_data(N, d, seed);
    std::printf("Sequential DSDBSCAN  N=%lld  eps=%.3f  min_pts=%d\n",
                (long long)N, eps, min_pts);
    auto res = dbscan_sequential(pts.data(), (int)N, d, eps, min_pts);
    int64_t nc = 0, nn = 0;
    for (auto l : res.labels) { if (l >= 0) nc = std::max(nc,l+1); else ++nn; }
    std::printf("  clusters=%lld  noise=%lld  time=%.4fs\n",
                (long long)nc, (long long)nn, res.elapsed_sec);
    return 0;
}
