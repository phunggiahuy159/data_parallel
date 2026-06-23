/*
 * main.cpp — MPI entry point for PDSDBSCAN-D
 *
 * Usage (single machine, 4 processes):
 *   mpirun -np 4 ./pdsdbscan --n 5000 --eps 0.5 --min-pts 5 --verify
 *
 * Usage (3-node cluster, 12 processes):
 *   mpirun -np 12 --hostfile cluster_setup/hosts.conf \
 *          ./pdsdbscan --input data/dataset.bin \
 *                      --eps 0.5 --min-pts 5 --output results/out.bin \
 *                      --verify --timing
 *
 * Data format (.bin):
 *   Header : int32 N, int32 d
 *   Points : N * d  float64 (row-major)
 *   Output : int32 N, then N int64 labels (-1 = noise)
 */

#include "pdsdbscan.hpp"
#include "dbscan_seq.hpp"

#include <mpi.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>
#include <algorithm>
#include <numeric>
#include <random>
#include <unordered_map>

// Source fingerprint, injected by the Makefile (-DBUILD_HASH). Lets every rank
// prove it was built from identical source. Falls back to "dev" if unset.
#ifndef BUILD_HASH
#define BUILD_HASH "dev"
#endif

// ---------------------------------------------------------------------------
// Argument parsing
// ---------------------------------------------------------------------------
struct Args {
    int64_t N        = 10000;
    int     d        = 2;
    double  eps      = 0.5;
    int     min_pts  = 5;
    int     seed     = 42;
    int     n_clust  = 10;
    std::string input;
    std::string output;
    bool verify  = false;
    bool timing  = false;
};

static void usage(const char* prog) {
    std::fprintf(stderr,
        "Usage: mpirun -np P %s [options]\n"
        "  --n N          Number of points to generate (default 10000)\n"
        "  --d D          Dimensions (default 2)\n"
        "  --eps E        DBSCAN epsilon (default 0.5)\n"
        "  --min-pts M    DBSCAN min_pts (default 5)\n"
        "  --clusters K   Number of clusters to generate (default 10)\n"
        "  --seed S       Random seed (default 42)\n"
        "  --input FILE   Read points from .bin file (overrides --n)\n"
        "  --output FILE  Write labels to .bin file\n"
        "  --verify       Also run sequential DSDBSCAN and compute ARI\n"
        "  --timing       Print per-rank compute/comm breakdown\n", prog);
    std::exit(1);
}

static Args parse_args(int argc, char** argv) {
    Args a;
    for (int i = 1; i < argc; ++i) {
        std::string s(argv[i]);
        auto next = [&]() -> std::string {
            if (i+1 >= argc) { std::fprintf(stderr,"Missing value for %s\n",argv[i]); std::exit(1); }
            return std::string(argv[++i]);
        };
        if      (s == "--n")        a.N       = std::stoll(next());
        else if (s == "--d")        a.d       = std::stoi(next());
        else if (s == "--eps")      a.eps     = std::stod(next());
        else if (s == "--min-pts")  a.min_pts = std::stoi(next());
        else if (s == "--clusters") a.n_clust = std::stoi(next());
        else if (s == "--seed")     a.seed    = std::stoi(next());
        else if (s == "--input")    a.input   = next();
        else if (s == "--output")   a.output  = next();
        else if (s == "--verify")   a.verify  = true;
        else if (s == "--timing")   a.timing  = true;
        else if (s == "--help")     usage(argv[0]);
        else { std::fprintf(stderr,"Unknown option: %s\n", s.c_str()); usage(argv[0]); }
    }
    return a;
}

// ---------------------------------------------------------------------------
// Data I/O
// ---------------------------------------------------------------------------
static std::vector<double> read_bin(const std::string& path, int& N, int& d) {
    std::ifstream f(path, std::ios::binary);
    if (!f) { std::fprintf(stderr,"Cannot open %s\n", path.c_str()); std::exit(1); }
    int32_t iN, id;
    f.read(reinterpret_cast<char*>(&iN), 4);
    f.read(reinterpret_cast<char*>(&id), 4);
    N = iN;  d = id;
    std::vector<double> pts(static_cast<size_t>(N)*d);
    f.read(reinterpret_cast<char*>(pts.data()),
           static_cast<std::streamsize>(pts.size() * sizeof(double)));
    return pts;
}

static void write_bin(const std::string& path,
                      const std::vector<int64_t>& labels) {
    std::ofstream f(path, std::ios::binary);
    int32_t N = static_cast<int32_t>(labels.size());
    f.write(reinterpret_cast<const char*>(&N), 4);
    f.write(reinterpret_cast<const char*>(labels.data()),
            static_cast<std::streamsize>(labels.size() * sizeof(int64_t)));
}

// ---------------------------------------------------------------------------
// Synthetic data generator (Gaussian blobs)
// ---------------------------------------------------------------------------
static std::vector<double> generate_data(int64_t N, int d, int n_clusters,
                                         double noise_frac, int seed) {
    std::mt19937_64 rng(seed);
    std::uniform_real_distribution<double> centre_dist(1.0, 9.0);
    std::normal_distribution<double>       gauss(0.0, 0.35);
    std::uniform_real_distribution<double> noise_dist(0.0, 10.0);

    // Cluster centres
    std::vector<std::vector<double>> centres(n_clusters, std::vector<double>(d));
    for (int c = 0; c < n_clusters; ++c)
        for (int k = 0; k < d; ++k)
            centres[c][k] = centre_dist(rng);

    int64_t n_noise   = static_cast<int64_t>(N * noise_frac);
    int64_t n_cluster = N - n_noise;

    std::vector<double> pts(static_cast<size_t>(N) * d);
    int64_t idx = 0;
    for (int64_t i = 0; i < n_cluster; ++i) {
        int c = static_cast<int>(i % n_clusters);
        for (int k = 0; k < d; ++k)
            pts[idx*d+k] = centres[c][k] + gauss(rng);
        ++idx;
    }
    for (int64_t i = 0; i < n_noise; ++i) {
        for (int k = 0; k < d; ++k)
            pts[idx*d+k] = noise_dist(rng);
        ++idx;
    }
    // Shuffle
    std::vector<int64_t> perm(N);
    std::iota(perm.begin(), perm.end(), 0LL);
    std::shuffle(perm.begin(), perm.end(), rng);
    std::vector<double> shuffled(static_cast<size_t>(N) * d);
    for (int64_t i = 0; i < N; ++i)
        for (int k = 0; k < d; ++k)
            shuffled[i*d+k] = pts[perm[i]*d+k];
    return shuffled;
}

// ---------------------------------------------------------------------------
// ARI  (Adjusted Rand Index, naive O(N²) computation)
// ---------------------------------------------------------------------------
static double adjusted_rand_index(const std::vector<int64_t>& a,
                                   const std::vector<int64_t>& b) {
    int64_t N = static_cast<int64_t>(a.size());
    // Count contingency table n_ij, a_i, b_j
    std::unordered_map<int64_t, std::unordered_map<int64_t,int64_t>> cont;
    std::unordered_map<int64_t,int64_t> sum_a, sum_b;
    int64_t n_both = 0;
    for (int64_t i = 0; i < N; ++i) {
        if (a[i] < 0 || b[i] < 0) continue;
        cont[a[i]][b[i]]++;
        sum_a[a[i]]++;
        sum_b[b[i]]++;
        ++n_both;
    }
    auto C2 = [](int64_t x) -> double { return x*(x-1)/2.0; };
    double sum_cij = 0;
    for (auto& [ai, row] : cont) for (auto& [bi, n] : row) sum_cij += C2(n);
    double sum_ai = 0;
    for (auto& [ai, n] : sum_a) sum_ai += C2(n);
    double sum_bi = 0;
    for (auto& [bi, n] : sum_b) sum_bi += C2(n);
    double Cn = C2(n_both);
    double expected = sum_ai * sum_bi / std::max(Cn, 1.0);
    double denom    = 0.5 * (sum_ai + sum_bi) - expected;
    if (std::abs(denom) < 1e-12) return 1.0;
    return (sum_cij - expected) / denom;
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------
int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // --- Build-consistency guard --------------------------------------------
    // Every rank carries a fingerprint of the source it was compiled from. If
    // the nodes are running different binaries (the classic "a teammate has an
    // old version" bug), detect it now and abort with a clear message instead
    // of producing silently-wrong clustering results.
    {
        char my[16];
        std::memset(my, 0, sizeof(my));
        std::strncpy(my, BUILD_HASH, sizeof(my) - 1);
        std::vector<char> all(static_cast<size_t>(size) * 16);
        MPI_Gather(my, 16, MPI_CHAR,
                   rank == 0 ? all.data() : nullptr, 16, MPI_CHAR,
                   0, MPI_COMM_WORLD);
        int bad = 0;
        if (rank == 0) {
            for (int r = 1; r < size; ++r)
                if (std::memcmp(all.data(), all.data() + r*16, 16) != 0) { bad = 1; break; }
            if (bad) {
                std::fprintf(stderr,
                    "\n*** BUILD MISMATCH — ranks are running DIFFERENT binaries ***\n");
                for (int r = 0; r < size; ++r)
                    std::fprintf(stderr, "    rank %2d : build %s\n", r, all.data() + r*16);
                std::fprintf(stderr,
                    "All nodes must run the SAME build. From the master node run\n"
                    "`make deploy` (rebuilds identically everywhere), then re-run.\n\n");
            }
        }
        MPI_Bcast(&bad, 1, MPI_INT, 0, MPI_COMM_WORLD);
        if (bad) { MPI_Finalize(); return 2; }
        if (rank == 0)
            std::printf("[rank 0] build %s — consistent across %d rank(s)\n",
                        BUILD_HASH, size);
    }

    Args a = parse_args(argc, argv);

    // ----------------------------------------------------------------
    // Load / generate data on rank 0
    // ----------------------------------------------------------------
    std::vector<double> pts;
    int64_t N = 0;
    int d = a.d;

    if (rank == 0) {
        if (!a.input.empty()) {
            int iN, id;
            pts = read_bin(a.input, iN, id);
            N = iN;  d = id;
        } else {
            pts = generate_data(a.N, a.d, a.n_clust, 0.05, a.seed);
            N = a.N;
        }
        std::printf("[rank 0] Dataset: %lld points, %d dims\n",
                    (long long)N, d);
        std::printf("[rank 0] eps=%.3f  min_pts=%d  procs=%d\n",
                    a.eps, a.min_pts, size);
        std::fflush(stdout);
    }

    // ----------------------------------------------------------------
    // Run PDSDBSCAN-D
    // ----------------------------------------------------------------
    ParResult res = pdsdbscan_d(pts, N, d, a.eps, a.min_pts, MPI_COMM_WORLD);

    if (rank == 0) {
        int64_t n_clusters = 0, n_noise = 0;
        for (auto l : res.labels) {
            if (l >= 0) n_clusters = std::max(n_clusters, l + 1);
            else        ++n_noise;
        }

        std::printf("\n=== PARALLEL RESULTS (PDSDBSCAN-D) ===\n");
        std::printf("  clusters    : %lld\n",  (long long)n_clusters);
        std::printf("  noise pts   : %lld\n",  (long long)n_noise);
        std::printf("  wall time   : %.4f s\n", res.t_wall);
        std::printf("  max compute : %.4f s\n", res.t_compute);
        std::printf("  max comm    : %.4f s\n", res.t_comm);

        if (a.timing) {
            std::printf("\n%5s %12s %10s\n", "rank", "compute(s)", "comm(s)");
            for (auto& row : res.per_rank)
                std::printf("%5d %12.4f %10.4f\n",
                            (int)row[0], row[1], row[2]);
        }

        // -------- Correctness verification --------
        if (a.verify) {
            std::printf("\n[verify] Running sequential DSDBSCAN...\n");
            std::fflush(stdout);
            SeqResult sr = dbscan_sequential(pts.data(), (int)N, d,
                                             a.eps, a.min_pts);
            int64_t n_cl_seq = 0, n_ns_seq = 0;
            for (auto l : sr.labels) {
                if (l >= 0) n_cl_seq = std::max(n_cl_seq, l+1);
                else        ++n_ns_seq;
            }
            double ari = adjusted_rand_index(sr.labels, res.labels);
            std::printf("\n=== VERIFICATION ===\n");
            std::printf("  seq clusters: %lld\n",  (long long)n_cl_seq);
            std::printf("  seq noise   : %lld\n",  (long long)n_ns_seq);
            std::printf("  seq time    : %.4f s\n", sr.elapsed_sec);
            std::printf("  ARI score   : %.6f  (1.0 = perfect)\n", ari);
            std::printf("  speedup     : %.2fx\n",
                        sr.elapsed_sec / res.t_wall);
        }

        // -------- Save output --------
        if (!a.output.empty()) {
            write_bin(a.output, res.labels);
            std::printf("\nSaved labels to %s\n", a.output.c_str());
        }
    }

    MPI_Finalize();
    return 0;
}
