#pragma once

#include <algorithm>
#include <cstdint>
#include <fstream>
#include <limits>
#include <string>
#include <vector>

namespace dblsh_im {

// ---------------------------------------------------------------------------
// Data
// ---------------------------------------------------------------------------

// LSHBOX binary layout:
//   int32 bytes_per_value (4), int32 vector_count, int32 dimension,
//   vector_count * dimension * float32 stored row-major.
//
// The first `query_count` vectors are used as queries; the rest form the
// searchable database. This is exactly the convention used by the original
// DB-LSH implementation.
struct Dataset {
    int total = 0;        // vectors in the file
    int query_count = 0;  // first min(200, total-1) vectors, reserved as queries
    int eval_count = 0;   // how many of those queries this run evaluates
    int n = 0;            // indexed vectors
    int dim = 0;
    std::vector<float> values;  // (query_count + n) * dim

    // Database id 0 is the vector after the reserved query block.
    const float* point(int id) const {
        return values.data() + static_cast<size_t>(query_count + id) * dim;
    }
    const float* query(int qi) const {
        return values.data() + static_cast<size_t>(qi) * dim;
    }
};

Dataset load_dataset(const std::string& path, int requested_queries);

// ---------------------------------------------------------------------------
// Options and results
// ---------------------------------------------------------------------------

struct Options {
    std::string dataset_name = "audio";
    std::string data_path;
    std::string output_csv = "results/DB-LSH-IM_result.csv";
    float c = 1.5f;         // c-ANN approximation ratio
    int k = 50;             // returned neighbours
    int L = 5;              // number of hash tables
    int K = 10;             // projection dimension per table
    float beta = 0.1f;      // candidate budget as a fraction of N
    float r_min = 0.3f;     // initial radius
    int queries = 100;      // number of queries (<= 200 in the source format)
    int seed = 0;           // 0 reproduces the original hash functions
    int threads = 0;        // 0 = OpenMP default
    int leaf_size = 128;     // kd-tree leaf size
    bool compute_gt = true;
    bool refresh_gt = false;  // ignore cached ground truth
    bool quiet = false;
};

struct Result {
    int id = -1;
    float dist = std::numeric_limits<float>::infinity();
};

struct QueryDetail {
    int visited = 0;       // unique candidates whose distance was computed
    int rounds = 0;        // widening rounds actually executed
    long long nodes = 0;   // kd-tree nodes examined
    long long hits = 0;    // projected points inside the window (before dedup)
    long long early = 0;   // distance computations cut short by the heap bound
};

struct BenchmarkReport {
    int queries = 0;
    int n = 0;
    int dim = 0;
    int k = 0;
    double index_build_ms = 0.0;
    double query_wall_ms = 0.0;
    double recall = 0.0;
    double ratio = 0.0;
    double cost = 0.0;
    double rounds = 0.0;
    double verified = 0.0;
    double nodes = 0.0;
    double hits = 0.0;
    double early = 0.0;
    int threads = 1;
    bool has_ground_truth = false;
    bool ground_truth_cached = false;
};

// ---------------------------------------------------------------------------
// Improved DB-LSH index
// ---------------------------------------------------------------------------

class LSHIndex {
public:
    LSHIndex(const Dataset& dataset, const Options& options);

    // Evaluation API. `scratch` is per-thread and reused across queries.
    void query(const float* query_vector, int k, float beta, float c,
               float r_min, struct QueryScratch& scratch,
               std::vector<Result>& out, QueryDetail& detail) const;

    int tables() const { return L_; }
    int hash_dim() const { return K_; }
    int hash_functions() const { return S_; }
    int size() const { return N_; }
    long long total_nodes() const;
    double build_milliseconds() const { return build_ms_; }

private:
    struct Table {
        std::vector<float> proj;   // N*K, rows in leaf order
        std::vector<int> order;    // database ids in leaf order
        std::vector<int> left;     // -1 marks a leaf
        std::vector<int> right;
        std::vector<int> begin;
        std::vector<int> count;
        std::vector<float> bbox;   // 2*K per node: all min, then all max
        int root = 0;
    };

    void generate_coefficients();
    void compute_table_projections(int table_id, std::vector<float>& out) const;
    Table build_table(int table_id, const std::vector<float>& base_proj) const;
    bool window_query(const Table& table, const float* query_hash,
                      float half_width, const float* query_vector,
                      struct QueryScratch& scratch, int k, int budget,
                      int& visited, QueryDetail& detail) const;

    const Dataset& ds_;
    Options opt_;
    int N_, dim_, L_, K_, S_;
    std::vector<float> coeff_;  // S*dim Gaussian projections
    std::vector<Table> tables_;
    double build_ms_ = 0.0;
};

// Per-thread reusable buffers. Avoids thousands of heap allocations per query.
struct QueryScratch {
    std::vector<uint32_t> seen;
    uint32_t generation = 0;
    struct Candidate {
        int id = -1;
        float d2 = std::numeric_limits<float>::infinity();
        // Default comparison for std::*_heap: worst (largest) distance on top.
        bool operator<(const Candidate& other) const { return d2 < other.d2; }
    };
    std::vector<Candidate> heap;  // max-heap, worst candidate at front
    std::vector<int> stack;
    std::vector<float> query_hash;
    std::vector<float> lower;
    std::vector<float> upper;

    QueryScratch(int n, int s, int k, int kdim)
        : seen(n, 0), query_hash(s), lower(kdim), upper(kdim) {
        heap.reserve(k + 1);
        stack.reserve(128);
    }
};

// Exact ground truth (top-k by Euclidean distance), one query at a time.
void exact_topk(const Dataset& dataset, int query_index, int k,
                std::vector<int>& ids, std::vector<float>& distances);

BenchmarkReport run_benchmark(const Dataset& dataset, const LSHIndex& index,
                              const Options& options);

// Append one row to the benchmark CSV, writing the header when needed.
void append_csv(const Options& options, const Dataset& dataset,
                const BenchmarkReport& report);

void print_report(const Options& options, const BenchmarkReport& report);

}  // namespace dblsh_im
