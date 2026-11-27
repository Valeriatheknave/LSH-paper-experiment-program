#include "db_lsh.hpp"

#include <omp.h>

#include <chrono>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <functional>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <random>
#include <stdexcept>

namespace dblsh2 {
namespace {

constexpr float kInf = std::numeric_limits<float>::infinity();

std::filesystem::path gt_cache_path(const Options& options) {
    const std::filesystem::path csv(options.output_csv);
    const std::string name = options.dataset_name + ".gt";
    return csv.parent_path().empty() ? std::filesystem::path(name)
                                     : csv.parent_path() / name;
}

uint64_t dataset_fingerprint(const Options& options, const Dataset& dataset) {
    namespace fs = std::filesystem;
    std::error_code ec;
    const uint64_t bytes = fs::file_size(options.data_path, ec);
    const auto stamp = fs::last_write_time(options.data_path, ec);
    return bytes ^ static_cast<uint64_t>(stamp.time_since_epoch().count()) ^
           (static_cast<uint64_t>(dataset.total) << 32) ^
           static_cast<uint64_t>(dataset.dim);
}

bool load_ground_truth(const Options& options, const Dataset& dataset, int k,
                       std::vector<int>& ids, std::vector<float>& distances) {
    if (options.refresh_gt) return false;
    const auto path = gt_cache_path(options);
    std::ifstream in(path, std::ios::binary);
    if (!in) return false;
    char magic[12] = {0};
    uint64_t fingerprint = 0;
    int32_t q = 0, stored_k = 0, n = 0, dim = 0;
    in.read(magic, sizeof(magic));
    in.read(reinterpret_cast<char*>(&fingerprint), sizeof(fingerprint));
    in.read(reinterpret_cast<char*>(&q), sizeof(q));
    in.read(reinterpret_cast<char*>(&stored_k), sizeof(stored_k));
    in.read(reinterpret_cast<char*>(&n), sizeof(n));
    in.read(reinterpret_cast<char*>(&dim), sizeof(dim));
    if (std::strncmp(magic, "DBLSH2GT001", sizeof(magic)) != 0 ||
        fingerprint != dataset_fingerprint(options, dataset) ||
        q != dataset.eval_count || stored_k != k || n != dataset.n ||
        dim != dataset.dim) {
        return false;
    }
    ids.resize(static_cast<size_t>(q) * k);
    distances.resize(static_cast<size_t>(q) * k);
    in.read(reinterpret_cast<char*>(ids.data()), ids.size() * sizeof(int32_t));
    in.read(reinterpret_cast<char*>(distances.data()), distances.size() * sizeof(float));
    return static_cast<bool>(in);
}

void save_ground_truth(const Options& options, const Dataset& dataset, int k,
                       const std::vector<int>& ids,
                       const std::vector<float>& distances) {
    const auto path = gt_cache_path(options);
    if (!path.parent_path().empty()) {
        std::filesystem::create_directories(path.parent_path());
    }
    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    if (!out) return;  // caching is an optimization, never a failure
    const uint64_t fingerprint = dataset_fingerprint(options, dataset);
    const int32_t q = dataset.eval_count, stored_k = k;
    const int32_t n = dataset.n, dim = dataset.dim;
    out.write("DBLSH2GT001", 12);
    out.write(reinterpret_cast<const char*>(&fingerprint), sizeof(fingerprint));
    out.write(reinterpret_cast<const char*>(&q), sizeof(q));
    out.write(reinterpret_cast<const char*>(&stored_k), sizeof(stored_k));
    out.write(reinterpret_cast<const char*>(&n), sizeof(n));
    out.write(reinterpret_cast<const char*>(&dim), sizeof(dim));
    out.write(reinterpret_cast<const char*>(ids.data()), ids.size() * sizeof(int32_t));
    out.write(reinterpret_cast<const char*>(distances.data()),
              distances.size() * sizeof(float));
}

inline float squared_distance(const float* a, const float* b, int dim) {
    float sum = 0.0f;
    for (int d = 0; d < dim; ++d) {
        const float delta = a[d] - b[d];
        sum += delta * delta;
    }
    return sum;
}

inline bool box_overlaps_window(const float* box, const float* low,
                                const float* high, int kdim) {
    for (int d = 0; d < kdim; ++d) {
        if (box[d] > high[d] || box[kdim + d] < low[d]) return false;
    }
    return true;
}

// Lower bound on the squared distance from the query's projected point to a
// node's bounding box. Used only to visit the more promising child first.
inline float box_distance(const float* box, const float* point, int kdim) {
    float sum = 0.0f;
    for (int d = 0; d < kdim; ++d) {
        float delta = 0.0f;
        if (point[d] < box[d]) delta = box[d] - point[d];
        else if (point[d] > box[kdim + d]) delta = point[d] - box[kdim + d];
        sum += delta * delta;
    }
    return sum;
}

struct ThreadStats {
    long long visited = 0;
    long long rounds = 0;
    long long exact_hits = 0;
    long long returned = 0;
    long long nodes = 0;
    long long window_hits = 0;
    long long early = 0;
    double ratio_sum = 0.0;
};

}  // namespace

// ---------------------------------------------------------------------------
// Dataset loader
// ---------------------------------------------------------------------------

Dataset load_dataset(const std::string& path, int requested_queries) {
    std::ifstream in(path, std::ios::binary | std::ios::ate);
    if (!in) throw std::runtime_error("cannot open data file: " + path);

    const std::streamoff file_bytes = in.tellg();
    in.seekg(0, std::ios::beg);

    int32_t header[3] = {0, 0, 0};
    in.read(reinterpret_cast<char*>(header), sizeof(header));
    if (header[0] != 4) throw std::runtime_error("expected float32 data");
    if (header[1] <= 1 || header[2] <= 0) throw std::runtime_error("invalid dataset header");

    const size_t total = static_cast<size_t>(header[1]);
    const size_t dim = static_cast<size_t>(header[2]);
    const size_t expected = 12 + total * dim * sizeof(float);
    if (static_cast<size_t>(file_bytes) < expected) {
        throw std::runtime_error("truncated data file: expected " +
                                 std::to_string(expected) + " bytes, got " +
                                 std::to_string(file_bytes));
    }

    Dataset ds;
    ds.total = static_cast<int>(total);
    ds.dim = static_cast<int>(dim);
    ds.query_count = std::min(200, ds.total - 1);
    ds.eval_count = std::max(1, std::min(requested_queries, ds.query_count));
    ds.n = ds.total - ds.query_count;
    ds.values.resize(total * dim);
    in.read(reinterpret_cast<char*>(ds.values.data()),
            static_cast<std::streamsize>(total * dim * sizeof(float)));
    if (!in) throw std::runtime_error("failed while reading vectors");
    return ds;
}

// ---------------------------------------------------------------------------
// Index construction
// ---------------------------------------------------------------------------

LSHIndex::LSHIndex(const Dataset& dataset, const Options& options)
    : ds_(dataset), opt_(options),
      N_(dataset.n), dim_(dataset.dim),
      L_(options.L), K_(options.K), S_(options.L * options.K) {
    if (L_ <= 0 || K_ <= 0 || N_ < 2) throw std::runtime_error("invalid index parameters");

    const auto start = std::chrono::steady_clock::now();
    generate_coefficients();

    tables_.resize(L_);
    #pragma omp parallel for schedule(static)
    for (int t = 0; t < L_; ++t) {
        // One contiguous N*K block per table: the query vector is read L
        // times instead of L*K times, and kd-tree splits stay cache friendly.
        std::vector<float> base(static_cast<size_t>(N_) * K_);
        compute_table_projections(t, base);
        tables_[t] = build_table(t, base);
    }

    build_ms_ = std::chrono::duration<double, std::milli>(
                    std::chrono::steady_clock::now() - start).count();
}

// Uses the same random stream as the reference implementation:
// mt19937(seed) + N(0,1) per coefficient, with one unused draw per hash
// function (the original's rndAs2). Seed 0 therefore reproduces its hashes.
void LSHIndex::generate_coefficients() {
    coeff_.resize(static_cast<size_t>(S_) * dim_);
    std::mt19937 rng(static_cast<unsigned>(opt_.seed));
    std::normal_distribution<float> normal;
    for (int s = 0; s < S_; ++s) {
        float* a = coeff_.data() + static_cast<size_t>(s) * dim_;
        for (int d = 0; d < dim_; ++d) a[d] = normal(rng);
        static_cast<void>(normal(rng));  // keep the reference RNG alignment
    }
}

void LSHIndex::compute_table_projections(int table_id, std::vector<float>& out) const {
    const float* table_coeff = coeff_.data() + static_cast<size_t>(table_id) * K_ * dim_;
    for (int id = 0; id < N_; ++id) {
        const float* x = ds_.point(id);
        float* row = out.data() + static_cast<size_t>(id) * K_;
        for (int d = 0; d < K_; ++d) {
            const float* a = table_coeff + static_cast<size_t>(d) * dim_;
            float sum = 0.0f;
            for (int j = 0; j < dim_; ++j) sum += x[j] * a[j];
            row[d] = sum;
        }
    }
}

LSHIndex::Table LSHIndex::build_table(int table_id,
                                      const std::vector<float>& base_proj) const {
    static_cast<void>(table_id);
    Table table;
    const int leaf = std::max(1, opt_.leaf_size);

    std::vector<int> ids(N_);
    std::iota(ids.begin(), ids.end(), 0);

    const size_t max_nodes = 2 * (static_cast<size_t>(N_ / leaf) + 2) + 8;
    table.left.reserve(max_nodes);
    table.right.reserve(max_nodes);
    table.begin.reserve(max_nodes);
    table.count.reserve(max_nodes);
    table.bbox.reserve(max_nodes * 2 * K_);

    auto new_node = [&]() -> int {
        const int node = static_cast<int>(table.left.size());
        table.left.push_back(-1);
        table.right.push_back(-1);
        table.begin.push_back(0);
        table.count.push_back(0);
        table.bbox.resize(static_cast<size_t>(node + 1) * 2 * K_, 0.0f);
        return node;
    };

    std::function<int(int, int)> build = [&](int b, int e) -> int {
        const int node = new_node();
        float* box = table.bbox.data() + static_cast<size_t>(node) * 2 * K_;
        for (int d = 0; d < K_; ++d) {
            box[d] = kInf;
            box[K_ + d] = -kInf;
        }
        for (int pos = b; pos < e; ++pos) {
            const int id = ids[pos];
            for (int d = 0; d < K_; ++d) {
                const float v = base_proj[static_cast<size_t>(id) * K_ + d];
                box[d] = std::min(box[d], v);
                box[K_ + d] = std::max(box[K_ + d], v);
            }
        }

        if (e - b <= leaf) {
            table.begin[node] = b;
            table.count[node] = e - b;
            return node;
        }

        int split_dim = 0;
        float best_span = -1.0f;
        for (int d = 0; d < K_; ++d) {
            const float span = box[K_ + d] - box[d];
            if (span > best_span) {
                best_span = span;
                split_dim = d;
            }
        }

        const int mid = b + (e - b) / 2;
        std::nth_element(ids.begin() + b, ids.begin() + mid, ids.begin() + e,
                         [&](int lhs, int rhs) {
                             const float vl = base_proj[static_cast<size_t>(lhs) * K_ + split_dim];
                             const float vr = base_proj[static_cast<size_t>(rhs) * K_ + split_dim];
                             return vl != vr ? vl < vr : lhs < rhs;
                         });

        const int left = build(b, mid);
        const int right = build(mid, e);
        table.left[node] = left;
        table.right[node] = right;
        return node;
    };

    table.root = build(0, N_);
    table.order = std::move(ids);

    // Transpose into leaf-ordered rows: projections of one point are adjacent,
    // which makes the window filter a single linear scan.
    table.proj.resize(static_cast<size_t>(N_) * K_);
    for (int row = 0; row < N_; ++row) {
        const int id = table.order[row];
        float* dst = table.proj.data() + static_cast<size_t>(row) * K_;
        for (int d = 0; d < K_; ++d) {
            dst[d] = base_proj[static_cast<size_t>(id) * K_ + d];
        }
    }
    return table;
}

long long LSHIndex::total_nodes() const {
    long long sum = 0;
    for (const Table& table : tables_) sum += static_cast<long long>(table.left.size());
    return sum;
}

// ---------------------------------------------------------------------------
// Query execution
// ---------------------------------------------------------------------------

bool LSHIndex::window_query(const Table& table, const float* query_hash,
                            float half_width, const float* query_vector,
                            QueryScratch& scratch, int k, int budget,
                            int& visited, QueryDetail& detail) const {
    for (int d = 0; d < K_; ++d) {
        scratch.lower[d] = query_hash[d] - half_width;
        scratch.upper[d] = query_hash[d] + half_width;
    }

    scratch.stack.clear();
    scratch.stack.push_back(table.root);

    while (!scratch.stack.empty()) {
        const int node = scratch.stack.back();
        scratch.stack.pop_back();
        ++detail.nodes;

        const float* box = table.bbox.data() + static_cast<size_t>(node) * 2 * K_;
        if (!box_overlaps_window(box, scratch.lower.data(), scratch.upper.data(), K_)) {
            continue;
        }

        if (table.left[node] < 0) {
            const int begin = table.begin[node];
            const int end = begin + table.count[node];
            for (int row = begin; row < end; ++row) {
                const float* projected = table.proj.data() + static_cast<size_t>(row) * K_;
                bool inside = true;
                for (int d = 0; d < K_; ++d) {
                    if (projected[d] < scratch.lower[d] || projected[d] > scratch.upper[d]) {
                        inside = false;
                        break;
                    }
                }
                if (!inside) continue;
                ++detail.hits;

                const int id = table.order[row];
                if (scratch.seen[id] == scratch.generation) continue;
                scratch.seen[id] = scratch.generation;
                ++visited;

                const bool full = scratch.heap.size() >= static_cast<size_t>(k);
                const float worst = full ? scratch.heap.front().d2 : kInf;
                const float* x = ds_.point(id);
                float sum = 0.0f;
                for (int d = 0; d < dim_; ++d) {
                    const float delta = x[d] - query_vector[d];
                    sum += delta * delta;
                    if (full && sum >= worst) {
                        ++detail.early;
                        break;  // cannot improve the heap
                    }
                }

                QueryScratch::Candidate candidate{id, sum};
                if (!full) {
                    scratch.heap.push_back(candidate);
                    std::push_heap(scratch.heap.begin(), scratch.heap.end());
                } else if (sum < worst) {
                    std::pop_heap(scratch.heap.begin(), scratch.heap.end());
                    scratch.heap.back() = candidate;
                    std::push_heap(scratch.heap.begin(), scratch.heap.end());
                }

                if (visited >= budget) return true;
            }
        } else {
            const int left = table.left[node];
            const int right = table.right[node];
            const float dl = box_distance(table.bbox.data() + static_cast<size_t>(left) * 2 * K_,
                                          query_hash, K_);
            const float dr = box_distance(table.bbox.data() + static_cast<size_t>(right) * 2 * K_,
                                          query_hash, K_);
            // LIFO stack: push the farther child first so the nearer one is
            // explored first when the candidate budget is reached early.
            if (dl <= dr) {
                scratch.stack.push_back(right);
                scratch.stack.push_back(left);
            } else {
                scratch.stack.push_back(left);
                scratch.stack.push_back(right);
            }
        }
    }
    return false;
}

void LSHIndex::query(const float* query_vector, int k, float beta, float c,
                     float r_min, QueryScratch& scratch, std::vector<Result>& out,
                     QueryDetail& detail) const {
    ++scratch.generation;
    if (scratch.generation == 0) {  // stamp wrap-around, practically unreachable
        std::fill(scratch.seen.begin(), scratch.seen.end(), 0u);
        scratch.generation = 1;
    }
    scratch.heap.clear();

    for (int s = 0; s < S_; ++s) {
        const float* a = coeff_.data() + static_cast<size_t>(s) * dim_;
        float sum = 0.0f;
        for (int d = 0; d < dim_; ++d) sum += query_vector[d] * a[d];
        scratch.query_hash[s] = sum;
    }

    const int budget = std::max(k, static_cast<int>(std::ceil(beta * N_)) + k);
    float half_width = 2.0f * c * c * r_min;  // original: init_w = 4*c*c*R_min
    int visited = 0;
    int rounds = 0;

    while (visited < budget && rounds < 30) {
        ++rounds;
        bool exhausted = false;
        for (int t = 0; t < L_; ++t) {
            if (window_query(tables_[t], scratch.query_hash.data() + t * K_,
                             half_width, query_vector, scratch, k, budget,
                             visited, detail)) {
                exhausted = true;
                break;
            }
        }
        if (exhausted) break;
        half_width *= c;
    }

    std::sort(scratch.heap.begin(), scratch.heap.end(),
              [](const QueryScratch::Candidate& a, const QueryScratch::Candidate& b) {
                  return a.d2 < b.d2;
              });
    out.resize(scratch.heap.size());
    for (size_t i = 0; i < scratch.heap.size(); ++i) {
        out[i].id = scratch.heap[i].id;
        out[i].dist = std::sqrt(std::max(0.0f, scratch.heap[i].d2));
    }

    detail.visited = visited;
    detail.rounds = rounds;
}

// ---------------------------------------------------------------------------
// Benchmarking
// ---------------------------------------------------------------------------

void exact_topk(const Dataset& dataset, int query_index, int k,
                std::vector<int>& ids, std::vector<float>& distances) {
    const float* q = dataset.query(query_index);
    std::vector<QueryScratch::Candidate> heap;
    heap.reserve(k + 1);
    for (int id = 0; id < dataset.n; ++id) {
        const float d2 = squared_distance(dataset.point(id), q, dataset.dim);
        QueryScratch::Candidate candidate{id, d2};
        if (heap.size() < static_cast<size_t>(k)) {
            heap.push_back(candidate);
            std::push_heap(heap.begin(), heap.end());
        } else if (d2 < heap.front().d2) {
            std::pop_heap(heap.begin(), heap.end());
            heap.back() = candidate;
            std::push_heap(heap.begin(), heap.end());
        }
    }
    std::sort(heap.begin(), heap.end(),
              [](const QueryScratch::Candidate& a, const QueryScratch::Candidate& b) {
                  return a.d2 < b.d2;
              });
    ids.resize(heap.size());
    distances.resize(heap.size());
    for (size_t i = 0; i < heap.size(); ++i) {
        ids[i] = heap[i].id;
        distances[i] = std::sqrt(std::max(0.0f, heap[i].d2));
    }
}

BenchmarkReport run_benchmark(const Dataset& dataset, const LSHIndex& index,
                              const Options& options) {
    const int q = dataset.eval_count;
    const int k = std::min(options.k, dataset.n);
    const int threads = options.threads > 0
                            ? std::min(options.threads, omp_get_max_threads())
                            : omp_get_max_threads();

    BenchmarkReport report;
    report.queries = q;
    report.n = dataset.n;
    report.dim = dataset.dim;
    report.k = k;
    report.threads = threads;
    report.has_ground_truth = options.compute_gt;
    report.index_build_ms = index.build_milliseconds();

    std::vector<int> gt_ids;
    std::vector<float> gt_dist;
    if (options.compute_gt) {
        gt_ids.resize(static_cast<size_t>(q) * k);
        gt_dist.resize(static_cast<size_t>(q) * k);
        report.ground_truth_cached = load_ground_truth(options, dataset, k,
                                                        gt_ids, gt_dist);
        if (!report.ground_truth_cached) {
            #pragma omp parallel for schedule(dynamic, 2) num_threads(threads)
            for (int qi = 0; qi < q; ++qi) {
                std::vector<int> ids;
                std::vector<float> dists;
                exact_topk(dataset, qi, k, ids, dists);
                for (int j = 0; j < k; ++j) {
                    gt_ids[static_cast<size_t>(qi) * k + j] = ids[j];
                    gt_dist[static_cast<size_t>(qi) * k + j] = dists[j];
                }
            }
            save_ground_truth(options, dataset, k, gt_ids, gt_dist);
        }
    }

    std::vector<ThreadStats> stats(threads);
    std::vector<std::vector<Result>> all_results(q);

    const double start = omp_get_wtime();

    #pragma omp parallel num_threads(threads)
    {
        const int tid = omp_get_thread_num();
        QueryScratch scratch(dataset.n, index.hash_functions(), k, index.hash_dim());
        std::vector<Result> results;
        QueryDetail detail;

        #pragma omp for schedule(dynamic, 2)
        for (int qi = 0; qi < q; ++qi) {
            index.query(dataset.query(qi), k, options.beta, options.c,
                        options.r_min, scratch, results, detail);

            all_results[qi] = results;

            stats[tid].visited += detail.visited;
            stats[tid].rounds += detail.rounds;
            stats[tid].returned += static_cast<long long>(results.size());
            stats[tid].nodes += detail.nodes;
            stats[tid].window_hits += detail.hits;
            stats[tid].early += detail.early;
        }
    }

    report.query_wall_ms = (omp_get_wtime() - start) * 1000.0;

    // Ground-truth evaluation is deliberately outside the timed query interval.
    if (options.compute_gt) {
        for (int qi = 0; qi < q; ++qi) {
            const auto gt_begin = gt_ids.begin() + static_cast<size_t>(qi) * k;
            const auto gt_end = gt_begin + k;

            for (size_t j = 0; j < all_results[qi].size(); ++j) {
                const Result& result = all_results[qi][j];

                if (std::find(gt_begin, gt_end, result.id) != gt_end) {
                    ++stats[0].exact_hits;
                }

                const size_t pos = static_cast<size_t>(qi) * k + j;
                const float reference = gt_dist[pos];
                const float ratio = reference > 0.0f
                                        ? result.dist / reference
                                        : 1.0f;
                stats[0].ratio_sum += ratio;
            }
        }
    }

    long long visited = 0, rounds = 0, returned = 0, hits = 0;
    long long nodes = 0, window_hits = 0, early = 0;
    double ratio_sum = 0.0;
    for (const ThreadStats& s : stats) {
        visited += s.visited;
        rounds += s.rounds;
        returned += s.returned;
        hits += s.exact_hits;
        ratio_sum += s.ratio_sum;
        nodes += s.nodes;
        window_hits += s.window_hits;
        early += s.early;
    }

    report.nodes = static_cast<double>(nodes) / q;
    report.hits = static_cast<double>(window_hits) / q;
    report.early = static_cast<double>(early) / q;
    report.cost = static_cast<double>(visited) /
                  std::max<long long>(1, static_cast<long long>(q) * dataset.n);
    report.rounds = static_cast<double>(rounds) / q;
    report.verified = static_cast<double>(visited) / q;

    if (options.compute_gt) {
        report.recall = static_cast<double>(hits) /
                        std::max<long long>(1, static_cast<long long>(q) * k);
        report.ratio = ratio_sum / std::max<long long>(1, returned);
    }

    return report;
}

void append_csv(const Options& options, const Dataset& dataset,
                const BenchmarkReport& report) {
    static_cast<void>(dataset);
    const std::filesystem::path path(options.output_csv);
    if (path.parent_path().empty() == false) {
        std::filesystem::create_directories(path.parent_path());
    }
    const bool needs_header = !std::filesystem::exists(path) ||
                              std::filesystem::file_size(path) == 0;
    std::ofstream out(path, std::ios::app);
    if (!out) throw std::runtime_error("cannot write CSV: " + options.output_csv);
    if (needs_header) {
        out << "Dataset,c,k,L,K,beta,R_min,seed,threads,Q,Recall,Ratio,Cost,Rounds,"
               "Verified,Nodes,Hits,EarlyExits,IndexBuild_ms,Query_ms\n";
    }
    out << options.dataset_name << ',' << options.c << ',' << options.k << ','
        << options.L << ',' << options.K << ',' << options.beta << ','
        << options.r_min << ',' << options.seed << ',' << report.threads << ','
        << report.queries << ',' << std::fixed << std::setprecision(6)
        << report.recall << ','
        << report.ratio << ',' << report.cost << ',' << std::setprecision(2)
        << report.rounds << ',' << std::setprecision(0) << report.verified
        << ',' << report.nodes << ',' << report.hits << ',' << report.early
        << ',' << std::setprecision(2) << report.index_build_ms
        << ',' << report.query_wall_ms << '\n';
}

void print_report(const Options& options, const BenchmarkReport& report) {
    std::cout << "\nRESULTS (" << options.dataset_name << ", " << report.queries
              << " queries, " << report.threads << " threads)\n";
    std::cout << "  recall@k        : " << std::fixed << std::setprecision(4)
              << report.recall << '\n';
    std::cout << "  distance ratio  : " << std::setprecision(5) << report.ratio << '\n';
    std::cout << "  cost (fraction) : " << std::setprecision(5) << report.cost << '\n';
    std::cout << "  avg rounds      : " << std::setprecision(2) << report.rounds << '\n';
    std::cout << "  avg verified    : " << std::setprecision(0) << report.verified << '\n';
    std::cout << "  avg nodes/hits  : " << std::setprecision(0) << report.nodes
              << " / " << report.hits << " (early exits " << report.early << ")\n";
    std::cout << "  index build     : " << std::setprecision(1)
              << report.index_build_ms << " ms\n";
    std::cout << "  query wall time : " << std::setprecision(2)
              << report.query_wall_ms << " ms total, "
              << report.query_wall_ms / report.queries << " ms/query\n";
    if (report.has_ground_truth) {
        std::cout << "  ground truth       : "
                  << (report.ground_truth_cached ? "loaded from cache" : "computed")
                  << '\n';
    } else {
        std::cout << "  (ground truth skipped: recall and ratio are 0)\n";
    }
}

}  // namespace dblsh2
