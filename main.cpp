#include "db_lsh.hpp"

#include <omp.h>

#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

void usage(const char* program) {
    std::cout
        << "DB-LSH2 - faster, memory-friendly DB-LSH (same c-ANN algorithm)\n\n"
        << "Usage:\n"
        << "  " << program
        << " <dataset> [c k L K beta R_min] [options]\n\n"
        << "Positional arguments (defaults in brackets):\n"
        << "  dataset   dataset name; loads <dataset>.data_new            [audio]\n"
        << "  c         approximation ratio, > 1                         [1.5]\n"
        << "  k         neighbours to return                             [50]\n"
        << "  L         number of projection tables                      [5]\n"
        << "  K         projection dimension per table                   [10]\n"
        << "  beta      candidate budget as fraction of N                 [0.1]\n"
        << "  R_min     initial search radius (auto for known datasets)\n\n"
        << "Options:\n"
        << "  --data PATH     explicit path to the LSHBOX .data_new file\n"
        << "  --queries N     number of queries, at most 200              [100]\n"
        << "  --seed N        RNG seed; 0 matches the original hashes     [0]\n"
        << "  --threads N     OpenMP threads, 0 = all available           [0]\n"
        << "  --leaf N        kd-tree leaf size                           [128]\n"
        << "  --out PATH      benchmark CSV path       [results/DB-LSH2_result.csv]\n"
        << "  --no-gt         skip exact ground truth (faster, no recall)\n"
        << "  --refresh-gt    recompute ground truth instead of using the cache\n"
        << "  --quiet         suppress configuration printout\n"
        << "  -h, --help      show this message\n\n"
        << "Example:\n"
        << "  " << program << " audio 1.5 50 5 10 0.1 0.3 --threads 4\n\n"
        << "The first up-to-200 vectors in the file are treated as queries; the\n"
        << "remaining vectors form the searchable database (as in the original).\n";
}

float default_radius(const std::string& name) {
    if (name == "audio") return 0.3f;
    if (name == "mnist") return 500.0f;
    if (name == "cifar") return 300.0f;
    if (name == "deep1m") return 0.2f;
    if (name == "NUS") return 6.5f;
    if (name == "Trevi") return 700.0f;
    if (name == "gist") return 0.3f;
    return 1.0f;
}

std::string resolve_data_path(const std::string& name) {
    namespace fs = std::filesystem;
    const std::vector<fs::path> candidates = {
        fs::path("data") / (name + ".data_new"),
        fs::path("..") / "dataset" / (name + ".data_new"),
        fs::path(name + ".data_new"),
    };
    for (const fs::path& candidate : candidates) {
        if (fs::exists(candidate)) return candidate.string();
    }
    return candidates.front().string();
}

template <typename T>
T parse_number(const std::string& value, const std::string& flag) {
    try {
        size_t consumed = 0;
        const T result = static_cast<T>(std::stold(value, &consumed));
        if (consumed != value.size()) throw std::invalid_argument("trailing characters");
        return result;
    } catch (const std::exception&) {
        throw std::runtime_error("invalid value for " + flag + ": " + value);
    }
}

}  // namespace

int main(int argc, char** argv) {
    using namespace dblsh2;
    try {
        Options options;
        std::vector<std::string> positional;

        for (int i = 1; i < argc; ++i) {
            const std::string arg = argv[i];
            if (arg == "-h" || arg == "--help") {
                usage(argv[0]);
                return 0;
            }
            auto value = [&](const std::string& flag) -> std::string {
                if (i + 1 >= argc) throw std::runtime_error("missing value for " + flag);
                return argv[++i];
            };
            if (arg == "--data") options.data_path = value(arg);
            else if (arg == "--queries") options.queries = parse_number<int>(value(arg), arg);
            else if (arg == "--seed") options.seed = parse_number<int>(value(arg), arg);
            else if (arg == "--threads") options.threads = parse_number<int>(value(arg), arg);
            else if (arg == "--leaf") options.leaf_size = parse_number<int>(value(arg), arg);
            else if (arg == "--out") options.output_csv = value(arg);
            else if (arg == "--rmin") options.r_min = parse_number<float>(value(arg), arg);
            else if (arg == "--no-gt") options.compute_gt = false;
            else if (arg == "--refresh-gt") options.refresh_gt = true;
            else if (arg == "--quiet") options.quiet = true;
            else if (!arg.empty() && arg[0] == '-') {
                throw std::runtime_error("unknown option: " + arg);
            } else {
                positional.push_back(arg);
            }
        }

        if (positional.empty()) {
            usage(argv[0]);
            return 1;
        }
        if (positional.size() > 7) {
            throw std::runtime_error("too many positional arguments (expected at most 7)");
        }

        options.dataset_name = positional[0];
        if (positional.size() >= 2) options.c = parse_number<float>(positional[1], "c");
        if (positional.size() >= 3) options.k = parse_number<int>(positional[2], "k");
        if (positional.size() >= 4) options.L = parse_number<int>(positional[3], "L");
        if (positional.size() >= 5) options.K = parse_number<int>(positional[4], "K");
        if (positional.size() >= 6) options.beta = parse_number<float>(positional[5], "beta");
        if (positional.size() >= 7) options.r_min = parse_number<float>(positional[6], "R_min");
        else options.r_min = default_radius(options.dataset_name);

        if (options.data_path.empty()) options.data_path = resolve_data_path(options.dataset_name);
        if (!(options.c > 1.0f)) throw std::runtime_error("c must be greater than 1");
        if (options.k < 1) throw std::runtime_error("k must be positive");
        if (options.L < 1 || options.K < 1) throw std::runtime_error("L and K must be positive");
        if (!(options.beta > 0.0f)) throw std::runtime_error("beta must be positive");
        if (!(options.r_min > 0.0f)) throw std::runtime_error("R_min must be positive");
        if (options.queries < 1 || options.queries > 200) {
            throw std::runtime_error("queries must be between 1 and 200");
        }
        if (options.threads < 0 || options.leaf_size < 1) {
            throw std::runtime_error("threads and leaf must be non-negative/positive");
        }

        if (!options.quiet) {
            std::cout << "DB-LSH2 | dataset=" << options.dataset_name
                      << " c=" << options.c << " k=" << options.k
                      << " L=" << options.L << " K=" << options.K
                      << " beta=" << options.beta << " R_min=" << options.r_min
                      << " seed=" << options.seed << '\n';
        }

        const Dataset dataset = load_dataset(options.data_path, options.queries);
        if (!options.quiet) {
            std::cout << "Loaded " << dataset.total << " vectors ("
                      << dataset.query_count << " reserved queries, evaluating "
                      << dataset.eval_count << ") + " << dataset.n
                      << " indexed points, dim=" << dataset.dim
                      << " from " << options.data_path << '\n';
        }

        const LSHIndex index(dataset, options);
        const BenchmarkReport report = run_benchmark(dataset, index, options);
        append_csv(options, dataset, report);
        print_report(options, report);
        std::cout << "CSV: " << options.output_csv << '\n';
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "error: " << error.what() << '\n';
        return 1;
    }
}
