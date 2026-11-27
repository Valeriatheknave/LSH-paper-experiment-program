# DB-LSH-IM — an in-memory reimplementation of DB-LSH

**DB-LSH-IM** (in-memory DB-LSH) is a C++17 reimplementation of **DB-LSH: Locality-Sensitive Hashing with Query-based Dynamic Bucketing** (Tian, Zhao, and Zhou; ICDE 2022 [41]).

It accompanies the paper:

> Fatemeh Mohammadkia and Faezeh Din-Mohammadi, *"Locality-Sensitive Hashing for Approximate Nearest Neighbor Search: A Controlled Comparison of DB-LSH and an In-Memory Reimplementation on Indexing and Query Performance,"* Department of Computer Science, Islamic Azad University, Karaj Branch, Karaj, Iran.

DB-LSH-IM follows the original hashing and query-based dynamic-bucketing approach while replacing the disk-backed R\*-tree index with bulk-loaded in-memory kd-trees and introducing other implementation optimizations.

> **Scope:** This is an engineering reimplementation, **not** a new LSH algorithm, and it is **not** the official *DB-LSH 2.0* extension (Tian, Zhao, and Zhou; TKDE 2024 [42]). Differences in data structures and traversal order can affect the candidates examined and the results returned. The measurements below characterize the reimplementation as a whole; they do not isolate individual optimizations or establish algorithm-level speedups.

## Approach

DB-LSH-IM preserves the Gaussian random projections, the query-window initialization and widening rule, and the candidate budget of the reference implementation, while replacing one disk-backed R\*-tree per table with one bulk-loaded in-memory kd-tree per table.

| Component | Original DB-LSH | DB-LSH-IM |
|---|---|---|
| Hash functions | Gaussian random projections | Same approach; seed 0 follows the reference random-number sequence in the tested environment |
| Index | One K-dimensional R\*-tree per table | One bulk-loaded kd-tree per table, with bounding-box pruning |
| Query window | Initialized using `4 · c² · R_min` and widened by `c` | Same window initialization and widening approach |
| Search budget | `β · N + k` candidates, with at most 30 rounds | Same budget and round limit |
| Candidate de-duplication | Per-query flag array | Generation-stamped array |
| Candidate ranking | Euclidean distance | Euclidean distance with early termination |
| Queries | First 200 vectors reserved as queries | Same file convention |

The implementations follow the same general approach, but the use of different data structures and traversal orders does not guarantee identical candidate sets or results.

## Implementation changes

- **In-memory kd-trees:** replace disk-backed R\*-trees with bulk-loaded kd-trees and bounding-box pruning.
- **Contiguous projections:** store each table's projected coordinates in an `N × K` block.
- **OpenMP parallelism:** parallelize independent work in projection, index construction, exact ground-truth calculation, and query processing (configured to one thread for the reported measurements).
- **Generation-stamped de-duplication:** avoid resetting the entire flag array for every query.
- **Bounded top-k heap:** maintain the best `k` candidates without sorting a larger result array.
- **Early distance termination:** stop calculating a candidate's distance when its partial squared distance cannot improve the current top-k.
- **Traversal ordering:** visit the more promising child first in projected space.
- **Ground-truth cache:** reuse exact nearest-neighbor results across runs when the cache is valid.
- **Wall-clock timing:** use `omp_get_wtime()` for DB-LSH-IM measurements; both implementations were measured using wall-clock time.

The in-memory index trades disk usage for RAM. Datasets that can be processed using a disk-backed index may not fit in memory with this implementation.

## Datasets

Audio was obtained from the original DB-LSH repository. MNIST training images were obtained from the MNIST database, and SIFT vectors from the TexMex SIFT1M dataset. MNIST and SIFT were converted to the required binary input format using the supplied scripts — no normalization, shuffling, or random sampling was applied.

| Dataset | Indexed vectors | Reserved queries | Total vectors | Dimensions |
|---|---:|---:|---:|---:|
| Audio | 54,187 | 200 | 54,387 | 192 |
| MNIST | 59,800 | 200 | 60,000 | 784 |
| SIFT 100k | 100,000 | 200 | 100,200 | 128 |

For Audio and MNIST, the first 200 vectors in each input file are reserved as queries and excluded from the index. For SIFT 100k, the first 200 vectors from `sift_query.fvecs` are placed before the first 100,000 vectors from `sift_base.fvecs`. The first 100 of the 200 reserved queries are evaluated; all 200 remain excluded from the index.

## Experimental configuration

The experiment was conducted on an Acer TravelMate P214-51 laptop equipped with an Intel Core i5-8250U processor and running Ubuntu 24.04.5 LTS. Both implementations were evaluated using a single thread. Each implementation–dataset configuration was repeated 20 times, resulting in 120 total benchmark runs.

| Parameter | Description | Value |
|---|---|---|
| `c` | Window expansion factor | 1.5 |
| `k` | Requested neighbors | 50 |
| `L` | Hash tables | 5 |
| `K` | Projections per table | 10 |
| `β` | Candidate-budget fraction | 0.1 |
| `Q` | Evaluated queries | 100 |
| Seed | DB-LSH-IM projection seed | 0 |
| `R_min` | Initial-radius parameter | Audio 0.3; MNIST 500; SIFT 1.0 |
| `leaf_size` | kd-tree leaf size | 128 |

Parameters were fixed rather than swept; results characterize the selected configurations rather than optimized performance across all configurations.

## Results

### Table 1 — Indexing and query performance

Values are mean ± standard deviation over 20 repetitions. Indexing time is in seconds; query time is in milliseconds per query.

| Dataset | Implementation | Indexing time (s) | Query time (ms/query) |
|---|---|---:|---:|
| Audio | Original DB-LSH | 0.24752 ± 0.01038 | 3.48453 ± 0.12277 |
| Audio | DB-LSH-IM | **0.05605 ± 0.00734** | **2.59550 ± 0.12298** |
| MNIST | Original DB-LSH | 0.61995 ± 0.01417 | 6.89610 ± 0.12144 |
| MNIST | DB-LSH-IM | **0.15807 ± 0.01159** | **6.13086 ± 0.21808** |
| SIFT 100k | Original DB-LSH | 0.45899 ± 0.01567 | 9.84344 ± 0.23579 |
| SIFT 100k | DB-LSH-IM | **0.10742 ± 0.00517** | **5.70514 ± 0.18600** |

Bold indicates the better measured value. Lower times are desirable.

### Table 2 — Search quality and search-effort measures

Values are means across 20 runs. These metrics were invariant across repetitions in the recorded experiment.

| Dataset | Implementation | Recall | Ratio | Cost | Rounds |
|---|---|---:|---:|---:|---:|
| Audio | Original DB-LSH | 0.9106 | 1.003950 | 0.100910 | 1.66 |
| Audio | DB-LSH-IM | **0.9186** | **1.003777** | 0.100928 | 1.66 |
| MNIST | Original DB-LSH | 0.9250 | 1.003920 | 0.100836 | 1.74 |
| MNIST | DB-LSH-IM | **0.9316** | **1.003690** | 0.100836 | 1.74 |
| SIFT 100k | Original DB-LSH | 0.9078 | 1.003650 | 0.100500 | 12.98 |
| SIFT 100k | DB-LSH-IM | **0.9138** | **1.003414** | 0.100500 | 12.98 |

### Table 3 — Relative timing improvements

Calculated using the mean timing of Original DB-LSH as the baseline.

| Dataset | Indexing reduction | Query-time reduction |
|---|---:|---:|
| Audio | 77.36% | 25.51% |
| MNIST | 74.50% | 11.10% |
| SIFT 100k | 76.60% | 42.04% |

Reduction (%) = (T<sub>Original</sub> − T<sub>IM</sub>) / T<sub>Original</sub> × 100. These percentages are descriptive improvements in the observed mean timings.

### Table 4 — Paired statistical comparison

The statistical analysis compares the corresponding 20 repetitions for each dataset and metric.

| Dataset | Metric | Mean paired difference* | 95% bootstrap CI | Adjusted p |
|---|---|---:|---|---|
| Audio | Indexing | 0.191473 s | [0.186243, 0.196731] | 0.00001144 |
| Audio | Query | 0.889031 ms | [0.812288, 0.960477] | 0.00001144 |
| MNIST | Indexing | 0.461885 s | [0.453226, 0.470227] | 0.00001144 |
| MNIST | Query | 0.765239 ms | [0.666209, 0.869227] | 0.00001144 |
| SIFT 100k | Indexing | 0.351569 s | [0.344363, 0.358964] | 0.00001144 |
| SIFT 100k | Query | 4.138300 ms | [4.018101, 4.269807] | 0.00001144 |

\*Difference = Original DB-LSH time − DB-LSH-IM time. Positive values favor DB-LSH-IM. CI is for the paired mean difference. Test: Wilcoxon signed-rank test, paired, exact. W⁺ = 210, W⁻ = 0 in all comparisons. Unadjusted p = 1.907 × 10⁻⁶; Bonferroni-adjusted p = 1.144 × 10⁻⁵. All six comparisons are statistically significant after Bonferroni correction.

### Timing caveats

The measurements characterize DB-LSH-IM as a whole, not any single implementation change. Because individual optimizations were not isolated, the lower recorded indexing and query times cannot be attributed specifically to the in-memory kd-tree index, the contiguous projections, the early distance termination, the bounded top-k heap, or the de-duplication scheme.

Both implementations were measured using wall-clock time, but their timing boundaries differ:

1. The DB-LSH-IM build timer includes coefficient generation, projections, and tree construction but excludes dataset loading; its query timer includes the query loop and in-loop metric calculations.
2. The original implementation's timing boundaries encompass the hash configuration and index construction for indexing time, and the hash and search operations for query time.

Although both measurements target the same phases, the internal operations included in each timing interval are not necessarily identical. The timing reductions in Tables 3 and 4 are therefore recorded timings under the stated protocol, not algorithm-level speedups.

Both implementations are deterministic under the fixed seed and single-thread configuration; the standard deviations across the 20 repetitions reflect system-level timing variability (OS scheduling, cache state, and thermal effects) rather than algorithmic variance.

Small Recall@50 differences were observed consistently across the 20 repetitions, with DB-LSH-IM higher by 0.0060–0.0080 on all three datasets. Their cause was not investigated and cannot be attributed to a specific implementation detail. Differences in data structures and traversal order can change the candidates examined and the results returned; similar aggregate cost and round counts do not establish that the two implementations examined identical candidate sets.

## Input format

The program reads the LSHBOX-style binary format used by the original implementation:

```text
int32 bytes_per_value (= 4)
int32 vector_count
int32 dimension
vector_count × dimension float32 values, in row-major order
```

The first 200 vectors in the file are reserved as queries and are excluded from the indexed dataset. The number actually evaluated is controlled by `--queries`.

## Build

### Requirements

- A C++17 compiler
- OpenMP
- `make`

The supplied Makefile uses GCC-compatible flags including `-O3`, `-march=native`, `-ffast-math`, and `-fopenmp`.

### Repository layout

```
.
├── Makefile
├── README.md
├── convert_mnist.py   # MNIST IDX → DB-LSH binary format
├── convert_sift.py    # SIFT fvecs → DB-LSH binary format
├── db_lsh.cpp         # index construction, query, benchmarking
├── db_lsh.hpp         # data structures and public API
└── main.cpp           # CLI entry point
```

### Compiling

```bash
make            # builds the dblsh_im executable
make clean      # removes object files and the executable
```

## Usage

```bash
./dblsh_im <dataset> [c k L K beta R_min] [options]
```

| Positional argument | Default | Description |
|---|---|---|
| `dataset` | `audio` | dataset name; loads `<dataset>.data_new` |
| `c` | 1.5 | approximation ratio, > 1 |
| `k` | 50 | neighbors to return |
| `L` | 5 | number of projection tables |
| `K` | 10 | projection dimension per table |
| `beta` | 0.1 | candidate budget as fraction of N |
| `R_min` | auto | initial search radius (auto for known datasets) |

| Option | Default | Description |
|---|---|---|
| `--data PATH` | auto | explicit path to the LSHBOX `.data_new` file |
| `--queries N` | 100 | number of queries, at most 200 |
| `--seed N` | 0 | RNG seed; 0 matches the original hashes |
| `--threads N` | 0 | OpenMP threads, 0 = all available |
| `--leaf N` | 128 | kd-tree leaf size |
| `--out PATH` | `results/DB-LSH-IM_result.csv` | benchmark CSV path |
| `--no-gt` | off | skip exact ground truth (faster, no recall) |
| `--refresh-gt` | off | recompute ground truth instead of using the cache |
| `--quiet` | off | suppress configuration printout |
| `-h, --help` | — | show the help message |

The data file is resolved from the first existing path among `data/<dataset>.data_new`, `../dataset/<dataset>.data_new`, and `<dataset>.data_new`.

### Exact invocations used in the reported measurements

```bash
./dblsh_im audio 1.5 50 5 10 0.1 0.3   --queries 100 --seed 0 --threads 1
./dblsh_im mnist 1.5 50 5 10 0.1 500   --queries 100 --seed 0 --threads 1
./dblsh_im sift  1.5 50 5 10 0.1 1.0   --queries 100 --seed 0 --threads 1
```

### Example

```bash
./dblsh_im audio 1.5 50 5 10 0.1 0.3 --threads 1
```

## Dataset conversion scripts

Both scripts write to a `data/` directory relative to the repository root. Place the raw source files at the expected paths before running them.

**MNIST** — converts the gzipped IDX training images to float32 vectors (0–255 range, no normalization):

```bash
# expects: data/mnist_raw/train-images-idx3-ubyte.gz
# produces: data/mnist.data_new
python3 convert_mnist.py
```

**SIFT 100k** — converts the official SIFT1M query and base fvecs, placing the 200 queries before 100,000 base vectors with per-vector dimension headers removed:

```bash
# expects: data/sift_download/sift/sift_query.fvecs
#          data/sift_download/sift/sift_base.fvecs
# produces: data/sift_download/sift_100k.data_new
python3 convert_sift.py
```

## Evaluation measures

| Measure | Definition |
|---|---|
| Indexing time | Reported index-construction time (wall-clock) |
| Query time | Reported query-phase time per evaluated query (wall-clock) |
| Recall@50 | Proportion of exact top-50 IDs recovered |
| Distance ratio | Comparison of returned and exact-neighbor distances (rank by rank; 1.0 when the exact distance at a rank is zero) |
| Cost | Unique candidates examined divided by indexed dataset size |
| Rounds | Search-window rounds averaged across queries |

## Limitations

The experiment is limited to three datasets, fixed configurations, one machine, one thread, and 100 evaluated queries. It does not isolate individual optimizations, measure resident memory, sweep parameters, or compare graph-based and quantization-based ANN methods. Repeated runs do not establish variability across different datasets, query selections, or seeds. The review portions of the paper are narrative rather than systematic, and the experiment is not a general ranking of ANN methods.

## Related resources

- **Reference DB-LSH implementation (baseline):** <https://github.com/Jacyhust/DB-LSH>
- **This repository:** <https://github.com/Valeriatheknave/LSH-paper-experiment-program>
- **Audio dataset:** obtained from the reference DB-LSH repository
- **MNIST:** the MNIST database
- **SIFT:** the TexMex SIFT1M collection

## References

- [41] Y. Tian, X. Zhao, and X. Zhou, "DB-LSH: Locality-Sensitive Hashing with Query-based Dynamic Bucketing," in Proc. IEEE 38th Int. Conf. Data Engineering (ICDE), 2022, pp. 2250–2262.
- [42] Y. Tian, X. Zhao, and X. Zhou, "DB-LSH 2.0: Locality-sensitive hashing with query-based dynamic bucketing," IEEE Trans. Knowl. Data Eng., vol. 36, no. 3, pp. 1000–1015, Mar. 2024.
