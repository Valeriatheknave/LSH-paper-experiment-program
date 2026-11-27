# DB-LSH-IM — an in-memory reimplementation of DB-LSH

**DB-LSH-IM** (in-memory DB-LSH) is a C++17 reimplementation of **DB-LSH: Locality-Sensitive Hashing with Query-based Dynamic Bucketing** (Tian, Zhao, and Zhou; ICDE 2022 [41]).

It accompanies the paper:

> Fatemeh Mohammadkia and Faezeh Din-Mohammadi, *"Locality-Sensitive Hashing for Approximate Nearest Neighbor Search: A Controlled Comparison of DB-LSH and an In-Memory Reimplementation on Indexing and Query Performance,"* Department of Computer Science, Islamic Azad University, Karaj Branch, Karaj, Iran.

DB-LSH-IM follows the original hashing and query-based dynamic-bucketing approach while replacing the disk-backed R\*-tree index with bulk-loaded in-memory kd-trees and introducing other implementation optimizations.

> **Scope:** This is an engineering reimplementation, **not** a new LSH algorithm, and it is **not** the official *DB-LSH 2.0* extension (Tian, Zhao, and Zhou; TKDE 2023/2024 [42]). Differences in data structures and traversal order can affect the candidates examined and the results returned. The measurements below characterize the reimplementation as a whole; they do not isolate individual optimizations or establish algorithm-level speedups.

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
- **Wall-clock timing:** use `omp_get_wtime()` for DB-LSH-IM measurements.

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

The experiment was conducted on an Intel Core i5-8250U system with 19 GiB of RAM running Ubuntu Linux, using one thread. Each implementation–dataset configuration was repeated 20 times.

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

Parameters were fixed rather than swept; results characterize the selected configurations rather than optimized performance across all configurations.

## Results

### Table 1 — Performance comparison

| Dataset | Implementation | Indexing time (s) | Query time (ms/query) | Recall@50 | Distance ratio |
|---|---|---:|---:|---:|---:|
| Audio | Original | 0.240104 | 3.434614 | 0.9106 | 1.003950 |
| Audio | DB-LSH-IM | **0.062958** | **2.318740** | **0.9186** | **1.003777** |
| MNIST | Original | 0.623759 | 6.757688 | 0.9250 | 1.003920 |
| MNIST | DB-LSH-IM | **0.217501** | **5.646030** | **0.9316** | **1.003690** |
| SIFT 100k | Original | 0.455335 | 9.878633 | 0.9078 | 1.003650 |
| SIFT 100k | DB-LSH-IM | **0.119412** | **5.181170** | **0.9138** | **1.003414** |

Bold indicates the better measured value within each dataset. Lower indexing time, query time, and ratio are desirable; higher recall is desirable.

### Table 2 — Relative performance of DB-LSH-IM

| Dataset | Recall improvement | Query reduction | Query speedup | Indexing reduction | Indexing speedup |
|---|---:|---:|---:|---:|---:|
| Audio | +0.0080 | 32.42% | 1.4827× | 73.74% | 3.8242× |
| MNIST | +0.0066 | 16.42% | 1.1976× | 65.09% | 2.8703× |
| SIFT 100k | +0.0060 | 47.54% | 1.9083× | 73.75% | 3.8203× |

### Table 3 — Paired statistical analysis of timing

| Dataset | Metric | Speedup | 95% Bootstrap CI | Bonferroni-adjusted p-value |
|---|---|---:|---|---|
| Audio | Indexing | 3.8242× | [3.7149, 3.9288] | 1.1444 × 10⁻⁵ |
| Audio | Query | 1.4827× | [1.4551, 1.5138] | 1.1444 × 10⁻⁵ |
| MNIST | Indexing | 2.8703× | [2.8143, 2.9299] | 1.1444 × 10⁻⁵ |
| MNIST | Query | 1.1976× | [1.1814, 1.2139] | 1.1444 × 10⁻⁵ |
| SIFT 100k | Indexing | 3.8203× | [3.7333, 3.9068] | 1.1444 × 10⁻⁵ |
| SIFT 100k | Query | 1.9083× | [1.8809, 1.9348] | 1.1444 × 10⁻⁵ |

Every adjusted p-value is below the 0.05 level, and none of the confidence intervals includes 1.

### Timing caveats

The measurements characterize DB-LSH-IM as a whole, not any single implementation change. Because individual optimizations were not isolated, the lower recorded indexing and query times cannot be attributed specifically to the in-memory kd-tree index, the contiguous projections, the early distance termination, the bounded top-k heap, or the de-duplication scheme.

Timing interpretation is subject to two measurement differences:

1. The reference implementation reports CPU time measured with `clock()`, while DB-LSH-IM uses elapsed-time clocks (`omp_get_wtime()`). CPU time and elapsed wall time are different measures.
2. The timing boundaries differ: the DB-LSH-IM build timer includes coefficient generation, projections, and tree construction but excludes dataset loading, and its query timer includes the query loop and in-loop metric calculations.

The query speedups and reductions in Tables 2 and 3 are therefore recorded timings under the stated protocol, not algorithm-level speedups, and the indexing-time differences may partly reflect the timer boundaries.

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
| Indexing time | Reported index-construction time |
| Query time | Reported query-phase time per evaluated query |
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
