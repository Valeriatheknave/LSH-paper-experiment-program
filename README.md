# DB-LSH2 — an in-memory implementation of DB-LSH

DB-LSH2 is a C++17 reimplementation of **DB-LSH: Locality-Sensitive Hashing with Query-based Dynamic Bucketing** (Tian, Zhao, and Zhou; ICDE 2022).

It follows the original hashing and dynamic-bucketing approach while replacing the disk-backed R*-tree index with in-memory kd-trees and introducing other implementation optimizations.

> **Scope:** This is an engineering reimplementation, not a new LSH algorithm. It is not the official *DB-LSH 2.0* (TKDE 2023). Differences in data structures and traversal order can affect the candidates examined and the results returned.

## Approach

| Component | Original DB-LSH | DB-LSH2 |
|---|---|---|
| Hash functions | Gaussian random projections | Same approach; seed 0 is designed to match the reference random sequence in the tested environment |
| Index | One K-dimensional R*-tree per table | One bulk-loaded kd-tree per table, with bounding-box pruning |
| Query window | Initialized using `4 · c² · R_min` and widened by `c` | Same window initialization and widening approach |
| Search budget | `β · N + k` candidates, with at most 30 rounds | Same budget and round limit |
| Candidate de-duplication | Per-query flag array | Generation-stamped array |
| Candidate ranking | Euclidean distance | Euclidean distance with early termination |
| Queries | First 200 vectors reserved as queries | Same file convention |

The implementations follow the same general approach, but the use of different data structures and traversal orders does not guarantee identical candidate sets or results.

### Input format

The program reads the LSHBOX-style binary format used by the original implementation:

```text
int32 bytes_per_value (= 4)
int32 vector_count
int32 dimension
vector_count × dimension float32 values, in row-major order
```

The first 200 vectors in the file are reserved as queries and are excluded from the indexed dataset. The number actually evaluated is controlled by `--queries`.

## Implementation changes

- **In-memory kd-trees:** replace disk-backed R*-trees with bulk-loaded kd-trees and bounding-box pruning.
- **Contiguous projections:** store each table's projected coordinates in an `N × K` block.
- **OpenMP parallelism:** parallelize independent work in projection, index construction, exact ground-truth calculation, and query processing.
- **Generation-stamped de-duplication:** avoid resetting the entire flag array for every query.
- **Bounded top-k heap:** maintain the best `k` candidates without sorting a larger result array.
- **Early distance termination:** stop calculating a candidate's distance when its partial squared distance cannot improve the current top-k.
- **Traversal ordering:** visit the more promising child first in projected space.
- **Ground-truth cache:** reuse exact nearest-neighbor results across runs when the cache is valid.
- **Wall-clock timing:** use `omp_get_wtime()` for DB-LSH2 measurements.

The in-memory index trades disk usage for RAM. Datasets that can be processed using a disk-backed index may not fit in memory with this implementation.

## Build

### Requirements

- A C++17 compiler
- OpenMP
- `make`

The supplied Makefile uses GCC-compatible flags including `-O3`, `-march=native`, `-ffast-math`, and `-fopenmp`.

### Current repository layout

The current Makefile expects the source files in a `src/` directory, while the files in the repository are at the 
