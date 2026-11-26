# DB-LSH2 — a faster, memory-friendly DB-LSH

A C++17 reimplementation of **DB-LSH: Locality-Sensitive Hashing with
Query-based Dynamic Bucketing** (Tian, Zhao, Zhou; ICDE 2022). It keeps the
reference algorithm and reproduces its results, but replaces the disk-backed
R*-tree implementation with a compact in-memory index and adds parallelism.

## Algorithm preserved

| Reference DB-LSH | This program |
|---|---|
| `S = L·K` Gaussian vectors `a_s ~ N(0,1)` | same vectors, and `--seed 0` uses the *identical* random stream as the reference |
| projection `h_s(x) = <a_s, x>` | identical |
| one `K`-dimensional R*-tree per table | one bulk-loaded kd-tree per table, with MBR pruning |
| query window `[h_j ± init_w/2]`, `init_w = 4·c²·R_min`, widened by `c` each round | identical |
| candidate budget `T = β·N + k`, max 30 rounds, `flag[]` de-duplication | identical, with generation-stamped de-duplication |
| exact Euclidean re-ranking of candidates | identical, with an early-exit bound |
| first 200 vectors of the file reserved as queries | identical |

The binary data format is unchanged:

```
int32 bytes_per_value (=4), int32 vector_count, int32 dimension,
vector_count * dimension * float32 (row-major)
```

## Improvements

1. **In-memory kd-trees instead of page-file R*-trees.** No `RStar_index_file`
   directory, no page cache, no disk writes. Bulk-loaded by median split on
   the widest projected dimension; leaves hold contiguous point blocks.
2. **Cache-friendly projections.** Each table's `N × K` coordinates live in one
   contiguous block. A data vector is read `L` times, not `L·K` times.
3. **OpenMP everywhere independent.** Projection, index construction, exact
   ground truth and the query loop are parallel; each query thread owns a
   reusable scratch buffer (stamps, heap, traversal stack, hash vector).
4. **O(1) de-duplication.** A generation stamp replaces the reference's
   `int flag[N]` reset and avoids re-initialising it per query.
5. **Bounded top-k heap.** Only `k` candidates are kept, instead of a `βN`
   result array that is fully sorted at the end.
6. **Early distance termination.** As soon as a partial squared distance cannot
   beat the current k-th candidate, the remaining dimensions are skipped.
7. **Better traversal order.** Children are visited nearest-first in projected
   space, so the most useful candidates are found before the budget is spent.
8. **Ground-truth cache.** Exact top-k is cached beside the CSV with a data
   fingerprint; repeated runs skip the O(Q·N·dim) exact scan.
9. **Correct wall-clock timing** with `omp_get_wtime()` (the reference uses
   `clock()`, which under-reports work when threads are used).
10. **Builds cleanly with modern GCC.** The reference needed `<array>` and
    `<string>` includes added before it would compile here.

## Build

```bash
cd /home/user/dblsh-improved
make            # produces ./dblsh2
make test       # beta=1 sanity check: recall must be 1.0
make clean
```

Requirements: a C++17 compiler and OpenMP (`g++ -O3 -march=native -ffast-math
-fopenmp`). No third-party libraries.

## Run

```bash
./dblsh2 audio 1.5 50 5 10 0.1 0.3 --threads 2
```

Positional arguments match the reference:

```
dblsh2 <dataset> [c k L K beta R_min] [options]
```

| Option | Meaning | Default |
|---|---|---|
| `--data PATH` | explicit `.data_new` file | `data/<dataset>.data_new` |
| `--queries N` | queries to evaluate (1–200) | 100 |
| `--seed N` | hash RNG seed; 0 reproduces reference hashes | 0 |
| `--threads N` | OpenMP threads, 0 = all available | 0 |
| `--leaf N` | kd-tree leaf size | 128 |
| `--out PATH` | benchmark CSV path | `results/DB-LSH2_result.csv` |
| `--rmin X` | override initial radius | dataset table in `main.cpp` |
| `--no-gt` | skip ground truth (faster; no recall/ratio) | off |
| `--refresh-gt` | recompute ground truth instead of loading cache | off |
| `--quiet` | suppress configuration printout | off |
| `-h, --help` | usage | — |

Default `R_min`: audio 0.3, mnist 500, cifar 300, deep1m 0.2, NUS 6.5,
Trevi 700, gist 0.3, other 1.0.

## Output

Console: recall@k (set intersection, as in the reference), distance ratio,
cost (verified candidates / N), average rounds, index build time and query
wall time. Each run appends a row to the CSV with all parameters and metrics.
A sibling `<dataset>.gt` file caches exact ground truth.

Example CSV row:

```
Dataset,c,k,L,K,beta,R_min,seed,threads,Q,Recall,Ratio,Cost,Rounds,Verified,Nodes,Hits,EarlyExits,IndexBuild_ms,Query_ms
audio,1.5,50,5,10,0.1,0.3,0,2,100,0.918600,1.003777,0.100928,1.66,5469,44055,173395,134917,75.38,146.82
```

## Measured comparison

Same dataset, parameters and hash seed; see `results/comparison.md` for details.

| Metric | Original DB-LSH | DB-LSH2 (1 thread) | DB-LSH2 (2 threads) |
|---|---:|---:|---:|
| recall@50 | 0.9106 | 0.9186 | 0.9186 |
| distance ratio | 1.00395 | 1.00378 | 1.00378 |
| cost | 0.10091 | 0.10093 | 0.10093 |
| avg rounds | 1.66 | 1.66 | 1.66 |
| index build | 112 ms | ~75 ms | ~75 ms |
| query time | 4.86 ms/query (CPU) | 2.88 ms/query | 1.47 ms/query |
| total wall (GT cached) | 0.95 s | ~0.5 s | 0.35 s |

The recall/ratio difference is due to tie ordering in the reference's
position-based reporting; comparing result *sets* gives identical neighbours.

## Files

```
src/db_lsh.hpp / db_lsh.cpp   index, query engine, benchmarking
src/main.cpp                  command-line interface
data/audio.data_new           sample dataset (from the reference repository)
results/DB-LSH2_result.csv    appended benchmark rows
results/audio.gt              cached exact ground truth
results/comparison.md         baseline vs improved report
Makefile
```
