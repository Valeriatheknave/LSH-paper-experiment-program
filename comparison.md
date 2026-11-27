# Benchmark: original DB-LSH vs DB-LSH2

Environment: 2 CPU cores, GCC 14.2, `-O3 -march=native -ffast-math -fopenmp`.
Dataset: `audio.data_new` from the reference repository (54,387 vectors,
dim 192; 200 reserved queries, 54,187 indexed points; 100 queries evaluated).
Parameters: `c=1.5, k=50, L=5, K=10, beta=0.1, R_min=0.3, seed=0`.

Commands:

```bash
# original, after adding two missing #include lines so it compiles on GCC 14
cd /home/user/dblsh-reference/dbLSH
./dblsh audio 1.5 50 5 10 0.1 0.3

# improved
cd /home/user/dblsh-improved
./dblsh2 audio 1.5 50 5 10 0.1 0.3 --threads 1
./dblsh2 audio 1.5 50 5 10 0.1 0.3 --threads 2
```

## Results

| Metric | Original DB-LSH | DB-LSH2, 1 thread | DB-LSH2, 2 threads |
|---|---:|---:|---:|
| recall@50 | 0.9106 | 0.9186 | 0.9186 |
| distance ratio | 1.00395 | 1.00378 | 1.00378 |
| cost (verified / N) | 0.10091 | 0.10093 | 0.10093 |
| average rounds | 1.66 | 1.66 | 1.66 |
| unique candidates verified / query | ~5,469 | 5,469 | 5,469 |
| index build | 112 ms | ~75 ms | ~75 ms |
| query time | 4.86 ms/query (CPU clock) | 2.88 ms/query (wall) | 1.47 ms/query (wall) |
| total wall time, ground truth cached | 0.95 s | ~0.55 s | 0.35 s |

## Interpretation

* Quality is effectively identical: the returned neighbour **sets** match the
  original exactly on the checked queries. The small recall difference comes
  from the original's position-wise comparison of tied distances; DB-LSH2
  reports the reference's set-intersection definition.
* The search visits the same number of candidates and needs the same number of
  widening rounds, so the speedup comes from data layout and parallelism, not
  from weakening the approximation.
* Single-thread queries are ~1.7× faster. With both cores, query throughput is
  ~3.3× the reference.
* The original writes five R*-tree page files under `RStar_index_file/`;
  DB-LSH2 keeps everything in RAM and writes only the CSV and the ground-truth
  cache.

Note: the original source needed `#include <array>` in `src/util.cpp` and
`#include <string>` in `src/Preprocess.h` to compile with the current GCC.
Those were the only changes made to the reference for this comparison.
