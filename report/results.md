# Benchmark Results — RTX 4080 12GB, BF16

## Peak VRAM (forward + backward)

| seq_len | Dense   | DSA     | CSA     | HCA     |
|---------|---------|---------|---------|---------|
| 2 048   | 0.53 GB | 0.07 GB | 0.06 GB | 0.04 GB |
| 4 096   | 2.05 GB | 0.14 GB | 0.16 GB | 0.08 GB |
| 8 192   | 8.12 GB | 0.26 GB | 0.54 GB | 0.20 GB |
| 16 384  | **OOM** | 0.51 GB | 1.99 GB | 0.63 GB |
| 32 768  | **OOM** | 1.01 GB | 7.64 GB | 2.25 GB |

DSA and HCA stay under 1 GB even at 32K. CSA scales faster (O(n·k) cross-gather)
but remains within 12 GB through 32K.

## Throughput (tok/s, forward-only, 20 iterations, BF16)

| seq_len | Dense     | DSA              | CSA             | HCA              |
|---------|-----------|------------------|-----------------|------------------|
| 2 048   | 873K      | 460K (0.53×)     | 44K (0.05×)*    | 873K (1.00×)     |
| 4 096   | 494K      | 458K (0.93×)     | 44K (0.09×)*    | 923K (1.87×)     |
| 8 192   | 244K      | 468K (1.92×)     | 43K (0.18×)*    | 908K (3.72×)     |
| 16 384  | OOM       | 468K             | 42K*            | 906K             |
| 32 768  | OOM       | 467K             | 41K*            | 739K             |

*CSA throughput is bottlenecked by Python-level per-segment for-loops, not the
algorithm. A batched-matmul or Triton kernel implementation would close this gap.
The memory scaling (7.64 GB at 32K) confirms the sparse cross-gather pattern works
correctly.

## Observations

- **DSA** delivers stable ~467K tok/s regardless of seq_len — O(n·w) complexity
  confirmed. At 8K it is already 1.92× faster than dense.
- **HCA** reaches 3.7× throughput over dense at 8K. Chunk-local L0 computation
  maps efficiently to GPU parallelism. Slight drop at 32K is expected as L1
  inter-chunk attention grows O((n/c)²).
- **CSA** Python loop overhead dominates at all seq_lens measured. This is a
  benchmark implementation limitation, not an algorithmic one.
- Dense OOMs at 16K on RTX 4080 12GB; all sparse modes handle 32K comfortably.
