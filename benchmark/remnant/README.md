<!-- Author: Winston Cai. -->

# Native / Packed Triton / Packed Kernel microbench

Model-free production-function comparison on one H100/SM90. Protocol
`remnant-three-modes-v1`, schema version 2, intentionally changes the historical
Native baseline: Native now stores **unpruned raw FP32 producer inputs**, rather
than a matched Packed control. The old ≤2% Native gate remains with the archived
benchmark; current timing is descriptive. Microbenchmarks do not establish
serving TPOT, TTFT, SLO capacity, or task quality.

## Measured paths

| Mode | Decode | Post-compressor store |
|---|---|---|
| `native` | Unchanged Native records, stock FlashMLA, combine | Production normalization, RoPE, quantization and Native storage |
| `triton` | Packed unpacking, RoPE, Native materialization, stock FlashMLA, combine | Production raw-magnitude selection and Packed storage |
| `kernel` | Direct Packed FlashMLA and combine | Fused FP32 selection, normalization and Packed storage |

The default benchmark retains 256/512 coordinates in 328 bytes. Native has
584 logical bytes. Startup per-layer counts are supported only by the Triton
serving path (64..448 in steps of 64). The optimized Kernel benchmark remains
256-only; the frozen timing matrix is unchanged. Smaller configured records
require separate GPU correctness qualification, not reinterpretation of these
historical timings.
Stores exclude compressor projection and common compressor computation. Native
invalid destinations are encoded as skipped plan entries before timing; its
production writer does not accept negative destination pointers.

Decode defaults: H64/H128 × B8/B16 × K512/K317, Q1, SWA64, D512,
seed 20260919, 10 warmups, 30 rounds, 100 CUDA graph replays per round sample.
B32 is supplemental via `--batches 8,16,32`. Modes share raw inputs, queries,
logical positions, physical selections, lengths and SWA inputs. Mode-owned caches,
workspaces, metadata, graph outputs and allocations persist through measurement.
Initialization, compilation, cache construction and validation are untimed.
Timing order is deterministically shuffled per round; ratios pair matching rounds.

Store uses FP32 latent rows and weights across the established 25 layouts:
decode sizes/boundaries, prefill sizes, shuffled mappings, padded entries,
invalid destinations and empty writes. Definitions are shared with the fused
store test in `python/sglang/benchmark/remnant/cases.py`. Historical BF16 fixtures
are unchanged. Decode stores use graphs/100 repeats; prefill stores use eager
calls/20 repeats, with the same 10 warmups and 30 rounds.

## Run

The parent repository's maintained Modal entry points own CPU builds and verified
extension loading. A separate authorization is required before remote execution:

```bash
modal run --detach scripts/modal/kernels.py::benchmark_kernels_worker --run-id <unique-id>
modal run scripts/modal/kernels.py::profile_kernels --run-id <unique-id>
```

Direct execution requires installed fork extensions and an explicitly CPU-built
Native FP32 D512/RoPE64/page64 store library. No native compilation is permitted
on the measured GPU:

```bash
python benchmark/remnant/microbench.py \
  --native-library /path/to/verified/native-store.so --suite all \
  --modes native,triton,kernel --seed 20260919 --output /path/to/timing
python -m pytest benchmark/remnant/test_microbench.py -q
```

For GPU tests set `REMNANT_NATIVE_STORE_LIBRARY` to that verified library.
Options: `--suite all|decode|store`, `--modes`, `--heads`, `--batches`,
`--topk-lengths`, `--seed`, `--warmup`, `--rounds`, `--repeats`,
`--prefill-repeats`, `--output`. Existing outputs are never overwritten.

JSON is canonical; CSV carries raw samples and metadata alongside summary
statistics. All times are milliseconds per complete operation. p95 is the
percentile of round-average timing samples, not request latency. Ratios below
one mean the numerator is faster. Save source/content hashes, binary manifests,
input identities, environment and validity before interpreting performance.

## Validity and profiling

Native storage uses an independent FP64 normalization reference with explicit FP32 rounding-error intervals propagated through BF16 and FP8 quantization. Scale codes and quantized values must lie within those intervals; attention atol/rtol stay 2e-2. This avoids rejecting a valid adjacent quantization code solely because two FP32 reduction orders differ at a midpoint.
Packed Triton/Kernel bytes must match exactly; attention output/LSE are compared
with a matched Packed Torch reference at established atol/rtol 2e-2. Tests cover
all eight decode and 25 store cases, partial lengths, invalid/padded entries,
duplicate selections, mappings and changing graph inputs. GPU skips are missing
coverage, never passing validity.

Nsight Compute captures warmed eager operations after initialization and
compilation, using `--profile-mode` and one `--case`. The Modal profile action
collects all 24 decode combinations and all three modes for decode-r25-all_valid
and prefill-r8192-all_valid. It records every selected operation kernel including
selection, reconstruction and combine; duration, DRAM traffic/bandwidth, L2,
occupancy, registers, local spill traffic and warp stalls are required. Reports,
commands, versions and exported summaries are separate from timing records.
Profiling does not refresh or replace unprofiled timing.
