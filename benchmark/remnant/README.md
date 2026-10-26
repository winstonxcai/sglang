<!-- Author: Winston Cai. -->

# Native / Packed Triton / Packed Kernel microbench

**v0.5.22 migration:** this checkout is a source port, not GPU-qualified.
Past GPU results below refer to the v0.5.18 studies and do not validate this
version. From the parent, set `REMNANT_INTEGRATION=v0.5.22` for all maintained
Modal commands. The default selector still uses v0.5.18.

Model-free production-function comparison on one H100/SM90. Protocol
`remnant-three-modes-v1`, schema version 2, intentionally changes the historical
Native baseline: Native now stores **unpruned raw FP32 producer inputs**, rather
than a matched Packed control. The old ≤2% Native gate remains with the archived
benchmark; current timing is descriptive. Microbenchmarks do not establish
serving TPOT, TTFT, SLO capacity, or task quality.

## V4.1 FP4 Native / Packed Direct follow-up

Author: Winston Cai. `--suite v41-fp4-decode` uses V4.1 production H100
geometry: **64 query heads, SWA128/page256, Q1/K512**, with batches
8/16/24/28/32. Both compressed global page sizes are covered: **C1/page256**
and **C2/page128**, for ten cases. The model has one KV head and a separate
32-head indexer; the indexer is outside this attention timing. Tensor-parallel
shards with fewer query heads are padded to 64 by this SM90 runtime. Batches
remain a synthetic sweep, not measured production traffic.

Protocol `remnant-v41-fp4-decode-v3` distinguishes the 128-token attention
window from the 256-token paged SWA allocation. The runtime factory
`kv_cache_configurator.py` requires 256-token SWA pages.
It supersedes the historical v1 20-case matrix and the untimed v2 fixture
that incorrectly used page128. Historical receipts retain their original protocol;
they do not qualify this matrix. SWA64/H128 remain correctness coverage only.
Native uses dense 288-byte FP4; Packed Direct uses 224-byte keep256 records.
An independent CPU quantization/pruning/attention oracle validates each path.
Shared post-normalization/post-RoPE BF16 rows, query/indices/SWA/sink hashes bind
both paths. Selection/store, setup, allocation, references and capture are
untimed. Complete Direct, including scheduling/split/combine, is timed in CUDA
graphs with10 warmups,30 randomized paired rounds and100 replays per sample.
Existing JSON/CSV retain raw samples and per-case ratios; p95 is a round-average
graph timing statistic. No minimum speedup or serving claim is implied.

From the parent, after identical source/binary/tool-qualified strict checks:

```bash
REMNANT_INTEGRATION=v0.5.22 REMNANT_VALIDATION_SUITE=v41-fp4 \
  modal run scripts/modal/kernels.py::benchmark_kernels \
  --run-id <unique-id> --suite v41-fp4-decode \
  --qualification-runs <validation-run-id>[,<validation-run-id>]
```

The runner rejects incomplete memcheck/racecheck/synccheck receipts, changed
identities, missing cases and skipped checks. H100 only; no model volumes,
weights, serving or profiling. Budget authorization is owned by the parent
experiment README. Default V4 suites are unchanged. Rollback omits this suite.

## Prefill-read experiment

Question: does compact C4 storage save enough historical-KV preparation and
attention time to contribute to the serving target of 2.07 requests/s versus
Native's 1.8? The changed mechanism is the persistent record: unchanged Native,
Packed192 and Packed256. Both Packed variants use the existing production
prefill reader; this experiment changes no runtime kernel or frontend.

### Fused unpack/RoPE candidate

Question: can one Packed read kernel remove the measured separate RoPE pass
without changing the reconstructed BF16 rows? `--prefill-reader split` remains
the default and the serving path. The opt-in `--prefill-reader fused` candidate
shares unpack arithmetic, rounds dequantized values to BF16 before rotation,
then writes the final row once. Selection, persistent format, Native, SWA and
attention are unchanged. H100 numerical and sanitizer qualification passed; this
remains an opt-in candidate rather than the serving default.

Use the current matrix and seed for a new candidate run; record the reader in
JSON/CSV and retain the verified split study in
`benchmarks/microbench/prefill-read-20261018/` as historical evidence. New timing
includes a fresh Native control; comparisons against the saved split timing
are cross-run, not paired. Never overwrite or relabel the original measurements.
Before interpreting speed, require exact BF16 equality with the split reader
for every supported survivor count, the independent references, invalid/padded
and duplicate selections, strided indices, changing-input graph replay, context
growth without recompilation, and sanitizer checks. Profile coverage must show
one fused C4 read and no standalone tail-RoPE launch. Stop on a parity, coverage,
sanitizer, deadline or budget failure. Promote no serving default without passing
these checks and a measured complete-path improvement. Roll back by selecting
`split`; no cache conversion is needed. New Modal execution needs scoped budget
authorization. The maintained `scripts/modal/kernels.py` forwards the reader
option and reuses the batched profiler. Numerical implementation and fixtures
remain in this SGLang fork. The exact controller used for the recorded study is
preserved in its verified source archive; the maintained runner retains its
normal resource caps.

```bash
python third_party/sglang-v0.5.22/benchmark/remnant/microbench.py \
  --suite prefill-read --prefill-reader fused \
  --native-library <CPU-built-library> --output <fresh-output-prefix>
```

From the parent repository, run timing/validation before profiling:

```bash
modal run --detach scripts/modal/kernels.py::benchmark_kernels \
  --suite prefill-read --prefill-reader fused --run-id <unique-id>
modal run --detach scripts/modal/kernels.py::profile_kernels \
  --suite prefill-read --prefill-reader fused --run-id <unique-id> \
  --max-seconds 3500
```

These actions run serially, without automatic retries, weights or model volumes.
Candidate timing runs the existing unpack and CUDA-graph tests plus memcheck,
racecheck and synccheck before measurement. Timing has a 900-second hard limit;
maintained profiling and CPU builds have 3,600-second hard limits. The recorded
study used the saved controller under
`raw/experiments/prefill-read-fused-rope/20261018T144353Z/source/scripts/modal/`,
with its separately authorized 900-second GPU and 3,000-second CPU limits.
Retain that verified archive for exact historical reproduction.

The 2026-10-18 study passed 276 tests plus 15 subtests, and 67 checks under each
sanitizer. All 36 timing cases and 108 captured operations were verified. Fusion
reduces reconstruction cost but still loses to Native in these prefill cases;
the parent repository records results in
`benchmarks/microbench/prefill-read-fused-rope-20261018/`. Its execution used the
frozen fork copies under the snapshot's `source/third_party/`, as recorded in
the manifest. The GPU deadline expired after all three reports were captured,
during CSV export. The archived local NVIDIA report-API exporter completed all
exports and checked them against every existing CLI export. This recovery does
not recapture or allocate a GPU; retain the original timeout record.

The opt-in `--suite prefill-read` compares all three variants in paired rounds
on identical FP32 latents/weights, queries, logical positions, shuffled physical
locations, sparse selections, SWA and sink. It measures reconstruction (including
SWA preparation), attention alone, and the complete preparation/attention path.
Initialization, selection/store, metadata construction and validation are untimed.
Native remains an unpruned baseline; each Packed budget has its own reference.

Current serving-informed synthetic shapes use history 143,360, query chunks
128/512/1,536/4,096/8,192 and H64/H128. `--chunked-prefill-size 8192` is a
scheduler budget, not a requirement to fill each request's query chunk. The
128-token case represents small continuations, 512/1,536 approximate typical
uncached additions, 4,096 covers larger extensions, and 8,192 is a full-chunk
ceiling case. This matrix is not a measured or workload-weighted distribution;
actual scheduler per-request lengths and batch composition are not captured here.
Recent replay records have approximately 142k-token
median histories and 550/1,400 median/p90 uncached additions; they do not contain
actual query/KV tensors or selected indices. These are not literal request replays.
The SWA window is 128; C4/SWA storage pages are 64/128, following the current
model configuration defaults (`cfg.window_size` controls SWA storage pages).
Warm reuse and four-buffer rotation are separate fixture policies. Rotation is a
cache-sensitivity control, not a simulation of the full model's weight traffic.

Acceptance requires complete three-variant coverage, independent numerical
checks, identical shared-input hashes, positive finite raw samples and nonempty
Nsight Compute reports with DRAM and L2 metrics. Performance is descriptive;
no microbench result establishes the serving target or task quality. Stop on
missing validity, profiler coverage, deadline or budget; preserve partial records.
The existing decode/store defaults and historical records remain unchanged.

```bash
modal run --detach scripts/modal/kernels.py::benchmark_kernels \
  --run-id <unique-id> --suite prefill-read
modal run --detach scripts/modal/kernels.py::profile_kernels \
  --run-id <unique-id> --suite prefill-read --max-seconds 3500
```

Wait for timing/validity to finish successfully before launching profiling; the
two functions must not allocate H100s concurrently for this experiment.

Timing covers 60 cases (ten shapes × two cache policies × three stages), with
10 warmups, 30 paired rounds and 20 eager repetitions. Nsight Compute covers
the same 60 cases in all three variants: 180 labeled operations in three reports
(one report per variant). Batching amortizes application-replay process startup;
NVTX-filtered exports retain each operation's kernel names and counters separately.
`--cases` can select a smaller set when explicitly requested. Capture initialization and
validation are inactive; application replay with cache control disabled preserves
the preceding cache-preparation sequence. The selected traffic metrics are
GPU duration, DRAM read/write bytes, L2 read/write sectors and L2 hit rate.
Raw `.ncu-rep`/CSV exports and canonical per-launch JSON stay separate from timing.
Complete-path latency is authoritative; isolated-stage timings need not add up
because the cache state differs. Preserve per-launch hit rates rather than
averaging percentages across unequal traffic volumes. Application replay follows
[NVIDIA's cache-state guidance](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html#application-replay).

The 2026-10-18 records above retain their original 36 cases / 108 operations.
The new 128/4,096 shapes have CPU contract coverage only; no new GPU validation,
timing or profiling is implied. Resource caps are unchanged, and a larger matrix
requires a fresh execution budget review rather than automatic retries or a
timeout increase. To select the historical direct-run matrix explicitly, use
`--query-chunks 512,1536,8192`; the current full-default coverage judge still
requires all 60 cases. Use the archived source/controller for exact historical
qualification. Existing case names, input seeds and measurements are unchanged.

Direct execution uses the same CPU-built Native library as the other suites:

```bash
python benchmark/remnant/microbench.py --suite prefill-read \
  --native-library /path/to/verified/native-store.so --output /path/to/prefill-read
python benchmark/remnant/microbench.py --suite prefill-read \
  --native-library /path/to/verified/native-store.so \
  --case prefill_h64_s143360_q512_rotating_complete \
  --profile-path packed192 --output /path/to/prefill-profile
```

The low-level SM90 sparse prefill kernel returns natural-log max logits/LSE;
the compiled SGL compatibility shim converts both to log2. Validation checks
that public API contract, including the sink in the output softmax. Attention
timing and traffic include both conversion kernels. All workspace rows are checked;
attention output, max logits
and LSE use eight deterministic query samples plus a full-output finiteness check.
SWA bytes come from the existing production Native writer, retiled untimed from
its CPU-built page64 format. No new GPU-side native compilation is introduced.
CPU import preflight uses CUDA's driver stub only in its child-process loader
environment; H100 validation and measurement use the real driver.

Existing outputs remain protected. Only an owning `--profile-replay-id` with
identical configuration, source/binary hashes and input hash may reuse its own
validated output during Nsight application replay. This does not authorize retries.
Rollback is to omit `--suite prefill-read`; Native and production serving are unchanged.

## Measured paths

| Mode | Decode | Post-compressor store |
|---|---|---|
| `native` | Unchanged Native records, stock FlashMLA, combine | Production normalization, RoPE, quantization and Native storage |
| `triton` | Packed unpacking, RoPE, Native materialization, stock FlashMLA, combine | Production raw-magnitude selection and Packed storage |
| `kernel` | Direct Packed FlashMLA and combine | Fused FP32 selection, normalization and Packed storage |

The benchmark accepts `--keep-count 64|128|192|256|320|384|448` (default 256).
This is retained coordinates per record, independent of the `--topk-lengths`
selected-record count. Packed records use `keep_count + 72` logical bytes;
Native remains unpruned at 584 bytes. At 256, modes default to all three. Other
counts default to `native,triton`; explicitly requesting Kernel fails because
its production store and Direct reader still require 256. Only requested modes
are initialized. This benchmark extension does not extend the production kernels.

The question is whether a smaller persistent record reduces complete-operation
latency and memory traffic after selection/reconstruction costs. Compare the
same cases and raw-input hashes across budgets, keeping Native unchanged. Case
seeds, geometric matrix and default timing settings remain unchanged. Survivor
count and logical record size are separate identities in JSON/CSV; logical bytes
are not measured DRAM traffic. Timing is descriptive, with numerical validity
required before interpretation. New counts require H100 qualification; historical
256 results cannot be relabeled as smaller-budget measurements.
Stores exclude compressor projection and common compressor computation. Native
invalid destinations are encoded as skipped plan entries before timing; its
production writer does not accept negative destination pointers.

Decode defaults: ten cases, H64/H128 × B8/B16/B24/B28/B32 × K512, Q1, SWA64, D512,
seed 20260919, 10 warmups, 30 rounds, 100 CUDA graph replays per round sample.
B24/B28 cover the serving concurrency targets; B32 also covers the padded graph
used for 29 or 30 decoding sequences with the local extended graph configuration.
Larger batches are supplemental via `--batches 64`. K317 is removed from default
timing/profiling; partial lengths remain covered by correctness tests. Historical
results are unchanged, and explicit `--topk-lengths` remains available for timing
reproduction. Modes share raw inputs, queries,
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
modal run --detach scripts/modal/kernels.py::benchmark_kernels --run-id <unique-id> --keep-count 192
modal run scripts/modal/kernels.py::profile_kernels --run-id <unique-id> --keep-count 192
```

Direct execution requires installed fork extensions and an explicitly CPU-built
Native FP32 D512/RoPE64/page64 store library. No native compilation is permitted
on the measured GPU:

```bash
python benchmark/remnant/microbench.py \
  --native-library /path/to/verified/native-store.so --suite all \
  --modes native,triton,kernel --seed 20260919 --output /path/to/timing
python benchmark/remnant/microbench.py \
  --native-library /path/to/verified/native-store.so --keep-count 192 \
  --modes native,triton --output /path/to/timing-192
python -m pytest benchmark/remnant/test_microbench.py -q
```

For GPU tests set `REMNANT_NATIVE_STORE_LIBRARY` to that verified library.
Options: `--suite all|decode|store`, `--keep-count`, `--modes`, `--heads`, `--batches`,
`--topk-lengths`, `--seed`, `--warmup`, `--rounds`, `--repeats`,
`--prefill-repeats`, `--output`. Existing outputs are never overwritten.

JSON is canonical; CSV carries raw samples and metadata alongside summary
statistics. All times are milliseconds per complete operation. p95 is the
percentile of round-average timing samples, not request latency. Ratios below
one mean the numerator is faster. Save source/content hashes, binary manifests,
input identities, environment and validity before interpreting performance.

## Validity and profiling

Native storage uses an independent FP64 normalization reference with explicit FP32 rounding-error intervals propagated through BF16 and FP8 quantization. Scale codes and quantized values must lie within those intervals; attention atol/rtol stay 2e-2. This avoids rejecting a valid adjacent quantization code solely because two FP32 reduction orders differ at a midpoint.
At 256, Packed Triton/Kernel producer bytes must match exactly. Every selected
budget checks independent selection bitmaps and FP8 quantization intervals;
attention output/LSE are compared
with a matched Packed Torch reference at established atol/rtol 2e-2. Tests cover
all ten decode and 25 store cases, partial lengths, invalid/padded entries,
duplicate selections, mappings and changing graph inputs. GPU skips are missing
coverage, never passing validity.

Nsight Compute captures warmed eager operations after initialization and
compilation, using `--profile-mode` and one `--case`. The Modal profile action
collects ten decode cases and decode-r25-all_valid/prefill-r8192-all_valid in
the selected supported modes: 36 captures at default 256, or 24 at other counts.
It records every selected operation kernel including
selection, reconstruction and combine; duration, DRAM traffic/bandwidth, L2,
occupancy, registers, local spill traffic and warp stalls are required. Reports,
commands, versions and exported summaries are separate from timing records.
Profiling does not refresh or replace unprofiled timing.
