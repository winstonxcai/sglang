# Remnant microbenchmarks

This directory contains portable Native/Packed C4 microbenchmarks for the
SGLang fork. It measures pack/write cost and selected-row reconstruction with
fixed tensors, CUDA events, warmup iterations, and synchronized measurements.

Serving throughput, LongSWE, SLO, and other study benchmarks remain in the
outer `flash-optimizations` repository. Fused reconstruction is outside this
milestone.

Run on a CUDA host from the SGLang repository root:

```bash
PYTHONPATH=python python benchmark/remnant/bench_packed.py --rows 1024 --repeats 100
```

The direct FlashMLA decode benchmark compares stock Native decode, the
existing Packed-to-Native adapter plus stock decode, and direct Packed decode:

```bash
PYTHONPATH=python python benchmark/remnant/bench_flashmla_decode.py \
  --batches 1,2,8 --repeats 50
```

It reports total CUDA-event time and the split-kernel plus combine-kernel time
for H64 and H128. It requires a compiled SM90 FlashMLA extension and does not
load model weights.
