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
