<!-- Authored by Winston Cai. -->

# Remnant microbenchmarks

This directory contains the model-free Native/Adapter/Direct C4 microbenchmark
for the SGLang fork. It uses an explicit decode-case matrix, CUDA Graph replay,
synchronized measurements, and timestamped JSON/CSV artifacts.

Serving throughput, LongSWE, SLO, and other study benchmarks remain in the
outer `flash-optimizations` repository.

Run on an H100 from the SGLang repository root:

```bash
PYTHONPATH=python python benchmark/remnant/microbench.py \
  --heads 64,128 --batches 8,16 --topk-lengths 512,317 \
  --warmup 10 --repeats 100 --rounds 30
```

The default matrix compares stock Native decode, the existing Packed-to-Native
Adapter plus stock decode, and direct Packed decode for H64/H128, batch 8/16,
and full/partial C4 top-k lengths. It writes:

```bash
microbench-<UTC timestamp>.json
microbench-<UTC timestamp>.csv
```

The JSON is canonical and includes raw round samples, summary statistics,
source identifiers, environment metadata, and case definitions. The CSV is a
flat summary. The benchmark requires a compiled SM90 FlashMLA extension and
does not load model weights.
