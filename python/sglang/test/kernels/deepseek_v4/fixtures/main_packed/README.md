<!-- Author: Winston Cai. -->

# Historical Packed parity fixture

Question: does current Triton preserve the original Packed selection, stored
bytes, reconstructed active records, and stock FlashMLA output/LSE?

The six files listed in `source.json` are unchanged snapshots of parent `main`
commit `b69841acf0d29181f84999d9b7e3cd5505eea9ad`, under `mustafar/`.
Tests verify every SHA256 before importing this isolated test package. The
snapshot is necessary because importing the current implementation as its own
baseline would not detect historical drift. This is a permanent regression
fixture, never imported by production or used to patch a serving tree.

Consumer: `test_remnant_unpack_kernel.py::test_triton_matches_main_packed`.
Run from the SGLang checkout:

```bash
python3 -m pytest -v python/sglang/test/kernels/deepseek_v4/test_remnant_unpack_kernel.py -k main_packed
```

Acceptance: exact selection masks, complete Packed buffers (including untouched
rows), BF16 gather output, active Native record bytes/scales, temporary indices,
and stock FlashMLA output/LSE on the same FP32 compressor inputs and weights.
Cases cover prefill/decode stores, zeros and ties, nontrivial destination mapping,
duplicate selections, partial lengths, and two reconstruction widths. Historical
unsafe inputs (out-of-bounds indices, padded prefill plans, strided selections,
workspace capacity differing from selected width) are excluded from historical
parity; existing independent-reference tests cover corrected current behavior.
No historical implementation changes or tolerance relaxation are permitted.
Stop on any mismatch and record the failing stage before interpreting timings.
This is numerical compatibility, not a serving performance gate.

Validation on 2026-10-13: Modal run `main-packed-parity-20261013-aligned`
passed all four historical parity cases plus compilation reuse on one H100,
with zero failures or skips and no model load. CPU preflight passed 38 tests
plus 18 subtests. Logs, JUnit and source/binary manifests are retained under
`remnant-stage2a-results:/non-model/main-packed-parity-20261013-aligned/`.
The earlier CPU snapshot failure (missing nested CUTLASS) and attention fixture
failure (unaligned SWA index width) are recorded separately; neither changed
production code or the numerical acceptance rules.
