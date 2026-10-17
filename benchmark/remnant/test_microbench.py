"""Three-mode benchmark contracts and independent GPU validity checks.

Author: Winston Cai. CPU checks import no CUDA modules; GPU skips are explicit.
"""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

import pytest


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


ROOT = Path(__file__).resolve().parents[2]
cases = load_module(
    "_benchmark_case_contract", ROOT / "python/sglang/benchmark/remnant/cases.py"
)


def test_established_matrix_and_profile_coverage():
    matrix = cases.profile_matrix()
    assert len(matrix) == 10
    assert len(set(matrix)) == 10
    assert len([x for x in matrix if x[0] == "decode"]) == 8
    assert {x[1] for x in matrix if x[0] == "decode"} == {
        f"h{h}_b{b}_k{k}" for h in (64, 128) for b in (8, 16) for k in (512, 317)
    }
    assert cases.MODES == ("native", "triton", "kernel")


def test_store_matrix_layouts_are_valid_and_reproducible():
    specs = cases.store_cases()
    assert len(specs) == 25 and len({c.name for c in specs}) == 25
    for case in specs:
        seq, ragged, loc = cases.store_layout(case, cases.SEED)
        assert sorted(ragged) == list(range(case.rows))
        assert (seq, ragged, loc) == cases.store_layout(case, cases.SEED)
        writes = [
            loc[i if case.phase == "decode" else ragged[i]]
            for i, s in enumerate(seq)
            if s >= 0 and (case.phase == "prefill" or s % 4 == 0)
        ]
        writes = [x for x in writes if x >= 0]
        assert len(set(writes)) == len(writes)
        assert all(x < case.rows + 7 for x in writes)
    assert (
        cases.store_layout(cases.StoreCase("decode", 32, "no_boundary"), cases.SEED)[0]
        == [5] * 32
    )


@pytest.mark.parametrize("value", ["", "0", "-1", "8,8", "x", "8,"])
def test_invalid_matrix_values(value):
    with pytest.raises(ValueError):
        cases.positive_ints(value, "batch")


@pytest.mark.parametrize("value", ["", "packed", "adapter", "native,native"])
def test_invalid_modes(value):
    with pytest.raises(ValueError):
        cases.parse_modes(value)


def test_paired_ratios_and_round_percentile():
    result = cases.paired_ratios(
        {"native": [1.0, 2.0], "triton": [2.0, 4.0], "kernel": [1.0, 2.0]}
    )
    assert result["kernel_over_triton"]["mean"] == 0.5
    assert result["triton_over_native"]["samples"] == [2.0, 2.0]
    assert cases.summary([1.0, 2.0, 3.0])["p95"] == pytest.approx(2.9)
    with pytest.raises(ValueError):
        cases.paired_ratios({"native": [1.0, 2.0], "kernel": [1.0]})


@pytest.fixture
def benchmark():
    import torch

    if not torch.cuda.is_available():
        pytest.skip("H100 numerical/graph validation requires CUDA")
    library = os.environ.get("REMNANT_NATIVE_STORE_LIBRARY")
    if not library:
        pytest.fail("CPU-built Native library required; no automatic GPU build")
    module = load_module(
        "_three_mode_benchmark", ROOT / "benchmark/remnant/microbench.py"
    )
    from tvm_ffi import load_module as load_native

    module.NATIVE_STORE_MODULE = load_native(library)
    yield module
    module.configure("native")


@pytest.mark.parametrize(
    "heads,batch,length",
    [(h, b, k) for h in (64, 128) for b in (8, 16) for k in (512, 317)],
)
def test_gpu_decode_outputs(heads, batch, length, benchmark):
    import torch

    with torch.inference_mode():
        ops, identity = benchmark.decode_operations(
            benchmark.DecodeCase(heads, batch, length), cases.SEED
        )
        assert identity["native_baseline"] == "unpruned production Native"
        for op in ops:
            op.validate()


@pytest.mark.parametrize("case", cases.store_cases(), ids=lambda c: c.name)
def test_gpu_store_contract(case, benchmark):
    import torch

    with torch.inference_mode():
        ops, _ = benchmark.store_operations(
            benchmark.StoreCase(case.phase, case.rows, case.layout), cases.SEED
        )
        for op in ops:
            op.validate()


def test_gpu_graph_replay_changes_queries_lengths_and_selections(benchmark):
    import torch

    with torch.inference_mode():
        ops, _ = benchmark.decode_operations(
            benchmark.DecodeCase(64, 8, 512), cases.SEED
        )
        captured = [benchmark.capture(op, 3) for op in ops]
        # Shared tensors retain their addresses. Duplicate selections, invalid
        # entries and partial lengths exercise reconstruction and direct reads.
        _, q, _, _, _, _, _, _, physical, raw, lengths = ops[0].owned
        q.mul_(0.75)
        physical[:, :, 1] = physical[:, :, 0]
        raw[:, :, 1] = raw[:, :, 0]
        physical[:, :, 2] = -1
        raw[:, :, 2] = -1
        lengths.fill_(317)
        for op, (graph, output) in zip(ops, captured):
            benchmark.configure(op.mode)
            graph.replay()
            torch.cuda.synchronize()
            # Validate actual captured outputs against fresh independent
            # references, rather than letting an eager call hide graph errors.
            _, swa_storage = op.owned[3:5]
            swa = benchmark.decode_native(swa_storage).reshape(8, 64, 512)
            rows = benchmark.decode_native(op.owned[3])[:4096]
            if op.mode != "native":
                cache = op.owned[2][op.mode]
                decoded = benchmark.unpack_rows_ref(
                    *(t.flatten(0, 1)[:4096] for t in cache)
                )
                positions = (torch.arange(4096, device="cuda") * 3 + 17) * 4
                rows = benchmark.rotate_tail(decoded, positions, op.owned[7])
            expected = benchmark.attention_reference(q, swa, rows, physical, lengths)
            benchmark.check_output(output, expected, f"graph/{op.mode}")


def test_cli_defaults_and_supplemental_matrix():
    args = cases.parser().parse_args([])
    assert (
        args.seed,
        args.warmup,
        args.rounds,
        args.repeats,
        args.prefill_repeats,
    ) == (20260919, 10, 30, 100, 20)
    assert len(cases.selected_cases(args)) == 33
    args = cases.parser().parse_args(["--suite", "decode", "--batches", "32"])
    assert len(cases.selected_cases(args)) == 4
    assert all(c.batch == 32 for c in cases.selected_cases(args))
    for argv in (
        ["--case", "missing"],
        ["--profile-mode", "triton"],
        ["--rounds", "1"],
        ["--repeats", "0"],
    ):
        with pytest.raises(ValueError):
            cases.selected_cases(cases.parser().parse_args(argv))


def test_case_seed_independent_of_enumeration_and_matrix_identity():
    a, b = cases.store_cases()[:2]
    assert cases.case_seed(a, cases.SEED) != cases.case_seed(b, cases.SEED)
    assert cases.case_seed(a, cases.SEED) != cases.case_seed(a, cases.SEED + 1)
    assert cases.matrix_digest([a, b]) != cases.matrix_digest([b, a])


def test_json_and_flat_csv_schema(tmp_path):
    import csv
    import json

    result = {
        "protocol": cases.PROTOCOL,
        "schema_version": 2,
        "environment": {"device": "fixture"},
        "source": {"binaries": {"test.so": "hash"}},
        "cases": [
            {
                "suite": "decode",
                "name": "example",
                "case": {"batch": 8},
                "common_inputs_sha256": "inputs",
                "execution": "cuda_graph",
                "valid": True,
                "paths": {"native": cases.summary([1.0, 2.0])},
                "paired_ratios": {},
            }
        ],
    }
    base = tmp_path / "timing"
    cases.write_results(result, base)
    assert json.loads(base.with_suffix(".json").read_text()) == result
    row = list(csv.DictReader(base.with_suffix(".csv").open()))[0]
    assert json.loads(row["samples_ms"]) == [1.0, 2.0]
    assert json.loads(row["source"]) == result["source"]
    assert row["protocol"] == cases.PROTOCOL and row["input_sha256"] == "inputs"


def test_gpu_store_graph_replay_changes_inputs_and_destinations(benchmark):
    import torch

    with torch.inference_mode():
        ops, _ = benchmark.store_operations(
            benchmark.StoreCase("decode", 25), cases.SEED
        )
        captured = [benchmark.capture(op, 3) for op in ops]
        (
            x,
            weight,
            metadata,
            native_plan,
            native_cache,
            caches_,
            freqs,
            loc,
            native_loc,
        ) = ops[0].owned
        x.mul_(0.37).add_(0.013)
        weight.mul_(1.2)
        loc.copy_(loc.roll(3))
        native_loc.copy_(loc)
        for op, (graph, output) in zip(ops, captured):
            benchmark.configure(op.mode)
            graph.replay()
        torch.cuda.synchronize()
        ids = loc.long()
        expected = benchmark.pack_rows_ref(
            x, benchmark.topmag_keep_mask(x, 0.5), weight, benchmark.EPS
        )
        for mode in ("triton", "kernel"):
            actual = tuple(
                benchmark.indexed_rows(t.flatten(0, 1), ids) for t in caches_[mode]
            )
            assert torch.equal(actual[1], expected[1])
            torch.testing.assert_close(
                benchmark.unpack_rows_ref(*actual),
                benchmark.unpack_rows_ref(*expected),
                atol=benchmark.ATOL,
                rtol=benchmark.RTOL,
            )
        for a, b in zip(caches_["triton"], caches_["kernel"]):
            assert torch.equal(a, b)
        expected_native = benchmark.native_reference(
            x, weight, torch.zeros(25, device="cuda", dtype=torch.long), freqs
        )
        torch.testing.assert_close(
            benchmark.decode_native(native_cache)[ids],
            expected_native,
            atol=benchmark.ATOL,
            rtol=benchmark.RTOL,
        )


def test_quantization_reference_accepts_rounding_and_rejects_wrong_codes():
    import torch

    reference = load_module(
        "_quantization_contract",
        ROOT / "python/sglang/benchmark/remnant/quantization.py",
    )
    x, weight = torch.ones(1, 512), torch.ones(512)
    values, codes = torch.ones(1, 512), torch.full((1, 8), 119, dtype=torch.uint8)
    reference.check_fp8_storage(values, codes, x, weight, 1e-6, "cpu")
    for field in ("value", "scale"):
        bad_values, bad_codes = values.clone(), codes.clone()
        if field == "value":
            bad_values[0, 0] = 1.125
        else:
            bad_codes[0, 0] = 120
        with pytest.raises(AssertionError):
            reference.check_fp8_storage(bad_values, bad_codes, x, weight, 1e-6, "cpu")
