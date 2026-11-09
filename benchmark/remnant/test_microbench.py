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
    assert len(matrix) == 12
    assert len(set(matrix)) == 12
    assert len([x for x in matrix if x[0] == "decode"]) == 10
    assert {x[1] for x in matrix if x[0] == "decode"} == {
        f"h{h}_b{b}_k512" for h in (64, 128) for b in (8, 16, 24, 28, 32)
    }
    defaults = cases.parser().parse_args(["--suite", "decode"])
    assert [("decode", c.name) for c in cases.selected_cases(defaults)] == [
        row for row in matrix if row[0] == "decode"
    ]
    assert cases.MODES == ("native", "triton", "kernel")


def test_prefill_fixture_matches_combine_keyword_contract():
    """Catch upstream required-argument additions before paying for GPU setup."""
    import ast

    tree = ast.parse((ROOT / "benchmark/remnant/microbench.py").read_text())
    call = next(
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.Call)
        and isinstance(n.func, ast.Name)
        and n.func.id == "combine_topk_swa_indices"
    )
    api = ast.parse(
        (
            ROOT / "python/sglang/srt/layers/attention/dsv4/sparse_prefill_utils.py"
        ).read_text()
    )
    function = next(
        n
        for n in api.body
        if isinstance(n, ast.FunctionDef) and n.name == "combine_topk_swa_indices"
    )
    required = function.args.args[
        : len(function.args.args) - len(function.args.defaults)
    ]
    assert {arg.arg for arg in required} <= {keyword.arg for keyword in call.keywords}


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


@pytest.mark.parametrize("keep_count", range(64, 512, 64))
def test_survivor_counts_select_only_supported_paths(keep_count):
    args = cases.parser().parse_args(["--keep-count", str(keep_count)])
    assert len(cases.selected_cases(args)) == 35
    expected = cases.MODES if keep_count == 256 else ("native", "triton")
    assert cases.benchmark_modes(args.keep_count, args.modes) == expected
    matrix = cases.selected_profile_matrix(keep_count=keep_count)
    assert len(matrix) == 12 * len(expected)
    assert {mode for _, _, mode in matrix} == set(expected)
    assert cases.benchmark_modes(keep_count, "native") == ("native",)
    # Survivor count never changes the raw-fixture seed or geometric matrix.
    baseline = cases.selected_cases(cases.parser().parse_args([]))
    assert cases.selected_cases(args) == baseline
    assert cases.matrix_digest(cases.selected_cases(args)) == cases.matrix_digest(
        baseline
    )


@pytest.mark.parametrize("keep_count", [0, 1, 63, 65, 511, 512, 576, -64, True, 192.0])
def test_invalid_survivor_counts(keep_count):
    with pytest.raises(ValueError, match="keep-count"):
        cases.benchmark_modes(keep_count)


def test_unsupported_kernel_rejected_before_fixture_setup():
    for keep_count in cases.KEEP_COUNTS:
        if keep_count == 256:
            continue
        for modes in ("kernel", "native,triton,kernel"):
            args = cases.parser().parse_args(
                ["--keep-count", str(keep_count), "--modes", modes]
            )
            with pytest.raises(ValueError, match="Kernel requires"):
                cases.selected_cases(args)
        with pytest.raises(ValueError, match="profile-mode"):
            cases.selected_cases(
                cases.parser().parse_args(
                    [
                        "--keep-count",
                        str(keep_count),
                        "--case",
                        "h64_b8_k512",
                        "--profile-mode",
                        "kernel",
                    ]
                )
            )


@pytest.mark.parametrize("keep_count", cases.KEEP_COUNTS)
def test_benchmark_configure_uses_real_runtime_survivor_profile(keep_count):
    import ast

    # Execute the real lightweight configuration function against runtime config,
    # without importing CUDA-only attention/FFI dependencies on the CPU host.
    runtime = load_module(
        "_microbench_runtime_config", ROOT / "python/sglang/srt/remnant/config.py"
    )
    tree = ast.parse((ROOT / "benchmark/remnant/microbench.py").read_text())
    function = next(
        n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "configure"
    )
    namespace = {"remnant": runtime, "benchmark_modes": cases.benchmark_modes}
    exec(
        compile(ast.Module(body=[function], type_ignores=[]), "microbench.py", "exec"),
        namespace,
    )
    configure = namespace["configure"]
    assert runtime.VALID_KEEP_COUNTS == cases.KEEP_COUNTS
    for mode in cases.benchmark_modes(keep_count):
        configure(mode, keep_count)
        assert runtime.cache_format() == ("native" if mode == "native" else "remnant")
        assert runtime.packed_profile().default_count == (
            256 if mode == "native" else keep_count
        )
        assert runtime.cache_store_implementation() == (
            "kernel" if mode == "native" else mode
        )
    configure("native", keep_count)
    assert runtime.packed_profile().default_count == 256


def test_paired_ratios_and_round_percentile():
    result = cases.paired_ratios(
        {"native": [1.0, 2.0], "triton": [2.0, 4.0], "kernel": [1.0, 2.0]}
    )
    assert result["kernel_over_triton"]["mean"] == 0.5
    assert result["triton_over_native"]["samples"] == [2.0, 2.0]
    assert cases.summary([1.0, 2.0, 3.0])["p95"] == pytest.approx(2.9)
    with pytest.raises(ValueError):
        cases.paired_ratios({"native": [1.0, 2.0], "kernel": [1.0]})


def test_prefill_read_matrix_shared_seeds_and_explicit_selection():
    args = cases.parser().parse_args(["--suite", "prefill-read"])
    matrix = cases.selected_cases(args)
    assert len(matrix) == 60
    assert {c.history_length for c in matrix} == {143360}
    assert {c.query_length for c in matrix} == {128, 512, 1536, 4096, 8192}
    assert {c.heads for c in matrix} == {64, 128}
    assert {c.stage for c in matrix} == set(cases.PREFILL_STAGES)
    assert {c.cache_policy for c in matrix} == {"warm", "rotating"}
    fixtures = {}
    for case in matrix:
        seed = cases.case_seed(case, cases.SEED)
        assert fixtures.setdefault(case.fixture_name, seed) == seed
        assert cases.case_suite(case) == "prefill-read"
    assert len(fixtures) == 10
    assert len(cases.selected_profile_matrix(suite="prefill-read")) == 180
    # Keep original cases/names/seeds and explicit historical reproduction.
    legacy = cases.parser().parse_args(
        ["--suite", "prefill-read", "--query-chunks", "512,1536,8192"]
    )
    original = cases.selected_cases(legacy)
    assert len(original) == 36
    assert original == [c for c in matrix if c.query_length in (512, 1536, 8192)]
    selected = cases.parser().parse_args(
        ["--suite", "prefill-read", "--case", "prefill_h64_s143360_q128_warm_attention"]
    )
    assert [c.name for c in cases.selected_cases(selected)] == [selected.case]
    assert args.prefill_reader == "split"
    fused_args = cases.parser().parse_args(
        ["--suite", "prefill-read", "--prefill-reader", "fused"]
    )
    assert cases.selected_cases(fused_args) == matrix
    for argv in (
        ["--suite", "prefill-read", "--modes", "native,triton"],
        ["--suite", "prefill-read", "--keep-count", "192"],
        ["--profile-path", "packed192"],
        ["--suite", "prefill-read", "--histories", "1024"],
        ["--prefill-reader", "fused"],
        ["--suite", "decode", "--prefill-reader", "fused"],
    ):
        with pytest.raises(ValueError):
            cases.selected_cases(cases.parser().parse_args(argv))


def test_prefill_three_way_ratios_and_timing_coverage():
    import copy
    import hashlib
    from dataclasses import asdict

    samples = {"native": [2.0, 4.0], "packed192": [1.0, 2.0], "packed256": [1.5, 3.0]}
    ratios = cases.paired_ratios(samples)
    assert ratios["packed192_over_native"]["samples"] == [0.5, 0.5]
    assert ratios["packed192_over_packed256"]["mean"] == pytest.approx(2 / 3)
    payload = {
        "protocol": cases.PREFILL_PROTOCOL,
        "schema_version": 2,
        "status": "passed",
        "cases": [
            {
                "suite": "prefill-read",
                "name": c.name,
                "valid": True,
                "case": asdict(c),
                "common_inputs_sha256": hashlib.sha256(
                    c.fixture_name.encode()
                ).hexdigest(),
                "path_layouts": cases.prefill_path_layouts(),
                "paths": {m: {"samples": [1.0] * 30} for m in cases.PREFILL_VARIANTS},
            }
            for c in cases.prefill_read_cases()
        ],
    }
    cases.validate_timing(payload, suite="prefill-read")
    candidate = copy.deepcopy(payload)
    candidate["config"] = {"prefill_reader": "fused"}
    for record in candidate["cases"]:
        record["prefill_reader"] = "fused"
    cases.validate_timing(candidate, suite="prefill-read", prefill_reader="fused")
    with pytest.raises(RuntimeError):
        cases.validate_timing(candidate, suite="prefill-read")
    candidate["cases"][0]["prefill_reader"] = "split"
    with pytest.raises(RuntimeError, match="reader mismatch"):
        cases.validate_timing(candidate, suite="prefill-read", prefill_reader="fused")
    for change in (
        lambda p: p["cases"].pop(),
        lambda p: p.update(
            cases=[
                c for c in p["cases"] if c["case"]["query_length"] in (512, 1536, 8192)
            ]
        ),
        lambda p: p["cases"][0]["paths"].pop("packed256"),
        lambda p: p["cases"][0]["path_layouts"]["packed192"].update(record_bytes=328),
        lambda p: p.update(protocol=cases.PROTOCOL),
        lambda p: p["cases"][0].update(common_inputs_sha256="a" * 64),
    ):
        bad = copy.deepcopy(payload)
        change(bad)
        with pytest.raises(RuntimeError):
            cases.validate_timing(bad, suite="prefill-read")


def test_profile_application_replay_cannot_overwrite_another_result():
    import copy

    config = {
        "suite": "prefill-read",
        "profile_path": "packed192",
        "profile_replay_id": "owner:case",
    }
    payload = {
        "config": config.copy(),
        "protocol": cases.PREFILL_PROTOCOL,
        "status": "passed",
        "cases": [{"valid": True, "path": "packed192"}],
    }
    cases.verify_profile_replay(payload, config)
    for change in (
        lambda p: p["config"].update(profile_replay_id="other"),
        lambda p: p.update(status="failed"),
        lambda p: p["cases"][0].update(path="native"),
    ):
        bad = copy.deepcopy(payload)
        change(bad)
        with pytest.raises(FileExistsError):
            cases.verify_profile_replay(bad, config)


def test_prefill_profile_batches_preserve_case_selection_and_replay_identity():
    import copy

    catalog = cases.prefill_read_cases()
    names = [c.name for c in catalog[:2]]
    args = cases.parser().parse_args(
        ["--suite", "prefill-read", "--profile-path", "packed192"]
    )
    assert cases.selected_cases(args) == catalog
    args.case = ",".join(reversed(names))
    assert cases.selected_cases(args) == catalog[:2]
    for invalid in (names[0] + ",", names[0] + "," + names[0], "missing"):
        args.case = invalid
        with pytest.raises(ValueError, match="unknown or ambiguous"):
            cases.selected_cases(args)
    config = {
        "suite": "prefill-read",
        "profile_path": "packed192",
        "profile_replay_id": "owner:batch",
    }
    payload = {
        "config": config.copy(),
        "protocol": cases.PREFILL_PROTOCOL,
        "status": "passed",
        "cases": [{"name": n, "valid": True, "path": "packed192"} for n in names],
    }
    cases.verify_profile_replay(payload, config)
    for change in (
        lambda p: p["cases"][1].update(name=names[0]),
        lambda p: p["cases"][1].update(valid=False),
        lambda p: p["cases"].clear(),
    ):
        bad = copy.deepcopy(payload)
        change(bad)
        with pytest.raises(FileExistsError):
            cases.verify_profile_replay(bad, config)


@pytest.mark.parametrize("fail", [False, True])
def test_prefill_profile_nvtx_wraps_only_warmed_operation_and_cleans_up(fail):
    import ast
    from contextlib import contextmanager, nullcontext
    from types import SimpleNamespace

    function = next(
        n
        for n in ast.parse((ROOT / "benchmark/remnant/microbench.py").read_text()).body
        if isinstance(n, ast.FunctionDef) and n.name == "profile_case"
    )
    state = {"profiler": False, "nvtx": None}
    observations = []

    def start():
        state["profiler"] = True
        return (0,)

    def stop():
        state["profiler"] = False
        return (0,)

    @contextmanager
    def nvtx(name):
        state["nvtx"] = name
        try:
            yield
        finally:
            state["nvtx"] = None

    def call():
        observations.append(state.copy())
        if fail and state["profiler"]:
            raise RuntimeError("operation failed")

    cuda = SimpleNamespace(
        cudart=lambda: SimpleNamespace(cudaProfilerStart=start, cudaProfilerStop=stop),
        synchronize=lambda: None,
        nvtx=SimpleNamespace(range=nvtx),
    )
    namespace = {
        "torch": SimpleNamespace(cuda=cuda),
        "Operation": object,
        "PrefillReadCase": cases.PrefillReadCase,
        "configure": lambda *a: None,
        "nullcontext": nullcontext,
    }
    exec(
        compile(ast.Module(body=[function], type_ignores=[]), "microbench.py", "exec"),
        namespace,
    )
    case = cases.PrefillReadCase(64, 4100, 8, "complete")
    op = SimpleNamespace(
        mode="triton",
        key="packed192",
        keep_count=192,
        validate=lambda: None,
        call=call,
        expected_kernels=(),
    )
    if fail:
        with pytest.raises(RuntimeError, match="operation failed"):
            namespace["profile_case"](case, op, SimpleNamespace(warmup=10))
    else:
        result = namespace["profile_case"](case, op, SimpleNamespace(warmup=10))
        assert result["capture_range"] == f"remnant_{case.name}_packed192"
        assert result["minimum_launches"]["elementwise.*MulFunctor"] == 2
    assert observations[:10] == [{"profiler": False, "nvtx": None}] * 10
    assert observations[10:] == [
        {"profiler": True, "nvtx": f"remnant_{case.name}_packed192"}
    ]
    assert state == {"profiler": False, "nvtx": None}


def test_prefill_timing_preserves_variant_configuration_and_pairing():
    import ast
    import random
    from types import SimpleNamespace
    from unittest.mock import Mock

    function = next(
        n
        for n in ast.parse((ROOT / "benchmark/remnant/microbench.py").read_text()).body
        if isinstance(n, ast.FunctionDef) and n.name == "time_case"
    )
    configure = Mock()
    ops = [
        SimpleNamespace(
            key=key, mode=mode, keep_count=count, call=lambda v=v: v, validate=Mock()
        )
        for key, mode, count, v in (
            ("native", "native", 256, 1.0),
            ("packed192", "triton", 192, 2.0),
            ("packed256", "triton", 256, 3.0),
        )
    ]
    for op in ops:

        def checked_call(op=op):
            assert configure.call_args.args == (op.mode, op.keep_count)
            return {"native": 1.0, "packed192": 2.0, "packed256": 3.0}[op.key]

        op.call = checked_call
    namespace = {
        "Operation": object,
        "DecodeCase": cases.DecodeCase,
        "configure": configure,
        "measure": lambda call, repeats: call(),
        "paired_samples": shared_timing.paired_samples,
        "case_seed": cases.case_seed,
        "random": random,
        "summary": cases.summary,
        "paired_ratios": cases.paired_ratios,
    }
    exec(
        compile(ast.Module(body=[function], type_ignores=[]), "microbench.py", "exec"),
        namespace,
    )
    args = SimpleNamespace(
        repeats=100, prefill_repeats=20, warmup=0, rounds=2, seed=cases.SEED
    )
    result = namespace["time_case"](cases.PrefillReadCase(64, 4096, 8), ops, args)
    assert result["paths"]["packed192"]["samples"] == [2.0, 2.0]
    assert result["paired_ratios"]["packed192_over_native"]["mean"] == 2.0
    assert ("triton", 192) in [c.args for c in configure.call_args_list]
    assert all(op.validate.call_count == 2 for op in ops)


def test_prefill_traffic_requires_reads_writes_hits_and_launch_coverage():
    import csv
    import io

    metrics = (
        "gpu__time_duration.sum",
        "dram__bytes_read.sum",
        "dram__bytes_write.sum",
        "lts__t_sectors_op_read.sum",
        "lts__t_sectors_op_write.sum",
        "lts__t_sector_hit_rate.pct",
    )

    def export(omit=None, launches=2, omit_launch=None):
        output = io.StringIO()
        writer = csv.writer(output)
        writer.writerow(
            ["ID", "Kernel Name", "Metric Name", "Metric Unit", "Metric Value"]
        )
        for launch in range(launches):
            for metric in metrics:
                if metric != omit or (
                    omit_launch is not None and launch != omit_launch
                ):
                    writer.writerow(
                        [str(launch), "dequantize_k_cache_paged", metric, "unit", "1"]
                    )
        return output.getvalue()

    result = cases.profile_summary(
        export(), ["dequantize"], traffic_only=True, minimum_launches={"dequantize": 2}
    )
    assert len(result["metrics"]) == 12
    for metric in metrics:
        with pytest.raises(RuntimeError, match="coverage"):
            cases.profile_summary(
                export(omit=metric), ["dequantize"], traffic_only=True
            )
        with pytest.raises(RuntimeError, match="per-launch metric coverage"):
            cases.profile_summary(
                export(omit=metric, omit_launch=1), ["dequantize"], traffic_only=True
            )
    with pytest.raises(RuntimeError, match="launch coverage"):
        cases.profile_summary(
            export(launches=1),
            ["dequantize"],
            traffic_only=True,
            minimum_launches={"dequantize": 2},
        )


def test_native_repage_preserves_values_scales_and_zero_padding():
    import ast

    import torch

    tree = ast.parse((ROOT / "benchmark/remnant/microbench.py").read_text())
    function = next(
        n
        for n in tree.body
        if isinstance(n, ast.FunctionDef) and n.name == "repage_native"
    )
    namespace = {"torch": torch, "PAGE_SIZE": 64}
    exec(
        compile(ast.Module(body=[function], type_ignores=[]), "microbench.py", "exec"),
        namespace,
    )
    source = torch.randint(0, 256, (3, 37632), dtype=torch.uint8)
    for page_size in (128, 256):
        target = namespace["repage_native"](source, 130, page_size)
        for row in range(130):
            sp, sr = divmod(row, 64)
            tp, tr = divmod(row, page_size)
            assert torch.equal(
                source[sp, sr * 576 : (sr + 1) * 576],
                target[tp, tr * 576 : (tr + 1) * 576],
            )
            assert torch.equal(
                source[sp, 64 * 576 + sr * 8 : 64 * 576 + (sr + 1) * 8],
                target[tp, page_size * 576 + tr * 8 : page_size * 576 + (tr + 1) * 8],
            )
        for row in range(130, len(target) * page_size):
            page, offset = divmod(row, page_size)
            assert not bool(target[page, offset * 576 : (offset + 1) * 576].any())


@pytest.mark.parametrize("heads", [64, 128])
@pytest.mark.parametrize("stage", cases.PREFILL_STAGES)
@pytest.mark.parametrize("policy", cases.PREFILL_POLICIES)
@pytest.mark.parametrize("fused_rope", [False, True], ids=["split", "fused"])
def test_gpu_prefill_read_complete_fixture(heads, stage, policy, fused_rope, benchmark):
    import torch

    with torch.inference_mode():
        operations, identity = benchmark.prefill_read_operations(
            benchmark.PrefillReadCase(heads, 4100, 8, stage, policy),
            cases.SEED,
            fused_rope=fused_rope,
        )
        assert [op.key for op in operations] == list(cases.PREFILL_VARIANTS)
        assert identity["path_layouts"] == cases.prefill_path_layouts()
        assert identity["prefill_reader"] == ("fused" if fused_rope else "split")
        if stage != "attention":
            for op in operations[1:]:
                assert ("unpack_gather_bf16_rope" in op.expected_kernels) == fused_rope
                assert ("rope_tail_complex" in op.expected_kernels) != fused_rope
        assert identity["buffer_banks"] == (4 if policy == "rotating" else 1)
        assert len({op.owned[0].data_ptr() for op in operations}) == 1
        operations[0].owned[0].mul_(0.9)
        indices, lengths = operations[0].owned[4:6]
        indices[0, 0, 1] = indices[0, 0, 0]
        indices[0, 0, 2] = -1
        lengths[1] -= 3
        for op in operations:
            benchmark.configure(op.mode, op.keep_count)
            op.call()
            op.validate()


@pytest.mark.parametrize("valid", [False, True])
def test_fused_reader_preserves_bf16_rounding_and_rotation_on_cpu(valid):
    """Execute the actual Triton arithmetic body with CPU memory primitives.

    This checks rounding/sign/position/store semantics against independent
    Torch complex arithmetic, not GPU compilation, layout or synchronization.
    The H100 tests separately cover the real gather, compilation and graphs.
    """
    import ast
    from types import SimpleNamespace

    import torch

    class Pointer:
        def __init__(self, data, offsets=0):
            self.data, self.offsets = data, offsets

        def __add__(self, offsets):
            return Pointer(self.data, self.offsets + offsets)

    def load(pointer, mask, other):
        result = torch.full(mask.shape, other, dtype=torch.float32)
        result[mask] = pointer.data[pointer.offsets[mask]]
        return result

    def store(pointer, value, mask):
        pointer.data[pointer.offsets[mask]] = value[mask].to(pointer.data.dtype)

    # Deliberately non-BF16 inputs make a missing intermediate rounding visible.
    value = torch.linspace(-1.137, 1.377, 512)
    rounded = value.to(torch.bfloat16).float()
    angles = torch.linspace(0.13, 1.17, 16 * 32).reshape(16, 32)
    freqs = torch.polar(torch.ones_like(angles), angles)
    raw = torch.tensor(2 if valid else 2**31 - 1, dtype=torch.int64)
    output = torch.full((3 * 512,), -91, dtype=torch.bfloat16)
    tl = SimpleNamespace(
        constexpr=int,
        bfloat16=torch.bfloat16,
        float32=torch.float32,
        arange=torch.arange,
        program_id=lambda axis: 1,
        gather=lambda x, indices, axis: torch.gather(x, axis, indices.long()),
        load=load,
        store=store,
        where=torch.where,
    )
    path = ROOT / "python/sglang/srt/remnant/triton/kernels.py"
    function = next(
        node
        for node in ast.parse(path.read_text()).body
        if isinstance(node, ast.FunctionDef)
        and node.name == "_unpack_gather_bf16_rope_kernel"
    )
    function.decorator_list = []
    namespace = {
        "tl": tl,
        "_unpack_gather_row": lambda *args: (
            value if valid else torch.zeros_like(value),
            raw,
            torch.tensor(valid),
        ),
    }
    exec(
        compile(ast.Module(body=[function], type_ignores=[]), str(path), "exec"),
        namespace,
    )
    args = {arg.arg: None for arg in function.args.args}
    args.update(
        freq_ptr=Pointer(torch.view_as_real(freqs).flatten()),
        output_ptr=Pointer(output),
        HEAD_DIM=512,
        NOPE_DIM=448,
        ROPE_PAIRS=32,
        BLOCK_D=512,
    )
    namespace[function.name](**args)
    expected = torch.zeros(512, dtype=torch.bfloat16)
    if valid:
        expected.copy_(rounded)
        tail = torch.view_as_complex(rounded[448:].reshape(32, 2)) * freqs[8]
        expected[448:] = torch.view_as_real(tail).flatten().to(torch.bfloat16)
        unrounded = torch.view_as_complex(value[448:].reshape(32, 2)) * freqs[8]
        assert not torch.equal(
            expected[448:], torch.view_as_real(unrounded).flatten().to(torch.bfloat16)
        ), "fixture must detect removal of the pre-RoPE BF16 round trip"
    assert torch.equal(output[512:1024].view(torch.int16), expected.view(torch.int16))
    assert torch.all(output[:512] == -91) and torch.all(output[1024:] == -91)


def test_prefill_workspace_dtype_and_independent_constant_attention_on_cpu():
    import ast
    import math
    from types import SimpleNamespace
    from unittest.mock import patch

    import torch

    tree = ast.parse((ROOT / "benchmark/remnant/microbench.py").read_text())
    factory = next(
        n for n in tree.body if getattr(n, "name", None) == "prefill_read_operations"
    )
    validate = next(
        n
        for n in ast.walk(factory)
        if isinstance(n, ast.FunctionDef) and n.name == "validate"
    )
    # Two all-one KV rows, queries producing unit scores, and a zero sink:
    # SGL's public max/LSE outputs are log2(e) and log2(2e), while output
    # coordinates are 2e/(2e+1). Nonzero logits expose a log-base mismatch.
    # The reference stays FP32; the production workspace/output are BF16.
    workspaces = [torch.ones((2, 1, 512), dtype=torch.bfloat16)]
    q = torch.zeros((1, 2, 512))
    q[..., 0] = math.sqrt(512)
    cursor = [7]
    namespace = dict(
        torch=torch,
        math=math,
        ATOL=2e-2,
        RTOL=2e-2,
        reconstruct=lambda bank: workspaces[bank],
        attention=lambda bank: (
            torch.full(
                (1, 2, 512), 2 * math.e / (2 * math.e + 1), dtype=torch.bfloat16
            ),
            torch.full((1, 2), 1 / math.log(2)),
            torch.full((1, 2), (1 + math.log(2)) / math.log(2)),
        ),
        configure=lambda *args: None,
        mode="native",
        keep_count=256,
        expected_workspace=torch.ones((2, 512), dtype=torch.float32),
        workspaces=workspaces,
        split_workspace=None,
        sources=[],
        cursor=cursor,
        label="native",
        banks=1,
        case=SimpleNamespace(name="constant-prefill"),
        q=q,
        indices=torch.tensor([[[0, 1]]]),
        lengths=torch.tensor([2]),
        sample_ids=torch.tensor([0]),
        sink=torch.zeros(2),
    )
    exec(
        compile(ast.Module(body=[validate], type_ignores=[]), "microbench.py", "exec"),
        namespace,
    )
    arange = torch.arange
    with patch.object(
        torch,
        "arange",
        side_effect=lambda *a, **kw: arange(*a, **{**kw, "device": "cpu"}),
    ):
        namespace["validate"]()
        assert cursor == [0]
        workspaces[0][0, 0, 0] = 0
        with pytest.raises(AssertionError):
            namespace["validate"]()
        workspaces[0][0, 0, 0] = 1
        workspaces[0] = workspaces[0].float()
        with pytest.raises(RuntimeError, match="expected BF16 workspace"):
            namespace["validate"]()


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


@pytest.mark.parametrize("keep_count", [192, 256])
@pytest.mark.parametrize(
    "heads,batch,length",
    [
        (case.heads, case.batch, case.topk_length)
        for case in cases.selected_cases(
            cases.parser().parse_args(["--suite", "decode"])
        )
    ],
)
def test_gpu_decode_outputs(heads, batch, length, keep_count, benchmark):
    import torch

    with torch.inference_mode():
        ops, identity = benchmark.decode_operations(
            benchmark.DecodeCase(heads, batch, length), cases.SEED, keep_count
        )
        assert identity["native_baseline"] == "unpruned production Native"
        for op in ops:
            op.validate()


@pytest.mark.parametrize("keep_count", [192, 256])
@pytest.mark.parametrize("case", cases.store_cases(), ids=lambda c: c.name)
def test_gpu_store_contract(case, keep_count, benchmark):
    import torch

    with torch.inference_mode():
        ops, _ = benchmark.store_operations(
            benchmark.StoreCase(case.phase, case.rows, case.layout),
            cases.SEED,
            keep_count,
        )
        for op in ops:
            op.validate()


def test_gpu_all_survivor_counts_preserve_native_and_raw_inputs(benchmark):
    import torch

    identity = native_hash = None
    with torch.inference_mode():
        for keep_count in cases.KEEP_COUNTS:
            ops, record = benchmark.decode_operations(
                benchmark.DecodeCase(64, 8, 512),
                cases.SEED,
                keep_count,
                ("native", "triton"),
            )
            assert [op.mode for op in ops] == ["native", "triton"]
            assert record["packed_keep_count"] == keep_count
            assert record["packed_record_bytes"] == keep_count + 72
            assert ops[1].owned[2]["triton"].values.shape[-1] == keep_count
            current_native = benchmark.tensor_hash(dict(cache=ops[0].owned[3]))
            if identity is not None:
                assert record["common_inputs_sha256"] == identity
                assert current_native == native_hash
            identity, native_hash = record["common_inputs_sha256"], current_native
            for op in ops:
                op.validate()


@pytest.mark.parametrize("keep_count", [64, 192, 256, 448])
def test_gpu_graph_replay_changes_queries_lengths_and_selections(keep_count, benchmark):
    import torch

    with torch.inference_mode():
        ops, _ = benchmark.decode_operations(
            benchmark.DecodeCase(64, 8, 512), cases.SEED, keep_count
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
            benchmark.configure(op.mode, op.keep_count)
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
    assert args.batches == "8,16,24,28,32"
    assert args.topk_lengths == "512"
    assert len(cases.selected_cases(args)) == 35
    args = cases.parser().parse_args(["--suite", "decode", "--batches", "64"])
    assert len(cases.selected_cases(args)) == 2
    assert all(c.batch == 64 for c in cases.selected_cases(args))
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
                "packed_keep_count": 192,
                "packed_record_bytes": 264,
                "common_inputs_sha256": "inputs",
                "execution": "cuda_graph",
                "valid": True,
                "paths": {m: cases.summary([1.0, 2.0]) for m in ("native", "triton")},
                "paired_ratios": {},
            }
        ],
    }
    base = tmp_path / "timing"
    cases.write_results(result, base)
    assert json.loads(base.with_suffix(".json").read_text()) == result
    rows = list(csv.DictReader(base.with_suffix(".csv").open()))
    row = rows[0]
    assert json.loads(row["samples_ms"]) == [1.0, 2.0]
    assert json.loads(row["source"]) == result["source"]
    assert row["protocol"] == cases.PROTOCOL and row["input_sha256"] == "inputs"
    assert [r["packed_keep_count"] for r in rows] == ["192", "192"]
    assert [r["record_bytes"] for r in rows] == ["584", "264"]


@pytest.mark.parametrize("keep_count", cases.KEEP_COUNTS)
def test_timing_validation_matches_requested_modes_and_survivor_identity(keep_count):
    import copy

    modes = cases.benchmark_modes(keep_count)
    payload = {
        "protocol": cases.PROTOCOL,
        "schema_version": 2,
        "status": "passed",
        "config": {"keep_count": keep_count},
        "cases": [
            {
                "suite": "decode" if isinstance(case, cases.DecodeCase) else "store",
                "name": case.name,
                "valid": True,
                "packed_keep_count": keep_count,
                "packed_record_bytes": keep_count + 72,
                "paths": {m: {"samples": [1.0] * 30} for m in modes},
            }
            for case in cases.selected_cases(cases.parser().parse_args([]))
        ],
    }
    cases.validate_timing(payload, keep_count=keep_count)
    for field, value in (("packed_keep_count", 512), ("packed_record_bytes", 584)):
        bad = copy.deepcopy(payload)
        bad["cases"][0][field] = value
        with pytest.raises(RuntimeError):
            cases.validate_timing(bad, keep_count=keep_count)
    bad = copy.deepcopy(payload)
    bad["cases"][0]["paths"].pop("triton")
    with pytest.raises(RuntimeError):
        cases.validate_timing(bad, keep_count=keep_count)
    bad = copy.deepcopy(payload)
    bad["config"]["keep_count"] = 512
    with pytest.raises(RuntimeError):
        cases.validate_timing(bad, keep_count=keep_count)


@pytest.mark.parametrize("keep_count", [64, 192, 256, 448])
def test_gpu_store_graph_replay_changes_inputs_and_destinations(keep_count, benchmark):
    import torch

    with torch.inference_mode():
        ops, _ = benchmark.store_operations(
            benchmark.StoreCase("decode", 25), cases.SEED, keep_count
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
            benchmark.configure(op.mode, op.keep_count)
            graph.replay()
        torch.cuda.synchronize()
        ids = loc.long()
        expected = benchmark.pack_rows_ref(
            x,
            benchmark.topmag_keep_mask(x, keep_count / 512),
            weight,
            benchmark.EPS,
            keep_count=keep_count,
        )
        for mode in caches_:
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
        if "kernel" in caches_:
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


@pytest.mark.parametrize("count", cases.KEEP_COUNTS)
def test_portable_packed_codec_and_profile_contract(count):
    """Real Torch numerics without importing SGLang's CUDA-only package startup."""
    import importlib

    import torch

    runtime = load_module(
        "_portable_remnant", ROOT / "python/sglang/srt/remnant/__init__.py"
    )
    reference = importlib.import_module("_portable_remnant.reference")
    runtime.configure_cache_format("remnant", "triton", keep_count=count)
    assert runtime.packed_profile().default_count == count
    x = torch.arange(512, dtype=torch.float32).repeat(3, 1) / 127
    x[1].zero_()
    mask = reference.topmag_keep_mask(x, count / 512)
    assert torch.equal(mask.sum(-1), torch.full((3,), count))
    assert mask[0, : 512 - count].sum() == 0
    values, bitmap, scales = reference.pack_rows_ref(
        x, mask, torch.ones(512), 1e-6, keep_count=count
    )
    assert values.numel() + bitmap.nbytes + scales.numel() == 3 * (count + 72)
    decoded = reference.unpack_rows_ref(values, bitmap, scales)
    assert decoded.dtype == torch.bfloat16
    assert torch.isfinite(decoded).all() and (decoded[~mask] == 0).all()
    assert (decoded[1] == 0).all()
    masked = x.masked_fill(~mask, 0)
    normalized = (
        masked * torch.rsqrt(masked.square().mean(-1, keepdim=True) + 1e-6)
    ).bfloat16()
    # E4M3 nearest rounding has <= 1/16 relative error (plus BF16 rounding).
    torch.testing.assert_close(
        decoded.float(), normalized.float(), atol=0.001, rtol=0.065
    )
    runtime.configure_cache_format("native")
    assert not runtime.packed_enabled()


def test_packed_hicache_routes_all_planes_without_cuda():
    """Run real routing methods; only the external transfer operations are spies."""
    import ast
    from types import SimpleNamespace

    path = ROOT / "python/sglang/srt/mem_cache/memory_pool_host.py"
    tree = ast.parse(path.read_text())
    cls = next(
        n
        for n in tree.body
        if isinstance(n, ast.ClassDef) and n.name == "RemnantPackedHostPool"
    )
    names = {
        "_transfer_plane_all_layers",
        "_transfer_plane_per_layer",
        "backup_from_device_all_layer",
        "load_to_device_per_layer",
    }
    calls = []
    namespace = {
        name: lambda **kwargs: calls.append(kwargs)
        for name in (
            "transfer_kv_direct",
            "transfer_kv_all_layer_mla",
            "transfer_kv_per_layer_mla",
        )
    }
    for node in cls.body:
        if isinstance(node, ast.FunctionDef) and node.name in names:
            exec(
                compile(ast.Module([node], type_ignores=[]), str(path), "exec"),
                namespace,
            )
    pool = SimpleNamespace(
        layer_num=2,
        pool_name="C4",
        plane_item_bytes=(16384, 4096, 512),
        device_plane_buffers=[[object(), object()] for _ in range(3)],
        host_plane_buffers=[[object(), object()] for _ in range(3)],
        device_plane_ptrs=[object() for _ in range(3)],
        host_plane_ptrs=[object() for _ in range(3)],
        _transfer_page_indices=lambda host, device: (host, device),
    )
    from types import MethodType

    for name in names:
        setattr(pool, name, MethodType(namespace[name], pool))
    host, device = object(), object()
    for backend in ("direct", "kernel"):
        calls.clear()
        pool.backup_from_device_all_layer(None, host, device, backend)
        assert len(calls) == 3
        for plane, call in enumerate(calls):
            assert call["src_indices"] is device and call["dst_indices"] is host
            assert call["src_layers"] is (
                pool.device_plane_buffers[plane]
                if backend == "direct"
                else pool.device_plane_ptrs[plane]
            )
        calls.clear()
        pool.load_to_device_per_layer(None, host, device, 1, backend)
        assert len(calls) == 3
        for plane, call in enumerate(calls):
            assert call["src_indices"] is host and call["dst_indices"] is device
            if backend == "kernel":
                assert call["src"] is pool.host_plane_buffers[plane][1]
                assert call["dst"] is pool.device_plane_buffers[plane][1]
    with pytest.raises(ValueError, match="Unsupported"):
        pool.backup_from_device_all_layer(None, host, device, "storage")
    with pytest.raises(IndexError):
        pool.load_to_device_per_layer(None, host, device, 2, "direct")


def test_v0522_reserved_pages_and_packed_allocator_coefficient():
    """Execute portable allocator arithmetic, not a mocked runtime boot."""
    import ast
    from types import SimpleNamespace

    runtime = load_module(
        "_allocator_config", ROOT / "python/sglang/srt/remnant/config.py"
    )
    pool = ast.parse(
        (ROOT / "python/sglang/srt/mem_cache/deepseek_v4_memory_pool.py").read_text()
    )
    pages = next(
        n
        for n in pool.body
        if isinstance(n, ast.FunctionDef) and n.name == "_num_dsv4_physical_kv_pages"
    )
    namespace = {"ceil_div": lambda x, y: (x + y - 1) // y}
    exec(compile(ast.Module([pages], type_ignores=[]), "pool.py", "exec"), namespace)
    for size, physical, logical, expected in (
        (130, 64, 256, 7),
        (128, 64, 256, 6),
        (0, 64, 256, 4),
    ):
        assert namespace[pages.name](size, physical, logical) == expected
    tree = ast.parse(
        (ROOT / "python/sglang/srt/model_executor/pool_configurator.py").read_text()
    )
    cls = next(
        n
        for n in tree.body
        if isinstance(n, ast.ClassDef) and n.name == "DSV4PoolConfigurator"
    )
    method = next(
        n
        for n in cls.body
        if isinstance(n, ast.FunctionDef) and n.name == "_get_paged_kv_bytes_per_token"
    )
    namespace = {"_sg_lr": runtime}
    exec(
        compile(ast.Module([method], type_ignores=[]), "configurator.py", "exec"),
        namespace,
    )
    runtime.configure_cache_format("remnant", "triton", keep_count=192)
    layouts = tuple(
        runtime.PackedLayerLayout(i, k) for i, k in enumerate((128, 192, 256, 320))
    )
    assert namespace[method.name](SimpleNamespace(packed_c4_layouts=layouts), 4) == 296


def test_context_dispatch_snapshot_is_atomic_without_runtime_imports():
    from types import SimpleNamespace

    config = load_module(
        "_dispatch_config", ROOT / "python/sglang/srt/remnant/config.py"
    )
    config.configure_cache_format("remnant", "triton", keep_count=192)
    previous = config._snapshot_dispatch()
    native = config._dispatch_from_args(SimpleNamespace())
    assert config._snapshot_dispatch() == previous
    config._restore_dispatch(native)
    assert not config.packed_enabled()
    config._restore_dispatch(previous)
    assert config.packed_profile().default_count == 192
    with pytest.raises(ValueError):
        config._dispatch_from_args(SimpleNamespace(kv_cache_format="invalid"))
    assert config._snapshot_dispatch() == previous
    config._restore_dispatch(None)
    assert not config.packed_enabled()


def test_packed_cli_fields_build_as_real_msgspec_declarations():
    """Catch required/default ordering with the real upstream Arg and msgspec types."""
    import ast
    from typing import Optional

    import msgspec

    args = load_module(
        "_v0522_arg_utils", ROOT / "python/sglang/srt/arg_groups/arg_utils.py"
    )
    tree = ast.parse(
        (ROOT / "python/sglang/srt/arg_groups/fields/model.py").read_text()
    )
    model = next(
        n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "Model"
    )
    model.body = [
        n
        for n in model.body
        if isinstance(n, ast.AnnAssign)
        and (
            n.target.id in ("model_path", "kv_cache_format")
            or n.target.id.startswith("dsv4_c4_cache_")
        )
    ]
    namespace = {"msgspec": msgspec, "A": args.A, "Arg": args.Arg, "Optional": Optional}
    exec(
        compile(ast.Module([model], type_ignores=[]), "model-fields.py", "exec"),
        namespace,
    )
    record = namespace["Model"](model_path="dummy")
    assert record.kv_cache_format == "native"
    assert record.dsv4_c4_cache_keep_count == 256
    assert record.dsv4_c4_cache_store_implementation is None


# GLM owns a separate protocol; importing it must not load torch/FlashMLA.
glm = load_module(
    "_glm_attention_contract", ROOT / "python/sglang/benchmark/remnant/glm_attention.py"
)
shared_timing = load_module(
    "_remnant_shared_timing", ROOT / "python/sglang/benchmark/remnant/timing.py"
)


def test_glm_matrix_order_and_baseline_identity():
    matrix = glm.cases()
    assert len(matrix) == len({c.name for c in matrix}) == 128
    assert sum(c.phase == "decode" for c in matrix) == 48
    assert sum(c.phase == "prefill" for c in matrix) == 80
    assert all(c.heads == 16 for c in matrix[:64])
    assert all(c.heads == 8 for c in matrix[64:])
    for heads in (16, 8):
        subset = [c for c in matrix if c.heads == heads]
        assert [c.phase for c in subset[:24]] == ["decode"] * 24
    assert glm.MODES == ("native", "direct")


def test_glm_graph_probe_metadata():
    """Validate the lightweight graph fixture's public metadata without CUDA."""
    import ast

    source = ROOT / "test/registered/attention/unittests/test_glm5_remnant.py"
    helper = next(
        node
        for node in ast.parse(source.read_text()).body
        if isinstance(node, ast.FunctionDef) and node.name == "_benchmark_bank_graph"
    )
    identity = next(
        node.value
        for node in helper.body
        if isinstance(node, ast.Assign)
        and any(ast.unparse(target) == "f.identity" for target in node.targets)
    )
    expression = compile(ast.Expression(identity), str(source), "eval")
    for inner in (1, 2, 33, 66):
        metadata = eval(expression, {"inner": inner})
        assert metadata["inner_iter"] == inner
        assert metadata["path_geometry"] == {
            mode: dict(tile_rows=32, inner_iter=inner, partial_groups=66 // inner)
            for mode in ("native", "direct")
        }


def test_glm_grouped_causal_prefill_and_shared_prefix():
    import numpy as np

    for phase in (0, 3):
        for locality in ("independent", "shared"):
            case = glm.Case("prefill", 16, 8, phase, locality, "warm")
            ids = glm.logical_selection(case)
            assert np.array_equal(ids, glm.logical_selection(case))
            first = glm.HISTORY + (4 if phase == 0 else 3)
            for q, row in enumerate(ids):
                groups = row[:2048].reshape(512, 4)
                assert len(np.unique(groups[:, 0])) == 512
                assert np.all(groups[:, 0] % 4 == 0)
                assert np.array_equal(
                    groups - groups[:, :1], np.tile(np.arange(4), (512, 1))
                )
                length = first + q
                assert np.all(row[row >= 0] < length)
                tail = length % 4
                assert np.array_equal(
                    row[2048 : 2048 + tail], np.arange(length - tail, length)
                )
                assert np.all(row[2048 + tail :] == -1)
            if locality == "shared":
                assert np.all(ids[:, :2048] == ids[0, :2048])
            else:
                assert not np.array_equal(ids[0, :2048], ids[1, :2048])


def test_glm_separate_decode_requests_and_page_mapping():
    import numpy as np

    case = glm.Case("decode", 8, 8, 3, "independent", "warm")
    logical, physical = glm.logical_selection(case), glm.physical_selection(case)
    valid = logical >= 0
    assert np.array_equal(logical[valid] % 256, physical[valid] % 256)
    assert np.all(physical[valid] >= 256)
    for q in range(case.queries):
        selected = physical[q][physical[q] >= 0]
        assert np.all((selected - 256) // case.request_slots == q)
    assert np.array_equal(physical, glm.physical_selection(case))
    assert not np.array_equal(physical, glm.physical_selection(case, 1))


def test_glm_rotation_and_call_normalization():
    assert glm.rotation_count(2048, 50 * 1024**2) == 161
    assert (161 - 1) * 2048 * 320 <= 2 * 50 * 1024**2 < 161 * 2048 * 320
    assert glm.cycles_per_sample(1) == 100
    assert glm.cycles_per_sample(3) == 34
    assert glm.cycles_per_sample(161) == 1
    for value in (0, -1):
        with pytest.raises(ValueError):
            glm.cycles_per_sample(value)
        with pytest.raises(ValueError):
            glm.rotation_count(value, 1024)


def test_glm_shared_round_schedule_and_preconditioning():
    calls, before = [], []
    values, order = shared_timing.paired_samples(
        {m: lambda m=m: calls.append(m) for m in glm.MODES},
        30,
        123,
        34,
        balanced=True,
        before=before.append,
        sample=lambda call, repeat: (call(), float(repeat))[1],
    )
    assert order.count(["native", "direct"]) == order.count(["direct", "native"]) == 15
    assert before == calls == [m for row in order for m in row]
    assert values == {m: [34.0] * 30 for m in glm.MODES}
    assert shared_timing.round_orders(glm.MODES, 30, 123, balanced=True) == order
    with pytest.raises(ValueError):
        shared_timing.round_orders(glm.MODES, 3, 1, balanced=True)


def test_glm_cli_rejects_adapter_and_deepseek_options():
    for argv in (
        ["--modes", "native,adapter"],
        ["--keep-count", "256"],
        ["--native-library", "/tmp/native.so"],
        ["--suite", "store"],
        ["--profile-mode", "native"],
    ):
        with pytest.raises(SystemExit):
            glm.parser().parse_args(argv)
    args = glm.parser().parse_args(
        ["--model-family", "glm53", "--suite", "glm-attention"]
    )
    assert args.modes is None
    assert glm.experiment(args.suite, args.phase)[1] == ("native", "direct")


def test_glm_entrypoint_dispatches_before_deepseek_imports():
    source = (ROOT / "benchmark/remnant/microbench.py").read_text()
    assert source.index("glm_main()") < source.index("from sgl_kernel import flash_mla")


def test_glm_rejects_incomplete_timing():
    with pytest.raises(RuntimeError, match="incomplete"):
        glm.validate_timing({"protocol": glm.PROTOCOL, "status": "incomplete"})
    with pytest.raises(RuntimeError, match="128-case"):
        glm.validate_timing({"protocol": glm.PROTOCOL, "status": "passed", "cases": []})


def _glm_complete_timing_payload():
    from dataclasses import asdict

    records = []
    for definition in glm.cases():
        banks = 1 if definition.policy == "warm" else 2
        records.append(
            dict(
                name=definition.name,
                case=asdict(definition),
                valid=True,
                fixture=dict(
                    banks=banks,
                    selected_rows_per_bank=[1] * banks,
                    l2_bytes=200,
                    projected_bytes=1,
                    memory_limit_bytes=2,
                    common_inputs_sha256="0" * 64,
                ),
                paths={m: cases.summary([1.0] * 30) for m in glm.MODES},
                round_orders=[["native", "direct"], ["direct", "native"]] * 15,
                execution="cuda_graph",
                cycles_per_sample=100 // banks,
                calls_per_sample=100,
                paired_ratios={"direct_over_native": cases.summary([1.0] * 30)},
            )
        )
    return dict(protocol=glm.PROTOCOL, status="passed", cases=records)


def test_glm_full_timing_coverage_rejects_sample_and_cache_corruption():
    import copy

    payload = _glm_complete_timing_payload()
    glm.validate_timing(payload)
    for field, value in [
        ("calls_per_sample", 99),
        ("round_orders", [["native", "direct"]] * 30),
    ]:
        changed = copy.deepcopy(payload)
        changed["cases"][0][field] = value
        with pytest.raises(RuntimeError):
            glm.validate_timing(changed)
    changed = copy.deepcopy(payload)
    changed["cases"][1]["fixture"]["l2_bytes"] = 10000
    with pytest.raises(RuntimeError, match="cache policy"):
        glm.validate_timing(changed)
    changed = copy.deepcopy(payload)
    changed["cases"][0]["paths"]["direct"]["samples"] = [1.0] * 29
    with pytest.raises(RuntimeError, match="rounds"):
        glm.validate_timing(changed)


def test_glm_graph_pool_accounting_ignores_global_cache_changes():
    from types import SimpleNamespace

    pool = (7, 3)
    graph = SimpleNamespace(pool=lambda: pool)
    calls = []

    def snapshot(pool_id):
        calls.append(pool_id)
        # total_size includes inactive bytes, which also belong to this graph.
        return [
            dict(address=4096, total_size=8192, active_size=1024),
            dict(address=16384, total_size=4096, active_size=0),
        ]

    # No global memory_reserved query is available: its delta can be negative.
    cuda = SimpleNamespace(memory_snapshot=snapshot)
    assert glm.graph_reserved_bytes(cuda, graph) == 12288
    assert calls == [pool]
    for segments in (
        [],
        [dict(address=1, total_size=0)],
        [dict(address=1, total_size=2)] * 2,
    ):
        cuda.memory_snapshot = lambda pool_id: segments
        with pytest.raises(RuntimeError, match="Graph pool snapshot"):
            glm.graph_reserved_bytes(cuda, graph)


def test_glm_selected_initialization_covers_real_backing_locations():
    import numpy as np

    case = glm.Case("prefill", 16, 128, 3, "independent", "warm")
    ids = glm.physical_selection(case)
    locations = glm.selected_physical_rows(ids)
    assert len(locations) == len(set(ids[ids >= 0].tolist()))
    assert np.all(locations >= glm.PAGE)
    # Independent expected gather/scatter on shuffled pages and duplicated,
    # invalid, tail and request-offset positions. Unselected storage stays zero.
    native = np.zeros((locations.max() + 1, 2), dtype=np.int64)
    dense = np.arange(len(native) * 2, dtype=np.int64).reshape(-1, 2)
    native[locations] = dense[locations]
    np.testing.assert_array_equal(native[ids[ids >= 0]], dense[ids[ids >= 0]])
    assert not native[0].any()
    np.testing.assert_array_equal(
        glm.selected_physical_rows(np.array([-1, 257, 256, 257, -1, 1024])),
        [256, 257, 1024],
    )


def _glm_source_with_artifacts(tmp_path):
    import hashlib

    binary = tmp_path / "runtime.so"
    cubin = tmp_path / "partial.cubin"
    binary.write_bytes(b"runtime artifact")
    cubin.write_bytes(b"TileLang artifact")

    def digest(path):
        return hashlib.sha256(path.read_bytes()).hexdigest()

    versions = dict(torch="torch", cuda="cuda", triton="triton", tilelang="tilelang")
    return dict(
        sources=glm.benchmark_source_hashes(ROOT / "python/sglang"),
        compiled_build=dict(
            model_family="glm53",
            sources={"sglang": "source-tree"},
            source_identities={"sglang": {"commit": "source-commit"}},
            binaries={str(binary): digest(binary)},
            glm_compile=dict(
                attention_scale=glm.SCALE,
                triton=["store", "unpack"],
                tilelang=[dict(cubin_path=str(cubin), cubin_sha256=digest(cubin))],
                cache_files={str(cubin): digest(cubin)},
            ),
            **versions,
        ),
        **versions,
    )


def test_glm_resume_requires_identical_valid_ordered_prefix(tmp_path):
    import copy

    source = _glm_source_with_artifacts(tmp_path)
    previous = _glm_complete_timing_payload()
    previous.update(
        source=source, seed=glm.SEED, rounds=glm.ROUNDS, status="incomplete"
    )
    previous["cases"] = previous["cases"][:3] + [dict(valid=False)]
    accepted = glm.resume_prefix(previous, source)
    assert len(accepted) == 3
    assert previous["cases"][-1] == dict(valid=False)  # input retained as evidence
    for key, value in [("source", {}), ("seed", 0), ("rounds", 29)]:
        changed = copy.deepcopy(previous)
        changed[key] = value
        with pytest.raises(RuntimeError, match="identity"):
            glm.resume_prefix(changed, source)
    changed = copy.deepcopy(previous)
    changed["cases"][1]["valid"] = False
    with pytest.raises(RuntimeError, match="Incomplete GLM case"):
        glm.resume_prefix(changed, source)
    changed = copy.deepcopy(previous)
    changed["cases"][:2] = changed["cases"][1::-1]
    with pytest.raises(RuntimeError, match="128-case"):
        glm.resume_prefix(changed, source)
    changed = copy.deepcopy(previous)
    changed["cases"][0]["paths"]["native"]["samples"] = [1.0] * 29
    with pytest.raises(RuntimeError, match="rounds"):
        glm.resume_prefix(changed, source)


@pytest.mark.parametrize(
    "dependency",
    [
        "srt/remnant/reference.py",
        "srt/layers/layernorm.py",
        "benchmark/remnant/cases.py",
        "srt/remnant/config.py",
        "srt/remnant/glm_config.py",
        "srt/remnant/bitmap.py",
    ],
)
def test_glm_resume_rejects_changed_fixture_dependencies(
    dependency, tmp_path, monkeypatch
):
    import copy

    source = _glm_source_with_artifacts(tmp_path)
    previous = _glm_complete_timing_payload()
    previous.update(source=source, seed=glm.SEED, rounds=glm.ROUNDS)
    root = ROOT / "python/sglang"
    changed_path = root / dependency
    original_read = Path.read_bytes
    monkeypatch.setattr(
        Path,
        "read_bytes",
        lambda path: (
            b"changed fixture dependency"
            if path == changed_path
            else original_read(path)
        ),
    )
    changed_source = copy.deepcopy(source)
    changed_source["sources"] = glm.benchmark_source_hashes(root)
    assert {
        name
        for name, digest in changed_source["sources"].items()
        if digest != source["sources"][name]
    } == {dependency}
    with pytest.raises(RuntimeError, match="identity mismatch"):
        glm.resume_prefix(previous, changed_source)


def test_glm_fresh_diagnostic_identity_needs_no_build_manifest(monkeypatch):
    import torch

    monkeypatch.delenv("REMNANT_BUILD_MANIFEST_PATH", raising=False)
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda: "H100")
    monkeypatch.setattr(glm.importlib.metadata, "version", lambda name: name)
    source = glm.source_manifest()
    assert source["compiled_build"] == {}
    assert source["sources"] == glm.benchmark_source_hashes(ROOT / "python/sglang")
    previous = _glm_complete_timing_payload()
    previous.update(source=source, seed=glm.SEED, rounds=glm.ROUNDS)
    with pytest.raises(RuntimeError, match="REMNANT_BUILD_MANIFEST_PATH"):
        glm.resume_prefix(previous, source)


@pytest.mark.parametrize(
    "missing", ["sources", "source_identities", "binaries", "glm_compile"]
)
def test_glm_resume_rejects_incomplete_build_manifest(missing, tmp_path):
    source = _glm_source_with_artifacts(tmp_path)
    source["compiled_build"].pop(missing)
    previous = _glm_complete_timing_payload()
    previous.update(source=source, seed=glm.SEED, rounds=glm.ROUNDS)
    with pytest.raises(RuntimeError, match="complete GLM build manifest"):
        glm.resume_prefix(previous, source)


def test_glm_resume_rejects_other_model_manifest(tmp_path):
    source = _glm_source_with_artifacts(tmp_path)
    source["compiled_build"].update(model_family="dsv4", glm_compile=None)
    with pytest.raises(RuntimeError, match="complete GLM build manifest"):
        glm.verify_resume_artifacts(source)


@pytest.mark.parametrize("artifact", ["runtime.so", "partial.cubin"])
@pytest.mark.parametrize("change", ["missing", "modified"])
def test_glm_resume_verifies_current_artifact_bytes(artifact, change, tmp_path):
    source = _glm_source_with_artifacts(tmp_path)
    previous = _glm_complete_timing_payload()
    previous.update(source=source, seed=glm.SEED, rounds=glm.ROUNDS)
    path = tmp_path / artifact
    if change == "missing":
        path.unlink()
    else:
        path.write_bytes(b"different build")
    with pytest.raises(
        RuntimeError, match=f"artifact missing or changed: .*{artifact}"
    ):
        glm.resume_prefix(previous, source)


@pytest.mark.parametrize("mode", ["native", "direct", "direct_over_native"])
@pytest.mark.parametrize("statistic", ["mean", "median", "p95", "stdev"])
def test_glm_validation_and_resume_reject_inconsistent_statistics(
    mode, statistic, tmp_path
):
    source = _glm_source_with_artifacts(tmp_path)
    previous = _glm_complete_timing_payload()
    previous.update(source=source, seed=glm.SEED, rounds=glm.ROUNDS)
    record = previous["cases"][0]
    stats = record["paired_ratios"] if mode == "direct_over_native" else record["paths"]
    stats[mode][statistic] = 1234.0
    with pytest.raises(RuntimeError, match="derived"):
        glm.validate_timing(previous)
    with pytest.raises(RuntimeError, match="derived"):
        glm.resume_prefix(previous, source)


def test_glm_twofold_gate_uses_raw_ratios():
    records = _glm_complete_timing_payload()["cases"][:2]
    records[0]["paired_ratios"]["direct_over_native"]["mean"] = 3.0
    records[1]["paired_ratios"]["direct_over_native"].update(
        samples=[2.1] * glm.ROUNDS, mean=1.0
    )
    assert glm.weighted_overhead(records)["unacceptable_2x_cases"] == [
        records[1]["name"]
    ]


def test_glm_measured_capture_preserves_ten_warmups():
    import inspect

    assert (
        inspect.signature(glm.Fixture.capture).parameters["warmup_calls"].default == 10
    )


def test_glm_cpu_oracle_preserves_fp8_rounding_and_empty_rows():
    import torch

    q = torch.zeros(2, 1, 512, dtype=torch.bfloat16)
    q[0, 0, 0] = 1.0625  # midpoint: E4M3 RNE rounds to 1.0
    rows = torch.zeros(2, 512)
    rows[:, 0] = torch.tensor([1.0, -1.0])
    codes = rows.to(torch.float8_e4m3fn).view(torch.uint8)
    ids = torch.tensor([[0, 1], [-1, -1]], dtype=torch.int32)
    expected = torch.zeros(1, 2, 1, 512, dtype=torch.bfloat16)
    expected[0, 0, 0, 0] = torch.tanh(torch.tensor(glm.SCALE))
    actual = glm.attention_reference(q, codes, ids, device="cpu")
    assert actual.device.type == "cpu"
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)


def test_glm_decode_scope_and_weighting():
    assert len(glm.cases("decode")) == 48
    assert glm.cases("decode") == [c for c in glm.cases() if c.phase == "decode"]
    assert glm.parser().parse_args(["--phase", "decode"]).phase == "decode"
    payload = _glm_complete_timing_payload()
    payload.update(
        protocol=glm.DECODE_PROTOCOL,
        phase="decode",
        cases=[c for c in payload["cases"] if c["case"]["phase"] == "decode"],
    )
    payload["summary"] = glm.weighted_overhead(payload["cases"])
    glm.validate_timing(payload)
    with pytest.raises(RuntimeError):
        glm.validate_timing({**payload, "protocol": glm.PROTOCOL})
    for c in payload["cases"]:
        ratio = 1.01 if c["case"]["queries"] == 1 else 1.2
        c["paired_ratios"]["direct_over_native"].update(
            samples=[ratio] * 30, mean=ratio
        )
    score = glm.weighted_overhead(payload["cases"])
    assert score["completed_cases"] == 48
    assert score["batch_weighted_overhead_percent"] == pytest.approx(
        (0.01 + 144 * 0.2) / 145 * 100
    )
    with pytest.raises(RuntimeError, match="identity"):
        glm.resume_prefix(
            {**payload, "source": {}, "seed": glm.SEED, "rounds": 30}, {}, "all"
        )


def test_glm_full_prefill_scope_and_resume(tmp_path):
    import numpy as np

    matrix = glm.cases("prefill")
    assert len(matrix) == 80
    assert matrix == [case for case in glm.cases() if case.phase == "prefill"]
    assert glm.parser().parse_args(["--phase", "prefill"]).phase == "prefill"
    for case in (matrix[2], matrix[-1]):
        indices = glm.physical_selection(case, bank=1)
        np.testing.assert_array_equal(
            glm.selected_physical_rows(indices, shared_history=True),
            glm.selected_physical_rows(indices),
        )

    source = _glm_source_with_artifacts(tmp_path)
    payload = _glm_complete_timing_payload()
    payload.update(
        protocol=glm.PREFILL_PROTOCOL,
        phase="prefill",
        packed_prefill_tile=64,
        cases=[c for c in payload["cases"] if c["case"]["phase"] == "prefill"],
        source=source,
        seed=glm.SEED,
        rounds=glm.ROUNDS,
    )
    for case in payload["cases"]:
        case["fixture"].update(
            inner_iter=66,
            path_geometry={
                "native": dict(tile_rows=32, inner_iter=66, partial_groups=1),
                "direct": dict(tile_rows=64, inner_iter=17, partial_groups=2),
            },
        )
    glm.validate_timing(payload)
    checkpoint = {**payload, "cases": payload["cases"][:3]}
    assert (
        glm.resume_prefix(checkpoint, source, "prefill", packed_prefill_tile=64)
        == checkpoint["cases"]
    )
    with pytest.raises(RuntimeError, match="identity"):
        glm.resume_prefix(checkpoint, source, "prefill")
    with pytest.raises(RuntimeError, match="80-case"):
        glm.validate_timing(checkpoint)


def test_glm_decode_summary_must_match_raw_samples(tmp_path):
    source = _glm_source_with_artifacts(tmp_path)
    payload = _glm_complete_timing_payload()
    payload.update(
        protocol=glm.DECODE_PROTOCOL,
        phase="decode",
        cases=[c for c in payload["cases"] if c["case"]["phase"] == "decode"],
        source=source,
        seed=glm.SEED,
        rounds=glm.ROUNDS,
    )
    payload["summary"] = glm.weighted_overhead(payload["cases"])
    payload["summary"]["batch_weighted_overhead_percent"] = 999.0
    with pytest.raises(RuntimeError, match="derived decode summary"):
        glm.validate_timing(payload)
    with pytest.raises(RuntimeError, match="derived decode summary"):
        glm.resume_prefix(payload, source, "decode")


def test_glm_completed_resume_retains_derived_decode_summary(tmp_path, monkeypatch):
    import json

    source = _glm_source_with_artifacts(tmp_path)
    payload = _glm_complete_timing_payload()
    payload.update(
        protocol=glm.DECODE_PROTOCOL,
        phase="decode",
        cases=[c for c in payload["cases"] if c["case"]["phase"] == "decode"],
        source=source,
        seed=glm.SEED,
        rounds=glm.ROUNDS,
    )
    payload["summary"] = glm.weighted_overhead(payload["cases"])
    checkpoint = tmp_path / "checkpoint.json"
    checkpoint.write_text(json.dumps(payload))
    base = tmp_path / "resumed"
    monkeypatch.setattr(glm, "source_manifest", lambda: source)
    monkeypatch.setattr(
        glm,
        "Fixture",
        lambda *args: pytest.fail("A complete checkpoint must not rerun GPU cases"),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "microbench",
            "--phase",
            "decode",
            "--resume",
            str(checkpoint),
            "--output",
            str(base),
        ],
    )
    glm.main()
    result = json.loads(base.with_suffix(".json").read_text())
    assert result["status"] == "passed"
    assert result["summary"] == payload["summary"]
    assert result["resumed_from"]["accepted_cases"] == 48


def test_glm_selected_packed_reference_preserves_bytes_and_duplicates():
    torch = pytest.importorskip("torch")
    values = torch.arange(256, dtype=torch.uint8).repeat(2, 1)
    words = torch.tensor(
        [[2**64 - 1] * 4 + [0] * 4, [0] * 4 + [2**64 - 1] * 4], dtype=torch.uint64
    )
    ids = torch.tensor([[1, -1, 0, 1]], dtype=torch.int32)
    dense, local = glm.selected_packed_reference(values, words, ids)
    assert local.tolist() == [[1, -1, 0, 1]]
    assert torch.equal(dense[0, :256], values[0]) and not dense[0, 256:].any()
    assert torch.equal(dense[1, 256:], values[1]) and not dense[1, :256].any()


@pytest.mark.parametrize("mode", ["native", "triton", "kernel"])
def test_single_mode_benchmark_timing(mode):
    """Exercise the shared loop through the existing benchmark entry point."""
    import ast
    from types import SimpleNamespace
    from unittest.mock import Mock

    args = cases.parser().parse_args(["--suite", "decode", "--modes", mode])
    assert cases.benchmark_modes(args.keep_count, args.modes) == (mode,)
    args.rounds, args.warmup, args.repeats = 2, 0, 1
    function = next(
        node
        for node in ast.parse(
            (ROOT / "benchmark/remnant/microbench.py").read_text()
        ).body
        if isinstance(node, ast.FunctionDef) and node.name == "time_case"
    )
    operation = SimpleNamespace(
        key=mode,
        mode=mode,
        keep_count=256,
        call=lambda: 3.0,
        validate=Mock(),
        validate_output=None,
    )
    namespace = dict(
        Operation=object,
        DecodeCase=cases.DecodeCase,
        configure=Mock(),
        capture=lambda op, warmup: (SimpleNamespace(replay=op.call), None),
        paired_samples=shared_timing.paired_samples,
        measure=lambda call, repeats: call(),
        case_seed=cases.case_seed,
        summary=cases.summary,
        paired_ratios=cases.paired_ratios,
    )
    exec(
        compile(ast.Module(body=[function], type_ignores=[]), "microbench.py", "exec"),
        namespace,
    )
    result = namespace["time_case"](cases.selected_cases(args)[0], [operation], args)
    assert result["paths"][mode]["samples"] == [3.0, 3.0]
    assert operation.validate.call_count == 2


def test_balanced_timing_still_requires_two_paths():
    with pytest.raises(ValueError, match="Balanced"):
        shared_timing.round_orders(("native",), 30, 1, balanced=True)


def test_glm_artifact_metadata_binds_verified_source_binary_and_abi(
    tmp_path, monkeypatch
):
    import hashlib
    from types import ModuleType

    artifacts = load_module(
        "_glm_artifact_contract",
        ROOT / "python/sglang/benchmark/remnant/glm_artifacts.py",
    )
    generated = tmp_path / "device_kernel.cu"
    generated.write_text(
        "void glm_packed_partial_kernel(uint64_t* bitmap, int* indices, "
        "uint8_t* kv_fp8, float* partial_lse, float* partial_o, "
        "uint8_t* q_fp8, int num_pages, int seq_len) {}"
    )
    (tmp_path / "host_kernel.cu").write_text("v_int64) = ((int64_t)37248);")
    cubin = tmp_path / "test.cubin"
    cubin.write_bytes(b"verified cubin")
    cache = {
        str(p): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in (generated, cubin, tmp_path / "host_kernel.cu")
    }
    tilelang = ModuleType("tilelang")
    tilelang.__file__ = str(tmp_path / "tilelang/__init__.py")
    cuda_cache = ModuleType("tilelang.cache.cuda_binary_cache")

    class Cache:
        @staticmethod
        def make_key(**kwargs):
            assert (
                kwargs["target_arch"] == "90a" and kwargs["compile_format"] == "cubin"
            )
            return "verified-key"

        @staticmethod
        def get_path(key, suffix):
            assert (key, suffix) == ("verified-key", "cubin")
            return str(cubin)

    cuda_cache.CUDABinaryCache = Cache
    monkeypatch.setitem(sys.modules, "tilelang", tilelang)
    monkeypatch.setitem(sys.modules, "tilelang.cache.cuda_binary_cache", cuda_cache)
    variant = dict(heads=8, inner=2, packed=True, source_sha256=cache[str(generated)])
    manifest = dict(tilelang=[variant], cache_files=cache)
    artifacts.bind_partial_launch_metadata(manifest)
    assert variant["cubin_sha256"] == cache[str(cubin)]
    assert variant["dynamic_shared"] == 37248
    assert variant["symbol"] == "glm_packed_partial_kernel"
    assert [p["name"] for p in variant["parameters"]][-2:] == ["num_pages", "seq_len"]
    cubin.write_bytes(b"tampered cubin")
    with pytest.raises(RuntimeError, match="Unverified production GLM cubin"):
        artifacts.bind_partial_launch_metadata(manifest)
    cubin.write_bytes(b"verified cubin")
    generated.write_text(generated.read_text().replace("num_pages", "wrong_abi"))
    variant["source_sha256"] = hashlib.sha256(generated.read_bytes()).hexdigest()
    cache[str(generated)] = variant["source_sha256"]
    with pytest.raises(RuntimeError, match="ABI changed"):
        artifacts.bind_partial_launch_metadata(manifest)


def test_glm_cpu_preparation_requires_explicit_cache_paths(monkeypatch):
    artifacts = load_module(
        "_glm_artifact_paths", ROOT / "python/sglang/benchmark/remnant/glm_artifacts.py"
    )
    for name in (
        "TRITON_CACHE_DIR",
        "TILELANG_CACHE_DIR",
        "REMNANT_GLM_COMPILE_MANIFEST",
    ):
        monkeypatch.delenv(name, raising=False)
    with pytest.raises(RuntimeError, match="requires explicit paths"):
        artifacts.precompile_sm90()


def test_v41_native_reference_preserves_tail_and_rounds_fp8_ties():
    import torch

    reference = load_module(
        "_v41_native_quantization",
        ROOT / "python/sglang/benchmark/remnant/quantization.py",
    ).v41_native_fp8_reference
    values = torch.zeros((2, 512), dtype=torch.bfloat16)
    values[0, :3] = torch.tensor([1.0, 1.0625, 1.1875])
    values[:, 448:] = torch.arange(64).bfloat16() / 7
    decoded = reference(values)
    assert decoded.dtype == torch.bfloat16
    assert decoded[0, :3].tolist() == [1.0, 1.0, 1.25]
    assert torch.equal(decoded[1, :448], values[1, :448])
    assert torch.equal(decoded[:, 448:], values[:, 448:])
    for invalid in (values.float(), values[:, :448], values.reshape(1, 2, 512)):
        with pytest.raises(ValueError, match="Native reference requires"):
            reference(invalid)


def test_v41_three_path_ratios():
    ratios = cases.paired_ratios(
        {"native": [1.0, 2.0], "dense_fp4": [2.0, 4.0], "kernel": [3.0, 6.0]}
    )
    assert ratios["kernel_over_native"]["mean"] == 3.0
    assert ratios["dense_fp4_over_native"]["mean"] == 2.0
    assert ratios["kernel_over_dense_fp4"]["mean"] == 1.5


def test_v41_native_call_does_not_reuse_warmup_metadata():
    """Exercise the actual callback without CUDA; fresh schedules must reach the API."""
    import ast
    from types import SimpleNamespace

    tree = ast.parse((ROOT / "benchmark/remnant/microbench.py").read_text())
    factory = next(
        n
        for n in tree.body
        if isinstance(n, ast.FunctionDef) and n.name == "fp4_decode_operations"
    )
    callback = next(
        n
        for n in ast.walk(factory)
        if isinstance(n, ast.FunctionDef) and n.name == "call"
    )
    received = []

    def decode(*args, **kwargs):
        received.append(args[5])
        return "output", "lse"

    namespace = {
        name: object()
        for name in (
            "q",
            "swa_view",
            "swa_indices",
            "swa_lengths",
            "cache_view",
            "physical",
            "lengths",
            "sink",
        )
    }
    namespace.update(
        packed=False,
        mode="native",
        upstream_flash_mla=SimpleNamespace(
            get_mla_metadata=lambda: (object(), None), flash_mla_with_kvcache=decode
        ),
    )
    exec(
        compile(ast.Module(body=[callback], type_ignores=[]), "microbench.py", "exec"),
        namespace,
    )
    assert namespace["call"]() == namespace["call"]() == ("output", "lse")
    assert len(received) == 2 and received[0] is not received[1]


@pytest.mark.parametrize("global_page_size", [128, 256])
def test_gpu_v41_upstream_native_comparison(global_page_size, benchmark):
    """Both production page sizes; original control, graphs and a nondefault stream."""
    import torch

    case = next(
        c
        for c in cases.fp4_decode_cases()
        if c.batch == 8 and c.global_page_size == global_page_size
    )
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.inference_mode(), torch.cuda.stream(stream):
        operations, identity = benchmark.fp4_decode_operations(case, cases.SEED)
        assert tuple(op.key for op in operations) == cases.FP4_DECODE_MODES
        assert (
            identity["native_control"]["source_commit"] == cases.UPSTREAM_FLASHMLA_SHA
        )
        for operation in operations:
            operation.validate()
            graph, output = benchmark.capture(operation, 3)
            # Selection order changes in place, while the selected set and
            # lengths stay fixed. Upstream permits reusing this same schedule.
            indices = operation.owned[2]
            indices.copy_(indices.flip(-1))
            graph.replay()
            operation.validate_output(output)
    torch.cuda.current_stream().wait_stream(stream)


def test_v41_fp4_matrix_and_timing_gate():
    """Production V4.1 geometry only; stale or incomplete timing cannot qualify."""
    import copy
    from dataclasses import asdict

    args = cases.parser().parse_args(["--suite", "v41-fp4-decode"])
    assert cases.FP4_DECODE_MODES == ("native", "dense_fp4", "kernel")
    for modes in ("native,kernel", "dense_fp4", "native,triton,kernel"):
        args.modes = modes
        with pytest.raises(ValueError, match="V4.1 compares"):
            cases.selected_cases(args)
    args.modes = ",".join(cases.FP4_DECODE_MODES)
    for flag, value in (
        ("--heads", "128"),
        ("--batches", "1"),
        ("--topk-lengths", "317"),
    ):
        with pytest.raises(ValueError, match="V4.1 FP4 requires"):
            cases.selected_cases(
                cases.parser().parse_args(["--suite", "v41-fp4-decode", flag, value])
            )
    matrix = cases.selected_cases(args)
    assert len(matrix) == len({c.name for c in matrix}) == 10
    assert {(c.heads, c.batch, c.topk_length) for c in matrix} == {
        (64, b, 512) for b in (8, 16, 24, 28, 32)
    }
    assert {(c.swa_length, c.swa_page_size, c.global_page_size) for c in matrix} == {
        (128, 256, 256),
        (128, 256, 128),
    }
    payload = {
        "status": "passed",
        "protocol": "remnant-v41-fp4-decode-v5",
        "config": vars(args),
        "cases": [],
    }
    for c in matrix:
        payload["cases"].append(
            {
                "name": c.name,
                "case": asdict(c),
                "valid": True,
                "execution": "cuda_graph",
                "repeats": 100,
                "baseline": "upstream H100 auto FP8/BF16; synchronization corrected",
                "native_scheduler": "first use per ratio; regenerated inside graph",
                "native_control": {
                    "source_commit": cases.UPSTREAM_FLASHMLA_SHA,
                    "source_sha256": "b" * 64,
                    "binary_sha256": "c" * 64,
                    "python_sha256": "d" * 64,
                    "architectures": ["sm_90a"],
                },
                "common_inputs_sha256": "a" * 64,
                "paths": {
                    mode: {"samples": [1.0] * 30}
                    for mode in ("native", "dense_fp4", "kernel")
                },
                "path_layouts": {
                    "native": {
                        "implementation": "Upstream H100 auto FP8/BF16",
                        "record_bytes": 584,
                        "packed_keep_count": 512,
                    },
                    "dense_fp4": {
                        "implementation": "Custom dense FP4 Direct",
                        "record_bytes": 288,
                        "packed_keep_count": 512,
                    },
                    "kernel": {
                        "implementation": "Packed FP4 Direct",
                        "record_bytes": 224,
                        "packed_keep_count": 256,
                    },
                },
            }
        )
    cases.validate_timing(payload, suite="v41-fp4-decode")
    for fault in (
        "missing-case",
        "missing-mode",
        "bad-sample",
        "wrong-layout",
        "wrong-geometry",
        "nonproduction-swa",
        "window-as-page",
        "nonproduction-heads",
        "old-protocol",
        "missing-diagnostic",
        "custom-baseline",
        "fork-control",
        "missing-control",
        "warmed-scheduler",
    ):
        broken = copy.deepcopy(payload)
        if fault == "missing-case":
            broken["cases"].pop()
        elif fault == "missing-mode":
            broken["cases"][0]["paths"].pop("kernel")
        elif fault == "bad-sample":
            broken["cases"][0]["paths"]["native"]["samples"][0] = float("nan")
        elif fault == "wrong-layout":
            broken["cases"][0]["path_layouts"]["native"]["record_bytes"] = 288
        elif fault == "missing-diagnostic":
            broken["cases"][0]["paths"].pop("dense_fp4")
        elif fault == "custom-baseline":
            broken["cases"][0]["baseline"] = "unpruned dense Native FP4"
        elif fault == "fork-control":
            broken["cases"][0]["native_control"]["source_commit"] = "3eb8594"
        elif fault == "missing-control":
            broken["cases"][0].pop("native_control")
        elif fault == "warmed-scheduler":
            broken["cases"][0]["native_scheduler"] = "cached before capture"
        elif fault == "wrong-geometry":
            broken["cases"][0]["case"]["swa_page_size"] = 64 + 128
        elif fault == "nonproduction-swa":
            broken["cases"][0]["case"]["swa_length"] = 64
        elif fault == "window-as-page":
            broken["cases"][0]["case"]["swa_page_size"] = 128
        elif fault == "nonproduction-heads":
            broken["cases"][0]["case"]["heads"] = 128
        else:
            broken["protocol"] = "remnant-v41-fp4-decode-v3"
        with pytest.raises((RuntimeError, ValueError)):
            cases.validate_timing(broken, suite="v41-fp4-decode")


def test_shared_statistics_preserve_exact_summary_and_paired_rounds():
    import statistics

    samples = [4.0, 1.0, 3.0, 2.0]
    expected = dict(
        samples=samples,
        mean=2.5,
        median=2.5,
        stdev=statistics.stdev(samples),
        p95=3.8499999999999996,
    )
    assert shared_timing.summary(samples) == cases.summary(samples) == expected
    assert shared_timing.paired_ratio_summary(
        [2.0, 3.0], [1.0, 2.0]
    ) == shared_timing.summary([2.0, 1.5])
    for left, right in (
        ([1.0, 2.0], [1.0]),
        ([1.0, float("nan")], [1.0, 2.0]),
        ([1.0, 2.0], [0.0, 2.0]),
    ):
        with pytest.raises(ValueError):
            shared_timing.paired_ratio_summary(left, right)


upstream = load_module(
    "_glm_upstream_contract", ROOT / "python/sglang/benchmark/remnant/glm_upstream.py"
)


def test_glm_upstream_decode_matrix_and_labels():
    protocol, modes, matrix = glm.experiment(glm.UPSTREAM_SUITE, "decode")
    assert protocol == "remnant-glm-upstream-decode-v1"
    assert modes == ("upstream", "direct")
    assert len(matrix) == len({c.name for c in matrix}) == 24
    assert {c.heads for c in matrix} == {16}
    assert {c.queries for c in matrix} == {1, 8, 16, 24, 32, 64}
    assert {c.tail for c in matrix} == {0, 3}
    assert {c.policy for c in matrix} == {"warm", "rotating"}
    assert matrix == [c for c in glm.cases("decode") if c.heads == 16]
    for phase in ("all", "prefill", "prefill-small"):
        with pytest.raises(ValueError, match="phase decode"):
            glm.experiment(glm.UPSTREAM_SUITE, phase)
    fixture = glm.Fixture.__new__(glm.Fixture)
    assert fixture.modes == ("native", "direct")
    fixture.upstream = True
    assert fixture.modes == modes


def test_glm_upstream_source_and_helper_provenance(tmp_path):
    import hashlib
    import json
    import subprocess

    directory = tmp_path / "upstream"
    package = tmp_path / "package"
    directory.mkdir()
    for name in upstream.FILES:
        path = (
            directory / "tilelang_kernel.py"
            if name == upstream.MODULE
            else package / name
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        if name == upstream.MODULE:
            prepared = os.environ.get(upstream.SOURCE_ENV)
            data = (
                (Path(prepared) / "tilelang_kernel.py").read_bytes()
                if prepared
                else subprocess.check_output(
                    [
                        "git",
                        "-C",
                        str(ROOT),
                        "show",
                        f"{upstream.COMMIT}:python/sglang/{name}",
                    ],
                    timeout=30,
                )
            )
        else:
            data = (ROOT / "python/sglang" / name).read_bytes()
        path.write_bytes(data)
        assert hashlib.sha256(data).hexdigest() == upstream.FILES[name]
    (directory / "source.json").write_text(json.dumps(upstream.identity()))
    assert upstream.verify_source(directory, package) == upstream.identity()
    for name in upstream.FILES:
        path = (
            directory / "tilelang_kernel.py"
            if name == upstream.MODULE
            else package / name
        )
        original = path.read_bytes()
        path.write_bytes(original + b"\n# changed\n")
        with pytest.raises(RuntimeError, match="changed"):
            upstream.verify_source(directory, package)
        path.write_bytes(original)
    (directory / "source.json").write_text(json.dumps({"commit": "wrong"}))
    with pytest.raises(RuntimeError, match="receipt"):
        upstream.verify_source(directory, package)


def test_glm_upstream_compile_artifacts_are_bound(tmp_path):
    import copy
    import hashlib

    cubin = tmp_path / "kernel.cubin"
    cubin.write_bytes(b"independent CPU fixture")
    digest = hashlib.sha256(cubin.read_bytes()).hexdigest()
    compiled = dict(
        source=upstream.identity(),
        cache_files={str(cubin): digest},
        tilelang=[
            dict(
                heads=16,
                inner=i,
                packed=False,
                cubin_path=str(cubin),
                cubin_sha256=digest,
            )
            for i in (1, 2)
        ],
    )
    upstream.verify_compilation(compiled)
    wgmma = dict(
        cubin_path=str(cubin), cubin_sha256=digest,
        library_path=str(cubin), library_sha256=digest,
    )
    candidate = copy.deepcopy(compiled)
    candidate["decode_wgmma"] = {
        str(batch): dict(
            partial=dict(wgmma), combine=dict(wgmma), qgmma_sites=16, warp_mma_sites=32
        )
        for batch in upstream.WGMMA_INNER
    }
    upstream.verify_compilation(candidate)
    for change in ("geometry", "qgmma", "warp_mma", "library"):
        broken = copy.deepcopy(candidate)
        variants = broken["decode_wgmma"]
        if change == "geometry":
            variants.pop("64")
        elif change == "qgmma":
            variants["64"]["qgmma_sites"] = 32
        elif change == "warp_mma":
            variants["64"].pop("warp_mma_sites")
        else:
            variants["64"]["partial"]["library_sha256"] = "wrong"
        with pytest.raises(RuntimeError, match="WGMMA|QK/PV"):
            upstream.verify_compilation(broken)
    for change in ("missing", "source", "binary", "missing_binding"):
        candidate = copy.deepcopy(compiled)
        if change == "missing":
            candidate["tilelang"].pop()
        elif change == "source":
            candidate["source"]["commit"] = "wrong"
        elif change == "missing_binding":
            candidate["tilelang"][0].pop("cubin_path")
            candidate["tilelang"][0].pop("cubin_sha256")
        else:
            candidate["tilelang"][0]["cubin_sha256"] = "wrong"
        with pytest.raises(RuntimeError):
            upstream.verify_compilation(candidate)
    cubin.write_bytes(b"changed")
    with pytest.raises(RuntimeError, match="changed"):
        upstream.verify_compilation(compiled)


def test_glm_wgmma_shared_ranges_reject_async_operand_aliases():
    offsets = dict(q=0, kv=8192, scratch=40960, p=49664, pv_scale=50688)

    def generated() -> str:
        return "\n".join(
            f"void* {name} = ((void*)((char*)buf_dyn_shmem + {offset}));"
            for name, offset in offsets.items()
        )

    assert upstream.wgmma_shared_ranges(generated())["kv"] == (8192, 40960)
    offsets["p"] = 0
    with pytest.raises(RuntimeError, match="Aliased WGMMA"):
        upstream.wgmma_shared_ranges(generated())


def test_glm_decode_cache_binding_uses_exact_wrapper_entry(tmp_path, monkeypatch):
    import hashlib
    from types import SimpleNamespace

    artifacts = load_module(
        "_glm_decode_artifact_contract",
        ROOT / "python/sglang/benchmark/remnant/glm_artifacts.py",
    )
    device, host = "device program", "selected host wrapper"
    entry, stale = tmp_path / "selected", tmp_path / "stale"
    cubin = tmp_path / "device.cubin"
    cache = {
        str(entry / "device_kernel.cu"): hashlib.sha256(device.encode()).hexdigest(),
        str(stale / "device_kernel.cu"): hashlib.sha256(device.encode()).hexdigest(),
        str(entry / "host_kernel.cu"): hashlib.sha256(host.encode()).hexdigest(),
        str(entry / "executable.so"): "selected binary",
        str(stale / "executable.so"): "old binary",
        str(cubin): "cubin",
    }
    monkeypatch.setattr(artifacts, "_verified_sm90_cubin", lambda *_: cubin)
    kernel = SimpleNamespace(
        _tilelang_cache_path=entry,
        get_kernel_source=lambda: device,
        adapter=SimpleNamespace(get_host_source=lambda: host),
    )
    assert artifacts.bind_decode_library(kernel, cache)["library_path"] == str(
        entry / "executable.so"
    )
    kernel._tilelang_cache_path = stale
    with pytest.raises(RuntimeError, match="does not match"):
        artifacts.bind_decode_library(kernel, cache)


def _glm_upstream_payload():
    payload = _glm_complete_timing_payload()
    payload.update(
        suite=glm.UPSTREAM_SUITE,
        protocol=glm.UPSTREAM_PROTOCOL,
        phase="decode",
        source={"upstream": upstream.identity(), "device": "NVIDIA H200"},
        cases=[
            c
            for c in payload["cases"]
            if c["case"]["phase"] == "decode" and c["case"]["heads"] == 16
        ],
    )
    for case in payload["cases"]:
        inner = 1 if case["case"]["queries"] == 1 else 2
        case["fixture"].update(
            changing_input_replay="passed",
            path_geometry={
                mode: dict(tile_rows=32, inner_iter=inner, partial_groups=66 // inner)
                for mode in ("upstream", "direct")
            },
        )
        case["paths"]["upstream"] = case["paths"].pop("native")
        case["paired_ratios"]["direct_over_upstream"] = case["paired_ratios"].pop(
            "direct_over_native"
        )
        case["round_orders"] = [["upstream", "direct"], ["direct", "upstream"]] * 15
    payload["summary"] = glm.weighted_overhead(payload["cases"], baseline="upstream")
    return payload


def test_glm_upstream_result_identity_and_csv(tmp_path):
    import copy

    payload = _glm_upstream_payload()
    glm.validate_timing(payload)
    glm.write_results(payload, tmp_path / "timing")
    csv = (tmp_path / "timing.csv").read_text()
    assert "direct_over_upstream" in csv and "direct_over_native" not in csv
    for change in ("scope", "source", "label", "rounds", "ratio"):
        bad = copy.deepcopy(payload)
        if change == "scope":
            bad["cases"].pop()
        elif change == "source":
            bad["source"]["upstream"]["commit"] = "wrong"
        elif change == "label":
            bad["cases"][0]["paths"]["native"] = bad["cases"][0]["paths"].pop(
                "upstream"
            )
        elif change == "rounds":
            bad["cases"][0]["round_orders"][0] = ["direct", "upstream"]
        else:
            bad["cases"][0]["paired_ratios"]["direct_over_upstream"]["samples"][0] = 2
        with pytest.raises(RuntimeError):
            glm.validate_timing(bad)
    with pytest.raises(RuntimeError, match="identity"):
        glm.resume_prefix(payload, payload["source"], "decode")


def test_glm_upstream_goal_scores_all_cases_equally():
    _, _, cases = glm.experiment(glm.UPSTREAM_SUITE, "decode")
    records = [
        {
            "name": case.name,
            "valid": True,
            "paired_ratios": {
                "direct_over_upstream": {
                    "samples": [0.4, 0.8] * 15 if i % 2 else [0.96] * 30
                }
            },
        }
        for i, case in enumerate(cases)
    ]
    score = glm.upstream_decode_score(records)
    assert score["geometric_mean_direct_over_upstream"] == pytest.approx(0.6 * 1.6**0.5)
    assert score["speedup"] == pytest.approx(1 / (0.6 * 1.6**0.5))
    assert score["meets_target"]
    with pytest.raises(ValueError, match="all 24"):
        glm.upstream_decode_score(records[:-1])
    records[-1]["paired_ratios"]["direct_over_upstream"]["samples"][-1] = float("nan")
    with pytest.raises(ValueError, match="Invalid"):
        glm.upstream_decode_score(records)


def test_glm_decode_trace_distinguishes_identical_native_symbols():
    events = [
        dict(cat="kernel", ph="X", pid=0, tid=1, ts=i * 100, name=symbol, dur=duration)
        for i, (symbol, duration) in enumerate(
            [("copy", 2), ("main_kernel", 8), ("main_kernel", 5)] * 2
        )
    ]
    stages = glm.decode_trace_stages(events[::-1], 2)
    assert [stage["us_per_call"] for stage in stages.values()] == [2, 8, 5]
    assert stages["partial_attention"]["symbol"] == stages["combine"]["symbol"]
    with pytest.raises(RuntimeError, match="three CUDA kernels"):
        glm.decode_trace_stages(events[:-1], 2)


def test_glm_upstream_direct_split_policy_bounds_partial_traffic():
    policy = glm._cpu_contract("glm_config")
    inner = policy.H200_PACKED_DECODE_INNER
    assert list(inner) == [1, 8, 16, 24, 32, 64]
    assert all(66 % split == 0 for split in inner.values())
    assert inner[1] == 1 and inner[8] == 2
    assert all(batch * (66 // inner[batch]) >= 2 * 132 for batch in inner if batch >= 8)
    assert [66 // inner[batch] for batch in (16, 24, 32, 64)] == [22, 11, 11, 6]
    assert set(policy.H200_PACKED_DECODE_COMBINE) == {
        (66, 128),
        (33, 512),
        (22, 512),
        (11, 512),
        (6, 512),
    }


def test_glm_upstream_geometry_is_bound_to_combine_artifact_scope():
    cfg = glm._cpu_contract("glm_config")
    for tuned in (False, True):
        payload = _glm_upstream_payload()
        variants = cfg.H200_PACKED_DECODE_COMBINE if tuned else ((66, 128), (33, 512))
        compiled = {"decode_combine": {f"{g}x{c}": {} for g, c in variants}}
        payload["source"]["compiled_build"] = {"glm_compile": compiled}
        for row in payload["cases"]:
            queries = row["case"]["queries"]
            fixture = row["fixture"]
            fixture["direct_combine_columns"] = 128 if queries == 1 else 512
            if tuned:
                inner = cfg.H200_PACKED_DECODE_INNER[queries]
                fixture["path_geometry"]["direct"] = dict(
                    tile_rows=32, inner_iter=inner, partial_groups=66 // inner
                )
        glm.validate_timing(payload)
        payload["cases"][0]["fixture"].pop("direct_combine_columns")
        with pytest.raises(RuntimeError, match="geometry"):
            glm.validate_timing(payload)
