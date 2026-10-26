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
        config._dispatch_from_args(SimpleNamespace(dsv4_c4_cache_format="invalid"))
    assert config._snapshot_dispatch() == previous
    config._restore_dispatch(None)
    assert not config.packed_enabled()


def test_packed_cli_fields_build_as_real_msgspec_declarations():
    """Catch required/default ordering with the real upstream Arg and msgspec types."""
    import ast
    from typing import Optional

    import msgspec

    args = load_module("_v0522_arg_utils", ROOT / "python/sglang/srt/arg_groups/arg_utils.py")
    tree = ast.parse((ROOT / "python/sglang/srt/arg_groups/fields/model.py").read_text())
    model = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "Model")
    model.body = [n for n in model.body if isinstance(n, ast.AnnAssign) and (
        n.target.id == "model_path" or n.target.id.startswith("dsv4_c4_cache_")
    )]
    namespace = {"msgspec": msgspec, "A": args.A, "Arg": args.Arg, "Optional": Optional}
    exec(compile(ast.Module([model], type_ignores=[]), "model-fields.py", "exec"), namespace)
    record = namespace["Model"](model_path="dummy")
    assert record.dsv4_c4_cache_format == "native"
    assert record.dsv4_c4_cache_keep_count == 256
    assert record.dsv4_c4_cache_store_implementation is None


def test_v41_fp4_matrix_and_timing_gate():
    """Production V4.1 geometry only; stale or incomplete timing cannot qualify."""
    import copy
    from dataclasses import asdict

    args = cases.parser().parse_args(["--suite", "v41-fp4-decode"])
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
        "protocol": "remnant-v41-fp4-decode-v3",
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
                "common_inputs_sha256": "a" * 64,
                "paths": {
                    mode: {"samples": [1.0] * 30} for mode in ("native", "kernel")
                },
                "path_layouts": {
                    "native": {
                        "implementation": "Native FP4 Direct",
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
    ):
        broken = copy.deepcopy(payload)
        if fault == "missing-case":
            broken["cases"].pop()
        elif fault == "missing-mode":
            broken["cases"][0]["paths"].pop("kernel")
        elif fault == "bad-sample":
            broken["cases"][0]["paths"]["native"]["samples"][0] = float("nan")
        elif fault == "wrong-layout":
            broken["cases"][0]["path_layouts"]["native"]["record_bytes"] = 584
        elif fault == "wrong-geometry":
            broken["cases"][0]["case"]["swa_page_size"] = 64 + 128
        elif fault == "nonproduction-swa":
            broken["cases"][0]["case"]["swa_length"] = 64
        elif fault == "window-as-page":
            broken["cases"][0]["case"]["swa_page_size"] = 128
        elif fault == "nonproduction-heads":
            broken["cases"][0]["case"]["heads"] = 128
        else:
            broken["protocol"] = "remnant-v41-fp4-decode-v2"
        with pytest.raises((RuntimeError, ValueError)):
            cases.validate_timing(broken, suite="v41-fp4-decode")
