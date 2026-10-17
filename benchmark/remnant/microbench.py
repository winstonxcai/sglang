"""Current Native, Packed Triton and Packed Kernel on one H100, without weights.

Author: Winston Cai.
Decode and post-compressor store are separate measurements. Native retains all
coordinates; Packed modes prune 256/512. A matched Packed Torch reference is a
validity control, never a Native timing baseline. Times are milliseconds per
complete operation; p95 describes round-average timings, not request latency.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import importlib.metadata
import json
import math
import os
import random
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

import torch
from sgl_kernel import flash_mla
from sglang.benchmark.remnant.cases import (
    MODES,
    PROTOCOL,
    DecodeCase,
    StoreCase,
    case_seed,
    matrix_digest,
    paired_ratios,
    parse_modes,
    parser,
    selected_cases,
    store_layout,
    summary,
    write_results,
)
from sglang.kernels.ops.attention.dsv4.compress import (
    CompressorDecodePlan,
    CompressorPrefillPlan,
)
from sglang.benchmark.remnant.quantization import check_fp8_storage
from sglang.srt import remnant
from sglang.srt.remnant.packed import (
    NativeWorkspace,
    PackedBuffers,
    pack_rows,
    pack_rows_fused,
    unpack_gather_native,
)
from sglang.srt.remnant.reference import (
    _ue8m0_quantize,
    pack_rows_ref,
    topmag_keep_mask,
    unpack_rows_ref,
)

PAGE_SIZE = 64  # Production C4 geometry: base page 256 / compress ratio 4.
EPS = 1e-6
ATOL = RTOL = 2e-2
NATIVE_STORE_MODULE = None


def configure(mode: str) -> None:
    remnant.configure_cache_format(
        "native" if mode == "native" else "remnant",
        "kernel" if mode == "native" else mode,
    )
    if remnant.cache_format() != ("native" if mode == "native" else "remnant"):
        raise RuntimeError(f"cache format did not resolve for {mode}")
    if mode != "native" and remnant.cache_implementation() != mode:
        raise RuntimeError(f"cache implementation did not resolve for {mode}")


def frequencies(positions: int) -> torch.Tensor:
    angle = (
        (torch.arange(positions, device="cuda", dtype=torch.float32)[:, None] + 1)
        * (torch.arange(32, device="cuda") + 1)
        * 0.0017
    )
    return torch.polar(torch.ones_like(angle), angle).contiguous()


def plan(seq: list[int] | torch.Tensor, ragged: list[int] | torch.Tensor, phase: str):
    seq = torch.as_tensor(seq, dtype=torch.int32, device="cuda")
    ragged = torch.as_tensor(ragged, dtype=torch.int32, device="cuda")
    words = torch.zeros((len(seq), 4), dtype=torch.int32, device="cuda")
    words[:, 0], words[:, 1] = seq, torch.where(seq < 0, 65535, ragged)
    blob = words.view(torch.uint8)
    return (
        CompressorDecodePlan(4, blob)
        if phase == "decode"
        else CompressorPrefillPlan(
            4, blob, torch.empty((0, 8), dtype=torch.uint8, device="cuda")
        )
    )


def native_storage(rows: int) -> torch.Tensor:
    pages = (rows + PAGE_SIZE - 1) // PAGE_SIZE
    size = ((PAGE_SIZE * 584 + 575) // 576) * 576
    return torch.zeros((pages, size), device="cuda", dtype=torch.uint8)


def native_view(storage: torch.Tensor) -> torch.Tensor:
    return storage.as_strided(
        (len(storage), PAGE_SIZE, 1, 584), (storage.stride(0), 584, 584, 1)
    )


def packed_storage(rows: int) -> PackedBuffers:
    pages = (rows + PAGE_SIZE - 1) // PAGE_SIZE
    return PackedBuffers(
        torch.zeros((pages, PAGE_SIZE, 256), device="cuda", dtype=torch.uint8),
        torch.zeros((pages, PAGE_SIZE, 8), device="cuda", dtype=torch.uint64),
        torch.zeros((pages, PAGE_SIZE, 8), device="cuda", dtype=torch.uint8),
    )


def native_write(x, weight, metadata, locations, freqs, storage):
    if NATIVE_STORE_MODULE is None:
        raise RuntimeError(
            "Native store library must be built on CPU before GPU execution"
        )
    # Exact underlying production FFI entry point; loading its verified library
    # directly prevents a cache miss from triggering nvcc on the GPU worker.
    NATIVE_STORE_MODULE.forward(
        x,
        metadata[1],
        weight,
        EPS,
        torch.view_as_real(freqs).flatten(-2),
        locations,
        storage,
        metadata.is_decode,
        metadata.compress_ratio,
    )


def packed_write(mode, x, weight, metadata, locations, buffers):
    if mode == "triton":
        # Selection and the mask allocation belong to the measured store path.
        pack_rows(
            x, topmag_keep_mask(x, 0.5), weight, EPS, metadata, locations, buffers
        )
    else:
        pack_rows_fused(x, weight, EPS, metadata, locations, buffers)


def decode_native(storage: torch.Tensor) -> torch.Tensor:
    """Independent page reader: value bytes and scales have separate regions."""
    codes = storage[:, : PAGE_SIZE * 576].reshape(-1, 576)
    scales = storage[:, PAGE_SIZE * 576 : PAGE_SIZE * 584].reshape(-1, 8)
    nope = codes[:, :448].contiguous().view(torch.float8_e4m3fn).float()
    nope *= torch.exp2(scales[:, :7].float() - 127).repeat_interleave(64, dim=-1)
    tail = codes[:, 448:576].contiguous().view(torch.bfloat16).float()
    return torch.cat((nope, tail), dim=-1)


def rotate_tail(
    dense: torch.Tensor, positions: torch.Tensor, freqs: torch.Tensor
) -> torch.Tensor:
    tail = torch.view_as_complex(dense[:, 448:].float().reshape(-1, 32, 2).contiguous())
    rotated = (
        torch.view_as_real(tail * freqs[positions.long()])
        .flatten(-2)
        .to(torch.bfloat16)
    )
    return torch.cat((dense[:, :448].float(), rotated.float()), dim=-1)


def native_reference(
    x: torch.Tensor, weight: torch.Tensor, positions: torch.Tensor, freqs: torch.Tensor
) -> torch.Tensor:
    normalized = (
        x.float()
        * torch.rsqrt(x.float().square().mean(-1, keepdim=True) + EPS)
        * weight.float()
    )
    quantized, scales = _ue8m0_quantize(normalized.to(torch.bfloat16))
    nope = quantized[:, :448].float() * torch.exp2(
        scales[:, :7].float() - 127
    ).repeat_interleave(64, dim=-1)
    # Stock Native rotates FP32 normalized tail before its single BF16 rounding.
    tail = torch.view_as_complex(normalized[:, 448:].reshape(-1, 32, 2).contiguous())
    rotated = (
        torch.view_as_real(tail * freqs[positions.long()])
        .flatten(-2)
        .to(torch.bfloat16)
    )
    return torch.cat((nope, rotated.float()), dim=-1)


def attention_reference(q, swa, rows, physical, lengths):
    selected = rows[physical.clamp_min(0).long()[:, 0]]
    valid = (
        torch.arange(physical.shape[-1], device="cuda")[None] < lengths[:, None]
    ) & (physical[:, 0] >= 0)
    kv = torch.cat((swa, selected), dim=1).float()
    valid = torch.cat(
        (torch.ones(swa.shape[:2], device="cuda", dtype=torch.bool), valid), dim=-1
    )
    score = q[:, 0].float() @ kv.transpose(-1, -2) * 512**-0.5
    score.masked_fill_(~valid[:, None], -math.inf)
    output = (score.softmax(-1) @ kv).to(torch.bfloat16)[:, None]
    # FlashMLA's completed decode returns natural-log LSE, [B,H,Q].
    return output, score.logsumexp(-1)[:, :, None]


def check_output(actual, expected, name: str) -> None:
    for label, a, e in zip(("output", "lse"), actual, expected):
        if not bool(torch.isfinite(a).all()):
            raise RuntimeError(f"{name}: nonfinite {label}")
        torch.testing.assert_close(a, e, atol=ATOL, rtol=RTOL, msg=f"{name}: {label}")


def tensor_hash(values: dict) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(values.items()):
        tensors = value if isinstance(value, (tuple, list)) else (value,)
        for t in tensors:
            digest.update(str((name, tuple(t.shape), str(t.dtype))).encode())
            digest.update(
                t.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes()
            )
    return digest.hexdigest()


@dataclass
class Operation:
    mode: str
    call: Callable
    # Retain every allocation referenced by closures/graphs, including outputs.
    owned: tuple
    validate: Callable
    expected_kernels: tuple[str, ...]


def check_native_storage(storage, x, weight, positions, freqs, name, destinations=None):
    decoded = decode_native(storage)
    scales = storage[:, PAGE_SIZE * 576 : PAGE_SIZE * 584].reshape(-1, 8)[:, :7]
    if destinations is not None:
        decoded, scales = decoded[destinations], scales[destinations]
    else:
        decoded, scales = decoded[: len(x)], scales[: len(x)]
    check_fp8_storage(decoded[:, :448], scales, x, weight, EPS, name)
    torch.testing.assert_close(
        decoded[:, 448:],
        native_reference(x, weight, positions, freqs)[:, 448:],
        atol=ATOL,
        rtol=RTOL,
    )


def indexed_rows(tensor: torch.Tensor, indices) -> torch.Tensor:
    # CUDA advanced indexing does not implement UInt64. Reinterpret bitmap bits
    # as signed integers for indexing, then restore the original unsigned type.
    if tensor.dtype == torch.uint64:
        return tensor.view(torch.int64)[indices].view(torch.uint64)
    return tensor[indices]


def check_packed_bytes(left, right, name: str) -> None:
    differences = []
    for label, a, b in zip(("values", "bitmaps", "scales"), left, right):
        if not torch.equal(a, b):
            aa, bb = a.contiguous().view(torch.uint8), b.contiguous().view(torch.uint8)
            positions = (aa != bb).nonzero()
            first = tuple(positions[0].tolist())
            differences.append(
                {
                    "buffer": label,
                    "count": len(positions),
                    "first": first,
                    "triton": int(aa[first]),
                    "kernel": int(bb[first]),
                }
            )
    if differences:
        raise AssertionError(f"{name}: Packed producer bytes differ: {differences}")


def decode_operations(case: DecodeCase, seed: int) -> tuple[list[Operation], dict]:
    rows = case.batch * case.topk
    rng = torch.Generator(device="cuda").manual_seed(case_seed(case, seed))
    x = torch.randn((rows, 512), device="cuda", generator=rng, dtype=torch.float32)
    weight = torch.linspace(0.5, 1.5, 512, device="cuda", dtype=torch.float32)
    q = torch.randn(
        (case.batch, 1, case.heads, 512),
        device="cuda",
        generator=rng,
        dtype=torch.bfloat16,
    )
    raw_rows = torch.arange(rows, device="cuda", dtype=torch.int32) * 3 + 17
    freqs = frequencies(int(raw_rows[-1]) * 4 + 1)
    loc = torch.arange(rows, device="cuda", dtype=torch.int64)
    metadata = plan((raw_rows + 1) * 4, torch.arange(rows, device="cuda"), "prefill")
    physical = torch.arange(rows, device="cuda", dtype=torch.int32).reshape(
        case.batch, 1, case.topk
    )
    raw = raw_rows.reshape(case.batch, 1, case.topk).clone()
    lengths = torch.full(
        (case.batch,), case.topk_length, device="cuda", dtype=torch.int32
    )
    # Keep the frozen matrix's padding: physical=0, raw=-1 beyond active length.
    physical[:, :, case.topk_length :] = 0
    raw[:, :, case.topk_length :] = -1
    swa_x = torch.randn(
        (case.batch * case.swa_length, 512), device="cuda", generator=rng
    )
    swa_storage = native_storage(len(swa_x))
    swa_plan = plan([4] * len(swa_x), list(range(len(swa_x))), "prefill")
    native_write(
        swa_x,
        weight,
        swa_plan,
        torch.arange(len(swa_x), device="cuda", dtype=torch.int64),
        freqs,
        swa_storage,
    )
    swa = decode_native(swa_storage)[: len(swa_x)].reshape(
        case.batch, case.swa_length, 512
    )
    swa_indices = torch.arange(len(swa_x), device="cuda", dtype=torch.int32).reshape(
        case.batch, 1, case.swa_length
    )
    swa_lengths = torch.full_like(lengths, case.swa_length)
    native_cache = native_storage(rows)
    native_write(x, weight, metadata, loc, freqs, native_cache)
    check_native_storage(
        native_cache, x, weight, raw_rows * 4, freqs, case.name + "/native"
    )
    caches = {}
    for mode in ("triton", "kernel"):
        configure(mode)
        caches[mode] = packed_storage(rows)
        packed_write(mode, x, weight, metadata, loc, caches[mode])
    check_packed_bytes(caches["triton"], caches["kernel"], case.name)
    decoded = unpack_rows_ref(*(t.flatten(0, 1)[:rows] for t in caches["triton"]))
    matched = rotate_tail(decoded, raw_rows * 4, freqs)
    operations = []
    for mode in MODES:
        configure(mode)
        mla_meta = flash_mla.get_mla_metadata()[0]
        ws = (
            NativeWorkspace.allocate(case.batch, case.topk, PAGE_SIZE, "cuda")
            if mode == "triton"
            else None
        )

        def call(mode=mode, mla_meta=mla_meta, ws=ws):
            kwargs = {}
            indices = physical
            if mode == "native":
                kwargs["extra_k_cache"] = native_view(native_cache)
            elif mode == "triton":
                cache, indices = unpack_gather_native(
                    caches[mode], physical[:, 0], raw[:, 0], lengths, freqs, ws
                )
                kwargs["extra_k_cache"] = native_view(cache)
                indices = indices[:, None]
            else:
                kwargs.update(
                    remnant_buffers=caches[mode],
                    remnant_raw_indices=raw,
                    remnant_freqs=torch.view_as_real(freqs),
                )
            return flash_mla.flash_mla_with_kvcache(
                q,
                native_view(swa_storage),
                None,
                None,
                512,
                mla_meta,
                softmax_scale=512**-0.5,
                is_fp8_kvcache=True,
                indices=swa_indices,
                topk_length=swa_lengths,
                extra_topk_length=lengths,
                extra_indices_in_kvcache=indices,
                **kwargs,
            )

        def reference(mode=mode):
            dense = decode_native(native_cache)[:rows] if mode == "native" else matched
            return attention_reference(q, swa, dense, physical, lengths)

        def validate(mode=mode, call=call, reference=reference):
            configure(mode)
            check_output(call(), reference(), f"{case.name}/{mode}")

        kernels = (
            (
                "unpack_gather_bf16",
                "rope_tail_complex",
                "bf16_to_native",
                "splitkv",
                "combine",
            )
            if mode == "triton"
            else ("splitkv", "combine")
        )
        operations.append(
            Operation(
                mode,
                call,
                (
                    x,
                    q,
                    caches,
                    native_cache,
                    swa_storage,
                    ws,
                    mla_meta,
                    freqs,
                    physical,
                    raw,
                    lengths,
                ),
                validate,
                kernels,
            )
        )
    return operations, {
        "common_inputs_sha256": tensor_hash(
            dict(
                x=x,
                weight=weight,
                q=q,
                physical=physical,
                raw=raw,
                lengths=lengths,
                freqs=freqs,
                swa=swa,
            )
        ),
        "packed_bytes_sha256": tensor_hash(dict(buffers=caches["triton"])),
        "native_baseline": "unpruned production Native",
        "page_size": PAGE_SIZE,
    }


def store_operations(case: StoreCase, seed: int) -> tuple[list[Operation], dict]:
    seq, ragged, locations = store_layout(case, seed)
    rng = torch.Generator(device="cuda").manual_seed(case_seed(case, seed))
    x = torch.randn((case.rows, 512), device="cuda", dtype=torch.float32, generator=rng)
    weight = torch.linspace(0.5, 1.5, 512, device="cuda", dtype=torch.float32)
    loc = torch.tensor(locations, device="cuda", dtype=torch.int64)
    metadata = plan(seq, ragged, case.phase)
    destinations = [
        locations[i if case.phase == "decode" else ragged[i]] for i in range(case.rows)
    ]
    valid = [
        i
        for i, s in enumerate(seq)
        if s >= 0 and (case.phase == "prefill" or s % 4 == 0) and destinations[i] >= 0
    ]
    # Native's API accepts only valid destinations. Represent identical skipped
    # writes in its plan, before timing, without passing negative pointers.
    native_seq = [
        s if destinations[i] >= 0 else (5 if case.phase == "decode" else -1)
        for i, s in enumerate(seq)
    ]
    if case.phase == "decode":
        native_seq = [5 if s < 0 else s for s in native_seq]
    native_plan = plan(native_seq, ragged, case.phase)
    native_loc = loc.clamp_min(0)
    freqs = frequencies(max(4, max(seq) + 1))
    capacity = case.rows + 7
    native_cache = native_storage(capacity)
    caches = {m: packed_storage(capacity) for m in ("triton", "kernel")}
    expected_packed = pack_rows_ref(x, topmag_keep_mask(x, 0.5), weight, EPS)
    operations = []
    for mode in MODES:

        def call(mode=mode):
            if mode == "native":
                native_write(x, weight, native_plan, native_loc, freqs, native_cache)
                return native_cache
            packed_write(mode, x, weight, metadata, loc, caches[mode])
            return caches[mode]

        def validate(mode=mode, call=call):
            configure(mode)
            call()
            if valid:
                ids = torch.tensor(valid, device="cuda", dtype=torch.long)
                dest = torch.tensor(
                    [destinations[i] for i in valid], device="cuda", dtype=torch.long
                )
                if mode == "native":
                    check_native_storage(
                        native_cache,
                        x[ids],
                        weight,
                        torch.tensor(
                            [max(seq[i] - 4, 0) for i in valid], device="cuda"
                        ),
                        freqs,
                        case.name + "/native",
                        dest,
                    )
                else:
                    actual = tuple(
                        indexed_rows(t.flatten(0, 1), dest) for t in caches[mode]
                    )
                    assert torch.equal(
                        actual[1], indexed_rows(expected_packed[1], ids)
                    ), f"{case.name}/{mode}: selection differs"
                    check_fp8_storage(
                        unpack_rows_ref(*actual),
                        actual[2],
                        x[ids],
                        weight,
                        EPS,
                        case.name + "/" + mode,
                        topmag_keep_mask(x[ids], 0.5),
                    )
            written = set(destinations[i] for i in valid)
            untouched = [i for i in range(capacity) if i not in written]
            if untouched:
                if mode == "native":
                    assert (
                        torch.count_nonzero(decode_native(native_cache)[untouched]) == 0
                    )
                else:
                    for t in caches[mode]:
                        assert (
                            torch.count_nonzero(
                                indexed_rows(t.flatten(0, 1), untouched).view(
                                    torch.uint8
                                )
                            )
                            == 0
                        )

        kernels = (
            ("fused_norm_rope_flashmla",)
            if mode == "native"
            else (
                ("topk|top_k|radix|sort", "pack_fp8")
                if mode == "triton"
                else ("pack_fused_fp32",)
            )
        )
        operations.append(
            Operation(
                mode,
                call,
                (
                    x,
                    weight,
                    metadata,
                    native_plan,
                    native_cache,
                    caches,
                    freqs,
                    loc,
                    native_loc,
                ),
                validate,
                kernels,
            )
        )
    for op in operations:
        op.validate()
    check_packed_bytes(caches["triton"], caches["kernel"], case.name)
    return operations, {
        "common_inputs_sha256": tensor_hash(
            dict(x=x, weight=weight, plan=metadata[1], locations=loc, freqs=freqs)
        ),
        "valid_store_rows": len(valid),
        "capacity_rows": capacity,
        "dtype": "float32",
        "native_invalid_destinations": "represented by skipped plan entries before timing",
    }


def capture(op: Operation, warmup: int) -> tuple[torch.cuda.CUDAGraph, object]:
    configure(op.mode)
    for _ in range(max(3, warmup)):
        op.call()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        output = op.call()
    graph.replay()
    torch.cuda.synchronize()
    op.validate()
    return graph, output


def measure(call: Callable, repeats: int) -> float:
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(
        enable_timing=True
    )
    start.record()
    for _ in range(repeats):
        call()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) / repeats


def time_case(case, operations: list[Operation], args) -> dict:
    graph_mode = isinstance(case, DecodeCase) or case.phase == "decode"
    repeats = args.repeats if graph_mode else args.prefill_repeats
    calls, retained = {}, []
    for op in operations:
        op.validate()
        if graph_mode:
            graph, output = capture(op, args.warmup)
            calls[op.mode] = graph.replay
            retained.append((graph, output, op))
        else:
            configure(op.mode)
            for _ in range(args.warmup):
                op.call()
            calls[op.mode] = op.call
    samples = {op.mode: [] for op in operations}
    rng = random.Random(case_seed(case, args.seed))
    for _ in range(args.rounds):
        order = list(samples)
        rng.shuffle(order)
        for mode in order:
            configure(mode)
            samples[mode].append(measure(calls[mode], repeats))
    for op in operations:
        op.validate()
    return {
        "execution": "cuda_graph" if graph_mode else "eager",
        "repeats": repeats,
        "paths": {m: summary(v) for m, v in samples.items()},
        "paired_ratios": paired_ratios(samples),
    }


def profile_case(case, op: Operation, args) -> dict:
    op.validate()
    configure(op.mode)
    for _ in range(max(3, args.warmup)):
        op.call()
    torch.cuda.synchronize()
    start = torch.cuda.cudart().cudaProfilerStart()
    if (start[0] if isinstance(start, tuple) else start) != 0:
        raise RuntimeError("cudaProfilerStart failed")
    try:
        output = op.call()
        torch.cuda.synchronize()
    finally:
        stop = torch.cuda.cudart().cudaProfilerStop()
        if (stop[0] if isinstance(stop, tuple) else stop) != 0:
            raise RuntimeError("cudaProfilerStop failed")
    op.validate()
    return {
        "execution": "profiled_eager",
        "mode": op.mode,
        "calls": 1,
        "expected_kernels": op.expected_kernels,
        "acceptance_timing": False,
    }


def source_manifest() -> dict:
    import sglang.benchmark.remnant.cases as cases_module
    import sglang.benchmark.remnant.quantization as quantization_module
    import sglang.kernels.ops.attention.dsv4.compress as compress_module
    import sglang.srt.remnant.packed as packed_module

    roots = [
        Path(packed_module.__file__).parent,
        Path(compress_module.__file__).parents[3] / "jit",
    ]
    files = [
        Path(__file__),
        Path(cases_module.__file__),
        Path(quantization_module.__file__),
        Path(compress_module.__file__),
        Path(flash_mla.__file__),
    ]
    for root in roots:
        files += [
            p
            for p in root.rglob("*")
            if p.is_file() and p.suffix in {".py", ".cu", ".cuh", ".h", ".hpp"}
        ]
    sources = {
        str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(set(files))
    }
    binaries = {}
    for name, module in tuple(sys.modules.items()):
        path = getattr(module, "__file__", None)
        if name.startswith("sgl_kernel") and path and path.endswith(".so"):
            binaries[path] = hashlib.sha256(Path(path).read_bytes()).hexdigest()
    return {
        "revisions": {
            k: os.environ.get("REMNANT_" + k.upper() + "_SHA", "unknown")
            for k in ("parent", "sglang", "flashmla")
        },
        "sources": sources,
        "binaries": binaries,
        "compiled_build": (
            json.loads(Path(os.environ["REMNANT_BUILD_MANIFEST_PATH"]).read_text())
            if "REMNANT_BUILD_MANIFEST_PATH" in os.environ
            else {}
        ),
    }


def main() -> None:
    args = parser().parse_args()
    cases = selected_cases(args)
    modes = parse_modes(args.modes)
    if args.check_imports:
        from sgl_kernel import remnant_ops  # noqa: F401 -- registers checked operator

        if getattr(flash_mla, "_flashmla_import_error", None) is not None:
            raise RuntimeError(
                "verified FlashMLA extension did not load"
            ) from flash_mla._flashmla_import_error
        if not hasattr(torch.ops.sgl_kernel, "remnant_sparse_decode_fwd"):
            raise RuntimeError("verified Direct Packed operator is missing")
        if args.native_library:
            from tvm_ffi import load_module

            module = load_module(args.native_library)
            if not callable(module.forward):
                raise RuntimeError(
                    "Native CPU binary lacks production forward entry point"
                )
        print(json.dumps({"protocol": PROTOCOL, "cases": len(cases), "modes": modes}))
        return
    if (
        not torch.cuda.is_available()
        or torch.cuda.get_device_capability() != (9, 0)
        or "H100" not in torch.cuda.get_device_name()
    ):
        raise RuntimeError("Requires one H100/SM90")
    global NATIVE_STORE_MODULE
    if not args.native_library or not Path(args.native_library).is_file():
        raise RuntimeError(
            "--native-library must point to the CPU-built Native store library"
        )
    from tvm_ffi import load_module

    NATIVE_STORE_MODULE = load_module(args.native_library)
    stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    base = Path(args.output or f"microbench-{stamp}").with_suffix("")
    if base.with_suffix(".json").exists() or base.with_suffix(".csv").exists():
        raise FileExistsError(f"refusing to overwrite {base}")
    payload = {
        "schema_version": 2,
        "protocol": PROTOCOL,
        "config": vars(args),
        "matrix_sha256": matrix_digest(cases),
        "environment": {
            "device": torch.cuda.get_device_name(),
            "capability": list(torch.cuda.get_device_capability()),
            "python": sys.version,
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "triton": importlib.metadata.version("triton"),
            "tvm_ffi": importlib.metadata.version("apache-tvm-ffi"),
        },
        "scope": "model-free production functions; decode and post-compressor store measured separately",
        "started_at": stamp,
        "cases": [],
        "status": "running",
    }
    try:
        with torch.inference_mode():
            for case in cases:
                factory = (
                    decode_operations
                    if isinstance(case, DecodeCase)
                    else store_operations
                )
                record = {
                    "suite": "decode" if isinstance(case, DecodeCase) else "store",
                    "name": case.name,
                    "case": asdict(case),
                    "valid": False,
                }
                payload["cases"].append(record)
                write_results(payload, base)
                operations, identity = factory(case, args.seed)
                record.update(identity)
                if args.profile_mode:
                    record.update(
                        profile_case(
                            case,
                            next(o for o in operations if o.mode == args.profile_mode),
                            args,
                        )
                    )
                else:
                    record.update(
                        time_case(
                            case, [o for o in operations if o.mode in modes], args
                        )
                    )
                record["valid"] = True
                write_results(payload, base)
                print(
                    json.dumps(
                        {k: record[k] for k in ("suite", "name", "valid", "execution")}
                    ),
                    flush=True,
                )
        payload["status"] = "passed"
    except BaseException as e:
        payload.update(status="failed", error=repr(e))
        raise
    finally:
        configure("native")
        payload["source"] = source_manifest()
        payload["source"]["binaries"][args.native_library] = hashlib.sha256(
            Path(args.native_library).read_bytes()
        ).hexdigest()
        payload["finished_at"] = dt.datetime.now(dt.timezone.utc).isoformat()
        write_results(payload, base)
    print(f'json_output={base.with_suffix(".json")}')


if __name__ == "__main__":
    main()
