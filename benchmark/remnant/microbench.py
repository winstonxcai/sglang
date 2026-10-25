"""Current Native, Packed Triton and Packed Kernel on one H100, without weights.

Author: Winston Cai.
Decode and post-compressor store are separate measurements. Native retains all
coordinates; Packed modes retain a configured multiple of 64, below 512.
A matched Packed Torch reference is a validity control, never a Native timing
baseline. Times are milliseconds per
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
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

import torch
from sgl_kernel import flash_mla
from sglang.benchmark.remnant.cases import (
    PROTOCOL,
    PREFILL_PROTOCOL,
    DecodeCase,
    FP4DecodeCase,
    PrefillReadCase,
    StoreCase,
    benchmark_modes,
    case_seed,
    case_suite,
    matrix_digest,
    paired_ratios,
    parser,
    prefill_path_layouts,
    selected_cases,
    store_layout,
    summary,
    verify_profile_replay,
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
    unpack_gather_bf16,
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


def configure(mode: str, keep_count: int = 256) -> None:
    benchmark_modes(keep_count, mode)
    remnant.configure_cache_format(
        "native" if mode == "native" else "remnant",
        "kernel" if mode == "native" else mode,
        keep_count=256 if mode == "native" else keep_count,
    )
    if remnant.cache_format() != ("native" if mode == "native" else "remnant"):
        raise RuntimeError(f"cache format did not resolve for {mode}")
    if mode != "native" and remnant.cache_implementation() != mode:
        raise RuntimeError(f"cache implementation did not resolve for {mode}")
    if mode != "native" and remnant.packed_profile().default_count != keep_count:
        raise RuntimeError(
            f"Packed survivor count did not resolve for {mode}/{keep_count}"
        )
    if mode != "native" and remnant.cache_store_implementation() != mode:
        raise RuntimeError(f"Packed store implementation did not resolve for {mode}")


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


def native_storage(rows: int, page_size: int = PAGE_SIZE) -> torch.Tensor:
    pages = (rows + page_size - 1) // page_size
    size = ((page_size * 584 + 575) // 576) * 576
    return torch.zeros((pages, size), device="cuda", dtype=torch.uint8)


def native_view(storage: torch.Tensor) -> torch.Tensor:
    return storage.as_strided(
        (len(storage), PAGE_SIZE, 1, 584), (storage.stride(0), 584, 584, 1)
    )


def repage_native(storage: torch.Tensor, rows: int, page_size: int) -> torch.Tensor:
    """Retile CPU-built page64 producer bytes for the SWA page geometry, untimed."""
    pages = (rows + page_size - 1) // page_size
    size = ((page_size * 584 + 575) // 576) * 576
    result = torch.zeros((pages, size), dtype=torch.uint8, device=storage.device)
    for width, source_start, target_start in (
        (576, 0, 0),
        (8, PAGE_SIZE * 576, page_size * 576),
    ):
        source = storage[:, source_start : source_start + PAGE_SIZE * width]
        target = result[:, target_start : target_start + page_size * width]
        # Page padding means reshape may copy. Assign through page views.
        flat = torch.zeros(
            (pages * page_size, width), dtype=torch.uint8, device=storage.device
        )
        flat[:rows].copy_(source.reshape(-1, width)[:rows])
        target.copy_(flat.reshape(pages, page_size * width))
    return result


def packed_storage(rows: int, keep_count: int = 256) -> PackedBuffers:
    pages = (rows + PAGE_SIZE - 1) // PAGE_SIZE
    return PackedBuffers(
        torch.zeros((pages, PAGE_SIZE, keep_count), device="cuda", dtype=torch.uint8),
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
            x,
            topmag_keep_mask(x, buffers.values.shape[-1] / 512),
            weight,
            EPS,
            metadata,
            locations,
            buffers,
        )
    else:
        pack_rows_fused(x, weight, EPS, metadata, locations, buffers)


def decode_native(storage: torch.Tensor, page_size: int = PAGE_SIZE) -> torch.Tensor:
    """Independent page reader: value bytes and scales have separate regions."""
    codes = storage[:, : page_size * 576].reshape(-1, 576)
    scales = storage[:, page_size * 576 : page_size * 584].reshape(-1, 8)
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
    keep_count: int = 256
    label: str | None = None
    validate_output: Callable | None = None

    @property
    def key(self) -> str:
        return self.label or self.mode


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


def fp4_decode_operations(case: FP4DecodeCase, seed: int, *_unused):
    """Paired production FP4 APIs; fixture/reference work is outside timing.

    Author: Winston Cai. CPU oracle independently quantizes and selects; Native
    is dense FP4, never V4 FP8 or a pruned matched control.
    """
    from sglang.kernels.ops.attention.dsv4.fp4_cache import store_fp4_cache
    from sglang.test.kernels.deepseek_v4.dsv41_kv_quant_reference import (
        quantize_k_cache_v41_fp4,
        dequantize_k_cache_v41_fp4,
    )

    page, rows = case.global_page_size, case.batch * 512
    rng = torch.Generator().manual_seed(case_seed(case, seed))
    # Post-normalization/post-RoPE BF16 producer boundary, shared by both paths.
    x = torch.randn((rows, 512), generator=rng).bfloat16()
    dense_bytes = quantize_k_cache_v41_fp4(x.reshape(-1, page, 512))
    native_values = dequantize_k_cache_v41_fp4(dense_bytes, page).reshape(rows, 512)
    selected = torch.argsort(native_values.abs(), dim=-1, descending=True, stable=True)[
        :, :256
    ]
    packed_values = torch.zeros_like(native_values).scatter_(
        1, selected, native_values.gather(1, selected)
    )
    q_cpu = torch.randn((case.batch, 1, case.heads, 512), generator=rng).bfloat16()
    q = q_cpu.cuda()
    physical_cpu = torch.randperm(rows, generator=rng).reshape(case.batch, 1, 512)
    physical = (physical_cpu + page).to(device="cuda", dtype=torch.int32)
    lengths = torch.full((case.batch,), 512, device="cuda", dtype=torch.int32)
    # Existing production V4 writer is also the unchanged H100 SWA writer.
    swa_x = torch.randn(
        (case.batch * case.swa_length, 512),
        device="cuda",
        generator=torch.Generator(device="cuda").manual_seed(case_seed(case, seed)),
    )
    weight = torch.ones(512, device="cuda", dtype=torch.float32)
    freqs = frequencies(len(swa_x) * 4 + 1)
    swa_plan = plan([4] * len(swa_x), list(range(len(swa_x))), "prefill")
    swa_storage = native_storage(len(swa_x))
    native_write(
        swa_x,
        weight,
        swa_plan,
        torch.arange(len(swa_x), device="cuda", dtype=torch.int64),
        freqs,
        swa_storage,
    )
    swa_values = (
        decode_native(swa_storage)[: len(swa_x)]
        .cpu()
        .reshape(case.batch, case.swa_length, 512)
    )
    swa_cache = repage_native(swa_storage, len(swa_x), case.swa_page_size)
    swa_view = swa_cache.as_strided(
        (len(swa_cache), case.swa_page_size, 1, 584), (swa_cache.stride(0), 584, 584, 1)
    )
    swa_indices = torch.arange(len(swa_x), device="cuda", dtype=torch.int32).reshape(
        case.batch, 1, case.swa_length
    )
    swa_lengths = torch.full_like(lengths, case.swa_length)
    sink_cpu = torch.linspace(-1, 1, case.heads)
    sink = sink_cpu.cuda()
    loc = torch.arange(page, page + rows, device="cuda", dtype=torch.int64)
    x_gpu = x.cuda()
    operations = []
    for mode, values, record_bytes in (
        ("native", native_values, 288),
        ("kernel", packed_values, 224),
    ):
        packed = mode == "kernel"
        cache = torch.zeros(
            (rows // page + 1, page * record_bytes), device="cuda", dtype=torch.uint8
        )
        store_fp4_cache(x_gpu, cache, loc, page, packed=packed)
        cache_view = cache.view(-1, page, 1, record_bytes)
        gathered = values[physical_cpu.long()]
        kv = torch.cat((swa_values[:, None], gathered), dim=2).float()
        scores = torch.einsum("bshd,bskd->bshk", q_cpu.float(), kv) * 512**-0.5
        lse = scores.logsumexp(-1)
        probabilities = torch.exp(
            scores - torch.logaddexp(lse, sink_cpu[None, None, :])[:, :, :, None]
        )
        expected = (
            torch.einsum("bshk,bskd->bshd", probabilities, kv).bfloat16(),
            lse.transpose(1, 2),
        )

        def call(cache_view=cache_view, packed=packed):
            return flash_mla.flash_mla_with_fp4_kvcache(
                q,
                swa_view,
                swa_indices,
                cache_view,
                physical,
                packed=packed,
                topk_length=swa_lengths,
                extra_topk_length=lengths,
                attn_sink=sink,
            )

        def validate_output(output, expected=expected, mode=mode):
            for actual, target in zip(output, expected):
                torch.testing.assert_close(actual.cpu(), target, atol=ATOL, rtol=RTOL,
                                           msg=f"{case.name}/{mode}")

        def validate(call=call, validate_output=validate_output):
            validate_output(call())

        operations.append(
            Operation(
                mode=mode,
                call=call,
                owned=(cache, q, physical, swa_cache, sink, lengths, swa_lengths),
                validate=validate,
                expected_kernels=(),
                keep_count=256,
                validate_output=validate_output,
            )
        )
    return operations, {
        "common_inputs_sha256": tensor_hash(
            {
                "x": x,
                "q": q_cpu,
                "physical": physical_cpu,
                "swa": swa_values,
                "sink": sink_cpu,
            }
        ),
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
        "global_page_size": page,
        "baseline": "unpruned dense Native FP4",
    }


def decode_operations(
    case: DecodeCase,
    seed: int,
    keep_count: int = 256,
    modes: tuple[str, ...] | None = None,
) -> tuple[list[Operation], dict]:
    modes = benchmark_modes(keep_count, None if modes is None else ",".join(modes))
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
    for mode in modes:
        if mode == "native":
            continue
        configure(mode, keep_count)
        caches[mode] = packed_storage(rows, keep_count)
        packed_write(mode, x, weight, metadata, loc, caches[mode])
    if "triton" in caches and "kernel" in caches:
        check_packed_bytes(caches["triton"], caches["kernel"], case.name)
    matched = None
    for mode, cache in caches.items():
        buffers = tuple(t.flatten(0, 1)[:rows] for t in cache)
        mask = topmag_keep_mask(x, keep_count / 512)
        expected = pack_rows_ref(x, mask, weight, EPS, keep_count=keep_count)
        assert torch.equal(
            buffers[1], expected[1]
        ), f"{case.name}/{mode}: selection differs"
        decoded = unpack_rows_ref(*buffers)
        check_fp8_storage(
            decoded, buffers[2], x, weight, EPS, case.name + "/" + mode, mask
        )
        matched = rotate_tail(decoded, raw_rows * 4, freqs)
    operations = []
    for mode in modes:
        configure(mode, keep_count)
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
            configure(mode, keep_count)
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
                keep_count,
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
        "packed_bytes_sha256": (
            tensor_hash(dict(buffers=next(iter(caches.values())))) if caches else None
        ),
        "packed_keep_count": keep_count,
        "packed_record_bytes": keep_count + 72,
        "native_baseline": "unpruned production Native",
        "page_size": PAGE_SIZE,
    }


def store_operations(
    case: StoreCase,
    seed: int,
    keep_count: int = 256,
    modes: tuple[str, ...] | None = None,
) -> tuple[list[Operation], dict]:
    modes = benchmark_modes(keep_count, None if modes is None else ",".join(modes))
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
    caches = {m: packed_storage(capacity, keep_count) for m in modes if m != "native"}
    expected_packed = (
        pack_rows_ref(
            x, topmag_keep_mask(x, keep_count / 512), weight, EPS, keep_count=keep_count
        )
        if caches
        else None
    )
    operations = []
    for mode in modes:

        def call(mode=mode):
            if mode == "native":
                native_write(x, weight, native_plan, native_loc, freqs, native_cache)
                return native_cache
            packed_write(mode, x, weight, metadata, loc, caches[mode])
            return caches[mode]

        def validate(mode=mode, call=call):
            configure(mode, keep_count)
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
                        topmag_keep_mask(x[ids], keep_count / 512),
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
                keep_count,
            )
        )
    for op in operations:
        op.validate()
    if "triton" in caches and "kernel" in caches:
        check_packed_bytes(caches["triton"], caches["kernel"], case.name)
    return operations, {
        "common_inputs_sha256": tensor_hash(
            dict(x=x, weight=weight, plan=metadata[1], locations=loc, freqs=freqs)
        ),
        "valid_store_rows": len(valid),
        "capacity_rows": capacity,
        "packed_keep_count": keep_count,
        "packed_record_bytes": keep_count + 72,
        "dtype": "float32",
        "native_invalid_destinations": "represented by skipped plan entries before timing",
    }


def prefill_read_operations(
    case: PrefillReadCase,
    seed: int,
    *,
    fused_rope: bool = False,
) -> tuple[list[Operation], dict]:
    """Production C4/SWA preparation and sparse prefill, with variant-owned banks.

    Queries are BF16 [Q,H,512]. C4 rows are raw FP32 [S//4,512], stored in
    shuffled physical slots; raw IDs retain their logical RoPE positions.
    Each bank owns its persistent caches and BF16 [C4+SWA,1,512] workspace.
    Four rotating banks approximate a larger working set, without a timed flush.
    """
    from sglang.kernels.ops.attention.dsv4.dequant_k_cache import (
        dequantize_k_cache_paged,
    )
    from sglang.srt.layers.attention.dsv4.sparse_prefill_utils import (
        combine_topk_swa_indices,
    )

    rng = torch.Generator(device="cuda").manual_seed(case_seed(case, seed))
    rows = case.history_length // 4
    swa_rows = case.query_length + case.swa_length - 1
    x = torch.randn((rows, 512), device="cuda", dtype=torch.float32, generator=rng)
    weight = torch.linspace(0.5, 1.5, 512, device="cuda", dtype=torch.float32)
    q = torch.randn(
        (case.query_length, case.heads, 512),
        device="cuda",
        dtype=torch.bfloat16,
        generator=rng,
    )
    freqs = frequencies(case.history_length + 4)
    raw = torch.arange(rows, device="cuda", dtype=torch.int32)
    gather_lengths = torch.full((1,), rows, device="cuda", dtype=torch.int32)
    physical = torch.randperm(rows, device="cuda", generator=rng).to(torch.int32)
    metadata = plan((raw + 1) * 4, raw, "prefill")
    native = native_storage(rows)
    native_write(x, weight, metadata, physical.long(), freqs, native)
    check_native_storage(
        native, x, weight, raw * 4, freqs, case.name + "/native", physical.long()
    )
    swa_x = torch.randn(
        (swa_rows, 512), device="cuda", dtype=torch.float32, generator=rng
    )
    swa_positions = torch.arange(
        case.history_length - swa_rows,
        case.history_length,
        device="cuda",
        dtype=torch.int32,
    )
    swa_plan = plan(swa_positions + 4, torch.arange(swa_rows, device="cuda"), "prefill")
    swa_source = native_storage(swa_rows)
    native_write(
        swa_x,
        weight,
        swa_plan,
        torch.arange(swa_rows, device="cuda"),
        freqs,
        swa_source,
    )
    check_native_storage(
        swa_source, swa_x, weight, swa_positions, freqs, case.name + "/swa"
    )
    swa_source = repage_native(swa_source, swa_rows, case.swa_page_size)
    swa_expected = decode_native(swa_source, case.swa_page_size)[:swa_rows]
    swa_ids = torch.arange(swa_rows, device="cuda", dtype=torch.int32)
    positions = torch.arange(
        case.history_length - case.query_length,
        case.history_length,
        device="cuda",
        dtype=torch.int32,
    )
    available = ((positions + 1) // 4).clamp_max(rows)
    selected = torch.randint(
        0,
        rows,
        (case.query_length, case.topk),
        device="cuda",
        dtype=torch.int32,
        generator=rng,
    )
    selected = (selected.long() % available[:, None]).to(torch.int32)
    combined, lengths = combine_topk_swa_indices(
        topk_indices=selected,
        query_pos=positions,
        query_start_loc=torch.tensor(
            [0, case.query_length], device="cuda", dtype=torch.int32
        ),
        seq_lens=torch.tensor([case.history_length], device="cuda", dtype=torch.int32),
        gather_lens=torch.tensor([swa_rows], device="cuda", dtype=torch.int32),
        compressed_base=torch.tensor([0], device="cuda", dtype=torch.int32),
        swa_base=torch.tensor([rows], device="cuda", dtype=torch.int32),
        window_size=case.swa_length,
        compress_ratio=4,
        topk=case.topk,
    )
    indices = combined[:, None].contiguous()
    sink = torch.linspace(-1, 1, case.heads, device="cuda", dtype=torch.float32)
    sample_ids = (
        torch.linspace(
            0, case.query_length - 1, min(8, case.query_length), device="cuda"
        )
        .long()
        .unique()
    )
    banks = 4 if case.cache_policy == "rotating" else 1
    common_hash = tensor_hash(
        dict(
            x=x,
            weight=weight,
            q=q,
            physical=physical,
            raw=raw,
            freqs=freqs,
            swa=swa_expected,
            indices=indices,
            lengths=lengths,
            sink=sink,
        )
    )
    operations = []
    for label, layout in prefill_path_layouts().items():
        mode = layout["implementation"]
        keep_count = layout["packed_keep_count"] or 256
        configure(mode, keep_count)
        if mode == "native":
            source = native
            expected_c4 = decode_native(native)[physical.long()]
        else:
            source = packed_storage(rows, keep_count)
            packed_write("triton", x, weight, metadata, physical.long(), source)
            actual = tuple(
                indexed_rows(t.flatten(0, 1), physical.long()) for t in source
            )
            mask = topmag_keep_mask(x, keep_count / 512)
            expected = pack_rows_ref(x, mask, weight, EPS, keep_count=keep_count)
            assert torch.equal(
                actual[1], expected[1]
            ), f"{case.name}/{label}: selection differs"
            decoded = unpack_rows_ref(*actual)
            check_fp8_storage(
                decoded, actual[2], x, weight, EPS, case.name + "/" + label, mask
            )
            expected_c4 = rotate_tail(decoded, raw * 4, freqs)
        expected_workspace = torch.cat((expected_c4, swa_expected), dim=0)
        sources = [
            (
                source
                if i == 0
                else (
                    source.clone()
                    if mode == "native"
                    else PackedBuffers(*(t.clone() for t in source))
                )
            )
            for i in range(banks)
        ]
        swa_sources = [swa_source.clone() for _ in range(banks)]
        workspaces = [
            torch.empty((rows + swa_rows, 1, 512), device="cuda", dtype=torch.bfloat16)
            for _ in range(banks)
        ]
        # Candidate qualification uses a distinct, retained split-reader buffer.
        # Its allocation and every baseline comparison stay outside timing and
        # profiler capture; Native's measured path remains unchanged.
        split_workspace = (
            torch.empty_like(workspaces[0][:rows])
            if fused_rope and mode != "native"
            else None
        )

        def reconstruct(
            bank,
            mode=mode,
            sources=sources,
            swa_sources=swa_sources,
            workspaces=workspaces,
        ):
            workspace = workspaces[bank]
            if mode == "native":
                dequantize_k_cache_paged(
                    sources[bank], physical, PAGE_SIZE, out=workspace[:rows]
                )
            else:
                unpack_gather_bf16(
                    sources[bank],
                    physical[None],
                    raw[None],
                    gather_lengths,
                    freqs,
                    workspace[:rows],
                    fused_rope=fused_rope,
                )
            dequantize_k_cache_paged(
                swa_sources[bank], swa_ids, case.swa_page_size, out=workspace[rows:]
            )
            return workspace

        def attention(bank, workspaces=workspaces):
            return flash_mla.flash_mla_sparse_fwd(
                q, workspaces[bank], indices, 512**-0.5, 512, sink, lengths
            )

        for bank in range(banks):
            reconstruct(bank)
        cursor = [0]

        def call(cursor=cursor, reconstruct=reconstruct, attention=attention):
            bank = cursor[0] % banks
            cursor[0] += 1
            if case.stage == "reconstruction":
                return reconstruct(bank)
            if case.stage == "complete":
                reconstruct(bank)
            return attention(bank)

        def validate(
            reconstruct=reconstruct,
            attention=attention,
            mode=mode,
            keep_count=keep_count,
            expected_workspace=expected_workspace,
            workspaces=workspaces,
            cursor=cursor,
            label=label,
            split_workspace=split_workspace,
            source_banks=sources,
        ):
            configure(mode, keep_count)
            for bank in range(banks):
                reconstruct(bank)
                if split_workspace is not None:
                    unpack_gather_bf16(
                        source_banks[bank],
                        physical[None],
                        raw[None],
                        gather_lengths,
                        freqs,
                        split_workspace,
                    )
                    if not torch.equal(
                        workspaces[bank][:rows].view(torch.int16),
                        split_workspace.view(torch.int16),
                    ):
                        raise RuntimeError(
                            f"{case.name}/{label}: fused/split BF16 byte parity failed"
                        )
                if workspaces[bank].dtype != torch.bfloat16:
                    raise RuntimeError(f"{case.name}/{label}: expected BF16 workspace")
                # Promote only the validation view to the independent FP32
                # reference's dtype; the measured buffer remains BF16.
                torch.testing.assert_close(
                    workspaces[bank][:, 0].float(),
                    expected_workspace,
                    atol=ATOL,
                    rtol=RTOL,
                )
            out, max_logits, lse = attention(0)
            if not bool(torch.isfinite(out).all()):
                raise RuntimeError(f"{case.name}/{label}: nonfinite prefill output")
            selected_ids = indices[sample_ids, 0].long()
            valid = (
                torch.arange(indices.shape[-1], device="cuda")[None]
                < lengths[sample_ids, None]
            ) & (selected_ids >= 0)
            kv = expected_workspace[selected_ids.clamp_min(0)].float()
            scores = q[sample_ids].float() @ kv.transpose(-1, -2) * 512**-0.5
            scores.masked_fill_(~valid[:, None], -math.inf)
            expected_lse = scores.logsumexp(-1)
            denominator = torch.logaddexp(expected_lse, sink[None])
            expected_out = ((scores - denominator[..., None]).exp() @ kv).to(
                torch.bfloat16
            )
            torch.testing.assert_close(
                out[sample_ids], expected_out, atol=ATOL, rtol=RTOL
            )
            torch.testing.assert_close(
                max_logits[sample_ids],
                scores.amax(-1) / math.log(2),
                atol=ATOL,
                rtol=RTOL,
            )
            # SM90 phase1 returns natural logs, but the compiled SGL
            # compatibility shim (FlashMLA csrc/python_api.cpp) multiplies
            # both max_logits and LSE by LOG_2_E. Check the public API units;
            # keep the independent softmax/sink arithmetic above in ln space.
            torch.testing.assert_close(
                lse[sample_ids], expected_lse / math.log(2), atol=ATOL, rtol=RTOL
            )
            cursor[0] = 0

        kernels = []
        if case.stage != "attention":
            kernels.append("dequantize_k_cache_paged")
            if mode != "native":
                kernels += (
                    ["unpack_gather_bf16_rope"]
                    if fused_rope
                    else ["unpack_gather_bf16_kernel", "rope_tail_complex"]
                )
        if case.stage != "reconstruction":
            kernels += ["sparse_attn_fwd_kernel", "elementwise.*MulFunctor"]
        operations.append(
            Operation(
                mode,
                call,
                (
                    q,
                    sources,
                    swa_sources,
                    workspaces,
                    indices,
                    lengths,
                    sink,
                    freqs,
                    physical,
                    raw,
                    expected_workspace,
                    x,
                    weight,
                    split_workspace,
                ),
                validate,
                tuple(kernels),
                keep_count,
                label,
            )
        )
    for op in operations:
        op.validate()
    return operations, {
        "common_inputs_sha256": common_hash,
        "prefill_reader": "fused" if fused_rope else "split",
        "path_layouts": prefill_path_layouts(),
        "c4_rows": rows,
        "swa_rows": swa_rows,
        "workspace_bytes": (rows + swa_rows) * 1024,
        "cache_policy": case.cache_policy,
        "buffer_banks": banks,
        "validation": {
            "workspace_rows": rows + swa_rows,
            "sampled_query_ids": sample_ids.tolist(),
            "attention_reference": "independent Torch softmax/sink; SGL compatibility log2 max/LSE",
            "packed_reader_reference": (
                "bitwise split-reader parity for every C4 row and bank"
                if fused_rope
                else "independent BF16 workspace reference"
            ),
        },
        "fixture": "serving-informed synthetic, one request; metadata and cache construction untimed",
    }


def capture(op: Operation, warmup: int) -> tuple[torch.cuda.CUDAGraph, object]:
    configure(op.mode, op.keep_count)
    for _ in range(max(3, warmup)):
        op.call()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        output = op.call()
    graph.replay()
    torch.cuda.synchronize()
    if op.validate_output is not None:
        op.validate_output(output)
    else:
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
            calls[op.key] = graph.replay
            retained.append((graph, output, op))
        else:
            configure(op.mode, op.keep_count)
            for _ in range(args.warmup):
                op.call()
            calls[op.key] = op.call
    samples = {op.key: [] for op in operations}
    by_key = {op.key: op for op in operations}
    rng = random.Random(case_seed(case, args.seed))
    for _ in range(args.rounds):
        order = list(samples)
        rng.shuffle(order)
        for key in order:
            op = by_key[key]
            configure(op.mode, op.keep_count)
            samples[key].append(measure(calls[key], repeats))
    for op in operations:
        op.validate()
    for _, output, op in retained:
        if op.validate_output is not None:
            op.validate_output(output)
    return {
        "execution": "cuda_graph" if graph_mode else "eager",
        "repeats": repeats,
        "paths": {m: summary(v) for m, v in samples.items()},
        "paired_ratios": paired_ratios(samples),
    }


def profile_case(case, op: Operation, args) -> dict:
    op.validate()
    configure(op.mode, op.keep_count)
    for _ in range(max(3, args.warmup)):
        op.call()
    torch.cuda.synchronize()
    start = torch.cuda.cudart().cudaProfilerStart()
    if (start[0] if isinstance(start, tuple) else start) != 0:
        raise RuntimeError("cudaProfilerStart failed")
    capture_range = f"remnant_{case.name}_{op.key}"
    try:
        # NVTX identifies each operation inside a batched application replay.
        # It wraps only the warmed operation, never fixture setup/validation.
        context = (
            torch.cuda.nvtx.range(capture_range)
            if isinstance(case, PrefillReadCase)
            else nullcontext()
        )
        with context:
            output = op.call()
            torch.cuda.synchronize()
    finally:
        stop = torch.cuda.cudart().cudaProfilerStop()
        if (stop[0] if isinstance(stop, tuple) else stop) != 0:
            raise RuntimeError("cudaProfilerStop failed")
    op.validate()
    minimum_launches = {}
    if isinstance(case, PrefillReadCase):
        if case.stage != "attention":
            minimum_launches["dequantize_k_cache_paged"] = (
                2 if op.mode == "native" else 1
            )
        if case.stage != "reconstruction":
            # The production compatibility wrapper converts max and LSE
            # separately; both ATen kernels belong to the measured operation.
            minimum_launches["elementwise.*MulFunctor"] = 2
    return {
        "execution": "profiled_eager",
        "mode": op.mode,
        "path": op.key,
        "keep_count": op.keep_count,
        "calls": 1,
        "capture_range": capture_range,
        "expected_kernels": op.expected_kernels,
        "minimum_launches": minimum_launches,
        "acceptance_timing": False,
    }


def source_manifest() -> dict:
    import sglang.benchmark.remnant.cases as cases_module
    import sglang.benchmark.remnant.quantization as quantization_module
    import sglang.kernels.ops.attention.dsv4.compress as compress_module
    import sglang.srt.remnant.packed as packed_module
    from sglang.kernels.ops.attention.dsv4 import (
        dequant_k_cache,
        sparse_prefill_kernels,
    )
    from sglang.srt.layers.attention.dsv4 import sparse_prefill_utils

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
        Path(dequant_k_cache.__file__),
        Path(sparse_prefill_kernels.__file__),
        Path(sparse_prefill_utils.__file__),
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
    is_prefill = args.suite == "prefill-read"
    is_fp4 = args.suite == "v41-fp4-decode"
    if is_fp4:
        args.modes = "native,kernel"
    modes = () if is_prefill else benchmark_modes(args.keep_count, args.modes)
    if not is_prefill:
        args.modes = ",".join(modes)
    if args.check_imports:
        from sgl_kernel import remnant_ops  # noqa: F401 -- registers checked operator

        if getattr(flash_mla, "_flashmla_import_error", None) is not None:
            raise RuntimeError(
                "verified FlashMLA extension did not load"
            ) from flash_mla._flashmla_import_error
        if not hasattr(torch.ops.sgl_kernel, "remnant_sparse_decode_fwd"):
            raise RuntimeError("verified Direct Packed operator is missing")
        if is_prefill and not hasattr(torch.ops.sgl_kernel, "sparse_prefill_fwd"):
            raise RuntimeError("verified sparse prefill operator is missing")
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
    replay_payload = None
    if base.with_suffix(".json").exists() or base.with_suffix(".csv").exists():
        if not base.with_suffix(".json").is_file():
            raise FileExistsError(f"refusing to overwrite {base}")
        replay_payload = json.loads(base.with_suffix(".json").read_text())
        verify_profile_replay(replay_payload, vars(args))
        if replay_payload.get("matrix_sha256") != matrix_digest(cases) or [
            r["name"] for r in replay_payload["cases"]
        ] != [c.name for c in cases]:
            raise RuntimeError("Profiler application replay matrix changed")
        for path, digest in replay_payload.get("source", {}).get("sources", {}).items():
            if hashlib.sha256(Path(path).read_bytes()).hexdigest() != digest:
                raise RuntimeError(
                    f"Profiler application replay source changed: {path}"
                )
        for path, digest in (
            replay_payload.get("source", {}).get("binaries", {}).items()
        ):
            if hashlib.sha256(Path(path).read_bytes()).hexdigest() != digest:
                raise RuntimeError(
                    f"Profiler application replay binary changed: {path}"
                )
    can_write = replay_payload is None
    payload = {
        "schema_version": 2,
        "protocol": "remnant-v41-fp4-decode-v1"
        if is_fp4
        else (PREFILL_PROTOCOL if is_prefill else PROTOCOL),
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
        "scope": (
            "model-free historical C4/SWA reconstruction, attention and complete prefill-read path"
            if is_prefill
            else "model-free production functions; decode and post-compressor store measured separately"
        ),
        "started_at": stamp,
        "cases": [],
        "status": "running",
    }
    try:
        with torch.inference_mode():
            for case in cases:
                factory = (
                    prefill_read_operations
                    if isinstance(case, PrefillReadCase)
                    else (
                        (
                            fp4_decode_operations
                            if isinstance(case, FP4DecodeCase)
                            else decode_operations
                        )
                        if isinstance(case, DecodeCase)
                        else store_operations
                    )
                )
                record = {
                    "suite": case_suite(case),
                    "name": case.name,
                    "case": asdict(case),
                    "valid": False,
                }
                payload["cases"].append(record)
                if can_write:
                    write_results(payload, base)
                operations, identity = (
                    factory(case, args.seed, fused_rope=args.prefill_reader == "fused")
                    if is_prefill
                    else factory(case, args.seed, args.keep_count, modes)
                )
                record.update(identity)
                if replay_payload is not None:
                    prior = next(
                        (r for r in replay_payload["cases"] if r["name"] == case.name),
                        None,
                    )
                    if (
                        prior is None
                        or identity["common_inputs_sha256"]
                        != prior["common_inputs_sha256"]
                    ):
                        raise RuntimeError(
                            "Profiler application replay input hash changed"
                        )
                    can_write = True
                if args.profile_mode or args.profile_path:
                    record.update(
                        profile_case(
                            case,
                            next(
                                o
                                for o in operations
                                if (
                                    o.key == args.profile_path
                                    if is_prefill
                                    else o.mode == args.profile_mode
                                )
                            ),
                            args,
                        )
                    )
                else:
                    record.update(time_case(case, operations, args))
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
        if can_write:
            write_results(payload, base)
    print(f'json_output={base.with_suffix(".json")}')


if __name__ == "__main__":
    main()
