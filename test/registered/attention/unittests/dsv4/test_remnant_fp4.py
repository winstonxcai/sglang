"""Model-free SM90 FP4 contracts using the upstream independent CPU oracle.

Author: Winston Cai.
These fixtures qualify storage/attention engineering, never model/task quality.
"""

from __future__ import annotations

import gc
import json
import math
import os
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from sglang.kernels.ops.attention.dsv4.kv_layout import KVLayout
from sglang.srt.remnant.config import validate_v41_cache_args
from sglang.test.kernels.deepseek_v4.dsv41_kv_quant_reference import (
    dequantize_k_cache_v41_fp4,
    quantize_k_cache_v41_fp4,
)

from .test_remnant_flashmla_direct import _make_swa_cache
from .test_remnant_pool import _gpu_storages

GPU = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires Modal H100")
RATIOS = [0] * 2 + [2] * 18 + [1] * 20
SOURCES = (2, 8, 14, 20)


def _oracle(x: torch.Tensor, packed: bool) -> tuple[torch.Tensor, torch.Tensor]:
    """CPU-only oracle: dense quantization precedes independent stable selection."""
    assert x.device.type == "cpu" and x.ndim == 3
    x = x.detach()
    dense = quantize_k_cache_v41_fp4(x)
    p = x.shape[1]
    values = dequantize_k_cache_v41_fp4(dense, p)
    if not packed:
        return dense, values
    pages = x.shape[0]
    out = torch.zeros((pages, p * 224), dtype=torch.uint8)
    data = dense[:, : p * 256].reshape(pages, p, 256)
    codes = torch.stack((data & 15, data >> 4), dim=-1).flatten(-2)
    selected = (
        torch.argsort(values.abs(), dim=-1, descending=True, stable=True)[..., :256]
        .sort(dim=-1)
        .values
    )
    kept_codes = codes.gather(-1, selected)
    dst = out[:, : p * 192].reshape(pages, p, 192)
    dst[..., :128] = kept_codes[..., 0::2] | (kept_codes[..., 1::2] << 4)
    mask = torch.zeros_like(values, dtype=torch.bool).scatter_(-1, selected, True)
    # Form bytes in coordinate order, then reverse each word's eight bytes for
    # little-endian uint64 storage. This avoids production rank/prefix logic.
    weights = torch.tensor([128, 64, 32, 16, 8, 4, 2, 1], dtype=torch.int64)
    bitmap_bytes = (mask.reshape(pages, p, 8, 8, 8) * weights).sum(-1).flip(-1)
    dst[..., 128:] = bitmap_bytes.flatten(-2).to(torch.uint8)
    values.masked_fill_(~mask, 0)
    out[:, p * 192 :] = dense[:, p * 256 : p * 288]
    return out, values


def test_fp4_startup_contract(monkeypatch):
    from sglang.srt.environ import envs

    for field in (envs.SGLANG_DSV4_KV_LAYOUT, envs.SGLANG_DSV4_COMPRESSED_KV_LAYOUT):
        monkeypatch.delenv(field.name, raising=False)
    validate_v41_cache_args(SimpleNamespace())
    for mode in ("fp4", "remnant"):
        validate_v41_cache_args(SimpleNamespace(dsv41_main_kv_cache_format=mode))
        validate_v41_cache_args(
            SimpleNamespace(dsv41_main_kv_cache_format=mode, dsv4_attn_backend="auto")
        )
        for field in (
            envs.SGLANG_DSV4_KV_LAYOUT,
            envs.SGLANG_DSV4_COMPRESSED_KV_LAYOUT,
        ):
            with field.override("auto"):
                validate_v41_cache_args(
                    SimpleNamespace(dsv41_main_kv_cache_format=mode)
                )
            with field.override("v4"):
                with pytest.raises(ValueError, match="legacy SGLANG_DSV4"):
                    validate_v41_cache_args(
                        SimpleNamespace(dsv41_main_kv_cache_format=mode)
                    )
        for key, value in [
            ("enable_hierarchical_cache", True),
            ("speculative_algorithm", "DSpark"),
            ("dsv4_c4_cache_format", "remnant"),
            ("dsv4_c4_cache_keep_count", 192),
            ("disaggregation_mode", "decode"),
            ("enable_encoder_swa_bounded_replay", True),
        ]:
            with pytest.raises(ValueError, match=key):
                validate_v41_cache_args(
                    SimpleNamespace(dsv41_main_kv_cache_format=mode, **{key: value})
                )
    with pytest.raises(ValueError):
        validate_v41_cache_args(SimpleNamespace(dsv41_main_kv_cache_format="fp8"))
    import argparse
    from sglang.srt.server_args import ServerArgs
    from sglang.srt.runtime_context import get_context, get_model

    parser = argparse.ArgumentParser()
    ServerArgs.add_cli_args(parser)
    args = ServerArgs.from_cli_args(
        parser.parse_args(
            ["--model", "dummy", "--dsv41-main-kv-cache-format", "remnant"]
        )
    )
    assert args.dsv41_main_kv_cache_format == "remnant"
    with get_context().override_server_args(dsv41_main_kv_cache_format="remnant"):
        assert get_model().dsv41_main_kv_cache_format == "remnant"
    assert KVLayout.V41_FP4.bytes_per_token == 288
    assert KVLayout.V41_FP4_REMNANT.bytes_per_token == 224
    for p in (128, 256):
        assert KVLayout.V41_FP4_REMNANT.page_bytes(p) == p * 224


@GPU
@pytest.mark.parametrize("packed", [False, True])
@pytest.mark.parametrize("page", [128, 256])
@pytest.mark.parametrize("pattern", ["random", "ties", "zeros", "boundaries"])
def test_fp4_store_and_reader(packed, page, pattern):
    from sglang.kernels.ops.attention.dsv4.fp4_cache import (
        read_fp4_cache,
        store_fp4_cache,
    )

    g = torch.Generator().manual_seed(73)
    x = torch.randn((page, 512), generator=g).bfloat16()
    if pattern == "ties":
        x[:] = 1
        x[:, 448:] = 2  # selected tail coordinates exercise bitmap word 7
    elif pattern == "zeros":
        x.zero_()
        x[:, 1::2] = -0.0
    elif pattern == "boundaries":
        x[:] = torch.tensor(
            [0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0, 6.0] * 64
        ).bfloat16()
    # Per-row RoPE, BF16 tail rounding, then quantize exactly once.
    angle = torch.arange(page * 32, dtype=torch.float32).reshape(page, 32) * 0.003
    freq = torch.polar(torch.ones_like(angle), angle)
    rotated = x.clone()
    pairs = x[:, 448:].float().reshape(page, 32, 2)
    rot = torch.view_as_complex(pairs) * freq
    rotated[:, 448:] = torch.view_as_real(rot).flatten(1).bfloat16()
    source = torch.zeros((2, page, 512), dtype=torch.bfloat16)
    source[1] = rotated
    expected, values = _oracle(source, packed)
    # Page zero is reserved and remains byte-zero (not quantized zero scales).
    expected[0].zero_()
    cache = torch.zeros_like(expected, device="cuda")
    locations = torch.arange(page, page * 2, device="cuda", dtype=torch.int64)
    x_gpu, f_gpu = x.cuda(), freq.cuda()
    store_fp4_cache(x_gpu, cache, locations, page, packed=packed, freqs_cis=f_gpu)
    torch.testing.assert_close(cache.cpu(), expected, rtol=0, atol=0)
    before = cache.clone()
    bad = torch.tensor([0, -1, page * 2], device="cuda", dtype=torch.int64)
    store_fp4_cache(x_gpu[:3], cache, bad, page, packed=packed, freqs_cis=f_gpu[:3])
    assert torch.equal(cache, before)
    ids = torch.tensor(
        [page, page + 1, 2 * page - 1, page, -1, 2 * page, 2**31 - 1],
        device="cuda",
        dtype=torch.int32,
    )
    out = torch.empty((ids.numel() + 2, 1, 512), device="cuda", dtype=torch.bfloat16)[
        1:-1
    ]
    assert read_fp4_cache(cache, ids, page, packed=packed, out=out) is out
    target = torch.cat(
        (values[1, [0, 1, page - 1, 0]], torch.zeros((3, 512), dtype=torch.bfloat16))
    )
    torch.testing.assert_close(out[:, 0].cpu(), target, rtol=0, atol=0)


def _attention_case(packed: bool, heads: int, page: int):
    from sglang.kernels.ops.attention.dsv4.fp4_cache import store_fp4_cache

    torch.manual_seed(37)
    x = torch.randn((2, page, 512)).bfloat16()
    x[0].zero_()
    expected, values = _oracle(x, packed)
    expected[0].zero_()
    cache = torch.zeros_like(expected, device="cuda")
    loc = torch.arange(page, 2 * page, device="cuda", dtype=torch.int64)
    x_gpu = x[1].cuda()
    store_fp4_cache(x_gpu, cache, loc, page, packed=packed)
    swa = _make_swa_cache(2, 128, torch.device("cuda"))
    swa_raw = swa.view(torch.uint8).as_strided((2, swa.stride(0)), (swa.stride(0), 1))
    # Balance SWA and global contributions so loader faults cannot hide behind
    # a large SWA output or the relative-tolerance allowance. UE8M0 120 = 2^-7.
    swa_raw[:, 128 * 576 : 128 * 584].fill_(120)
    swa_values = torch.cat(
        (
            swa_raw[:, : 128 * 576]
            .reshape(2, 128, 576)[:, :, :448]
            .contiguous()
            .view(torch.float8_e4m3fn)
            .bfloat16()
            * 2.0**-7,
            swa_raw[:, : 128 * 576]
            .reshape(2, 128, 576)[:, :, 448:]
            .contiguous()
            .view(torch.bfloat16),
        ),
        dim=-1,
    ).reshape(-1, 512)
    b = 8
    q = torch.randn((b, 1, heads, 512), device="cuda").bfloat16()
    indices = (
        torch.arange(64, device="cuda", dtype=torch.int32)
        .reshape(1, 1, 64)
        .expand(b, 1, 64)
        .contiguous()
    )
    extra = (
        (torch.arange(512, device="cuda", dtype=torch.int32) % page + page)
        .reshape(1, 1, 512)
        .expand(b, 1, 512)
        .contiguous()
    )
    lengths = torch.tensor(
        [0, 1, 63, 64, 65, 317, 511, 512], device="cuda", dtype=torch.int32
    )
    valid = torch.arange(512, device="cuda")[None, None, :] < lengths[:, None, None]
    extra[~valid] = 2**31 - 1
    extra[:, 0, 2] = -1
    extra[-1, 0, 5] = 2**31 - 1  # invalid physical selection inside the live length
    sink = torch.linspace(-1, 1, heads, device="cuda")
    return SimpleNamespace(
        q=q,
        swa=swa,
        indices=indices,
        cache=cache,
        cache_view=cache.view(2, page, 1, 224 if packed else 288),
        extra=extra,
        lengths=lengths,
        swa_values=swa_values,
        values=values.reshape(-1, 512).cuda(),
        sink=sink,
        x=x_gpu,
        loc=loc,
        page=page,
        packed=packed,
    )


def _attention_reference(t):
    idx = t.extra.long()
    mask = (
        (idx >= 0)
        & (idx < t.values.shape[0])
        & (torch.arange(512, device="cuda")[None, None, :] < t.lengths[:, None, None])
    )
    gathered = t.values[idx.clamp(0, t.values.shape[0] - 1)]
    kv = torch.cat((t.swa_values[t.indices.long()], gathered), dim=2).float()
    valid = torch.cat((torch.ones_like(t.indices, dtype=torch.bool), mask), dim=-1)
    scores = torch.einsum("bshd,bskd->bshk", t.q.float(), kv) * 512**-0.5
    scores.masked_fill_(~valid[:, :, None, :], -float("inf"))
    lse = scores.logsumexp(-1)
    weights = torch.exp(
        scores - torch.logaddexp(lse, t.sink[None, None, :])[:, :, :, None]
    )
    out = torch.einsum("bshk,bskd->bshd", weights, kv).bfloat16()
    return out, lse.transpose(1, 2)


@GPU
@pytest.mark.parametrize("packed", [False, True])
@pytest.mark.parametrize("heads", [64, 128])
@pytest.mark.parametrize("page", [128, 256])
def test_fp4_direct(packed, heads, page):
    from sgl_kernel.flash_mla import flash_mla_with_fp4_kvcache

    t = _attention_case(packed, heads, page)
    actual = flash_mla_with_fp4_kvcache(
        t.q,
        t.swa,
        t.indices,
        t.cache_view,
        t.extra,
        packed=packed,
        extra_topk_length=t.lengths,
        attn_sink=t.sink,
    )
    expected = _attention_reference(t)
    original = t.values
    try:
        for fault in ("missing-global", "missing-global-tail"):
            t.values = original.clone()
            if fault == "missing-global":
                t.values.zero_()
            else:
                t.values[:, 448:].zero_()
            faulty_output = _attention_reference(t)[0]
            assert not torch.allclose(
                expected[0], faulty_output, atol=0.02, rtol=0.02
            ), (
                f"Insensitive attention fixture: {fault}, packed={packed}, "
                f"heads={heads}, page={page}"
            )
    finally:
        t.values = original
    for a, e in zip(actual, expected):
        torch.testing.assert_close(a, e, atol=0.02, rtol=0.02)
    for use_lengths in (False, True):
        adapter = _fp4_adapter(t, use_lengths=use_lengths)
        torch.testing.assert_close(actual[0][:, 0], adapter, atol=0.02, rtol=0.02)


def _fp4_adapter(t, *, use_lengths):
    """Exercise production reconstruction and FP4-only BF16 prefill."""
    from sglang.kernels.ops.attention.dsv4.fp4_cache import read_fp4_cache
    from sgl_kernel.flash_mla import flash_mla_fp4_sparse_fwd

    ids = torch.arange(t.values.shape[0], device="cuda", dtype=torch.int32)
    reconstructed = torch.empty(
        (ids.numel(), 1, 512), device="cuda", dtype=torch.bfloat16
    )
    read_fp4_cache(t.cache, ids, t.page, packed=t.packed, out=reconstructed)
    valid = (
        (t.extra >= 0)
        & (t.extra < ids.numel())
        & (torch.arange(512, device="cuda")[None, None, :] < t.lengths[:, None, None])
    )
    combined = torch.cat(
        (t.indices, torch.where(valid, t.extra + t.swa_values.shape[0], -1)), dim=-1
    )
    # SM90 sparse prefill consumes two 64-entry tiles per iteration
    # (phase1.cuh: 2 * B_TOPK). Invalid padding preserves the selected set.
    combined = torch.nn.functional.pad(
        combined, (0, -combined.shape[-1] % 128), value=-1
    )
    return flash_mla_fp4_sparse_fwd(
        t.q[:, 0],
        torch.cat((t.swa_values[:, None, :], reconstructed)),
        combined,
        512**-0.5,
        attn_sink=t.sink,
        topk_length=64 + t.lengths if use_lengths else None,
    )[0]


@GPU
@pytest.mark.parametrize("packed", [False, True])
@pytest.mark.parametrize("heads", [64, 128])
def test_fp4_graph_and_stream(packed, heads):
    from sgl_kernel.flash_mla import flash_mla_with_fp4_kvcache
    from sglang.kernels.ops.attention.dsv4.fp4_cache import (
        read_fp4_cache,
        store_fp4_cache,
    )

    t = _attention_case(packed, heads, 128)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    out = torch.empty((t.loc.numel(), 1, 512), device="cuda", dtype=torch.bfloat16)
    with torch.cuda.stream(stream):

        def run():
            store_fp4_cache(t.x, t.cache, t.loc, t.page, packed=packed)
            read_fp4_cache(t.cache, t.loc, t.page, packed=packed, out=out)
            direct = flash_mla_with_fp4_kvcache(
                t.q,
                t.swa,
                t.indices,
                t.cache_view,
                t.extra,
                packed=packed,
                extra_topk_length=t.lengths,
                attn_sink=t.sink,
            )
            return direct, _fp4_adapter(t, use_lengths=True)

        for _ in range(2):
            run()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            captured = run()
        for step in range(2):
            t.x.copy_(t.x.roll(17, 1) * -0.7)
            t.extra[-1, 0, :4] = torch.tensor(
                [128 + step, -1, 129, 128], device="cuda", dtype=torch.int32
            )
            t.lengths[-1] = 65 + step
            graph.replay()
            expected_pages, expected_values = _oracle(
                torch.cat((torch.zeros_like(t.x.cpu())[None], t.x.cpu()[None])), packed
            )
            t.values = expected_values.reshape(-1, 512).cuda()
            for a, e in zip(captured[0], _attention_reference(t)):
                torch.testing.assert_close(a, e, atol=0.02, rtol=0.02)
            torch.testing.assert_close(
                captured[1], _attention_reference(t)[0][:, 0], atol=0.02, rtol=0.02
            )
            torch.testing.assert_close(out[:, 0], t.values[128:], rtol=0, atol=0)
    torch.cuda.current_stream().wait_stream(stream)


@GPU
def test_fp4_pool_allocations(tmp_path):
    from sglang.srt.mem_cache.deepseek_v4_memory_pool import (
        DeepSeekV4SingleKVPool,
        DeepSeekV4TokenToKVPool,
    )
    from sglang.srt.runtime_context import get_context
    from sglang.srt.model_executor.pool_configurator import DSV4PoolConfigurator
    from sglang.srt.speculative.spec_info import SpeculativeAlgorithm

    # Include the complete Direct call's measured temporary/output peak at the
    # declared B8/H128 fixture, equally in both capacity budgets. Warm-up is
    # outside measurement; this records memory, never timing or throughput.
    from sgl_kernel.flash_mla import flash_mla_with_fp4_kvcache

    direct_peaks = []
    for packed in (False, True):
        t = _attention_case(packed, 128, 128)
        for _ in range(2):
            warm = flash_mla_with_fp4_kvcache(
                t.q,
                t.swa,
                t.indices,
                t.cache_view,
                t.extra,
                packed=packed,
                extra_topk_length=t.lengths,
                attn_sink=t.sink,
            )
            del warm
        torch.cuda.synchronize()
        before = torch.cuda.memory_allocated()
        torch.cuda.reset_peak_memory_stats()
        result = flash_mla_with_fp4_kvcache(
            t.q,
            t.swa,
            t.indices,
            t.cache_view,
            t.extra,
            packed=packed,
            extra_topk_length=t.lengths,
            attn_sink=t.sink,
        )
        torch.cuda.synchronize()
        direct_peaks.append(torch.cuda.max_memory_allocated() - before)
        del result, t
        gc.collect()
    assert direct_peaks[0] == direct_peaks[1]
    direct_workspace_bytes = direct_peaks[0]
    report = {
        "direct_workspace_peak_bytes": direct_workspace_bytes,
        "direct_workspace_case": "B8/S1/H128/K512 plus K64 SWA, page128",
        "status": "running",
        "modes": [],
        "edges": [],
        "scope": "model-free actual pool storage; no serving capacity claim",
    }
    with get_context().override_server_args(
        device="cuda",
        dsv4_attn_backend="flashmla",
        page_size=256,
        max_running_requests=8,
    ):
        for layout in (KVLayout.V41_FP4, KVLayout.V41_FP4_REMNANT):
            for p in (128, 256):
                for rows in (0, 1, 127, 128, 129, 255, 256, 257):
                    pool = DeepSeekV4SingleKVPool(
                        rows,
                        p,
                        torch.float8_e4m3fn,
                        448,
                        64,
                        1,
                        "cuda",
                        False,
                        global_page_size=256,
                        kv_layout=layout,
                    )
                    expected = math.ceil((rows + 256) / p) * p * layout.bytes_per_token
                    assert sum(_gpu_storages(pool).values()) == expected
                    report["edges"].append(
                        dict(
                            layout=layout.value,
                            page=p,
                            rows=rows,
                            storage_bytes=expected,
                        )
                    )
                    del pool
    for mode in ("fp4", "remnant"):
        with get_context().override_server_args(
            device="cuda",
            dsv4_attn_backend="flashmla",
            page_size=256,
            max_running_requests=8,
            chunked_prefill_size=512,
            dsv41_main_kv_cache_format=mode,
        ):
            metadata = SimpleNamespace(
                qk_nope_head_dim=448,
                qk_rope_head_dim=64,
                index_head_dim=128,
                context_len=131072,
                compress_ratios=RATIOS,
                window_size=128,
                hf_config=SimpleNamespace(kv_source_layer_ids=SOURCES),
            )
            sizing = DSV4PoolConfigurator(
                SimpleNamespace(
                    kv_cache_dtype_str="fp8_e4m3",
                    model_config=metadata,
                    layer_info=SimpleNamespace(start_layer=0, end_layer=40),
                    pp_size=1,
                    attn_dp_size=1,
                    page_size=256,
                    sliding_window_size=128,
                    spec_algorithm=SpeculativeAlgorithm.NONE,
                )
            )
            record_bytes = 288 if mode == "fp4" else 224
            assert sizing._compressed_bytes_per_full_token(1) == record_bytes + 68
            assert sizing._compressed_bytes_per_full_token(2) == (record_bytes + 68) / 2
            workspace_bytes = 1024 * 512 * 2 + direct_workspace_bytes
            byte_budget = 1 << 30
            sizes = sizing.calculate_pool_sizes(byte_budget - workspace_bytes, 256)
            full = sizes.full_max_total_num_tokens
            assert sizing.swa_cap_tokens is not None
            pool = DeepSeekV4TokenToKVPool(
                max_num_reqs=8,
                swa_size=sizes.swa_max_total_num_tokens,
                c4_size=0,
                c128_size=0,
                c4_state_pool_size=0,
                c128_state_pool_size=0,
                page_size=256,
                swa_page_size=256,
                dtype=torch.float8_e4m3fn,
                c4_state_dtype=torch.float32,
                c128_state_dtype=torch.float32,
                qk_nope_head_dim=448,
                qk_rope_head_dim=64,
                indexer_head_dim=128,
                layer_num=40,
                device="cuda",
                enable_memory_saver=False,
                compression_ratios=RATIOS,
                kv_source_layers=SOURCES,
                full_size=full,
                kv_layout=KVLayout.V4,
                compressed_kv_layout="sm90_fp4" if mode == "fp4" else "sm90_remnant",
            )
            assert pool.kv_pools[2].layer_num == 3 and pool.kv_pools[1].layer_num == 1
            assert (
                pool.get_extra_key_page_size(2) == 128
                and pool.get_extra_key_page_size(20) == 256
            )
            assert (
                pool.get_extra_key_buffer(2).data_ptr()
                == pool.get_extra_key_buffer(3).data_ptr()
            )
            ownership = {}
            groups = {}
            for name, obj in dict(
                swa=pool.swa_kv_pool,
                global_kv=pool.kv_pools,
                indexer=pool.index_pools,
                compressor_state=(
                    pool.compress_state_pools,
                    pool.indexer_compress_state_pools,
                ),
            ).items():
                storages = _gpu_storages(obj)
                assert ownership.keys().isdisjoint(storages), name
                ownership.update(storages)
                groups[name] = sum(storages.values())
            assert ownership == _gpu_storages(pool), "unaccounted pool-owned storage"
            # Real C2 request rings, across split calls and an in-batch chain.
            from sglang.srt.layers.attention.deepseek_v4_backend import (
                DeepseekV4AttnBackend,
            )

            backend = SimpleNamespace(token_to_kv_pool=pool)
            partner = DeepseekV4AttnBackend._low_ratio_pair_partners
            req = torch.tensor([1, 2], device="cuda", dtype=torch.int64)
            even = (
                torch.arange(1024, device="cuda", dtype=torch.float32).reshape(2, 512)
                / 1024
            )
            score = even * 0.13
            pad = torch.zeros(2, device="cuda", dtype=torch.bool)
            carried = partner(
                backend,
                layer_id=2,
                kv=even,
                score=score,
                req=req,
                pos=torch.zeros_like(req),
                pad=pad,
            )
            assert torch.count_nonzero(carried[0]) == 0
            assert torch.isneginf(carried[1]).all()
            odd = even + 2
            carried = partner(
                backend,
                layer_id=2,
                kv=odd,
                score=score + 1,
                req=req,
                pos=torch.ones_like(req),
                pad=pad,
            )
            torch.testing.assert_close(carried[0], even, rtol=0, atol=0)
            torch.testing.assert_close(carried[1], score, rtol=0, atol=0)
            chain = (
                torch.arange(2048, device="cuda", dtype=torch.float32).reshape(4, 512)
                / 256
            )
            carried = partner(
                backend,
                layer_id=2,
                kv=chain,
                score=chain * 0.17,
                req=torch.ones(4, device="cuda", dtype=torch.int64),
                pos=torch.arange(2, 6, device="cuda", dtype=torch.int64),
                pad=torch.zeros(4, device="cuda", dtype=torch.bool),
            )
            torch.testing.assert_close(
                carried[0], torch.cat((odd[:1], chain[:-1])), rtol=0, atol=0
            )
            torch.testing.assert_close(
                carried[1],
                torch.cat((score[:1] + 1, chain[:-1] * 0.17)),
                rtol=0,
                atol=0,
            )
            expected_global = (
                math.ceil((full / 2 + 256) / 128) * 128 * 3
                + math.ceil((full + 256) / 256) * 256
            ) * record_bytes
            assert groups["global_kv"] == expected_global
            from sglang.srt.mem_cache.deepseek_v4_memory_pool import (
                DSV41_INDEX_PAGE_SIZE,
            )

            ip = DSV41_INDEX_PAGE_SIZE
            expected_index = (
                math.ceil(((full + 256) // 2 + ip) / ip) * ip * 3
                + math.ceil((full + 256 + ip) / ip) * ip
            ) * 68
            assert groups["indexer"] == expected_index
            # Explicit prefill workspace is identical in both configurations.
            workspace = torch.empty((1024, 1, 512), device="cuda", dtype=torch.bfloat16)
            direct_workspace = torch.empty(
                (direct_workspace_bytes,), device="cuda", dtype=torch.uint8
            )
            assert groups["compressor_state"] == sizing._get_c2_state_fixed_bytes(8)
            assert groups["swa"] == (sizes.swa_max_total_num_tokens + 256) * 585 * 40
            total_bytes = (
                sum(groups.values()) + workspace.nbytes + direct_workspace.nbytes
            )
            assert total_bytes <= byte_budget
            assert total_bytes + 256 * sizing.bytes_per_full_token > byte_budget
            report["modes"].append(
                dict(
                    mode="Native" if mode == "fp4" else "Packed",
                    groups=groups,
                    prefill_workspace_bytes=workspace.nbytes,
                    direct_workspace_bytes=direct_workspace.nbytes,
                    total_bytes=total_bytes,
                    byte_budget=byte_budget,
                    full_tokens=full,
                    swa_tokens=sizes.swa_max_total_num_tokens,
                    reserved_page_bytes=sizing._get_v41_reserved_page_bytes(),
                )
            )
            del pool, workspace, direct_workspace
            gc.collect()
    a, b = report["modes"]
    for group in ("swa", "compressor_state"):
        assert a["groups"][group] == b["groups"][group]
    report["capacity_gain_percent"] = (b["full_tokens"] / a["full_tokens"] - 1) * 100
    report["status"] = "passed"
    Path(
        os.environ.get("REMNANT_ALLOCATION_REPORT", str(tmp_path / "allocation.json"))
    ).write_text(json.dumps(report, indent=2) + "\n")


@GPU
@pytest.mark.parametrize("ratio", [1, 2])
@pytest.mark.parametrize("packed", [False, True])
def test_fp4_compressor_boundary(ratio, packed):
    from sglang.srt.layers.attention.dsv4.dsv41_sparse import DeepseekV41Compressor
    from sglang.srt.layers.attention.deepseek_v4_backend import DeepseekV4AttnBackend
    from sglang.kernels.ops.attention.dsv4.fp4_cache import store_fp4_cache

    torch.manual_seed(71)
    compressor = DeepseekV41Compressor(
        512, 512, ratio, 1e-6, fused_compress=False
    ).cuda()
    with torch.no_grad():
        compressor.norm.weight.copy_(torch.linspace(0.5, 1.5, 512, device="cuda"))
    x = torch.randn((8, 512), device="cuda", dtype=torch.bfloat16)
    kv, score = compressor.project(x)
    pooled = (
        kv
        if ratio == 1
        else compressor.pool_pairs(kv.reshape(4, 2, 512), score.reshape(4, 2, 512))
    )
    n = pooled.shape[0]
    captured = {}

    def index_keys(latent, freq):
        captured["index_latent"] = latent.clone()
        return torch.zeros((n, 128), device="cuda", dtype=torch.bfloat16)

    layout = KVLayout.V41_FP4_REMNANT if packed else KVLayout.V41_FP4
    cache = torch.zeros(
        (2, 128 * layout.bytes_per_token), device="cuda", dtype=torch.uint8
    )

    def store(**kwargs):
        store_fp4_cache(
            kwargs["cache_k"],
            cache,
            kwargs["loc"],
            128,
            packed=packed,
            freqs_cis=kwargs["freqs_cis"],
        )

    def index_store(**kwargs):
        captured["index_written"] = kwargs["cache_k"].clone()

    pool = SimpleNamespace(
        get_extra_key_layout=lambda _: layout,
        set_extra_key_buffer_fused=store,
        set_index_k_fp4=index_store,
    )
    angle = (
        torch.arange(n * 32, device="cuda", dtype=torch.float32).reshape(n, 32) * 0.021
    )
    freq = torch.polar(torch.ones_like(angle), angle)
    layer = SimpleNamespace(
        layer_id=20 if ratio == 1 else 2,
        compressor=compressor,
        freqs_cis=freq,
        rope_head_dim=64,
        indexer=SimpleNamespace(owns_k=True, index_keys=index_keys),
    )
    backend = SimpleNamespace(token_to_kv_pool=pool)
    slots = torch.arange(128, 128 + n, device="cuda", dtype=torch.int64)
    DeepseekV4AttnBackend._low_ratio_write_group(
        backend, layer, pooled, slots, torch.arange(n, device="cuda", dtype=torch.int64)
    )
    latent = captured["index_latent"]
    assert latent.dtype == torch.bfloat16 and captured["index_written"].shape == (
        n,
        128,
    )
    # Independent normalization check also catches pruning before normalization.
    v = pooled.bfloat16().float()
    expected_norm = (
        v
        * torch.rsqrt(v.square().mean(-1, keepdim=True) + 1e-6)
        * compressor.norm.weight.float()
    ).bfloat16()
    torch.testing.assert_close(latent, expected_norm, rtol=2**-6, atol=2**-20)
    from sglang.srt.layers.attention.dsv4.dsv41_sparse import rope_tail

    rotated = rope_tail(latent, freq, 64).cpu()
    source = torch.zeros((2, 128, 512), dtype=torch.bfloat16)
    source[1, :n] = rotated
    expected, _ = _oracle(source, packed)
    expected[0].zero_()
    # Untouched rows stay byte-zero rather than receiving quantized zero scales.
    expected[1, n * layout.data_bytes : 128 * layout.data_bytes].zero_()
    expected[1, 128 * layout.data_bytes + n * 32 :].zero_()
    torch.testing.assert_close(cache.cpu(), expected, rtol=0, atol=0)


def test_fp4_backend_dispatch():
    from unittest import mock
    import sys
    from sglang.srt.layers.attention import deepseek_v4_backend as module
    from sglang.srt.layers.attention.deepseek_v4_backend import DSV4AttnMetadata
    from sglang.srt.remnant import capture

    for layout in (KVLayout.V41_FP4, KVLayout.V41_FP4_REMNANT):
        for ratio in (1, 2):
            core = mock.Mock(spec=DSV4AttnMetadata)
            core.swa_page_indices = torch.zeros((1, 64), dtype=torch.int32)
            core.swa_topk_lengths = torch.tensor([64], dtype=torch.int32)
            core.sparse_page_indices.return_value = torch.ones(
                (1, 64), dtype=torch.int32
            )
            core.sparse_topk_lengths.return_value = torch.tensor([1], dtype=torch.int32)
            pool = mock.Mock(spec=module.DeepSeekV4TokenToKVPool)
            pool.request_window = None
            pool.get_swa_key_buffer_radix.return_value = torch.zeros(
                (1, 128 * 584), dtype=torch.uint8
            )
            pool.get_swa_key_page_size.return_value = 128
            pool.get_swa_key_bytes_per_token.return_value = 584
            pool.get_extra_key_buffer.return_value = torch.zeros(
                (1, 128 * layout.bytes_per_token), dtype=torch.uint8
            )
            pool.get_extra_key_page_size.return_value = 128
            pool.get_extra_key_bytes_per_token.return_value = layout.bytes_per_token
            pool.get_extra_key_layout.return_value = layout
            backend = SimpleNamespace(
                mtp_enabled=False,
                trtllm_attn=False,
                is_dsv41=True,
                _use_sparse_prefill=lambda *a, **k: False,
                remnant_workspace=None,
                forward_metadata=SimpleNamespace(
                    core_attn_metadata=core, late_layer_tail=None
                ),
                token_to_kv_pool=pool,
                head_dim_v=512,
                softmax_scale=512**-0.5,
            )
            q = torch.ones((1, 64, 512), dtype=torch.bfloat16)
            direct = mock.Mock(return_value=(q[:, None], torch.zeros((1, 64, 1))))
            with (
                mock.patch.object(
                    module,
                    "get_platform",
                    return_value=SimpleNamespace(
                        is_sm90=True, is_sm100=False, is_sm120=False
                    ),
                ),
                mock.patch.object(capture, "enabled", return_value=False),
                mock.patch.dict(
                    sys.modules,
                    {
                        "sgl_kernel.flash_mla": SimpleNamespace(
                            flash_mla_with_fp4_kvcache=direct
                        )
                    },
                ),
            ):
                actual = module.DeepseekV4AttnBackend._forward_attention(
                    backend,
                    q,
                    q,
                    q,
                    SimpleNamespace(layer_id=2),
                    SimpleNamespace(),
                    compress_ratio=ratio,
                    save_kv_cache=False,
                    attn_sink=torch.zeros(64),
                )
            assert torch.equal(actual, q)
            direct.assert_called_once()
            assert direct.call_args.kwargs["packed"] == (
                layout is KVLayout.V41_FP4_REMNANT
            )
            assert direct.call_args.args[3].shape[-1] == layout.bytes_per_token

    for mode in ("auto", "fp4", "remnant"):
        for ratio in (0, 1, 2):
            q = torch.ones((1, 1, 64, 512), dtype=torch.bfloat16)
            workspace = torch.zeros((2, 1, 512), dtype=torch.bfloat16)
            indices = torch.zeros((1, 128), dtype=torch.int32)
            lengths = torch.ones(1, dtype=torch.int32)
            cache = SimpleNamespace(
                swa_token_ids=torch.zeros(1, dtype=torch.int32),
                swa_page_size=128,
                c0_combined_indices=indices,
                c0_combined_lens=lengths,
                layer_inputs=lambda *a: (
                    torch.zeros(1, dtype=torch.int32),
                    indices,
                    lengths,
                ),
            )
            backend = SimpleNamespace(
                forward_metadata=SimpleNamespace(sparse_prefill_cache=cache),
                sparse_prefill_workspace=SimpleNamespace(get=lambda n: workspace[:n]),
                softmax_scale=512**-0.5,
                head_dim_v=512,
            )
            pool = mock.Mock(spec=module.DeepSeekV4TokenToKVPool)
            pool.get_extra_key_page_size.return_value = 128
            stock = mock.Mock(return_value=(q[:, 0], None, None))
            fp4 = mock.Mock(return_value=(q[:, 0], None, None))
            with (
                mock.patch.object(
                    module, "get_platform", return_value=SimpleNamespace(is_sm90=True)
                ),
                mock.patch.object(
                    module,
                    "get_model",
                    return_value=SimpleNamespace(dsv41_main_kv_cache_format=mode),
                ),
                mock.patch.object(module, "dequantize_k_cache_paged"),
                mock.patch.dict(
                    sys.modules,
                    {
                        "sgl_kernel.flash_mla": SimpleNamespace(
                            flash_mla_sparse_fwd=stock, flash_mla_fp4_sparse_fwd=fp4
                        )
                    },
                ),
            ):
                module.DeepseekV4AttnBackend._forward_prefill_sparse(
                    backend,
                    q,
                    2,
                    ratio,
                    SimpleNamespace(),
                    pool,
                    SimpleNamespace(),
                    torch.zeros(64),
                )
            (stock if mode == "auto" else fp4).assert_called_once()
            (fp4 if mode == "auto" else stock).assert_not_called()
