"""Model-free GLM production codec, TileLang, graph and allocation qualification.

Author: Winston Cai.

Run via scripts/modal/kernels.py --model-family glm53. No weights. The Native
reference reads independently expanded codes with stock FP8 TileLang arithmetic.
"""

import gc
import hashlib
import json
import os
from pathlib import Path
from unittest.mock import patch

import pytest
import torch

from sglang.srt.layers.layernorm import RMSNorm
from sglang.srt.remnant.glm import (
    PackedCache,
    attention,
    normalize_and_store,
    reconstruct,
)
from sglang.srt.remnant.reference import topmag_keep_mask
from sglang.srt.runtime_context import get_context
from sglang.kernels.ops.attention.dsa.tilelang_kernel import tilelang_sparse_fwd

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="H100 required")


def fixture(rows=513, implementation="direct"):
    raw = torch.randn(rows, 512, device="cuda", dtype=torch.bfloat16)
    raw[0] = 0
    raw[1] = 1  # exact cutoff ties; preserve the original producer mask
    raw[2, ::2] = 0
    raw[3] = torch.arange(512, device="cuda").to(torch.bfloat16)
    norm = RMSNorm(512, eps=1e-5).cuda().to(torch.bfloat16)
    norm.weight.data.copy_(torch.linspace(0.5, 1.5, 512, device="cuda"))
    values = torch.zeros(rows + 256, 256, dtype=torch.uint8, device="cuda")
    bitmap = torch.zeros(rows + 256, 8, dtype=torch.uint64, device="cuda")
    workspace = (
        torch.empty(128 * 2112, 512, dtype=torch.uint8, device="cuda")
        if implementation == "adapter"
        else None
    )
    cache = PackedCache(values, bitmap, workspace, implementation)
    return raw, norm, cache


def expected(raw, norm):
    # Selection is the established shared primitive; codec expansion below is
    # independent (boolean scatter, no rank/popcount implementation reuse).
    mask = topmag_keep_mask(raw, 0.5)
    codes = norm(raw.masked_fill(~mask, 0)).to(torch.float8_e4m3fn).view(torch.uint8)
    dense = torch.where(mask, codes, 0)
    words = torch.zeros(raw.shape[0], 8, dtype=torch.int64, device="cpu")
    for row, indices in enumerate(mask.cpu().nonzero().split(256)):
        for coord in indices[:, 1].tolist():
            word, lane = divmod(coord, 64)
            value = 1 << (63 - lane)
            if value >= 1 << 63:
                value -= 1 << 64
            words[row, word] |= value
    return mask, codes[mask].reshape(-1, 256), words, dense


def test_producer_reconstruction():
    with get_context().override_server_args(device="cuda"):
        raw, norm, cache = fixture()
        mask, values, words, dense = expected(raw, norm)
        loc = torch.arange(raw.shape[0], device="cuda")
        normalize_and_store(raw, norm, loc, cache.values, cache.bitmaps)
        assert torch.equal(cache.values[: len(raw)], values)
        assert torch.equal(cache.bitmaps[: len(raw)].cpu().view(torch.int64), words)
        selected = torch.tensor(
            [-1, 0, 1, 63, 64, 255, 256, 512, 512, 9000],
            device="cuda",
            dtype=torch.int32,
        )
        out = torch.empty(len(selected), 512, device="cuda", dtype=torch.uint8)
        reconstruct(cache, selected, out)
        ref = torch.zeros_like(out)
        ref[1:-1] = dense[selected[1:-1].long()]
        assert torch.equal(out, ref)
        # Invalid writes must leave BOTH planes unchanged.
        before_v, before_b = cache.values.clone(), cache.bitmaps.clone()
        normalize_and_store(
            raw[:2],
            norm,
            torch.tensor([-1, 9000], device="cuda"),
            cache.values,
            cache.bitmaps,
        )
        assert torch.equal(cache.values, before_v)
        assert torch.equal(cache.bitmaps.view(torch.int64), before_b.view(torch.int64))


def test_invalid_attention():
    with get_context().override_server_args(device="cuda"):
        raw, norm, cache = fixture(implementation="adapter")
        normalize_and_store(
            raw,
            norm,
            torch.arange(len(raw), device="cuda"),
            cache.values,
            cache.bitmaps,
        )
        indices = selection(8, 3)
        indices[0] = -1
        indices[1, 0, 2048] = 9000
        q = torch.randn(8, 8, 512, dtype=torch.bfloat16, device="cuda")
        mask = topmag_keep_mask(raw, 0.5)
        dense = (
            norm(raw.masked_fill(~mask, 0)).to(torch.float8_e4m3fn).view(torch.uint8)
        )
        dense.masked_fill_(~mask, 0)
        valid = torch.where(indices < len(raw), indices, -1)
        ref = tilelang_sparse_fwd(
            q, dense.view(torch.float8_e4m3fn).unsqueeze(1), valid, 512**-0.5
        )
        for impl in ("adapter", "direct"):
            actual = attention(
                q,
                PackedCache(cache.values, cache.bitmaps, cache.workspace, impl),
                indices,
                512**-0.5,
            )
            torch.testing.assert_close(actual, ref, atol=0.02, rtol=0.02)


def selection(batch, tails):
    indices = torch.full((batch, 1, 2112), -1, dtype=torch.int32, device="cuda")
    indices[:, :, :2048] = torch.arange(
        2048, device="cuda", dtype=torch.int32
    ).remainder(513)
    indices[0, :, :1024] = -1  # partial selection / cached prefix
    if tails:
        indices[:, :, 2048 : 2048 + tails] = torch.arange(
            510, 510 + tails, device="cuda", dtype=torch.int32
        )
    return indices


@pytest.mark.parametrize("heads", [8, 16, 32, 64])
@pytest.mark.parametrize("batch", [1, 8, 24, 32, 128])
@pytest.mark.parametrize("tails", [0, 1, 2, 3])
def test_attention(heads, batch, tails):
    with get_context().override_server_args(device="cuda"):
        raw, norm, cache = fixture(implementation="adapter")
        normalize_and_store(
            raw,
            norm,
            torch.arange(len(raw), device="cuda"),
            cache.values,
            cache.bitmaps,
        )
        mask = topmag_keep_mask(raw, 0.5)
        dense = (
            norm(raw.masked_fill(~mask, 0)).to(torch.float8_e4m3fn).view(torch.uint8)
        )
        dense.masked_fill_(~mask, 0)
        dense = dense.view(torch.float8_e4m3fn).unsqueeze(1)
        indices = selection(batch, tails)
        q = torch.randn(batch, heads, 512, dtype=torch.bfloat16, device="cuda")
        reference = tilelang_sparse_fwd(q, dense, indices, 512**-0.5)
        for implementation in ("adapter", "direct"):
            candidate = PackedCache(
                cache.values, cache.bitmaps, cache.workspace, implementation
            )
            out = attention(q, candidate, indices, 512**-0.5)
            torch.testing.assert_close(out, reference, atol=0.02, rtol=0.02)


@pytest.mark.parametrize("implementation", ["adapter", "direct"])
def test_changing_input_graph(implementation):
    with get_context().override_server_args(device="cuda"):
        raw, norm, cache = fixture(implementation=implementation)
        loc = torch.arange(len(raw), device="cuda")
        indices = selection(8, 3)
        q = torch.randn(8, 8, 512, dtype=torch.bfloat16, device="cuda")

        def run():
            normalize_and_store(raw, norm, loc, cache.values, cache.bitmaps)
            return attention(q, cache, indices, 512**-0.5)

        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):
                run()
        torch.cuda.current_stream().wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            actual = run()
        for step in range(3):
            raw.normal_()
            q.normal_()
            indices.copy_(selection(8, step))
            indices[1] = -1  # empty selection must remain safe and deterministic
            graph.replay()
            reference = run()
            torch.testing.assert_close(actual, reference, atol=0, rtol=0)


def make_pool(size, implementation="native", layers=11, mamba=None):
    from sglang.srt.mem_cache.memory_pool import HybridLinearKVPool, DSATokenToKVPool
    from sglang.srt.remnant.glm_pool import GLMPackedTokenToKVPool

    return HybridLinearKVPool(
        size=size,
        dtype=torch.float8_e4m3fn,
        page_size=256,
        head_num=1,
        head_dim=512,
        full_attention_layer_ids=list(range(3, 44, 4))[:layers],
        device="cuda",
        mamba_pool=mamba,
        use_mla=True,
        use_dsa=True,
        kv_lora_rank=512,
        qk_rope_head_dim=0,
        index_head_dim=128,
        kv_cache_dim=512,
        index_kpool=4,
        index_kpool_compress=True,
        max_running_requests=32,
        full_kv_pool_class=(
            DSATokenToKVPool if implementation == "native" else GLMPackedTokenToKVPool
        ),
    )


def test_pool_lifecycle():
    with get_context().override_server_args(
        device="cuda", glm5_mla_cache_implementation="direct"
    ):
        pool = make_pool(768, "direct", layers=1)
        raw, norm, _ = fixture()
        pool.normalize_and_store_glm(
            3, raw, norm, torch.arange(len(raw), device="cuda")
        )
        cache = pool.get_glm_packed_buffer(3)
        source = torch.arange(256, 512, device="cuda")
        target = torch.arange(512, 768, device="cuda")
        pool.full_kv_pool.index_key_cache.buffer[0][1].fill_(123)
        pool.move_kv_cache(target, source)
        assert bool((pool.full_kv_pool.index_key_cache.buffer[0][2] == 123).all())
        assert torch.equal(cache.values[source], cache.values[target])
        assert torch.equal(
            cache.bitmaps.view(torch.int64)[source],
            cache.bitmaps.view(torch.int64)[target],
        )
        req_source = torch.tensor([2], device="cuda")
        req_target = torch.tensor([3], device="cuda")
        pool.full_kv_pool._compress_tail_k[0][2].fill_(1.5)
        pool.full_kv_pool._compress_tail_score[0][2].fill_(-0.5)
        saved = pool.get_cpu_copy(source, req_pool_index=req_source)
        pool.load_cpu_copy(saved, target, req_pool_index=req_target)
        assert torch.equal(cache.values[source], cache.values[target])
        assert torch.equal(
            cache.bitmaps.view(torch.int64)[source],
            cache.bitmaps.view(torch.int64)[target],
        )
        assert torch.equal(
            pool.full_kv_pool._compress_tail_k[0][2],
            pool.full_kv_pool._compress_tail_k[0][3],
        )
        assert torch.equal(
            pool.full_kv_pool._compress_tail_score[0][2],
            pool.full_kv_pool._compress_tail_score[0][3],
        )
        # Reuse after restoring a different request cannot expose stale bitmap bits.
        pool.normalize_and_store_glm(3, raw[:256].flip(0), norm, target)
        pool.load_cpu_copy(saved, target, req_pool_index=req_target)
        assert torch.equal(cache.values[source], cache.values[target])
        with pytest.raises(RuntimeError, match="no dense mirror"):
            pool.get_key_buffer(3)
        pool.full_kv_pool._clear_buffers()


def test_production_backend_constructor():
    from types import SimpleNamespace
    from sglang.srt.configs.glm5_next import Glm5NextConfig
    from sglang.srt.layers.attention.dsa_backend import DeepseekSparseAttnBackend

    with get_context().override_server_args(
        device="cuda",
        tp_size=8,
        dsa_prefill_backend="tilelang",
        dsa_decode_backend="tilelang",
        dsa_topk_backend="torch",
        glm5_mla_cache_implementation="direct",
    ):
        pool = make_pool(768, "direct")
        hf = Glm5NextConfig(
            **json.loads(Path(os.environ["REMNANT_GLM_MODEL_CONFIG"]).read_text())
        )
        model = SimpleNamespace(
            hf_config=hf,
            context_len=4096,
            num_attention_heads=64,
            qk_nope_head_dim=256,
            kv_lora_rank=512,
            qk_rope_head_dim=0,
        )
        req = SimpleNamespace(
            size=32,
            req_to_token=torch.zeros(33, 4096, dtype=torch.int32, device="cuda"),
        )
        runner = SimpleNamespace(
            device="cuda",
            page_size=256,
            model_config=model,
            token_to_kv_pool=pool,
            req_to_token_pool=req,
            hisparse_coordinator=None,
            is_draft_worker=False,
            kv_cache_dtype=torch.float8_e4m3fn,
            max_running_requests=32,
        )
        backend = DeepseekSparseAttnBackend(runner)
        assert backend.num_q_heads == 8
        assert not backend.supports_mha_one_shot
        raw, norm, _ = fixture()
        pool.normalize_and_store_glm(
            3, raw, norm, torch.arange(len(raw), device="cuda")
        )
        q = torch.randn(8, 8, 512, dtype=torch.bfloat16, device="cuda")
        cache = pool.get_glm_packed_buffer(3)
        out = backend._forward_tilelang(
            q, cache, 512, selection(8, 3)[:, 0, :2051], 512**-0.5
        )
        ref = attention(q, cache, selection(8, 3), 512**-0.5)
        torch.testing.assert_close(out, ref, atol=0, rtol=0)


def test_real_gpu_allocations(tmp_path):
    from sglang.srt.configs.mamba_utils import (
        KimiLinearCacheParams,
        KimiLinearStateShape,
        Mamba2StateDType,
    )
    from sglang.srt.mem_cache.memory_pool import MambaPool
    from dsv4.test_remnant_pool import _gpu_storages

    source = Path(os.environ["REMNANT_GLM_MODEL_CONFIG"])
    assert (
        hashlib.sha256(source.read_bytes()).hexdigest()
        == "bb8f01c42cb92a52ca72e65afb4d5bd8d11aef083cd210e8de25dfb904f23e9f"
    )
    text = json.loads(source.read_text())["text_config"]
    assert text["num_hidden_layers"] == 45 and text["index_kpool"] == 4
    report = {
        "status": "running",
        "modes": [],
        "edges": [],
        "tokens": 131072,
        "requests": 32,
        "tp": 8,
    }
    output = Path(
        os.environ.get("REMNANT_ALLOCATION_REPORT", tmp_path / "allocation.json")
    )
    with get_context().override_server_args(device="cuda"):
        for mode in ("native", "adapter", "direct"):
            with get_context().override_server_args(
                glm5_mla_cache_implementation=mode if mode != "native" else "direct"
            ):
                row_bytes = 512 if mode == "native" else 320
                for size in (0, 1, 255, 256, 257, 131072):
                    mamba = None
                    if size == 131072:
                        linear = [i for i in range(45) if i % 4 != 3]
                        params = KimiLinearCacheParams(
                            shape=KimiLinearStateShape.create(
                                tp_world_size=8, num_heads=64, head_dim=128
                            ),
                            layers=linear,
                            dtype=Mamba2StateDType(
                                conv=torch.bfloat16, temporal=torch.float32
                            ),
                        )
                        mamba = MambaPool(
                            size=160,
                            spec_state_size=0,
                            cache_params=params,
                            mamba_layer_ids=linear,
                            device="cuda",
                        )
                    pool = make_pool(size, mode, mamba=mamba)
                    owned = _gpu_storages(vars(pool.full_kv_pool))
                    if mamba is not None:
                        owned.update(_gpu_storages(mamba))
                    measured = sum(owned.values())
                    workspace = 128 * 2112 * 512 if mode == "adapter" else 0
                    latent = 11 * (size + 256) * row_bytes
                    indexer = 11 * ((size + 257) // 256) * 64 * 132
                    tails = 11 * 2 * 33 * 4 * 128 * 2
                    pointers = 11 * 8
                    kda = (
                        161 * 18452480 + 32 if mamba is not None else 0
                    )  # four one-conv descriptor arrays
                    expected_bytes = (
                        latent + indexer + tails + pointers + workspace + kda
                    )
                    assert measured == expected_bytes, (
                        mode,
                        size,
                        measured,
                        expected_bytes,
                    )
                    entry = dict(
                        mode=mode,
                        tokens=size,
                        backing_bytes=measured,
                        latent=latent,
                        indexer=indexer,
                        tails=tails,
                        kda=kda,
                        workspace=workspace,
                        pointers=pointers,
                    )
                    report["modes" if size == 131072 else "edges"].append(entry)
                    output.write_text(json.dumps(report, indent=2) + "\n")
                    del pool, mamba
                    gc.collect()
                    torch.cuda.empty_cache()
    report["status"] = "passed"
    output.write_text(json.dumps(report, indent=2) + "\n")


def precompile_sm90():
    """Build exact CPU artifacts without allocating a CUDA tensor or context."""
    from sglang.srt.remnant.glm import compile_kernels
    from sglang.kernels.ops.attention.dsa.tilelang_kernel import (
        sparse_mla_fwd_decode_partial_fp8,
        sparse_mla_fwd_decode_combine,
    )

    store, unpack = compile_kernels()
    artifacts = {"triton": [store.hash, unpack.hash], "tilelang": []}
    for heads in (8, 16, 32, 64):
        for inner in (1, 2, 33):
            for packed in (False, True):
                kernel = sparse_mla_fwd_decode_partial_fp8(
                    heads,
                    512,
                    0,
                    2112,
                    packed=packed,
                    sm_scale=512**-0.5,
                    block_I=32,
                    inner_iter=inner,
                    threads=128,
                )
                artifacts["tilelang"].append(
                    dict(
                        heads=heads,
                        inner=inner,
                        packed=packed,
                        source_sha256=hashlib.sha256(
                            kernel.get_kernel_source().encode()
                        ).hexdigest(),
                    )
                )
            sparse_mla_fwd_decode_combine(
                heads,
                512,
                (66 // inner) * 32,
                head_per_block=4,
                block_I=32,
                threads=128,
            )
    Path(os.environ["REMNANT_GLM_COMPILE_MANIFEST"]).write_text(
        json.dumps(artifacts, indent=2) + "\n"
    )


if __name__ == "__main__":
    import sys

    if sys.argv[1:] != ["--compile-only"]:
        raise SystemExit("Expected --compile-only; use pytest for execution")
    precompile_sm90()
