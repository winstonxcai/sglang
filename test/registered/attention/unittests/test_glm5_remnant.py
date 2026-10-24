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

ATTENTION_SCALE = 256**-0.5  # Native scales the original head, before 512-D absorption.

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="H100 required")


@pytest.fixture(autouse=True)
def model_free_rank_context():
    from sglang.srt.runtime_context import get_parallel

    with torch.inference_mode(), get_parallel().override(
        attn_dcp_rank=0, attn_cp_rank=0, attn_tp_rank=0
    ):
        yield


def test_compiled_launch_contract():
    from sglang.srt.remnant import glm

    calls = []

    class Kernel:
        def __getitem__(self, grid):
            assert len(grid) == 3

            def launch(*args):
                calls.append((grid, args))

            return launch

    raw = torch.ones(4, 512, dtype=torch.bfloat16)
    values = torch.zeros(8, 256, dtype=torch.uint8)
    bitmap = torch.zeros(8, 8, dtype=torch.uint64)
    indices = torch.arange(4, dtype=torch.int32)
    output = torch.empty(4, 512, dtype=torch.uint8)
    with pytest.MonkeyPatch.context() as patcher:
        patcher.setattr(glm, "compile_kernels", lambda: (Kernel(), Kernel()))
        glm.normalize_and_store(raw, lambda x: x, indices.long(), values, bitmap)
        glm.reconstruct(PackedCache(values, bitmap, None, "direct"), indices, output)
    assert [grid for grid, _ in calls] == [(4, 1, 1), (4, 1, 1)]
    assert [len(args) for _, args in calls] == [6, 5]


def fixture(rows=513, implementation="direct"):
    torch.manual_seed(20261024)
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


def test_native_fp8_conversion():
    with get_context().override_server_args(device="cuda"):
        raw, norm, cache = fixture()
        norm.weight.data.fill_(1024)
        _, values, words, _ = expected(raw[:4], norm)
        normalize_and_store(
            raw[:4], norm, torch.arange(4, device="cuda"), cache.values, cache.bitmaps
        )
        assert torch.equal(cache.values[:4], values)
        assert torch.equal(cache.bitmaps[:4].cpu().view(torch.int64), words)


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
            q, dense.view(torch.float8_e4m3fn).unsqueeze(1), valid, ATTENTION_SCALE
        )
        for impl in ("adapter", "direct"):
            actual = attention(
                q,
                PackedCache(cache.values, cache.bitmaps, cache.workspace, impl),
                indices,
                ATTENTION_SCALE,
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


def test_partial_outputs():
    from sglang.kernels.ops.attention.dsa.tilelang_kernel import (
        sparse_mla_fwd_decode_partial_fp8,
    )

    with get_context().override_server_args(device="cuda"):
        raw, norm, cache = fixture()
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
        native = dense.view(torch.float8_e4m3fn).unsqueeze(0).unsqueeze(2)
        packed = cache.values.view(torch.float8_e4m3fn).unsqueeze(0).unsqueeze(2)
        dummy = dense.view(torch.uint64).reshape(-1)[:1].reshape(1, 1)
        for heads in (8, 64):
            for batch, inner in ((1, 1), (8, 2), (128, 33)):
                q = (
                    torch.randn(batch, heads, 512, device="cuda")
                    .to(torch.float8_e4m3fn)
                    .unsqueeze(0)
                )
                indices = selection(batch, 3).unsqueeze(0)
                outputs = []
                for mode, kv, bitmap in (
                    (False, native, dummy),
                    (True, packed, cache.bitmaps),
                ):
                    kernel = sparse_mla_fwd_decode_partial_fp8(
                        heads,
                        512,
                        0,
                        2112,
                        packed=mode,
                        sm_scale=ATTENTION_SCALE,
                        block_I=32,
                        inner_iter=inner,
                        threads=128,
                    )
                    outputs.append(kernel(q, kv, bitmap, indices))
                for actual, reference in zip(outputs[1], outputs[0]):
                    torch.testing.assert_close(actual, reference, atol=0.02, rtol=0.02)
                inactive = outputs[0][1] == -(2**30)
                assert torch.equal(outputs[1][1][inactive], outputs[0][1][inactive])


def test_adapter_chunk_boundary():
    test_attention(8, 129, 3)


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
        reference = tilelang_sparse_fwd(q, dense, indices, ATTENTION_SCALE)
        for implementation in ("adapter", "direct"):
            candidate = PackedCache(
                cache.values, cache.bitmaps, cache.workspace, implementation
            )
            out = attention(q, candidate, indices, ATTENTION_SCALE)
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
            return attention(q, cache, indices, ATTENTION_SCALE)

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
        max_running_requests=33,  # req_to_token.shape[0], including its dummy row
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
        pool.normalize_and_store_glm(
            3, raw[:2], norm, torch.zeros(2, dtype=torch.int64, device="cuda")
        )
        assert not bool(cache.values[0].any())
        assert not bool(cache.bitmaps[0].view(torch.int64).any())
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
            max_running_requests=33,  # req_to_token.shape[0], including its dummy row
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
            q, cache, 512, selection(8, 3)[:, 0, :2051], ATTENTION_SCALE
        )
        ref = attention(q, cache, selection(8, 3), ATTENTION_SCALE)
        torch.testing.assert_close(out, ref, atol=0, rtol=0)


def test_allocator_accounting():
    from types import SimpleNamespace
    from sglang.srt.configs.glm5_next import Glm5NextConfig
    from sglang.srt.model_executor.pool_configurator import DefaultPoolConfigurator
    from sglang.srt.speculative.spec_info import SpeculativeAlgorithm

    hf = Glm5NextConfig(
        **json.loads(Path(os.environ["REMNANT_GLM_MODEL_CONFIG"]).read_text())
    )
    model = SimpleNamespace(
        hf_config=hf,
        is_draft_model=False,
        kv_lora_rank=512,
        qk_rope_head_dim=0,
        context_len=1048576,
    )
    kvc = SimpleNamespace(
        model_config=model,
        kv_cache_dtype_str="fp8_e4m3",
        kv_cache_dtype=torch.float8_e4m3fn,
        use_mla_backend=True,
        layer_info=SimpleNamespace(
            start_layer=0, end_layer=45, num_effective_layers=45
        ),
        server_args=SimpleNamespace(enable_hisparse=False),
        is_draft_worker=False,
        mambaish_config=hf.get_text_config(),
        pp_size=1,
        spec_algorithm=SpeculativeAlgorithm.NONE,
    )
    with get_context().override_server_args(
        device="cuda",
        tp_size=8,
        dsa_prefill_backend="tilelang",
        dsa_decode_backend="tilelang",
    ):
        for mode, cell in (("native", 5995), ("adapter", 3883), ("direct", 3883)):
            with get_context().override_server_args(
                device="cuda",
                tp_size=8,
                dsa_prefill_backend="tilelang",
                dsa_decode_backend="tilelang",
                glm5_mla_cache_format="native" if mode == "native" else "remnant",
                glm5_mla_cache_implementation=(
                    "adapter" if mode == "adapter" else "direct"
                ),
            ):
                profile = DefaultPoolConfigurator(kvc)
                overhead = 128 * 2112 * 512 if mode == "adapter" else 0
                assert profile._cell_size == cell, (mode, profile._cell_size, cell)
                assert profile._fixed_overhead_bytes == overhead
                assert (
                    profile.calculate_pool_sizes(
                        overhead + cell * 131072, 256
                    ).max_total_num_tokens
                    == 131072
                )
                assert (
                    profile.calculate_pool_sizes(
                        overhead + cell * 131072 - 1, 256
                    ).max_total_num_tokens
                    == 131072 - 256
                )


def test_real_gpu_allocations(tmp_path):
    from sglang.srt.configs.mamba_utils import (
        KimiLinearCacheParams,
        KimiLinearStateShape,
        Mamba2StateDType,
    )
    from sglang.srt.mem_cache.memory_pool import HybridReqToTokenPool
    from dsv4.test_remnant_pool import _gpu_storages

    source = Path(os.environ["REMNANT_GLM_MODEL_CONFIG"])
    assert (
        hashlib.sha256(source.read_bytes()).hexdigest()
        == "bb8f01c42cb92a52ca72e65afb4d5bd8d11aef083cd210e8de25dfb904f23e9f"
    )
    text = json.loads(source.read_text())["text_config"]
    assert text["num_hidden_layers"] == 45 and text["index_kpool"] == 4
    assert text["qk_nope_head_dim"] == 256 and text["qk_rope_head_dim"] == 0
    assert text["rms_norm_eps"] == 1e-5
    assert [
        i
        for i, kind in enumerate(text["layer_types"])
        if kind == "deepseek_sparse_attention"
    ] == list(range(3, 44, 4))
    request_context = text["max_position_embeddings"] + 4
    report = {
        "status": "running",
        "modes": [],
        "edges": [],
        "tokens": 131072,
        "requests": 32,
        "tp": 8,
        "context_length": text["max_position_embeddings"],
        "extra_context_rows": 4,
    }
    output = Path(
        os.environ.get("REMNANT_ALLOCATION_REPORT", tmp_path / "allocation.json")
    )
    with get_context().override_server_args(
        device="cuda",
        tp_size=8,
        dsa_prefill_backend="tilelang",
        dsa_decode_backend="tilelang",
    ):
        for mode in ("native", "adapter", "direct"):
            with get_context().override_server_args(
                device="cuda",
                tp_size=8,
                dsa_prefill_backend="tilelang",
                dsa_decode_backend="tilelang",
                glm5_mla_cache_implementation=mode if mode != "native" else "direct",
                glm5_mla_cache_format="native" if mode == "native" else "remnant",
            ):
                row_bytes = 512 if mode == "native" else 320
                for size in (0, 1, 255, 256, 257, 131072):
                    torch.cuda.synchronize()
                    mamba = None
                    request_pool = None
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
                        request_pool = HybridReqToTokenPool(
                            size=32,
                            mamba_size=160,
                            mamba_spec_state_size=32,
                            max_context_len=request_context,
                            device="cuda",
                            enable_memory_saver=False,
                            cache_params=params,
                            mamba_layer_ids=linear,
                            enable_mamba_extra_buffer=True,
                        )
                        mamba = request_pool.mamba_pool
                    pool = make_pool(size, mode, mamba=mamba)
                    owned = _gpu_storages(vars(pool.full_kv_pool))
                    if mamba is not None:
                        owned.update(_gpu_storages(request_pool))
                    measured = sum(owned.values())
                    # Reconcile actual active CUDA blocks as well as requested
                    # tensor storages. Allocator reservation remains diagnostic.
                    wanted = {
                        address: nbytes
                        for (_, address), nbytes in owned.items()
                        if nbytes
                    }
                    blocks = {}
                    for segment in torch.cuda.memory_snapshot():
                        address = segment["address"]
                        for block in segment["blocks"]:
                            block_address = block.get("address", address)
                            if (
                                block["state"] == "active_allocated"
                                and block_address in wanted
                            ):
                                assert block["requested_size"] == wanted[block_address]
                                blocks[block_address] = block["size"]
                            address += block["size"]
                    assert (
                        blocks.keys() == wanted.keys()
                    ), "Some pool backing blocks were not reconciled"
                    physical_bytes = sum(blocks.values())
                    workspace = 128 * 2112 * 512 if mode == "adapter" else 0
                    latent = 11 * (size + 256) * row_bytes
                    indexer = 11 * ((size + 257) // 256) * 64 * 132
                    tails = 11 * 2 * 34 * 4 * 128 * 2
                    pointers = 11 * 8
                    kda = (
                        161 * 18452480 + 32 if mamba is not None else 0
                    )  # four one-conv descriptor arrays
                    request_metadata = (
                        33 * request_context * 4 + 160 * 8 + 33 * 4 + 33 * 2 * 8
                        if request_pool is not None
                        else 0
                    )
                    expected_bytes = (
                        latent
                        + indexer
                        + tails
                        + pointers
                        + workspace
                        + kda
                        + request_metadata
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
                        active_block_bytes=physical_bytes,
                        allocator_padding_bytes=physical_bytes - measured,
                        allocator_reserved_bytes=torch.cuda.memory_reserved(),
                        latent=latent,
                        indexer=indexer,
                        tails=tails,
                        kda=kda,
                        workspace=workspace,
                        pointers=pointers,
                        request_metadata=request_metadata,
                    )
                    report["modes" if size == 131072 else "edges"].append(entry)
                    output.write_text(json.dumps(report, indent=2) + "\n")
                    del pool, mamba, request_pool
                    gc.collect()
                    torch.cuda.empty_cache()
    report["status"] = "passed"
    output.write_text(json.dumps(report, indent=2) + "\n")


def _compile_sm90_variant(case: tuple[int, int]) -> list[dict]:
    """One independent head/split pair, owned by a CPU compiler process."""
    from sglang.kernels.ops.attention.dsa.tilelang_kernel import (
        sparse_mla_fwd_decode_partial_fp8,
        sparse_mla_fwd_decode_combine,
    )

    heads, inner = case
    variants = []
    for packed in (False, True):
        kernel = sparse_mla_fwd_decode_partial_fp8(
            heads,
            512,
            0,
            2112,
            packed=packed,
            sm_scale=ATTENTION_SCALE,
            block_I=32,
            inner_iter=inner,
            threads=128,
        )
        variants.append(
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
        heads, 512, (66 // inner) * 32, head_per_block=4, block_I=32, threads=128
    )
    return variants


def precompile_sm90():
    """Build exact CPU artifacts without allocating a CUDA tensor or context."""
    import multiprocessing
    from concurrent.futures import ProcessPoolExecutor
    from sglang.srt.remnant.glm import compile_kernels

    store, unpack = compile_kernels()
    artifacts = {
        "attention_scale": ATTENTION_SCALE,
        "precompile_workers": 4,
        "triton": [store.hash, unpack.hash],
        "tilelang": [],
    }
    cases = [(heads, inner) for heads in (8, 16, 32, 64) for inner in (1, 2, 33, 66)]
    # Separate processes isolate TVM compiler state; each pair owns its combine
    # variant. The maintained runner supplies the overall bounded subprocess.
    with ProcessPoolExecutor(
        max_workers=4, mp_context=multiprocessing.get_context("spawn")
    ) as executor:
        for variants in executor.map(_compile_sm90_variant, cases):
            artifacts["tilelang"].extend(variants)
            print(
                "Compiled GLM variants",
                [(v["heads"], v["inner"], v["packed"]) for v in variants],
                flush=True,
            )
    artifacts["cache_files"] = {
        str(path): hashlib.sha256(path.read_bytes()).hexdigest()
        for root in (
            Path(os.environ["TRITON_CACHE_DIR"]),
            Path(os.environ["TILELANG_CACHE_DIR"]),
        )
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }
    Path(os.environ["REMNANT_GLM_COMPILE_MANIFEST"]).write_text(
        json.dumps(artifacts, indent=2) + "\n"
    )


if __name__ == "__main__":
    import sys

    if sys.argv[1:] != ["--compile-only"]:
        raise SystemExit("Expected --compile-only; use pytest for execution")
    precompile_sm90()
