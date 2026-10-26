"""Model-free GLM production codec, TileLang, graph and allocation qualification.

Author: Winston Cai.

Run via scripts/modal/kernels.py --model-family glm53. No weights.
Focused partial-graph tests consume REMNANT_GLM_COMPILE_MANIFEST from
python -m sglang.benchmark.remnant.glm_artifacts; no parent metadata is needed. The Native
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

from sglang.benchmark.remnant.glm_artifacts import ATTENTION_SCALE, packed_loader_probe
from sglang.kernels.ops.attention.dsa.tilelang_kernel import tilelang_sparse_fwd
from sglang.srt.layers.layernorm import RMSNorm
from sglang.srt.remnant.glm import (
    PackedCache,
    attention,
    normalize_and_store,
    reconstruct,
)
from sglang.srt.remnant.reference import topmag_keep_mask
from sglang.srt.runtime_context import get_context
from sglang.test.ci.ci_register import register_cuda_ci

# This suite requires the explicitly owned preparation/qualification workflow.
# A disabled generic CI registration does not skip the maintained pytest runner.
register_cuda_ci(
    est_time=0,
    stage="base-a",
    runner_config="1-gpu-small",
    disabled="Manual H100 qualification with CPU-prepared artifacts; run scripts/modal/kernels.py --model-family glm53",
)

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="H100 required")


@pytest.fixture(autouse=True)
def model_free_rank_context():
    from sglang.srt.runtime_context import get_parallel

    with (
        torch.inference_mode(),
        get_parallel().override(attn_dcp_rank=0, attn_cp_rank=0, attn_tp_rank=0),
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
    from sglang.benchmark.remnant.glm_attention import packed_reference

    mask, values, words, dense = packed_reference(raw, norm)
    return mask, values, words.cpu(), dense


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
    from sglang.srt.mem_cache.memory_pool import DSATokenToKVPool, HybridLinearKVPool
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
    from dsv4.test_remnant_pool import _gpu_storages

    from sglang.srt.configs.mamba_utils import (
        KimiLinearCacheParams,
        KimiLinearStateShape,
        Mamba2StateDType,
    )
    from sglang.srt.mem_cache.memory_pool import HybridReqToTokenPool

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
                    assert blocks.keys() == wanted.keys(), (
                        "Some pool backing blocks were not reconciled"
                    )
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


@pytest.mark.parametrize("heads,banks", [(8, 2), (16, 4)])
def test_benchmark_bank_graph_ownership(heads, banks):
    """Small correctness fixture, not the 128K timing/cache-residency protocol."""
    _benchmark_bank_graph(heads, banks, 128 if heads == 8 else 512)


def _benchmark_bank_graph(
    heads, banks, queries, *, warmup_calls=10, reference_device=None, phase="prefill"
):
    import time

    from sglang.benchmark.remnant.glm_attention import Case, Fixture
    from sglang.kernels.ops.attention.dsa.tilelang_kernel import _pick_inner_iter

    raw, norm, cache = fixture()
    f = Fixture.__new__(Fixture)
    f.case = Case(phase, heads, queries, 3, "independent", "rotating")
    f.deadline = time.monotonic() + 1200
    f.banks, f.memory_limit = (
        banks,
        int(torch.cuda.get_device_properties(0).total_memory * 0.7),
    )
    f.q = torch.randn(queries, heads, 512, dtype=torch.bfloat16, device="cuda")
    f.native, f.packed, f.ids, f.graphs, f.outputs = [], [], [], {}, {}
    inner = _pick_inner_iter(
        queries, 66, torch.cuda.get_device_properties(0).multi_processor_count, 1
    )
    f.identity = {"inner_iter": inner}
    for bank in range(banks):
        changed = raw.roll(bank, 0).contiguous()
        values, bitmaps = (
            torch.zeros_like(cache.values),
            torch.zeros_like(cache.bitmaps),
        )
        normalize_and_store(
            changed, norm, torch.arange(len(raw), device="cuda"), values, bitmaps
        )
        native = torch.zeros(len(values), 512, dtype=torch.uint8, device="cuda")
        native[: len(raw)] = norm(changed).to(torch.float8_e4m3fn).view(torch.uint8)
        f.native.append(native)
        f.packed.append(PackedCache(values, bitmaps, None, "direct"))
        f.ids.append(selection(queries, bank % 4))
    if reference_device is None:
        # Unprofiled full-size smoke establishes CPU/GPU oracle parity on actual
        # probe inputs; focused tools then keep cuBLAS oracle work off the GPU.
        from sglang.benchmark.remnant.glm_attention import attention_reference

        sample = min(8, queries)
        args = (f.q[:sample], f.native[0], f.ids[0][:sample, 0])
        torch.testing.assert_close(
            attention_reference(*args),
            attention_reference(*args, device="cpu"),
            atol=0.02,
            rtol=0.02,
        )
    f.validate(reference_device=reference_device)
    f.capture(
        warmup_calls=warmup_calls,
        capture_stream=torch.cuda.current_stream() if phase == "decode" else None,
        reference_device=reference_device,
    )
    f.q.normal_()
    for ids in f.ids:
        ids[1] = -1
    f.validate_replay(reference_device=reference_device)
    f.validate(reference_device=reference_device)
    assert f.identity["native_graph_bytes"] > 0
    assert f.identity["direct_graph_bytes"] > 0
    return f.identity


@pytest.mark.parametrize("heads,banks,queries,inner", [(8, 2, 2, 33), (16, 4, 2, 66)])
def test_benchmark_sanitizer_bank_graph(heads, banks, queries, inner, record_property):
    """Instrument identical compiled variants at explicit representative grids.

    Full Q128/Q512 ownership/dispatch smoke remains separate. Runtime query count
    changes the launch grid, not the compiled partial/combine specialization.
    This probe does not qualify full-size allocation/grid sanitizer coverage.
    """
    target = "sglang.kernels.ops.attention.dsa.tilelang_kernel._pick_inner_iter"
    with patch(target, return_value=inner) as dispatch:
        identity = _benchmark_bank_graph(
            heads, banks, queries, warmup_calls=1, reference_device="cpu"
        )
        assert identity["inner_iter"] == inner
        assert dispatch.call_count > 1
    record_property(
        "probe_geometry",
        json.dumps(
            dict(
                probe="focused-bank-graph",
                heads=heads,
                banks=banks,
                queries=queries,
                forced_inner=inner,
                warmup_calls=1,
                actual_warmup_calls_per_mode=banks,
                reference_device="cpu",
                measured_warmup_calls=10,
            )
        ),
    )


@pytest.mark.parametrize("inner", [1, 2])
@pytest.mark.parametrize("heads", [8, 16])
def test_decode_bank_graph(heads, inner, record_property):
    """Focused decode loader, arithmetic, alternating-bank and changing-input replay."""
    target = "sglang.kernels.ops.attention.dsa.tilelang_kernel._pick_inner_iter"
    record_property(
        "decode_geometry",
        json.dumps(
            dict(
                heads=heads,
                inner=inner,
                banks=4,
                queries=2,
                tails=[0, 1, 2, 3],
                nondefault_stream=True,
            )
        ),
    )
    owner = torch.cuda.current_stream()
    stream = torch.cuda.Stream()
    stream.wait_stream(owner)
    try:
        with torch.cuda.stream(stream), patch(target, return_value=inner):
            identity = _benchmark_bank_graph(
                heads, 4, 2, warmup_calls=1, reference_device="cpu", phase="decode"
            )
            assert identity["inner_iter"] == inner
    finally:
        owner.wait_stream(stream)
        stream.synchronize()


def test_packed_loader_bytes():
    import numpy as np

    probe = packed_loader_probe()
    # 256 * index crosses signed 32-bit at index 8388608.
    slots = 8388610
    values = torch.empty(slots, 256, dtype=torch.uint8, device="cuda")
    bitmap = torch.empty(slots, 8, dtype=torch.uint64, device="cuda")
    host_values = np.tile(np.arange(256, dtype=np.uint8), (16, 1))
    host_words = np.zeros((16, 8), dtype=np.uint64)
    for row in range(16):
        mask = np.zeros(512, dtype=bool)
        if row == 0:
            mask[:256] = True
        elif row == 1:
            mask[256:] = True
        elif row == 2:
            mask[:] = True
        elif row == 3:
            pass
        else:
            mask[
                np.random.default_rng(20261024 + row).choice(512, 256, replace=False)
            ] = True
        for w in range(8):
            host_words[row, w] = sum(
                int(mask[w * 64 + j]) << (63 - j) for j in range(64)
            )
    locations = [*range(15), slots - 1]
    for row, loc in enumerate(locations):
        values[loc].copy_(torch.from_numpy(host_values[row]).cuda())
        bitmap[loc].copy_(torch.from_numpy(host_words[row]).cuda())
    host_ids = np.array([*locations, -1, slots, -9, *locations[:13]], dtype=np.int32)
    expected = np.zeros((32, 512), dtype=np.uint8)
    for row, loc in enumerate(host_ids):
        if loc not in locations:
            continue
        src = locations.index(loc)
        rank = 0
        for j in range(512):
            if int(host_words[src, j // 64]) & (1 << (63 - j % 64)):
                if rank < 256:
                    expected[row, j] = host_values[src, rank]
                rank += 1
    actual = probe(slots)(values, bitmap, torch.from_numpy(host_ids).cuda())
    torch.cuda.synchronize()
    assert np.array_equal(actual.cpu().numpy(), expected)
    # Do not carry the >2 GiB address probe's cached allocation into graph tools.
    del actual, values, bitmap
    torch.cuda.synchronize()
    torch.cuda.empty_cache()


class _CachedPartial:
    """Launch one partition of the exact CPU-built production cubin.

    Restricting the grid does not change the binary, tensor strides, barriers or
    arithmetic. It removes 32/65 empty CTAs and the unchanged combine kernel from
    sanitizer iteration. All arguments and shared bytes come from verified CPU
    metadata, never a guessed ABI. CUDA modules outlive their captured graphs.
    """

    def __init__(self, variant):
        import ctypes as C

        self.C = C
        self.driver = C.CDLL("libcuda.so.1")
        self.module, self.function = C.c_void_p(), C.c_void_p()
        self.variant = variant
        path = Path(variant["cubin_path"])
        assert hashlib.sha256(path.read_bytes()).hexdigest() == variant["cubin_sha256"]
        self._check(self.driver.cuModuleLoad(C.byref(self.module), str(path).encode()))
        try:
            self._check(
                self.driver.cuModuleGetFunction(
                    C.byref(self.function), self.module, variant["symbol"].encode()
                )
            )
        except BaseException:
            self.close()
            raise

    @staticmethod
    def _check(status):
        if status:
            raise RuntimeError(
                f"Focused production partial CUDA Driver error: {status}"
            )

    def __call__(self, tensors, *, slots, stream):
        C = self.C
        arguments = []
        for param in self.variant["parameters"]:
            if param["pointer"]:
                arguments.append(C.c_void_p(tensors[param["name"]].data_ptr()))
            else:
                assert param["type"] == "int"
                arguments.append(C.c_int(slots if param["name"] == "num_pages" else 1))
        pointers = (C.c_void_p * len(arguments))(*(C.addressof(x) for x in arguments))
        self._check(
            self.driver.cuLaunchKernel(
                self.function,
                1,
                1,
                1,
                128,
                1,
                1,
                self.variant["dynamic_shared"],
                C.c_void_p(stream.cuda_stream),
                pointers,
                None,
            )
        )

    def close(self):
        if self.module.value:
            self._check(self.driver.cuModuleUnload(self.module))
            self.module.value = None


@pytest.mark.parametrize("inner", [1, 2])
@pytest.mark.parametrize("heads", [8, 16])
def test_decode_partial_graph(heads, inner, record_property):
    """Essential sanitizer: one exact production CTA, two banks, tails 0..3.

    Fixture creation, reconstruction, finiteness and comparisons run on CPU.
    The instrumented GPU work is only the changed partial kernel, with stock
    Native on independently reconstructed Packed rows as its control. The full
    66/33-partition attention and combine path has a separate uninstrumented gate.
    """
    import numpy as np

    manifest = os.environ.get("REMNANT_GLM_COMPILE_MANIFEST")
    if not manifest:
        pytest.fail(
            "Set REMNANT_GLM_COMPILE_MANIFEST to the output of "
            "python -m sglang.benchmark.remnant.glm_artifacts before focused tests"
        )
    compiled = json.loads(Path(manifest).read_text())
    variants = {
        v["packed"]: v
        for v in compiled["tilelang"]
        if v["heads"] == heads and v["inner"] == inner
    }
    record_property(
        "decode_geometry",
        json.dumps(
            dict(
                heads=heads,
                inner=inner,
                banks=2,
                queries=1,
                tails=[0, 1, 2, 3],
                nondefault_stream=True,
                launched_partitions=1,
                binary_partitions=66 // inner,
            )
        ),
    )
    record_property(
        "partial_cubins",
        json.dumps({str(k): v["cubin_sha256"] for k, v in variants.items()}),
    )
    slots = 513  # Both sides of 256-token page boundaries.
    rng = np.random.default_rng(20261024 + heads * 10 + inner)
    owner, stream = torch.cuda.current_stream(), torch.cuda.Stream()
    stream.wait_stream(owner)
    kernels, graphs = {}, {}
    try:
        with torch.cuda.stream(stream):
            banks = []
            for bank in range(2):
                payload = rng.integers(0, 112, (slots, 256), dtype=np.uint8)
                words = np.zeros((slots, 8), dtype=np.uint64)
                dense = np.zeros((slots, 512), dtype=np.uint8)
                for row in range(slots):
                    coords = np.sort(rng.choice(512, 256, replace=False))
                    # Independent ascending-coordinate scatter; finite raw codes.
                    dense[row, coords] = payload[row]
                    for coord in coords:
                        words[row, coord // 64] |= np.uint64(1) << np.uint64(
                            63 - coord % 64
                        )
                q = torch.from_numpy(
                    rng.normal(0, 0.15, (1, 1, heads, 512)).astype(np.float32)
                ).to(torch.float8_e4m3fn)
                ids = np.full((1, 1, 1, 2112), -1, dtype=np.int32)
                # First partition contains duplicates, invalid and partial rows;
                # tails occupy its final three entries to reach the changed loader.
                ids[0, 0, 0, : 32 * inner] = np.resize(
                    np.array([0, 1, 255, 256, 511, 512, -1, slots], np.int32),
                    32 * inner,
                )
                if inner == 2:
                    ids[0, 0, 0, :32] = -1  # Empty tile followed by a live tile.
                ids[0, 0, 0, 32 * inner - 3 :] = -1
                ids[0, 0, 0, 32 * inner - 3 : 32 * inner - 3 + bank] = (
                    np.arange(bank) + 510
                )
                common = dict(
                    q_fp8=q.view(torch.uint8).cuda(),
                    indices=torch.from_numpy(ids).cuda(),
                    partial_o=torch.empty(
                        1,
                        1,
                        66 // inner,
                        heads,
                        512,
                        dtype=torch.bfloat16,
                        device="cuda",
                    ),
                    partial_lse=torch.empty(
                        1, 1, 66 // inner, heads, dtype=torch.float32, device="cuda"
                    ),
                )
                native_ids = ids.copy()
                native_ids[
                    native_ids >= slots
                ] = -1  # Stock Native guards negative IDs only.
                banks.append(
                    {
                        False: dict(
                            common,
                            indices=torch.from_numpy(native_ids).cuda(),
                            kv_fp8=torch.from_numpy(dense).cuda(),
                            bitmap=torch.zeros(1, 1, dtype=torch.uint64, device="cuda"),
                        ),
                        True: dict(
                            common,
                            kv_fp8=torch.from_numpy(payload).cuda(),
                            bitmap=torch.from_numpy(words).cuda(),
                        ),
                    }
                )
            for mode in (False, True):
                kernels[mode] = _CachedPartial(variants[mode])
                # Preload the module/function before capture; no lazy compilation.
                kernels[mode](banks[0][mode], slots=slots, stream=stream)
            stream.synchronize()
            for mode in (False, True):
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph, stream=stream):
                    for bank in banks:
                        kernels[mode](bank[mode], slots=slots, stream=stream)
                graphs[mode] = graph
            for phase in range(2):
                if phase:
                    for b, bank in enumerate(banks):
                        new_q = (
                            torch.from_numpy(
                                rng.normal(0, 0.15, (1, 1, heads, 512)).astype(
                                    np.float32
                                )
                            )
                            .to(torch.float8_e4m3fn)
                            .view(torch.uint8)
                        )
                        new_ids = bank[True]["indices"].cpu()
                        new_ids[0, 0, 0, 32 * inner - 3 : 32 * inner] = -1
                        new_ids[0, 0, 0, 32 * inner - 3 : 32 * inner - 3 + 2 + b] = (
                            torch.arange(2 + b, dtype=torch.int32) + 510
                        )
                        bank[False]["q_fp8"].copy_(new_q)
                        bank[True]["indices"].copy_(new_ids)
                        new_ids[new_ids >= slots] = -1
                        bank[False]["indices"].copy_(new_ids)
                results = {}
                # Opposite order after mutation verifies alternating mode ownership.
                for mode in (False, True) if not phase else (True, False):
                    graphs[mode].replay()
                    stream.synchronize()
                    results[mode] = [
                        (
                            bank[mode]["partial_o"][0, 0, 0].cpu().float(),
                            bank[mode]["partial_lse"][0, 0, 0].cpu(),
                        )
                        for bank in banks
                    ]
                for actual, expected in zip(results[True], results[False]):
                    assert torch.isfinite(actual[0]).all()
                    assert torch.isfinite(actual[1]).all()
                    for a, e in zip(actual, expected):
                        torch.testing.assert_close(a, e, atol=0.02, rtol=0.02)
    finally:
        # Graph nodes retain CUDA function handles. Destroy them before unloading.
        stream.synchronize()
        for graph in graphs.values():
            graph.reset()
        for kernel in kernels.values():
            kernel.close()
        owner.wait_stream(stream)


def test_adapter_reconstruction_above_2gib():
    """Valid int32 locations must retain 64-bit offsets into the value plane."""
    slots = 8388610
    rows = [8388607, 8388608, 8388609]
    values = torch.empty((slots, 256), dtype=torch.uint8, device="cuda")
    bitmap = torch.empty((slots, 8), dtype=torch.uint64, device="cuda")
    try:
        codes = torch.arange(256, dtype=torch.int32, device="cuda").to(torch.uint8)
        for i, row in enumerate(rows):
            values[row] = codes.roll(i)
            bitmap[row] = torch.tensor(
                [2**64 - 1] * 4 + [0] * 4, dtype=torch.uint64, device="cuda"
            )
        ids = torch.tensor(
            [-1, *rows, rows[1], slots], dtype=torch.int32, device="cuda"
        )
        output = torch.empty((len(ids), 512), dtype=torch.uint8, device="cuda")
        reconstruct(PackedCache(values, bitmap, None, "adapter"), ids, output)
        expected = torch.zeros_like(output)
        for i in range(3):
            expected[i + 1, :256] = codes.roll(i)
        expected[4, :256] = codes.roll(1)
        assert torch.equal(output, expected)
    finally:
        torch.cuda.synchronize()
        del values, bitmap
        torch.cuda.empty_cache()
