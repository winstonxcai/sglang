"""Pool layout and capacity contracts for the Remnant C4 format.

Author: Winston Cai.
"""

import gc
import hashlib
import json
import math
import os
import unittest
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import pytest
import torch

from sglang.srt import remnant
from sglang.srt.mem_cache import deepseek_v4_memory_pool as pool_module
from sglang.srt.mem_cache.deepseek_v4_memory_pool import RemnantPackedKVPool
from sglang.srt.model_executor.pool_configurator import DSV4PoolConfigurator
from sglang.srt.remnant import config
from sglang.test.test_utils import CustomTestCase

MODEL_CONFIG_SHA256 = "6c8f3d2d3b48707541b88f32f22ef3f0f8a6b57d8523281e2b8d3cdb0ae9a023"


def _gpu_storages(value, visited=None):
    """Count pool-owned CUDA backing allocations once, including aliased views."""
    if visited is None:
        visited = set()
    if id(value) in visited:
        return {}
    visited.add(id(value))
    if isinstance(value, torch.Tensor):
        if value.device.type != "cuda":
            return {}
        storage = value.untyped_storage()
        return {(str(value.device), storage.data_ptr()): storage.nbytes()}
    if isinstance(value, dict):
        children = value.values()
    elif isinstance(value, (tuple, list)):
        children = value
    elif type(value).__module__.startswith("sglang.srt.mem_cache"):
        children = vars(value).values()
    else:
        return {}
    result = {}
    for child in children:
        result.update(_gpu_storages(child, visited))
    return result


def _native_pool_bytes(size, page, reserve, layers):
    # Independent V4 geometry: 576 value bytes + eight scales, page-aligned to 576.
    pages = math.ceil((size + reserve) / page)
    return pages * math.ceil(page * 584 / 576) * 576 * layers


def _cuda_snapshot():
    torch.cuda.synchronize()
    return {
        "allocated": torch.cuda.memory_allocated(),
        "reserved": torch.cuda.memory_reserved(),
        "peak_allocated": torch.cuda.max_memory_allocated(),
        "active_requested_bytes": sum(
            block["requested_size"]
            for segment in torch.cuda.memory_snapshot()
            for block in segment["blocks"]
            if block["state"] == "active_allocated"
        ),
    }


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_real_gpu_allocations(tmp_path):
    """Normal constructors, one TP4 rank's geometry; no weights or process groups.

    REMNANT_MODEL_CONFIG supplies the pinned config-only download from CPU setup.
    REMNANT_ALLOCATION_REPORT optionally retains incremental measurements outside
    pytest's temporary directory. Allocator reservation is diagnostic, not storage.
    """
    from sglang.srt.runtime_context import get_context
    from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
    from sglang.srt.remnant import NativeWorkspace

    source = Path(
        os.environ.get("REMNANT_MODEL_CONFIG", "/opt/remnant-0731-config.json")
    )
    payload = source.read_bytes()
    assert hashlib.sha256(payload).hexdigest() == MODEL_CONFIG_SHA256
    model = json.loads(payload)
    layers = model["num_hidden_layers"]
    ratios = tuple(model["compress_ratios"])
    assert layers == 43 and model["num_attention_heads"] == 64
    assert [i for i in range(layers) if ratios[i] == 4] == list(range(2, 43, 2))
    assert model["head_dim"] == 512 and model["qk_rope_head_dim"] == 64
    output = Path(
        os.environ.get("REMNANT_ALLOCATION_REPORT", str(tmp_path / "allocation.json"))
    )
    report = {
        "schema": 1,
        "status": "running",
        "model_config_sha256": MODEL_CONFIG_SHA256,
        "scope": "model-free single-rank pools, not usable serving capacity",
        "original_tokens": 131072,
        "request_slots": 32,
        "target_layers": layers,
        "c4_layers": 21,
        "edge_cases": [],
        "modes": [],
        "workspaces": [],
    }

    def save():
        temporary = output.with_suffix(output.suffix + ".tmp")
        temporary.write_text(json.dumps(report, indent=2) + "\n")
        temporary.replace(output)

    def clean():
        gc.collect()
        torch.cuda.synchronize()
        torch.cuda.empty_cache()

    torch.empty(0, device="cuda")  # Establish the CUDA context before snapshots.
    with get_context().override_server_args(
        device="cuda", dsv4_attn_backend="flashmla"
    ):
        for rows in (0, 1, 63, 64, 65, 130, 32768):
            for count in (None, 64, 128, 192, 256, 320, 384, 448):
                remnant.configure_cache_format(
                    "native" if count is None else "remnant", "triton"
                )
                cls = (
                    pool_module.DeepSeekV4SingleKVPool
                    if count is None
                    else RemnantPackedKVPool
                )
                options = (
                    {}
                    if count is None
                    else {"layouts": (config.PackedLayerLayout(2, count),)}
                )
                clean()
                torch.cuda.reset_peak_memory_stats()
                before = _cuda_snapshot()
                pool = cls(
                    rows,
                    64,
                    torch.float8_e4m3fn,
                    448,
                    64,
                    1,
                    "cuda",
                    False,
                    global_page_size=256,
                    kv_layout=pool_module.KVLayout.V4,
                    **options
                )
                pages = math.ceil((rows + 256) / 64)
                expected = (
                    _native_pool_bytes(rows, 64, 256, 1)
                    if count is None
                    else pages * 64 * (count + 72)
                )
                measured = sum(_gpu_storages(pool).values())
                assert measured == expected
                plane_bytes = (
                    {"native": measured}
                    if count is None
                    else {
                        name: tensor.nbytes
                        for name, tensor in zip(
                            ("values", "bitmaps", "scales"), pool.get_packed_buffers(0)
                        )
                    }
                )
                report["edge_cases"].append(
                    {
                        "rows": rows,
                        "survivors": count,
                        "pages": pages,
                        "physical_rows": pages * 64,
                        "reserved_and_padding_rows": pages * 64 - rows,
                        "page_padding_bytes": (
                            pages * (64 * 585 - 64 * 584) if count is None else 0
                        ),
                        "plane_bytes": plane_bytes,
                        "storage_bytes": measured,
                        "cuda_before": before,
                        "cuda_after": _cuda_snapshot(),
                    }
                )
                save()
                del pool
                clean()

    previous = None
    for mode in ("native", "triton", "kernel"):
        with get_context().override_server_args(
            device="cuda",
            dsv4_attn_backend="flashmla",
            max_running_requests=32,
            chunked_prefill_size=8192,
            page_size=256,
            dsv4_c4_cache_format="native" if mode == "native" else "remnant",
            dsv4_c4_cache_implementation="triton" if mode == "triton" else "kernel",
            _dsv4_c4_cache_compression_ratios=ratios,
        ):
            hf = SimpleNamespace(**model, kv_source_layer_ids=())
            metadata = SimpleNamespace(
                qk_nope_head_dim=448,
                qk_rope_head_dim=64,
                index_head_dim=128,
                context_len=model["max_position_embeddings"],
                compress_ratios=ratios,
                window_size=128,
                hf_config=hf,
            )
            sizing = DSV4PoolConfigurator(
                SimpleNamespace(
                    kv_cache_dtype_str="fp8_e4m3",
                    model_config=metadata,
                    layer_info=SimpleNamespace(start_layer=0, end_layer=layers),
                    pp_size=1,
                    attn_dp_size=1,
                    page_size=256,
                    sliding_window_size=128,
                    spec_algorithm=SpeculativeAlgorithm.NONE,
                )
            )
            sizes = sizing.calculate_pool_sizes_from_max_tokens(131072, 256)
            sizes.max_running_requests = 32
            sizes = sizing.finalize_with_max_running_requests(sizes)
            clean()
            torch.cuda.reset_peak_memory_stats()
            before = _cuda_snapshot()
            pool = pool_module.DeepSeekV4TokenToKVPool(
                max_num_reqs=32,
                swa_size=sizes.swa_max_total_num_tokens,
                c4_size=sizes.c4_max_total_num_tokens,
                c128_size=sizes.c128_max_total_num_tokens,
                c4_state_pool_size=sizes.c4_state_pool_size,
                c128_state_pool_size=sizes.c128_state_pool_size,
                page_size=256,
                swa_page_size=256,
                dtype=torch.float8_e4m3fn,
                c4_state_dtype=torch.float32,
                c128_state_dtype=torch.float32,
                qk_nope_head_dim=448,
                qk_rope_head_dim=64,
                indexer_head_dim=128,
                layer_num=layers,
                device="cuda",
                enable_memory_saver=False,
                compression_ratios=ratios,
                start_layer=0,
                end_layer=layers,
                full_size=131072,
                kv_layout=pool_module.KVLayout.V4,
            )
            assert not pool._unified_kv and not pool.uniform_fp8
            assert pool.request_window is None and len(pool.c4_kv_pool.kv_buffer) == 21
            groups = {
                name: _gpu_storages(owner)
                for name, owner in {
                    "swa": pool.swa_kv_pool,
                    "c4": pool.c4_kv_pool,
                    "c128": pool.c128_kv_pool,
                    "indexer": pool.index_pools,
                    "compressor_state": (
                        pool.compress_state_pools,
                        pool.indexer_compress_state_pools,
                    ),
                }.items()
            }
            ownership = {}
            for name, storages in groups.items():
                assert ownership.keys().isdisjoint(storages), name
                ownership.update(storages)
            assert ownership == _gpu_storages(pool), "unaccounted pool-owned storage"
            actual = {name: sum(storages.values()) for name, storages in groups.items()}
            c4_pages = math.ceil((sizes.c4_max_total_num_tokens + 256) / 64)
            expected_c4 = c4_pages * 64 * 21 * (585 if mode == "native" else 328)
            assert actual["c4"] == expected_c4
            assert actual["swa"] == _native_pool_bytes(
                sizes.swa_max_total_num_tokens, 256, 256, layers
            )
            assert actual["c128"] == _native_pool_bytes(
                sizes.c128_max_total_num_tokens, 2, 256, 20
            )
            # Default FP8 index rows contain 128 values and one FP32 scale.
            assert actual["indexer"] == c4_pages * 64 * 132 * 21
            # Offline C4 rings follow SWA pages; C128 rings follow request slots
            # (including the dummy slot). Each state contains KV and score;
            # C4 overlap doubles both. All state values are FP32.
            c4_state_rows = 4 * math.ceil(
                (8 * (sizes.swa_max_total_num_tokens // 256) + 9) / 4
            )
            c128_state_rows = 128 * math.ceil((33 * 128 + 129) / 128)
            expected_states = (
                c4_state_rows * 4 * (512 + 128) * 4 * 21
                + c128_state_rows * 2 * 512 * 4 * 20
            )
            assert actual["compressor_state"] == expected_states
            if previous is not None:
                assert all(actual[k] == previous[k] for k in actual if k != "c4")
            previous = actual
            report["modes"].append(
                {
                    "mode": mode,
                    "pool_sizes": vars(sizes),
                    "component_bytes": actual,
                    "persistent_bytes": sum(actual.values()),
                    "cuda_before": before,
                    "cuda_after": _cuda_snapshot(),
                }
            )
            save()
            del pool, groups, ownership
            clean()
            for batch in (8, 24, 32):
                with get_context().override_server_args(
                    dsv4_c4_cache_format="native" if mode == "native" else "remnant",
                    dsv4_c4_cache_implementation=(
                        "triton" if mode == "triton" else "kernel"
                    ),
                ):
                    workspace = (
                        None
                        if mode == "native"
                        else NativeWorkspace.allocate(batch, 512, 64, "cuda")
                    )
                tensors = (
                    ()
                    if workspace is None
                    else (
                        workspace.native_bytes,
                        workspace.temporary_indices,
                        workspace.dense_bf16,
                    )
                )
                measured = sum(_gpu_storages(tensors).values())
                expected = (
                    0
                    if mode == "native"
                    else batch * 512 // 64 * 37440
                    + batch * 512 * 4
                    + (batch * 512 * 512 * 2 if mode == "triton" else 0)
                )
                assert measured == expected
                report["workspaces"].append(
                    {
                        "mode": mode,
                        "batch": batch,
                        "selected_k": 512,
                        "storage_bytes": measured,
                        "owner": "one backend, reused across layers",
                    }
                )
                save()
                del workspace, tensors
                clean()
    report["status"] = "passed"
    save()


class TestRemnantPool(CustomTestCase):

    def setUp(self):
        super().setUp()
        patcher = mock.patch(
            "sglang.srt.model_executor.pool_configurator.get_exec",
            return_value=SimpleNamespace(
                kernel=SimpleNamespace(dsv4_attn_backend="flashmla")
            ),
        )
        patcher.start()
        self.addCleanup(patcher.stop)
        layout = mock.patch.object(
            pool_module,
            "select_dsv4_kv_layout",
            return_value=(pool_module.KVLayout.V4, None),
        )
        layout.start()
        self.addCleanup(layout.stop)

    def tearDown(self):
        remnant.configure_cache_format("native")

    def _pool(self, counts):
        pool = object.__new__(RemnantPackedKVPool)
        pool.layouts = tuple(
            config.PackedLayerLayout(i * 2 + 2, k) for i, k in enumerate(counts)
        )
        pool.layer_num, pool.size, pool.page_size = len(counts), 130, 64
        pool.global_page_size = 256
        pool.kv_layout = pool_module.KVLayout.V4
        pool.start_layer, pool.device, pool.custom_mem_pool = 0, "cpu", None
        pool.memory_saver_adapter = SimpleNamespace(region=lambda unused: nullcontext())
        pool._create_buffers()
        return pool

    def test_actual_allocation_matches_layer_profiles(self):
        for counts in [(k,) for k in config.VALID_KEEP_COUNTS] + [(128, 192, 256, 320)]:
            pool = self._pool(counts)
            pages = 7  # v0.5.22: ceil((130 + logical reserve 256) / 64).
            self.assertEqual(
                pool.get_kv_size_bytes(), pages * 64 * sum(k + 72 for k in counts)
            )
            ptrs, lengths, items = pool.get_buf_infos()
            self.assertEqual(sum(lengths), pool.get_kv_size_bytes())
            self.assertEqual(len(ptrs), 3 * len(counts))
            for layer, count in enumerate(counts):
                values, bitmap, scales = pool.get_packed_buffers(layer)
                self.assertEqual(values.shape, (pages, 64, count))
                self.assertEqual(bitmap.dtype, torch.uint64)
                self.assertEqual(scales.shape, (pages, 64, 8))
                self.assertEqual(pool.get_bytes_per_token(layer), count + 72)
                self.assertEqual(
                    items[3 * layer : 3 * layer + 3], [64 * count, 64 * 64, 64 * 8]
                )
            if len(set(counts)) > 1:
                with self.assertRaisesRegex(ValueError, "layer index"):
                    pool.get_bytes_per_token()
            for invalid_layer in (-1, len(counts)):
                with self.assertRaises(IndexError):
                    pool.get_packed_buffers(invalid_layer)

    def test_stage_pool_uses_absolute_model_layer_layouts(self):
        ratios = (0, 4, 128, 4, 128, 4, 4)
        remnant.configure_cache_format(
            "remnant",
            "triton",
            keep_count=128,
            layer_keep_counts='{"3":192,"5":320}',
            compression_ratios=ratios,
        )
        stage = SimpleNamespace(
            compression_ratios=ratios,
            _stage_start=3,
            _stage_end=7,
            qk_nope_head_dim=448,
            qk_rope_head_dim=64,
        )
        with mock.patch.object(pool_module, "RemnantPackedKVPool") as constructor:
            pool_module.DeepSeekV4TokenToKVPool._make_kv_pool(
                stage,
                cls=constructor,
                size=128,
                page_size=64,
                dtype=torch.float8_e4m3fn,
                layer_num=3,
                device="cpu",
                enable_memory_saver=False,
                global_page_size=256,
            )
        layouts = constructor.call_args.kwargs["layouts"]
        self.assertEqual(
            [(x.layer_id, x.keep_count) for x in layouts],
            [(3, 192), (5, 320), (6, 128)],
        )

    def test_record_layout_is_328_bytes(self):
        values = config.PACKED_KEPT_VALUES
        bitmap = config.BITMAP_WORDS * 8
        scales = config.BITMAP_WORDS
        self.assertEqual(values + bitmap + scales, 328)
        self.assertEqual(self._pool((256,)).get_bytes_per_token(), 328)
        self.assertEqual(config.NATIVE_RECORD_BYTES, 584)

    def test_pool_sizing_uses_packed_record_for_c4_only(self):
        configurator = object.__new__(DSV4PoolConfigurator)
        configurator.qk_nope_head_dim = 448
        configurator.qk_rope_head_dim = 64
        configurator.indexer_head_dim = 128
        configurator.swa_ratio = 0.5
        configurator.c4_shrink_factor = 1
        configurator.page_size = 256
        configurator._unified = False
        configurator.swa_cap_tokens = None
        configurator.bytes_per_swa_token = 0
        configurator.indexer_bytes_per_token = 132
        configurator.stage_owner_layers = {4: [2, 4], 128: [1, 3]}
        configurator.swa_page_size = 128
        configurator.c4_ring_size = 8
        configurator.num_layers_total = 4
        configurator.num_layers_ca4 = 2
        configurator.num_layers_ca128 = 2
        configurator.packed_c4_layouts = (
            config.PackedLayerLayout(2),
            config.PackedLayerLayout(4),
        )

        with mock.patch.object(remnant, "packed_enabled", return_value=False):
            native_bytes = configurator._get_bytes_per_full_token()
        with mock.patch.object(remnant, "packed_enabled", return_value=True):
            packed_bytes = configurator._get_bytes_per_full_token()

        native_c4_bytes = pool_module.KVLayout.V4.page_bytes(64) / 64
        expected_saving = (native_c4_bytes - 328) / 4 * configurator.num_layers_ca4
        self.assertAlmostEqual(native_bytes - packed_bytes, expected_saving)
        self.assertLess(packed_bytes, native_bytes)
        configurator.num_layers_total = 6
        configurator.num_layers_ca4 = 4
        configurator.stage_owner_layers[4] = [0, 1, 2, 3]
        configurator.packed_c4_layouts = tuple(
            config.PackedLayerLayout(i, count)
            for i, count in enumerate((128, 192, 256, 320))
        )
        with mock.patch.object(remnant, "packed_enabled", return_value=False):
            native_bytes = configurator._get_bytes_per_full_token()
        with mock.patch.object(remnant, "packed_enabled", return_value=True):
            mixed_bytes = configurator._get_bytes_per_full_token()
        self.assertAlmostEqual(
            native_bytes - mixed_bytes,
            sum(native_c4_bytes - count - 72 for count in (128, 192, 256, 320)) / 4,
        )


if __name__ == "__main__":
    unittest.main()
