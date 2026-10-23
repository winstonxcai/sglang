"""Pool layout and capacity contracts for the Remnant C4 format.

Author: Winston Cai.
"""

import unittest
from contextlib import nullcontext
from types import SimpleNamespace
from unittest import mock

import torch

from sglang.srt import remnant
from sglang.srt.mem_cache import deepseek_v4_memory_pool as pool_module
from sglang.srt.mem_cache.deepseek_v4_memory_pool import RemnantPackedKVPool
from sglang.srt.model_executor.pool_configurator import DSV4PoolConfigurator
from sglang.srt.remnant import config
from sglang.test.test_utils import CustomTestCase


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
