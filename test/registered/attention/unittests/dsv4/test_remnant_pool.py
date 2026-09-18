"""Pool layout and capacity contracts for the Remnant C4 format."""

import unittest
from unittest import mock

from sglang.srt import remnant
from sglang.srt.mem_cache.deepseek_v4_memory_pool import RemnantPackedKVPool
from sglang.srt.model_executor.pool_configurator import DSV4PoolConfigurator
from sglang.srt.remnant import config
from sglang.test.test_utils import CustomTestCase


class TestRemnantPool(CustomTestCase):
    def test_record_layout_is_328_bytes(self):
        values = config.PACKED_KEPT_VALUES
        bitmap = config.BITMAP_WORDS * 8
        scales = config.BITMAP_WORDS
        self.assertEqual(values + bitmap + scales, 328)
        self.assertEqual(RemnantPackedKVPool.get_bytes_per_token(object()), 328)
        self.assertEqual(config.NATIVE_RECORD_BYTES, 584)

    def test_pool_sizing_uses_packed_record_for_c4_only(self):
        configurator = object.__new__(DSV4PoolConfigurator)
        configurator.qk_nope_head_dim = 448
        configurator.qk_rope_head_dim = 64
        configurator.indexer_head_dim = 128
        configurator.swa_ratio = 0.5
        configurator.c4_shrink_factor = 1
        configurator.swa_page_size = 128
        configurator.c4_ring_size = 8
        configurator.num_layers_total = 4
        configurator.num_layers_ca4 = 2
        configurator.num_layers_ca128 = 2

        with mock.patch.object(remnant, "packed_enabled", return_value=False):
            native_bytes = configurator._get_bytes_per_full_token()
        with mock.patch.object(remnant, "packed_enabled", return_value=True):
            packed_bytes = configurator._get_bytes_per_full_token()

        expected_saving = (584 - 328) / 4 * configurator.num_layers_ca4
        self.assertAlmostEqual(native_bytes - packed_bytes, expected_saving)
        self.assertLess(packed_bytes, native_bytes)


if __name__ == "__main__":
    unittest.main()
