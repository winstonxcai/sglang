"""Independent reference tests for the persistent packed row format.

Author: Winston Cai.
"""

import unittest
from types import SimpleNamespace

import torch

from sglang.srt import remnant
from sglang.srt.remnant import config
from sglang.srt.remnant.packed import _as_buffers, pack_rows_fused, unpack_gather_bf16
from sglang.srt.remnant.reference import (
    pack_rows_ref,
    topmag_keep_mask,
    unpack_rows_ref,
)
from sglang.test.test_utils import CustomTestCase


class TestRemnantPack(CustomTestCase):

    def tearDown(self):
        remnant.configure_cache_format("native")

    def test_bf16_reader_opt_in_dispatch_and_empty_inputs(self):
        import sys
        from unittest.mock import patch

        calls = []

        class Kernel:
            def __init__(self, name):
                self.name = name

            def __getitem__(self, grid):
                return lambda *args, **kwargs: calls.append((self.name, grid))

        exports = SimpleNamespace(
            _unpack_gather_bf16_kernel=Kernel("split"),
            _rope_tail_complex_inplace_kernel=Kernel("rope"),
            _unpack_gather_bf16_rope_kernel=Kernel("fused"),
        )
        buffers = (
            torch.zeros((1, 256), dtype=torch.uint8),
            torch.zeros((1, 8), dtype=torch.uint64),
            torch.zeros((1, 8), dtype=torch.uint8),
        )
        index = torch.zeros((1, 1), dtype=torch.int32)
        lengths = torch.ones(1, dtype=torch.int32)
        freqs = torch.ones((16, 32), dtype=torch.complex64)
        output = torch.empty((1, 512), dtype=torch.bfloat16)
        package = unpack_gather_bf16.__module__.rsplit(".", 1)[0]
        with patch.dict(sys.modules, {package + ".triton": exports}):
            unpack_gather_bf16(buffers, index, index, lengths, freqs, output)
            self.assertEqual(calls, [("split", (1,)), ("rope", (1,))])
            calls.clear()
            unpack_gather_bf16(
                buffers, index, index, lengths, freqs, output, fused_rope=True
            )
            self.assertEqual(calls, [("fused", (1,))])
            calls.clear()
            for physical, raw, active in (
                (index[:0], index[:0], lengths[:0]),
                (index[:, :0], index[:, :0], lengths),
            ):
                unpack_gather_bf16(
                    buffers, physical, raw, active, freqs, output, fused_rope=True
                )
            self.assertEqual(calls, [])
            for invalid in (1, 0, "fused", None):
                with self.assertRaisesRegex(TypeError, "fused_rope must be a bool"):
                    unpack_gather_bf16(
                        buffers,
                        index,
                        index,
                        lengths,
                        freqs,
                        output,
                        fused_rope=invalid,
                    )

    def test_profiles_validate_counts_ids_and_atomic_updates(self):
        ratios = (0, 128, 4, 128, 4, 0, 4)
        for count in config.VALID_KEEP_COUNTS:
            remnant.configure_cache_format(
                "remnant",
                "triton",
                keep_count=count,
                layer_keep_counts='{"4":320}',
                compression_ratios=ratios,
            )
            layouts = remnant.packed_layouts(ratios, 3, 7)
            self.assertEqual(
                [(x.layer_id, x.keep_count) for x in layouts], [(4, 320), (6, count)]
            )
            self.assertEqual(layouts[0].record_bytes, 392)
        before = remnant.packed_profile()
        for bad in (0, 63, 65, 512, True, 192.0):
            with self.subTest(count=bad), self.assertRaises(ValueError):
                remnant.configure_cache_format("remnant", "triton", keep_count=bad)
            self.assertIs(remnant.packed_profile(), before)
        for value in (
            "[]",
            "{",
            '{"2":192,"2":256}',
            '{"2":192,"02":256}',
            '{"1":192}',
            '{"99":192}',
            '{"2":512}',
            '{"-2":192}',
        ):
            with self.subTest(overrides=value), self.assertRaises(ValueError):
                remnant.configure_cache_format(
                    "remnant",
                    "triton",
                    layer_keep_counts=value,
                    compression_ratios=ratios,
                )
            self.assertIs(remnant.packed_profile(), before)
        for kwargs in (
            {"implementation": "kernel"},
            {"store_implementation": "kernel"},
            {"hierarchical_cache": True},
        ):
            options = {"implementation": "triton", "keep_count": 192, **kwargs}
            with self.assertRaises(ValueError):
                remnant.configure_cache_format("remnant", **options)
        for kwargs in ({"keep_count": 192}, {"layer_keep_counts": "{}"}):
            with self.assertRaises(ValueError):
                remnant.configure_cache_format("native", **kwargs)
        remnant.configure_from_args(SimpleNamespace())
        self.assertEqual(remnant.packed_profile(), config.PackedProfile())

    def test_resolution_and_worker_profile_restore(self):
        before = remnant.packed_profile()
        remnant.configure_cache_format(
            "remnant",
            "kernel",
            keep_count=192,
            layer_keep_counts='{"2":256,"4":256}',
            publish=False,
        )
        self.assertIs(remnant.packed_profile(), before)
        # Only effective C4 counts govern dispatch: every C4 layer is 256 here.
        ratios = (0, 128, 4, 128, 4)
        args = SimpleNamespace(
            dsv4_c4_cache_format="remnant",
            dsv4_c4_cache_keep_count=192,
            dsv4_c4_cache_layer_keep_counts='{"2":256,"4":256}',
            _dsv4_c4_cache_compression_ratios=ratios,
        )
        remnant.configure_from_args(args)
        self.assertFalse(remnant.packed_profile().nondefault)
        self.assertEqual(remnant.packed_profile().compression_ratios, ratios)
        remnant.configure_from_args(SimpleNamespace())
        self.assertEqual(remnant.packed_profile(), config.PackedProfile())

    def test_buffer_geometry_and_optimized_dispatch_guards(self):
        def buffers(count):
            return (
                torch.empty((2, count), dtype=torch.uint8),
                torch.empty((2, 8), dtype=torch.uint64),
                torch.empty((2, 8), dtype=torch.uint8),
            )

        remnant.configure_cache_format("remnant", "kernel")
        with self.assertRaisesRegex(ValueError, "Triton"):
            _as_buffers(buffers(192))
        remnant.configure_cache_format("remnant", "triton")
        with self.assertRaisesRegex(ValueError, "matching row geometry"):
            _as_buffers((buffers(192)[0], torch.empty(3, 8), torch.empty(2, 8)))
        pool = SimpleNamespace(
            get_packed_buffers=lambda layer: buffers(192),
            get_packed_layout=lambda layer: config.PackedLayerLayout(2, 320),
        )
        with self.assertRaisesRegex(ValueError, "resolved layer layout"):
            _as_buffers(pool, 0)
        with self.assertRaisesRegex(ValueError, "only 256"):
            pack_rows_fused(
                torch.ones(1, 512),
                torch.ones(512),
                1e-6,
                None,
                torch.zeros(1, dtype=torch.int32),
                buffers(192),
            )

    def test_explicit_count_references_across_all_budgets(self):
        # Strictly ordered magnitudes give an independent expected mask; other
        # rows exercise natural zeros, ties and FP32 values straddling BF16.
        x = torch.arange(512, dtype=torch.float32).repeat(4, 1)
        x[1].zero_()
        x[2].fill_(1)
        x[3] = 1 + torch.arange(512).float() / 65536
        for count in config.VALID_KEEP_COUNTS:
            mask = topmag_keep_mask(x, count / 512)
            self.assertTrue(torch.all(mask.sum(-1) == count))
            self.assertTrue(torch.equal(mask[0], torch.arange(512) >= 512 - count))
            values, bitmap, scales = pack_rows_ref(
                x, mask, torch.ones(512), 1e-6, keep_count=count
            )
            self.assertEqual(values.shape, (4, count))
            self.assertEqual(
                values.numel() + bitmap.numel() * 8 + scales.numel(), 4 * (count + 72)
            )
            decoded = unpack_rows_ref(values, bitmap, scales)
            self.assertTrue(torch.isfinite(decoded).all())
            self.assertTrue(torch.all(decoded[~mask] == 0))
            self.assertTrue(torch.all(decoded[1] == 0))
            with self.assertRaises(ValueError):
                pack_rows_ref(
                    x,
                    ~mask,
                    torch.ones(512),
                    1e-6,
                    keep_count=count if count != 256 else 192,
                )

    def test_topmag_keeps_exactly_256_coordinates_per_row(self):
        latent = torch.arange(2 * remnant.HEAD_DIM, dtype=torch.float32).reshape(
            2, remnant.HEAD_DIM
        )
        mask = topmag_keep_mask(latent, 0.5)
        self.assertEqual(tuple(mask.shape), tuple(latent.shape))
        self.assertTrue(torch.all(mask.sum(dim=-1) == 256))
        self.assertTrue(torch.equal(mask[0], latent[0].abs() >= latent[0].abs().kthvalue(257).values))

    def test_reference_pack_round_trip_preserves_kept_values(self):
        torch.manual_seed(7)
        latent = torch.randn(3, remnant.HEAD_DIM, dtype=torch.float32)
        mask = topmag_keep_mask(latent, 0.5)
        norm_weight = torch.ones(remnant.HEAD_DIM, dtype=torch.float32)

        values, bitmaps, scales = pack_rows_ref(
            latent, mask, norm_weight, norm_eps=1e-6
        )
        restored = unpack_rows_ref(values, bitmaps, scales)
        expected = latent.masked_fill(~mask, 0).to(torch.bfloat16)
        expected = expected * torch.rsqrt(
            expected.float().square().mean(-1, keepdim=True) + 1e-6
        ).to(torch.bfloat16)

        self.assertEqual(tuple(values.shape), (3, 256))
        self.assertEqual(tuple(bitmaps.shape), (3, 8))
        self.assertEqual(tuple(scales.shape), (3, 8))
        self.assertTrue(torch.allclose(restored, expected, atol=0.25, rtol=0.05))

    def test_native_mode_does_not_enable_packed_storage(self):
        remnant.configure_cache_format("native")
        self.assertFalse(remnant.packed_enabled())
        remnant.configure_cache_format("remnant")
        self.assertTrue(remnant.packed_enabled())
        remnant.configure_cache_format("native")


if __name__ == "__main__":
    unittest.main()
