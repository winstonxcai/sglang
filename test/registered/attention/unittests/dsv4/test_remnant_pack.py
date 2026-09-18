"""Independent reference tests for the persistent packed row format."""

import unittest

import torch

from sglang.srt import remnant
from sglang.srt.remnant.reference import (
    pack_rows_ref,
    topmag_keep_mask,
    unpack_rows_ref,
)
from sglang.test.test_utils import CustomTestCase


class TestRemnantPack(CustomTestCase):
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
