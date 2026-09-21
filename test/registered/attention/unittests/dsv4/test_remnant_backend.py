"""Backend metadata and selected-row reconstruction contracts.

Author: Winston Cai.
"""

import unittest
from unittest import mock

import torch

from sglang.srt import remnant
from sglang.srt.layers.attention.deepseek_v4_backend import DSV4AttnMetadata
from sglang.srt.remnant import NativeWorkspace
from sglang.srt.remnant.packed import unpack_gather_native
from sglang.test.test_utils import CustomTestCase


class TestRemnantBackend(CustomTestCase):
    def test_packed_metadata_retains_raw_indices(self):
        metadata = DSV4AttnMetadata(
            page_size=256,
            page_table=torch.zeros((1, 1), dtype=torch.int32),
            raw_out_loc=torch.zeros(1, dtype=torch.int32),
            cuda_int32_kwargs={},
            seq_lens_casual=torch.ones(1, dtype=torch.int32),
            positions_casual=torch.zeros(1, dtype=torch.int32),
            swa_page_indices=torch.zeros((1, 1), dtype=torch.int32),
            swa_topk_lengths=torch.ones(1, dtype=torch.int32),
            c4_sparse_topk=512,
            c4_topk_lengths_clamp1=torch.ones(1, dtype=torch.int32),
            c128_page_indices=torch.zeros((1, 1), dtype=torch.int32),
            c128_topk_lengths_clamp1=torch.ones(1, dtype=torch.int32),
        )
        with mock.patch(
            "sglang.srt.layers.attention.deepseek_v4_backend._create_flashmla_metadata",
            return_value=None,
        ):
            remnant.configure_cache_format("native")
            metadata.init_flashmla_related()
            self.assertIsNone(metadata.c4_sparse_raw_indices)

            remnant.configure_cache_format("remnant")
            metadata.init_flashmla_related()
            self.assertEqual(metadata.c4_sparse_raw_indices.shape, (1, 512))
        remnant.configure_cache_format("native")

    def test_native_reconstruction_rejects_workspace_overflow(self):
        remnant.configure_cache_format("remnant")
        buffers = tuple(
            torch.empty(shape, dtype=dtype)
            for shape, dtype in (
                ((1, 256, 256), torch.uint8),
                ((1, 256, 8), torch.uint64),
                ((1, 256, 8), torch.uint8),
            )
        )
        workspace = NativeWorkspace.allocate(1, 1, 64, "cpu")
        with self.assertRaisesRegex(ValueError, "top-k capacity"):
            unpack_gather_native(
                buffers,
                torch.zeros((1, 2), dtype=torch.int32),
                torch.zeros((1, 2), dtype=torch.int32),
                torch.ones(1, dtype=torch.int32),
                torch.ones(256, dtype=torch.complex64),
                workspace,
            )
        remnant.configure_cache_format("native")


if __name__ == "__main__":
    unittest.main()
