"""Packed Remnant HiCache L2 host-pool round-trip coverage.

Author: Winston Cai.
"""

import unittest

import torch

from sglang.srt import remnant
from sglang.srt.mem_cache.deepseek_v4_memory_pool import (
    DeepSeekV4TokenToKVPool,
    RemnantPackedKVPool,
)
from sglang.srt.mem_cache.hybrid_cache.hybrid_pool_assembler import (
    RemnantPackedHostPool,
)
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase


register_cuda_ci(est_time=15, stage="base-b", runner_config="1-gpu-large")


@unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
class TestRemnantHiCache(CustomTestCase):
    slot_page_size = 256
    c4_rows_per_page = slot_page_size // 4

    class DevicePool:
        layer_num = 2

        def __init__(self, device):
            num_pages = 4
            self.packed_values = [
                torch.zeros(
                    num_pages,
                    TestRemnantHiCache.c4_rows_per_page,
                    256,
                    dtype=torch.uint8,
                    device=device,
                )
                for _ in range(self.layer_num)
            ]
            self.packed_bitmaps = [
                torch.zeros(
                    num_pages,
                    TestRemnantHiCache.c4_rows_per_page,
                    8,
                    dtype=torch.uint64,
                    device=device,
                )
                for _ in range(self.layer_num)
            ]
            self.packed_scales = [
                torch.zeros(
                    num_pages,
                    TestRemnantHiCache.c4_rows_per_page,
                    8,
                    dtype=torch.uint8,
                    device=device,
                )
                for _ in range(self.layer_num)
            ]

        def get_packed_buffers(self, layer):
            return (
                self.packed_values[layer],
                self.packed_bitmaps[layer],
                self.packed_scales[layer],
            )

    def setUp(self):
        remnant.configure_cache_format("remnant")
        self.addCleanup(remnant.configure_cache_format, "native")
        self.device = torch.device("cuda")
        self.device_pool = self.DevicePool(self.device)

    def _make_host_pool(self, num_host_pages=2):
        return RemnantPackedHostPool(
            pool_name="test-c4",
            device_pool=self.device_pool,
            num_host_pages=num_host_pages,
            slot_page_size=self.slot_page_size,
        )

    def _fill_page(self, layer, page, tag):
        values, bitmaps, scales = self.device_pool.get_packed_buffers(layer)
        row_values = (
            torch.arange(
                self.c4_rows_per_page * 256,
                dtype=torch.int32,
                device=self.device,
            )
            .reshape(self.c4_rows_per_page, 256)
            .add(tag * 7)
            .remainder(251)
            .to(torch.uint8)
        )
        values[page].copy_(row_values)
        bitmaps[page].fill_(0x1111111111111111 * tag)
        scales[page].fill_(tag * 13)
        return values[page].clone(), bitmaps[page].clone(), scales[page].clone()

    def test_backup_restore_preserves_all_planes_across_layers_and_pages(self):
        host_pool = self._make_host_pool()
        source_pages = (1, 3)
        destination_pages = (2, 0)
        expected = {}
        for layer in range(self.device_pool.layer_num):
            for page_index, page in enumerate(source_pages):
                expected[layer, page_index] = self._fill_page(
                    layer, page, tag=1 + layer * 2 + page_index
                )

        host_indices = torch.arange(2 * self.slot_page_size, dtype=torch.int32)
        device_indices = torch.cat(
            [
                torch.arange(
                    page * self.slot_page_size,
                    (page + 1) * self.slot_page_size,
                    dtype=torch.int32,
                    device=self.device,
                )
                for page in source_pages
            ]
        )
        host_pool.backup_from_device_all_layer(
            self.device_pool, host_indices, device_indices, io_backend="kernel"
        )
        torch.cuda.synchronize()

        for layer in range(self.device_pool.layer_num):
            for page_index, host_page in enumerate((0, 1)):
                host_planes = (
                    host_pool.host_values[layer][host_page],
                    host_pool.host_bitmaps[layer][host_page],
                    host_pool.host_scales[layer][host_page],
                )
                for actual, wanted in zip(host_planes, expected[layer, page_index]):
                    self.assertTrue(torch.equal(actual, wanted.cpu()))

        for layer in range(self.device_pool.layer_num):
            values, bitmaps, scales = self.device_pool.get_packed_buffers(layer)
            for page in source_pages:
                values[page].zero_()
                bitmaps[page].zero_()
                scales[page].zero_()

        destination_indices = torch.cat(
            [
                torch.arange(
                    page * self.slot_page_size,
                    (page + 1) * self.slot_page_size,
                    dtype=torch.int32,
                    device=self.device,
                )
                for page in destination_pages
            ]
        )
        for layer in range(self.device_pool.layer_num):
            host_pool.load_to_device_per_layer(
                self.device_pool,
                host_indices,
                destination_indices,
                layer,
                io_backend="kernel",
            )
        torch.cuda.synchronize()

        for layer in range(self.device_pool.layer_num):
            planes = self.device_pool.get_packed_buffers(layer)
            for page_index, page in enumerate(destination_pages):
                for actual, wanted in zip(
                    (buffer[page] for buffer in planes), expected[layer, page_index]
                ):
                    self.assertTrue(torch.equal(actual, wanted))

    def test_reusing_host_page_replaces_every_plane(self):
        host_pool = self._make_host_pool(num_host_pages=1)
        host_indices = torch.arange(self.slot_page_size, dtype=torch.int32)
        first_device_indices = torch.arange(
            self.slot_page_size, dtype=torch.int32, device=self.device
        )
        second_device_indices = first_device_indices + self.slot_page_size
        destination_indices = first_device_indices + 2 * self.slot_page_size

        self._fill_page(layer=0, page=0, tag=1)
        host_pool.backup_from_device_all_layer(
            self.device_pool,
            host_indices,
            first_device_indices,
            io_backend="direct",
        )
        expected = self._fill_page(layer=0, page=1, tag=2)
        host_pool.backup_from_device_all_layer(
            self.device_pool,
            host_indices,
            second_device_indices,
            io_backend="direct",
        )
        host_pool.load_to_device_per_layer(
            self.device_pool,
            host_indices,
            destination_indices,
            layer_id=0,
            io_backend="direct",
        )
        torch.cuda.synchronize()

        for actual, wanted in zip(
            (buffer[2] for buffer in self.device_pool.get_packed_buffers(0)),
            expected,
        ):
            self.assertTrue(torch.equal(actual, wanted))

    def test_partial_page_transfer_is_rejected(self):
        host_pool = self._make_host_pool()
        partial_host_indices = torch.arange(0, self.slot_page_size - 1)
        partial_device_indices = torch.arange(
            self.slot_page_size - 1, dtype=torch.int32, device=self.device
        )

        with self.assertRaisesRegex(RuntimeError, "whole-page"):
            host_pool.backup_from_device_all_layer(
                self.device_pool,
                partial_host_indices,
                partial_device_indices,
                io_backend="direct",
            )

    def test_rope_table_can_be_installed_without_a_packed_write(self):
        # HiCache can restore packed rows before any compressor write runs.
        # Exercise the same pool accessors used by that cache-hit path without
        # constructing or loading a model.
        packed_pool = object.__new__(RemnantPackedKVPool)
        packed_pool.start_layer = 0
        packed_pool._remnant_rope_freqs = [None]
        token_pool = object.__new__(DeepSeekV4TokenToKVPool)
        token_pool.layer_mapping = [(4, 0, packed_pool)]
        freqs_cis = torch.arange(16, dtype=torch.float32).to(torch.complex64)

        token_pool.set_packed_rope_freqs(0, freqs_cis)

        self.assertIs(token_pool.get_packed_freqs(0), freqs_cis)


if __name__ == "__main__":
    unittest.main()
