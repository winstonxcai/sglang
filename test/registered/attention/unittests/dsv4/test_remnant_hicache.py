"""Packed HiCache host mirror round-trip coverage."""

import unittest

import torch

from sglang.srt import remnant
from sglang.srt.mem_cache.hybrid_cache.hybrid_pool_assembler import (
    RemnantPackedHostPool,
)
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase


register_cuda_ci(est_time=10, stage="base-b", runner_config="1-gpu-large")


@unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
class TestRemnantHiCache(CustomTestCase):
    def test_backup_and_restore_preserve_all_fragments(self):
        remnant.configure_cache_format("remnant")
        device = torch.device("cuda")

        class DevicePool:
            layer_num = 1

            def __init__(self):
                self.packed_values = [
                    torch.zeros((2, 256, 256), dtype=torch.uint8, device=device)
                ]
                self.packed_bitmaps = [
                    torch.zeros((2, 256, 8), dtype=torch.uint64, device=device)
                ]
                self.packed_scales = [
                    torch.zeros((2, 256, 8), dtype=torch.uint8, device=device)
                ]

            def get_packed_buffers(self, layer):
                return (
                    self.packed_values[layer],
                    self.packed_bitmaps[layer],
                    self.packed_scales[layer],
                )

        device_pool = DevicePool()
        host_pool = RemnantPackedHostPool(
            pool_name="test-c4",
            device_pool=device_pool,
            num_host_pages=1,
            slot_page_size=256,
        )
        expected = (
            torch.arange(256 * 256, device=device, dtype=torch.int32)
            .remainder(251)
            .to(torch.uint8)
            .reshape(256, 256)
        )
        device_pool.packed_values[0][1].copy_(expected)
        device_pool.packed_bitmaps[0][1].fill_(0xAAAAAAAAAAAAAAAA)
        device_pool.packed_scales[0][1].fill_(127)
        host_indices = torch.arange(256, dtype=torch.int32)
        device_indices = torch.arange(256, dtype=torch.int32, device=device) + 256

        host_pool.backup_from_device_all_layer(
            device_pool, host_indices, device_indices, "direct"
        )
        torch.cuda.synchronize()
        self.assertTrue(torch.equal(host_pool.host_values[0][0], expected.cpu()))
        self.assertTrue(torch.all(host_pool.host_bitmaps[0][0] == 0xAAAAAAAAAAAAAAAA))
        self.assertTrue(torch.all(host_pool.host_scales[0][0] == 127))
        device_pool.packed_values[0][0].zero_()
        device_pool.packed_bitmaps[0][0].zero_()
        device_pool.packed_scales[0][0].zero_()
        host_pool.load_to_device_per_layer(
            device_pool, host_indices, torch.arange(256, device=device), 0, "direct"
        )
        torch.cuda.synchronize()

        self.assertTrue(torch.equal(device_pool.packed_values[0][0], expected))
        self.assertTrue(
            torch.equal(
                device_pool.packed_bitmaps[0][0],
                torch.full((256, 8), 0xAAAAAAAAAAAAAAAA, dtype=torch.uint64, device=device),
            )
        )
        self.assertTrue(torch.all(device_pool.packed_scales[0][0] == 127))
        remnant.configure_cache_format("native")


if __name__ == "__main__":
    unittest.main()
