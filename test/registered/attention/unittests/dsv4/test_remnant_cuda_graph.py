"""CUDA graph replay coverage for the fixed packed reconstruction path.

Author: Winston Cai.
"""

import importlib.util
import unittest

import torch

from sglang.srt import remnant
from sglang.srt.remnant.packed import unpack_gather_bf16
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase


register_cuda_ci(est_time=10, stage="base-b", runner_config="1-gpu-large")


@unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
@unittest.skipUnless(
    importlib.util.find_spec("triton") is not None, "Triton is required"
)
class TestRemnantCudaGraph(CustomTestCase):
    def test_selected_row_reconstruction_replays(self):
        device = torch.device("cuda")
        remnant.configure_cache_format("remnant")
        values = torch.randint(0, 255, (1, 256, 256), dtype=torch.uint8, device=device)
        bitmaps = torch.zeros((1, 256, 8), dtype=torch.uint64, device=device)
        all_bits = torch.tensor(-1, dtype=torch.int64, device=device).view(torch.uint64)
        bitmaps[:, :, :4] = all_bits
        scales = torch.full((1, 256, 8), 127, dtype=torch.uint8, device=device)
        physical = torch.zeros((1, 1), dtype=torch.int32, device=device)
        raw = torch.zeros((1, 1), dtype=torch.int32, device=device)
        lengths = torch.ones(1, dtype=torch.int32, device=device)
        freqs = torch.ones(512, dtype=torch.complex64, device=device)
        output = torch.empty((1, 512), dtype=torch.bfloat16, device=device)

        unpack_gather_bf16(
            (values, bitmaps, scales), physical, raw, lengths, freqs, output
        )
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            unpack_gather_bf16(
                (values, bitmaps, scales), physical, raw, lengths, freqs, output
            )
        first = output.clone()
        graph.replay()
        torch.cuda.synchronize()
        self.assertTrue(torch.equal(output, first))
        remnant.configure_cache_format("native")


if __name__ == "__main__":
    unittest.main()
