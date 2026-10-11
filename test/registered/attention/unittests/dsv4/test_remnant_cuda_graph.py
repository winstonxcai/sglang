"""CUDA graph replay coverage for the fixed packed reconstruction path.

Author: Winston Cai.
"""

import importlib.util
import unittest

import torch

from sglang.kernels.ops.attention.dsv4.compress import CompressorDecodePlan
from sglang.srt import remnant
from sglang.srt.remnant.packed import (
    NativeWorkspace,
    pack_rows,
    pack_rows_fused,
    unpack_gather_bf16,
    unpack_gather_native,
)
from sglang.srt.remnant.reference import pack_rows_ref, topmag_keep_mask
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=10, stage="base-b", runner_config="1-gpu-large")


@unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
@unittest.skipUnless(
    importlib.util.find_spec("triton") is not None, "Triton is required"
)
class TestRemnantCudaGraph(CustomTestCase):
    def tearDown(self):
        remnant.configure_cache_format("native")

    def test_store_and_native_reconstruction_replays_changed_inputs(self):
        for implementation in ("triton", "kernel"):
            with self.subTest(implementation=implementation):
                self._check_store_and_reconstruction_graph(implementation)

    def _check_store_and_reconstruction_graph(self, implementation):
        remnant.configure_cache_format("remnant", implementation)
        latent = torch.randn((4, 512), device="cuda", dtype=torch.float32)
        weight = torch.ones(512, device="cuda", dtype=torch.float32)
        words = torch.zeros((4, 4), device="cuda", dtype=torch.int32)
        words[:, 0] = 4
        plan = CompressorDecodePlan(4, words.view(torch.uint8))
        locations = torch.arange(4, device="cuda", dtype=torch.int32)
        buffers = (
            torch.zeros((4, 256), dtype=torch.uint8, device="cuda"),
            torch.zeros((4, 8), dtype=torch.uint64, device="cuda"),
            torch.zeros((4, 8), dtype=torch.uint8, device="cuda"),
        )
        reference_buffers = tuple(torch.zeros_like(t) for t in buffers)
        physical = torch.tensor(
            [[0, 1, 2, 3], [3, 2, 1, 0]], device="cuda", dtype=torch.int32
        )
        raw = physical.clone()
        lengths = torch.tensor([4, 3], device="cuda", dtype=torch.int32)
        angles = torch.arange(16 * 32, device="cuda").float().reshape(16, 32) * 0.0017
        freqs = torch.polar(torch.ones_like(angles), angles)
        actual = NativeWorkspace.allocate(2, 8, 4, "cuda")
        expected = NativeWorkspace.allocate(2, 8, 4, "cuda")

        def pipeline():
            if implementation == "triton":
                pack_rows(
                    latent,
                    topmag_keep_mask(latent, 0.5),
                    weight,
                    1e-6,
                    plan,
                    locations,
                    buffers,
                )
            else:
                pack_rows_fused(latent, weight, 1e-6, plan, locations, buffers)
            unpack_gather_native(buffers, physical, raw, lengths, freqs, actual)

        pipeline()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            pipeline()
        for iteration in range(2):
            latent.normal_()
            locations.copy_(
                torch.arange(4, device="cuda", dtype=torch.int32).roll(iteration + 1)
            )
            physical.copy_(physical.roll(1, dims=1))
            raw.copy_(raw.roll(2, dims=1))
            lengths.copy_(
                torch.tensor(
                    [iteration + 1, 4 - iteration], device="cuda", dtype=torch.int32
                )
            )
            graph.replay()
            # Build Packed records with the independent CPU numerical reference
            # outside capture, then compare reconstruction with these live inputs.
            source = latent.cpu()
            packed = pack_rows_ref(
                source, topmag_keep_mask(source, 0.5), weight.cpu(), 1e-6
            )
            for destination, reference in zip(reference_buffers, packed):
                destination.view(torch.uint8)[locations.long()] = reference.view(
                    torch.uint8
                ).to("cuda")
            unpack_gather_native(
                reference_buffers, physical, raw, lengths, freqs, expected
            )
            torch.cuda.synchronize()
            self.assertTrue(torch.equal(actual.native_bytes, expected.native_bytes))

    def test_selected_row_reconstruction_replays(self):
        device = torch.device("cuda")
        remnant.configure_cache_format("remnant")
        # Arbitrary byte codes can encode FP8 NaNs, breaking exact replay equality.
        values = (
            torch.randn((1, 256, 256), device=device)
            .to(torch.float8_e4m3fn)
            .view(torch.uint8)
        )
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
