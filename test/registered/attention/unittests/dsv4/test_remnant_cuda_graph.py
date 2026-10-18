"""CUDA graph replay coverage for startup-configured Packed reconstruction.

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
from sglang.srt.remnant.reference import (
    pack_rows_ref,
    topmag_keep_mask,
    unpack_rows_ref,
)
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
        for implementation, keep_count in [("kernel", 256)] + [
            ("triton", k) for k in (64, 128, 192, 256, 320, 384, 448)
        ]:
            with self.subTest(implementation=implementation, keep_count=keep_count):
                self._check_store_and_reconstruction_graph(implementation, keep_count)

    def _check_store_and_reconstruction_graph(self, implementation, keep_count):
        remnant.configure_cache_format("remnant", implementation, keep_count=keep_count)
        latent = torch.randn((4, 512), device="cuda", dtype=torch.float32)
        weight = torch.ones(512, device="cuda", dtype=torch.float32)
        words = torch.zeros((4, 4), device="cuda", dtype=torch.int32)
        words[:, 0] = 4
        plan = CompressorDecodePlan(4, words.view(torch.uint8))
        locations = torch.arange(4, device="cuda", dtype=torch.int32)
        buffers = (
            torch.zeros((4, keep_count), dtype=torch.uint8, device="cuda"),
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
                    topmag_keep_mask(latent, keep_count / 512),
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
                source,
                topmag_keep_mask(source, keep_count / 512),
                weight.cpu(),
                1e-6,
                keep_count=keep_count,
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

    def test_fused_read_replays_changed_planes_indices_lengths_and_frequencies(self):
        for count in (64, 128, 192, 256, 320, 384, 448):
            with self.subTest(keep_count=count):
                remnant.configure_cache_format("remnant", "triton", keep_count=count)
                torch.manual_seed(101 + count)
                source = torch.randn(4, 512)
                weight = torch.linspace(0.5, 1.5, 512)
                planes = pack_rows_ref(
                    source,
                    topmag_keep_mask(source, count / 512),
                    weight,
                    1e-6,
                    keep_count=count,
                )
                buffers = tuple(x.cuda() for x in planes)
                physical = torch.tensor(
                    [[0, 2, -1, 3], [3, 3, 1, 0]], device="cuda", dtype=torch.int32
                )
                raw = torch.tensor(
                    [[0, 2, 1, 3], [2, 0, 1, 4]], device="cuda", dtype=torch.int32
                )
                lengths = torch.tensor([4, 3], device="cuda", dtype=torch.int32)
                angles = torch.arange(32 * 32).float().reshape(32, 32) * 0.0037
                freqs = torch.polar(torch.ones_like(angles), angles).cuda()
                output = torch.empty((8, 512), device="cuda", dtype=torch.bfloat16)
                split = torch.empty_like(output)
                owned = (*buffers, physical, raw, lengths, freqs, output)
                addresses = [x.data_ptr() for x in owned]

                def call():
                    unpack_gather_bf16(
                        buffers,
                        physical,
                        raw,
                        lengths,
                        freqs,
                        output,
                        fused_rope=True,
                    )

                call()
                torch.cuda.synchronize()
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    call()
                for iteration in range(2):
                    source.normal_()
                    planes = pack_rows_ref(
                        source,
                        topmag_keep_mask(source, count / 512),
                        weight,
                        1e-6,
                        keep_count=count,
                    )
                    for target, values in zip(buffers, planes):
                        target.copy_(values)
                    physical.copy_(physical.roll(1, dims=1))
                    raw.copy_(raw.roll(1, dims=1))
                    raw[0, 1] = -1
                    lengths.copy_(torch.tensor([4 - iteration, 2 + iteration]))
                    phase = angles * (iteration + 2)
                    freq_cpu = torch.polar(torch.ones_like(phase), phase)
                    freqs.copy_(freq_cpu)
                    graph.replay()
                    unpack_gather_bf16(buffers, physical, raw, lengths, freqs, split)
                    self.assertTrue(
                        torch.equal(output.view(torch.int16), split.view(torch.int16))
                    )
                    decoded = unpack_rows_ref(*planes)
                    expected = torch.zeros((8, 512), dtype=torch.bfloat16)
                    p_cpu, r_cpu, lens = physical.cpu(), raw.cpu(), lengths.cpu()
                    for row in range(8):
                        query, k = divmod(row, 4)
                        p, r = int(p_cpu[query, k]), int(r_cpu[query, k])
                        if k < lens[query] and 0 <= p < 4 and 0 <= r * 4 < 32:
                            expected[row] = decoded[p]
                            tail = decoded[p, 448:].float().reshape(32, 2)
                            rotated = (
                                torch.view_as_complex(tail.contiguous())
                                * freq_cpu[r * 4]
                            )
                            expected[row, 448:] = (
                                torch.view_as_real(rotated).flatten().to(torch.bfloat16)
                            )
                    torch.testing.assert_close(
                        output.cpu(), expected, atol=2e-2, rtol=2e-2
                    )
                    self.assertEqual([x.data_ptr() for x in owned], addresses)

    def test_consecutive_mixed_layers_replay_with_stable_buffers(self):
        counts = (128, 192, 256, 320)
        remnant.configure_cache_format(
            "remnant",
            "triton",
            keep_count=128,
            layer_keep_counts='{"1":192,"2":256,"3":320}',
            compression_ratios=(4,) * 4,
        )
        latents = [torch.randn(4, 512, device="cuda") for _ in counts]
        weight = torch.ones(512, device="cuda")
        words = torch.zeros((4, 4), dtype=torch.int32, device="cuda")
        words[:, 0] = 4
        plan = CompressorDecodePlan(4, words.view(torch.uint8))
        locations = torch.arange(4, dtype=torch.int32, device="cuda")
        physical = torch.arange(4, dtype=torch.int32, device="cuda").view(1, 4)
        raw = torch.zeros_like(physical)
        lengths = torch.tensor([4], dtype=torch.int32, device="cuda")
        freqs = torch.ones((16, 32), dtype=torch.complex64, device="cuda")
        buffers = [
            (
                torch.zeros((4, k), dtype=torch.uint8, device="cuda"),
                torch.zeros((4, 8), dtype=torch.uint64, device="cuda"),
                torch.zeros((4, 8), dtype=torch.uint8, device="cuda"),
            )
            for k in counts
        ]
        outputs = [
            torch.empty((4, 512), dtype=torch.bfloat16, device="cuda") for _ in counts
        ]
        addresses = [tensor.data_ptr() for group in buffers for tensor in group]

        def pipeline():
            for x, k, planes, output in zip(latents, counts, buffers, outputs):
                pack_rows(
                    x,
                    topmag_keep_mask(x, k / 512),
                    weight,
                    1e-6,
                    plan,
                    locations,
                    planes,
                )
                unpack_gather_bf16(planes, physical, raw, lengths, freqs, output)

        pipeline()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            pipeline()
        for iteration in range(2):
            for x in latents:
                x.normal_()
            locations.copy_(locations.roll(1))
            physical.copy_(physical.roll(1, dims=1))
            lengths.fill_(4 - iteration)
            graph.replay()
            for x, k, output in zip(latents, counts, outputs):
                source = x.cpu()
                planes = pack_rows_ref(
                    source,
                    topmag_keep_mask(source, k / 512),
                    weight.cpu(),
                    1e-6,
                    keep_count=k,
                )
                dense = unpack_rows_ref(*planes)
                stored = torch.empty_like(dense)
                stored[locations.cpu().long()] = dense
                expected = stored[physical.cpu().long().flatten()]
                expected[4 - iteration :] = 0
                self.assertTrue(torch.equal(output.cpu(), expected))
            self.assertEqual(
                addresses, [t.data_ptr() for group in buffers for t in group]
            )


if __name__ == "__main__":
    unittest.main()
