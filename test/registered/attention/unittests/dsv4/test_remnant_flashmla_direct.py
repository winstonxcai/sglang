"""Parity between the fused adapter and direct Remnant FlashMLA decode.

Author: Winston Cai.
"""

import importlib.util

import pytest
import torch

from sglang.srt import remnant
from sglang.srt.remnant import NativeWorkspace
from sglang.srt.remnant.packed import pack_rows, unpack_gather_native
from sglang.srt.remnant.reference import topmag_keep_mask


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available()
    or importlib.util.find_spec("triton") is None,
    reason="CUDA, Triton, and the compiled FlashMLA extension are required",
)


@pytest.fixture(autouse=True)
def _reset_cache_format():
    yield
    remnant.configure_cache_format("native")


class _PackPlan:
    is_decode = False

    def __init__(self, rows: int, device: torch.device):
        self.plan_w = torch.zeros((rows, 8), dtype=torch.uint8, device=device)
        ids = torch.arange(rows, dtype=torch.int32, device=device)
        self.plan_w[:, :4] = ids.view(torch.uint8).reshape(rows, 4)

    def __getitem__(self, index: int) -> torch.Tensor:
        if index == 2:
            return self.plan_w
        raise IndexError(index)


def _frequencies(max_position: int, device: torch.device) -> torch.Tensor:
    pair = torch.arange(32, device=device, dtype=torch.float32)
    position = torch.arange(max_position, device=device, dtype=torch.float32)[:, None]
    angle = (position + 1.0) * (pair + 1.0) * 0.0017
    # The kernel uses four 32-pair rows per raw token position and addresses
    # the first row with raw * 4 * 32.  Keep the same layout as the model's
    # table so the adapter sees [positions * 4, 32, 2] and the direct path can
    # consume the identical contiguous storage.
    table = torch.zeros((max_position * 4, 32, 2), device=device)
    table[::4, :, 0] = torch.cos(angle)
    table[::4, :, 1] = torch.sin(angle)
    return torch.view_as_complex(table).contiguous()


def _bitmap_edge_keep_mask(rows: int, device: torch.device) -> torch.Tensor:
    """Balanced masks for empty/full nibbles, alternating bits, and rank 255."""
    coordinate = torch.arange(512, device=device)
    within_byte = coordinate.remainder(8)
    patterns = torch.stack(
        (
            within_byte < 4,
            within_byte >= 4,
            coordinate.remainder(2) == 0,
            coordinate.remainder(2) == 1,
            (within_byte == 0) | (within_byte == 1) | (within_byte == 6) | (within_byte == 7),
            (within_byte == 0) | (within_byte == 2) | (within_byte == 5) | (within_byte == 7),
            (coordinate < 255) | (coordinate == 511),
        )
    )
    assert torch.all(patterns.sum(dim=-1) == 256), "Each bitmap pattern must retain 256 values"
    row_pattern = torch.arange(rows, device=device) % patterns.shape[0]
    return patterns.index_select(0, row_pattern)


def _make_swa_cache(pages: int, page_size: int, device: torch.device) -> torch.Tensor:
    bytes_per_page = ((page_size * 584 + 575) // 576) * 576
    storage = torch.zeros((pages, bytes_per_page), dtype=torch.uint8, device=device)
    values = storage[:, : page_size * 576].view(pages, page_size, 576)
    rows = pages * page_size
    values[:, :, :448].copy_(
        (torch.arange(rows * 448, device=device) % 113 + 1)
        .to(torch.uint8)
        .view(pages, page_size, 448)
    )
    tail = torch.linspace(-0.25, 0.25, rows * 64, device=device)
    values[:, :, 448:576].copy_(
        tail.to(torch.bfloat16).view(torch.uint8).reshape(pages, page_size, 128)
    )
    storage[:, page_size * 576 : page_size * 584].fill_(127)
    return storage.as_strided(
        (pages, page_size, 1, 584), (bytes_per_page, 584, 584, 1)
    )


class TestRemnantFlashMLADirect:
    @pytest.mark.parametrize(
        "bitmap_patterns", [False, True], ids=["topmag", "bitmap-edges"]
    )
    @pytest.mark.parametrize("num_heads", [64, 128])
    @pytest.mark.parametrize("batch", [8, 16])
    @pytest.mark.parametrize("topk_length", [512, 317])
    def test_direct_matches_fused_adapter(
        self, num_heads: int, batch: int, topk_length: int, bitmap_patterns: bool
    ):
        from sgl_kernel import flash_mla

        device = torch.device("cuda")
        rows = batch * 64
        page_size = 64
        selected_k = 512
        remnant.configure_cache_format("remnant")

        latent = torch.randn((rows, 512), device=device, dtype=torch.float32)
        keep_mask = (
            _bitmap_edge_keep_mask(rows, device)
            if bitmap_patterns
            else topmag_keep_mask(latent, 0.5)
        )
        norm_weight = torch.ones(512, device=device)
        buffers = (
            torch.zeros((batch, page_size, 256), dtype=torch.uint8, device=device),
            torch.zeros((batch, page_size, 8), dtype=torch.uint64, device=device),
            torch.zeros((batch, page_size, 8), dtype=torch.uint8, device=device),
        )
        locations = torch.arange(rows, dtype=torch.int32, device=device)
        pack_rows(
            latent,
            keep_mask,
            norm_weight,
            1.0e-6,
            _PackPlan(rows, device),
            locations,
            buffers,
        )

        physical = torch.arange(selected_k, device=device, dtype=torch.int32).view(
            1, selected_k
        ) % 64
        physical = physical + torch.arange(batch, device=device, dtype=torch.int32).view(
            batch, 1
        ) * 64
        raw = physical + 3 + torch.arange(
            selected_k, device=device, dtype=torch.int32
        ).view(1, selected_k) % 11
        lengths = torch.full((batch,), topk_length, dtype=torch.int32, device=device)
        if topk_length < selected_k:
            physical = physical.clone()
            raw = raw.clone()
            lengths[1::2] = 64
            for request, length in enumerate(lengths.tolist()):
                physical[request, length:] = 0
                raw[request, length:] = -1
        freqs = _frequencies(int(raw.clamp_min(0).max().item()) + 2, device)

        workspace = NativeWorkspace.allocate(
            batch, selected_k, page_size, device
        )
        native_bytes, native_indices = unpack_gather_native(
            buffers,
            physical,
            raw,
            lengths,
            freqs,
            workspace,
        )
        native_cache = native_bytes.as_strided(
            (native_bytes.shape[0], page_size, 1, 584),
            (workspace.bytes_per_page, 584, 584, 1),
        )
        native_indices = native_indices.unsqueeze(1)
        if topk_length < selected_k:
            for request, length in enumerate(lengths.tolist()):
                physical[request, length:] = torch.iinfo(torch.int32).max

        q = torch.randn((batch, 1, num_heads, 512), device=device, dtype=torch.bfloat16)
        swa_cache = _make_swa_cache(batch, page_size, device)
        swa_indices = torch.arange(page_size, device=device, dtype=torch.int32).view(
            1, 1, page_size
        )
        swa_indices = swa_indices + torch.arange(batch, device=device, dtype=torch.int32).view(
            batch, 1, 1
        ) * page_size
        swa_indices = swa_indices.contiguous()
        swa_lengths = torch.full((batch,), page_size, dtype=torch.int32, device=device)
        sink = torch.linspace(-0.1, 0.1, num_heads, device=device)

        native_meta = flash_mla.get_mla_metadata()[0]
        native_out, native_lse = flash_mla.flash_mla_with_kvcache(
            q,
            swa_cache,
            None,
            None,
            512,
            native_meta,
            softmax_scale=512 ** -0.5,
            is_fp8_kvcache=True,
            indices=swa_indices,
            attn_sink=sink,
            extra_k_cache=native_cache,
            extra_indices_in_kvcache=native_indices,
            topk_length=swa_lengths,
            extra_topk_length=lengths,
        )

        direct_meta = flash_mla.get_mla_metadata()[0]
        direct_out, direct_lse = flash_mla.flash_mla_with_kvcache(
            q,
            swa_cache,
            None,
            None,
            512,
            direct_meta,
            softmax_scale=512 ** -0.5,
            is_fp8_kvcache=True,
            indices=swa_indices,
            attn_sink=sink,
            extra_indices_in_kvcache=physical.unsqueeze(1),
            topk_length=swa_lengths,
            extra_topk_length=lengths,
            remnant_buffers=buffers,
            remnant_raw_indices=raw.unsqueeze(1),
            remnant_freqs=torch.view_as_real(freqs),
        )

        packed_codes = buffers[0]
        assert not torch.any((packed_codes & 0x7F) == 0x7F), "Packed cache contains FP8 NaN codes"
        assert torch.isfinite(native_out).all(), "Native adapter output is non-finite"
        assert torch.isfinite(native_lse).all(), "Native adapter LSE is non-finite"
        assert torch.isfinite(direct_out).all(), "Direct output is non-finite"
        assert torch.isfinite(direct_lse).all(), "Direct LSE is non-finite"
        torch.testing.assert_close(direct_out, native_out, atol=2.0e-2, rtol=2.0e-2)
        torch.testing.assert_close(direct_lse, native_lse, atol=2.0e-2, rtol=2.0e-2)

    @pytest.mark.parametrize("num_heads", [64, 128])
    @pytest.mark.parametrize("batch", [8, 16])
    def test_direct_decode_cuda_graph_replay(self, num_heads: int, batch: int):
        from sgl_kernel import flash_mla

        device = torch.device("cuda")
        rows = batch * 64
        remnant.configure_cache_format("remnant")
        latent = torch.randn((rows, 512), device=device)
        keep_mask = topmag_keep_mask(latent, 0.5)
        buffers = (
            torch.zeros((batch, 64, 256), dtype=torch.uint8, device=device),
            torch.zeros((batch, 64, 8), dtype=torch.uint64, device=device),
            torch.zeros((batch, 64, 8), dtype=torch.uint8, device=device),
        )
        pack_rows(
            latent,
            keep_mask,
            torch.ones(512, device=device),
            1.0e-6,
            _PackPlan(rows, device),
            torch.arange(rows, dtype=torch.int32, device=device),
            buffers,
        )
        q = torch.randn((batch, 1, num_heads, 512), device=device, dtype=torch.bfloat16)
        swa_cache = _make_swa_cache(batch, 64, device)
        swa_indices = torch.arange(64, device=device, dtype=torch.int32).view(1, 1, 64)
        swa_indices = swa_indices + torch.arange(batch, device=device, dtype=torch.int32).view(
            batch, 1, 1
        ) * 64
        swa_indices = swa_indices.contiguous()
        swa_lengths = torch.full((batch,), 64, dtype=torch.int32, device=device)
        physical = torch.arange(512, device=device, dtype=torch.int32).view(1, 512) % 64
        physical = physical + torch.arange(batch, device=device, dtype=torch.int32).view(batch, 1) * 64
        physical = physical.contiguous()
        raw = physical + 5 + torch.arange(
            512, device=device, dtype=torch.int32
        ).view(1, 512) % 11
        lengths = torch.full((batch,), 512, dtype=torch.int32, device=device)
        freqs = _frequencies(int(raw.max().item()) + 2, device)

        def run(meta):
            return flash_mla.flash_mla_with_kvcache(
                q,
                swa_cache,
                None,
                None,
                512,
                meta,
                softmax_scale=512 ** -0.5,
                is_fp8_kvcache=True,
                indices=swa_indices,
                topk_length=swa_lengths,
                extra_topk_length=lengths,
                extra_indices_in_kvcache=physical.unsqueeze(1),
                remnant_buffers=buffers,
                remnant_raw_indices=raw.unsqueeze(1),
                remnant_freqs=torch.view_as_real(freqs),
            )

        meta = flash_mla.get_mla_metadata()[0]
        for _ in range(3):
            run(meta)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            captured = run(meta)
        graph.replay()
        torch.cuda.synchronize()
        first = tuple(x.clone() for x in captured)
        assert torch.isfinite(first[0]).all(), "Captured direct output is non-finite"
        assert torch.isfinite(first[1]).all(), "Captured direct LSE is non-finite"
        q.normal_()
        buffers[2].add_(1)
        physical.copy_(physical.roll(1, dims=-1))
        raw.copy_(raw.roll(1, dims=-1))
        lengths.sub_(13)
        graph.replay()
        torch.cuda.synchronize()
        second = tuple(x.clone() for x in captured)
        eager = run(meta)
        assert torch.isfinite(second[0]).all(), "Replayed direct output is non-finite"
        assert torch.isfinite(eager[0]).all(), "Eager direct output is non-finite"
        torch.testing.assert_close(second[0], eager[0], atol=2.0e-2, rtol=2.0e-2)
        torch.testing.assert_close(second[1], eager[1], atol=2.0e-2, rtol=2.0e-2)
        assert not torch.equal(first[0], second[0])


if __name__ == "__main__":
    pytest.main([__file__])
