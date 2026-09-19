"""Parity between the native adapter and direct Remnant FlashMLA decode."""

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
    table = torch.zeros((1, max_position * 128 + 32, 2), device=device)
    table[..., 0] = 1.0
    table[0, : max_position * 128].view(max_position, 128, 2)[:, :32, 0] = torch.cos(angle)
    table[0, : max_position * 128].view(max_position, 128, 2)[:, :32, 1] = torch.sin(angle)
    return table.view(torch.complex64).reshape(-1).contiguous()


def _fill_swa(cache: torch.Tensor) -> None:
    rows = cache.shape[0] * cache.shape[1]
    flat = cache.reshape(rows, 584)
    flat[:, :448].copy_(
        (torch.arange(rows * 448, device=cache.device) % 113 + 1)
        .to(torch.uint8)
        .view(rows, 448)
    )
    tail = torch.linspace(-0.25, 0.25, rows * 64, device=cache.device)
    flat[:, 448:576].copy_(
        tail.to(torch.bfloat16).view(torch.uint8).reshape(rows, 128)
    )
    flat[:, 576:584].fill_(127)


class TestRemnantFlashMLADirect:
    @pytest.mark.parametrize("num_heads", [64, 128])
    @pytest.mark.parametrize("batch", [8, 16])
    @pytest.mark.parametrize("topk_length", [512, 317])
    def test_direct_matches_native_adapter(
        self, num_heads: int, batch: int, topk_length: int
    ):
        from sgl_kernel import flash_mla

        device = torch.device("cuda")
        rows = batch * 64
        page_size = 64
        selected_k = 512
        remnant.configure_cache_format("remnant")

        latent = torch.randn((rows, 512), device=device, dtype=torch.float32)
        keep_mask = topmag_keep_mask(latent, 0.5)
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
            batch, selected_k, page_size, device, with_dense=True
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
        swa_cache = torch.zeros((batch, page_size, 1, 584), dtype=torch.uint8, device=device)
        _fill_swa(swa_cache)
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
            remnant_freqs=torch.view_as_real(freqs).unsqueeze(0),
        )

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
        swa_cache = torch.zeros((batch, 64, 1, 584), dtype=torch.uint8, device=device)
        _fill_swa(swa_cache)
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
                remnant_freqs=torch.view_as_real(freqs).unsqueeze(0),
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
        q.normal_()
        buffers[2].add_(1)
        physical.copy_(physical.roll(1, dims=-1))
        raw.copy_(raw.roll(1, dims=-1))
        lengths.sub_(13)
        graph.replay()
        torch.cuda.synchronize()
        second = tuple(x.clone() for x in captured)
        eager = run(meta)
        torch.testing.assert_close(second[0], eager[0], atol=2.0e-2, rtol=2.0e-2)
        torch.testing.assert_close(second[1], eager[1], atol=2.0e-2, rtol=2.0e-2)
        assert not torch.equal(first[0], second[0])


if __name__ == "__main__":
    pytest.main([__file__])
