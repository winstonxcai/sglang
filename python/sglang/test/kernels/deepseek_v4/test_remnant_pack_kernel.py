"""Triton pack and unpack correctness for the Remnant C4 record."""

import importlib.util

import pytest
import torch

from sglang.kernels.ops.attention.dsv4 import CompressorPrefillPlan
from sglang.srt import remnant
from sglang.srt.remnant.packed import pack_rows
from sglang.srt.remnant.reference import pack_rows_ref, topmag_keep_mask


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or importlib.util.find_spec("triton") is None,
    reason="CUDA and Triton are required",
)


def _prefill_plan(rows: int, device: torch.device) -> CompressorPrefillPlan:
    plan_c = torch.zeros((rows, 16), dtype=torch.uint8, device=device)
    plan_w = torch.zeros((rows, 8), dtype=torch.uint8, device=device)
    plan_w[:, :4] = torch.arange(rows, dtype=torch.int32, device=device).view(
        torch.uint8
    ).reshape(rows, 4)
    plan_w[:, 4:8] = torch.tensor([0, 0, 0, 0], dtype=torch.uint8, device=device)
    return CompressorPrefillPlan(4, plan_c, plan_w)


def test_remnant_triton_pack_matches_reference():
    device = torch.device("cuda")
    remnant.configure_cache_format("remnant")
    torch.manual_seed(11)
    latent = torch.randn((2, 512), device=device, dtype=torch.float32)
    mask = topmag_keep_mask(latent, 0.5)
    norm_weight = torch.ones(512, device=device, dtype=torch.float32)
    locations = torch.arange(2, device=device, dtype=torch.int32)
    buffers = (
        torch.empty((2, 256, 256), dtype=torch.uint8, device=device),
        torch.empty((2, 256, 8), dtype=torch.uint64, device=device),
        torch.empty((2, 256, 8), dtype=torch.uint8, device=device),
    )

    pack_rows(
        latent,
        mask,
        norm_weight,
        1e-6,
        _prefill_plan(2, device),
        locations,
        buffers,
    )
    expected = pack_rows_ref(latent, mask, norm_weight, 1e-6)
    selected_rows = locations.cpu().tolist()
    packed_values = torch.stack(
        [buffers[0].reshape(-1, 256)[i] for i in selected_rows]
    )
    packed_bitmaps = torch.stack(
        [buffers[1].reshape(-1, 8)[i] for i in selected_rows]
    )
    packed_scales = torch.stack(
        [buffers[2].reshape(-1, 8)[i] for i in selected_rows]
    )
    assert torch.equal(packed_values, expected[0])
    assert torch.equal(packed_bitmaps, expected[1])
    assert torch.equal(packed_scales, expected[2])
    remnant.configure_cache_format("native")
