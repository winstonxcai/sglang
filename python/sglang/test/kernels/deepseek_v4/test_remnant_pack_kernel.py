"""Triton pack and unpack correctness for the Remnant C4 record.

Author: Winston Cai.
"""

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
    words = plan_c.view(torch.int32)
    words[:, 0] = 4
    words[:, 1] = torch.arange(rows, dtype=torch.int32, device=device)
    plan_w = torch.zeros((rows, 8), dtype=torch.uint8, device=device)
    plan_w[:, :4] = (
        torch.arange(rows, dtype=torch.int32, device=device)
        .view(torch.uint8)
        .reshape(rows, 4)
    )
    plan_w[:, 4:8] = torch.tensor([0, 0, 0, 0], dtype=torch.uint8, device=device)
    return CompressorPrefillPlan(4, plan_c, plan_w)


@pytest.mark.parametrize("keep_count", [64, 128, 192, 256, 320, 384, 448])
def test_remnant_triton_pack_matches_reference(keep_count):
    device = torch.device("cuda")
    remnant.configure_cache_format("remnant", "triton", keep_count=keep_count)
    torch.manual_seed(11)
    latent = torch.randn((7, 512), device=device, dtype=torch.float32)
    latent[0].zero_()
    latent[1].fill_(1)  # cutoff ties
    latent[2] = 1 + torch.arange(512, device=device).float() / 65536
    mask = topmag_keep_mask(latent, keep_count / 512)
    norm_weight = torch.linspace(0.5, 2, 512, device=device, dtype=torch.float32)
    locations = torch.tensor(
        [0, 255, 256, 511, 23, -1, 42], device=device, dtype=torch.int32
    )
    plan = _prefill_plan(7, device)
    plan.plan_c.view(torch.int32)[6, 0] = -1
    buffers = (
        torch.zeros((2, 256, keep_count), dtype=torch.uint8, device=device),
        torch.zeros((2, 256, 8), dtype=torch.uint64, device=device),
        torch.zeros((2, 256, 8), dtype=torch.uint8, device=device),
    )

    pack_rows(
        latent,
        mask,
        norm_weight,
        1e-6,
        plan,
        locations,
        buffers,
    )
    expected = pack_rows_ref(latent, mask, norm_weight, 1e-6, keep_count=keep_count)
    selected_rows = locations[:5].cpu().tolist()
    packed_values = torch.stack(
        [buffers[0].reshape(-1, keep_count)[i] for i in selected_rows]
    )
    packed_bitmaps = torch.stack([buffers[1].reshape(-1, 8)[i] for i in selected_rows])
    packed_scales = torch.stack([buffers[2].reshape(-1, 8)[i] for i in selected_rows])
    assert torch.equal(packed_values, expected[0][:5])
    assert torch.equal(packed_bitmaps, expected[1][:5])
    assert torch.equal(packed_scales, expected[2][:5])
    # Padded plan entries must not write their otherwise-valid destination.
    for plane in buffers:
        assert not plane.reshape(-1, plane.shape[-1])[42].view(torch.uint8).any()
    remnant.configure_cache_format("native")
