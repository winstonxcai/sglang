"""Triton selected-row unpack correctness for the Remnant C4 record.

Author: Winston Cai.
"""

import importlib.util

import pytest
import torch

from sglang.srt import remnant
from sglang.srt.remnant.packed import (
    NativeWorkspace,
    unpack_gather_bf16,
    unpack_gather_native,
)
from sglang.srt.remnant.reference import (
    pack_rows_ref,
    topmag_keep_mask,
    unpack_rows_ref,
)

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or importlib.util.find_spec("triton") is None,
    reason="CUDA and Triton are required",
)


@pytest.fixture(autouse=True)
def _reset_cache_format():
    yield
    remnant.configure_cache_format("native")


def test_remnant_triton_unpack_reconstructs_selected_row():
    device = torch.device("cuda")
    remnant.configure_cache_format("remnant")
    values = torch.zeros((1, 256, 256), dtype=torch.uint8, device=device)
    bitmaps = torch.zeros((1, 256, 8), dtype=torch.uint64, device=device)
    all_bits = torch.tensor(-1, dtype=torch.int64, device=device).view(torch.uint64)
    bitmaps[:, :, :4] = all_bits
    scales = torch.full((1, 256, 8), 127, dtype=torch.uint8, device=device)
    physical = torch.zeros((1, 1), dtype=torch.int32, device=device)
    raw = torch.zeros((1, 1), dtype=torch.int32, device=device)
    lengths = torch.ones(1, dtype=torch.int32, device=device)
    freqs = torch.ones(512, dtype=torch.complex64, device=device)
    output = torch.empty((1, 512), dtype=torch.bfloat16, device=device)

    unpack_gather_bf16((values, bitmaps, scales), physical, raw, lengths, freqs, output)
    torch.cuda.synchronize()
    assert torch.isfinite(output.float()).all()
    assert output.shape == (1, 512)
    remnant.configure_cache_format("native")


@pytest.mark.parametrize("strided", [False, True])
@pytest.mark.parametrize("empty", [None, "queries", "selections"])
def test_triton_native_pages_match_independent_reference(strided, empty):
    remnant.configure_cache_format("remnant", "triton")
    torch.manual_seed(19)
    latent = torch.randn((6, 512), dtype=torch.float32)
    latent[0].zero_()
    latent[1].fill_(1)  # cutoff ties use the existing selection policy
    latent[2, ::3] = 0
    mask = topmag_keep_mask(latent, 0.5)
    values, bitmaps, scales = pack_rows_ref(latent, mask, torch.ones(512), 1e-6)
    buffers = tuple(t.cuda() for t in (values, bitmaps, scales))
    decoded = unpack_rows_ref(values, bitmaps, scales)
    dense_codes = torch.zeros((6, 512), dtype=torch.uint8)
    dense_codes[mask] = values.reshape(-1)
    physical_cpu = torch.tensor(
        [
            [3, 3, -1, 2**31 - 1, 0],
            [4, 1, 2, 5, 0],
            [2**31 - 1, 0, 0, 0, 0],
        ],
        dtype=torch.int32,
    )
    raw_cpu = torch.tensor(
        [
            [2, 3, 2**31 - 1, 2, -1],
            [1, 0, 4, 2**31 - 1, 1],
            [2**31 - 1, 0, 0, 0, 0],
        ],
        dtype=torch.int32,
    )
    lengths_cpu = torch.tensor([4, 3, 0], dtype=torch.int32)
    if empty == "queries":
        physical_cpu, raw_cpu, lengths_cpu = (
            physical_cpu[:0],
            raw_cpu[:0],
            lengths_cpu[:0],
        )
    if empty == "selections":
        physical_cpu, raw_cpu = physical_cpu[:, :0], raw_cpu[:, :0]
        lengths_cpu.zero_()
    physical, raw, lengths = (t.cuda() for t in (physical_cpu, raw_cpu, lengths_cpu))
    if strided:
        physical = torch.stack((physical, physical), dim=-1)[..., 0]
        raw = torch.stack((raw, raw, raw), dim=-1)[..., 1]
        lengths = torch.stack((lengths, lengths), dim=-1)[..., 0]
    angles = torch.arange(32 * 32, dtype=torch.float32).reshape(32, 32) * 0.0017
    freqs_cpu = torch.polar(torch.ones_like(angles), angles)
    workspace = NativeWorkspace.allocate(3, 8, 4, "cuda")
    native, indices = unpack_gather_native(
        buffers, physical, raw, lengths, freqs_cpu.cuda(), workspace
    )
    rows = physical.numel()
    assert torch.equal(indices.cpu().flatten(), torch.arange(rows, dtype=torch.int32))
    if not rows:
        assert torch.count_nonzero(native).item() == 0
        return
    native = native.cpu()
    selected_k = physical.shape[1]
    for row in range(rows):
        query, k = divmod(row, selected_k)
        p, r = int(physical_cpu[query, k]), int(raw_cpu[query, k])
        valid = k < lengths_cpu[query] and 0 <= p < 6 and 0 <= r * 4 < 32
        page, offset = divmod(row, workspace.page_size)
        value_base = offset * 576
        scale_base = workspace.page_size * 576 + offset * 8
        expected_codes = (
            dense_codes[p, :448] if valid else torch.zeros(448, dtype=torch.uint8)
        )
        expected_scales = scales[p, :7] if valid else torch.zeros(7, dtype=torch.uint8)
        assert torch.equal(native[page, value_base : value_base + 448], expected_codes)
        assert torch.equal(native[page, scale_base : scale_base + 7], expected_scales)
        assert native[page, scale_base + 7] == 0
        expected_tail = torch.zeros(64, dtype=torch.bfloat16)
        if valid:
            tail = decoded[p, 448:].float().reshape(32, 2)
            rotated = torch.view_as_complex(tail.contiguous()) * freqs_cpu[r * 4]
            expected_tail = torch.view_as_real(rotated).flatten().to(torch.bfloat16)
        actual_tail = native[page, value_base + 448 : value_base + 576].view(
            torch.bfloat16
        )
        torch.testing.assert_close(actual_tail, expected_tail, atol=2e-2, rtol=2e-2)
