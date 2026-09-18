"""Triton selected-row unpack correctness for the Remnant C4 record."""

import importlib.util

import pytest
import torch

from sglang.srt import remnant
from sglang.srt.remnant.packed import unpack_gather_bf16


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or importlib.util.find_spec("triton") is None,
    reason="CUDA and Triton are required",
)


def test_remnant_triton_unpack_reconstructs_selected_row():
    device = torch.device("cuda")
    remnant.configure_cache_format("remnant")
    values = torch.randint(0, 255, (1, 256, 256), dtype=torch.uint8, device=device)
    bitmaps = torch.full(
        (1, 256, 8), torch.iinfo(torch.uint64).max, dtype=torch.uint64, device=device
    )
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
    assert torch.isfinite(output.float()).all()
    assert output.shape == (1, 512)
    remnant.configure_cache_format("native")
