"""Triton selected-row unpack correctness for the Remnant C4 record.

Author: Winston Cai.
"""

import hashlib
import importlib.util
import json
import sys
from pathlib import Path

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


@pytest.mark.parametrize("keep_count", [64, 128, 192, 256, 320, 384, 448])
@pytest.mark.parametrize("strided", [False, True])
@pytest.mark.parametrize("empty", [None, "queries", "selections"])
def test_triton_native_pages_match_independent_reference(strided, empty, keep_count):
    # A 256 default must not override the individual buffer's survivor width.
    remnant.configure_cache_format("remnant", "triton")
    torch.manual_seed(19)
    latent = torch.randn((6, 512), dtype=torch.float32)
    latent[0].zero_()
    latent[1].fill_(1)  # cutoff ties use the existing selection policy
    latent[2, ::3] = 0
    mask = topmag_keep_mask(latent, keep_count / 512)
    values, bitmaps, scales = pack_rows_ref(
        latent, mask, torch.ones(512), 1e-6, keep_count=keep_count
    )
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
    # Candidate and split outputs own distinct storage. Extra rows are a
    # canary: neither empty nor partially valid selections may write them.
    fused = torch.full((rows + 3, 512), -91, dtype=torch.bfloat16, device="cuda")
    unpack_gather_bf16(
        buffers, physical, raw, lengths, freqs_cpu.cuda(), fused, fused_rope=True
    )
    assert torch.equal(
        fused[:rows].view(torch.int16),
        workspace.dense_bf16[:rows].view(torch.int16),
    ), f"fused/split BF16 bytes differ for k={keep_count}, strided={strided}"
    assert torch.all(fused[rows:] == -91)
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
        expected_dense = (
            decoded[p].clone() if valid else torch.zeros(512, dtype=torch.bfloat16)
        )
        expected_dense[448:] = expected_tail
        torch.testing.assert_close(
            fused[row].cpu(), expected_dense, atol=2e-2, rtol=2e-2
        )
        actual_tail = native[page, value_base + 448 : value_base + 576].view(
            torch.bfloat16
        )
        torch.testing.assert_close(actual_tail, expected_tail, atol=2e-2, rtol=2e-2)


@pytest.mark.parametrize("keep_count", [192, 256])
def test_fused_rope_rotates_into_pruned_partner_lanes(keep_count):
    """A retained real/imaginary lane must contribute to its pruned partner."""
    remnant.configure_cache_format("remnant", "triton", keep_count=keep_count)
    latent = torch.zeros((2, 512), dtype=torch.float32)
    latent[:, : keep_count - 2] = 2
    latent[:, 448] = 3
    latent[:, 451] = -3
    mask = latent != 0
    planes = pack_rows_ref(latent, mask, torch.ones(512), 1e-6, keep_count=keep_count)
    physical = torch.tensor([[1, 1]], dtype=torch.int32, device="cuda")
    raw = torch.tensor([[0, 2]], dtype=torch.int32, device="cuda")
    lengths = torch.tensor([2], dtype=torch.int32, device="cuda")
    angles = torch.linspace(0.2, 1.1, 16 * 32).reshape(16, 32)
    freqs = torch.polar(torch.ones_like(angles), angles)
    split = torch.empty((2, 512), dtype=torch.bfloat16, device="cuda")
    fused = torch.empty_like(split)
    buffers = tuple(t.cuda() for t in planes)
    for flag, output in ((False, split), (True, fused)):
        unpack_gather_bf16(
            buffers, physical, raw, lengths, freqs.cuda(), output, fused_rope=flag
        )
    assert torch.equal(fused.view(torch.int16), split.view(torch.int16))
    expected = unpack_rows_ref(*planes)[[1, 1]]
    pairs = expected[:, 448:].float().reshape(2, 32, 2)
    rotated = torch.view_as_complex(pairs.contiguous()) * freqs[[0, 8]]
    expected[:, 448:] = torch.view_as_real(rotated).flatten(1).to(torch.bfloat16)
    assert torch.all(expected[:, [449, 450]] != 0)
    torch.testing.assert_close(fused.cpu(), expected, atol=2e-2, rtol=2e-2)


def test_prefill_widths_reuse_gather_kernels(monkeypatch):
    """Growing contexts must not compile a new unpack/RoPE kernel per width."""
    from collections import Counter

    import triton

    remnant.configure_cache_format("remnant", "triton")
    torch.manual_seed(23)
    latent = torch.randn((19, 512), dtype=torch.float32)
    mask = topmag_keep_mask(latent, 0.5)
    values, bitmaps, scales = pack_rows_ref(latent, mask, torch.ones(512), 1e-6)
    decoded = unpack_rows_ref(values, bitmaps, scales)
    dense_codes = torch.zeros_like(latent, dtype=torch.uint8)
    dense_codes[mask] = values.reshape(-1)
    workspace = NativeWorkspace.allocate(2, 768, 64, "cuda")
    names = {
        "_unpack_gather_bf16_kernel",
        "_unpack_gather_bf16_rope_kernel",
        "_rope_tail_complex_inplace_kernel",
        "_bf16_to_native_kernel",
    }
    compilations = Counter()

    def record_compile(*args, **kwargs):
        name = kwargs["fn"].name.rsplit(".", 1)[-1]
        if name in names:
            compilations[name] += 1

    monkeypatch.setattr(triton.knobs.runtime, "jit_cache_hook", record_compile)
    warm_counts = None
    # Odd widths keep ordinary scalar alignment specialization unchanged.
    # Row strides, Packed capacity and RoPE bounds all change independently.
    for width, capacity, freq_rows in zip(
        (513, 529, 547, 563), (7, 11, 15, 19), (32, 48, 64, 80)
    ):
        buffers = tuple(t[:capacity].cuda() for t in (values, bitmaps, scales))
        physical = torch.arange(2 * width, dtype=torch.int32).reshape(2, width) % 7
        raw = torch.arange(2 * width, dtype=torch.int32).reshape(2, width) % 21
        physical[0, 0] = -1
        physical[1, 0] = capacity
        lengths = torch.tensor((width, width - 1), dtype=torch.int32)
        angles = torch.arange(freq_rows * 32).reshape(freq_rows, 32) * 0.0017
        freqs = torch.polar(torch.ones_like(angles), angles)
        native, indices = unpack_gather_native(
            buffers,
            physical.cuda(),
            raw.cuda(),
            lengths.cuda(),
            freqs.cuda(),
            workspace,
        )
        fused = torch.empty((2 * width, 512), dtype=torch.bfloat16, device="cuda")
        unpack_gather_bf16(
            buffers,
            physical.cuda(),
            raw.cuda(),
            lengths.cuda(),
            freqs.cuda(),
            fused,
            fused_rope=True,
        )
        assert torch.equal(
            fused.view(torch.int16),
            workspace.dense_bf16[: 2 * width].view(torch.int16),
        )
        torch.cuda.synchronize()
        if warm_counts is None:
            warm_counts = compilations.copy()
        else:
            assert compilations == warm_counts, (
                f"prefill width={width}, capacity={capacity}, freq_rows={freq_rows} "
                f"compiled additional gather kernels: {compilations - warm_counts}"
            )

        flat_physical, flat_raw = physical.flatten(), raw.flatten()
        valid = (
            (torch.arange(width)[None, :] < lengths[:, None]).flatten()
            & (flat_physical >= 0)
            & (flat_physical < capacity)
            & (flat_raw * 4 < freq_rows)
        )
        expected = torch.zeros((2 * width, 512), dtype=torch.bfloat16)
        expected[valid] = decoded[flat_physical[valid]]
        tails = expected[valid, 448:].float().reshape(-1, 32, 2)
        rotated = torch.view_as_complex(tails.contiguous()) * freqs[flat_raw[valid] * 4]
        expected[valid, 448:] = (
            torch.view_as_real(rotated).flatten(1).to(torch.bfloat16)
        )
        torch.testing.assert_close(
            workspace.dense_bf16[: 2 * width].cpu(), expected, atol=2e-2, rtol=2e-2
        )
        assert torch.equal(indices.cpu().flatten(), torch.arange(2 * width))
        native_cpu = native.cpu()
        records = native_cpu[:, : 64 * 576].reshape(-1, 576)[: 2 * width]
        expected_codes = torch.zeros((2 * width, 448), dtype=torch.uint8)
        expected_codes[valid] = dense_codes[flat_physical[valid], :448]
        assert torch.equal(records[:, :448], expected_codes)
        actual_tail = records[:, 448:].contiguous().view(torch.bfloat16)
        torch.testing.assert_close(actual_tail, expected[:, 448:], atol=2e-2, rtol=2e-2)


@pytest.fixture
def main_packed(monkeypatch):
    """Import immutable main source without importing its runtime patch machinery."""
    root = Path(__file__).parent / "fixtures/main_packed"
    manifest = json.loads((root / "source.json").read_text())
    assert manifest["revision"] == "b69841acf0d29181f84999d9b7e3cd5505eea9ad"
    for name, expected in manifest["files"].items():
        assert hashlib.sha256((root / name).read_bytes()).hexdigest() == expected, name
    # These switches configure only the frozen package; current production uses
    # configure_cache_format. They never alter the serving process or its source.
    for name, value in {
        "SGLANG_OPT_TOPMAG": "1",
        "KEEP": "0.5",
        "SGLANG_OPT_TOPMAG_PACKED": "1",
        "SGLANG_OPT_TOPMAG_FUSED": "0",
    }.items():
        monkeypatch.setenv(name, value)
    name = "_remnant_main_packed_fixture"
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(
            name, root / "__init__.py", submodule_search_locations=[str(root)]
        )
        package = importlib.util.module_from_spec(spec)
        sys.modules[name] = package
        spec.loader.exec_module(package)
    from importlib import import_module

    return (
        import_module(f"{name}.packed"),
        import_module(f"{name}.reference"),
    )


@pytest.mark.parametrize("is_decode", [False, True], ids=["prefill", "decode"])
@pytest.mark.parametrize("selected_k", [64, 512])
def test_triton_matches_main_packed(main_packed, is_decode, selected_k):
    """Exact parity on shared valid inputs, including stock attention output/LSE."""
    from sgl_kernel import flash_mla

    from sglang.srt.remnant.packed import pack_rows

    historical, historical_reference = main_packed
    remnant.configure_cache_format("remnant", "triton")
    device = torch.device("cuda")
    torch.manual_seed(71)
    rows, batch, page_size = 128, 2, 64
    latent = torch.randn((rows, 512), device=device, dtype=torch.float32)
    latent[0].zero_()
    latent[1].fill_(1)  # equal-magnitude cutoff ties
    latent[2, ::3] = 0
    norm_weight = torch.linspace(0.7, 1.3, 512, device=device)
    mask = topmag_keep_mask(latent, 0.5)
    assert torch.equal(mask, historical_reference.topmag_keep_mask(latent, 0.5))

    class Plan:
        def __init__(self):
            self.is_decode = is_decode
            self.words = torch.zeros((rows, 4), device=device, dtype=torch.int32)
            self.words[:, 0] = (torch.arange(rows, device=device) + 1) * 4
            self.words[:, 1] = torch.arange(rows, device=device)
            if is_decode:
                self.words[3, 0] += 1  # neither packer should write this row

        def __getitem__(self, index):
            assert index == 1
            return self.words

    plan = Plan()
    locations = torch.arange(rows - 1, -1, -1, device=device, dtype=torch.int32)

    def buffers():
        return (
            torch.zeros((2, page_size, 256), device=device, dtype=torch.uint8),
            torch.zeros((2, page_size, 8), device=device, dtype=torch.uint64),
            torch.zeros((2, page_size, 8), device=device, dtype=torch.uint8),
        )

    current_buffers, old_buffers = buffers(), buffers()
    pack_rows(latent, mask, norm_weight, 1e-6, plan, locations, current_buffers)
    historical.pack_rows(
        latent.clone(),
        mask.clone(),
        norm_weight.clone(),
        1e-6,
        plan,
        locations.clone(),
        old_buffers,
    )
    for label, current, old in zip(
        ("values", "bitmaps", "scales"), current_buffers, old_buffers
    ):
        assert torch.equal(current, old), f"Packed {label} differs from main"

    physical = (
        torch.arange(batch * selected_k, device=device, dtype=torch.int32)
        .reshape(batch, selected_k)
        .remainder(rows)
    )
    raw = physical.remainder(29)
    lengths = torch.tensor(
        (selected_k, selected_k - 7), device=device, dtype=torch.int32
    )
    raw[1, selected_k - 7 :] = -1
    angles = torch.arange(128 * 32, device=device).reshape(128, 32) * 0.0017
    freqs = torch.polar(torch.ones_like(angles), angles)
    current_ws = NativeWorkspace.allocate(batch, selected_k, page_size, device)
    old_ws = historical.NativeWorkspace.allocate(batch, selected_k, page_size, device)
    current_bytes, current_indices = unpack_gather_native(
        current_buffers, physical, raw, lengths, freqs, current_ws
    )
    old_bytes, old_indices = historical.unpack_gather_native(
        old_buffers,
        physical.clone(),
        raw.clone(),
        lengths.clone(),
        freqs.clone(),
        old_ws,
    )
    assert torch.equal(current_ws.dense_bf16, old_ws.dense_bf16), "BF16/RoPE parity"
    assert torch.equal(current_indices, old_indices), "temporary index parity"
    active = (
        torch.arange(selected_k, device=device)[None, :] < lengths[:, None]
    ).flatten()
    for offset, width, label in (
        (0, 576, "Native values"),
        (page_size * 576, 8, "Native scales"),
    ):
        current = current_bytes[:, offset : offset + page_size * width]
        old = old_bytes[:, offset : offset + page_size * width]
        assert torch.equal(
            current.reshape(-1, width)[active], old.reshape(-1, width)[active]
        ), f"{label} parity"

    def cache(storage, workspace):
        return storage.as_strided(
            (storage.shape[0], page_size, 1, 584),
            (workspace.bytes_per_page, 584, 584, 1),
        )

    current_cache = cache(current_bytes, current_ws)
    q = torch.randn((batch, 1, 64, 512), device=device, dtype=torch.bfloat16)
    sink = torch.linspace(-0.1, 0.1, 64, device=device)
    # Both calls share identical SWA/query/sink inputs; only the reconstructed
    # extra C4 cache differs. Native FlashMLA owns attention in both versions.
    outputs = []
    for storage, workspace, indices in (
        (current_bytes, current_ws, current_indices),
        (old_bytes, old_ws, old_indices),
    ):
        outputs.append(
            flash_mla.flash_mla_with_kvcache(
                q,
                current_cache,
                None,
                None,
                512,
                flash_mla.get_mla_metadata()[0],
                softmax_scale=512**-0.5,
                is_fp8_kvcache=True,
                # Native FlashMLA requires a 64-aligned index width, even when the
                # active SWA prefix contains only one entry.
                indices=current_indices[:, :page_size].unsqueeze(1).contiguous(),
                attn_sink=sink,
                extra_k_cache=cache(storage, workspace),
                extra_indices_in_kvcache=indices.unsqueeze(1),
                topk_length=torch.ones(batch, device=device, dtype=torch.int32),
                extra_topk_length=lengths,
            )
        )
    for label, current, old in zip(("attention output", "LSE"), *outputs):
        assert torch.isfinite(current).all(), label
        assert torch.equal(current, old), f"{label} parity"
