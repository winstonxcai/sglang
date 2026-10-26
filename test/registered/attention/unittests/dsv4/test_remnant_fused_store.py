"""Production fused-store bytes, plan routing and live graph inputs.

Author: Winston Cai.
"""

from unittest import mock

import pytest
import torch

from sglang.benchmark.remnant.cases import store_cases
from sglang.kernels.ops.attention.dsv4.compress import (
    CompressorDecodePlan,
    CompressorPrefillPlan,
)
from sglang.srt import remnant
from sglang.srt.remnant.packed import PackedBuffers, pack_rows, pack_rows_fused
from sglang.srt.remnant.reference import topmag_keep_mask
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(
    est_time=0,
    stage="base-a",
    runner_config="1-gpu-small",
    disabled="Requires the independently pinned SGLang/FlashMLA build; qualified by scripts/modal/kernels.py",
)

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


@pytest.fixture(autouse=True)
def cache_format():
    remnant.configure_cache_format("remnant")
    yield
    remnant.configure_cache_format("native")


def plan_and_locations(rows, phase, layout, device):
    words = torch.zeros((rows, 4), dtype=torch.int32, device=device)
    words[:, 0] = 4
    words[:, 1] = torch.arange(rows, device=device)
    locations = torch.arange(rows, device=device, dtype=torch.int32)
    if layout == "shuffled":
        locations = locations.flip(0)
    if layout in ("staggered", "no_boundary"):
        words[:, 0] = 5
        if layout == "staggered":
            words[::4, 0] = 4
    if layout in ("padded", "all_invalid"):
        words[:: 4 if layout == "padded" else 1, 0] = -1
        words[words[:, 0] < 0, 1] = 65535
    if layout == "invalid_dest":
        locations[::4] = -1
    blob = words.view(torch.uint8)
    if phase == "decode":
        plan = CompressorDecodePlan(4, blob)
    else:
        # Deliberately unrelated WritePlan catches plan[2] regressions.
        plan = CompressorPrefillPlan(
            4, blob, torch.zeros((1, 8), device=device, dtype=torch.uint8)
        )
    return plan, locations


def sentinel(rows, device):
    return PackedBuffers(
        torch.full((rows + 7, 256), 165, dtype=torch.uint8, device=device),
        torch.full((rows + 7, 8), 0x123456789ABCDEF, dtype=torch.uint64, device=device),
        torch.full((rows + 7, 8), 165, dtype=torch.uint8, device=device),
    )


def fixture(name):
    torch.manual_seed(20261010)
    rows = 8192 if name.startswith("large") else 257
    x = torch.randn((rows, 512), device="cuda", dtype=torch.float32)
    if name == "natural_zeros":
        x[:, ::3] = 0
    if name == "cutoff_ties":
        x.fill_(1)
        x[:, :128] = 0
    if name == "signed_zero":
        x.zero_()
        x[:, ::2] = -0.0
    if name in ("quantized_ties", "large_ties"):
        x = torch.randint(-4, 5, x.shape, device="cuda").float()
    if name == "nearby_fp32":
        x = (
            (1 + torch.arange(512, device="cuda", dtype=torch.float32) * 2**-20)[None]
            .expand(rows, -1)
            .contiguous()
        )
        x[:, 1::2] *= -1
    if name == "finite_extremes":
        bits = torch.randint(0, 0x7F800000, x.shape, device="cuda", dtype=torch.int32)
        bits[:, 1::2] |= -2147483648
        x = bits.view(torch.float32)
    return x


def assert_bytes(x, phase, layout, eps=1e-6):
    rows = len(x)
    plan, loc = plan_and_locations(rows, phase, layout, x.device)
    weight = torch.linspace(0.5, 1.5, 512, device=x.device, dtype=torch.float32)
    mask = topmag_keep_mask(x, 0.5)
    expected, actual = sentinel(rows, x.device), sentinel(rows, x.device)
    pack_rows(x, mask, weight, eps, plan, loc, expected)
    pack_rows_fused(x, weight, eps, plan, loc, actual)
    for a, b in zip(actual, expected):
        assert torch.equal(a.view(torch.uint8), b.view(torch.uint8))
    words = plan[1].view(torch.int32)
    eligible = ((words[:, 0] % 4 == 0) if phase == "decode" else (words[:, 0] >= 0)) & (
        loc >= 0
    )
    if bool(eligible.any()):
        kept = actual.bitmaps.view(torch.int64)[loc[eligible].long()]
        # The production byte comparison above covers empty/invalid rows too.
        # Independent selection is exact; Torch attention references elsewhere
        # use numerical tolerances rather than bitwise FP32 reduction parity.
        from sglang.srt.remnant.bitmap import bitmap_to_mask

        assert torch.equal(bitmap_to_mask(kept), mask[eligible])


@pytest.mark.parametrize(
    "name",
    [
        "normal",
        "natural_zeros",
        "cutoff_ties",
        "signed_zero",
        "quantized_ties",
        "nearby_fp32",
        "large_normal",
        "large_ties",
        "finite_extremes",
    ],
)
@pytest.mark.parametrize(
    "phase,layout",
    [("decode", "all_valid"), ("decode", "staggered"), ("prefill", "padded")],
)
def test_adversarial_masks_and_bytes(name, phase, layout):
    assert_bytes(fixture(name), phase, layout)


COVERAGE = [(c.phase, c.rows, c.layout) for c in store_cases()]


@pytest.mark.parametrize("phase,rows,layout", COVERAGE)
def test_shared_coverage(phase, rows, layout):
    assert_bytes(torch.randn((rows, 512), device="cuda"), phase, layout)


@pytest.mark.parametrize("phase", ["decode", "prefill"])
def test_graph_replay_with_changed_inputs_and_plans(phase):
    x = torch.randn((32, 512), device="cuda")
    w = torch.ones(512, device="cuda")
    plan, loc = plan_and_locations(32, phase, "all_valid", x.device)
    actual, expected = sentinel(32, x.device), sentinel(32, x.device)
    pack_rows_fused(x, w, 1e-5, plan, loc, actual)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        pack_rows_fused(x, w, 1e-5, plan, loc, actual)
    for iteration in range(2):
        x.normal_()
        loc.copy_(
            torch.arange(32, device=x.device, dtype=loc.dtype).roll(iteration + 1)
        )
        words = plan[1].view(torch.int32)
        words[:, 0] = 5 if phase == "decode" else -1
        words[iteration::2, 0] = 4
        for a, b in zip(actual, expected):
            a.copy_(b)
        graph.replay()
        pack_rows(x, topmag_keep_mask(x, 0.5), w, 1e-5, plan, loc, expected)
        torch.cuda.synchronize()
        for a, b in zip(actual, expected):
            assert torch.equal(a.view(torch.uint8), b.view(torch.uint8))


@pytest.mark.parametrize(
    "kind", ["bf16", "strided_latent", "strided_weight", "other_arch"]
)
def test_fallback_preserves_original_selection(kind):
    x = torch.randn((32, 1024 if kind == "strided_latent" else 512), device="cuda")
    if kind == "strided_latent":
        x = x[:, ::2]
    if kind == "bf16":
        x = x.bfloat16()
    w = torch.ones(1024 if kind == "strided_weight" else 512, device="cuda")
    if kind == "strided_weight":
        w = w[::2]
    plan, loc = plan_and_locations(32, "decode", "all_valid", x.device)
    actual, expected = sentinel(32, x.device), sentinel(32, x.device)
    pack_rows(
        x.contiguous(),
        topmag_keep_mask(x, 0.5),
        w.contiguous(),
        1e-6,
        plan,
        loc,
        expected,
    )
    with mock.patch(
        "torch.cuda.get_device_capability",
        return_value=(8, 0) if kind == "other_arch" else (9, 0),
    ):
        pack_rows_fused(x, w, 1e-6, plan, loc, actual)
    for a, b in zip(actual, expected):
        assert torch.equal(a.view(torch.uint8), b.view(torch.uint8))


def test_empty_does_not_select_or_launch():
    x = torch.empty((0, 512), device="cuda")
    plan, loc = plan_and_locations(0, "prefill", "all_valid", x.device)
    with mock.patch(
        "sglang.srt.remnant.reference.topmag_keep_mask",
        side_effect=AssertionError("selection on empty"),
    ):
        pack_rows_fused(
            x, torch.ones(512, device="cuda"), 1e-6, plan, loc, sentinel(0, x.device)
        )


@pytest.mark.parametrize(
    "heads,batch,length", [(64, 8, 512), (128, 16, 512), (128, 16, 317)]
)
def test_nonuniform_fp32_weights_preserve_production_pair_rounding(
    heads, batch, length
):
    from sglang.benchmark.remnant.cases import SEED, DecodeCase, case_seed

    generator = torch.Generator(device="cuda").manual_seed(
        case_seed(DecodeCase(heads, batch, length), SEED)
    )
    rows = batch * 512
    x = torch.randn(
        (rows, 512), device="cuda", dtype=torch.float32, generator=generator
    )
    weight = torch.linspace(0.5, 1.5, 512, device="cuda", dtype=torch.float32)
    plan, locations = plan_and_locations(rows, "prefill", "all_valid", x.device)
    locations = locations.to(torch.int64)
    actual, expected = sentinel(rows, x.device), sentinel(rows, x.device)
    pack_rows_fused(x, weight, 1e-6, plan, locations, actual)
    pack_rows(x, topmag_keep_mask(x, 0.5), weight, 1e-6, plan, locations, expected)
    for a, b in zip(actual, expected):
        assert torch.equal(a.view(torch.uint8), b.view(torch.uint8))


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
