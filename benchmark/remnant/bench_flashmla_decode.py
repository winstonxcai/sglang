"""Model-free FlashMLA Native, adapter, and direct Remnant decode timing."""

from __future__ import annotations

import argparse

import torch

from sgl_kernel import flash_mla
from sglang.srt import remnant
from sglang.srt.remnant import NativeWorkspace
from sglang.srt.remnant.packed import pack_rows, unpack_gather_native
from sglang.srt.remnant.reference import topmag_keep_mask


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


def _measure(fn, warmup: int, repeats: int) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(repeats):
        fn()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) / repeats


def _kernel_time(fn) -> float:
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as prof:
        fn()
    torch.cuda.synchronize()
    names = ("flash_fwd_splitkv_mla_fp8_sparse_kernel", "flash_fwd_mla_combine_kernel")
    return sum(
        event.device_time_total
        for event in prof.key_averages()
        if any(name in event.key for name in names)
    ) / 1000.0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batches", default="8,16")
    parser.add_argument("--repeats", type=int, default=50)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--max-regression-percent", type=float, default=2.0)
    parser.add_argument(
        "--path",
        choices=("all", "native", "adapter", "direct"),
        default="all",
        help="Run one path only when isolating it under a GPU profiler.",
    )
    args = parser.parse_args()
    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 9:
        raise RuntimeError("This benchmark requires an H100")

    device = torch.device("cuda")
    remnant.configure_cache_format("remnant")
    print("heads,batch,topk,native_total_ms,adapter_total_ms,direct_total_ms,"
          "native_kernel_ms,adapter_kernel_ms,direct_kernel_ms,direct_vs_native_pct,status")
    misses = []
    for heads in (64, 128):
        for batch in (int(value) for value in args.batches.split(",")):
            rows = batch * 64
            q = torch.randn((batch, 1, heads, 512), device=device, dtype=torch.bfloat16)
            swa_cache = torch.zeros((batch, 64, 1, 584), dtype=torch.uint8, device=device)
            swa_indices = torch.arange(64, dtype=torch.int32, device=device).view(1, 1, 64)
            swa_indices = (swa_indices + torch.arange(batch, device=device, dtype=torch.int32).view(batch, 1, 1) * 64).contiguous()
            swa_lengths = torch.full((batch,), 64, dtype=torch.int32, device=device)
            physical = torch.arange(512, dtype=torch.int32, device=device).view(1, 1, 512).remainder(64)
            physical = (physical + torch.arange(batch, device=device, dtype=torch.int32).view(batch, 1, 1) * 64).contiguous()
            raw = physical + 3
            lengths = torch.full((batch,), 512, dtype=torch.int32, device=device)
            latent = torch.randn((rows, 512), device=device)
            mask = topmag_keep_mask(latent, 0.5)
            buffers = (
                torch.zeros((batch, 64, 256), dtype=torch.uint8, device=device),
                torch.zeros((batch, 64, 8), dtype=torch.uint64, device=device),
                torch.zeros((batch, 64, 8), dtype=torch.uint8, device=device),
            )
            pack_rows(
                latent, mask, torch.ones(512, device=device), 1.0e-6,
                _PackPlan(rows, device), torch.arange(rows, dtype=torch.int32, device=device), buffers,
            )
            workspace = NativeWorkspace.allocate(batch, 512, 64, device, with_dense=True)
            freqs = torch.ones((max(128, int(raw.max().item()) + 2) * 128 + 32,), dtype=torch.complex64, device=device)
            baseline_bytes, baseline_indices = unpack_gather_native(
                buffers, physical.flatten(0, 1), raw.flatten(0, 1), lengths,
                freqs, workspace,
            )
            baseline_cache = baseline_bytes.as_strided(
                (baseline_bytes.shape[0], 64, 1, 584),
                (workspace.bytes_per_page, 584, 584, 1),
            )
            baseline_indices = baseline_indices.unsqueeze(1)

            def native():
                meta = flash_mla.get_mla_metadata()[0]
                return flash_mla.flash_mla_with_kvcache(
                    q, swa_cache, None, None, 512, meta,
                    softmax_scale=512 ** -0.5,
                    is_fp8_kvcache=True,
                    indices=swa_indices,
                    extra_k_cache=baseline_cache,
                    extra_indices_in_kvcache=baseline_indices,
                    topk_length=swa_lengths,
                    extra_topk_length=lengths,
                )

            def adapter():
                native_bytes, native_indices = unpack_gather_native(
                    buffers, physical.flatten(0, 1), raw.flatten(0, 1), lengths,
                    freqs, workspace,
                )
                native_cache = native_bytes.as_strided(
                    (native_bytes.shape[0], 64, 1, 584),
                    (workspace.bytes_per_page, 584, 584, 1),
                )
                meta = flash_mla.get_mla_metadata()[0]
                return flash_mla.flash_mla_with_kvcache(
                    q, swa_cache, None, None, 512, meta,
                    softmax_scale=512 ** -0.5,
                    is_fp8_kvcache=True,
                    indices=swa_indices,
                    extra_k_cache=native_cache,
                    extra_indices_in_kvcache=native_indices.unsqueeze(1),
                    topk_length=swa_lengths,
                    extra_topk_length=lengths,
                )

            def direct():
                meta = flash_mla.get_mla_metadata()[0]
                return flash_mla.flash_mla_with_kvcache(
                    q, swa_cache, None, None, 512, meta,
                    softmax_scale=512 ** -0.5,
                    is_fp8_kvcache=True,
                    indices=swa_indices,
                    extra_indices_in_kvcache=physical,
                    topk_length=swa_lengths,
                    extra_topk_length=lengths,
                    remnant_buffers=buffers, remnant_raw_indices=raw,
                    remnant_freqs=torch.view_as_real(freqs),
                )

            native_ms = adapter_ms = direct_ms = float("nan")
            native_kernel = adapter_kernel = direct_kernel = float("nan")
            selected = {
                "native": native,
                "adapter": adapter,
                "direct": direct,
            }
            paths = selected if args.path == "all" else {args.path: selected[args.path]}
            measurements = {
                name: _measure(fn, args.warmup, args.repeats)
                for name, fn in paths.items()
            }
            if args.path == "all":
                native_ms = measurements["native"]
                adapter_ms = measurements["adapter"]
                direct_ms = measurements["direct"]
                native_kernel = _kernel_time(native)
                adapter_kernel = _kernel_time(adapter)
                direct_kernel = _kernel_time(direct)
            elif args.path == "native":
                native_ms = measurements["native"]
            elif args.path == "adapter":
                adapter_ms = measurements["adapter"]
            else:
                direct_ms = measurements["direct"]
            delta = 100 * (direct_ms / native_ms - 1) if args.path == "all" else float("nan")
            status = "PASS" if args.path != "all" or delta <= args.max_regression_percent else "MISS"
            if status == "MISS":
                misses.append((heads, batch, delta))
            print(
                f"{heads},{batch},512,{native_ms:.4f},{adapter_ms:.4f},{direct_ms:.4f},"
                f"{native_kernel:.4f},{adapter_kernel:.4f},{direct_kernel:.4f},"
                f"{delta:.3f},{status}"
            )
    if misses:
        details = ", ".join(f"H{heads}/B{batch}={delta:.2f}%" for heads, batch, delta in misses)
        raise RuntimeError(
            f"direct decode exceeds the {args.max_regression_percent:.2f}% target: {details}"
        )
    remnant.configure_cache_format("native")


if __name__ == "__main__":
    main()
