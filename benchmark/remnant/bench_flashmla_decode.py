"""Model-free steady-state FlashMLA Native, fused adapter, and Remnant timing.

Author: Winston Cai.
"""

from __future__ import annotations

import argparse
import random
import statistics

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


def _frequencies(max_position: int, device: torch.device) -> torch.Tensor:
    pair = torch.arange(32, device=device, dtype=torch.float32)
    position = torch.arange(max_position, device=device, dtype=torch.float32)[:, None]
    angle = (position + 1.0) * (pair + 1.0) * 0.0017
    table = torch.zeros((1, max_position * 128 + 32, 2), device=device)
    table[..., 0] = 1.0
    shaped = table[0, : max_position * 128].view(max_position, 128, 2)
    shaped[:, :32, 0] = torch.cos(angle)
    shaped[:, :32, 1] = torch.sin(angle)
    return table.view(torch.complex64).reshape(-1).contiguous()


def _fill_swa(cache: torch.Tensor) -> None:
    """Populate valid, non-zero MODEL1 records without affecting comparisons."""
    rows = cache.shape[0] * cache.shape[1]
    flat = cache.reshape(rows, 584)
    values = (torch.arange(rows * 448, device=cache.device) % 113 + 1).to(torch.uint8)
    flat[:, :448].copy_(values.view(rows, 448))
    tail = torch.linspace(-0.25, 0.25, rows * 64, device=cache.device).to(torch.bfloat16)
    flat[:, 448:576].copy_(tail.view(torch.uint8).reshape(rows, 128))
    flat[:, 576:584].fill_(127)


def _capture(fn, warmup: int) -> torch.cuda.CUDAGraph:
    for _ in range(max(3, warmup)):
        fn()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        fn()
    graph.replay()
    torch.cuda.synchronize()
    return graph


def _measure_graph(graph: torch.cuda.CUDAGraph, repeats: int) -> float:
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(repeats):
        graph.replay()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) / repeats


def _measure_eager(fn, repeats: int) -> float:
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(repeats):
        fn()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) / repeats


def _percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int((len(ordered) - 1) * fraction + 0.5))]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--heads", default="64,128")
    parser.add_argument("--batches", default="8,16")
    parser.add_argument("--repeats", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--rounds", type=int, default=9)
    parser.add_argument("--max-regression-percent", type=float, default=2.0)
    parser.add_argument("--path", choices=("all", "native", "adapter", "direct"), default="all")
    args = parser.parse_args()
    if args.repeats <= 0 or args.rounds <= 0 or args.warmup < 0:
        raise ValueError("repeats/rounds must be positive and warmup nonnegative")
    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 9:
        raise RuntimeError("This benchmark requires an H100")

    torch.manual_seed(20260919)
    device = torch.device("cuda")
    remnant.configure_cache_format("remnant")
    print("heads,batch,topk,native_decode_ms,direct_decode_ms,adapter_total_ms,median_regression_pct,p95_regression_pct,status")
    misses = []
    for heads in (int(value) for value in args.heads.split(",")):
        for batch in (int(value) for value in args.batches.split(",")):
            selected_k = 512
            rows = batch * selected_k
            q = torch.randn((batch, 1, heads, 512), device=device, dtype=torch.bfloat16)
            swa_cache = torch.zeros((batch, 64, 1, 584), dtype=torch.uint8, device=device)
            _fill_swa(swa_cache)
            swa_indices = torch.arange(64, dtype=torch.int32, device=device).view(1, 1, 64)
            swa_indices = (swa_indices + torch.arange(batch, device=device, dtype=torch.int32).view(batch, 1, 1) * 64).contiguous()
            swa_lengths = torch.full((batch,), 64, dtype=torch.int32, device=device)
            physical = torch.arange(rows, dtype=torch.int32, device=device).view(batch, 1, selected_k)
            raw = (torch.arange(rows, dtype=torch.int32, device=device) * 3 + 17).view(batch, 1, selected_k)
            lengths = torch.full((batch,), selected_k, dtype=torch.int32, device=device)

            latent = torch.randn((rows, 512), device=device)
            mask = topmag_keep_mask(latent, 0.5)
            pages = rows // 64
            buffers = (
                torch.zeros((pages, 64, 256), dtype=torch.uint8, device=device),
                torch.zeros((pages, 64, 8), dtype=torch.uint64, device=device),
                torch.zeros((pages, 64, 8), dtype=torch.uint8, device=device),
            )
            pack_rows(
                latent, mask, torch.ones(512, device=device), 1.0e-6,
                _PackPlan(rows, device), torch.arange(rows, dtype=torch.int32, device=device), buffers,
            )
            freqs = _frequencies(int(raw.max().item()) + 2, device)
            workspace = NativeWorkspace.allocate(batch, selected_k, 64, device)
            native_bytes, native_indices = unpack_gather_native(
                buffers, physical.flatten(0, 1), raw.flatten(0, 1), lengths, freqs, workspace
            )
            native_cache = native_bytes.as_strided(
                (native_bytes.shape[0], 64, 1, 584),
                (workspace.bytes_per_page, 584, 584, 1),
            )
            native_indices = native_indices.unsqueeze(1)
            native_meta = flash_mla.get_mla_metadata()[0]
            adapter_meta = flash_mla.get_mla_metadata()[0]
            direct_meta = flash_mla.get_mla_metadata()[0]

            def native():
                return flash_mla.flash_mla_with_kvcache(
                    q, swa_cache, None, None, 512, native_meta,
                    softmax_scale=512 ** -0.5, is_fp8_kvcache=True,
                    indices=swa_indices, extra_k_cache=native_cache,
                    extra_indices_in_kvcache=native_indices,
                    topk_length=swa_lengths, extra_topk_length=lengths,
                )

            def adapter():
                adapter_bytes, adapter_indices = unpack_gather_native(
                    buffers, physical.flatten(0, 1), raw.flatten(0, 1), lengths, freqs, workspace,
                )
                adapter_cache = adapter_bytes.as_strided(
                    (adapter_bytes.shape[0], 64, 1, 584),
                    (workspace.bytes_per_page, 584, 584, 1),
                )
                return flash_mla.flash_mla_with_kvcache(
                    q, swa_cache, None, None, 512, adapter_meta,
                    softmax_scale=512 ** -0.5, is_fp8_kvcache=True,
                    indices=swa_indices, extra_k_cache=adapter_cache,
                    extra_indices_in_kvcache=adapter_indices.unsqueeze(1),
                    topk_length=swa_lengths, extra_topk_length=lengths,
                )

            def direct():
                return flash_mla.flash_mla_with_kvcache(
                    q, swa_cache, None, None, 512, direct_meta,
                    softmax_scale=512 ** -0.5, is_fp8_kvcache=True,
                    indices=swa_indices, extra_indices_in_kvcache=physical,
                    topk_length=swa_lengths, extra_topk_length=lengths,
                    remnant_buffers=buffers, remnant_raw_indices=raw,
                    remnant_freqs=torch.view_as_real(freqs).unsqueeze(0),
                )

            graphs = {}
            if args.path in ("all", "native"):
                graphs["native"] = _capture(native, args.warmup)
            if args.path in ("all", "direct"):
                graphs["direct"] = _capture(direct, args.warmup)
            if args.path in ("all", "adapter"):
                graphs["adapter"] = _capture(adapter, args.warmup)

            samples = {name: [] for name in ("native", "direct", "adapter")}
            rng = random.Random(heads * 1000 + batch)
            for _ in range(args.rounds):
                order = list(graphs)
                rng.shuffle(order)
                for name in order:
                    samples[name].append(_measure_graph(graphs[name], args.repeats))

            native_ms = statistics.median(samples["native"]) if samples["native"] else float("nan")
            direct_ms = statistics.median(samples["direct"]) if samples["direct"] else float("nan")
            adapter_ms = statistics.median(samples["adapter"]) if samples["adapter"] else float("nan")
            if args.path == "all":
                paired = [100.0 * (d / n - 1.0) for n, d in zip(samples["native"], samples["direct"])]
                delta = statistics.median(paired)
                p95 = _percentile(paired, 0.95)
                status = "PASS" if p95 <= args.max_regression_percent else "MISS"
            else:
                delta = p95 = float("nan")
                status = "N/A"
            if status == "MISS":
                misses.append((heads, batch, delta, p95))
            print(f"{heads},{batch},512,{native_ms:.5f},{direct_ms:.5f},{adapter_ms:.5f},{delta:.3f},{p95:.3f},{status}")
    remnant.configure_cache_format("native")
    if misses:
        details = ", ".join(
            f"H{heads}/B{batch}=median {delta:.2f}%, p95 {p95:.2f}%"
            for heads, batch, delta, p95 in misses
        )
        raise RuntimeError(f"direct decode exceeds the {args.max_regression_percent:.2f}% p95 target: {details}")


if __name__ == "__main__":
    main()
