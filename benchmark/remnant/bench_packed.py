"""Benchmark Native and Remnant C4 write/reconstruction primitives."""

from __future__ import annotations

import argparse
import time

import torch

from sglang.srt import remnant
from sglang.srt.remnant import config
from sglang.srt.remnant.packed import pack_rows, unpack_gather_bf16
from sglang.srt.remnant.reference import topmag_keep_mask


def _measure(fn, repeats: int, warmup: int) -> float:
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


def _native_write(latent, keep_mask, norm_weight, output, norm_eps):
    def run():
        masked = latent.masked_fill(~keep_mask, 0.0)
        normalized = masked * torch.rsqrt(
            masked.square().mean(-1, keepdim=True) + norm_eps
        ) * norm_weight
        output[:, :448].copy_(
            normalized[:, :448].to(torch.float8_e4m3fn).view(torch.uint8)
        )
        output[:, 448:576].copy_(
            normalized[:, 448:].to(torch.bfloat16).view(torch.uint8)
        )
        output[:, 576:].zero_()

    return run


def _packed_write(latent, keep_mask, norm_weight, buffers, locations, norm_eps):
    class Plan:
        is_decode = False

        def __getitem__(self, index):
            if index == 2:
                return self.plan_w
            raise IndexError(index)

    plan = Plan()
    plan.plan_w = torch.zeros(
        (latent.shape[0], 8), dtype=torch.uint8, device=latent.device
    )
    plan.plan_w[:, :4] = torch.arange(
        latent.shape[0], dtype=torch.int32, device=latent.device
    ).view(torch.uint8).reshape(latent.shape[0], 4)

    def run():
        pack_rows(
            latent,
            keep_mask,
            norm_weight,
            norm_eps,
            plan,
            locations,
            buffers,
        )

    return run


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, default=1024)
    parser.add_argument("--repeats", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=10)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("bench_packed.py requires CUDA")
    if args.rows <= 0 or args.repeats <= 0 or args.warmup < 0:
        raise ValueError("rows and repeats must be positive; warmup must be nonnegative")

    device = torch.device("cuda")
    torch.manual_seed(0)
    latent = torch.randn((args.rows, remnant.HEAD_DIM), device=device)
    keep_mask = topmag_keep_mask(latent, 0.5)
    norm_weight = torch.ones(remnant.HEAD_DIM, device=device)
    native_output = torch.empty(
        (args.rows, config.NATIVE_RECORD_BYTES), dtype=torch.uint8, device=device
    )
    locations = torch.arange(args.rows, dtype=torch.int32, device=device)
    packed_buffers = (
        torch.empty((args.rows, 256, 256), dtype=torch.uint8, device=device),
        torch.empty((args.rows, 256, 8), dtype=torch.uint64, device=device),
        torch.empty((args.rows, 256, 8), dtype=torch.uint8, device=device),
    )
    physical = locations.view(args.rows, 1)
    raw = locations.view(args.rows, 1)
    lengths = torch.ones(args.rows, dtype=torch.int32, device=device)
    freqs = torch.ones(
        args.rows * 4 * (config.ROPE_DIM // 2) + config.ROPE_DIM // 2,
        dtype=torch.complex64,
        device=device,
    )
    reconstructed = torch.empty(
        (args.rows, 512), dtype=torch.bfloat16, device=device
    )

    remnant.configure_cache_format("native")
    native_ms = _measure(
        _native_write(latent, keep_mask, norm_weight, native_output, 1e-6),
        args.repeats,
        args.warmup,
    )
    remnant.configure_cache_format("remnant")
    packed_ms = _measure(
        _packed_write(
            latent, keep_mask, norm_weight, packed_buffers, locations, 1e-6
        ),
        args.repeats,
        args.warmup,
    )
    reconstruct_ms = _measure(
        lambda: unpack_gather_bf16(
            packed_buffers, physical, raw, lengths, freqs, reconstructed
        ),
        args.repeats,
        args.warmup,
    )
    remnant.configure_cache_format("native")

    print(f"rows={args.rows} repeats={args.repeats}")
    print(f"native_write_ms={native_ms:.4f}")
    print(f"packed_write_ms={packed_ms:.4f}")
    print(f"packed_selected_row_reconstruct_ms={reconstruct_ms:.4f}")


if __name__ == "__main__":
    main()
