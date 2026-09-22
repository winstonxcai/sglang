"""Model-free Native, Adapter, and direct Remnant FlashMLA microbenchmark.

Author: Winston Cai.

The benchmark uses an explicit decode-case matrix, CUDA Graph replay, and
timestamped JSON/CSV output.  The JSON file is the canonical artifact; the CSV
is a flat summary for quick inspection.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import math
import os
import platform
import random
import statistics
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Optional

import torch

from sgl_kernel import flash_mla
from sglang.srt import remnant
from sglang.srt.remnant import NativeWorkspace
from sglang.srt.remnant.packed import pack_rows, unpack_gather_native
from sglang.srt.remnant.reference import topmag_keep_mask


@dataclass(frozen=True)
class DecodeCase:
    heads: int
    batch: int
    topk_length: int
    topk: int = 512
    swa_length: int = 64
    query_length: int = 1
    dtype: str = "bfloat16"

    @property
    def name(self) -> str:
        return f"h{self.heads}_b{self.batch}_k{self.topk_length}"


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
    table = torch.zeros((max_position * 4, 32, 2), device=device)
    table[::4, :, 0] = torch.cos(angle)
    table[::4, :, 1] = torch.sin(angle)
    return torch.view_as_complex(table).contiguous()


def _make_swa_cache(pages: int, page_size: int, device: torch.device) -> torch.Tensor:
    """Populate valid MODEL1 pages with values followed by page scale bytes."""
    bytes_per_page = ((page_size * 584 + 575) // 576) * 576
    storage = torch.zeros((pages, bytes_per_page), dtype=torch.uint8, device=device)
    values = storage[:, : page_size * 576].view(pages, page_size, 576)
    rows = pages * page_size
    codes = (torch.arange(rows * 448, device=device) % 113 + 1).to(torch.uint8)
    values[:, :, :448].copy_(codes.view(pages, page_size, 448))
    tail = torch.linspace(-0.25, 0.25, rows * 64, device=device).to(torch.bfloat16)
    values[:, :, 448:576].copy_(tail.view(torch.uint8).reshape(pages, page_size, 128))
    storage[:, page_size * 576 : page_size * 584].fill_(127)
    return storage.as_strided(
        (pages, page_size, 1, 584), (bytes_per_page, 584, 584, 1)
    )


def _capture(fn: Callable[[], object], warmup: int) -> torch.cuda.CUDAGraph:
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


def _percentile(values: list[float], fraction: float) -> float:
    if not values:
        raise ValueError("cannot calculate a percentile of an empty sample")
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _summary(values: list[float]) -> Optional[dict[str, object]]:
    if not values:
        return None
    return {
        "samples": [round(value, 8) for value in values],
        "median": statistics.median(values),
        "mean": statistics.mean(values),
        "stdev": statistics.stdev(values) if len(values) > 1 else 0.0,
        "p95": _percentile(values, 0.95),
    }


def _parse_ints(value: str, name: str) -> list[int]:
    try:
        values = [int(item.strip()) for item in value.split(",") if item.strip()]
    except ValueError as exc:
        raise ValueError(f"{name} must be a comma-separated integer list") from exc
    if not values:
        raise ValueError(f"{name} must contain at least one integer")
    return values


def _build_cases(args: argparse.Namespace) -> list[DecodeCase]:
    heads = _parse_ints(args.heads, "--heads")
    batches = _parse_ints(args.batches, "--batches")
    topk_lengths = _parse_ints(args.topk_lengths, "--topk-lengths")
    if any(heads <= 0 for heads in heads):
        raise ValueError("--heads values must be positive")
    if any(batch <= 0 for batch in batches):
        raise ValueError("--batches values must be positive")
    if any(length <= 0 or length > 512 for length in topk_lengths):
        raise ValueError("--topk-lengths values must be in [1, 512]")
    return [
        DecodeCase(heads=head, batch=batch, topk_length=topk_length)
        for head in heads
        for batch in batches
        for topk_length in topk_lengths
    ]


def _run_case(
    case: DecodeCase,
    *,
    path: str,
    repeats: int,
    warmup: int,
    rounds: int,
    max_regression_percent: float,
    device: torch.device,
) -> dict[str, object]:
    selected_k = case.topk
    rows = case.batch * selected_k
    q = torch.randn(
        (case.batch, case.query_length, case.heads, 512),
        device=device,
        dtype=torch.bfloat16,
    )
    swa_cache = _make_swa_cache(case.batch, case.swa_length, device)
    swa_indices = torch.arange(
        case.swa_length, dtype=torch.int32, device=device
    ).view(1, 1, case.swa_length)
    swa_indices = (
        swa_indices
        + torch.arange(case.batch, device=device, dtype=torch.int32).view(case.batch, 1, 1)
        * case.swa_length
    ).contiguous()
    swa_lengths = torch.full(
        (case.batch,), case.swa_length, dtype=torch.int32, device=device
    )
    physical = torch.arange(rows, dtype=torch.int32, device=device).view(
        case.batch, 1, selected_k
    )
    raw = (
        torch.arange(rows, dtype=torch.int32, device=device) * 3 + 17
    ).view(case.batch, 1, selected_k)
    lengths = torch.full(
        (case.batch,), case.topk_length, dtype=torch.int32, device=device
    )
    if case.topk_length < selected_k:
        physical[:, :, case.topk_length:] = 0
        raw[:, :, case.topk_length:] = -1

    latent = torch.randn((rows, 512), device=device)
    mask = topmag_keep_mask(latent, 0.5)
    pages = rows // 64
    buffers = (
        torch.zeros((pages, 64, 256), dtype=torch.uint8, device=device),
        torch.zeros((pages, 64, 8), dtype=torch.uint64, device=device),
        torch.zeros((pages, 64, 8), dtype=torch.uint8, device=device),
    )
    pack_rows(
        latent,
        mask,
        torch.ones(512, device=device),
        1.0e-6,
        _PackPlan(rows, device),
        torch.arange(rows, dtype=torch.int32, device=device),
        buffers,
    )
    freqs = _frequencies(int(raw.clamp_min(0).max().item()) + 2, device)
    run_native = path in ("all", "native")
    run_adapter = path in ("all", "adapter")
    run_direct = path in ("all", "direct")
    workspace = (
        NativeWorkspace.allocate(case.batch, selected_k, 64, device)
        if run_native or run_adapter
        else None
    )

    native_cache = native_indices = None
    if run_native:
        assert workspace is not None
        native_bytes, native_indices = unpack_gather_native(
            buffers,
            physical.flatten(0, 1),
            raw.flatten(0, 1),
            lengths,
            freqs,
            workspace,
        )
        native_cache = native_bytes.as_strided(
            (native_bytes.shape[0], 64, 1, 584),
            (workspace.bytes_per_page, 584, 584, 1),
        )
        native_indices = native_indices.unsqueeze(1)

    native_meta = flash_mla.get_mla_metadata()[0] if run_native else None
    adapter_meta = flash_mla.get_mla_metadata()[0] if run_adapter else None
    direct_meta = flash_mla.get_mla_metadata()[0] if run_direct else None

    def native():
        return flash_mla.flash_mla_with_kvcache(
            q,
            swa_cache,
            None,
            None,
            512,
            native_meta,
            softmax_scale=512**-0.5,
            is_fp8_kvcache=True,
            indices=swa_indices,
            extra_k_cache=native_cache,
            extra_indices_in_kvcache=native_indices,
            topk_length=swa_lengths,
            extra_topk_length=lengths,
        )

    def adapter():
        assert workspace is not None
        adapter_bytes, adapter_indices = unpack_gather_native(
            buffers,
            physical.flatten(0, 1),
            raw.flatten(0, 1),
            lengths,
            freqs,
            workspace,
        )
        adapter_cache = adapter_bytes.as_strided(
            (adapter_bytes.shape[0], 64, 1, 584),
            (workspace.bytes_per_page, 584, 584, 1),
        )
        return flash_mla.flash_mla_with_kvcache(
            q,
            swa_cache,
            None,
            None,
            512,
            adapter_meta,
            softmax_scale=512**-0.5,
            is_fp8_kvcache=True,
            indices=swa_indices,
            extra_k_cache=adapter_cache,
            extra_indices_in_kvcache=adapter_indices.unsqueeze(1),
            topk_length=swa_lengths,
            extra_topk_length=lengths,
        )

    def direct():
        return flash_mla.flash_mla_with_kvcache(
            q,
            swa_cache,
            None,
            None,
            512,
            direct_meta,
            softmax_scale=512**-0.5,
            is_fp8_kvcache=True,
            indices=swa_indices,
            extra_indices_in_kvcache=physical,
            topk_length=swa_lengths,
            extra_topk_length=lengths,
            remnant_buffers=buffers,
            remnant_raw_indices=raw,
            remnant_freqs=torch.view_as_real(freqs),
        )

    outputs: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
    if run_native:
        outputs["native"] = native()
    if run_direct:
        outputs["direct"] = direct()
    if run_adapter:
        outputs["adapter"] = adapter()
    for name, (output, lse) in outputs.items():
        if not torch.isfinite(output).all() or not torch.isfinite(lse).all():
            raise RuntimeError(f"non-finite {name} output at {case.name}")
    if path == "all":
        torch.testing.assert_close(outputs["direct"][0], outputs["native"][0], atol=2e-2, rtol=2e-2)
        torch.testing.assert_close(outputs["direct"][1], outputs["native"][1], atol=2e-2, rtol=2e-2)
        torch.testing.assert_close(outputs["adapter"][0], outputs["native"][0], atol=2e-2, rtol=2e-2)
        torch.testing.assert_close(outputs["adapter"][1], outputs["native"][1], atol=2e-2, rtol=2e-2)

    graphs: dict[str, torch.cuda.CUDAGraph] = {}
    if run_native:
        graphs["native"] = _capture(native, warmup)
    if run_direct:
        graphs["direct"] = _capture(direct, warmup)
    if run_adapter:
        graphs["adapter"] = _capture(adapter, warmup)

    samples = {name: [] for name in graphs}
    rng = random.Random(case.heads * 1000 + case.batch * 10 + case.topk_length)
    for _ in range(rounds):
        order = list(graphs)
        rng.shuffle(order)
        for name in order:
            samples[name].append(_measure_graph(graphs[name], repeats))

    summaries = {name: _summary(values) for name, values in samples.items()}
    comparison = None
    status = "N/A"
    if path == "all":
        paired = [
            100.0 * (direct_ms / native_ms - 1.0)
            for native_ms, direct_ms in zip(samples["native"], samples["direct"])
        ]
        comparison = {"direct_vs_native_percent": _summary(paired)}
        status = (
            "PASS"
            if comparison["direct_vs_native_percent"]["p95"] <= max_regression_percent
            else "MISS"
        )

    record = {
        "name": case.name,
        "case": asdict(case),
        "paths": summaries,
        "comparison": comparison,
        "status": status,
    }
    native_ms = summaries.get("native")
    direct_ms = summaries.get("direct")
    adapter_ms = summaries.get("adapter")
    regression = comparison["direct_vs_native_percent"] if comparison else None
    print(
        f"{case.name},{case.heads},{case.batch},{case.topk_length},"
        f"{native_ms['median'] if native_ms else 'NA'},"
        f"{direct_ms['median'] if direct_ms else 'NA'},"
        f"{adapter_ms['median'] if adapter_ms else 'NA'},"
        f"{regression['median'] if regression else 'NA'},"
        f"{regression['p95'] if regression else 'NA'},{status}"
    )
    return record


def _output_paths(output: Optional[str]) -> tuple[Path, Path]:
    if output:
        base = Path(output)
        if base.suffix.lower() in {".json", ".csv"}:
            base = base.with_suffix("")
    else:
        stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        base = Path(os.environ.get("REMNANT_RESULTS_DIR", ".")) / f"microbench-{stamp}"
    return base.with_suffix(".json"), base.with_suffix(".csv")


def _write_results(payload: dict[str, object], json_path: Path, csv_path: Path) -> None:
    json_path.parent.mkdir(parents=True, exist_ok=True)
    with json_path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2)
        file.write("\n")
    with csv_path.open("w", newline="", encoding="utf-8") as file:
        fields = [
            "name",
            "heads",
            "batch",
            "topk_length",
            "native_median_ms",
            "direct_median_ms",
            "adapter_median_ms",
            "direct_vs_native_median_percent",
            "direct_vs_native_p95_percent",
            "status",
        ]
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for record in payload["cases"]:
            case = record["case"]
            paths = record["paths"]
            comparison = record["comparison"]
            regression = comparison["direct_vs_native_percent"] if comparison else None
            writer.writerow(
                {
                    "name": record["name"],
                    "heads": case["heads"],
                    "batch": case["batch"],
                    "topk_length": case["topk_length"],
                    "native_median_ms": paths.get("native", {}).get("median", ""),
                    "direct_median_ms": paths.get("direct", {}).get("median", ""),
                    "adapter_median_ms": paths.get("adapter", {}).get("median", ""),
                    "direct_vs_native_median_percent": regression.get("median", "") if regression else "",
                    "direct_vs_native_p95_percent": regression.get("p95", "") if regression else "",
                    "status": record["status"],
                }
            )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--heads", default="64,128")
    parser.add_argument("--batches", default="8,16")
    parser.add_argument("--topk-lengths", default="512,317")
    parser.add_argument("--repeats", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--rounds", type=int, default=30)
    parser.add_argument("--max-regression-percent", type=float, default=2.0)
    parser.add_argument("--path", choices=("all", "native", "adapter", "direct"), default="all")
    parser.add_argument("--output", help="Output basename or .json/.csv path")
    args = parser.parse_args()
    if args.repeats <= 0 or args.rounds < 2 or args.warmup < 0:
        raise ValueError("repeats must be positive, rounds must be at least 2, and warmup nonnegative")
    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 9:
        raise RuntimeError("This benchmark requires an H100")

    cases = _build_cases(args)
    torch.manual_seed(20260919)
    torch.cuda.manual_seed_all(20260919)
    torch.cuda.set_device(0)
    device = torch.device("cuda")
    json_path, csv_path = _output_paths(args.output)
    started_at = dt.datetime.now(dt.timezone.utc)
    payload = {
        "schema_version": 1,
        "started_at": started_at.isoformat().replace("+00:00", "Z"),
        "source": {
            "parent": os.environ.get("REMNANT_PARENT_SHA", "unknown"),
            "sglang": os.environ.get("REMNANT_SGLANG_SHA", "unknown"),
            "flashmla": os.environ.get("REMNANT_FLASHMLA_SHA", "unknown"),
        },
        "environment": {
            "hostname": platform.node(),
            "python": sys.version,
            "pytorch": torch.__version__,
            "cuda": torch.version.cuda,
            "device": torch.cuda.get_device_name(0),
            "capability": list(torch.cuda.get_device_capability(0)),
        },
        "config": {
            "heads": args.heads,
            "batches": args.batches,
            "topk_lengths": args.topk_lengths,
            "repeats": args.repeats,
            "warmup": args.warmup,
            "rounds": args.rounds,
            "path": args.path,
            "max_regression_percent": args.max_regression_percent,
        },
        "cases": [],
    }
    misses = []
    print("case,heads,batch,topk_length,native_median_ms,direct_median_ms,adapter_median_ms,direct_vs_native_median_percent,direct_vs_native_p95_percent,status")
    remnant.configure_cache_format("remnant")
    try:
        with torch.inference_mode():
            for case in cases:
                record = _run_case(
                    case,
                    path=args.path,
                    repeats=args.repeats,
                    warmup=args.warmup,
                    rounds=args.rounds,
                    max_regression_percent=args.max_regression_percent,
                    device=device,
                )
                payload["cases"].append(record)
                if record["status"] == "MISS":
                    misses.append(record)
    finally:
        remnant.configure_cache_format("native")
    payload["finished_at"] = dt.datetime.now(dt.timezone.utc).isoformat().replace("+00:00", "Z")
    _write_results(payload, json_path, csv_path)
    print(f"json_output={json_path}")
    print(f"csv_output={csv_path}")
    if misses:
        details = ", ".join(
            f"{record['name']} p95={record['comparison']['direct_vs_native_percent']['p95']:.2f}%"
            for record in misses
        )
        raise RuntimeError(f"direct decode exceeds the {args.max_regression_percent:.2f}% p95 target: {details}")


if __name__ == "__main__":
    main()
