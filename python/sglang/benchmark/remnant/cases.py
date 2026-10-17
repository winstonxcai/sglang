"""Versioned model-free three-mode cases and measurement summaries.

Author: Winston Cai. No CUDA imports: shared by the benchmark and contract tests.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import random
import re
import statistics
from dataclasses import asdict, dataclass
from pathlib import Path

PROTOCOL = "remnant-three-modes-v1"
SEED = 20260919
MODES = ("native", "triton", "kernel")


@dataclass(frozen=True)
class DecodeCase:
    heads: int
    batch: int
    topk_length: int
    topk: int = 512
    swa_length: int = 64
    query_length: int = 1

    @property
    def name(self) -> str:
        return f"h{self.heads}_b{self.batch}_k{self.topk_length}"


@dataclass(frozen=True)
class StoreCase:
    phase: str
    rows: int
    layout: str = "all_valid"

    @property
    def name(self) -> str:
        return f"{self.phase}-r{self.rows}-{self.layout}"


def store_cases() -> list[StoreCase]:
    return (
        [StoreCase("decode", n) for n in (1, 8, 16, 24, 25, 32, 64, 128)]
        + [StoreCase("decode", n, "staggered") for n in (1, 25, 32, 64)]
        + [
            StoreCase("decode", 32, s)
            for s in ("no_boundary", "shuffled", "invalid_dest")
        ]
        + [StoreCase("prefill", n) for n in (1, 64, 257, 1024, 4096, 8192)]
        + [
            StoreCase("prefill", 8192, s)
            for s in ("shuffled", "padded", "invalid_dest", "all_invalid")
        ]
    )


def case_seed(case: DecodeCase | StoreCase, seed: int) -> int:
    # Local generators make a case independent of selection or iteration order.
    return int.from_bytes(
        hashlib.sha256(f"{PROTOCOL}:{seed}:{case.name}".encode()).digest()[:4], "little"
    )


def store_layout(case: StoreCase, seed: int) -> tuple[list[int], list[int], list[int]]:
    if case not in store_cases():
        raise ValueError(f"unsupported store case: {case}")
    n = case.rows
    seq, ragged, loc = [4] * n, list(range(n)), list(range(n))
    rng = random.Random(case_seed(case, seed))
    if case.layout in ("shuffled", "padded", "invalid_dest"):
        rng.shuffle(loc)
        if case.phase == "prefill":
            rng.shuffle(ragged)
    if case.layout == "staggered":
        seq = [4 + i % 4 for i in range(n)]
    elif case.layout == "no_boundary":
        seq = [5] * n
    elif case.layout == "padded":
        seq = [-1 if i % 4 == 0 else 4 for i in range(n)]
    elif case.layout == "all_invalid":
        seq = [-1] * n
    elif case.layout == "invalid_dest":
        loc = [-1 if i % 4 == 0 else v for i, v in enumerate(loc)]
    return seq, ragged, loc


def positive_ints(value: str, name: str, maximum: int | None = None) -> list[int]:
    try:
        result = [int(x) for x in value.split(",")]
    except ValueError as e:
        raise ValueError(f"{name}: expected comma-separated positive integers") from e
    if (
        not result
        or len(set(result)) != len(result)
        or any(x < 1 or (maximum and x > maximum) for x in result)
    ):
        raise ValueError(f"{name}: invalid or duplicate values {result}")
    return result


def parse_modes(value: str) -> tuple[str, ...]:
    result = tuple(value.split(","))
    if (
        not result
        or len(set(result)) != len(result)
        or any(x not in MODES for x in result)
    ):
        raise ValueError(f"--modes: choose unique modes from {MODES}")
    return result


def summary(samples: list[float]) -> dict:
    if len(samples) < 2 or any(not math.isfinite(x) or x <= 0 for x in samples):
        raise ValueError("timing requires at least two positive finite samples")
    ordered = sorted(samples)
    position = (len(samples) - 1) * 0.95
    lo, hi = math.floor(position), math.ceil(position)
    return {
        "samples": samples,
        "mean": statistics.mean(samples),
        "median": statistics.median(samples),
        "stdev": statistics.stdev(samples),
        "p95": ordered[lo] + (ordered[hi] - ordered[lo]) * (position - lo),
    }


def paired_ratios(samples: dict[str, list[float]]) -> dict:
    result = {}
    for numerator, denominator in (
        ("triton", "native"),
        ("kernel", "native"),
        ("kernel", "triton"),
    ):
        if numerator in samples and denominator in samples:
            a, b = samples[numerator], samples[denominator]
            if len(a) != len(b):
                raise ValueError("paired timings must have equal round counts")
            summary(a)
            summary(b)
            result[f"{numerator}_over_{denominator}"] = summary(
                [x / y for x, y in zip(a, b)]
            )
    return result


def profile_matrix() -> list[tuple[str, str]]:
    decode = [
        DecodeCase(h, b, k).name for h in (64, 128) for b in (8, 16) for k in (512, 317)
    ]
    return [
        (suite, name)
        for suite, names in (
            ("decode", decode),
            ("store", ["decode-r25-all_valid", "prefill-r8192-all_valid"]),
        )
        for name in names
    ]


def matrix_digest(cases: list) -> str:
    return hashlib.sha256(
        json.dumps([asdict(c) for c in cases], sort_keys=True).encode()
    ).hexdigest()


def write_results(payload: dict, base: Path) -> None:
    base.parent.mkdir(parents=True, exist_ok=True)
    base.with_suffix(".json").write_text(json.dumps(payload, indent=2) + "\n")
    with base.with_suffix(".csv").open("w", newline="") as f:
        fields = [
            "protocol",
            "suite",
            "case",
            "mode",
            "execution",
            "mean_ms",
            "median_ms",
            "p95_ms",
            "valid",
            "samples_ms",
            "paired_ratios",
            "case_definition",
            "input_sha256",
            "source",
            "environment",
        ]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for case in payload["cases"]:
            for mode, values in case.get("paths", {}).items():
                writer.writerow(
                    dict(
                        suite=case["suite"],
                        case=case["name"],
                        mode=mode,
                        execution=case["execution"],
                        mean_ms=values["mean"],
                        median_ms=values["median"],
                        p95_ms=values["p95"],
                        valid=case["valid"],
                        protocol=payload["protocol"],
                        samples_ms=json.dumps(values["samples"]),
                        paired_ratios=json.dumps(case["paired_ratios"], sort_keys=True),
                        case_definition=json.dumps(case["case"], sort_keys=True),
                        input_sha256=case["common_inputs_sha256"],
                        source=json.dumps(payload.get("source", {}), sort_keys=True),
                        environment=json.dumps(payload["environment"], sort_keys=True),
                    )
                )


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--suite", choices=("all", "decode", "store"), default="all")
    p.add_argument("--modes", default=",".join(MODES))
    p.add_argument("--heads", default="64,128")
    p.add_argument("--batches", default="8,16")
    p.add_argument("--topk-lengths", default="512,317")
    p.add_argument("--seed", type=int, default=SEED)
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--rounds", type=int, default=30)
    p.add_argument("--repeats", type=int, default=100)
    p.add_argument("--prefill-repeats", type=int, default=20)
    p.add_argument("--output")
    p.add_argument("--profile-mode", choices=MODES)
    p.add_argument("--case", help="Select one case, required for profiling")
    p.add_argument("--check-imports", action="store_true")
    p.add_argument(
        "--native-library", default=os.environ.get("REMNANT_NATIVE_STORE_LIBRARY")
    )
    return p


def selected_cases(args) -> list:
    h = positive_ints(args.heads, "--heads")
    b = positive_ints(args.batches, "--batches")
    k = positive_ints(args.topk_lengths, "--topk-lengths", 512)
    cases = []
    if args.suite in ("all", "decode"):
        cases += [
            DecodeCase(head, batch, length) for head in h for batch in b for length in k
        ]
    if args.suite in ("all", "store"):
        cases += store_cases()
    if args.case:
        cases = [c for c in cases if c.name == args.case]
        if len(cases) != 1:
            raise ValueError(f"unknown or ambiguous case: {args.case}")
    if args.profile_mode and (not args.case or len(cases) != 1):
        raise ValueError("profiling requires one --case")
    if (
        args.rounds < 2
        or min(args.repeats, args.prefill_repeats) < 1
        or args.warmup < 0
    ):
        raise ValueError(
            "rounds >= 2, positive repeats and nonnegative warmup required"
        )
    return cases


def validate_timing(payload: dict) -> None:
    """Require the full default matrix and all measured rounds before promotion."""
    args = parser().parse_args([])
    expected = {
        ("decode" if isinstance(c, DecodeCase) else "store", c.name)
        for c in selected_cases(args)
    }
    cases = payload.get("cases", [])
    actual = {(c["suite"], c["name"]) for c in cases}
    if (
        payload.get("status") != "passed"
        or actual != expected
        or len(cases) != len(expected)
        or payload.get("protocol") != PROTOCOL
        or payload.get("schema_version") != 2
    ):
        raise RuntimeError(
            "Incomplete default decode/store matrix or protocol mismatch"
        )
    for case in cases:
        if not case["valid"] or set(case.get("paths", {})) != set(MODES):
            raise RuntimeError(f"Missing valid three-mode timing: {case['name']}")
        for path in case["paths"].values():
            samples = path.get("samples", [])
            if len(samples) != args.rounds or any(
                not isinstance(x, (int, float))
                or isinstance(x, bool)
                or not math.isfinite(x)
                or x <= 0
                for x in samples
            ):
                raise RuntimeError(f"Missing or invalid timing rounds: {case['name']}")


def selected_profile_matrix(case_names: str = "") -> list[tuple[str, str, str]]:
    """Select documented profile cases without changing the timing judge."""
    selected = profile_matrix()
    if case_names:
        requested = case_names.split(",")
        catalog = {name: suite for suite, name in selected}
        catalog.update((case.name, "store") for case in store_cases())
        if len(set(requested)) != len(requested) or any(
            name not in catalog for name in requested
        ):
            raise ValueError(f"unknown or duplicate profile cases: {case_names}")
        selected = [(catalog[name], name) for name in requested]
    return [(suite, name, mode) for suite, name in selected for mode in MODES]


def profile_summary(raw_csv: str, required: list[str]) -> dict:
    import io

    rows = list(csv.reader(io.StringIO(raw_csv)))
    start = next(
        (i for i, row in enumerate(rows) if "Kernel Name" in row),
        None,
    )
    if start is None:
        raise RuntimeError("Nsight Compute export has no kernel metric table")
    fields = rows[start]
    if "Metric Name" in fields:
        metrics = [
            dict(zip(fields, row))
            for row in rows[start + 1 :]
            if len(row) == len(fields)
        ]
    else:
        # Raw-page exports use a units row followed by one wide row per launch.
        units = rows[start + 1]
        if len(units) != len(fields):
            raise RuntimeError("Nsight Compute wide export has no metric units row")
        metrics = []
        for row in rows[start + 2 :]:
            if len(row) != len(fields) or not row[fields.index("Kernel Name")]:
                continue
            launch = dict(zip(fields, row))
            for index, name in enumerate(fields):
                if "__" in name:
                    metrics.append(
                        {
                            "ID": launch.get("ID"),
                            "Kernel Name": launch["Kernel Name"],
                            "Metric Name": name,
                            "Metric Unit": units[index],
                            "Metric Value": row[index],
                        }
                    )
    kernels = sorted({r["Kernel Name"] for r in metrics})
    missing = [
        pattern
        for pattern in required
        if not any(re.search(pattern, k, re.I) for k in kernels)
    ]
    if missing:
        raise RuntimeError(
            f"Nsight Compute kernel coverage missing {missing}; observed {kernels}"
        )
    names = {r["Metric Name"] for r in metrics}
    categories = {
        "duration": r"gpu__time_duration",
        "dram": r"dram__",
        "l2": r"lts__",
        "occupancy": r"(warps_active|occupancy)",
        "registers": r"launch__registers",
        "spills": r"(local_op|local.*sector|local.*load|local.*store)",
        "stalls": r"warps?_issue_stalled",
    }
    for category, pattern in categories.items():
        if not any(re.search(pattern, n, re.I) for n in names):
            raise RuntimeError(f"Nsight Compute metric coverage missing {category}")
    selected = []
    for row in metrics:
        if any(re.search(p, row["Metric Name"], re.I) for p in categories.values()):
            try:
                value = float(row["Metric Value"].replace(",", ""))
            except (ValueError, KeyError) as error:
                raise RuntimeError(f"Non-numeric required metric: {row}") from error
            if not math.isfinite(value):
                raise RuntimeError(f"Non-finite required metric: {row}")
            selected.append(
                {
                    "kernel": row["Kernel Name"],
                    "metric": row["Metric Name"],
                    "unit": row.get("Metric Unit"),
                    "value": value,
                    "id": row.get("ID"),
                }
            )
    return {"kernels": kernels, "metrics": selected, "acceptance_timing": False}
