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
PREFILL_PROTOCOL = "remnant-prefill-read-v1"
FP4_DECODE_PROTOCOL = "remnant-v41-fp4-decode-v2"
SEED = 20260919
MODES = ("native", "triton", "kernel")
DEFAULT_DECODE_BATCHES = (8, 16, 24, 28, 32)
DEFAULT_PREFILL_QUERY_CHUNKS = (128, 512, 1536, 4096, 8192)
KEEP_COUNTS = tuple(range(64, 512, 64))
PREFILL_VARIANTS = ("native", "packed192", "packed256")
PREFILL_STAGES = ("reconstruction", "attention", "complete")
PREFILL_POLICIES = ("warm", "rotating")


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
class FP4DecodeCase(DecodeCase):
    """Production V4.1 H100 decode geometry with explicit C1/C2 pages."""

    swa_length: int = 128

    global_page_size: int = 256
    swa_page_size: int = 128

    @property
    def name(self) -> str:
        return f"v41_{super().name}_swa{self.swa_length}_p{self.swa_page_size}_g{self.global_page_size}"


def fp4_decode_cases() -> list[FP4DecodeCase]:
    return [
        # V4.1 has 64 query heads; SM90 pads smaller TP shards back to 64.
        # Its 256-token logical pages hold 256 C1 or 128 C2 compressed rows.
        FP4DecodeCase(64, b, 512, global_page_size=page)
        for page in (256, 128)
        for b in DEFAULT_DECODE_BATCHES
    ]


@dataclass(frozen=True)
class StoreCase:
    phase: str
    rows: int
    layout: str = "all_valid"

    @property
    def name(self) -> str:
        return f"{self.phase}-r{self.rows}-{self.layout}"


@dataclass(frozen=True)
class PrefillReadCase:
    heads: int
    history_length: int
    query_length: int
    stage: str = "complete"
    cache_policy: str = "warm"
    topk: int = 512
    swa_length: int = 128
    swa_page_size: int = 128

    def __post_init__(self) -> None:
        if (
            any(
                type(x) is not int
                for x in (
                    self.heads,
                    self.history_length,
                    self.query_length,
                    self.topk,
                    self.swa_length,
                    self.swa_page_size,
                )
            )
            or self.heads not in (64, 128)
            or self.query_length < 1
            or self.history_length < max(2048, self.query_length + 127)
            or self.topk != 512
            or self.swa_length != 128
            or self.swa_page_size not in (64, 128, 256)
            or self.stage not in PREFILL_STAGES
            or self.cache_policy not in PREFILL_POLICIES
        ):
            raise ValueError(f"unsupported prefill-read case: {self}")

    @property
    def fixture_name(self) -> str:
        # Stage/cache policy must not change shared numerical inputs.
        return f"prefill_h{self.heads}_s{self.history_length}_q{self.query_length}"

    @property
    def name(self) -> str:
        return f"{self.fixture_name}_{self.cache_policy}_{self.stage}"

    @property
    def phase(self) -> str:
        return "prefill"


def prefill_read_cases(
    heads: tuple[int, ...] = (64, 128),
    histories: tuple[int, ...] = (143360,),
    queries: tuple[int, ...] = DEFAULT_PREFILL_QUERY_CHUNKS,
) -> list[PrefillReadCase]:
    return [
        PrefillReadCase(h, s, q, stage, policy)
        for h in heads
        for s in histories
        for q in queries
        for policy in PREFILL_POLICIES
        for stage in PREFILL_STAGES
    ]


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


def case_seed(case: DecodeCase | StoreCase | PrefillReadCase, seed: int) -> int:
    # Local generators make a case independent of selection or iteration order.
    protocol, name = (
        (PREFILL_PROTOCOL, case.fixture_name)
        if isinstance(case, PrefillReadCase)
        else (PROTOCOL, case.name)
    )
    return int.from_bytes(
        hashlib.sha256(f"{protocol}:{seed}:{name}".encode()).digest()[:4], "little"
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


def benchmark_modes(keep_count: int = 256, modes: str | None = None) -> tuple[str, ...]:
    """Resolve supported paths before allocating or capturing GPU buffers."""
    if type(keep_count) is not int or keep_count not in KEEP_COUNTS:
        raise ValueError(f"--keep-count must be one of {KEEP_COUNTS}; 512 is invalid")
    result = (
        parse_modes(modes)
        if modes is not None
        else (MODES if keep_count == 256 else ("native", "triton"))
    )
    if keep_count != 256 and "kernel" in result:
        raise ValueError(
            "Kernel requires --keep-count 256; use native,triton otherwise"
        )
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
        ("packed192", "native"),
        ("packed256", "native"),
        ("packed192", "packed256"),
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
        DecodeCase(h, b, 512).name for h in (64, 128) for b in DEFAULT_DECODE_BATCHES
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
            "implementation",
            "packed_keep_count",
            "record_bytes",
            "execution",
            "prefill_reader",
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
                layout = case.get("path_layouts", {}).get(mode, {})
                writer.writerow(
                    dict(
                        suite=case["suite"],
                        case=case["name"],
                        mode=mode,
                        implementation=layout.get("implementation", mode),
                        packed_keep_count=layout.get(
                            "packed_keep_count", case.get("packed_keep_count", 256)
                        ),
                        record_bytes=(
                            layout.get(
                                "record_bytes",
                                (
                                    584
                                    if mode == "native"
                                    else case.get("packed_record_bytes", 328)
                                ),
                            )
                        ),
                        execution=case["execution"],
                        prefill_reader=(
                            case.get("prefill_reader", "split")
                            if case["suite"] == "prefill-read"
                            else ""
                        ),
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
    p.add_argument(
        "--suite",
        choices=("all", "decode", "store", "prefill-read", "v41-fp4-decode"),
        default="all",
    )
    p.add_argument("--modes", help="Default: all modes at 256, native,triton otherwise")
    p.add_argument(
        "--keep-count",
        type=int,
        choices=KEEP_COUNTS,
        default=256,
        help="Retained coordinates per C4 record; independent of --topk-lengths",
    )
    p.add_argument("--heads", default="64,128")
    p.add_argument("--batches", default=",".join(map(str, DEFAULT_DECODE_BATCHES)))
    p.add_argument("--topk-lengths", default="512")
    p.add_argument("--histories", default="143360")
    p.add_argument(
        "--query-chunks",
        default=",".join(map(str, DEFAULT_PREFILL_QUERY_CHUNKS)),
        help="Per-request uncached query tokens, not the server's chunked-prefill ceiling",
    )
    p.add_argument(
        "--prefill-reader",
        choices=("split", "fused"),
        default="split",
        help="Opt-in Packed unpack/RoPE candidate; only for --suite prefill-read",
    )
    p.add_argument("--seed", type=int, default=SEED)
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--rounds", type=int, default=30)
    p.add_argument("--repeats", type=int, default=100)
    p.add_argument("--prefill-repeats", type=int, default=20)
    p.add_argument("--output")
    p.add_argument("--profile-mode", choices=MODES)
    p.add_argument("--profile-path", choices=PREFILL_VARIANTS)
    p.add_argument(
        "--profile-replay-id",
        help="Owning profiler identity for deterministic application replay",
    )
    p.add_argument("--case", help="Select one case, required for profiling")
    p.add_argument("--check-imports", action="store_true")
    p.add_argument(
        "--native-library", default=os.environ.get("REMNANT_NATIVE_STORE_LIBRARY")
    )
    return p


def selected_cases(args) -> list:
    if args.suite == "prefill-read":
        if args.modes is not None or args.keep_count != 256 or args.profile_mode:
            raise ValueError(
                "prefill-read compares native/packed192/packed256 together; use --profile-path"
            )
    elif args.profile_path:
        raise ValueError("--profile-path requires --suite prefill-read")
    if args.suite != "prefill-read" and args.prefill_reader != "split":
        raise ValueError("--prefill-reader fused requires --suite prefill-read")
    if args.profile_replay_id and not args.profile_path:
        raise ValueError("--profile-replay-id requires a prefill --profile-path")
    modes = benchmark_modes(args.keep_count, args.modes)
    if args.profile_mode and args.profile_mode not in modes:
        raise ValueError("--profile-mode must be included in the supported --modes")
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
    if args.suite == "prefill-read":
        cases = prefill_read_cases(
            tuple(h),
            tuple(positive_ints(args.histories, "--histories")),
            tuple(positive_ints(args.query_chunks, "--query-chunks")),
        )
    if args.suite == "v41-fp4-decode":
        if args.modes not in (None, "native,kernel") or args.keep_count != 256:
            raise ValueError("V4.1 FP4 compares dense Native and Packed Direct keep256")
        if args.profile_mode or args.profile_path:
            raise ValueError("V4.1 FP4 profiling is deferred")
        cases = fp4_decode_cases()
    if args.case:
        requested = args.case.split(",")
        if len(set(requested)) != len(requested) or set(requested) - {
            c.name for c in cases
        }:
            raise ValueError(f"unknown or ambiguous case: {args.case}")
        cases = [c for c in cases if c.name in requested]
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


def verify_profile_replay(payload: dict, config: dict) -> None:
    """Permit only the owning deterministic NCU application replay to reuse output."""
    if (
        not config.get("profile_replay_id")
        or not config.get("profile_path")
        or config.get("suite") != "prefill-read"
        or payload.get("protocol") != PREFILL_PROTOCOL
        or payload.get("config") != config
        or payload.get("status") != "passed"
        or not payload.get("cases")
        or len({item.get("name") for item in payload["cases"]}) != len(payload["cases"])
        or any(
            not item.get("valid") or item.get("path") != config["profile_path"]
            for item in payload["cases"]
        )
    ):
        raise FileExistsError(
            "Existing output is not the same validated profiler application replay"
        )


def validate_timing(
    payload: dict,
    *,
    keep_count: int = 256,
    modes: str | None = None,
    suite: str = "all",
    prefill_reader: str = "split",
) -> None:
    """Require the full default matrix and all measured rounds before promotion."""
    if suite == "v41-fp4-decode":
        definitions = {c.name: asdict(c) for c in fp4_decode_cases()}
        records = payload.get("cases", [])
        if (
            payload.get("status") != "passed"
            or payload.get("protocol") != FP4_DECODE_PROTOCOL
            or len(records) != len(definitions)
            or {r["name"] for r in records} != set(definitions)
            or payload.get("config", {}).get("warmup") != 10
            or payload["config"].get("rounds") != 30
            or payload["config"].get("repeats") != 100
        ):
            raise RuntimeError("Incomplete V4.1 FP4 timing matrix/protocol")
        for record in records:
            if (
                not record.get("valid")
                or record.get("case") != definitions[record["name"]]
                or record.get("execution") != "cuda_graph"
                or record.get("repeats") != 100
                or set(record.get("paths", {})) != {"native", "kernel"}
                or not re.fullmatch(
                    r"[a-f0-9]{64}", record.get("common_inputs_sha256", "")
                )
                or record.get("path_layouts")
                != {
                    "native": {
                        "implementation": "Native FP4 Direct",
                        "record_bytes": 288,
                        "packed_keep_count": 512,
                    },
                    "kernel": {
                        "implementation": "Packed FP4 Direct",
                        "record_bytes": 224,
                        "packed_keep_count": 256,
                    },
                }
            ):
                raise RuntimeError(f"Invalid V4.1 FP4 case: {record['name']}")
            for path in record["paths"].values():
                if len(path.get("samples", [])) != 30:
                    raise RuntimeError(f"Missing FP4 rounds: {record['name']}")
                summary(path["samples"])
        return
    if suite not in ("all", "prefill-read"):
        raise ValueError(f"unsupported timing judge suite: {suite}")
    if prefill_reader not in ("split", "fused") or (
        suite != "prefill-read" and prefill_reader != "split"
    ):
        raise ValueError("prefill_reader fused requires suite prefill-read")
    args = parser().parse_args([])
    args.suite = suite
    requested_modes = (
        PREFILL_VARIANTS
        if suite == "prefill-read"
        else benchmark_modes(keep_count, modes)
    )
    definitions = {c.name: c for c in selected_cases(args)}
    expected = {(case_suite(c), c.name) for c in definitions.values()}
    cases = payload.get("cases", [])
    actual = {(c["suite"], c["name"]) for c in cases}
    if (
        payload.get("status") != "passed"
        or actual != expected
        or len(cases) != len(expected)
        or payload.get("protocol")
        != (PREFILL_PROTOCOL if suite == "prefill-read" else PROTOCOL)
        or payload.get("schema_version") != 2
        or payload.get("config", {}).get("keep_count", 256) != keep_count
        or (
            suite == "prefill-read"
            and payload.get("config", {}).get("prefill_reader", "split")
            != prefill_reader
        )
    ):
        raise RuntimeError(
            "Incomplete default decode/store matrix or protocol mismatch"
        )
    fixture_hashes = {}
    for case in cases:
        if (
            not case["valid"]
            or set(case.get("paths", {})) != set(requested_modes)
            or (
                suite != "prefill-read"
                and (
                    case.get("packed_keep_count", 256) != keep_count
                    or case.get("packed_record_bytes", 328) != keep_count + 72
                )
            )
        ):
            raise RuntimeError(f"Missing valid mode/count timing: {case['name']}")
        if (
            suite == "prefill-read"
            and case.get("path_layouts") != prefill_path_layouts()
        ):
            raise RuntimeError(f"Wrong prefill variant layouts: {case['name']}")
        if suite == "prefill-read":
            if case.get("prefill_reader", "split") != prefill_reader:
                raise RuntimeError(f"Prefill reader mismatch: {case['name']}")
            definition = definitions[case["name"]]
            digest = case.get("common_inputs_sha256", "")
            if (
                case.get("case") != asdict(definition)
                or not re.fullmatch(r"[a-f0-9]{64}", digest)
                or fixture_hashes.setdefault(definition.fixture_name, digest) != digest
            ):
                raise RuntimeError(
                    f"Prefill case or shared-input identity mismatch: {case['name']}"
                )
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


def case_suite(case: DecodeCase | StoreCase | PrefillReadCase) -> str:
    if isinstance(case, PrefillReadCase):
        return "prefill-read"
    return "decode" if isinstance(case, DecodeCase) else "store"


def prefill_path_layouts() -> dict:
    return {
        "native": {
            "implementation": "native",
            "packed_keep_count": None,
            "record_bytes": 584,
        },
        "packed192": {
            "implementation": "triton",
            "packed_keep_count": 192,
            "record_bytes": 264,
        },
        "packed256": {
            "implementation": "triton",
            "packed_keep_count": 256,
            "record_bytes": 328,
        },
    }


def selected_profile_matrix(
    case_names: str = "",
    keep_count: int = 256,
    modes: str | None = None,
    suite: str = "all",
) -> list[tuple[str, str, str]]:
    """Select documented profile cases without changing the timing judge."""
    if suite not in ("all", "prefill-read"):
        raise ValueError(f"unsupported profile suite: {suite}")
    requested_modes = (
        PREFILL_VARIANTS
        if suite == "prefill-read"
        else benchmark_modes(keep_count, modes)
    )
    if suite == "prefill-read":
        # The new suite profiles every timed shape/stage/cache policy.
        catalog = {c.name: "prefill-read" for c in prefill_read_cases()}
        selected = [("prefill-read", c.name) for c in prefill_read_cases()]
    else:
        selected = profile_matrix()
        catalog = {name: kind for kind, name in selected}
        catalog.update((case.name, "store") for case in store_cases())
    if case_names:
        requested = case_names.split(",")
        if len(set(requested)) != len(requested) or any(
            name not in catalog for name in requested
        ):
            raise ValueError(f"unknown or duplicate profile cases: {case_names}")
        selected = [(catalog[name], name) for name in requested]
    return [(suite, name, mode) for suite, name in selected for mode in requested_modes]


def profile_summary(
    raw_csv: str,
    required: list[str],
    *,
    traffic_only: bool = False,
    minimum_launches: dict[str, int] | None = None,
) -> dict:
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
    for pattern, minimum in (minimum_launches or {}).items():
        launches = {
            r.get("ID") for r in metrics if re.search(pattern, r["Kernel Name"], re.I)
        }
        if None in launches or len(launches) < minimum:
            raise RuntimeError(
                f"Nsight Compute launch coverage missing {pattern}: {len(launches)} < {minimum}"
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
    if traffic_only:
        categories = {
            "duration": r"^gpu__time_duration.sum$",
            "dram_read": r"^dram__bytes_read.sum$",
            "dram_write": r"^dram__bytes_write.sum$",
            "l2_read": r"^lts__t_sectors_op_read.sum$",
            "l2_write": r"^lts__t_sectors_op_write.sum$",
            "l2_hit": r"^lts__t_sector_hit_rate.pct$",
        }
    for category, pattern in categories.items():
        if not any(re.search(pattern, n, re.I) for n in names):
            raise RuntimeError(f"Nsight Compute metric coverage missing {category}")
    if traffic_only:
        launches = {(r.get("ID"), r["Kernel Name"]) for r in metrics}
        for identity in launches:
            launch_metrics = {
                r["Metric Name"]
                for r in metrics
                if (r.get("ID"), r["Kernel Name"]) == identity
            }
            missing = [
                category
                for category, pattern in categories.items()
                if not any(re.search(pattern, n, re.I) for n in launch_metrics)
            ]
            if missing:
                raise RuntimeError(
                    f"Nsight Compute per-launch metric coverage missing {missing}: {identity}"
                )
    selected = []
    for row in metrics:
        if any(re.search(p, row["Metric Name"], re.I) for p in categories.values()):
            try:
                value = float(row["Metric Value"].replace(",", ""))
            except (ValueError, KeyError) as error:
                raise RuntimeError(f"Non-numeric required metric: {row}") from error
            if not math.isfinite(value):
                raise RuntimeError(f"Non-finite required metric: {row}")
            if row["Metric Name"] == "gpu__time_duration.sum" and value <= 0:
                raise RuntimeError(f"Non-positive GPU duration: {row}")
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
