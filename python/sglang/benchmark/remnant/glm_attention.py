"""Model-free Native FP8 / Packed Direct TileLang attention protocol.

Author: Winston Cai.
CPU case/selection contracts import no CUDA modules. GPU fixtures own all cache,
query, selection and graph allocations; no producer or metadata work is timed.
Consumer: benchmark/remnant/microbench.py --model-family glm53 --suite glm-attention.
"""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import importlib.metadata
import json
import math
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path

PROTOCOL = "remnant-glm-attention-v1"
DECODE_PROTOCOL = "remnant-glm-decode-v1"
MODES = ("native", "direct")
SEED = 20261024
HISTORY, PAGE, DIM, KEEP, COLUMNS, GROUPS = 131072, 256, 512, 256, 2112, 512
SCALE, ATOL, RTOL = 256**-0.5, 0.02, 0.02
ROUNDS = 30


@dataclass(frozen=True)
class Case:
    phase: str
    heads: int
    queries: int
    tail: int
    locality: str
    policy: str

    @property
    def name(self) -> str:
        return f"{self.phase}_h{self.heads}_q{self.queries}_t{self.tail}_{self.locality}_{self.policy}"

    @property
    def requests(self) -> int:
        return self.queries if self.phase == "decode" else 1

    @property
    def request_slots(self) -> int:
        # Common 128K completed prefix, up to three pre-existing live tokens,
        # contiguous prefill queries, and full production page rounding.
        extension = 1 if self.phase == "decode" else self.queries
        return math.ceil((HISTORY + 4 + extension) / PAGE) * PAGE


def cases(phase_scope: str = "all") -> list[Case]:
    if phase_scope not in ("all", "decode"):
        raise ValueError(f"Unsupported GLM phase: {phase_scope}")
    return [
        Case(phase, heads, queries, tail, locality, policy)
        for heads in (16, 8)
        for phase, sizes in (
            ("decode", (1, 8, 16, 24, 32, 64)),
            ("prefill", (128, 512, 1536, 4096, 8192)),
        )
        if phase_scope == "all" or phase == phase_scope
        for queries in sizes
        for tail in (0, 3)
        for locality in (
            ("independent",) if phase == "decode" else ("independent", "shared")
        )
        for policy in ("warm", "rotating")
    ]


def case_seed(case: Case, bank: int = 0) -> int:
    # Cache policy changes bank count, not the first bank's numerical inputs.
    identity = asdict(case)
    identity.pop("policy")
    return int.from_bytes(
        hashlib.sha256(
            json.dumps([PROTOCOL, SEED, identity, bank], sort_keys=True).encode()
        ).digest()[:4],
        "little",
    )


def logical_selection(case: Case, bank: int = 0):
    """Int32 [Q,2112] logical slots; separate requests use disjoint ranges.

    Initial completed prefix is 128K. The first query closes a group for phase
    zero (sequence length H+4), or has three live tokens (H+3). Subsequent
    prefill queries advance normally. KPool group IDs expand in ascending slot
    order within each group, preserving the unsorted group-selection order.
    """
    import numpy as np

    rng = np.random.default_rng(case_seed(case, bank))
    out = np.full((case.queries, COLUMNS), -1, dtype=np.int32)
    first_length = HISTORY + (4 if case.tail == 0 else 3)
    shared = (
        rng.choice(HISTORY // 4, GROUPS, replace=False)
        if case.locality == "shared"
        else None
    )
    for q in range(case.queries):
        length = first_length + (q if case.phase == "prefill" else 0)
        chosen = (
            shared
            if shared is not None
            else rng.choice(length // 4, GROUPS, replace=False)
        )
        req = q if case.phase == "decode" else 0
        offset = req * case.request_slots
        out[q, :2048] = (chosen[:, None] * 4 + np.arange(4)).reshape(-1) + offset
        tail = length % 4
        out[q, 2048 : 2048 + tail] = np.arange(length - tail, length) + offset
    return out


def physical_selection(case: Case, bank: int = 0):
    """Shuffle pages within each request; retain page offsets and dummy page."""
    import numpy as np

    logical = logical_selection(case, bank)
    pages = case.request_slots // PAGE
    rng = np.random.default_rng(case_seed(case, bank) ^ 0xCAFE)
    mapping = np.concatenate(
        [rng.permutation(pages) + r * pages for r in range(case.requests)]
    )
    valid = logical >= 0
    physical = np.full_like(logical, -1)
    physical[valid] = (
        PAGE + mapping[logical[valid] // PAGE] * PAGE + logical[valid] % PAGE
    )
    return physical


def selected_physical_rows(indices):
    """Sorted distinct valid locations; padding is never initialized or loaded."""
    import numpy as np

    return np.unique(indices[indices >= 0])


def rotation_count(selected_rows: int, l2_bytes: int) -> int:
    if min(selected_rows, l2_bytes) <= 0:
        raise ValueError(
            "Rotation requires positive selected-row count and device L2 size"
        )
    # Both planes are independent 32-byte-sector-aligned allocations: 256+64 B.
    return (2 * l2_bytes) // (selected_rows * 320) + 1


def graph_reserved_bytes(cuda, graph) -> int:
    """Count graph-pool segments, including inactive storage, exactly once.

    Global reserved-memory deltas are invalid: capture may release unrelated
    cached segments. The public pool-filtered snapshot isolates graph ownership.
    """
    segments = cuda.memory_snapshot(graph.pool())
    addresses = [segment["address"] for segment in segments]
    if not segments or len(set(addresses)) != len(addresses):
        raise RuntimeError(
            "Graph pool snapshot is empty or contains duplicate segments"
        )
    sizes = [segment["total_size"] for segment in segments]
    if any(size <= 0 for size in sizes):
        raise RuntimeError("Graph pool snapshot contains a nonpositive segment size")
    return sum(sizes)


def cycles_per_sample(banks: int) -> int:
    if banks < 1:
        raise ValueError("A sample must contain at least one bank")
    return math.ceil(100 / banks)


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model-family", choices=("glm53",), default="glm53")
    p.add_argument("--suite", choices=("glm-attention",), default="glm-attention")
    p.add_argument("--phase", choices=("all", "decode"), default="all")
    p.add_argument("--modes", choices=("native,direct",), default="native,direct")
    p.add_argument(
        "--resume",
        help="Prior timing JSON from identical source/artifacts; resume its valid ordered prefix",
    )
    p.add_argument("--output")
    p.add_argument("--max-seconds", type=int, default=850)
    p.add_argument("--check-imports", action="store_true")
    return p


def packed_reference(raw, norm):
    """Independent Torch byte/bitmap reference; MSB-first words, ascending values."""
    import torch
    from sglang.srt.remnant.reference import topmag_keep_mask

    keep = topmag_keep_mask(raw, 0.5)
    codes = norm(raw.masked_fill(~keep, 0)).to(torch.float8_e4m3fn).view(torch.uint8)
    bits = torch.tensor(
        [-(1 << 63)] + [1 << i for i in range(62, -1, -1)],
        dtype=torch.int64,
        device=raw.device,
    )
    words = (keep.reshape(-1, 8, 64).to(torch.int64) * bits).sum(-1)
    dense = torch.where(keep, codes, 0)
    return keep, codes[keep].reshape(-1, KEEP), words, dense


def attention_reference(q, dense, indices, *, device=None):
    """Independent FP32 attention on selected queries, with Native FP8 Q rounding.

    Mirrors the Native primitive's FP8 probabilities before PV, BF16 partials
    and softmax merge only within established output tolerance, not byte parity.
    No production TileLang function participates in this reference.
    """
    import torch

    output_device = q.device
    rows = dense.view(torch.float8_e4m3fn).reshape(-1, DIM)
    if device is not None:
        # Focused sanitizers instrument production CUDA, not the independent
        # oracle's cuBLAS kernels. Math and FP8 Q rounding remain identical.
        q, indices = q.to(device), indices.to(device)
        rows = rows.to(device=device, dtype=torch.float32)
    out = []
    for query, selected in zip(q, indices):
        valid = selected >= 0
        kv = rows[selected[valid].long()].float()
        if not len(kv):
            out.append(torch.zeros_like(query))
            continue
        score = query.to(torch.float8_e4m3fn).float() @ kv.T * SCALE
        out.append((score.softmax(-1) @ kv).to(torch.bfloat16))
    return torch.stack(out).unsqueeze(0).to(output_device)


def _hash_tensor(digest, tensor) -> None:
    import torch

    digest.update(
        tensor.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes()
    )


def _device_contract() -> tuple[int, int]:
    import torch
    from cuda.bindings import driver

    if (
        not torch.cuda.is_available()
        or torch.cuda.get_device_capability() != (9, 0)
        or "H100" not in torch.cuda.get_device_name()
    ):
        raise RuntimeError("GLM attention benchmark requires one H100/SM90")
    result, l2 = driver.cuDeviceGetAttribute(
        driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_L2_CACHE_SIZE,
        torch.cuda.current_device(),
    )
    if result != driver.CUresult.CUDA_SUCCESS or l2 <= 0:
        raise RuntimeError(f"Cannot query actual device L2 size: {result}, {l2}")
    return int(l2), torch.cuda.get_device_properties(0).total_memory


class Fixture:
    """Own disjoint Native and Packed banks, common BF16 Q and int32 selections.

    Cached tensor shapes: Native uint8 [slots,512], values uint8 [slots,256],
    bitmap uint64 [slots,8]; all contiguous CUDA tensors. Slot zero/page zero
    remain dummy. No Adapter workspace, KDA pool or model weights are allocated.
    """

    def __init__(self, case: Case, deadline: float):
        import torch
        from sglang.srt.layers.layernorm import RMSNorm
        from sglang.srt.remnant.glm import PackedCache, normalize_and_store

        self.case, self.deadline = case, deadline
        self.l2, total = _device_contract()
        first = physical_selection(case)
        selected_rows = len(selected_physical_rows(first))
        self.banks = (
            1 if case.policy == "warm" else rotation_count(selected_rows, self.l2)
        )
        self.slots = PAGE + case.requests * case.request_slots
        # Bank-specific random selections can have different union sizes. Make
        # the rotation bound exact using all actual selections, not just bank 0.
        selections = [first]
        touched = [selected_rows]
        while len(selections) < self.banks:
            self.check()
            ids = physical_selection(case, len(selections))
            selections.append(ids)
            touched.append(len(selected_physical_rows(ids)))
        if case.policy == "rotating":
            while sum(touched) * 320 <= 2 * self.l2:
                self.check()
                ids = physical_selection(case, len(selections))
                selections.append(ids)
                touched.append(len(selected_physical_rows(ids)))
            # Remove a final unnecessary bank: smallest prefix exceeding 2*L2.
            while len(touched) > 1 and sum(touched[:-1]) * 320 > 2 * self.l2:
                touched.pop()
                selections.pop()
        self.banks = len(selections)
        cache_bytes = self.banks * self.slots * (512 + 320)
        input_bytes = (
            self.banks * case.queries * COLUMNS * 4
            + case.queries * case.heads * DIM * 2
        )
        # Four BF16-sized outputs per mode conservatively cover capture liveness
        # (input Q conversion, partial O/LSE and final O) for every split geometry.
        output_budget = 8 * case.queries * case.heads * DIM * 2 * 66
        self.memory_limit = int(total * 0.7)
        # Use actual selected split count rather than 66 for the graph budget.
        from sglang.kernels.ops.attention.dsa.tilelang_kernel import _pick_inner_iter

        inner = _pick_inner_iter(
            case.queries,
            66,
            torch.cuda.get_device_properties(0).multi_processor_count,
            1,
        )
        output_budget //= inner
        self.projected_bytes = cache_bytes + input_bytes + output_budget + 512 * 1024**2
        if self.projected_bytes >= self.memory_limit:
            raise RuntimeError(
                f"{case.name}: projected fixture {self.projected_bytes} exceeds 70% memory cap {self.memory_limit}; banks={self.banks}"
            )
        self.q = torch.randn(
            (case.queries, case.heads, DIM),
            generator=torch.Generator(device="cuda").manual_seed(
                case_seed(case) ^ 0x1234
            ),
            dtype=torch.bfloat16,
            device="cuda",
        )
        self.ids, self.native, self.packed = [], [], []
        self.digest = hashlib.sha256()
        _hash_tensor(self.digest, self.q)
        norm = RMSNorm(DIM, eps=1e-5).cuda().to(torch.bfloat16)
        norm.weight.data.copy_(torch.linspace(0.5, 1.5, DIM, device="cuda"))
        _hash_tensor(self.digest, norm.weight)
        for bank, selected in enumerate(selections):
            self.check()
            ids = torch.from_numpy(selected).cuda().unsqueeze(1)
            native = torch.zeros(self.slots, DIM, dtype=torch.uint8, device="cuda")
            values = torch.zeros(self.slots, KEEP, dtype=torch.uint8, device="cuda")
            bitmap = torch.zeros(self.slots, 8, dtype=torch.uint64, device="cuda")
            gen = torch.Generator(device="cuda").manual_seed(case_seed(case, bank))
            locations = selected_physical_rows(selected)
            for start in range(0, len(locations), 65536):
                self.check()
                end = min(start + 65536, len(locations))
                raw = torch.randn(
                    (end - start, DIM),
                    generator=gen,
                    device="cuda",
                    dtype=torch.bfloat16,
                )
                loc = torch.from_numpy(locations[start:end]).to(
                    device="cuda", dtype=torch.int64
                )
                # Assignment scatters to backing storage; advanced-index copy_
                # would mutate a temporary. Both paths consume identical raw rows.
                native[loc] = norm(raw).to(torch.float8_e4m3fn).view(torch.uint8)
                normalize_and_store(raw, norm, loc, values, bitmap)
                mask, expected_values, words, dense = packed_reference(raw, norm)
                if not torch.equal(values[loc], expected_values) or not torch.equal(
                    bitmap.view(torch.int64)[loc], words
                ):
                    raise RuntimeError(
                        f"{case.name}/bank{bank}/rows{start}:{end}: Packed bytes/bitmap mismatch"
                    )
                _hash_tensor(self.digest, loc)
                _hash_tensor(self.digest, raw)
                del raw, loc, mask, expected_values, words, dense
            self.ids.append(ids)
            self.native.append(native)
            self.packed.append(PackedCache(values, bitmap, None, "direct"))
            _hash_tensor(self.digest, ids)
        del selections, norm
        self.identity = dict(
            common_inputs_sha256=self.digest.hexdigest(),
            initialization="selected-physical-union-v1; unselected backing rows zero",
            initialized_rows_per_bank=touched,
            banks=self.banks,
            l2_bytes=self.l2,
            selected_rows_per_bank=touched,
            packed_selected_bytes=sum(touched) * 320,
            native_selected_bytes=sum(touched) * 512,
            backing_cache_bytes=cache_bytes,
            input_bytes=input_bytes,
            projected_bytes=self.projected_bytes,
            memory_limit_bytes=self.memory_limit,
            inner_iter=inner,
            partial_groups=66 // inner,
            first_query_sequence_length=HISTORY + (4 if case.tail == 0 else 3),
        )
        self.graphs = {}
        self.outputs = {}
        self.validate()

    def check(self) -> None:
        if time.monotonic() >= self.deadline - 5:
            raise TimeoutError(f"{self.case.name}: fixture/timing deadline reached")

    def call(self, mode: str, bank: int):
        import torch
        from sglang.kernels.ops.attention.dsa.tilelang_kernel import tilelang_sparse_fwd
        from sglang.srt.remnant.glm import attention

        if mode == "native":
            return tilelang_sparse_fwd(
                self.q,
                self.native[bank].view(torch.float8_e4m3fn).unsqueeze(1),
                self.ids[bank],
                SCALE,
                DIM,
            )
        if mode != "direct":
            raise ValueError(f"Unsupported GLM path {mode}")
        return attention(self.q, self.packed[bank], self.ids[bank], SCALE)

    def validate(self, *, reference_device=None) -> None:
        import torch
        from sglang.kernels.ops.attention.dsa.tilelang_kernel import tilelang_sparse_fwd

        sample = torch.linspace(
            0, self.case.queries - 1, min(8, self.case.queries), device="cuda"
        ).long()
        for bank in range(self.banks):
            self.check()
            selected = self.ids[bank][sample, 0]
            cache = self.packed[bank]
            # Independently expand only the selected rows with boolean scatter.
            unique, inverse = torch.unique(selected[selected >= 0], return_inverse=True)
            words = cache.bitmaps.view(torch.int64)[unique.long()]
            coords = torch.arange(DIM, device="cuda")
            keep = ((words[:, coords // 64] >> (63 - coords % 64)) & 1).bool()
            dense = torch.zeros(len(unique), DIM, dtype=torch.uint8, device="cuda")
            dense[keep] = cache.values[unique.long()].flatten()
            local = torch.full_like(selected, -1)
            local[selected >= 0] = inverse.to(local.dtype)
            native_ref = attention_reference(
                self.q[sample], self.native[bank], selected, device=reference_device
            )
            direct_ref = attention_reference(
                self.q[sample], dense, local, device=reference_device
            )
            for mode, reference in (("native", native_ref), ("direct", direct_ref)):
                output = self.call(mode, bank)
                if not bool(torch.isfinite(output).all()):
                    raise RuntimeError(
                        f"{self.case.name}/{bank}/{mode}: nonfinite output"
                    )
                torch.testing.assert_close(
                    output[:, sample], reference, atol=ATOL, rtol=RTOL
                )
                if mode == "direct":
                    stock = tilelang_sparse_fwd(
                        self.q[sample],
                        dense.view(torch.float8_e4m3fn).unsqueeze(1),
                        local.unsqueeze(1),
                        SCALE,
                        DIM,
                    )
                    torch.testing.assert_close(
                        output[:, sample], stock, atol=ATOL, rtol=RTOL
                    )
                del output

    def capture(self, *, warmup_calls: int = 10, capture_stream=None) -> None:
        import torch

        if warmup_calls < 1:
            raise ValueError("Capture requires positive warmup count")
        for mode in MODES:
            self.check()
            for i in range(max(warmup_calls, self.banks)):
                output = self.call(mode, i % self.banks)
                del output
            torch.cuda.synchronize()
            graph = torch.cuda.CUDAGraph()
            # Tests can capture directly on their explicitly owned stream.
            # Timing leaves this None and preserves Torch's default side stream.
            capture = (torch.cuda.graph(graph) if capture_stream is None else
                       torch.cuda.graph(graph, stream=capture_stream))
            with capture:
                for bank in range(self.banks):
                    output = self.call(mode, bank)
                    if bank != self.banks - 1:
                        del output  # stream-ordered graph allocator can reuse it
            self.graphs[mode], self.outputs[mode] = graph, output
            graph.replay()
            torch.cuda.synchronize()
            graph_bytes = graph_reserved_bytes(torch.cuda, graph)
            # Constant per-call bound, independent of bank count; includes all
            # split partials, Q conversion, and allocator padding.
            bound = (
                8
                * self.case.queries
                * self.case.heads
                * DIM
                * 2
                * (66 // self.identity["inner_iter"])
                + 64 * 1024**2
            )
            if graph_bytes > bound or torch.cuda.memory_reserved() >= self.memory_limit:
                raise RuntimeError(
                    f"{self.case.name}/{mode}: graph allocation {graph_bytes} exceeds bounded liveness {bound} or fixture memory cap"
                )
            self.identity[f"{mode}_graph_bytes"] = graph_bytes
        self.validate_replay()

    def validate_replay(self) -> None:
        import torch

        for mode in ("direct", "native", "direct", "native"):
            self.graphs[mode].replay()
            expected = self.call(mode, self.banks - 1)
            torch.testing.assert_close(self.outputs[mode], expected, atol=0, rtol=0)

    def time(self) -> dict:
        from sglang.benchmark.remnant.cases import summary
        from sglang.benchmark.remnant.timing import paired_samples

        self.capture()
        repeats = cycles_per_sample(self.banks)
        samples, orders = paired_samples(
            {mode: self.graphs[mode].replay for mode in MODES},
            ROUNDS,
            case_seed(self.case),
            repeats,
            balanced=True,
            before=lambda mode: self.graphs[mode].replay(),
            check=self.check,
        )
        samples = {
            mode: [t / self.banks for t in times] for mode, times in samples.items()
        }
        self.validate_replay()
        self.validate()
        return dict(
            execution="cuda_graph",
            paths={m: summary(t) for m, t in samples.items()},
            paired_ratios={
                "direct_over_native": summary(
                    [d / n for d, n in zip(samples["direct"], samples["native"])]
                )
            },
            round_orders=orders,
            cycles_per_sample=repeats,
            calls_per_sample=repeats * self.banks,
        )


def runtime_hashes(root: Path) -> dict[str, str]:
    paths = (
        "kernels/ops/attention/dsa/tilelang_kernel.py",
        "kernels/jit/include/sgl_kernel/glm_packed_decode.cuh",
        "srt/remnant/glm.py",
        "srt/remnant/glm_pool.py",
        "srt/remnant/reference.py",
        "srt/layers/layernorm.py",
    )
    return {p: hashlib.sha256((root / p).read_bytes()).hexdigest() for p in paths}


def source_manifest() -> dict:
    import torch
    import sglang.srt.remnant.glm as codec
    import sglang.kernels.ops.attention.dsa.tilelang_kernel as tile
    import sglang.benchmark.remnant.timing as timing

    sources = {
        str(Path(p).resolve()): hashlib.sha256(Path(p).read_bytes()).hexdigest()
        for p in (__file__, codec.__file__, tile.__file__, timing.__file__,
                  Path(tile.__file__).resolve().parents[3] / "jit/include/sgl_kernel/glm_packed_decode.cuh")
    }
    build_path = os.environ.get("REMNANT_BUILD_MANIFEST_PATH")
    return dict(
        sources=sources,
        compiled_build=json.loads(Path(build_path).read_text()) if build_path else {},
        device=torch.cuda.get_device_name(),
        torch=torch.__version__,
        cuda=torch.version.cuda,
        triton=importlib.metadata.version("triton"),
        tilelang=importlib.metadata.version("tilelang"),
    )


def write_results(payload: dict, base: Path) -> None:
    base.parent.mkdir(parents=True, exist_ok=True)
    tmp = base.with_suffix(".json.partial")
    tmp.write_text(json.dumps(payload, indent=2) + "\n")
    tmp.replace(base.with_suffix(".json"))
    with base.with_suffix(".csv").open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=(
                "case",
                "mode",
                "mean_ms",
                "median_ms",
                "p95_round_average_ms",
                "samples_ms",
                "direct_over_native",
                "fixture",
            ),
            lineterminator="\n",
        )
        writer.writeheader()
        for case in payload["cases"]:
            for mode, stats in case.get("paths", {}).items():
                writer.writerow(
                    dict(
                        case=case["name"],
                        mode=mode,
                        mean_ms=stats["mean"],
                        median_ms=stats["median"],
                        p95_round_average_ms=stats["p95"],
                        samples_ms=json.dumps(stats["samples"]),
                        direct_over_native=json.dumps(case["paired_ratios"]),
                        fixture=json.dumps(case["fixture"]),
                    )
                )


def validate_timing(payload: dict, *, prefix: bool = False) -> None:
    phase = payload.get("phase", "all")
    protocol = DECODE_PROTOCOL if phase == "decode" else PROTOCOL
    if payload.get("protocol") != protocol or (
        not prefix and payload.get("status") != "passed"
    ):
        raise RuntimeError("GLM timing protocol incomplete")
    expected = cases(phase)
    if prefix:
        if len(payload["cases"]) > len(expected):
            raise RuntimeError("GLM checkpoint exceeds the required matrix")
        expected = expected[: len(payload["cases"])]
    if [c["name"] for c in payload["cases"]] != [c.name for c in expected]:
        raise RuntimeError(f"GLM timing requires the ordered {len(cases(phase))}-case {phase} matrix")
    for case, definition in zip(payload["cases"], expected):
        if (
            not case.get("valid")
            or set(case.get("paths", {})) != set(MODES)
            or case.get("case") != asdict(definition)
        ):
            raise RuntimeError(f"Incomplete GLM case {case['name']}")
        fixture = case["fixture"]
        banks = fixture["banks"]
        touched = fixture["selected_rows_per_bank"]
        if (
            banks != len(touched)
            or banks < 1
            or any(n <= 0 for n in touched)
            or len(fixture["common_inputs_sha256"]) != 64
        ):
            raise RuntimeError(f"Invalid GLM fixture identity {case['name']}")
        if definition.policy == "warm":
            valid_policy = banks == 1
        else:
            valid_policy = (
                sum(touched) * 320 > 2 * fixture["l2_bytes"]
                and sum(touched[:-1]) * 320 <= 2 * fixture["l2_bytes"]
            )
        if (
            not valid_policy
            or fixture["projected_bytes"] >= fixture["memory_limit_bytes"]
        ):
            raise RuntimeError(f"Invalid GLM memory/cache policy {case['name']}")
        orders = case["round_orders"]
        if (
            len(orders) != ROUNDS
            or orders.count(["native", "direct"]) != ROUNDS // 2
            or orders.count(["direct", "native"]) != ROUNDS // 2
        ):
            raise RuntimeError(f"Unbalanced GLM timing {case['name']}")
        if (
            case["execution"] != "cuda_graph"
            or case["calls_per_sample"] != case["cycles_per_sample"] * banks
            or case["calls_per_sample"] < 100
        ):
            raise RuntimeError(f"Invalid GLM sample normalization {case['name']}")
        for path in case["paths"].values():
            if len(path["samples"]) != ROUNDS or any(
                not math.isfinite(x) or x <= 0 for x in path["samples"]
            ):
                raise RuntimeError(f"Missing/invalid GLM rounds {case['name']}")
        ratios = case["paired_ratios"]["direct_over_native"]["samples"]
        if ratios != [
            d / n
            for d, n in zip(
                case["paths"]["direct"]["samples"], case["paths"]["native"]["samples"]
            )
        ]:
            raise RuntimeError(f"Invalid GLM paired ratios {case['name']}")


def resume_prefix(previous: dict, source: dict, phase: str = "all") -> list[dict]:
    """Reuse only validated cases from an identical fixture/toolchain snapshot."""
    if (
        previous.get("phase", "all") != phase
        or previous.get("source") != source
        or previous.get("seed") != SEED
        or previous.get("rounds") != ROUNDS
    ):
        raise RuntimeError(
            "GLM checkpoint source/artifact/seed/round identity mismatch"
        )
    records = list(previous.get("cases", []))
    # A killed/failed final case has no accepted measurement. Interior failures
    # are never skipped or converted into coverage.
    if records and not records[-1].get("valid"):
        records.pop()
    validate_timing({**previous, "cases": records}, prefix=True)
    return records


def weighted_overhead(records: list[dict]) -> dict:
    """Batch-weighted mean of paired round overheads; other axes equal weight."""
    def score(rows):
        weight = sum(c["case"]["queries"] for c in rows)
        return (100 * sum(c["case"]["queries"] *
                (sum(c["paired_ratios"]["direct_over_native"]["samples"]) / ROUNDS - 1)
                for c in rows) / weight) if weight else None
    rows = [c for c in records if c.get("valid") and c["case"]["phase"] == "decode"]
    return dict(batch_weighted_overhead_percent=score(rows),
                by_heads={str(h): score([c for c in rows if c["case"]["heads"] == h])
                          for h in (16, 8)}, completed_cases=len(rows),
                unacceptable_2x_cases=[c["name"] for c in rows if
                    c["paired_ratios"]["direct_over_native"]["mean"] >= 2])


def main() -> None:
    args = parser().parse_args()
    protocol = DECODE_PROTOCOL if args.phase == "decode" else PROTOCOL
    if args.check_imports:
        from sglang.srt.remnant.glm import compile_kernels
        from sglang.kernels.ops.attention.dsa.tilelang_kernel import tilelang_sparse_fwd

        assert callable(tilelang_sparse_fwd) and callable(compile_kernels)
        print(json.dumps(dict(protocol=protocol, cases=len(cases(args.phase)), modes=MODES)))
        return
    if not args.output or not 60 <= args.max_seconds <= 1200:
        raise ValueError("GLM requires --output and --max-seconds between 60 and 1200")
    import torch

    base = Path(args.output)
    if any(
        base.with_suffix(ext).exists() for ext in (".json", ".csv", ".json.partial")
    ):
        raise FileExistsError(f"Refusing to overwrite GLM run {base}")
    payload = dict(
        protocol=protocol,
        phase=args.phase,
        status="running",
        seed=SEED,
        rounds=ROUNDS,
        scope="model-free complete graph attention; no store/indexer/model/TP communication",
        source=source_manifest(),
        cases=[],
    )
    if args.resume:
        checkpoint = Path(args.resume)
        payload["cases"] = resume_prefix(
            json.loads(checkpoint.read_text()), payload["source"], args.phase
        )
        payload["resumed_from"] = dict(
            path=str(checkpoint),
            sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
            accepted_cases=len(payload["cases"]),
        )
    deadline = time.monotonic() + args.max_seconds
    fixture = None
    try:
        with torch.inference_mode():
            for case in cases(args.phase)[len(payload["cases"]) :]:
                record = dict(name=case.name, case=asdict(case), valid=False)
                payload["cases"].append(record)
                write_results(payload, base)
                fixture = Fixture(case, deadline)
                record["fixture"] = fixture.identity
                record.update(fixture.time())
                record["valid"] = True
                if args.phase == "decode":
                    payload["summary"] = weighted_overhead(payload["cases"])
                write_results(payload, base)
                del fixture
                fixture = None
                gc.collect()
                torch.cuda.empty_cache()
        payload["status"] = "passed"
        validate_timing(payload)
    except BaseException as error:
        payload.update(status="incomplete", error=repr(error))
        raise
    finally:
        write_results(payload, base)
