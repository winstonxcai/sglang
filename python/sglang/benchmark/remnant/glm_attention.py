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
from collections.abc import Callable
from dataclasses import asdict, dataclass
from functools import cache
from pathlib import Path

PROTOCOL = "remnant-glm-attention-v1"
DECODE_PROTOCOL = "remnant-glm-decode-v1"
PREFILL_PROTOCOL = "remnant-glm-prefill-v1"
PREFILL_SMALL_PROTOCOL = "remnant-glm-prefill-small-v1"
UPSTREAM_PROTOCOL = "remnant-glm-upstream-decode-v1"
UPSTREAM_SUITE = "glm-upstream-decode"
PROTOCOLS = {
    "all": PROTOCOL,
    "decode": DECODE_PROTOCOL,
    "prefill": PREFILL_PROTOCOL,
    "prefill-small": PREFILL_SMALL_PROTOCOL,
}
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
    if phase_scope == "prefill-small":
        return [
            Case("prefill", 16, queries, 0, locality, "warm")
            for queries in (128, 512)
            for locality in ("independent", "shared")
        ]
    if phase_scope not in PROTOCOLS:
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


def experiment(suite: str, phase: str) -> tuple[str, tuple[str, str], list[Case]]:
    """Keep the pristine upstream experiment distinct from historical matrices."""
    if suite == UPSTREAM_SUITE:
        if phase != "decode":
            raise ValueError("glm-upstream-decode requires --phase decode")
        return (
            UPSTREAM_PROTOCOL,
            ("upstream", "direct"),
            [case for case in cases("decode") if case.heads == 16],
        )
    if suite != "glm-attention":
        raise ValueError(f"Unsupported GLM suite: {suite}")
    return PROTOCOLS[phase], MODES, cases(phase)


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
    offsets = np.arange(4)
    if shared is not None:
        out[:, :2048] = (shared[:, None] * 4 + offsets).reshape(-1)
    for q in range(case.queries):
        length = first_length + (q if case.phase == "prefill" else 0)
        req = q if case.phase == "decode" else 0
        offset = req * case.request_slots
        if shared is None:
            chosen = rng.choice(length // 4, GROUPS, replace=False)
            out[q, :2048] = (chosen[:, None] * 4 + offsets).reshape(-1) + offset
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
    shared = case.locality == "shared"
    # Shared prefill selects the same 2,048 history rows for every query.
    # Map them once, then map the changing tails and broadcast the history.
    parts = (logical[:1, :2048], logical[:, 2048:2051]) if shared else (logical,)
    for part in parts:
        valid = part >= 0
        positions = part[valid]
        part[valid] = PAGE + mapping[positions // PAGE] * PAGE + positions % PAGE
    if shared:
        logical[1:, :2048] = logical[0, :2048]
    return logical


def selected_physical_rows(indices, *, shared_history: bool = False):
    """Sorted distinct valid locations; padding is never initialized or loaded."""
    import numpy as np

    if shared_history:
        indices = np.concatenate((indices[0, :2048], indices[:, 2048:2051].ravel()))
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
    p.add_argument(
        "--suite", choices=("glm-attention", UPSTREAM_SUITE), default="glm-attention"
    )
    p.add_argument("--phase", choices=tuple(PROTOCOLS), default="all")
    p.add_argument("--packed-prefill-tile", type=int, choices=(32, 64, 128), default=32)
    p.add_argument(
        "--modes", choices=("native,direct", "upstream,direct"), default=None
    )
    p.add_argument(
        "--resume",
        help="Prior timing JSON from identical source/artifacts; requires REMNANT_BUILD_MANIFEST_PATH",
    )
    p.add_argument("--profile-decode", action="store_true")
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


def selected_packed_reference(values, bitmaps, selected):
    """Independent bitmap scatter on the inputs' device; no production unpacker.

    values uint8 [slots,256], bitmaps uint64 [slots,8], selected int32 [Q,K].
    Returns raw uint8 dense rows and int32 local indices, preserving duplicates.
    """
    import torch

    unique, inverse = torch.unique(selected[selected >= 0], return_inverse=True)
    words = bitmaps.view(torch.int64)[unique.long()]
    coords = torch.arange(DIM, device=selected.device)
    keep = ((words[:, coords // 64] >> (63 - coords % 64)) & 1).bool()
    dense = torch.zeros(len(unique), DIM, dtype=torch.uint8, device=selected.device)
    dense[keep] = values[unique.long()].flatten()
    local = torch.full_like(selected, -1)
    local[selected >= 0] = inverse.to(local.dtype)
    return dense, local


def _hash_tensor(digest, tensor) -> None:
    import torch

    digest.update(
        tensor.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes()
    )


def _device_contract(gpu_family: str = "H100") -> tuple[int, int]:
    import torch
    from cuda.bindings import driver

    if (
        not torch.cuda.is_available()
        or torch.cuda.get_device_capability() != (9, 0)
        or gpu_family not in torch.cuda.get_device_name()
    ):
        raise RuntimeError(f"GLM attention benchmark requires one {gpu_family}/SM90")
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

    def __init__(
        self,
        case: Case,
        deadline: float,
        *,
        packed_prefill_tile: int = 32,
        gpu_family: str = "H100",
        upstream: bool = False,
        wgmma: dict | None = None,
    ):
        self.upstream = upstream
        self.packed_prefill_tile = packed_prefill_tile
        self.wgmma_kernels = None
        import torch

        from sglang.srt.layers.layernorm import RMSNorm
        from sglang.srt.remnant.glm import PackedCache, normalize_and_store

        self.case, self.deadline = case, deadline
        self.l2, total = _device_contract(gpu_family)
        first = physical_selection(case)
        shared_history = case.locality == "shared"
        first_rows = selected_physical_rows(first, shared_history=shared_history)
        selected_rows = len(first_rows)
        self.slots = PAGE + case.requests * case.request_slots
        # Bank-specific random selections can have different union sizes. Make
        # the rotation bound exact using all actual selections, not just bank 0.
        selections = [first]
        locations_by_bank = [first_rows]
        touched = [selected_rows]
        # Actual union sizes can differ across banks. Stop at the smallest
        # prefix of disjoint banks whose selected Packed bytes exceed 2*L2.
        while case.policy == "rotating" and sum(touched) * 320 <= 2 * self.l2:
            self.check()
            ids = physical_selection(case, len(selections))
            locations = selected_physical_rows(ids, shared_history=shared_history)
            selections.append(ids)
            locations_by_bank.append(locations)
            touched.append(len(locations))
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
        if upstream and inner != (1 if case.queries == 1 else 2):
            raise RuntimeError(
                f"Unprepared H200 decode split: {inner}; refusing GPU compilation"
            )
        output_budget //= min(inner, 33) if packed_prefill_tile != 32 else inner
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
        for bank, (selected, locations) in enumerate(
            zip(selections, locations_by_bank)
        ):
            self.check()
            ids = torch.from_numpy(selected).cuda().unsqueeze(1)
            native = torch.zeros(self.slots, DIM, dtype=torch.uint8, device="cuda")
            values = torch.zeros(self.slots, KEEP, dtype=torch.uint8, device="cuda")
            bitmap = torch.zeros(self.slots, 8, dtype=torch.uint64, device="cuda")
            gen = torch.Generator(device="cuda").manual_seed(case_seed(case, bank))
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
        del selections, locations_by_bank, norm
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
            path_geometry={
                self.baseline_name: dict(
                    tile_rows=32, inner_iter=inner, partial_groups=66 // inner
                ),
                "direct": dict(
                    tile_rows=packed_prefill_tile,
                    inner_iter={64: 17, 128: 9}.get(packed_prefill_tile, inner),
                    partial_groups=2 if packed_prefill_tile != 32 else 66 // inner,
                ),
            },
            first_query_sequence_length=HISTORY + (4 if case.tail == 0 else 3),
        )
        if upstream:
            from sglang.benchmark.remnant.glm_artifacts import (
                verify_decode_library,
                verify_decode_partial_library,
            )
            from sglang.benchmark.remnant.glm_upstream import wgmma_geometry
            from sglang.srt.remnant.glm_config import H200_PACKED_DECODE_INNER
            from sglang.kernels.ops.attention.dsa.tilelang_kernel import (
                glm_packed_decode_combine,
                glm_packed_wgmma_partial,
                sparse_mla_fwd_decode_partial_fp8,
            )

            columns = 128 if case.queries == 1 else 512
            if wgmma is not None and case.queries > 1:
                variant = wgmma[str(case.queries)]
                geometry = wgmma_geometry(case.queries)
                inner = geometry["inner_iter"]
                tile_rows = geometry["tile_rows"]
                groups = geometry["partial_groups"]
                partial = glm_packed_wgmma_partial(inner, SCALE)
                verify_decode_library(partial, variant["partial"])
                combine = glm_packed_decode_combine(case.heads, DIM, groups, columns)
                verify_decode_library(combine, variant["combine"])
                self.wgmma_kernels = partial, combine
            else:
                tile_rows = 32
                inner = self.packed_decode_inner = H200_PACKED_DECODE_INNER[case.queries]
                groups = 66 // inner
                manifest = json.loads(
                    Path(os.environ["REMNANT_GLM_COMPILE_MANIFEST"]).read_text()
                )
                (partial_variant,) = (
                    item
                    for item in manifest["tilelang"]
                    if item["packed"]
                    and item["heads"] == case.heads
                    and item.get("block_i", 32) == tile_rows
                    and item["inner"] == inner
                )
                partial = sparse_mla_fwd_decode_partial_fp8(
                    case.heads, DIM, 0, COLUMNS, packed=True, sm_scale=SCALE,
                    block_I=tile_rows, inner_iter=inner, threads=128,
                )
                verify_decode_partial_library(partial, partial_variant)
                variant = manifest["decode_combine"][f"{groups}x{columns}"]
                combine = glm_packed_decode_combine(case.heads, DIM, groups, columns)
                verify_decode_library(combine, variant)
            self.identity["path_geometry"]["direct"].update(
                tile_rows=tile_rows, inner_iter=inner, partial_groups=groups
            )
            self.identity["direct_combine_columns"] = columns
        self.graphs = {}
        self.outputs = {}
        self.validate()

    @property
    def baseline_name(self) -> str:
        return "upstream" if getattr(self, "upstream", False) else "native"

    @property
    def modes(self) -> tuple[str, str]:
        return self.baseline_name, "direct"

    def check(self) -> None:
        if time.monotonic() >= self.deadline - 5:
            raise TimeoutError(f"{self.case.name}: fixture/timing deadline reached")

    def call(self, mode: str, bank: int):
        import torch

        from sglang.kernels.ops.attention.dsa.tilelang_kernel import tilelang_sparse_fwd
        from sglang.srt.remnant.glm import attention

        if mode == self.baseline_name:
            if self.baseline_name == "upstream":
                from sglang.benchmark.remnant.glm_upstream import load_baseline

                tilelang_sparse_fwd = load_baseline().tilelang_sparse_fwd
            return tilelang_sparse_fwd(
                self.q,
                self.native[bank].view(torch.float8_e4m3fn).unsqueeze(1),
                self.ids[bank],
                SCALE,
                DIM,
            )
        if mode != "direct":
            raise ValueError(f"Unsupported GLM path {mode}")
        cache = self.packed[bank]
        if self.wgmma_kernels is not None:
            partial, combine = self.wgmma_kernels
            return combine(*partial(
                self.q.to(torch.float8_e4m3fn).unsqueeze(0),
                cache.values.view(torch.float8_e4m3fn)[None, :, None, :],
                cache.bitmaps,
                self.ids[bank].unsqueeze(0),
            ))
        if self.packed_prefill_tile == 32 and self.baseline_name != "upstream":
            return attention(self.q, cache, self.ids[bank], SCALE)
        return tilelang_sparse_fwd(
            self.q,
            cache.values.view(torch.float8_e4m3fn).unsqueeze(1),
            self.ids[bank],
            SCALE,
            DIM,
            packed_bitmap=cache.bitmaps,
            packed_prefill_tile=self.packed_prefill_tile,
            packed_decode_inner=getattr(self, "packed_decode_inner", 0),
            packed_decode_combine_columns=(128 if self.case.queries == 1 else 512)
            if self.baseline_name == "upstream"
            else 0,
        )

    def validate(self, *, reference_device=None) -> None:
        import torch

        from sglang.kernels.ops.attention.dsa.tilelang_kernel import tilelang_sparse_fwd

        if self.baseline_name == "upstream":
            from sglang.benchmark.remnant.glm_upstream import load_baseline

            tilelang_sparse_fwd = load_baseline().tilelang_sparse_fwd
        cpu = reference_device == "cpu"
        sample = torch.linspace(
            0,
            self.case.queries - 1,
            min(8, self.case.queries),
            device="cpu" if cpu else "cuda",
        ).long()
        for bank in range(self.banks):
            self.check()
            cache = self.packed[bank]
            # Focused tools keep ALL reference-only metadata/check operations
            # on CPU, not just matmuls. Actual attention calls remain on H100.
            ids = self.ids[bank].cpu() if cpu else self.ids[bank]
            selected = ids[sample, 0]
            dense, local = selected_packed_reference(
                cache.values.cpu() if cpu else cache.values,
                cache.bitmaps.cpu() if cpu else cache.bitmaps,
                selected,
            )
            query = self.q.cpu()[sample] if cpu else self.q[sample]
            native_ref = attention_reference(
                query,
                self.native[bank].cpu() if cpu else self.native[bank],
                selected,
                device=reference_device,
            )
            direct_ref = attention_reference(
                query, dense, local, device=reference_device
            )
            for mode, reference in (
                (self.baseline_name, native_ref),
                ("direct", direct_ref),
            ):
                output = self.call(mode, bank)
                checked = output.cpu() if cpu else output
                if not bool(torch.isfinite(checked).all()):
                    raise RuntimeError(
                        f"{self.case.name}/{bank}/{mode}: nonfinite output"
                    )
                torch.testing.assert_close(
                    checked[:, sample], reference, atol=ATOL, rtol=RTOL
                )
                if mode == "direct":
                    stock = tilelang_sparse_fwd(
                        query.to("cuda") if cpu else query,
                        (dense.to("cuda") if cpu else dense)
                        .view(torch.float8_e4m3fn)
                        .unsqueeze(1),
                        (local.to("cuda") if cpu else local).unsqueeze(1),
                        SCALE,
                        DIM,
                    )
                    torch.testing.assert_close(
                        checked[:, sample],
                        stock.cpu() if cpu else stock,
                        atol=ATOL,
                        rtol=RTOL,
                    )
                del output, checked

    def capture(
        self, *, warmup_calls: int = 10, capture_stream=None, reference_device=None
    ) -> None:
        import torch

        if warmup_calls < 1:
            raise ValueError("Capture requires positive warmup count")
        for mode in self.modes:
            self.check()
            for i in range(max(warmup_calls, self.banks)):
                output = self.call(mode, i % self.banks)
                del output
            torch.cuda.synchronize()
            graph = torch.cuda.CUDAGraph()
            # Tests can capture directly on their explicitly owned stream.
            # Timing leaves this None and preserves Torch's default side stream.
            capture = (
                torch.cuda.graph(graph)
                if capture_stream is None
                else torch.cuda.graph(graph, stream=capture_stream)
            )
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
                * self.identity["path_geometry"][mode]["partial_groups"]
                + 64 * 1024**2
            )
            if graph_bytes > bound or torch.cuda.memory_reserved() >= self.memory_limit:
                raise RuntimeError(
                    f"{self.case.name}/{mode}: graph allocation {graph_bytes} exceeds bounded liveness {bound} or fixture memory cap"
                )
            self.identity[f"{mode}_graph_bytes"] = graph_bytes
        self.validate_replay(reference_device=reference_device)

    def validate_replay(self, *, reference_device=None) -> None:
        import torch

        for mode in ("direct", self.baseline_name, "direct", self.baseline_name):
            self.graphs[mode].replay()
            expected = self.call(mode, self.banks - 1)
            actual = self.outputs[mode]
            if reference_device == "cpu":
                actual, expected = actual.cpu(), expected.cpu()
            torch.testing.assert_close(actual, expected, atol=0, rtol=0)

        if self.baseline_name == "upstream":
            # Q keeps its address but changes in place after capture. This
            # tests replay inputs in both the pristine and production paths.
            self.q.neg_()
            try:
                for mode in self.modes:
                    self.graphs[mode].replay()
                    expected = self.call(mode, self.banks - 1)
                    torch.testing.assert_close(
                        self.outputs[mode], expected, atol=0, rtol=0
                    )
            finally:
                self.q.neg_()
                for mode in self.modes:
                    self.graphs[mode].replay()
                torch.cuda.synchronize()
            self.identity["changing_input_replay"] = "passed"

    def time(self) -> dict:
        from sglang.benchmark.remnant.cases import summary
        from sglang.benchmark.remnant.timing import paired_ratio_summary, paired_samples

        self.capture()
        repeats = cycles_per_sample(self.banks)
        samples, orders = paired_samples(
            {mode: self.graphs[mode].replay for mode in self.modes},
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
                f"direct_over_{self.baseline_name}": paired_ratio_summary(
                    samples["direct"], samples[self.baseline_name]
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
        "kernels/jit/include/sgl_kernel/glm_wgmma_pv.cuh",
        "srt/remnant/glm.py",
        "srt/remnant/glm_pool.py",
        "srt/remnant/reference.py",
        "srt/layers/layernorm.py",
    )
    return {p: hashlib.sha256((root / p).read_bytes()).hexdigest() for p in paths}


def benchmark_source_hashes(root: Path) -> dict[str, str]:
    """Bind fixture producers and statistics as well as the measured runtime.

    A diagnostic run may lack a build manifest, so its local source identity must
    still include shared selection, normalization and configuration dependencies.
    The complete compiled artifact identity is additionally required for resume.
    """
    paths = (
        "benchmark/remnant/glm_attention.py",
        "benchmark/remnant/glm_upstream.py",
        "benchmark/remnant/timing.py",
        "benchmark/remnant/cases.py",
        "srt/remnant/config.py",
        "srt/remnant/glm_config.py",
        "srt/remnant/bitmap.py",
    )
    return {
        **runtime_hashes(root),
        **{p: hashlib.sha256((root / p).read_bytes()).hexdigest() for p in paths},
    }


def source_manifest() -> dict:
    import torch

    build_path = os.environ.get("REMNANT_BUILD_MANIFEST_PATH")
    return dict(
        sources=benchmark_source_hashes(Path(__file__).resolve().parents[2]),
        compiled_build=json.loads(Path(build_path).read_text()) if build_path else {},
        device=torch.cuda.get_device_name(),
        torch=torch.__version__,
        cuda=torch.version.cuda,
        triton=importlib.metadata.version("triton"),
        tilelang=importlib.metadata.version("tilelang"),
    )


def verify_resume_artifacts(source: dict) -> None:
    """Require the maintained build identity and verify its artifacts in place."""
    build = source.get("compiled_build", {})
    compiled = build.get("glm_compile") or {}
    variants = compiled.get("tilelang", [])
    if (
        build.get("model_family") != "glm53"
        or not build.get("sources")
        or not build.get("source_identities")
        or not build.get("binaries")
        or not compiled.get("cache_files")
        or compiled.get("attention_scale") != SCALE
        or len(compiled.get("triton", [])) != 2
        or not variants
    ):
        raise RuntimeError(
            "GLM --resume requires REMNANT_BUILD_MANIFEST_PATH pointing to a "
            "complete GLM build manifest with source identity, binaries and "
            "Triton/TileLang artifacts. Prepare it through the maintained GLM "
            "CPU workflow in scripts/modal/kernels.py, or start a fresh "
            "diagnostic run without --resume."
        )
    for key in ("torch", "cuda", "triton", "tilelang"):
        if not source.get(key) or build.get(key) != source[key]:
            raise RuntimeError(f"GLM resume build/runtime {key} identity mismatch")
    for variant in variants:
        if (
            not variant.get("cubin_sha256")
            or compiled["cache_files"].get(variant.get("cubin_path"))
            != variant["cubin_sha256"]
        ):
            raise RuntimeError(f"GLM resume has an unbound TileLang cubin: {variant}")
    for artifacts in (build["binaries"], compiled["cache_files"]):
        for name, expected in artifacts.items():
            path = Path(name)
            if (
                not path.is_file()
                or hashlib.sha256(path.read_bytes()).hexdigest() != expected
            ):
                raise RuntimeError(
                    f"GLM resume artifact missing or changed: {path}; "
                    "restore the verified build or start a fresh diagnostic run"
                )


def write_results(payload: dict, base: Path) -> None:
    base.parent.mkdir(parents=True, exist_ok=True)
    tmp = base.with_suffix(".json.partial")
    tmp.write_text(json.dumps(payload, indent=2) + "\n")
    tmp.replace(base.with_suffix(".json"))
    ratio_name = (
        "direct_over_upstream"
        if payload.get("protocol") == UPSTREAM_PROTOCOL
        else "direct_over_native"
    )
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
                ratio_name,
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
                        **{ratio_name: json.dumps(case["paired_ratios"])},
                        fixture=json.dumps(case["fixture"]),
                    )
                )


@cache
def _cpu_contract(owner: str):
    """Import a pure sibling contract without initializing the serving package."""
    import importlib.util
    import sys

    name = __name__ + "_" + owner
    path = (
        Path(__file__).parents[2] / "srt/remnant/glm_config.py"
        if owner == "glm_config"
        else Path(__file__).with_name(owner + ".py")
    )
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@cache
def _summary_function() -> Callable[[list[float]], dict]:
    """Load the shared CPU statistics without executing SGLang's package startup.

    Local preflights import this contract by file path on hosts with no serving
    dependencies. Keep the same summary implementation as GPU measurement.
    """
    return _cpu_contract("timing").summary


def validate_timing(payload: dict, *, prefix: bool = False) -> None:
    phase = payload.get("phase", "all")
    protocol, modes, expected = experiment(payload.get("suite", "glm-attention"), phase)
    baseline = modes[0]
    ratio_name = f"direct_over_{baseline}"
    wgmma = {}
    upstream_contract = None
    if protocol == UPSTREAM_PROTOCOL:
        upstream_contract = _cpu_contract("glm_upstream")
        source = payload.get("source", {})
        if source.get("upstream") != upstream_contract.identity():
            raise RuntimeError("Missing pristine upstream baseline identity")
        wgmma = (
            source.get("compiled_build", {})
            .get("upstream_compile", {})
            .get("decode_wgmma", {})
        )
        expected_candidate = upstream_contract.WGMMA_CANDIDATE if wgmma else None
        if payload.get("direct_candidate") != expected_candidate or (
            wgmma and set(wgmma) != {str(b) for b in upstream_contract.WGMMA_INNER}
        ):
            raise RuntimeError("Missing or mismatched WGMMA candidate identity")
    if payload.get("protocol") != protocol or (
        not prefix and payload.get("status") != "passed"
    ):
        raise RuntimeError("GLM timing protocol incomplete")
    if phase == "prefill" and payload.get("packed_prefill_tile") != 64:
        raise RuntimeError("Full GLM prefill requires the 64-row Packed tile")
    expected_count = len(expected)
    if prefix:
        if len(payload["cases"]) > len(expected):
            raise RuntimeError("GLM checkpoint exceeds the required matrix")
        expected = expected[: len(payload["cases"])]
    if [c["name"] for c in payload["cases"]] != [c.name for c in expected]:
        raise RuntimeError(
            f"GLM timing requires the ordered {expected_count}-case {phase} matrix"
        )
    for case, definition in zip(payload["cases"], expected):
        if (
            not case.get("valid")
            or set(case.get("paths", {})) != set(modes)
            or case.get("case") != asdict(definition)
        ):
            raise RuntimeError(f"Incomplete GLM case {case['name']}")
        fixture = case["fixture"]
        if protocol == UPSTREAM_PROTOCOL:
            inner = 1 if definition.queries == 1 else 2
            geometry = {
                mode: dict(tile_rows=32, inner_iter=inner, partial_groups=66 // inner)
                for mode in modes
            }
            compiled = (
                payload.get("source", {})
                .get("compiled_build", {})
                .get("glm_compile", {})
            )
            variants = set(compiled.get("decode_combine", {}))
            expected_columns = None
            if variants:
                config = _cpu_contract("glm_config")
                tuned = {f"{g}x{c}" for g, c in config.H200_PACKED_DECODE_COMBINE}
                if variants == tuned:
                    direct_inner = config.H200_PACKED_DECODE_INNER[definition.queries]
                    geometry["direct"].update(
                        inner_iter=direct_inner, partial_groups=66 // direct_inner
                    )
                elif variants != {"66x128", "33x512"}:
                    raise RuntimeError("Unknown upstream decode combine geometry")
                expected_columns = 128 if definition.queries == 1 else 512
            if wgmma and definition.queries > 1:
                geometry["direct"] = upstream_contract.wgmma_geometry(definition.queries)
                expected_columns = 512
            if (
                fixture.get("changing_input_replay") != "passed"
                or fixture.get("direct_combine_columns") != expected_columns
                or fixture.get("path_geometry") != geometry
                or "H200" not in payload.get("source", {}).get("device", "")
            ):
                raise RuntimeError(
                    f"Missing upstream decode graph/device geometry: {case['name']}"
                )
        if phase == "prefill":
            native_inner = fixture.get("inner_iter")
            if native_inner not in (33, 66) or fixture.get("path_geometry") != {
                "native": dict(
                    tile_rows=32,
                    inner_iter=native_inner,
                    partial_groups=66 // native_inner,
                ),
                "direct": dict(tile_rows=64, inner_iter=17, partial_groups=2),
            }:
                raise RuntimeError(f"Invalid full GLM prefill geometry {case['name']}")
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
            or orders.count([baseline, "direct"]) != ROUNDS // 2
            or orders.count(["direct", baseline]) != ROUNDS // 2
        ):
            raise RuntimeError(f"Unbalanced GLM timing {case['name']}")
        if (
            case["execution"] != "cuda_graph"
            or case["calls_per_sample"] != case["cycles_per_sample"] * banks
            or case["calls_per_sample"] < 100
        ):
            raise RuntimeError(f"Invalid GLM sample normalization {case['name']}")
        for mode, path in case["paths"].items():
            if len(path["samples"]) != ROUNDS or any(
                not math.isfinite(x) or x <= 0 for x in path["samples"]
            ):
                raise RuntimeError(f"Missing/invalid GLM rounds {case['name']}")
            if path != _summary_function()(path["samples"]):
                raise RuntimeError(
                    f"Invalid GLM derived statistics {case['name']}/{mode}"
                )
        ratios = case["paired_ratios"][ratio_name]["samples"]
        if ratios != [
            d / n
            for d, n in zip(
                case["paths"]["direct"]["samples"], case["paths"][baseline]["samples"]
            )
        ]:
            raise RuntimeError(f"Invalid GLM paired ratios {case['name']}")
        if case["paired_ratios"][ratio_name] != _summary_function()(ratios):
            raise RuntimeError(f"Invalid GLM derived paired statistics {case['name']}")
    if phase == "decode" and (not prefix or "summary" in payload):
        if payload.get("summary") != weighted_overhead(
            payload["cases"], baseline=baseline
        ):
            raise RuntimeError("Invalid GLM derived decode summary")
    if wgmma and not prefix and payload.get("upstream_decode_score") != upstream_decode_score(
        payload["cases"], target_ratio=upstream_contract.WGMMA_TARGET_RATIO
    ):
        raise RuntimeError("Invalid WGMMA upstream decode score")


def resume_prefix(
    previous: dict,
    source: dict,
    phase: str = "all",
    *,
    packed_prefill_tile: int = 32,
    suite: str = "glm-attention",
) -> list[dict]:
    """Reuse only validated cases from an identical fixture/toolchain snapshot."""
    if (
        previous.get("suite", "glm-attention") != suite
        or previous.get("phase", "all") != phase
        or previous.get("source") != source
        or previous.get("seed") != SEED
        or previous.get("rounds") != ROUNDS
        or previous.get("packed_prefill_tile", 32) != packed_prefill_tile
    ):
        raise RuntimeError(
            "GLM checkpoint source/artifact/seed/round identity mismatch"
        )
    verify_resume_artifacts(source)
    records = list(previous.get("cases", []))
    # A killed/failed final case has no accepted measurement. Interior failures
    # are never skipped or converted into coverage.
    if records and not records[-1].get("valid"):
        records.pop()
    validate_timing({**previous, "cases": records}, prefix=True)
    return records


def weighted_overhead(records: list[dict], *, baseline: str = "native") -> dict:
    """Batch-weighted mean of paired round overheads; other axes equal weight."""

    ratio_name = f"direct_over_{baseline}"

    def score(rows):
        weight = sum(c["case"]["queries"] for c in rows)
        return (
            (
                100
                * sum(
                    c["case"]["queries"]
                    * (sum(c["paired_ratios"][ratio_name]["samples"]) / ROUNDS - 1)
                    for c in rows
                )
                / weight
            )
            if weight
            else None
        )

    rows = [c for c in records if c.get("valid") and c["case"]["phase"] == "decode"]
    return dict(
        batch_weighted_overhead_percent=score(rows),
        by_heads={
            str(h): score([c for c in rows if c["case"]["heads"] == h]) for h in (16, 8)
        },
        completed_cases=len(rows),
        unacceptable_2x_cases=[
            c["name"]
            for c in rows
            if sum(c["paired_ratios"][ratio_name]["samples"]) / ROUNDS >= 2
        ],
    )


def upstream_decode_score(
    records: list[dict], *, target_ratio: float = 1 / 1.20
) -> dict:
    """Equal-case geometric mean of the 30 paired Direct/Native ratios."""
    _, _, expected = experiment(UPSTREAM_SUITE, "decode")
    if [row["name"] for row in records] != [case.name for case in expected]:
        raise ValueError("Score requires all 24 upstream decode cases in order")
    means = []
    for row in records:
        samples = row["paired_ratios"]["direct_over_upstream"]["samples"]
        if (
            not row.get("valid")
            or len(samples) != ROUNDS
            or not all(math.isfinite(value) and value > 0 for value in samples)
        ):
            raise ValueError(f"Invalid upstream decode samples: {row['name']}")
        means.append(math.fsum(samples) / ROUNDS)
    ratio = math.exp(math.fsum(map(math.log, means)) / len(means))
    return {
        "geometric_mean_direct_over_upstream": ratio,
        "speedup": 1 / ratio,
        "meets_target": ratio <= target_ratio,
    }


def decode_trace_stages(events: list[dict], banks: int) -> dict:
    """Attribute one complete replay by launch order, even for identical symbols."""
    kernels = sorted(
        (
            event
            for event in events
            if event.get("cat") == "kernel" and event.get("ph") == "X"
        ),
        key=lambda event: event["ts"],
    )
    if (
        banks < 1
        or len(kernels) != 3 * banks
        or len({(event["pid"], event["tid"]) for event in kernels}) != 1
        or any(
            not math.isfinite(event["dur"]) or event["dur"] <= 0 for event in kernels
        )
    ):
        raise RuntimeError("Expected three CUDA kernels per bank on one graph stream")
    stages = {}
    for position, name in enumerate(
        ("query_conversion", "partial_attention", "combine")
    ):
        launches = kernels[position::3]
        symbols = {event["name"] for event in launches}
        if len(symbols) != 1:
            raise RuntimeError(f"Inconsistent {name} kernel across graph banks")
        stages[name] = {
            "symbol": symbols.pop(),
            "us_per_call": math.fsum(event["dur"] for event in launches) / banks,
        }
    return stages


def profile_upstream_decode(
    matrix: list[Case], base: Path, deadline: float, source: dict
):
    """Trace representative steady-state graph replays separately from timing."""
    import torch
    from sglang.benchmark.remnant.glm_upstream import WGMMA_CANDIDATE

    selected = [
        case for case in matrix if case.queries in (1, 8, 32, 64) and case.tail == 0
    ]
    wgmma = source["compiled_build"]["upstream_compile"]["decode_wgmma"]
    payload = dict(
        protocol="remnant-glm-upstream-decode-profile-v1",
        direct_candidate=WGMMA_CANDIDATE,
        source=source,
        status="running",
        cases=[],
    )
    record_path = base.with_suffix(".json")
    trace_dir = base.with_suffix(".traces")
    trace_dir.mkdir()
    try:
        with torch.inference_mode():
            for case in selected:
                fixture = Fixture(
                    case, deadline, gpu_family="H200", upstream=True, wgmma=wgmma
                )
                fixture.capture()
                record = dict(name=case.name, fixture=fixture.identity, paths={})
                payload["cases"].append(record)
                for mode in fixture.modes:
                    fixture.check()
                    graph = fixture.graphs[mode]
                    graph.replay()
                    torch.cuda.synchronize()
                    with torch.profiler.profile(
                        activities=[torch.profiler.ProfilerActivity.CUDA]
                    ) as profiler:
                        graph.replay()
                        torch.cuda.synchronize()
                    trace = trace_dir / f"{case.name}-{mode}.json"
                    profiler.export_chrome_trace(str(trace))
                    stages = decode_trace_stages(
                        json.loads(trace.read_text())["traceEvents"], fixture.banks
                    )
                    if mode == "direct" and case.queries > 1 and (
                        stages["partial_attention"]["symbol"]
                        != "glm_packed_partial_wgmma_kernel"
                    ):
                        raise RuntimeError(f"{case.name}: trace did not run the WGMMA partial")
                    record["paths"][mode] = {
                        "trace": str(trace.relative_to(base.parent)),
                        "stages": stages,
                    }
                del graph, fixture
                gc.collect()
                torch.cuda.empty_cache()
        payload["status"] = "passed"
    except BaseException as error:
        payload.update(status="incomplete", error=repr(error))
        raise
    finally:
        record_path.write_text(json.dumps(payload, indent=2) + "\n")


def main() -> None:
    args = parser().parse_args()
    protocol, modes, matrix = experiment(args.suite, args.phase)
    upstream = args.suite == UPSTREAM_SUITE
    baseline = modes[0]
    wgmma = None
    target_ratio = 1 / 1.20
    if args.profile_decode and (not upstream or args.resume or args.check_imports):
        raise ValueError("Decode profiling requires a fresh upstream decode run")
    if args.modes is not None and args.modes != ",".join(modes):
        raise ValueError(f"{args.suite} requires {modes}")
    if upstream and args.packed_prefill_tile != 32:
        raise ValueError("Upstream decode requires production Direct tile32")
    if args.phase == "prefill":
        if args.packed_prefill_tile != 64:
            raise ValueError("Full GLM prefill requires --packed-prefill-tile 64")
    elif args.packed_prefill_tile != 32 and (
        args.phase != "prefill-small" or args.resume
    ):
        raise ValueError(
            "Wide Packed tile requires prefill or a fresh prefill-small run"
        )
    if args.check_imports:
        from sglang.kernels.ops.attention.dsa.tilelang_kernel import tilelang_sparse_fwd
        from sglang.srt.remnant.glm import compile_kernels

        if upstream:
            from sglang.benchmark.remnant.glm_upstream import load_baseline

            tilelang_sparse_fwd = load_baseline().tilelang_sparse_fwd
        assert callable(tilelang_sparse_fwd) and callable(compile_kernels)
        print(json.dumps(dict(protocol=protocol, cases=len(matrix), modes=modes)))
        return
    limit = 2400 if args.phase == "prefill" else 1200
    if not args.output or not 60 <= args.max_seconds <= limit:
        raise ValueError(
            f"GLM requires --output and --max-seconds between 60 and {limit}"
        )
    import torch

    base = Path(args.output)
    if any(
        base.with_suffix(ext).exists() for ext in (".json", ".csv", ".json.partial")
    ):
        raise FileExistsError(f"Refusing to overwrite GLM run {base}")
    payload = dict(
        protocol=protocol,
        phase=args.phase,
        **({"suite": args.suite} if upstream else {}),
        packed_prefill_tile=args.packed_prefill_tile,
        status="running",
        seed=SEED,
        rounds=ROUNDS,
        scope="model-free complete graph attention; no store/indexer/model/TP communication",
        source=source_manifest(),
        cases=[],
    )
    if upstream:
        from sglang.srt.remnant.glm_config import (
            H200_PACKED_DECODE_COMBINE,
            H200_PACKED_DECODE_INNER,
        )

        build = payload["source"]["compiled_build"]
        compile_manifest = build.get("glm_compile", {})
        expected_inner = set(H200_PACKED_DECODE_INNER.values())
        partials = [
            variant
            for variant in compile_manifest.get("tilelang", [])
            if variant["packed"]
            and variant["heads"] == 16
            and variant.get("block_i", 32) == 32
            and variant["inner"] in expected_inner
        ]
        if (
            len(partials) != len(expected_inner)
            or {v["inner"] for v in partials} != expected_inner
        ):
            raise RuntimeError("Missing CPU-prepared Packed decode partials")
        cache_files = compile_manifest["cache_files"]
        if any(
            not v.get("libraries")
            or any(cache_files.get(path) != h for path, h in v["libraries"].items())
            for v in partials
        ):
            raise RuntimeError("Unbound CPU-prepared Packed decode partial libraries")
        combines = compile_manifest.get("decode_combine", {})
        if set(combines) != {f"{g}x{c}" for g, c in H200_PACKED_DECODE_COMBINE} or any(
            not variant.get("cubin_path")
            or not variant.get("cubin_sha256")
            or compile_manifest["cache_files"].get(variant["cubin_path"])
            != variant["cubin_sha256"]
            or not variant.get("library_path")
            or not variant.get("library_sha256")
            or compile_manifest["cache_files"].get(variant["library_path"])
            != variant["library_sha256"]
            for variant in combines.values()
        ):
            raise RuntimeError("Missing CPU-prepared Packed decode combine cubins")
        from sglang.benchmark.remnant.glm_upstream import (
            WGMMA_CANDIDATE,
            WGMMA_TARGET_RATIO,
            identity,
            load_baseline,
            verify_compilation,
        )

        module = load_baseline()
        compiled_upstream = build.get("upstream_compile", {})
        verify_compilation(compiled_upstream, require_wgmma=True)
        wgmma = compiled_upstream["decode_wgmma"]
        payload["direct_candidate"] = WGMMA_CANDIDATE
        target_ratio = WGMMA_TARGET_RATIO
        payload["source"]["upstream"] = identity()
        payload["source"]["upstream_module_path"] = module.__file__
    if args.profile_decode:
        profile_upstream_decode(
            matrix, base, time.monotonic() + args.max_seconds, payload["source"]
        )
        return
    if args.resume:
        checkpoint = Path(args.resume)
        payload["cases"] = resume_prefix(
            json.loads(checkpoint.read_text()),
            payload["source"],
            args.phase,
            packed_prefill_tile=args.packed_prefill_tile,
            suite=args.suite,
        )
        payload["resumed_from"] = dict(
            path=str(checkpoint),
            sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
            accepted_cases=len(payload["cases"]),
        )
        if args.phase == "decode":
            payload["summary"] = weighted_overhead(payload["cases"], baseline=baseline)
    deadline = time.monotonic() + args.max_seconds
    try:
        with torch.inference_mode():
            for case in matrix[len(payload["cases"]) :]:
                record = dict(name=case.name, case=asdict(case), valid=False)
                payload["cases"].append(record)
                write_results(payload, base)
                fixture = Fixture(
                    case,
                    deadline,
                    packed_prefill_tile=args.packed_prefill_tile,
                    gpu_family="H200"
                    if upstream or args.phase in ("prefill", "prefill-small")
                    else "H100",
                    upstream=upstream,
                    wgmma=wgmma,
                )
                record["fixture"] = fixture.identity
                record.update(fixture.time())
                record["valid"] = True
                if args.phase == "decode":
                    payload["summary"] = weighted_overhead(
                        payload["cases"], baseline=baseline
                    )
                write_results(payload, base)
                if upstream or args.phase == "prefill":
                    print(
                        f"{len(payload['cases'])}/{len(matrix)} {case.name}", flush=True
                    )
                del fixture
                gc.collect()
                torch.cuda.empty_cache()
        payload["status"] = "passed"
        if upstream:
            payload["upstream_decode_score"] = upstream_decode_score(
                payload["cases"], target_ratio=target_ratio
            )
        validate_timing(payload)
    except BaseException as error:
        payload.update(status="incomplete", error=repr(error))
        raise
    finally:
        write_results(payload, base)
