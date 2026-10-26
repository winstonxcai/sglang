"""GLM raw FP8 producer and bounded reconstruction on the caller's stream.

Author: Winston Cai.

Physical locations index independent uint8 [slots,256] and uint64 [slots,8]
planes. The producer's bool [rows,512] mask is computed exactly once, before
Native RMSNorm. Values follow ascending coordinates; bit 63 denotes lane 0.
Invalid writes are ignored. Invalid selections reconstruct to positive zero.
Buffers are owned by the pool; no dense persistent mirror exists.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cache

import torch
import triton
import triton.language as tl

from .glm_config import MAX_QUERY_CHUNK, SELECTION_COLUMNS
from .reference import topmag_keep_mask


@triton.jit
def _store(X, M, L, V, B, SX: tl.constexpr, SM: tl.constexpr, SLOTS):
    row = tl.program_id(0)
    c = tl.arange(0, 512)
    loc = tl.load(L + row)
    keep = tl.load(M + row * SM + c)
    rank = tl.cumsum(keep.to(tl.int32)) - 1
    x = tl.load(X + row * SX + c)
    valid = (loc >= 0) & (loc < SLOTS)
    tl.store(V + loc * 256 + rank, x, valid & keep)
    lanes = tl.arange(0, 64)
    words = tl.arange(0, 8)
    bits = tl.reshape(keep, (8, 64)).to(tl.uint64) << (63 - lanes[None, :])
    bitmap = tl.sum(bits, 1)
    tl.store(B + loc * 8 + words, bitmap, valid)


@triton.jit
def _reconstruct(V, B, I, O, SLOTS):
    row = tl.program_id(0)
    c = tl.arange(0, 512)
    # Read indices are int32; widen before multiplying into the value plane.
    idx = tl.load(I + row).to(tl.int64)
    valid = (idx >= 0) & (idx < SLOTS)
    word = tl.load(B + idx * 8 + c // 64, valid, 0)
    keep = ((word >> (63 - c % 64)) & 1).to(tl.int32)
    rank = tl.cumsum(keep) - 1
    value = tl.load(V + idx * 256 + rank, valid & (keep != 0), 0)
    tl.store(O + row * 512 + c, value)


@cache
def compile_kernels() -> tuple[
    triton.compiler.CompiledKernel, triton.compiler.CompiledKernel
]:
    """CPU-buildable SM90 artifacts; identical AST signatures at build and load.

    Explicit target avoids device discovery on CPU builders. Packed currently
    admits only contiguous uint8 Native FP8 codes, bool masks and int64 write
    locations. Read indices are contiguous int32. The Modal preflight records
    these hashes and image cache files before any H100 is requested.
    """
    from triton.backends.compiler import GPUTarget
    from triton.compiler import ASTSource

    target = GPUTarget("cuda", 90, 32)
    store = triton.compile(
        ASTSource(
            _store,
            signature={
                "X": "*u8",
                "M": "*i1",
                "L": "*i64",
                "V": "*u8",
                "B": "*u64",
                "SLOTS": "i32",
            },
            constexprs={"SX": 512, "SM": 512},
        ),
        target=target,
    )
    unpack = triton.compile(
        ASTSource(
            _reconstruct,
            signature={
                "V": "*u8",
                "B": "*u64",
                "I": "*i32",
                "O": "*u8",
                "SLOTS": "i32",
            },
        ),
        target=target,
    )
    return store, unpack


@dataclass(frozen=True)
class PackedCache:
    values: torch.Tensor
    bitmaps: torch.Tensor
    workspace: torch.Tensor | None
    implementation: str


def normalize_and_store(
    raw: torch.Tensor,
    norm: torch.nn.Module,
    loc: torch.Tensor,
    values: torch.Tensor,
    bitmaps: torch.Tensor,
) -> torch.Tensor:
    """Compose original Native norm/conversion with one raw-latent selection."""
    raw = raw.reshape(-1, 512)
    keep = topmag_keep_mask(raw, 0.5)
    normalized = norm(raw.masked_fill(~keep, 0))
    if (
        normalized.dtype != torch.bfloat16
        or not normalized.is_contiguous()
        or loc.dtype != torch.int64
    ):
        raise ValueError(
            "GLM Packed requires contiguous Native BF16 norm output and int64 write locations"
        )
    if raw.shape[0]:
        store, _ = compile_kernels()
        # Preserve Native overflow and signed-zero conversion exactly.
        codes = normalized.to(torch.float8_e4m3fn).view(torch.uint8)
        store[(raw.shape[0], 1, 1)](codes, keep, loc, values, bitmaps, values.shape[0])
    return normalized


def reconstruct(
    cache: PackedCache, indices: torch.Tensor, output: torch.Tensor
) -> torch.Tensor:
    """Fill caller-owned contiguous uint8 rows; handles duplicate/invalid indices."""
    if indices.numel():
        _, unpack = compile_kernels()
        unpack[(indices.numel(), 1, 1)](
            cache.values, cache.bitmaps, indices, output, cache.values.shape[0]
        )
    return output.view(torch.float8_e4m3fn).reshape(-1, 1, 512)


def attention(
    q: torch.Tensor, cache: PackedCache, indices: torch.Tensor, sm_scale: float
) -> torch.Tensor:
    from sglang.kernels.ops.attention.dsa.tilelang_kernel import tilelang_sparse_fwd

    if indices.shape[-1] != SELECTION_COLUMNS:
        raise ValueError("GLM Packed expects 2112 padded history/tail columns")
    if cache.implementation == "direct":
        return tilelang_sparse_fwd(
            q,
            cache.values.view(torch.float8_e4m3fn).unsqueeze(1),
            indices,
            sm_scale,
            512,
            packed_bitmap=cache.bitmaps,
        )
    if cache.workspace is None:
        raise RuntimeError(
            "GLM Adapter workspace must be allocated before graph capture"
        )
    chunks = []
    for start in range(0, q.shape[0], MAX_QUERY_CHUNK):
        end = min(start + MAX_QUERY_CHUNK, q.shape[0])
        selected = indices[start:end].contiguous()
        rows = selected.numel()
        kv = reconstruct(cache, selected, cache.workspace[:rows])
        # Rebase each selection, preserving invalid entries; no CPU reads.
        local = torch.arange(rows, device=q.device, dtype=torch.int32).reshape(
            selected.shape
        )
        local = torch.where(
            (selected >= 0) & (selected < cache.values.shape[0]), local, -1
        )
        chunks.append(tilelang_sparse_fwd(q[start:end], kv, local, sm_scale, 512))
    return torch.cat(chunks, dim=1)
