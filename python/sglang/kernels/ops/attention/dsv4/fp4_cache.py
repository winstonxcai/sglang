"""SM90 V4.1 Native/Packed FP4 store and prefill reconstruction.

Author: Winston Cai.

Pages are SoA: P data rows, then P 32-byte E4M3 scale rows. Native data
rows contain 256 nibble bytes; Packed data rows contain 128 nibble bytes and
8 MSB-first uint64 bitmap words. Caller owns page/slot allocation and stream.
The store takes normalized BF16 latents and optional per-row complex RoPE.
"""
from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _fp4_magnitude(code):
    return tl.where(code == 0, 0., tl.where(code == 1, .5, tl.where(
        code == 2, 1., tl.where(code == 3, 1.5, tl.where(code == 4, 2.,
        tl.where(code == 5, 3., tl.where(code == 6, 4., 6.)))))))


@triton.jit
def _store(X, F, LOC, C, XS: tl.constexpr, FS: tl.constexpr,
           P: tl.constexpr, STRIDE: tl.constexpr, N: tl.constexpr,
           PACKED: tl.constexpr, ROTATE: tl.constexpr):
    r = tl.program_id(0)
    d = tl.arange(0, 512)
    loc = tl.load(LOC + r).to(tl.int64)
    valid = (loc > 0) & (loc < N * P)
    v = tl.load(X + r * XS + d).to(tl.float32)
    if ROTATE:
        tail = d >= 448
        j = (d - 448) // 2
        re = tl.load(X + r * XS + 448 + 2*j, tail, 0).to(tl.float32)
        im = tl.load(X + r * XS + 449 + 2*j, tail, 0).to(tl.float32)
        fr = tl.load(F + r * FS + 2*j, tail, 1.)
        fi = tl.load(F + r * FS + 2*j + 1, tail, 0.)
        rot = tl.where(d % 2 == 0, re * fr - im * fi, re * fi + im * fr)
        v = tl.where(tail, rot.to(tl.bfloat16).to(tl.float32), v)
    block = tl.reshape(v, (32, 16))
    amax = tl.max(tl.abs(block), 1)
    nonfinite = tl.sum((tl.abs(block) == float('inf')) | (block != block), 1) > 0
    scale = tl.minimum(tl.maximum(amax * (1./6.), 2.**-9), 448.)
    scale = tl.where(nonfinite, float('nan'), scale).to(tl.float8e4nv)
    scaled = tl.reshape(tl.div_rn(block, scale.to(tl.float32)[:, None]), (512,))
    a = tl.minimum(tl.abs(scaled), 6.)
    a = tl.where(scaled != scaled, 0., a)
    # Midpoint parity implements cvt.rn.satfinite.e2m1x2 without SM100 PTX.
    code = ((a > .25).to(tl.int32) + (a >= .75).to(tl.int32)
            + (a > 1.25).to(tl.int32) + (a >= 1.75).to(tl.int32)
            + (a > 2.5).to(tl.int32) + (a >= 3.5).to(tl.int32)
            + (a > 5.).to(tl.int32))
    sign = (scaled.to(tl.uint32, bitcast=True) >> 28) & 8
    code = code | sign.to(tl.int32)
    data_bytes: tl.constexpr = 192 if PACKED else 256
    base = C + (loc // P) * STRIDE
    row = loc % P
    if PACKED:
        deq = tl.abs(_fp4_magnitude(code & 7) * tl.gather(scale.to(tl.float32), d // 16, 0))
        # Nonfinite groups sort first, deterministically; their codes/scales are
        # still preserved exactly. Finite latents are the production contract.
        deq = tl.where(deq != deq, float('inf'), deq)
        key = (deq.to(tl.uint32, bitcast=True).to(tl.uint64) << 9) | (511-d).to(tl.uint64)
        ordered = tl.sort(key, descending=True)
        cutoff = tl.sum(tl.where(d == 255, ordered, 0), 0)
        keep = key >= cutoff
        coords = tl.sort(tl.where(keep, d, 512), descending=False)
        codes = tl.gather(code, tl.minimum(coords, 511), 0)
        i = tl.arange(0, 128)
        packed = tl.gather(codes, i*2, 0) | (tl.gather(codes, i*2+1, 0) << 4)
        tl.store(base + row*192 + i, packed.to(tl.uint8), valid)
        words = tl.sum(tl.reshape(keep.to(tl.uint64) << (63-d%64), (8,64)), 1)
        bitmap = (base + row*192 + 128).to(tl.pointer_type(tl.uint64))
        tl.store(bitmap + tl.arange(0,8), words, valid)
    else:
        i = tl.arange(0, 256)
        packed = tl.gather(code, i*2, 0) | (tl.gather(code, i*2+1, 0) << 4)
        tl.store(base + row*256 + i, packed.to(tl.uint8), valid)
    tl.store(base + P*data_bytes + row*32 + tl.arange(0,32),
             scale.to(tl.uint8, bitcast=True), valid)


@triton.jit
def _read(C, IDS, OUT, OS: tl.constexpr, P: tl.constexpr, STRIDE: tl.constexpr,
          N: tl.constexpr, PACKED: tl.constexpr):
    r = tl.program_id(0)
    d = tl.arange(0,512)
    loc = tl.load(IDS+r).to(tl.int64)
    valid = (loc >= 0) & (loc < N*P)
    loc = tl.where(valid, loc, 0)
    base = C + (loc//P)*STRIDE
    row = loc%P
    data_bytes: tl.constexpr = 192 if PACKED else 256
    if PACKED:
        bitmap = (base + row*192 + 128).to(tl.pointer_type(tl.uint64))
        word = tl.load(bitmap + d//64, valid, 0)
        keep = ((word >> (63-d%64)) & 1).to(tl.int32)
        rank = tl.cumsum(keep)-1
        byte = tl.load(base + row*192 + tl.maximum(rank,0)//2, valid & (keep != 0), 0)
        code = (byte >> ((tl.maximum(rank,0)%2)*4)) & 15
    else:
        keep = tl.full((512,), 1, tl.int32)
        byte = tl.load(base + row*256 + d//2, valid, 0)
        code = (byte >> ((d%2)*4)) & 15
    raw_scale = tl.load(base + P*data_bytes + row*32 + d//16, valid, 0)
    scale = raw_scale.to(tl.float8e4nv, bitcast=True).to(tl.float32)
    mag = _fp4_magnitude(code & 7)
    value = tl.where((code & 8) != 0, -mag, mag) * scale
    tl.store(OUT+r*OS+d, tl.where(valid & (keep != 0), value, 0.).to(tl.bfloat16))


def store_fp4_cache(x: torch.Tensor, cache: torch.Tensor, loc: torch.Tensor,
                    page_size: int, *, packed: bool = False,
                    freqs_cis: torch.Tensor | None = None) -> None:
    """Store [rows,512] BF16 into caller-owned uint8 pages; slot zero is padding.

    No allocation/synchronization occurs when frequencies are already contiguous.
    All input rows and frequency pairs must be valid even for padding locations.
    """
    if x.ndim != 2 or x.shape[1] != 512 or x.dtype != torch.bfloat16 or x.stride(1) != 1:
        raise ValueError('FP4 store requires BF16 [rows,512] latents contiguous in dimension 1')
    if cache.dtype != torch.uint8 or cache.ndim != 2 or not cache.is_contiguous():
        raise ValueError('FP4 cache requires contiguous uint8 [pages,page_bytes] storage')
    if page_size <= 0 or cache.shape[1] < page_size * (224 if packed else 288):
        raise ValueError('FP4 cache page geometry cannot hold the selected format')
    if loc.ndim != 1 or loc.numel() != x.shape[0] or not loc.is_contiguous() or loc.dtype not in (torch.int32,torch.int64):
        raise ValueError('FP4 locations must be contiguous int32/int64 [rows]')
    if x.device != cache.device or loc.device != x.device or not x.is_cuda:
        raise ValueError('FP4 store tensors must share one CUDA device')
    f = x
    if freqs_cis is not None:
        if freqs_cis.shape != (x.shape[0],32) or freqs_cis.dtype != torch.complex64 or not freqs_cis.is_contiguous() or freqs_cis.device != x.device:
            raise ValueError('FP4 RoPE requires contiguous complex64 [rows,32] frequencies on the latent device')
        f = torch.view_as_real(freqs_cis).view(-1,64)
    if x.shape[0]:
        _store[(x.shape[0],)](x,f,loc,cache,x.stride(0),f.stride(0),page_size,
            cache.stride(0),cache.shape[0],packed,freqs_cis is not None,
            num_warps=4,enable_fp_fusion=False)


def read_fp4_cache(cache: torch.Tensor, ids: torch.Tensor, page_size: int, *,
                   packed: bool = False, out: torch.Tensor | None = None) -> torch.Tensor:
    """Gather physical IDs to BF16 [rows,1,512]; invalid IDs become zero.

    Prefill callers supply a preallocated contiguous-along-row destination.
    Stored global records already include RoPE; reconstruction never rotates.
    """
    if cache.dtype != torch.uint8 or cache.ndim != 2 or not cache.is_contiguous():
        raise ValueError('FP4 reader requires contiguous uint8 pages')
    if page_size <= 0 or cache.shape[1] < page_size*(224 if packed else 288):
        raise ValueError('FP4 reader page geometry cannot hold the selected format')
    if ids.ndim != 1 or ids.dtype not in (torch.int32,torch.int64) or not ids.is_contiguous() or ids.device != cache.device or not ids.is_cuda:
        raise ValueError('FP4 reader IDs must be contiguous int32/int64 on the cache CUDA device')
    if out is None:
        out = torch.empty((ids.numel(),1,512),device=cache.device,dtype=torch.bfloat16)
    if out.shape != (ids.numel(),1,512) or out.dtype != torch.bfloat16 or out.device != cache.device or out.stride(-1) != 1:
        raise ValueError('FP4 reader output must be BF16 [rows,1,512] on the cache device')
    if ids.numel():
        _read[(ids.numel(),)](cache,ids,out,out.stride(0),page_size,cache.stride(0),cache.shape[0],packed,num_warps=4)
    return out
