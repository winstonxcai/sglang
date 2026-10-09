"""Finite-FP32 D512/keep256 selector; no input rounding before selection.

Magnitude order uses the full 31 FP32 magnitude bits. Eight radix-16 passes
locate the cutoff; an ascending-coordinate tie rule is checked against the
production Torch reference. FP32 packing arithmetic stays unchanged.
"""

import triton
import triton.language as tl


@triton.jit
def _select_keep_fp32(x):
    magnitude = (x.to(tl.uint32, bitcast=True) & 0x7FFFFFFF).to(tl.int32)
    bins = tl.arange(0, 32)
    prefix = tl.full((), 0, tl.int32)
    remaining = tl.full((), 256, tl.int32)
    active = tl.full((512,), True, tl.int1)
    for shift in tl.static_range(28, -1, -4):
        digit = (magnitude >> shift) & 15
        counts = tl.histogram(tl.where(active, digit, 16), 32)
        cumulative = tl.cumsum(counts, 0)
        bucket = tl.min(tl.where((bins < 16) & (cumulative >= remaining), bins, 32), 0)
        below = tl.sum(tl.where(bins < bucket, counts, 0), 0)
        remaining -= below
        prefix = prefix | (bucket << shift)
        active = active & (digit == bucket)
    below_count = tl.sum((magnitude < prefix).to(tl.int32), 0)
    equal = magnitude == prefix
    equal_rank = tl.cumsum(equal.to(tl.int32), 0)
    drop = (magnitude < prefix) | (equal & (equal_rank <= 256 - below_count))
    return ~drop


@triton.jit
def _pack_fused_fp32_kernel(
    latent_ptr,
    norm_weight_ptr,
    plan_ptr,
    locations_ptr,
    values_ptr,
    bitmap_ptr,
    scales_ptr,
    n_rows,
    norm_eps: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    KEEP_K: tl.constexpr,
    TILE_SIZE: tl.constexpr,
    BITMAP_WORDS: tl.constexpr,
    FP8_MAX: tl.constexpr,
    IS_DECODE: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """RMSNorm, UE8M0/FP8 quantize and scatter one packed row.

    Selection is computed once from the unmodified FP32 latent.  ``seq_lens % 4`` guards decode non-boundary rows, matching the
    native compressor plan instead of allowing them to overwrite location 0.
    """
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK_D)
    in_bounds = offs < HEAD_DIM
    plan_base = plan_ptr + row * 4
    if IS_DECODE:
        seq_len = tl.load(plan_base).to(tl.int32)
        location = tl.load(locations_ptr + row).to(tl.int64)
    else:
        # CompressPlan word 0 is seq_len; word 1 contains ragged_id.
        seq_len = tl.load(plan_base).to(tl.int32)
        ragged_id = tl.load(plan_base + 1).to(tl.int32) & 0xFFFF
        location = tl.load(locations_ptr + ragged_id, mask=seq_len >= 0, other=-1).to(
            tl.int64
        )
    # Decode plans include sequence length in word 0 and only write C4
    # boundaries. Padded prefill CompressPlan entries have seq_len == -1.
    write_row = (row < n_rows) & (location >= 0)
    if IS_DECODE:
        write_row = write_row & (seq_len % 4 == 0)
    else:
        write_row = write_row & (seq_len >= 0)
    if write_row:
        x = tl.load(latent_ptr + row * HEAD_DIM + offs, mask=in_bounds, other=0.0)
        bits = _select_keep_fp32(x)
        x = tl.where(bits, x, 0.0).to(tl.float32)
        weight = tl.load(norm_weight_ptr + offs, mask=in_bounds, other=0.0).to(
            tl.float32
        )
        inv_rms = tl.rsqrt(tl.sum(x * x, axis=0) / HEAD_DIM + norm_eps)
        # Native DSV4 store quantizes a BF16 round trip of the normalized NoPE
        # coordinates. Apply it to all eight tiles, including our new tail tile.
        normalized = (x * inv_rms * weight).to(tl.bfloat16).to(tl.float32)

        tiles = normalized.reshape(8, 64)
        abs_max = tl.max(tl.abs(tiles), axis=1)
        raw_scale = tl.maximum(1.0e-4, abs_max) / FP8_MAX
        exponent = tl.ceil(tl.log2(raw_scale)).to(tl.int32)
        tile_codes = exponent + 127
        tile_scales = tl.exp2(exponent.to(tl.float32))
        quant_scale = tl.broadcast_to(tile_scales[:, None], (8, 64)).reshape(512)

        clipped = tl.maximum(tl.minimum(normalized / quant_scale, FP8_MAX), -FP8_MAX)
        fp8 = clipped.to(tl.float8e4nv)
        raw_code = fp8.to(tl.uint8, bitcast=True)
        rank = tl.cumsum(bits.to(tl.int32), axis=0) - 1

        tl.store(
            values_ptr + location * KEEP_K + rank,
            raw_code,
            mask=write_row & bits & (rank < KEEP_K),
        )

        lane = tl.arange(0, 64)
        word = tl.arange(0, 8)
        bit_values = tl.where(
            bits.reshape(8, 64),
            tl.full((8, 64), 1, tl.uint64) << (63 - lane[None, :]),
            tl.full((8, 64), 0, tl.uint64),
        )
        word_values = tl.sum(bit_values, axis=1)
        tl.store(bitmap_ptr + location * 8 + word, word_values, mask=write_row)
        tl.store(
            scales_ptr + location * 8 + word,
            tl.maximum(tile_codes, 0).to(tl.uint8),
            mask=write_row,
        )
