"""Triton kernels for persistent 328-byte FP8 records.

Author: Winston Cai.
"""

import triton.language as tl

import triton


@triton.jit
def _pack_fp8_kernel(
    latent_ptr,
    keep_mask_ptr,
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

    The input mask is the exact mask computed once at the compressor store
    boundary.  ``seq_lens % 4`` guards decode non-boundary rows, matching the
    native compressor plan instead of allowing them to overwrite location 0.
    """
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK_D)
    in_bounds = offs < HEAD_DIM
    bits = tl.load(keep_mask_ptr + row * HEAD_DIM + offs, mask=in_bounds, other=0).to(
        tl.int1
    )
    x = tl.load(latent_ptr + row * HEAD_DIM + offs, mask=in_bounds, other=0.0)
    x = tl.where(bits, x, 0.0).to(tl.float32)
    weight = tl.load(norm_weight_ptr + offs, mask=in_bounds, other=0.0).to(tl.float32)
    inv_rms = tl.rsqrt(tl.sum(x * x, axis=0) / HEAD_DIM + norm_eps)
    # Native DSV4 store quantizes a BF16 round trip of the normalized NoPE
    # coordinates. Apply it to all eight tiles, including our new tail tile.
    normalized = (x * inv_rms * weight).to(tl.bfloat16).to(tl.float32)

    tile = offs // TILE_SIZE
    scale_code = tl.zeros([BLOCK_D], tl.int32)
    quant_scale = tl.full([BLOCK_D], 1.0, tl.float32)
    for t in tl.static_range(0, BITMAP_WORDS):
        tile_mask = in_bounds & (tile == t)
        abs_max = tl.max(tl.where(tile_mask, tl.abs(normalized), 0.0), axis=0)
        raw_scale = tl.maximum(1.0e-4, abs_max) / FP8_MAX
        exponent = tl.ceil(tl.log2(raw_scale)).to(tl.int32)
        code = exponent + 127
        represented = tl.exp2(exponent.to(tl.float32))
        scale_code = tl.where(tile_mask, code, scale_code)
        quant_scale = tl.where(tile_mask, represented, quant_scale)

    clipped = tl.maximum(tl.minimum(normalized / quant_scale, FP8_MAX), -FP8_MAX)
    fp8 = clipped.to(tl.float8e4nv)
    raw_code = fp8.to(tl.uint8, bitcast=True)
    rank = tl.cumsum(bits.to(tl.int32), axis=0) - 1

    plan_base = plan_ptr + row * (4 if IS_DECODE else 2)
    if IS_DECODE:
        seq_len = tl.load(plan_base).to(tl.int32)
        location = tl.load(locations_ptr + row).to(tl.int64)
    else:
        # Prefill WritePlan word 0 stores ragged_id (low 16 bits); word 1 is
        # the native write location and is not the packed C4 destination.
        ragged_id = tl.load(plan_base).to(tl.int32) & 0xFFFF
        location = tl.load(locations_ptr + ragged_id).to(tl.int64)
    # Decode plans include sequence length in word 0 and only write C4
    # boundaries. Prefill WritePlan entries are already filtered to valid
    # writes, so their word 0 is ragged_id rather than sequence length.
    write_row = (row < n_rows) & (location >= 0)
    if IS_DECODE:
        write_row = write_row & (seq_len % 4 == 0)
    tl.store(
        values_ptr + location * KEEP_K + rank,
        raw_code,
        mask=write_row & bits & (rank < KEEP_K),
    )

    lane = offs % 64
    word = offs // 64
    bit_value = tl.where(
        bits,
        tl.full([BLOCK_D], 1, tl.uint64) << (63 - lane),
        tl.zeros([BLOCK_D], tl.uint64),
    )
    for w in tl.static_range(0, BITMAP_WORDS):
        word_value = tl.sum(tl.where(word == w, bit_value, 0), axis=0)
        tl.store(
            bitmap_ptr + location * BITMAP_WORDS + w,
            word_value,
            mask=write_row,
        )
        tile_scale = tl.max(tl.where(tile == w, scale_code, 0), axis=0)
        tl.store(
            scales_ptr + location * BITMAP_WORDS + w,
            tile_scale.to(tl.uint8),
            mask=write_row,
        )


@triton.jit
def _unpack_gather_bf16_kernel(
    values_ptr,
    bitmap_ptr,
    scales_ptr,
    physical_indices_ptr,
    raw_indices_ptr,
    topk_lengths_ptr,
    output_ptr,
    n_queries,
    selected_k,
    HEAD_DIM: tl.constexpr,
    KEEP_K: tl.constexpr,
    BITMAP_WORDS: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """Gather packed rows and reconstruct pre-RoPE BF16 rows."""
    selected_row = tl.program_id(0)
    query = selected_row // selected_k
    k = selected_row % selected_k
    offs = tl.arange(0, BLOCK_D)
    in_bounds = offs < HEAD_DIM
    physical = tl.load(physical_indices_ptr + selected_row).to(tl.int64)
    raw = tl.load(raw_indices_ptr + selected_row).to(tl.int32)
    topk_len = tl.load(topk_lengths_ptr + query).to(tl.int32)
    valid = (query < n_queries) & (k < topk_len) & (physical >= 0) & (raw >= 0)
    word = offs // 64
    lane = offs % 64
    bitmap = tl.load(
        bitmap_ptr + physical * BITMAP_WORDS + word,
        mask=valid & in_bounds,
        other=0,
    )
    kept = ((bitmap >> (63 - lane)) & 1).to(tl.int1)
    rank = tl.cumsum(kept.to(tl.int32), axis=0) - 1
    code = tl.load(
        values_ptr + physical * KEEP_K + rank,
        mask=valid & kept & in_bounds,
        other=0,
    ).to(tl.uint8)
    fp8 = code.to(tl.float8e4nv, bitcast=True).to(tl.float32)
    scale_code = tl.load(
        scales_ptr + physical * BITMAP_WORDS + word,
        mask=valid & in_bounds,
        other=0,
    ).to(tl.int32)
    scale = tl.exp2((scale_code - 127).to(tl.float32))
    value = tl.where(valid & kept & in_bounds, fp8 * scale, 0.0)
    tl.store(output_ptr + selected_row * HEAD_DIM + offs, value, mask=in_bounds)


@triton.jit
def _rope_tail_complex_inplace_kernel(
    dense_ptr,
    raw_indices_ptr,
    freq_ptr,
    n_rows,
    HEAD_DIM: tl.constexpr,
    NOPE_DIM: tl.constexpr,
    ROPE_PAIRS: tl.constexpr,
):
    """Apply RoPE from a contiguous complex64 table without real/imag copies.

    PyTorch stores complex64 as interleaved float32 real/imag values. The
    wrapper passes a zero-copy ``view_as_real`` view, so each pair is loaded at
    offsets ``2 * freq_base`` and ``2 * freq_base + 1``.
    """
    row = tl.program_id(0)
    pair = tl.arange(0, ROPE_PAIRS)
    raw = tl.load(raw_indices_ptr + row).to(tl.int64)
    valid = (row < n_rows) & (raw >= 0)
    position = raw * 4
    base = row * HEAD_DIM + NOPE_DIM + pair * 2
    x_real = tl.load(dense_ptr + base, mask=valid, other=0.0).to(tl.float32)
    x_imag = tl.load(dense_ptr + base + 1, mask=valid, other=0.0).to(tl.float32)
    freq_base = position * ROPE_PAIRS + pair
    c = tl.load(freq_ptr + freq_base * 2, mask=valid, other=1.0).to(tl.float32)
    s = tl.load(freq_ptr + freq_base * 2 + 1, mask=valid, other=0.0).to(tl.float32)
    tl.store(dense_ptr + base, x_real * c - x_imag * s, mask=valid)
    tl.store(dense_ptr + base + 1, x_real * s + x_imag * c, mask=valid)
