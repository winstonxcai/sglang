// Author: Winston Cai.
// Device-only raw FP8 expansion for GLM's 256-byte payload + 64-byte bitmap.
#pragma once
#include <stdint.h>

// The four bitmap bits describe coordinates in MSB-first order. Each output
// selector nibble (low nibble first) counts retained coordinates BEFORE that
// coordinate. Contributions 0x222, 0x440 and 0x800 add one to all later ranks;
// the final bitmap bit cannot affect a preceding-coordinate count.
// Example: mask 0b1011 -> ranks [0, 1, 1, 2] -> selectors 0x2110.
// Selecting compact bytes [a, b, c, ...] and then masking emits [a, 0, b, c].
__device__ __forceinline__ uint32_t glm_nibble_ranks(uint32_t x) {
  return (x & 8u) * 0x222u + (x & 4u) * 0x440u + (x & 2u) * 0x800u;
}
template <bool Prefill>
__device__ __forceinline__ uint32_t glm_expand4(uint32_t a, uint32_t b, uint32_t selectors, uint32_t mask) {
  // Scatter bitmap bits 3..0 into the sign bits of output bytes 0..3.
  // PRMT selectors 8,9,a,b replicate those sign bits to whole 0xff/0x00 bytes.
  // AND after byte selection supplies raw FP8 zero without a numerical cast.
  uint32_t signs = mask * 0x80402010u;
  // Sign-replicating PRMT ignores the lower seven bits of each source byte.
  if constexpr (!Prefill) signs &= 0x80808080u;
  uint32_t bytes;
  asm("prmt.b32 %0, %1, %2, 0xba98;" : "=r"(bytes) : "r"(signs), "r"(0u));
  return __byte_perm(a, b, selectors) & bytes;
}
template <bool Prefill>
__device__ __forceinline__ uint2 glm_expand8(const uint8_t* row, int rank, uint32_t mask) {
  // Sixteen zero padding bytes make the aligned window reads safe even for
  // empty/malformed bitmaps. Clamp also preserves zero for rank >= 256.
  rank = min(rank, 256);
  uint32_t w0, w1, w2;
  if constexpr (Prefill) {
    // An eight-byte window starting within a four-byte word spans at most
    // three words. At rank 256 the reads land entirely in the zero padding.
    const uint32_t* words = reinterpret_cast<const uint32_t*>(row + (rank & ~3));
    w0 = words[0];
    w1 = words[1];
    w2 = words[2];
  } else {
    const uint2* p = reinterpret_cast<const uint2*>(row + (rank & ~7));
    uint2 a = p[0], b = p[1];
    bool upper = (rank & 4) != 0;
    w0 = upper ? a.y : a.x;
    w1 = upper ? b.x : a.y;
    w2 = upper ? b.y : b.x;
  }
  uint32_t shift = (rank & 3) * 8;
  uint32_t lo = __funnelshift_r(w0, w1, shift);
  uint32_t hi = __funnelshift_r(w1, w2, shift);
  // The upper bitmap nibble describes the first four output coordinates.
  // The second four start after popcount(upper), added to each selector nibble.
  const uint32_t second_base = __popc(mask >> 4) * 0x1111u;
  uint32_t selectors;
  if constexpr (Prefill) {
    // Each 16-bit half holds four selector nibbles. Neither half can
    // carry into the other because every selector is at most seven.
    const uint32_t masks = ((mask >> 4) << 16) | (mask & 15);
    selectors = (masks & 0x00080008u) * 0x222u +
                (masks & 0x00040004u) * 0x440u +
                (masks & 0x00020002u) * 0x800u + second_base;
  } else {
    selectors = (glm_nibble_ranks(mask >> 4) << 16) |
                (glm_nibble_ranks(mask & 15) + second_base);
  }
  return uint2{glm_expand4<Prefill>(lo, hi, selectors >> 16, mask >> 4), glm_expand4<Prefill>(lo, hi, selectors & 0xffff, mask & 15)};
}

template <bool Prefill = false, bool GlobalIndices = false>
__device__ __forceinline__ int remnant_glm_expand_tile(
    const void* values_raw,
    const uint64_t* bitmap,
    const int32_t* indices,
    void* scratch_raw,
    void* tile0,
    void* tile1,
    void* tile2,
    void* tile3,
    int64_t slots,
    int64_t index_offset = 0) {
  const int row = threadIdx.x / 4, part = threadIdx.x % 4;
  const int32_t* row_index = indices + index_offset + row;
  // Aligned prefill reads immutable global indices. Other callers stage
  // their indices in shared memory and use the ordinary load.
  const int32_t index = GlobalIndices ? __ldg(row_index) : *row_index;
  // Sign-extend before converting to unsigned so negative indices exceed
  // any positive int64 slot count, even for caches above 2 GiB.
  const bool valid = Prefill
      ? static_cast<uint64_t>(static_cast<int64_t>(index)) < static_cast<uint64_t>(slots)
      : index >= 0 && static_cast<int64_t>(index) < slots;
  uint8_t* scratch = static_cast<uint8_t*>(scratch_raw) + row * 272;
  uint64_t word0 = 0, word1 = 0;
  // Address multiplication must be 64-bit: B64 histories exceed 2 GiB.
  if (valid) {
    const uint8_t* src = static_cast<const uint8_t*>(values_raw) + static_cast<int64_t>(index) * 256;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      // Adjacent lanes stage adjacent vectors. Expansion still reads the
      // same ascending-coordinate payload from the same scratch addresses.
      const int vector = Prefill ? 4 * i + part : 4 * part + i;
      reinterpret_cast<uint4*>(scratch)[vector] =
          __ldg(reinterpret_cast<const uint4*>(src) + vector);
    }
    const ulonglong2 words =
        __ldg(reinterpret_cast<const ulonglong2*>(bitmap + static_cast<int64_t>(index) * 8 + part * 2));
    word0 = words.x;
    word1 = words.y;
  } else {
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      const int vector = Prefill ? 4 * i + part : 4 * part + i;
      reinterpret_cast<uint4*>(scratch)[vector] = uint4{0, 0, 0, 0};
    }
  }
  if (part == 0) *reinterpret_cast<uint4*>(scratch + 256) = uint4{0, 0, 0, 0};
  const int count0 = __popcll(word0);
  int count = count0 + __popcll(word1);
  // All lanes participate, including invalid rows. Four-lane groups never
  // cross a warp, and no subgroup depends on another subgroup's scratch.
  int previous = __shfl_up_sync(0xffffffffu, count, 1, 4);
  if (part >= 1) count += previous;
  previous = __shfl_up_sync(0xffffffffu, count, 2, 4);
  if (part >= 2) count += previous;
  const int prefix0 = count - count0 - __popcll(word1);
  __syncwarp();
  uint8_t* tile = static_cast<uint8_t*>(part == 0 ? tile0 : part == 1 ? tile1 : part == 2 ? tile2 : tile3);
  if constexpr (Prefill) {
    // Invalid rows have no payload. Initialize the full swizzled output
    // row before returning to the caller's collective synchronization.
    if (!valid) {
#pragma unroll
      for (int vector = 0; vector < 8; ++vector) {
        reinterpret_cast<uint4*>(tile + row * 128)[vector] = uint4{0, 0, 0, 0};
      }
      return 0;
    }
  }
#pragma unroll
  for (int fragment = 0; fragment < 8; ++fragment) {
    const uint64_t word = fragment < 4 ? word0 : word1;
    const int offset = (fragment & 3) * 16;
    const int prefix = prefix0 + (fragment < 4 ? 0 : count0);
    int rank = prefix;
    if constexpr (Prefill) {
      // Split the word into independent 32-bit halves. This avoids a running
      // dependency from the previous fragment's two eight-bit counts.
      const uint32_t high = static_cast<uint32_t>(word >> 32);
      const uint32_t low = static_cast<uint32_t>(word);
      if (offset >= 32) rank += __popc(high);
      if (offset & 16) rank += __popc((offset < 32 ? high : low) >> 16);
    } else if (offset != 0) {
      rank += __popcll(word >> (64 - offset));
    }
    const uint32_t bits = static_cast<uint32_t>(word >> (48 - offset)) & 0xffff;
    const uint2 a = glm_expand8<Prefill>(scratch, rank, bits >> 8);
    const int second_rank = rank + __popc(bits >> 8);
    const uint2 b = glm_expand8<Prefill>(scratch, second_rank, bits & 255);
    const int j = fragment * 16;
    // Native's inferred 128-byte-row XOR layout; explicitly annotated in TIR.
    *reinterpret_cast<uint4*>(tile + row * 128 + (j ^ ((row & 7) << 4))) = uint4{a.x, a.y, b.x, b.y};
  }
  return 0;
}


// Rows are staged in 32-row waves. The swizzle repeats every eight rows.
template <int Rows, bool GlobalIndices = false>
__device__ __forceinline__ int remnant_glm_expand_wide_tile(
    const void* values,
    const uint64_t* bitmap,
    const int32_t* indices,
    void* scratch,
    void* tile0,
    void* tile1,
    void* tile2,
    void* tile3,
    int64_t slots,
    int64_t index_offset = 0) {
#pragma unroll
  for (int first = 0; first < Rows; first += 32) {
    if (first != 0) __syncwarp();
    const int offset = first * 128;
    remnant_glm_expand_tile<true, GlobalIndices>(
        values, bitmap, indices + index_offset + first, scratch,
        static_cast<uint8_t*>(tile0) + offset,
        static_cast<uint8_t*>(tile1) + offset,
        static_cast<uint8_t*>(tile2) + offset,
        static_cast<uint8_t*>(tile3) + offset, slots);
  }
  return 0;
}
