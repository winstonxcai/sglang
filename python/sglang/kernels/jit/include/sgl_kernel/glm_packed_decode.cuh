// Author: Winston Cai.
// Device-only raw FP8 expansion for GLM's 256-byte payload + 64-byte bitmap.
#pragma once
#include <stdint.h>

__device__ __forceinline__ uint32_t glm_nibble_ranks(uint32_t x) {
  return (x & 8u) * 0x222u + (x & 4u) * 0x440u + (x & 2u) * 0x800u;
}
__device__ __forceinline__ uint32_t glm_expand4(
    uint32_t a, uint32_t b, uint32_t selectors, uint32_t mask) {
  uint32_t signs = (mask * 0x80402010u) & 0x80808080u;
  uint32_t bytes;
  asm("prmt.b32 %0, %1, %2, 0xba98;" : "=r"(bytes) : "r"(signs), "r"(0u));
  return __byte_perm(a, b, selectors) & bytes;
}
__device__ __forceinline__ uint2 glm_expand8(
    const uint8_t* row, int rank, uint32_t mask) {
  // Sixteen zero padding bytes make both 8-byte window reads safe even for
  // empty/malformed bitmaps. Clamp also preserves zero for rank >= 256.
  rank = min(rank, 256);
  const uint2* p = reinterpret_cast<const uint2*>(row + (rank & ~7));
  uint2 a = p[0], b = p[1];
  bool upper = (rank & 4) != 0;
  uint32_t w0 = upper ? a.y : a.x;
  uint32_t w1 = upper ? b.x : a.y;
  uint32_t w2 = upper ? b.y : b.x;
  uint32_t shift = (rank & 3) * 8;
  uint32_t lo = __funnelshift_r(w0, w1, shift);
  uint32_t hi = __funnelshift_r(w1, w2, shift);
  uint32_t selectors = (glm_nibble_ranks(mask >> 4) << 16) |
      (glm_nibble_ranks(mask & 15) + __popc(mask >> 4) * 0x1111u);
  return uint2{glm_expand4(lo, hi, selectors >> 16, mask >> 4),
               glm_expand4(lo, hi, selectors & 0xffff, mask & 15)};
}

__device__ __forceinline__ int remnant_glm_expand_tile(
    const void* values_raw, const uint64_t* bitmap, const int32_t* indices,
    void* scratch_raw, void* tile0, void* tile1, void* tile2, void* tile3,
    int64_t slots) {
  const int row = threadIdx.x / 4, part = threadIdx.x % 4;
  const int32_t index = indices[row];
  const bool valid = index >= 0 && static_cast<int64_t>(index) < slots;
  uint8_t* scratch = static_cast<uint8_t*>(scratch_raw) + row * 272;
  uint64_t word0 = 0, word1 = 0;
  // Address multiplication must be 64-bit: B64 histories exceed 2 GiB.
  if (valid) {
    const uint8_t* src = static_cast<const uint8_t*>(values_raw) +
        static_cast<int64_t>(index) * 256 + part * 64;
    #pragma unroll
    for (int i = 0; i < 4; ++i)
      reinterpret_cast<uint4*>(scratch + part * 64)[i] =
          __ldg(reinterpret_cast<const uint4*>(src) + i);
    const ulonglong2 words = __ldg(reinterpret_cast<const ulonglong2*>(
        bitmap + static_cast<int64_t>(index) * 8 + part * 2));
    word0 = words.x; word1 = words.y;
  } else {
    #pragma unroll
    for (int i = 0; i < 4; ++i)
      reinterpret_cast<uint4*>(scratch + part * 64)[i] = uint4{0,0,0,0};
  }
  if (part == 0) *reinterpret_cast<uint4*>(scratch + 256) = uint4{0,0,0,0};
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
  uint8_t* tile = static_cast<uint8_t*>(part == 0 ? tile0 :
      part == 1 ? tile1 : part == 2 ? tile2 : tile3);
  #pragma unroll
  for (int fragment = 0; fragment < 8; ++fragment) {
    const uint64_t word = fragment < 4 ? word0 : word1;
    const int offset = (fragment & 3) * 16;
    const int prefix = prefix0 + (fragment < 4 ? 0 : count0);
    const int rank = prefix + (offset == 0 ? 0 : __popcll(word >> (64-offset)));
    const uint32_t bits = static_cast<uint32_t>(word >> (48-offset)) & 0xffff;
    const uint2 a = glm_expand8(scratch, rank, bits >> 8);
    const uint2 b = glm_expand8(scratch, rank + __popc(bits >> 8), bits & 255);
    const int j = fragment * 16;
    // Native's inferred 128-byte-row XOR layout; explicitly annotated in TIR.
    *reinterpret_cast<uint4*>(tile + row * 128 + (j ^ ((row & 7) << 4))) =
        uint4{a.x, a.y, b.x, b.y};
  }
  return 0;
}
