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
// A PRMT selector of four reads from the second operand, which is zero.
// Use one 32-bit shared word per mask so distinct nibbles use distinct banks.
__device__ __forceinline__ int remnant_glm_init_rank_table(uint32_t* table) {
  const unsigned int mask = threadIdx.x;
  if (mask < 16) {
    uint32_t selector = 0;
    uint32_t rank = 0;
#pragma unroll
    for (int coordinate = 0; coordinate < 4; ++coordinate) {
      const uint32_t keep = (mask >> (3 - coordinate)) & 1;
      selector |= (keep ? rank : 4) << (4 * coordinate);
      rank += keep;
    }
    table[mask] = selector;
  }
  return 0;
}

template <bool Prefill, bool UseRankTable = false>
__device__ __forceinline__ uint2 glm_expand8(
    const uint8_t* row, int rank, uint32_t mask, const uint32_t* rank_table) {
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
  if constexpr (UseRankTable) {
    const uint32_t upper = mask >> 4;
    // Clamp the shift at 32 so a full upper nibble selects hi verbatim.
    const uint32_t second = __funnelshift_rc(lo, hi, __popc(upper) * 8);
    return uint2{
        __byte_perm(lo, 0, rank_table[upper]),
        __byte_perm(second, 0, rank_table[mask & 15])};
  }
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

template <bool Prefill = false, bool GlobalIndices = false, bool UseRankTable = false>
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
    int64_t index_offset = 0,
    const uint32_t* rank_table = nullptr) {
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
    const uint2 a =
        glm_expand8<Prefill, UseRankTable>(scratch, rank, bits >> 8, rank_table);
    const int second_rank = rank + __popc(bits >> 8);
    const uint2 b = glm_expand8<Prefill, UseRankTable>(
        scratch, second_rank, bits & 255, rank_table);
    const int j = fragment * 16;
    // Native's inferred 128-byte-row XOR layout; explicitly annotated in TIR.
    *reinterpret_cast<uint4*>(tile + row * 128 + (j ^ ((row & 7) << 4))) = uint4{a.x, a.y, b.x, b.y};
  }
  return 0;
}


// Rows are staged in 32-row waves. The swizzle repeats every eight rows.
template <int Rows, bool GlobalIndices = false, bool UseRankTable = false>
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
    int64_t index_offset = 0,
    const uint32_t* rank_table = nullptr) {
#pragma unroll
  for (int first = 0; first < Rows; first += 32) {
    if (first != 0) __syncwarp();
    const int offset = first * 128;
    remnant_glm_expand_tile<true, GlobalIndices, UseRankTable>(
        values, bitmap, indices + index_offset + first, scratch,
        static_cast<uint8_t*>(tile0) + offset,
        static_cast<uint8_t*>(tile1) + offset,
        static_cast<uint8_t*>(tile2) + offset,
        static_cast<uint8_t*>(tile3) + offset, slots, 0, rank_table);
  }
  return 0;
}

// Transpose the 64x512 FP8 SW128 K tile into the 512x64 SW64 PV operand.
// A warp owns 16 history rows and processes two 16x16 byte tiles per step.
// Hopper's matrix instructions move 16-bit pairs, so permute the raw bytes
// between the transposed load and the store. FP8 bit patterns stay intact.
__device__ __forceinline__ int remnant_glm_transpose_wgmma_tile(
    const void* kv_raw, void* v_raw) {
  const auto* kv = static_cast<const uint8_t*>(kv_raw);
  auto* v = static_cast<uint8_t*>(v_raw);
  const int lane = threadIdx.x & 31;
  const int row_base = (threadIdx.x >> 5) * 16;
  // Eight lanes supply eight row addresses per b16 matrix. Within each
  // 16-row tile, load rows [0,1,4,5,...] in one matrix and [2,3,6,7,...]
  // in the other. A register pair then holds four consecutive history rows.
  const int matrix_row = lane & 7;
  const int matrix_parity = (lane >> 3) & 1;
  const int src_row = row_base + 4 * (matrix_row / 2) + matrix_row % 2 + 2 * matrix_parity;
#pragma unroll
  for (int base = 0; base < 512; base += 32) {
    const int src_col = base + (lane >> 4) * 16;
    const int kv_offset =
        (src_col / 128) * (64 * 128) + src_row * 128 + ((src_col % 128) ^ ((src_row & 7) << 4));
    const uint32_t kv_addr = static_cast<uint32_t>(__cvta_generic_to_shared(kv + kv_offset));
    uint32_t a0, b0, a1, b1;
    asm volatile(
        "ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];"
        : "=r"(a0), "=r"(b0), "=r"(a1), "=r"(b1) : "r"(kv_addr) : "memory");

    // Even and odd bytes become separate matrices whose rows are features
    // and whose columns are the consecutive history bytes.
    const uint32_t even0 = __byte_perm(a0, b0, 0x6420);
    const uint32_t odd0 = __byte_perm(a0, b0, 0x7531);
    const uint32_t even1 = __byte_perm(a1, b1, 0x6420);
    const uint32_t odd1 = __byte_perm(a1, b1, 0x7531);
    const int dst_col = src_col + 2 * matrix_row + matrix_parity;
    const int v_offset = dst_col * 64 + (row_base ^ ((dst_col & 6) << 3));
    const uint32_t v_addr = static_cast<uint32_t>(__cvta_generic_to_shared(v + v_offset));
    asm volatile(
        "stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};"
        :: "r"(v_addr), "r"(even0), "r"(odd0), "r"(even1), "r"(odd1) : "memory");
  }
  return 0;
}
