// Author: Winston Cai.
#pragma once

#include <stdint.h>

__device__ __forceinline__ uint32_t remnant_glm_query_fp8_pair(uint32_t bits) {
  const uint32_t lo = bits & 0xffff;
  const uint32_t hi = bits >> 16;
  uint16_t packed;
  // BF16 bits extend exactly to FP32. Hopper packs the two rounded E4M3
  // codes with the first source in the high byte.
  asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;"
      : "=h"(packed)
      : "f"(__uint_as_float(hi << 16)), "f"(__uint_as_float(lo << 16)));
  // 464 rounds to 448 at the even tie. Values beyond it, including infinities
  // and NaNs, use Torch's signed E4M3FN NaN code instead of saturation.
  constexpr uint32_t last_finite_bf16 = 0x43e8;
  const uint32_t code_lo = (lo & 0x7fff) > last_finite_bf16
      ? (lo >> 8 & 0x80) | 0x7f
      : packed & 0xff;
  const uint32_t code_hi = (hi & 0x7fff) > last_finite_bf16
      ? (hi >> 8 & 0x80) | 0x7f
      : packed >> 8;
  return code_lo | (code_hi << 8);
}

// Stage one contiguous [16,512] BF16 query in TileLang's SW128 FP8 layout.
// Each lane loads eight aligned BF16 coordinates and stores eight FP8 bytes.
__device__ __forceinline__ int remnant_glm_stage_wgmma_query(
    const void* q_bf16, void* q_shared) {
  const auto* input = static_cast<const uint4*>(q_bf16);
  auto* output = static_cast<uint8_t*>(q_shared);
#pragma unroll
  for (int step = 0; step < 8; ++step) {
    const int vector = threadIdx.x + step * 128;
    const int head = vector / 64;
    const int column = (vector % 64) * 8;
    const int offset = (column / 128) * (16 * 128) + head * 128 +
                       ((column % 128) ^ ((head & 7) << 4));
    const uint4 src = input[vector];
    *reinterpret_cast<uint2*>(output + offset) = {
        remnant_glm_query_fp8_pair(src.x) |
            (remnant_glm_query_fp8_pair(src.y) << 16),
        remnant_glm_query_fp8_pair(src.z) |
            (remnant_glm_query_fp8_pair(src.w) << 16)};
  }
  return 0;
}
