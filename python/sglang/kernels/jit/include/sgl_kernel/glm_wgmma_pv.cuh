// Author: Winston Cai.
#pragma once

#include <tl_templates/cuda/instruction/wgmma.h>
#include <tl_templates/cuda/intrin.h>

// Each warp supplies sixteen features to a 64-feature WGMMA panel. Its
// four b16 matrix loads pair histories [0,1,4,5,...] and [2,3,6,7,...]
// in each half of the 32-row slice. Separate the two feature bytes from
// each b16 element to build four contiguous history bytes per register.
__device__ __forceinline__ void remnant_glm_load_pv_registers(
    const uint8_t* kv, int feature, int history, uint32_t* fragment) {
  const int lane = threadIdx.x & 31;
  const int matrix_row = lane & 7;
  const int row = history + 4 * (matrix_row / 2) + (matrix_row & 1) +
                  2 * ((lane >> 3) & 1) + 16 * (lane >> 4);
  const int offset = (feature / 128) * (64 * 128) + row * 128 +
                     ((feature % 128) ^ ((row & 7) << 4));
  const uint32_t addr =
      static_cast<uint32_t>(__cvta_generic_to_shared(kv + offset));
  uint32_t a0, b0, a1, b1;
  asm volatile(
      "ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];"
      : "=r"(a0), "=r"(b0), "=r"(a1), "=r"(b1)
      : "r"(addr)
      : "memory");
  fragment[0] = __byte_perm(a0, b0, 0x6420);
  fragment[1] = __byte_perm(a0, b0, 0x7531);
  fragment[2] = __byte_perm(a1, b1, 0x6420);
  fragment[3] = __byte_perm(a1, b1, 0x7531);
}

// PV uses FP8 register A and shared SW64 probability B. The logical
// feature order within a warp is even/odd in the two hardware M halves.
// The TileLang accumulator layout maps those halves to adjacent features.
__device__ __forceinline__ int remnant_glm_wgmma_pv_registers(
    const void* kv_raw, const void* p_raw, float* numerator) {
  const auto* kv = static_cast<const uint8_t*>(kv_raw);
  const auto* p = static_cast<const uint8_t*>(p_raw);
  const int warp_feature = (threadIdx.x >> 5) * 16;
  tl::GmmaDescriptor descriptor;
  tl::initialize_wgmma_descriptor<2, 1, 32>(descriptor, p);
  constexpr int panels_per_group = 2;
  constexpr int registers_per_panel = 8;
#pragma unroll
  for (int first = 0; first < 8; first += panels_per_group) {
    uint32_t values[panels_per_group][2][4];
    float* accum = numerator + first * registers_per_panel;
#pragma unroll
    for (int panel = 0; panel < panels_per_group; ++panel) {
#pragma unroll
      for (int slice = 0; slice < 2; ++slice) {
        remnant_glm_load_pv_registers(
            kv, (first + panel) * 64 + warp_feature, slice * 32,
            values[panel][slice]);
        tl::warpgroup_fence_operand(values[panel][slice], 4);
      }
    }
    tl::warpgroup_fence_operand(accum, panels_per_group * registers_per_panel);
    tl::warpgroup_arrive();
#pragma unroll
    for (int panel = 0; panel < panels_per_group; ++panel) {
#pragma unroll
      for (int slice = 0; slice < 2; ++slice) {
        tl::wgmma_rs<tl::DataType::kFloat8_e4m3, tl::DataType::kFloat8_e4m3,
                     tl::DataType::kFloat32, 64, 16, 32, false, false>(
            values[panel][slice], uint64_t(descriptor + 2 * slice),
            reinterpret_cast<uint32_t*>(accum + panel * registers_per_panel),
            true);
      }
    }
    tl::warpgroup_commit_batch();
    tl::warpgroup_wait<0>();
    // Source and accumulator registers remain live until this group drains.
#pragma unroll
    for (int panel = 0; panel < panels_per_group; ++panel) {
#pragma unroll
      for (int slice = 0; slice < 2; ++slice) {
        tl::warpgroup_fence_operand(values[panel][slice], 4);
      }
    }
    tl::warpgroup_fence_operand(accum, panels_per_group * registers_per_panel);
  }
  return 0;
}

// Match the TileLang numerator fragment. Each lane owns four heads and
// two adjacent features in each 64-feature panel. Write each BF16 pair
// together so one store covers the full four bytes owned by the lane.
__device__ __forceinline__ int remnant_glm_write_wgmma_output(
    const float* numerator, const float* scale, void* output_raw) {
  auto* output = static_cast<uint32_t*>(output_raw);
  const int feature_pair = threadIdx.x / 4;
  const int first_head = 2 * (threadIdx.x % 4);
#pragma unroll
  for (int panel = 0; panel < 8; ++panel) {
#pragma unroll
    for (int head_group = 0; head_group < 2; ++head_group) {
#pragma unroll
      for (int head_in_pair = 0; head_in_pair < 2; ++head_in_pair) {
        const int head = first_head + 8 * head_group + head_in_pair;
        const int index = 8 * panel + 4 * head_group + head_in_pair;
        const float weight = scale[2 * head_group + head_in_pair];
        const float even = numerator[index] * weight;
        const float odd = numerator[index + 2] * weight;
        uint32_t pair;
        // The first PTX input occupies the high half of the packed word.
        asm("cvt.rn.bf16x2.f32 %0, %1, %2;"
            : "=r"(pair) : "f"(odd), "f"(even));
        output[head * 256 + panel * 32 + feature_pair] = pair;
      }
    }
  }
  return 0;
}
