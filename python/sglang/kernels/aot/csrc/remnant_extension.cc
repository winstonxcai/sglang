// Remnant additions authored by Winston Cai.

#include <torch/all.h>
#include <torch/library.h>

#include "sgl_kernel_ops.h"

// Keep the adapter registration in a small optional extension.  The normal
// SGLang build still includes this operator in common_ops; the split target is
// used by model-free validation images that overlay only the Remnant changes.
TORCH_LIBRARY_FRAGMENT(sgl_kernel, m) {
  m.def(
      "remnant_packed_to_native(Tensor values, Tensor bitmaps, Tensor scales, "
      "Tensor physical_indices, Tensor raw_indices, Tensor topk_lengths, "
      "Tensor freq_pairs, Tensor! native_out, int page_size, "
      "int bytes_per_page) -> ()");
  m.impl("remnant_packed_to_native", torch::kCUDA, &remnant_packed_to_native);
}

REGISTER_EXTENSION(remnant_ops)
