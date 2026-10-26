"""Paged fp8 / fp4 KV cache layouts of the DeepSeek-V4 family sparse MLA decode kernels.

A page block stores ``page_size`` data rows followed by ``page_size`` scale rows.
Author: Winston Cai. (Experimental sparse FP4 layout.)

The reader selects the format from the bytes per token (the last dim of the
``(num_pages, page_size, 1, bytes_per_token)`` view) and requires the page
stride to be a multiple of its TMA row stride, which :meth:`KVLayout.page_bytes`
pads to. Mirrors ``sgl_kernel/deepseek_v4/kv_layout.cuh``.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass
from typing import ClassVar, Union


@dataclass(frozen=True)
class FP4CacheLayout:
    """SM90 persistent FP4 contract; offsets refer to a page's SoA planes.

    Author: Winston Cai. Numerical references intentionally specify their own
    expected bytes, so changing this contract cannot silently change the oracle.
    """

    packed: bool = False
    head_dim: ClassVar[int] = 512
    scale_group: ClassVar[int] = 16
    bitmap_words: ClassVar[int] = 8
    alignment: ClassVar[int] = 16
    page_alignment: ClassVar[int] = 256

    @property
    def survivors(self) -> int:
        return self.head_dim // 2 if self.packed else self.head_dim

    @property
    def value_bytes(self) -> int:
        return self.survivors // 2

    @property
    def bitmap_bytes(self) -> int:
        return self.bitmap_words * 8 if self.packed else 0

    @property
    def data_bytes(self) -> int:
        return self.value_bytes + self.bitmap_bytes

    @property
    def scale_bytes(self) -> int:
        return self.head_dim // self.scale_group

    @property
    def record_bytes(self) -> int:
        return self.data_bytes + self.scale_bytes


NATIVE_FP4 = FP4CacheLayout()
PACKED_FP4 = FP4CacheLayout(packed=True)


class KVLayout(str, enum.Enum):
    # 448 fp8 nope + 64 bf16 rope, 7 ue8m0 scales (+1 pad) per 64 values.
    V4 = "v4"
    # 512 fp8 (rope quantized too), 16 ue8m0 scales per 32 values.
    V41 = "v41"
    # 512 e2m1 packed two per byte (even index low nibble), 32 e4m3 scales per 16 values.
    V41_FP4 = "v41_fp4"
    # SM90 opt-in: 256 survivors, 128 nibble bytes + 64 bitmap bytes, 32 scales.
    V41_FP4_REMNANT = "v41_fp4_remnant"
    # TRT-LLM only: 512 e4m3 values with a fixed external unit scale.
    UNIFORM_FP8 = "uniform_fp8"

    @property
    def data_bytes(self) -> int:
        return {
            KVLayout.V4: 576,
            KVLayout.V41: 512,
            KVLayout.V41_FP4: NATIVE_FP4.data_bytes,
            KVLayout.V41_FP4_REMNANT: PACKED_FP4.data_bytes,
            KVLayout.UNIFORM_FP8: 512,
        }[self]

    @property
    def scale_bytes(self) -> int:
        return {
            KVLayout.V4: 8,
            KVLayout.V41: 16,
            KVLayout.V41_FP4: NATIVE_FP4.scale_bytes,
            KVLayout.V41_FP4_REMNANT: PACKED_FP4.scale_bytes,
            KVLayout.UNIFORM_FP8: 0,
        }[self]

    @property
    def tile_size(self) -> int:
        """Values sharing one scale."""
        return {
            KVLayout.V4: 64,
            KVLayout.V41: 32,
            KVLayout.V41_FP4: 16,
            KVLayout.V41_FP4_REMNANT: 16,
            KVLayout.UNIFORM_FP8: 512,
        }[self]

    @property
    def bytes_per_token(self) -> int:
        return self.data_bytes + self.scale_bytes

    @property
    def page_align(self) -> int:
        """Unit the page stride is padded to: the reader's TMA row stride."""
        return {
            KVLayout.V4: 576,
            KVLayout.V41: 512,
            KVLayout.V41_FP4: 256,
            KVLayout.V41_FP4_REMNANT: 256,
            KVLayout.UNIFORM_FP8: 512,
        }[self]

    @property
    def is_fp4(self) -> bool:
        return self in (KVLayout.V41_FP4, KVLayout.V41_FP4_REMNANT)

    def page_bytes(self, page_size: int) -> int:
        raw = page_size * self.bytes_per_token
        return -(-raw // self.page_align) * self.page_align

    def scale_offset(self, page_size: int) -> int:
        """Byte offset of the scale rows inside a page."""
        return page_size * self.data_bytes

    @property
    def cpp_name(self) -> str:
        """The C++ enumerator, for JIT template arguments. Bare, because it is
        also part of the JIT module name; the headers `using enum` it in."""
        return self.name

    @classmethod
    def parse(cls, value: Union[str, KVLayout]) -> KVLayout:
        if isinstance(value, KVLayout):
            return value
        return cls(str(value).lower())


def is_valid_kv_layout_pair(kv: KVLayout, extra_kv: KVLayout) -> bool:
    """The (main, extra) cache pairs the decode kernel accepts: identical layouts,
    or the fp4 extra cache next to a V4.1 fp8 main cache."""
    return (
        extra_kv is kv
        or (kv is KVLayout.V41 and extra_kv is KVLayout.V41_FP4)
        or (kv is KVLayout.V4 and extra_kv.is_fp4)
    )
