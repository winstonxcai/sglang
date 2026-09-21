"""Native and Remnant cache-format configuration for DeepSeek V4.

Author: Winston Cai.
"""

# --- compressed-latent geometry (fixed by DeepSeek-V4) ----------------------
HEAD_DIM = 512  # full compressed-latent dimension
ROPE_DIM = 64  # rotary tail dims
NOPE_DIM = HEAD_DIM - ROPE_DIM  # 448
TILE_SIZE = 64  # native store's fp8 scale tile
BITMAP_WORDS = HEAD_DIM // 64  # 8  (uint64 words per packed row)
PACKED_KEPT_VALUES = HEAD_DIM // 2  # exact TopMag50 survivors
PACKED_VALUE_BYTES = PACKED_KEPT_VALUES  # raw FP8 E4M3 codes
PACKED_BITMAP_BYTES = BITMAP_WORDS * 8
PACKED_SCALE_BYTES = HEAD_DIM // TILE_SIZE
PACKED_RECORD_BYTES = (
    PACKED_VALUE_BYTES + PACKED_BITMAP_BYTES + PACKED_SCALE_BYTES
)  # 328 B
NATIVE_RECORD_BYTES = 584
FP8_E4M3_MAX = 448.0

# ServerArgs configures this once during startup, before any cache pool or
# CUDA graph workspace is allocated. Native remains the safe default.
_CACHE_FORMAT = "native"


def configure_cache_format(cache_format: str) -> None:
    """Set the process-wide DSV4 C4 cache format from ServerArgs."""
    if cache_format not in {"native", "remnant"}:
        raise ValueError(
            "dsv4_c4_cache_format must be 'native' or 'remnant', "
            f"got {cache_format!r}"
        )
    global _CACHE_FORMAT
    _CACHE_FORMAT = cache_format


def cache_format() -> str:
    return _CACHE_FORMAT


def topmag_enabled() -> bool:
    return packed_enabled()


def topmag_keep() -> float:
    return 0.5 if packed_enabled() else 1.0


def packed_enabled() -> bool:
    """Whether the startup flag selected the persistent packed layout."""
    return _CACHE_FORMAT == "remnant"


def fused_enabled() -> bool:
    """Fused reconstruction is intentionally outside this phase."""
    return False


def optimized_fused_enabled() -> bool:
    return False


def sparse_enabled() -> bool:
    return False


def validate_packed_static_config() -> None:
    """Validate the fixed Remnant ABI after the startup flag is parsed."""
    if packed_enabled() and topmag_keep() != 0.5:
        raise RuntimeError("Remnant requires exactly 256/512 retained dimensions")
