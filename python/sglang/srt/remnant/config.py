"""Native and Packed cache configuration for DeepSeek V4.

Author: Winston Cai.
"""

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

# --- compressed-latent geometry (fixed by DeepSeek-V4) ----------------------
HEAD_DIM = 512  # full compressed-latent dimension
ROPE_DIM = 64  # rotary tail dims
NOPE_DIM = HEAD_DIM - ROPE_DIM  # 448
TILE_SIZE = 64  # native store's fp8 scale tile
BITMAP_WORDS = HEAD_DIM // 64  # 8  (uint64 words per packed row)
# Default geometry and the unchanged optimized CUDA/Direct ABI. Triton derives
# each layer's geometry from its resolved layout and actual tensor metadata.
PACKED_KEPT_VALUES = HEAD_DIM // 2
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
_CACHE_IMPLEMENTATION = "kernel"
_STORE_IMPLEMENTATION = "kernel"
VALID_KEEP_COUNTS = (64, 128, 192, 256, 320, 384, 448)


def validate_keep_count(value: int) -> int:
    if type(value) is not int or value not in VALID_KEEP_COUNTS:
        raise ValueError(
            f"Packed keep count must be one of {VALID_KEEP_COUNTS}, got {value!r}"
        )
    return value


def parse_layer_keep_counts(
    value: str | Mapping[str | int, int] | None,
) -> tuple[tuple[int, int], ...]:
    """Normalize JSON/model-layer overrides without silently accepting duplicate IDs."""
    if value is None:
        return ()
    if isinstance(value, str):

        def pairs(items):
            result = {}
            for key, item in items:
                if key in result:
                    raise ValueError(f"duplicate Packed layer ID {key!r}")
                result[key] = item
            return result

        try:
            value = json.loads(value, object_pairs_hook=pairs)
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid Packed layer-count JSON: {exc}") from exc
    if not isinstance(value, Mapping):
        raise ValueError(
            "Packed layer counts must be a JSON object keyed by model layer ID"
        )
    result = {}
    for key, count in value.items():
        if type(key) is int:
            layer = key
        elif isinstance(key, str) and key.isascii() and key.isdigit():
            layer = int(key)
        else:
            raise ValueError(f"invalid Packed model layer ID {key!r}")
        if layer < 0 or layer in result:
            raise ValueError(f"invalid or duplicate Packed model layer ID {key!r}")
        result[layer] = validate_keep_count(count)
    return tuple(sorted(result.items()))


@dataclass(frozen=True)
class PackedLayerLayout:
    """Immutable geometry for one absolute model layer (not its C4-pool index)."""

    layer_id: int
    keep_count: int = PACKED_KEPT_VALUES

    def __post_init__(self) -> None:
        if type(self.layer_id) is not int or self.layer_id < 0:
            raise ValueError(f"invalid Packed model layer ID {self.layer_id!r}")
        validate_keep_count(self.keep_count)

    @property
    def record_bytes(self) -> int:
        return self.keep_count + PACKED_BITMAP_BYTES + PACKED_SCALE_BYTES


@dataclass(frozen=True)
class PackedProfile:
    default_count: int = PACKED_KEPT_VALUES
    overrides: tuple[tuple[int, int], ...] = ()
    compression_ratios: tuple[int, ...] | None = None

    def layouts(
        self,
        ratios: Sequence[int],
        start: int = 0,
        end: int | None = None,
    ) -> tuple[PackedLayerLayout, ...]:
        ratios = tuple(ratios)
        if self.compression_ratios is not None and ratios != self.compression_ratios:
            raise ValueError("Packed profile does not match model compression ratios")
        overrides = dict(self.overrides)
        for layer in overrides:
            if layer >= len(ratios) or ratios[layer] != 4:
                raise ValueError(
                    f"Packed override layer {layer} is missing or is not a C4 layer"
                )
        stop = len(ratios) if end is None else end
        if not 0 <= start <= stop <= len(ratios):
            raise ValueError(f"invalid Packed model layer range [{start}, {stop})")
        return tuple(
            PackedLayerLayout(i, overrides.get(i, self.default_count))
            for i in range(start, stop)
            if ratios[i] == 4
        )

    @property
    def nondefault(self) -> bool:
        counts = (
            [x.keep_count for x in self.layouts(self.compression_ratios)]
            if self.compression_ratios is not None
            else [self.default_count, *(count for _, count in self.overrides)]
        )
        return any(count != PACKED_KEPT_VALUES for count in counts)


_PROFILE = PackedProfile()


def packed_profile() -> PackedProfile:
    return _PROFILE


def packed_layouts(
    ratios: Sequence[int],
    start: int = 0,
    end: int | None = None,
) -> tuple[PackedLayerLayout, ...]:
    return _PROFILE.layouts(ratios, start, end)


def configure_cache_format(
    cache_format: str,
    implementation: str = "kernel",
    store_implementation: str | None = None,
    *,
    keep_count: int = PACKED_KEPT_VALUES,
    layer_keep_counts: str | Mapping[str | int, int] | None = None,
    compression_ratios: Sequence[int] | None = None,
    hierarchical_cache: bool = False,
    publish: bool = True,
) -> None:
    """Set startup configuration; Native ignores the Packed implementation."""
    if cache_format not in {"native", "remnant"}:
        raise ValueError(
            "dsv4_c4_cache_format must be 'native' or 'remnant', "
            f"got {cache_format!r}"
        )
    if implementation not in {"triton", "kernel"}:
        raise ValueError(
            "dsv4_c4_cache_implementation must be 'triton' or 'kernel', "
            f"got {implementation!r}"
        )
    if store_implementation not in {None, "triton", "kernel"}:
        raise ValueError(
            "dsv4_c4_cache_store_implementation must be None, 'triton' or 'kernel', "
            f"got {store_implementation!r}"
        )
    profile = PackedProfile(
        validate_keep_count(keep_count),
        parse_layer_keep_counts(layer_keep_counts),
        None if compression_ratios is None else tuple(compression_ratios),
    )
    if cache_format == "native" and (
        keep_count != PACKED_KEPT_VALUES or layer_keep_counts is not None
    ):
        raise ValueError(
            "Packed survivor options require --dsv4-c4-cache-format remnant"
        )
    if profile.compression_ratios is not None:
        profile.layouts(profile.compression_ratios)
    store = store_implementation or implementation
    # Before model resolution, overrides may cover every nondefault count.
    # Validate the effective dispatch only once the complete profile is known.
    check_dispatch = publish or compression_ratios is not None or not profile.overrides
    if (
        check_dispatch
        and profile.nondefault
        and (implementation != "triton" or store != "triton")
    ):
        raise ValueError(
            "non-256 Packed counts require Triton storage and reconstruction"
        )
    if check_dispatch and profile.nondefault and hierarchical_cache:
        raise ValueError("non-256 Packed counts do not support HiCache")
    if not publish:
        return
    global _CACHE_FORMAT, _CACHE_IMPLEMENTATION, _STORE_IMPLEMENTATION, _PROFILE
    _CACHE_FORMAT = cache_format
    _CACHE_IMPLEMENTATION = implementation
    _STORE_IMPLEMENTATION = store
    _PROFILE = profile


def configure_from_args(
    args: Any,
    compression_ratios: Sequence[int] | None = None,
    *,
    publish: bool = True,
) -> None:
    """Publish startup geometry atomically, including in spawned workers."""
    validate_v41_cache_args(args)
    configure_cache_format(
        getattr(args, "dsv4_c4_cache_format", "native"),
        getattr(args, "dsv4_c4_cache_implementation", "kernel"),
        getattr(args, "dsv4_c4_cache_store_implementation", None),
        keep_count=getattr(args, "dsv4_c4_cache_keep_count", PACKED_KEPT_VALUES),
        layer_keep_counts=getattr(args, "dsv4_c4_cache_layer_keep_counts", None),
        compression_ratios=(
            compression_ratios
            if compression_ratios is not None
            else getattr(args, "_dsv4_c4_cache_compression_ratios", None)
        ),
        hierarchical_cache=getattr(args, "enable_hierarchical_cache", False),
        publish=publish,
    )


def _snapshot_dispatch() -> tuple[str, str, str, PackedProfile]:
    return _CACHE_FORMAT, _CACHE_IMPLEMENTATION, _STORE_IMPLEMENTATION, _PROFILE


def _restore_dispatch(state: tuple[str, str, str, PackedProfile] | None) -> None:
    global _CACHE_FORMAT, _CACHE_IMPLEMENTATION, _STORE_IMPLEMENTATION, _PROFILE
    _CACHE_FORMAT, _CACHE_IMPLEMENTATION, _STORE_IMPLEMENTATION, _PROFILE = (
        state if state is not None else ("native", "kernel", "kernel", PackedProfile())
    )


def _dispatch_from_args(args: Any) -> tuple[str, str, str, PackedProfile]:
    """Validate a context's startup dispatch without changing the active model."""
    previous = _snapshot_dispatch()
    try:
        configure_from_args(args)
        return _snapshot_dispatch()
    finally:
        _restore_dispatch(previous)


def cache_format() -> str:
    return _CACHE_FORMAT


def cache_implementation() -> str:
    return _CACHE_IMPLEMENTATION


def cache_store_implementation() -> str:
    """Resolved store mode; omitted overrides follow the decode implementation."""
    return _STORE_IMPLEMENTATION


def packed_enabled() -> bool:
    """Whether the startup flag selected the persistent packed layout."""
    return _CACHE_FORMAT == "remnant"


def validate_packed_static_config() -> None:
    """Check dispatch compatibility without reading tensors during graph replay."""
    if packed_enabled() and _PROFILE.nondefault:
        if _CACHE_IMPLEMENTATION != "triton" or _STORE_IMPLEMENTATION != "triton":
            raise RuntimeError(
                "non-256 Packed counts require Triton storage and reconstruction"
            )


def validate_v41_cache_args(args: Any) -> None:
    """Validate opt-in FP4 independently of V4 C4 dispatch publication."""
    mode = getattr(args, "dsv41_main_kv_cache_format", "auto")
    if mode not in ("auto", "fp4", "remnant"):
        raise ValueError("dsv41_main_kv_cache_format must be auto, fp4 or remnant")
    if mode == "auto":
        return
    conflicts = []
    for name, default in (
        ("dsv4_c4_cache_format", "native"),
        ("dsv4_c4_cache_keep_count", 256),
        ("dsv4_c4_cache_layer_keep_counts", None),
        ("dsv4_c4_cache_implementation", "kernel"),
        ("dsv4_c4_cache_store_implementation", None),
    ):
        if getattr(args, name, default) != default:
            conflicts.append(name)
    for name in (
        "enable_hierarchical_cache",
        "enable_hisparse",
        "enable_prefill_cp",
        "enable_encoder_swa_bounded_replay",
        "enable_decoder_swa_bounded_replay",
        "disaggregation_decode_enable_offload_kvcache",
    ):
        if getattr(args, name, False):
            conflicts.append(name)
    if getattr(args, "disaggregation_mode", "null") not in (None, "null"):
        conflicts.append("disaggregation_mode")
    if getattr(args, "speculative_algorithm", None) is not None:
        conflicts.append("speculative_algorithm")
    if getattr(args, "cpu_offload_gb", 0) > 0:
        conflicts.append("cpu_offload_gb")
    if getattr(args, "dsv4_attn_backend", "flashmla") != "flashmla":
        conflicts.append("dsv4_attn_backend (requires flashmla)")
    if getattr(args, "dsv4_prefill_backend", "auto") not in ("auto", "flashmla_sparse"):
        conflicts.append("dsv4_prefill_backend (requires BF16 sparse prefill)")
    from sglang.srt.environ import envs

    if (
        envs.SGLANG_DSV4_KV_LAYOUT.get().lower() != "auto"
        or envs.SGLANG_DSV4_COMPRESSED_KV_LAYOUT.get().lower() != "auto"
    ):
        conflicts.append("legacy SGLANG_DSV4_*KV_LAYOUT settings")
    from sglang.kernels.ops.attention.dsv4.unified_kv_kernels.env_gate import (
        is_unified_kv_triton,
    )

    if is_unified_kv_triton():
        conflicts.append("unified KV storage")
    if conflicts:
        raise ValueError(
            "--dsv41-main-kv-cache-format "
            + mode
            + " is incompatible with "
            + ", ".join(conflicts)
        )
