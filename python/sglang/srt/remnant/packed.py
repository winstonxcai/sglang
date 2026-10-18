"""Persistent per-layer Packed layout and production wrappers.

Author: Winston Cai.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple

import torch

from . import config


class PackedBuffers(NamedTuple):
    """Pool-owned contiguous planes on one device, shared by store and readers.

    Values hold FP8 codes as uint8 [pages, rows, k] (or flattened [rows, k]);
    bitmap uint64 and scale uint8 planes share those row dimensions and have
    eight entries per row. The startup layout fixes k for the buffer lifetime.
    """

    values: torch.Tensor
    bitmaps: torch.Tensor
    scales: torch.Tensor


@dataclass
class NativeWorkspace:
    """Backend-owned buffers allocated before CUDA graph capture.

    ``native_bytes`` is uint8 [pages, bytes_per_page] in MODEL1's split layout;
    ``temporary_indices`` is contiguous int32 [max_queries, selected_k]. Only
    Triton mode owns ``dense_bf16``, contiguous BF16 [max_queries * selected_k,
    512]. Reconstruction reuses these buffers on the caller's current stream.
    """

    native_bytes: torch.Tensor
    temporary_indices: torch.Tensor
    page_size: int
    bytes_per_page: int
    dense_bf16: torch.Tensor | None = None

    @property
    def max_queries(self) -> int:
        """Maximum query rows this decode/small-extend workspace can hold."""
        return self.temporary_indices.shape[0]

    @property
    def selected_k(self) -> int:
        return self.temporary_indices.shape[1]

    @classmethod
    def allocate(
        cls,
        max_batch: int,
        selected_k: int,
        page_size: int,
        device: torch.device | str,
    ) -> NativeWorkspace:
        rows = max_batch * selected_k
        pages = (rows + page_size - 1) // page_size
        bytes_per_page = ((config.NATIVE_RECORD_BYTES * page_size + 575) // 576) * 576
        raw = torch.zeros(pages, bytes_per_page, dtype=torch.uint8, device=device)
        temporary_indices = torch.arange(
            rows, dtype=torch.int32, device=device
        ).reshape(max_batch, selected_k)
        dense = None
        if config.packed_enabled() and config.cache_implementation() == "triton":
            # Owned by the backend, allocated before graph capture and reused
            # by all three reconstruction launches on the current CUDA stream.
            dense = torch.empty(
                (rows, config.HEAD_DIM), dtype=torch.bfloat16, device=device
            )
        return cls(raw, temporary_indices, page_size, bytes_per_page, dense)


def _as_buffers(packed_buffers, layer_id: int | None = None) -> PackedBuffers:
    if isinstance(packed_buffers, PackedBuffers):
        _buffer_keep_count(packed_buffers)
        return packed_buffers
    # The injected SGLang pool accessor deliberately exposes the stable
    # value/bitmap/scale ABI as a plain tuple. Normalize that form before
    # CUDA-graph capture reaches the unpack wrapper.
    if isinstance(packed_buffers, (tuple, list)):
        if len(packed_buffers) != 3 or not all(
            isinstance(buffer, torch.Tensor) for buffer in packed_buffers
        ):
            raise TypeError(
                "packed buffer sequences must contain value, bitmap, and scale tensors"
            )
        buffers = PackedBuffers(*packed_buffers)
        _buffer_keep_count(buffers)
        return buffers
    if hasattr(packed_buffers, "get_packed_buffers"):
        if layer_id is None:
            raise ValueError("layer_id is required for a packed pool")
        buffers = _as_buffers(packed_buffers.get_packed_buffers(layer_id))
        if hasattr(packed_buffers, "get_packed_layout"):
            if (
                buffers.values.shape[-1]
                != packed_buffers.get_packed_layout(layer_id).keep_count
            ):
                raise ValueError(
                    "Packed buffer width differs from resolved layer layout"
                )
        return buffers
    raise TypeError(
        "packed_buffers must be PackedBuffers, a three-tensor sequence, "
        "or a packed pool"
    )


def _buffer_keep_count(buffers: PackedBuffers) -> int:
    """Inspect host tensor metadata only; no tensor reads or CUDA synchronization."""
    if buffers.values.ndim not in (2, 3):
        raise ValueError("Packed values must be [rows, k] or [pages, rows, k]")
    count = config.validate_keep_count(buffers.values.shape[-1])
    prefix = buffers.values.shape[:-1]
    if buffers.bitmaps.shape != (
        *prefix,
        config.BITMAP_WORDS,
    ) or buffers.scales.shape != (*prefix, config.BITMAP_WORDS):
        raise ValueError(
            "Packed values, bitmap and scale planes must have matching row geometry"
        )
    if count != config.PACKED_KEPT_VALUES:
        if (
            config.cache_implementation() != "triton"
            or config.cache_store_implementation() != "triton"
        ):
            raise ValueError(
                "non-256 Packed buffers require Triton storage and reconstruction"
            )
    return count


def _plan_rows(compressor_plan: object) -> torch.Tensor:
    # Both DecodePlan and CompressPlan have four words per compressed row.
    # plan[2] is prefill state-ring persistence, not compressed cache records.
    plan = compressor_plan[1]
    return plan.view(torch.int32)


def pack_rows(
    latent: torch.Tensor,
    keep_mask: torch.Tensor,
    norm_weight: torch.Tensor,
    norm_eps: float,
    compressor_plan: object,
    locations: torch.Tensor,
    packed_buffers,
    *,
    layer_id: int | None = None,
) -> None:
    """Pack valid compressor rows directly into persistent paged storage."""
    config.validate_packed_static_config()
    if latent.shape[-1] != config.HEAD_DIM:
        raise ValueError(f"packed expects dim 512, got {latent.shape}")
    if keep_mask.shape != latent.shape or keep_mask.dtype != torch.bool:
        raise ValueError("keep_mask must be bool and match latent")
    buffers = _as_buffers(packed_buffers, layer_id)
    if latent.numel() == 0:
        return
    from .triton import _pack_fp8_kernel

    plan_rows = _plan_rows(compressor_plan)
    _pack_fp8_kernel[(latent.shape[0],)](
        latent,
        keep_mask,
        norm_weight,
        plan_rows,
        locations,
        buffers.values,
        buffers.bitmaps,
        buffers.scales,
        latent.shape[0],
        norm_eps=norm_eps,
        HEAD_DIM=config.HEAD_DIM,
        KEEP_K=_buffer_keep_count(buffers),
        TILE_SIZE=config.TILE_SIZE,
        BITMAP_WORDS=config.BITMAP_WORDS,
        FP8_MAX=config.FP8_E4M3_MAX,
        IS_DECODE=bool(compressor_plan.is_decode),
        BLOCK_D=config.HEAD_DIM,
        num_warps=8,
    )


def pack_rows_fused(
    latent: torch.Tensor,
    norm_weight: torch.Tensor,
    norm_eps: float,
    compressor_plan: object,
    locations: torch.Tensor,
    packed_buffers,
    *,
    layer_id: int | None = None,
) -> None:
    """Select and pack finite FP32 C4 rows without an intermediate keep mask.

    The optimized path is qualified on SM90. Other layouts/dtypes/devices use
    the explicit-mask store; its selection still sees the original latent.
    """
    config.validate_packed_static_config()
    if latent.ndim != 2 or latent.shape[1] != config.HEAD_DIM:
        raise ValueError(f"packed expects [rows, 512], got {latent.shape}")
    if latent.numel() == 0:
        return
    buffers = _as_buffers(packed_buffers, layer_id)
    if _buffer_keep_count(buffers) != config.PACKED_KEPT_VALUES:
        raise ValueError("fused Packed storage supports only 256 survivors")
    supported = (
        latent.is_cuda
        and not torch.version.hip
        and latent.dtype == torch.float32
        and norm_weight.dtype == torch.float32
        and norm_weight.device == latent.device
        and norm_weight.shape == (config.HEAD_DIM,)
        and latent.is_contiguous()
        and norm_weight.is_contiguous()
        and locations.is_contiguous()
        and torch.cuda.get_device_capability(latent.device) == (9, 0)
    )
    if not supported:
        from .reference import topmag_keep_mask

        mask = topmag_keep_mask(latent, config.PACKED_KEPT_VALUES / config.HEAD_DIM)
        pack_rows(
            latent.contiguous(),
            mask.contiguous(),
            norm_weight.contiguous(),
            norm_eps,
            compressor_plan,
            locations.contiguous(),
            buffers,
        )
        return
    from .triton.fused_store import _pack_fused_fp32_kernel

    plan_rows = _plan_rows(compressor_plan)
    _pack_fused_fp32_kernel[(latent.shape[0],)](
        latent,
        norm_weight,
        plan_rows,
        locations,
        buffers.values,
        buffers.bitmaps,
        buffers.scales,
        latent.shape[0],
        norm_eps=norm_eps,
        HEAD_DIM=config.HEAD_DIM,
        KEEP_K=_buffer_keep_count(buffers),
        TILE_SIZE=config.TILE_SIZE,
        BITMAP_WORDS=config.BITMAP_WORDS,
        FP8_MAX=config.FP8_E4M3_MAX,
        IS_DECODE=bool(compressor_plan.is_decode),
        BLOCK_D=config.HEAD_DIM,
        num_warps=8,
    )


def unpack_gather_bf16(
    packed_buffers,
    physical_indices: torch.Tensor,
    raw_indices: torch.Tensor,
    topk_lengths: torch.Tensor,
    freqs_cis,
    output: torch.Tensor,
    *,
    layer_id: int | None = None,
    fused_rope: bool = False,
) -> torch.Tensor:
    """Gather selected records, dequantize, zero-fill and apply tail RoPE.

    Indices are int32 [queries, selected_k] views with independent strides;
    physical entries address Packed rows, raw entries address logical C4 rows.
    Lengths are int32 [queries]. The caller owns contiguous BF16 output and a
    contiguous complex RoPE table. Invalid/padded selections produce zero rows.
    ``fused_rope`` opts into the experimental one-launch reader; the verified
    split path remains the default. Neither reader changes the persistent ABI.
    """
    if type(fused_rope) is not bool:
        raise TypeError("fused_rope must be a bool")
    buffers = _as_buffers(packed_buffers, layer_id)
    if physical_indices.shape != raw_indices.shape:
        raise ValueError("physical_indices and raw_indices must have equal shape")
    n_queries, selected_k = physical_indices.shape
    flat_rows = n_queries * selected_k
    out2d = output.reshape(-1, config.HEAD_DIM)
    if out2d.shape[0] < flat_rows:
        raise ValueError("output workspace is too small")
    if output.dtype != torch.bfloat16 or not output.is_contiguous():
        raise ValueError("output must be a contiguous BF16 workspace")
    if topk_lengths.shape != (n_queries,):
        raise ValueError("topk_lengths must have one entry per query")
    if not freqs_cis.is_complex() or not freqs_cis.is_contiguous():
        raise ValueError("freqs_cis must be a contiguous complex tensor")
    if flat_rows == 0:
        return output
    if fused_rope:
        from .triton import _unpack_gather_bf16_rope_kernel

        _unpack_gather_bf16_rope_kernel[(flat_rows,)](
            buffers.values,
            buffers.bitmaps,
            buffers.scales,
            physical_indices,
            raw_indices,
            topk_lengths,
            torch.view_as_real(freqs_cis),
            out2d,
            n_queries,
            selected_k,
            HEAD_DIM=config.HEAD_DIM,
            NOPE_DIM=config.NOPE_DIM,
            ROPE_PAIRS=config.ROPE_DIM // 2,
            KEEP_K=_buffer_keep_count(buffers),
            BITMAP_WORDS=config.BITMAP_WORDS,
            BLOCK_D=config.HEAD_DIM,
            PHYSICAL_STRIDE_Q=physical_indices.stride(0),
            PHYSICAL_STRIDE_K=physical_indices.stride(1),
            RAW_STRIDE_Q=raw_indices.stride(0),
            RAW_STRIDE_K=raw_indices.stride(1),
            LENGTH_STRIDE=topk_lengths.stride(0),
            PACKED_ROWS=buffers.values.numel() // _buffer_keep_count(buffers),
            FREQ_ROWS=freqs_cis.numel() // (config.ROPE_DIM // 2),
            num_warps=8,
        )
        return output
    from .triton import (
        _rope_tail_complex_inplace_kernel,
        _unpack_gather_bf16_kernel,
    )

    _unpack_gather_bf16_kernel[(flat_rows,)](
        buffers.values,
        buffers.bitmaps,
        buffers.scales,
        physical_indices,
        raw_indices,
        topk_lengths,
        out2d,
        n_queries,
        selected_k,
        HEAD_DIM=config.HEAD_DIM,
        KEEP_K=_buffer_keep_count(buffers),
        BITMAP_WORDS=config.BITMAP_WORDS,
        BLOCK_D=config.HEAD_DIM,
        PHYSICAL_STRIDE_Q=physical_indices.stride(0),
        PHYSICAL_STRIDE_K=physical_indices.stride(1),
        RAW_STRIDE_Q=raw_indices.stride(0),
        RAW_STRIDE_K=raw_indices.stride(1),
        LENGTH_STRIDE=topk_lengths.stride(0),
        PACKED_ROWS=buffers.values.numel() // _buffer_keep_count(buffers),
        FREQ_ROWS=freqs_cis.numel() // (config.ROPE_DIM // 2),
        num_warps=8,
    )
    # This is a view only; unlike .real.contiguous()/.imag.contiguous(), it
    # performs no CUDA allocation and can be retained by the packed pool.
    freq_pairs = torch.view_as_real(freqs_cis)
    _rope_tail_complex_inplace_kernel[(flat_rows,)](
        out2d,
        physical_indices,
        raw_indices,
        topk_lengths,
        freq_pairs,
        flat_rows,
        selected_k,
        HEAD_DIM=config.HEAD_DIM,
        NOPE_DIM=config.NOPE_DIM,
        ROPE_PAIRS=config.ROPE_DIM // 2,
        PHYSICAL_STRIDE_Q=physical_indices.stride(0),
        PHYSICAL_STRIDE_K=physical_indices.stride(1),
        RAW_STRIDE_Q=raw_indices.stride(0),
        RAW_STRIDE_K=raw_indices.stride(1),
        LENGTH_STRIDE=topk_lengths.stride(0),
        PACKED_ROWS=buffers.values.numel() // _buffer_keep_count(buffers),
        FREQ_ROWS=freqs_cis.numel() // (config.ROPE_DIM // 2),
        num_warps=1,
    )
    return output


def unpack_gather_native(
    packed_buffers,
    physical_indices: torch.Tensor,
    raw_indices: torch.Tensor,
    topk_lengths: torch.Tensor,
    freqs_cis,
    native_workspace: NativeWorkspace,
    *,
    layer_id: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Materialize native hybrid pages and return cache plus remapped indices."""
    buffers = _as_buffers(packed_buffers, layer_id)
    n_queries, selected_k = physical_indices.shape
    if n_queries > native_workspace.max_queries:
        raise ValueError(
            "native workspace query capacity exceeded: "
            f"{n_queries} > {native_workspace.max_queries}; "
            "route this extend through sparse prefill"
        )
    if selected_k > native_workspace.selected_k:
        raise ValueError(
            "native workspace top-k capacity exceeded: "
            f"{selected_k} > {native_workspace.selected_k}"
        )
    rows = n_queries * selected_k
    # Materialization writes dense selected rows, even when selected_k is less
    # than capacity. A rectangular slice would retain the capacity row stride
    # and point the second query at unwritten records.
    temp = native_workspace.temporary_indices.view(-1)[:rows].view(
        n_queries, selected_k
    )
    config.validate_packed_static_config()
    if physical_indices.shape != raw_indices.shape:
        raise ValueError("physical_indices and raw_indices must have equal shape")
    if not freqs_cis.is_complex() or not freqs_cis.is_contiguous():
        raise ValueError("freqs_cis must be a contiguous complex tensor")
    if config.cache_implementation() == "triton":
        if native_workspace.dense_bf16 is None:
            raise RuntimeError(
                "Triton reconstruction requires a workspace allocated in Triton mode"
            )
        dense = native_workspace.dense_bf16[:rows]
        unpack_gather_bf16(
            buffers, physical_indices, raw_indices, topk_lengths, freqs_cis, dense
        )
        if rows:
            from .triton import _bf16_to_native_kernel

            _bf16_to_native_kernel[(rows,)](
                dense,
                buffers.bitmaps,
                buffers.values,
                buffers.scales,
                physical_indices,
                raw_indices,
                topk_lengths,
                native_workspace.native_bytes,
                rows,
                selected_k,
                page_size=native_workspace.page_size,
                bytes_per_page=native_workspace.bytes_per_page,
                HEAD_DIM=config.HEAD_DIM,
                NOPE_DIM=config.NOPE_DIM,
                KEEP_K=_buffer_keep_count(buffers),
                BITMAP_WORDS=config.BITMAP_WORDS,
                BLOCK_D=config.HEAD_DIM,
                PHYSICAL_STRIDE_Q=physical_indices.stride(0),
                PHYSICAL_STRIDE_K=physical_indices.stride(1),
                RAW_STRIDE_Q=raw_indices.stride(0),
                RAW_STRIDE_K=raw_indices.stride(1),
                LENGTH_STRIDE=topk_lengths.stride(0),
                PACKED_ROWS=buffers.values.numel() // _buffer_keep_count(buffers),
                FREQ_ROWS=freqs_cis.numel() // (config.ROPE_DIM // 2),
                num_warps=8,
            )
        return native_workspace.native_bytes, temp
    # Model-free validation images overlay the split Remnant extension on top
    # of the stock SGLang common_ops library.  Production builds already load
    # this operator from common_ops, so import the optional module only when
    # the operator is not registered yet.
    if not hasattr(torch.ops.sgl_kernel, "remnant_packed_to_native"):
        try:
            from sgl_kernel import remnant_ops  # noqa: F401
        except ImportError as exc:
            raise RuntimeError(
                "remnant_packed_to_native is unavailable; build the Remnant adapter extension"
            ) from exc
    torch.ops.sgl_kernel.remnant_packed_to_native.default(
        buffers.values,
        buffers.bitmaps,
        buffers.scales,
        physical_indices,
        raw_indices,
        topk_lengths,
        torch.view_as_real(freqs_cis),
        native_workspace.native_bytes,
        native_workspace.page_size,
        native_workspace.bytes_per_page,
    )
    return native_workspace.native_bytes, temp
