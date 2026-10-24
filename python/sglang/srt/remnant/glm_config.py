"""GLM raw FP8 Packed contracts; independent of DeepSeek's scaled codec.

Author: Winston Cai.
"""

from __future__ import annotations

LATENT_DIM = 512
KEEP_COUNT = 256
BITMAP_WORDS = 8
ROW_BYTES = 320
MAX_QUERY_CHUNK = 128
SELECTION_COLUMNS = 2112


def enabled() -> bool:
    from sglang.srt.runtime_context import get_model

    return get_model().glm5_mla_cache_format == "remnant"


def validate(args, hf_config) -> None:
    if args.glm5_mla_cache_format == "native":
        return
    text = getattr(hf_config, "text_config", hf_config)
    if getattr(text, "model_type", None) != "glm5_next_text":
        raise ValueError("GLM Packed requires GLM-5.3-Flash text configuration")
    if (
        getattr(text, "kv_lora_rank", None),
        getattr(text, "qk_rope_head_dim", None),
    ) != (512, 0):
        raise ValueError("GLM Packed requires latent dimension 512 and no RoPE tail")
    unsupported = []
    for name in (
        "enable_two_batch_overlap",
        "enable_hisparse",
        "enable_hierarchical_cache",
        "enable_unified_memory",
        "enable_multimodal",
        "enable_prefill_cp",
        "enable_cp_decode_attn_tp",
        "enable_dp_attention",
        "enable_dsa_prefill_context_parallel",
        "enable_dsa_decode_context_parallel",
        "disable_cuda_graph_padding",
    ):
        if getattr(args, name, False):
            unsupported.append(name)
    for name in ("speculative_algorithm", "disaggregation_mode"):
        if getattr(args, name, None) not in (None, "null", "none"):
            unsupported.append(name)
    if getattr(args, "cpu_offload_gb", 0) > 0:
        unsupported.append("cpu_offload_gb")
    for name in ("attn_dcp_size", "attn_cp_size", "dcp_size", "cp_size"):
        if getattr(args, name, 1) != 1:
            unsupported.append(name)
    if getattr(args, "kv_cache_dtype", None) != "fp8_e4m3":
        unsupported.append("kv_cache_dtype must be fp8_e4m3")
    for name in ("dsa_prefill_backend", "dsa_decode_backend"):
        if getattr(args, name, None) != "tilelang":
            unsupported.append(name + " must be tilelang")
    if getattr(args, "device", "cuda") != "cuda":
        unsupported.append("device must be CUDA/SM90")
    if unsupported:
        raise ValueError(
            "GLM Packed unsupported configuration: " + ", ".join(unsupported)
        )
