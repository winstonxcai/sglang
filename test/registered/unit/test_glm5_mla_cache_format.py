"""CPU-only GLM startup contract; runnable without serving dependencies.

Author: Winston Cai.
"""

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest

_path = Path(__file__).resolve().parents[3] / "python/sglang/srt/remnant/glm_config.py"
_spec = importlib.util.spec_from_file_location("glm_config_contract", _path)
config = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(config)


def args(**changes):
    values = dict(
        glm5_mla_cache_format="remnant",
        kv_cache_dtype="fp8_e4m3",
        dsa_prefill_backend="tilelang",
        dsa_decode_backend="tilelang",
    )
    values.update(changes)
    return SimpleNamespace(**values)


MODEL = SimpleNamespace(
    model_type="glm5_next_text", kv_lora_rank=512, qk_rope_head_dim=0
)


def test_native_default_is_unrestricted():
    config.validate(args(glm5_mla_cache_format="native"), SimpleNamespace())


def test_raw_fp8_tilelang_is_supported():
    config.validate(args(), SimpleNamespace(text_config=MODEL))
    assert config.ROW_BYTES == 256 + 8 * 8
    assert 11 * (512 + 33) / (11 * (config.ROW_BYTES + 33)) == 545 / 353


@pytest.mark.parametrize(
    "name,value",
    [
        ("enable_hisparse", True),
        ("enable_hierarchical_cache", True),
        ("enable_unified_memory", True),
        ("enable_multimodal", True),
        ("enable_prefill_cp", True),
        ("enable_cp_decode_attn_tp", True),
        ("speculative_algorithm", "EAGLE"),
        ("disaggregation_mode", "prefill"),
        ("cpu_offload_gb", 1),
        ("attn_dcp_size", 2),
        ("attn_cp_size", 2),
        ("kv_cache_dtype", "bfloat16"),
        ("device", "cpu"),
        ("dsa_prefill_backend", "flashmla_sparse"),
        ("dsa_decode_backend", "flashmla_kv"),
    ],
)
def test_rejects_unqualified_combinations(name, value):
    with pytest.raises(ValueError, match="unsupported"):
        config.validate(args(**{name: value}), MODEL)


def test_rejects_other_models():
    with pytest.raises(ValueError, match="GLM-5.3"):
        config.validate(args(), SimpleNamespace(model_type="deepseek_v4"))
