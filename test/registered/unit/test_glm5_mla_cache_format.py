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
        attention_backend="dsa",
        kv_cache_dtype="fp8_e4m3",
        dsa_prefill_backend="tilelang",
        dsa_decode_backend="tilelang",
    )
    values.update(changes)
    return SimpleNamespace(**values)


MODEL = SimpleNamespace(
    model_type="glm5_next_text",
    kv_lora_rank=512,
    qk_rope_head_dim=0,
    qk_nope_head_dim=256,
    v_head_dim=256,
    index_topk=2048,
    index_kpool=4,
    index_head_dim=128,
    num_hidden_layers=45,
    num_attention_heads=64,
    index_kpool_compress=True,
    index_kpool_always_select_tail=True,
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


def test_multimodal_checkpoint_requires_text_only_launch():
    multimodal = SimpleNamespace(text_config=MODEL, vision_config=object())
    with pytest.raises(ValueError, match="language-model-only"):
        config.validate(args(), multimodal)
    config.validate(args(language_model_only=True), multimodal)


@pytest.mark.parametrize(
    "name,value",
    [
        ("qk_nope_head_dim", 128),
        ("index_topk", 1024),
        ("index_kpool", 1),
        ("num_hidden_layers", 44),
        ("num_attention_heads", 128),
    ],
)
def test_rejects_unqualified_geometry(name, value):
    fields = vars(MODEL).copy()
    fields[name] = value
    with pytest.raises(ValueError, match="unqualified geometry"):
        config.validate(args(), SimpleNamespace(**fields))


@pytest.mark.parametrize("phase", ["prefill", "decode"])
@pytest.mark.parametrize("backend", ["fa3", "triton", "flashinfer"])
def test_rejects_effective_outer_backend_overrides(phase, backend):
    with pytest.raises(ValueError, match=f"{phase} attention backend must be dsa"):
        config.validate(args(**{f"{phase}_attention_backend": backend}), MODEL)


def test_split_dsa_overrides_take_precedence_over_base_backend():
    config.validate(
        args(
            attention_backend="fa3",
            prefill_attention_backend="dsa",
            decode_attention_backend="dsa",
        ),
        MODEL,
    )
    with pytest.raises(ValueError, match="attention backend must be dsa"):
        config.validate(args(attention_backend="fa3"), MODEL)
