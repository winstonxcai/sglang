"""CPU-only GLM startup contract; runnable without serving dependencies.

Author: Winston Cai.
"""

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[3]


def load_cpu_module(name, relative):
    # Both contracts are stdlib-only; avoid importing SGLang's serving package.
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


config = load_cpu_module(
    "glm_config_contract", "python/sglang/srt/remnant/glm_config.py"
)
register_cpu_ci = load_cpu_module(
    "glm_ci_contract", "python/sglang/test/ci/ci_register.py"
).register_cpu_ci
register_cpu_ci(est_time=1, suite="base-a-test-cpu")


def args(**changes):
    values = dict(
        kv_cache_format="direct",
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


@pytest.mark.parametrize("model_dtype", [None, "float16", "float32", "bfloat16"])
def test_native_default_is_unrestricted(model_dtype):
    config.validate(
        args(kv_cache_format="native"), SimpleNamespace(), model_dtype=model_dtype
    )


def test_raw_fp8_tilelang_is_supported():
    config.validate(args(), SimpleNamespace(text_config=MODEL), model_dtype="bfloat16")
    assert config.ROW_BYTES == 256 + 8 * 8
    assert 11 * (512 + 33) / (11 * (config.ROW_BYTES + 33)) == 545 / 353


@pytest.mark.parametrize("requested_dtype", ["auto", "bfloat16"])
def test_resolved_bfloat16_activations_are_supported(requested_dtype):
    config.validate(args(dtype=requested_dtype), MODEL, model_dtype="torch.bfloat16")


@pytest.mark.parametrize(
    "requested_dtype,model_dtype",
    [
        ("auto", "torch.float16"),
        ("auto", "torch.float32"),
        ("half", "torch.float16"),
        ("float16", "torch.float16"),
        ("float", "torch.float32"),
        ("float32", "torch.float32"),
    ],
)
def test_rejects_resolved_non_bfloat16_activations(requested_dtype, model_dtype):
    with pytest.raises(ValueError, match="model activation dtype must be bfloat16"):
        config.validate(args(dtype=requested_dtype), MODEL, model_dtype=model_dtype)


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
        config.validate(args(**{name: value}), MODEL, model_dtype="bfloat16")


def test_rejects_other_models():
    with pytest.raises(ValueError, match="GLM-5.3"):
        config.validate(
            args(), SimpleNamespace(model_type="deepseek_v4"), model_dtype="bfloat16"
        )


def test_multimodal_checkpoint_requires_text_only_launch():
    multimodal = SimpleNamespace(text_config=MODEL, vision_config=object())
    with pytest.raises(ValueError, match="language-model-only"):
        config.validate(args(), multimodal, model_dtype="bfloat16")
    config.validate(args(language_model_only=True), multimodal, model_dtype="bfloat16")


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
        config.validate(args(), SimpleNamespace(**fields), model_dtype="bfloat16")


@pytest.mark.parametrize("phase", ["prefill", "decode"])
@pytest.mark.parametrize("backend", ["fa3", "triton", "flashinfer"])
def test_rejects_effective_outer_backend_overrides(phase, backend):
    with pytest.raises(ValueError, match=f"{phase} attention backend must be dsa"):
        config.validate(
            args(**{f"{phase}_attention_backend": backend}),
            MODEL,
            model_dtype="bfloat16",
        )


def test_split_dsa_overrides_take_precedence_over_base_backend():
    config.validate(
        args(
            attention_backend="fa3",
            prefill_attention_backend="dsa",
            decode_attention_backend="dsa",
        ),
        MODEL,
        model_dtype="bfloat16",
    )
    with pytest.raises(ValueError, match="attention backend must be dsa"):
        config.validate(args(attention_backend="fa3"), MODEL, model_dtype="bfloat16")


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))


# Public-format resolution is pure: these checks do not initialize SGLang/Torch.
cache_config = load_cpu_module(
    "cache_config_contract", "python/sglang/srt/remnant/config.py"
)


@pytest.mark.parametrize(
    "family,model_type",
    [("dsv4", "deepseek_v4"), ("dsv41", "deepseek_v41"), ("glm53", "glm5_next_text")],
)
@pytest.mark.parametrize("mode", ["native", "direct"])
def test_generic_selector_keeps_c4_dispatch_model_scoped(
    family, model_type, mode, monkeypatch
):
    # V4.1 validation imports its environment registry; tested in its own CPU suite.
    monkeypatch.setattr(cache_config, "validate_v41_cache_args", lambda args: None)
    hf = SimpleNamespace(model_type=model_type, architectures=["DeepseekV4ForCausalLM"])
    assert cache_config.resolve_model_family(SimpleNamespace(text_config=hf)) == family
    cache_config.configure_from_args(
        SimpleNamespace(_kv_cache_model_family=family, kv_cache_format=mode)
    )
    assert cache_config.packed_enabled() == (family == "dsv4" and mode == "direct")
    assert cache_config.cache_implementation() == "kernel"
    cache_config.configure_from_args(SimpleNamespace())
    assert not cache_config.packed_enabled()


@pytest.mark.parametrize(
    "mode", ["auto", "fp4", "remnant", "adapter", "triton", "unknown"]
)
def test_generic_selector_rejects_removed_values(mode):
    with pytest.raises(ValueError, match="kv_cache_format"):
        cache_config.configure_from_args(SimpleNamespace(kv_cache_format=mode))


def test_direct_profiles_fail_before_mutating_dispatch():
    cache_config.configure_from_args(
        SimpleNamespace(_kv_cache_model_family="dsv4", kv_cache_format="direct")
    )
    for count in (64, 192, 320, 448):
        with pytest.raises(ValueError, match="256 survivors"):
            cache_config.configure_from_args(
                SimpleNamespace(
                    _kv_cache_model_family="dsv4",
                    kv_cache_format="direct",
                    dsv4_c4_cache_keep_count=count,
                )
            )
        assert cache_config.packed_enabled()
    cache_config.configure_from_args(SimpleNamespace())


@pytest.mark.parametrize("model_type", ["qwen2", "qwen3_vl", None])
def test_native_unrelated_models_accept_missing_architectures(model_type):
    text = SimpleNamespace(model_type=model_type, architectures=None)
    assert cache_config.resolve_model_family(text) == "unknown"
    assert (
        cache_config.resolve_model_family(SimpleNamespace(text_config=text))
        == "unknown"
    )


def test_direct_rejects_unresolved_family():
    with pytest.raises(ValueError, match="resolved supported model"):
        cache_config.configure_from_args(SimpleNamespace(kv_cache_format="direct"))


@pytest.mark.parametrize(
    "mode,expected", [("native", "sm90_fp4"), ("direct", "sm90_remnant")]
)
@pytest.mark.parametrize("backend", ["auto", "flashmla"])
def test_v41_default_backend_selects_fp4_before_pool_allocation(
    mode, expected, backend
):
    import ast

    source = ROOT / "python/sglang/srt/mem_cache/deepseek_v4_memory_pool.py"
    function = next(
        n
        for n in ast.parse(source.read_text()).body
        if isinstance(n, ast.FunctionDef) and n.name == "select_dsv4_kv_layout"
    )
    namespace = dict(
        get_model=lambda: SimpleNamespace(
            cache_model_family="dsv41", kv_cache_format=mode
        ),
        get_platform=lambda: SimpleNamespace(is_sm90=True),
        get_exec=lambda: SimpleNamespace(
            kernel=SimpleNamespace(dsv4_attn_backend=backend)
        ),
        KVLayout=SimpleNamespace(V4="unchanged-swa"),
    )
    exec("from __future__ import annotations\n" + ast.unparse(function), namespace)
    assert namespace["select_dsv4_kv_layout"]() == ("unchanged-swa", expected)


@pytest.mark.parametrize("fp4", [False, True])
def test_fused_store_frequency_guard_preserves_raw_arguments(fp4):
    import ast

    source = ROOT / "python/sglang/srt/mem_cache/deepseek_v4_memory_pool.py"
    cls = next(
        n
        for n in ast.parse(source.read_text()).body
        if isinstance(n, ast.ClassDef) and n.name == "DeepSeekV4TokenToKVPool"
    )
    method = next(
        n
        for n in cls.body
        if isinstance(n, ast.FunctionDef) and n.name == "set_extra_key_buffer_fused"
    )
    namespace = {}
    exec("from __future__ import annotations\n" + ast.unparse(method), namespace)
    seen = []
    subpool = SimpleNamespace(
        kv_layout=SimpleNamespace(is_fp4=fp4),
        set_key_buffer_fused=lambda *args: seen.append(args),
    )
    pool = SimpleNamespace(layer_mapping={3: (2, 7, subpool)})
    loc, latent, freq = object(), object(), object()
    if fp4:
        namespace[method.name](pool, 3, loc, latent, freq)
        assert seen == [(7, loc, latent, freq)]
    else:
        with pytest.raises(AssertionError, match="fp4 cache"):
            namespace[method.name](pool, 3, loc, latent, freq)
        assert not seen
