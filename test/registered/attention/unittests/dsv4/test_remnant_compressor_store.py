"""Exercise the actual compressor hook without model weights.

Author: Winston Cai.
"""

from types import SimpleNamespace
from unittest import mock

import pytest
import torch

from sglang.kernels.ops.attention.dsv4.compress import (
    CompressorDecodePlan,
    CompressorPrefillPlan,
)
from sglang.srt import remnant
from sglang.srt.layers.attention.dsv4 import compressor_v2 as module
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


@pytest.mark.parametrize(
    "implementation,keep_count",
    [("kernel", 256)] + [("triton", k) for k in (64, 128, 192, 256, 320, 384, 448)],
)
@pytest.mark.parametrize("phase", ["decode", "prefill"])
@pytest.mark.parametrize("mode", ["packed", "native", "c128", "indexer"])
def test_compressor_routes_store(phase, mode, implementation, keep_count):
    ratio = 128 if mode == "c128" else 4
    dim = 128 if mode == "indexer" else 512
    latent = torch.randn((2, dim), dtype=torch.float32)
    original_latent = latent.clone()
    plan_blob = torch.tensor([[4, 0, 0, 0], [8, 1, 0, 0]], dtype=torch.int32).view(
        torch.uint8
    )
    plan = (
        CompressorDecodePlan(ratio, plan_blob)
        if phase == "decode"
        else CompressorPrefillPlan(
            ratio, plan_blob, torch.zeros((1, 8), dtype=torch.uint8)
        )
    )
    backend = SimpleNamespace(_get_paged_compress_metadata=lambda unused: plan)
    pool = mock.Mock()
    pool.get_packed_buffers.return_value = (
        torch.empty((1, 64, keep_count), dtype=torch.uint8),
    )
    remnant.configure_cache_format(
        "native" if mode == "native" else "remnant", implementation
    )
    keep_mask = torch.zeros_like(latent, dtype=torch.bool)
    keep_mask[:, :keep_count] = True
    try:
        with (
            mock.patch.object(module, "compress_forward", return_value=latent),
            mock.patch.object(module, "is_overlap_compress", return_value=False),
            mock.patch.object(module, "_use_online_compress", return_value=False),
            mock.patch.object(module, "compress_norm_rope_store") as native,
            mock.patch.object(remnant, "pack_rows_fused") as fused,
            mock.patch.object(remnant, "pack_rows") as explicit,
            mock.patch.object(
                remnant,
                "topmag_keep_mask",
                return_value=keep_mask,
            ) as select,
        ):
            module.CompressorBackendMixin._forward_compress_all_in_one(
                backend,
                kv_score_buffer=torch.zeros((1, ratio * 2, dim * 2)),
                kv_score_input=latent,
                ape=latent,
                head_dim=dim,
                norm=SimpleNamespace(weight=torch.ones(dim), variance_epsilon=1e-5),
                freqs_cis_cache=torch.ones((16, 32), dtype=torch.complex64),
                kv_cache=torch.empty(1),
                is_indexer=mode == "indexer",
                rotate=mode == "indexer",
                compress_ratio=ratio,
                page_size=64,
                out_loc=torch.arange(2, dtype=torch.int32),
                packed_pool=pool,
                packed_layer_id=0,
            )
            if mode == "packed":
                if implementation == "triton":
                    select.assert_called_once_with(latent, keep_count / 512)
                    explicit.assert_called_once()
                    assert explicit.call_args.args[0] is latent
                    assert explicit.call_args.args[1] is keep_mask
                    assert explicit.call_args.args[3] == 1e-5
                    assert explicit.call_args.args[4] is plan
                    fused.assert_not_called()
                else:
                    fused.assert_called_once()
                    assert fused.call_args.args[0] is latent
                    assert fused.call_args.args[2] == 1e-5
                    assert fused.call_args.args[3] is plan
                    select.assert_not_called()
                    explicit.assert_not_called()
                pool.set_rope_freqs.assert_called_once()
                native.assert_not_called()
            else:
                native.assert_called_once()
                assert native.call_args.args[0] is latent
                fused.assert_not_called()
                explicit.assert_not_called()
                select.assert_not_called()
                pool.set_rope_freqs.assert_not_called()
            # Storage dispatch must preserve the raw FP32 compressor output.
            torch.testing.assert_close(latent, original_latent, atol=0, rtol=0)
    finally:
        remnant.configure_cache_format("native")


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
