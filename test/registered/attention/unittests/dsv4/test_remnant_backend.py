"""Backend metadata and selected-row reconstruction contracts.

Author: Winston Cai.
"""

import sys
import unittest
from types import SimpleNamespace
from unittest import mock

import torch

from sglang.srt import remnant
from sglang.srt.layers.attention import deepseek_v4_backend as module
from sglang.srt.layers.attention.deepseek_v4_backend import DSV4AttnMetadata
from sglang.srt.remnant import NativeWorkspace, capture
from sglang.srt.remnant.packed import unpack_gather_native
from sglang.test.test_utils import CustomTestCase


class TestRemnantBackend(CustomTestCase):
    def tearDown(self):
        remnant.configure_cache_format("native")

    def test_workspace_allocates_dense_buffer_only_for_triton(self):
        for cache_format, implementation in [
            ("native", "triton"),
            ("remnant", "triton"),
            ("remnant", "kernel"),
        ]:
            with self.subTest(cache_format=cache_format, implementation=implementation):
                remnant.configure_cache_format(cache_format, implementation)
                workspace = NativeWorkspace.allocate(2, 8, 64, "cpu")
                if cache_format == "remnant" and implementation == "triton":
                    self.assertEqual(workspace.dense_bf16.shape, (16, 512))
                    self.assertEqual(workspace.dense_bf16.dtype, torch.bfloat16)
                else:
                    self.assertIsNone(workspace.dense_bf16)

    def test_triton_materialization_uses_strides_and_dense_destination_indices(self):
        for keep_count in (64, 128, 192, 256, 320, 384, 448):
            with self.subTest(keep_count=keep_count):
                self._check_triton_materialization(keep_count)

    def _check_triton_materialization(self, keep_count):
        # Default stays 256: the wrapper must derive each layer's actual width.
        remnant.configure_cache_format("remnant", "triton")
        workspace = NativeWorkspace.allocate(2, 8, 64, "cpu")
        physical = torch.zeros((2, 10), dtype=torch.int32)[:, :3]
        raw = torch.zeros((2, 9), dtype=torch.int32)[:, :3]
        lengths = torch.tensor([3, 1], dtype=torch.int32)
        buffers = (
            torch.zeros((4, keep_count), dtype=torch.uint8),
            torch.zeros((4, 8), dtype=torch.uint64),
            torch.zeros((4, 8), dtype=torch.uint8),
        )
        materialize = mock.MagicMock()
        with (
            mock.patch("sglang.srt.remnant.packed.unpack_gather_bf16") as gather,
            mock.patch.dict(
                sys.modules,
                {
                    "sglang.srt.remnant.triton": SimpleNamespace(
                        _bf16_to_native_kernel=materialize
                    ),
                    "sgl_kernel.remnant_ops": None,
                },
            ),
        ):
            cache, indices = unpack_gather_native(
                buffers,
                physical,
                raw,
                lengths,
                torch.ones((16, 32), dtype=torch.complex64),
                workspace,
            )
        self.assertIs(cache, workspace.native_bytes)
        self.assertTrue(
            torch.equal(indices, torch.arange(6, dtype=torch.int32).reshape(2, 3))
        )
        self.assertEqual(indices.data_ptr(), workspace.temporary_indices.data_ptr())
        gather.assert_called_once()
        materialize.__getitem__.assert_called_once_with((6,))
        launch = materialize.__getitem__.return_value
        launch.assert_called_once()
        self.assertEqual(launch.call_args.kwargs["KEEP_K"], keep_count)
        self.assertEqual(launch.call_args.kwargs["PACKED_ROWS"], 4)
        self.assertEqual(launch.call_args.kwargs["PHYSICAL_STRIDE_Q"], 10)
        self.assertEqual(launch.call_args.kwargs["RAW_STRIDE_Q"], 9)
        self.assertEqual(
            launch.call_args.args[0].data_ptr(), workspace.dense_bf16.data_ptr()
        )

    def test_triton_reconstruction_rejects_missing_dense_workspace(self):
        remnant.configure_cache_format("remnant", "kernel")
        workspace = NativeWorkspace.allocate(1, 1, 64, "cpu")
        remnant.configure_cache_format("remnant", "triton")
        with self.assertRaisesRegex(RuntimeError, "allocated in Triton mode"):
            unpack_gather_native(
                (torch.empty(1, 256), torch.empty(1, 8), torch.empty(1, 8)),
                torch.zeros((1, 1), dtype=torch.int32),
                torch.zeros((1, 1), dtype=torch.int32),
                torch.ones(1, dtype=torch.int32),
                torch.ones((16, 32), dtype=torch.complex64),
                workspace,
            )

    def test_decode_and_small_extend_dispatch(self):
        for mode in ("native", "triton", "kernel"):
            for extend in (False, True):
                with self.subTest(mode=mode, extend=extend):
                    self._check_attention_dispatch(mode, extend)

    def _check_attention_dispatch(self, mode, extend):
        remnant.configure_cache_format(
            "native" if mode == "native" else "remnant",
            "triton" if mode == "triton" else "kernel",
        )
        workspace = NativeWorkspace.allocate(1, 64, 64, "cpu")
        core = mock.Mock(spec=DSV4AttnMetadata)
        core.c4_sparse_page_indices = torch.zeros((1, 64), dtype=torch.int32)
        core.c4_sparse_raw_indices = torch.ones((1, 64), dtype=torch.int32)
        core.c4_sparse_topk_lengths = torch.tensor([32], dtype=torch.int32)
        core.swa_page_indices = torch.zeros((1, 64), dtype=torch.int32)
        core.swa_topk_lengths = torch.tensor([64], dtype=torch.int32)
        pool = mock.Mock(spec=module.DeepSeekV4TokenToKVPool)
        pool.page_size = 256
        pool.swa_window_size = 64
        pool.swa_kv_pool = SimpleNamespace(kv_cache_total_dim=584)
        pool.get_swa_key_buffer_radix.return_value = torch.zeros((1, 64 * 584))
        pool.get_extra_key_buffer.return_value = workspace.native_bytes
        pool.get_packed_freqs.return_value = torch.ones((16, 32), dtype=torch.complex64)
        backend = SimpleNamespace(
            mtp_enabled=False,
            forward_metadata=SimpleNamespace(core_attn_metadata=core),
            token_to_kv_pool=pool,
            remnant_workspace=workspace,
            head_dim_v=512,
            softmax_scale=512**-0.5,
        )
        batch = SimpleNamespace(forward_mode=mock.Mock())
        batch.forward_mode.is_extend_without_speculative.return_value = extend
        q = torch.ones((1, 64, 512), dtype=torch.bfloat16)
        flashmla = mock.Mock(return_value=(q.unsqueeze(1), torch.zeros((1, 64))))
        with (
            mock.patch.object(module, "_is_sm90", True),
            mock.patch.object(module, "_is_sm120", False),
            mock.patch.object(capture, "enabled", return_value=False),
            mock.patch.object(
                module.envs.SGLANG_OPT_FLASHMLA_SPARSE_PREFILL,
                "get",
                return_value=False,
            ),
            mock.patch.object(
                remnant,
                "unpack_gather_native",
                return_value=(workspace.native_bytes, workspace.temporary_indices),
            ) as reconstruct,
            mock.patch.dict(
                sys.modules,
                {
                    "sgl_kernel.flash_mla": SimpleNamespace(
                        flash_mla_with_kvcache=flashmla
                    )
                },
            ),
        ):
            output = module.DeepseekV4AttnBackend.forward(
                backend,
                q,
                q,
                q,
                SimpleNamespace(layer_id=0),
                batch,
                compress_ratio=4,
                save_kv_cache=False,
                attn_sink=torch.zeros(64),
            )
        self.assertTrue(torch.equal(output, q))
        kwargs = flashmla.call_args.kwargs
        if mode == "triton":
            reconstruct.assert_called_once()
            self.assertIsNone(kwargs["remnant_buffers"])
            self.assertIsNotNone(kwargs["extra_k_cache"])
            self.assertTrue(
                torch.equal(reconstruct.call_args.args[2], core.c4_sparse_raw_indices)
            )
        elif mode == "kernel":
            reconstruct.assert_not_called()
            self.assertIs(
                kwargs["remnant_buffers"], pool.get_packed_buffers.return_value
            )
            self.assertIsNone(kwargs["extra_k_cache"])
        else:
            reconstruct.assert_not_called()
            pool.get_packed_buffers.assert_not_called()
            self.assertIsNone(kwargs["remnant_buffers"])
            self.assertIsNotNone(kwargs["extra_k_cache"])

    def test_packed_metadata_retains_raw_indices(self):
        metadata = DSV4AttnMetadata(
            page_size=256,
            page_table=torch.zeros((1, 1), dtype=torch.int32),
            raw_out_loc=torch.zeros(1, dtype=torch.int32),
            cuda_int32_kwargs={},
            seq_lens_casual=torch.ones(1, dtype=torch.int32),
            positions_casual=torch.zeros(1, dtype=torch.int32),
            swa_page_indices=torch.zeros((1, 1), dtype=torch.int32),
            swa_topk_lengths=torch.ones(1, dtype=torch.int32),
            c4_sparse_topk=512,
            c4_topk_lengths_clamp1=torch.ones(1, dtype=torch.int32),
            c128_page_indices=torch.zeros((1, 1), dtype=torch.int32),
            c128_topk_lengths_clamp1=torch.ones(1, dtype=torch.int32),
        )
        with mock.patch(
            "sglang.srt.layers.attention.deepseek_v4_backend._create_flashmla_metadata",
            return_value=None,
        ):
            remnant.configure_cache_format("native")
            metadata.init_flashmla_related()
            self.assertIsNone(metadata.c4_sparse_raw_indices)

            remnant.configure_cache_format("remnant")
            metadata.init_flashmla_related()
            self.assertEqual(metadata.c4_sparse_raw_indices.shape, (1, 512))
        remnant.configure_cache_format("native")

    def test_native_reconstruction_rejects_workspace_overflow(self):
        remnant.configure_cache_format("remnant")
        buffers = tuple(
            torch.empty(shape, dtype=dtype)
            for shape, dtype in (
                ((1, 256, 256), torch.uint8),
                ((1, 256, 8), torch.uint64),
                ((1, 256, 8), torch.uint8),
            )
        )
        workspace = NativeWorkspace.allocate(1, 1, 64, "cpu")
        with self.assertRaisesRegex(ValueError, "top-k capacity"):
            unpack_gather_native(
                buffers,
                torch.zeros((1, 2), dtype=torch.int32),
                torch.zeros((1, 2), dtype=torch.int32),
                torch.ones(1, dtype=torch.int32),
                torch.ones(256, dtype=torch.complex64),
                workspace,
            )
        remnant.configure_cache_format("native")


if __name__ == "__main__":
    unittest.main()
