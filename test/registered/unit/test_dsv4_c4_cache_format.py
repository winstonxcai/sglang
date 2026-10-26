"""Tests for the public DeepSeek V4 C4 cache-format switch.

Author: Winston Cai.
"""

import argparse
import multiprocessing
import unittest
from multiprocessing.connection import Connection
from types import SimpleNamespace
from unittest import mock

from sglang.srt import remnant
from sglang.srt.server_args import ServerArgs, set_global_server_args_for_scheduler
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def _spawned_worker_modes(
    server_args: list[ServerArgs], connection: Connection
) -> None:
    """Exercise the same publication hook used before ModelRunner pool sizing."""

    def state() -> tuple[str, str, bool, str]:
        return (
            remnant.cache_format(),
            remnant.cache_implementation(),
            remnant.packed_enabled(),
            remnant.cache_store_implementation(),
        )

    try:
        states = [state()]
        for args in server_args:
            set_global_server_args_for_scheduler(args)
            states.append(state())
        connection.send(states)
    finally:
        connection.close()


class TestDSV4C4CacheFormat(CustomTestCase):
    def setUp(self):
        super().setUp()
        from sglang.srt.runtime_context import ParallelContext, RuntimeContext

        # Publication tests must not replace another test's process context.
        context = RuntimeContext(ParallelContext())
        context_patch = mock.patch(
            "sglang.srt.runtime_context._CONTEXT",
            context,
        )
        context_patch.start()
        self.addCleanup(context_patch.stop)
        parallel_patch = mock.patch(
            "sglang.srt.runtime_context._PARALLEL", context.parallel
        )
        parallel_patch.start()
        self.addCleanup(parallel_patch.stop)

    def tearDown(self):
        remnant.configure_cache_format("native")

    @classmethod
    def setUpClass(cls):
        cls.parser = argparse.ArgumentParser()
        ServerArgs.add_cli_args(cls.parser)

    def _parse(self, values):
        args = self.parser.parse_args(["--model", "dummy", *values])
        record = ServerArgs.from_cli_args(args)
        record.resolve_once()
        set_global_server_args_for_scheduler(record)
        return record

    def test_native_is_the_default(self):
        args = self._parse([])
        self.assertEqual(args.dsv4_c4_cache_format, "native")
        self.assertEqual(args.dsv4_c4_cache_implementation, "kernel")
        self.assertFalse(remnant.packed_enabled())

    def test_remnant_is_explicit(self):
        self.assertEqual(
            self._parse(["--dsv4-c4-cache-format", "remnant"]).dsv4_c4_cache_format,
            "remnant",
        )

    def test_unknown_format_is_rejected(self):
        with self.assertRaises(SystemExit):
            self.parser.parse_args(
                ["--model", "dummy", "--dsv4-c4-cache-format", "packed"]
            )

    def test_packed_implementations_are_configured_at_startup(self):
        for implementation in ("triton", "kernel"):
            with self.subTest(implementation=implementation):
                args = self._parse(
                    [
                        "--dsv4-c4-cache-format",
                        "remnant",
                        "--dsv4-c4-cache-implementation",
                        implementation,
                    ]
                )
                self.assertEqual(args.dsv4_c4_cache_implementation, implementation)
                self.assertEqual(remnant.cache_implementation(), implementation)
                self.assertTrue(remnant.packed_enabled())

    def _worker_states(
        self, args: list[ServerArgs]
    ) -> list[tuple[str, str, bool, str]]:
        context = multiprocessing.get_context("spawn")
        receiver, sender = context.Pipe(duplex=False)
        worker = context.Process(target=_spawned_worker_modes, args=(args, sender))
        try:
            worker.start()
            sender.close()
            worker.join(timeout=60)
            self.assertFalse(worker.is_alive(), "worker mode initialization timed out")
            self.assertEqual(worker.exitcode, 0, "worker mode initialization failed")
            self.assertTrue(receiver.poll(), "worker returned no mode state")
            return receiver.recv()
        finally:
            if worker.is_alive():
                worker.terminate()
                worker.join(timeout=10)
                if worker.is_alive():
                    worker.kill()
                    worker.join(timeout=10)
            receiver.close()
            sender.close()
            if worker.exitcode is not None:
                worker.close()

    def test_all_modes_in_spawned_workers(self):
        for cache_format, implementation, packed in (
            ("native", "kernel", False),
            ("remnant", "triton", True),
            ("remnant", "kernel", True),
        ):
            with self.subTest(cache_format=cache_format, implementation=implementation):
                args = self._parse(
                    [
                        "--dsv4-c4-cache-format",
                        cache_format,
                        "--dsv4-c4-cache-implementation",
                        implementation,
                    ]
                )
                # Neither inherited globals nor rerunning the constructor may
                # be required: the resolved args are pickled into a fresh worker.
                remnant.configure_cache_format("remnant", "triton")
                self.assertEqual(
                    self._worker_states([args]),
                    [
                        ("native", "kernel", False, "kernel"),
                        (cache_format, implementation, packed, implementation),
                    ],
                )

    def test_store_override_in_spawned_worker(self):
        args = []
        expected = [("native", "kernel", False, "kernel")]
        for decode in ("triton", "kernel"):
            for store in ("triton", "kernel"):
                resolved = self._parse(
                    [
                        "--dsv4-c4-cache-format",
                        "remnant",
                        "--dsv4-c4-cache-implementation",
                        decode,
                        "--dsv4-c4-cache-store-implementation",
                        store,
                    ]
                )
                self.assertEqual(resolved.dsv4_c4_cache_store_implementation, store)
                self.assertEqual(remnant.cache_implementation(), decode)
                self.assertEqual(remnant.cache_store_implementation(), store)
                args.append(resolved)
                expected.append(("remnant", decode, True, store))
        # Republishing without an override must clear the previous override.
        args.append(
            self._parse(
                [
                    "--dsv4-c4-cache-format",
                    "remnant",
                    "--dsv4-c4-cache-implementation",
                    "triton",
                ]
            )
        )
        expected.append(("remnant", "triton", True, "triton"))
        self.assertEqual(self._worker_states(args), expected)

    def test_invalid_store_override_preserves_configuration(self):
        remnant.configure_cache_format("remnant", "triton", "kernel")
        with self.assertRaisesRegex(ValueError, "store_implementation"):
            remnant.configure_cache_format("native", "kernel", "cuda")
        self.assertEqual(remnant.cache_format(), "remnant")
        self.assertEqual(remnant.cache_implementation(), "triton")
        self.assertEqual(remnant.cache_store_implementation(), "kernel")
        with self.assertRaises(SystemExit):
            self.parser.parse_args(
                [
                    "--model",
                    "dummy",
                    "--dsv4-c4-cache-store-implementation",
                    "cuda",
                ]
            )

    def test_worker_republication_replaces_previous_mode(self):
        modes = [("remnant", "triton"), ("remnant", "kernel"), ("native", "kernel")]
        args = [
            self._parse(
                [
                    "--dsv4-c4-cache-format",
                    cache_format,
                    "--dsv4-c4-cache-implementation",
                    implementation,
                ]
            )
            for cache_format, implementation in modes
        ]
        self.assertEqual(
            self._worker_states(args),
            [
                ("native", "kernel", False, "kernel"),
                ("remnant", "triton", True, "triton"),
                ("remnant", "kernel", True, "kernel"),
                ("native", "kernel", False, "kernel"),
            ],
        )

    def test_default_record_publication_resets_to_native(self):
        remnant.configure_cache_format("remnant", "triton")
        set_global_server_args_for_scheduler(ServerArgs(model_path="dummy"))
        self.assertEqual(remnant.cache_format(), "native")
        self.assertEqual(remnant.cache_implementation(), "kernel")
        self.assertFalse(remnant.packed_enabled())

    def test_invalid_publication_preserves_worker_configuration(self):
        from sglang.srt.runtime_context import get_server_args

        args = self._parse(
            [
                "--dsv4-c4-cache-format",
                "remnant",
                "--dsv4-c4-cache-implementation",
                "triton",
            ]
        )
        set_global_server_args_for_scheduler(args)
        with self.assertRaises(ValueError):
            set_global_server_args_for_scheduler(
                ServerArgs(
                    model_path="dummy",
                    dsv4_c4_cache_format="remnant",
                    dsv4_c4_cache_implementation="cuda",
                )
            )
        self.assertIs(get_server_args(), args)
        self.assertEqual(remnant.cache_format(), "remnant")
        self.assertEqual(remnant.cache_implementation(), "triton")

    def test_native_ignores_packed_implementation(self):
        self._parse(["--dsv4-c4-cache-implementation", "triton"])
        self.assertFalse(remnant.packed_enabled())
        self.assertEqual(remnant.cache_format(), "native")
        self.assertEqual(remnant.cache_implementation(), "triton")

    def test_unknown_implementation_is_rejected(self):
        with self.assertRaises(SystemExit):
            self.parser.parse_args(
                [
                    "--model",
                    "dummy",
                    "--dsv4-c4-cache-implementation",
                    "cuda",
                ]
            )

    def test_per_layer_counts_publish_and_reset(self):
        args = self._parse(
            [
                "--dsv4-c4-cache-format",
                "remnant",
                "--dsv4-c4-cache-implementation",
                "triton",
                "--dsv4-c4-cache-keep-count",
                "192",
                "--dsv4-c4-cache-layer-keep-counts",
                '{"4":320}',
            ]
        )
        args._dsv4_c4_cache_compression_ratios = (0, 128, 4, 128, 4)
        set_global_server_args_for_scheduler(args)
        self.assertEqual(
            [
                (x.layer_id, x.keep_count)
                for x in remnant.packed_layouts(args._dsv4_c4_cache_compression_ratios)
            ],
            [(2, 192), (4, 320)],
        )
        previous = remnant.packed_profile()
        with self.assertRaises(ValueError):
            set_global_server_args_for_scheduler(
                ServerArgs(
                    model_path="dummy",
                    dsv4_c4_cache_format="remnant",
                    dsv4_c4_cache_implementation="triton",
                    dsv4_c4_cache_keep_count=512,
                )
            )
        from sglang.srt.runtime_context import get_server_args

        self.assertIs(get_server_args(), args)
        self.assertIs(remnant.packed_profile(), previous)
        set_global_server_args_for_scheduler(ServerArgs(model_path="dummy"))
        self.assertEqual(remnant.packed_profile().default_count, 256)
        self.assertEqual(remnant.packed_profile().overrides, ())
        self.assertIsNone(remnant.packed_profile().compression_ratios)

    def test_invalid_cli_counts_and_nontriton_modes(self):
        base = [
            "--dsv4-c4-cache-format",
            "remnant",
            "--dsv4-c4-cache-implementation",
            "triton",
        ]
        for count in (0, 65, 512):
            with self.subTest(count=count), self.assertRaises(ValueError):
                self._parse([*base, "--dsv4-c4-cache-keep-count", str(count)])
        for overrides in ("[]", '{"2":192,"2":256}', '{"2":true}'):
            with self.subTest(overrides=overrides), self.assertRaises(ValueError):
                self._parse([*base, "--dsv4-c4-cache-layer-keep-counts", overrides])
        for extra in (
            ["--dsv4-c4-cache-store-implementation", "kernel"],
            ["--enable-hierarchical-cache"],
        ):
            with self.assertRaises(ValueError):
                self._parse([*base, "--dsv4-c4-cache-keep-count", "192", *extra])

    def test_existing_callers_default_to_kernel(self):
        remnant.configure_cache_format("remnant", "triton")
        remnant.configure_cache_format("remnant")
        self.assertEqual(remnant.cache_implementation(), "kernel")

    def test_real_record_uses_current_cp_option_and_declared_hicache_layout(self):
        from sglang.srt.arg_groups.overrides import resolving_view

        for cp in (False, True):
            args = ServerArgs(
                model_path="dummy",
                dsv4_c4_cache_format="remnant",
                dsv4_c4_cache_implementation="triton",
                enable_prefill_cp=cp,
                enable_hierarchical_cache=True,
                hicache_mem_layout="page_first",
            )
            hf = SimpleNamespace(compress_ratios=(0, 128, 4, 128, 4))
            with (
                mock.patch(
                    "sglang.srt.server_args.model_config_of",
                    return_value=SimpleNamespace(hf_config=hf),
                ),
                mock.patch(
                    "sglang.srt.configs.model_config.is_deepseek_v4", return_value=True
                ),
            ):
                if cp:
                    with self.assertRaisesRegex(ValueError, "--enable-prefill-cp"):
                        args._configure_dsv4_c4_cache_format()
                else:
                    args._configure_dsv4_c4_cache_format()
                    self.assertEqual(args.hicache_mem_layout, "page_first")
                    self.assertEqual(
                        resolving_view(args).hicache_mem_layout, "layer_first"
                    )

    def test_model_context_preserves_and_restores_packed_dispatch(self):
        from sglang.srt.runtime_context import (
            create_context,
            reset_context,
            restore_context,
            snapshot_context,
            use_context,
        )

        packed = self._parse(
            [
                "--dsv4-c4-cache-format",
                "remnant",
                "--dsv4-c4-cache-implementation",
                "triton",
            ]
        )
        saved = snapshot_context()
        native = create_context(ServerArgs(model_path="dummy"), role="test")
        self.assertTrue(remnant.packed_enabled())
        with use_context(native):
            self.assertFalse(remnant.packed_enabled())
        self.assertTrue(remnant.packed_enabled())
        reset_context()
        self.assertFalse(remnant.packed_enabled())
        restore_context(saved)
        self.assertTrue(remnant.packed_enabled())
        self.assertIs(saved["_server_args"], packed)

    def test_context_overrides_restore_dispatch_on_nested_exception(self):
        from sglang.srt.runtime_context import get_context

        self._parse(
            [
                "--dsv4-c4-cache-format",
                "remnant",
                "--dsv4-c4-cache-implementation",
                "triton",
            ]
        )
        context = get_context()
        with self.assertRaisesRegex(RuntimeError, "scope exit"):
            with context.override_server_args(dsv4_c4_cache_format="native"):
                self.assertFalse(remnant.packed_enabled())
                with context.override_server_args(
                    dsv4_c4_cache_format="remnant",
                    dsv4_c4_cache_implementation="triton",
                    dsv4_c4_cache_keep_count=192,
                ):
                    self.assertTrue(remnant.packed_enabled())
                    self.assertEqual(remnant.packed_profile().default_count, 192)
                    self.assertEqual(
                        context.config_bag("model").dsv4_c4_cache_keep_count, 192
                    )
                self.assertFalse(remnant.packed_enabled())
                raise RuntimeError("scope exit")
        self.assertTrue(remnant.packed_enabled())
        self.assertEqual(remnant.packed_profile().default_count, 256)

    def test_context_reentry_keeps_republished_dispatch(self):
        from sglang.srt.runtime_context import (
            create_context,
            get_context,
            use_context,
        )

        self._parse([])
        context = get_context()
        with use_context(context):
            self._parse(
                [
                    "--dsv4-c4-cache-format",
                    "remnant",
                    "--dsv4-c4-cache-implementation",
                    "triton",
                ]
            )
        self.assertTrue(remnant.packed_enabled())
        other = create_context(ServerArgs(model_path="dummy"), role="test")
        with use_context(other):
            context.set_server_args(ServerArgs(model_path="dummy"))
            self.assertFalse(remnant.packed_enabled())
        self.assertFalse(remnant.packed_enabled())

    def test_invalid_configuration_does_not_change_state(self):
        remnant.configure_cache_format("remnant", "triton")
        for cache_format, implementation in [("packed", "kernel"), ("native", "cuda")]:
            with self.subTest(cache_format=cache_format, implementation=implementation):
                with self.assertRaises(ValueError):
                    remnant.configure_cache_format(cache_format, implementation)
                self.assertEqual(remnant.cache_format(), "remnant")
                self.assertEqual(remnant.cache_implementation(), "triton")

    def test_packed_restrictions_apply_to_both_implementations(self):
        for implementation in ("triton", "kernel"):
            args = SimpleNamespace(
                dsv4_c4_cache_format="remnant",
                dsv4_c4_cache_implementation=implementation,
                get_model_config=lambda: SimpleNamespace(hf_config=object()),
                enable_hisparse=True,
                speculative_algorithm=None,
                cpu_offload_gb=0,
                disaggregation_decode_enable_offload_kvcache=False,
                enable_hierarchical_cache=False,
                enable_prefill_cp=False,
            )
            with (
                self.subTest(implementation=implementation),
                mock.patch(
                    "sglang.srt.configs.model_config.is_deepseek_v4", return_value=True
                ),
                mock.patch(
                    "sglang.srt.server_args.model_config_of",
                    return_value=SimpleNamespace(hf_config=object()),
                ),
            ):
                with self.assertRaisesRegex(ValueError, "--enable-hisparse"):
                    ServerArgs._configure_dsv4_c4_cache_format(args)


if __name__ == "__main__":
    unittest.main()
