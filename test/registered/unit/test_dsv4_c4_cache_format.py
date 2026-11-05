"""Tests for the generic Native/Direct serving switch and internal C4 profiles.

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
        record._kv_cache_model_family = "dsv4"
        record.resolve_once()
        set_global_server_args_for_scheduler(record)
        return record

    def test_native_is_the_default(self):
        args = self._parse([])
        self.assertEqual(args.kv_cache_format, "native")
        self.assertFalse(remnant.packed_enabled())

    def test_remnant_is_explicit(self):
        self.assertEqual(
            self._parse(["--kv-cache-format", "direct"]).kv_cache_format, "direct"
        )
        self.assertTrue(remnant.packed_enabled())

    def test_unknown_format_is_rejected(self):
        for value in ("remnant", "packed", "adapter", "fp4", "auto"):
            with self.subTest(value=value), self.assertRaises(SystemExit):
                self.parser.parse_args(["--model", "dummy", "--kv-cache-format", value])

    def test_packed_implementations_are_configured_at_startup(self):
        self._parse(["--kv-cache-format", "direct"])
        self.assertEqual(remnant.cache_implementation(), "kernel")
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
        for mode in ("native", "direct"):
            args = self._parse(["--kv-cache-format", mode])
            remnant.configure_cache_format("remnant", "triton")
            self.assertEqual(
                self._worker_states([args]),
                [
                    ("native", "kernel", False, "kernel"),
                    (
                        "remnant" if mode == "direct" else "native",
                        "kernel",
                        mode == "direct",
                        "kernel",
                    ),
                ],
            )

    def test_store_override_in_spawned_worker(self):
        args, expected = [], [("native", "kernel", False, "kernel")]
        for store in ("triton", "kernel"):
            resolved = self._parse(
                [
                    "--kv-cache-format",
                    "direct",
                    "--dsv4-c4-cache-store-implementation",
                    store,
                ]
            )
            args.append(resolved)
            expected.append(("remnant", "kernel", True, store))
        args.append(self._parse(["--kv-cache-format", "direct"]))
        expected.append(("remnant", "kernel", True, "kernel"))
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
        args = [
            self._parse(["--kv-cache-format", mode])
            for mode in ("direct", "direct", "native")
        ]
        self.assertEqual(
            self._worker_states(args),
            [
                ("native", "kernel", False, "kernel"),
                ("remnant", "kernel", True, "kernel"),
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

        args = self._parse(["--kv-cache-format", "direct"])
        with self.assertRaises(ValueError):
            set_global_server_args_for_scheduler(
                ServerArgs(model_path="dummy", kv_cache_format="invalid")
            )
        self.assertIs(get_server_args(), args)
        self.assertEqual(remnant.cache_format(), "remnant")
        self.assertEqual(remnant.cache_implementation(), "kernel")

    def test_native_ignores_packed_implementation(self):
        # Reconstruction remains an internal benchmark/test choice.
        remnant.configure_cache_format("native", "triton")
        self.assertFalse(remnant.packed_enabled())
        self.assertEqual(remnant.cache_implementation(), "triton")
        self._parse([])
        self.assertEqual(remnant.cache_implementation(), "kernel")

    def test_unknown_implementation_is_rejected(self):
        for option in (
            "--dsv4-c4-cache-format",
            "--dsv4-c4-cache-implementation",
            "--dsv41-main-kv-cache-format",
            "--glm5-mla-cache-format",
            "--glm5-mla-cache-implementation",
        ):
            with self.subTest(option=option), self.assertRaises(SystemExit):
                self.parser.parse_args(["--model", "dummy", option, "native"])

    def test_per_layer_counts_publish_and_reset(self):
        # Internal reconstruction still supports nondefault profiles, including atomic failures.
        remnant.configure_cache_format(
            "remnant",
            "triton",
            keep_count=192,
            layer_keep_counts='{"4":320}',
            compression_ratios=(0, 128, 4, 128, 4),
        )
        self.assertEqual(
            [
                (x.layer_id, x.keep_count)
                for x in remnant.packed_layouts((0, 128, 4, 128, 4))
            ],
            [(2, 192), (4, 320)],
        )
        previous = remnant.packed_profile()
        with self.assertRaises(ValueError):
            remnant.configure_cache_format("remnant", "triton", keep_count=512)
        self.assertIs(remnant.packed_profile(), previous)
        set_global_server_args_for_scheduler(ServerArgs(model_path="dummy"))
        self.assertEqual(remnant.packed_profile().default_count, 256)
        self.assertEqual(remnant.packed_profile().overrides, ())
        self.assertIsNone(remnant.packed_profile().compression_ratios)

    def test_invalid_cli_counts_and_nontriton_modes(self):
        for count in (0, 64, 192, 320, 512):
            with self.subTest(count=count), self.assertRaises(ValueError):
                self._parse(
                    [
                        "--kv-cache-format",
                        "direct",
                        "--dsv4-c4-cache-keep-count",
                        str(count),
                    ]
                )
        for overrides in ("[]", '{"2":192,"2":256}', '{"2":true}', '{"2":192}'):
            with self.subTest(overrides=overrides), self.assertRaises(ValueError):
                self._parse(
                    [
                        "--kv-cache-format",
                        "direct",
                        "--dsv4-c4-cache-layer-keep-counts",
                        overrides,
                    ]
                )

    def test_existing_callers_default_to_kernel(self):
        remnant.configure_cache_format("remnant", "triton")
        remnant.configure_cache_format("remnant")
        self.assertEqual(remnant.cache_implementation(), "kernel")

    def test_real_record_uses_current_cp_option_and_declared_hicache_layout(self):
        from sglang.srt.arg_groups.overrides import resolving_view

        for cp in (False, True):
            args = ServerArgs(
                model_path="dummy",
                kv_cache_format="direct",
                enable_prefill_cp=cp,
                enable_hierarchical_cache=True,
                hicache_mem_layout="page_first",
            )
            hf = SimpleNamespace(
                model_type="deepseek_v4", compress_ratios=(0, 128, 4, 128, 4)
            )
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
                        args._configure_cache_format()
                else:
                    args._configure_cache_format()
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
                "--kv-cache-format",
                "direct",
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
                "--kv-cache-format",
                "direct",
            ]
        )
        context = get_context()
        with self.assertRaisesRegex(RuntimeError, "scope exit"):
            with context.override_server_args(kv_cache_format="native"):
                self.assertFalse(remnant.packed_enabled())
                with context.override_server_args(
                    _kv_cache_model_family="dsv4",
                    kv_cache_format="direct",
                    dsv4_c4_cache_keep_count=256,
                ):
                    self.assertTrue(remnant.packed_enabled())
                    self.assertEqual(remnant.packed_profile().default_count, 256)
                    self.assertEqual(
                        context.config_bag("model").dsv4_c4_cache_keep_count, 256
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
                    "--kv-cache-format",
                    "direct",
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
        args = ServerArgs(
            model_path="dummy", kv_cache_format="direct", enable_hisparse=True
        )
        with mock.patch(
            "sglang.srt.server_args.model_config_of",
            return_value=SimpleNamespace(
                hf_config=SimpleNamespace(
                    model_type="deepseek_v4", compress_ratios=(0, 128, 4)
                )
            ),
        ):
            with self.assertRaisesRegex(ValueError, "--enable-hisparse"):
                args._configure_cache_format()


if __name__ == "__main__":
    unittest.main()
