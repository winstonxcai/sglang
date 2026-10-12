"""Tests for the public DeepSeek V4 C4 cache-format switch.

Author: Winston Cai.
"""

import argparse
import multiprocessing
from multiprocessing.connection import Connection
from types import SimpleNamespace
from unittest import mock

from sglang.srt import remnant
from sglang.srt.server_args import ServerArgs, set_global_server_args_for_scheduler
from sglang.test.test_utils import CustomTestCase


def _spawned_worker_modes(
    server_args: list[ServerArgs], connection: Connection
) -> None:
    """Exercise the same publication hook used before ModelRunner pool sizing."""

    def state() -> tuple[str, str, bool]:
        return (
            remnant.cache_format(),
            remnant.cache_implementation(),
            remnant.packed_enabled(),
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
        context_patch = mock.patch(
            "sglang.srt.runtime_context._CONTEXT",
            RuntimeContext(ParallelContext()),
        )
        context_patch.start()
        self.addCleanup(context_patch.stop)

    def tearDown(self):
        remnant.configure_cache_format("native")

    @classmethod
    def setUpClass(cls):
        cls.parser = argparse.ArgumentParser()
        ServerArgs.add_cli_args(cls.parser)

    def _parse(self, values):
        args = self.parser.parse_args(["--model", "dummy", *values])
        return ServerArgs.from_cli_args(args)

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

    def _worker_states(self, args: list[ServerArgs]) -> list[tuple[str, str, bool]]:
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
                        ("native", "kernel", False),
                        (cache_format, implementation, packed),
                    ],
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
                ("native", "kernel", False),
                ("remnant", "triton", True),
                ("remnant", "kernel", True),
                ("native", "kernel", False),
            ],
        )

    def test_partial_publication_resets_to_native_defaults(self):
        remnant.configure_cache_format("remnant", "triton")
        set_global_server_args_for_scheduler(SimpleNamespace())
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
                SimpleNamespace(
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
        self.assertFalse(remnant.topmag_enabled())

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

    def test_existing_callers_default_to_kernel(self):
        remnant.configure_cache_format("remnant", "triton")
        remnant.configure_cache_format("remnant")
        self.assertEqual(remnant.cache_implementation(), "kernel")

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
                enable_prefill_context_parallel=False,
                enable_dsa_prefill_context_parallel=False,
            )
            with self.subTest(implementation=implementation), mock.patch(
                "sglang.srt.configs.model_config.is_deepseek_v4", return_value=True
            ):
                with self.assertRaisesRegex(ValueError, "--enable-hisparse"):
                    ServerArgs._configure_dsv4_c4_cache_format(args)
