"""Tests for the public DeepSeek V4 C4 cache-format switch.

Author: Winston Cai.
"""

import argparse
from types import SimpleNamespace
from unittest import mock

from sglang.srt import remnant
from sglang.srt.server_args import ServerArgs
from sglang.test.test_utils import CustomTestCase


class TestDSV4C4CacheFormat(CustomTestCase):
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
