"""Tests for the public DeepSeek V4 C4 cache-format switch."""

import argparse

from sglang.srt.server_args import ServerArgs
from sglang.test.test_utils import CustomTestCase


class TestDSV4C4CacheFormat(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        cls.parser = argparse.ArgumentParser()
        ServerArgs.add_cli_args(cls.parser)

    def _parse(self, values):
        args = self.parser.parse_args(["--model", "dummy", *values])
        return ServerArgs.from_cli_args(args)

    def test_native_is_the_default(self):
        self.assertEqual(self._parse([]).dsv4_c4_cache_format, "native")

    def test_remnant_is_explicit(self):
        self.assertEqual(
            self._parse(["--dsv4-c4-cache-format", "remnant"])
            .dsv4_c4_cache_format,
            "remnant",
        )

    def test_unknown_format_is_rejected(self):
        with self.assertRaises(SystemExit):
            self.parser.parse_args(
                ["--model", "dummy", "--dsv4-c4-cache-format", "packed"]
            )
