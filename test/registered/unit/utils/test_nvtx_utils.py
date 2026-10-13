"""CPU contracts for gated batch annotations. Author: Winston Cai.

Run: python3 test/registered/unit/utils/test_nvtx_utils.py
Loads the helper directly so these tests need neither CUDA nor the full serving
dependency stack. Only its environment gates are stubbed; CPU tensors are real.
"""

import importlib.util
import json
import sys
import unittest
from contextlib import contextmanager, nullcontext
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

import torch


def load_helper():
    environ = ModuleType("sglang.srt.environ")
    disabled = SimpleNamespace(get=lambda: False)
    environ.envs = SimpleNamespace(
        SGLANG_ENABLE_NVTX_SCHEDULER=disabled,
        SGLANG_ENABLE_NVTX_OPERATIONS=disabled,
    )
    source = (
        Path(__file__).resolve().parents[4]
        / "python/sglang/srt/utils/nvtx_utils.py"
    )
    spec = importlib.util.spec_from_file_location("batch_nvtx_test_helper", source)
    module = importlib.util.module_from_spec(spec)
    with patch.dict(sys.modules, {"sglang.srt.environ": environ}):
        spec.loader.exec_module(module)
    return module


helper = load_helper()


class TestBatchAnnotations(unittest.TestCase):
    def setUp(self):
        self.gate = patch.object(helper, "NVTX_SCHEDULER_ENABLED", True)
        self.gate.start()
        self.addCleanup(self.gate.stop)

    def metadata(self, batch, **kwargs):
        with patch.object(helper, "profile_range", return_value=nullcontext()) as emit:
            with helper.batch_profile_range("batch", batch, **kwargs):
                pass
        name = emit.call_args.args[0]
        self.assertTrue(emit.call_args.kwargs["nvtx_enabled"])
        return json.loads(name.split(" ", 1)[1])

    def test_decode_and_exact_padding(self):
        batch = SimpleNamespace(
            batch_size=2,
            forward_mode=SimpleNamespace(name="DECODE"),
            seq_lens_cpu=torch.tensor([100, 200]),
            extend_num_tokens=999,  # Stale prefill state must not count.
        )
        data = self.metadata(batch, padded_bs=4)
        self.assertEqual(data["requests"], 2)
        self.assertEqual(data["padded_bs"], 4)
        self.assertEqual(data["decode_requests"], 2)
        self.assertEqual(data["prefill_tokens"], 0)
        self.assertEqual(data["context_sum"], 300)
        self.assertEqual(data["context_min"], 100)
        self.assertEqual(data["context_max"], 200)

    def test_prefill_and_mixed_tokens(self):
        batch = SimpleNamespace(
            batch_size=lambda: 3,
            forward_mode=SimpleNamespace(name="MIXED"),
            decoding_reqs=[object()],
            extend_num_tokens=9,
        )
        data = self.metadata(batch)
        self.assertEqual((data["decode_requests"], data["prefill_requests"]), (1, 2))
        self.assertEqual(data["prefill_tokens"], 8)
        batch.forward_mode.name = "EXTEND"
        batch.decoding_reqs = None
        self.assertEqual(self.metadata(batch)["prefill_tokens"], 9)

    def test_missing_mixed_metadata_is_unknown(self):
        batch = SimpleNamespace(
            batch_size=3, forward_mode=SimpleNamespace(name="MIXED")
        )
        self.assertIsNone(self.metadata(batch)["decode_requests"])
        self.assertIsNone(self.metadata(batch)["prefill_tokens"])

    def test_never_reads_device_tensor(self):
        class DeviceTensor:
            device = SimpleNamespace(type="cuda")

            def tolist(self):
                raise AssertionError("device metadata read")

        batch = SimpleNamespace(
            batch_size=1,
            forward_mode=SimpleNamespace(name="DECODE"),
            seq_lens_cpu=DeviceTensor(),
        )
        self.assertEqual(self.metadata(batch)["context_source"], "unavailable")

    def test_empty_batch(self):
        batch = SimpleNamespace(
            batch_size=0,
            forward_mode=SimpleNamespace(name="DECODE"),
            seq_lens_cpu=torch.tensor([], dtype=torch.int64),
        )
        data = self.metadata(batch)
        self.assertIsNone(data["context_min"])
        self.assertEqual(data["context_sum"], 0)

    def test_disabled_path_does_not_inspect_batch(self):
        with patch.object(helper, "NVTX_SCHEDULER_ENABLED", False):
            with helper.batch_profile_range("batch", object()):
                pass
            wrapped = helper.scheduler_batch_nvtx_method(lambda self, batch: 7)
            self.assertEqual(wrapped(object(), object()), 7)

    def test_scheduler_step_and_exception_cleanup(self):
        events = []

        @contextmanager
        def record(name, batch, **kwargs):
            events.append(kwargs)
            try:
                yield
            finally:
                events.append("closed")

        @helper.scheduler_batch_nvtx_method
        def run(scheduler, batch):
            raise RuntimeError("forward failed")

        scheduler = SimpleNamespace(forward_ct=10, ps=SimpleNamespace(tp_rank=2))
        with patch.object(helper, "batch_profile_range", record):
            with self.assertRaisesRegex(RuntimeError, "forward failed"):
                run(scheduler, object())
        self.assertEqual(events, [{"step": 11, "rank": 2}, "closed"])


if __name__ == "__main__":
    unittest.main()
