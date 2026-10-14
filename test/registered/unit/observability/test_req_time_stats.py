"""Unit tests for ReqTimeStats IPC serialization.

ReqTimeStatsBase.__setstate__ rebases perf_counter fields onto the receiving
process's clock anchor. Rebasing a field that was never stamped (0.0) turns
the sentinel into a tiny epsilon (sender_diff - receiver_diff), which defeats
== 0.0 / > 0.0 "was this stamped?" checks downstream. Concretely, a PD decode
server never stamps prefill_finished_time locally; if the sentinel arrives at
the tokenizer as an epsilon, first-token bookkeeping mistakes it for a real
stamp and the TTFT / inter-token-latency histograms record ~node-uptime-sized
garbage samples.

Remnant first-output metadata tests: Author: Winston Cai.
"""

import pickle
import unittest
from unittest import mock

import sglang.srt.observability.req_time_stats as rts
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestSetstatePreservesUnsetTimeSentinels(CustomTestCase):
    def test_two_hop_round_trip(self):
        src = rts.SchedulerReqTimeStats()
        src.enable_metrics = True
        src.wait_queue_entry_time = 123.456
        src.prefill_finished_time = 0.0

        with mock.patch.object(rts, "global_diff_realtime_monotonic", 1_000_000.0):
            blob = pickle.dumps(src)
        with mock.patch.object(rts, "global_diff_realtime_monotonic", 1_000_005.0):
            hop1 = pickle.loads(blob)
            blob2 = pickle.dumps(hop1)
        with mock.patch.object(rts, "global_diff_realtime_monotonic", 1_000_009.0):
            hop2 = pickle.loads(blob2)

        self.assertEqual(hop2.prefill_finished_time, 0.0)
        self.assertAlmostEqual(hop2.wait_queue_entry_time, 123.456 - 9.0)


class TestFirstOutputMetadata(CustomTestCase):
    def test_first_output_uses_existing_stamp_for_single_token(self):
        stats = rts.APIServerReqTimeStats()
        stats.created_time = 100.0
        stats.first_token_time = 105.0
        stats.finished_time = 107.0
        with mock.patch.object(rts, "global_diff_realtime_monotonic", 1_000_000.0):
            metadata = stats.convert_to_output_meta_info(completion_tokens=1)
        self.assertEqual(metadata["first_output_ts"], 1_000_105.0)
        self.assertEqual(metadata["request_finished_ts"], 1_000_107.0)
        self.assertNotIn("decode_throughput", metadata)
        self.assertEqual(stats.first_token_time, 105.0)

    def test_unset_first_output_is_absent(self):
        stats = rts.APIServerReqTimeStats()
        stats.finished_time = 107.0
        self.assertNotIn("first_output_ts", stats.convert_to_output_meta_info())

    def test_rebased_scheduler_stamps_share_export_clock(self):
        scheduler = rts.SchedulerReqTimeStats()
        scheduler.enable_metrics = True
        scheduler.wait_queue_entry_time = 100.0
        scheduler.forward_entry_time = 102.0
        scheduler.prefill_finished_time = 106.0
        with mock.patch.object(rts, "global_diff_realtime_monotonic", 1_000_000.0):
            blob = pickle.dumps(scheduler)
        with mock.patch.object(rts, "global_diff_realtime_monotonic", 1_000_009.0):
            received = pickle.loads(blob)
            api = rts.APIServerReqTimeStats()
            api.first_token_time = 98.0  # Original clock 107.0 after rebasing.
            metadata = received.convert_to_output_meta_info()
            metadata.update(api.convert_to_output_meta_info())
        self.assertEqual(metadata["queue_time"], 2.0)
        self.assertEqual(metadata["forward_entry_time"], 1_000_102.0)
        self.assertEqual(metadata["prefill_finished_time"], 1_000_106.0)
        self.assertEqual(metadata["first_output_ts"], 1_000_107.0)


if __name__ == "__main__":
    unittest.main()
