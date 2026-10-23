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

    def test_preparation_timestamps_use_api_clock_and_remain_opt_in(self):
        stats = rts.APIServerReqTimeStats(
            input_preparation_times=(100.0, 102.0, 103.0, 104.0)
        )
        with (
            mock.patch.object(rts, "SGLANG_FRONTEND_DIAGNOSTICS", True),
            mock.patch.object(rts, "global_diff_realtime_monotonic", 1_000_000.0),
        ):
            metadata = stats.convert_to_output_meta_info()
        self.assertEqual(
            metadata,
            {
                "input_preparation_queued_ts": 1_000_100.0,
                "input_preparation_start_ts": 1_000_102.0,
                "input_preparation_finish_ts": 1_000_103.0,
                "input_preparation_resume_ts": 1_000_104.0,
            },
        )
        with mock.patch.object(rts, "SGLANG_FRONTEND_DIAGNOSTICS", False):
            self.assertEqual(stats.convert_to_output_meta_info(), {})
        with mock.patch.object(rts, "SGLANG_FRONTEND_DIAGNOSTICS", True):
            self.assertEqual(
                rts.APIServerReqTimeStats().convert_to_output_meta_info(), {}
            )

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


class TestFrontendFirstOutputDiagnostics(CustomTestCase):
    def test_disabled_preserves_serialization_and_metadata(self):
        with mock.patch.object(rts, "SGLANG_FRONTEND_DIAGNOSTICS", False):
            scheduler = rts.SchedulerReqTimeStats(first_output_send_time=100.0)
            self.assertEqual(scheduler.__getstate__(), {})

            scheduler.enable_metrics = True
            state = scheduler.__getstate__()
            self.assertEqual(
                set(state),
                {
                    "has_timing_data",
                    "wait_queue_entry_time",
                    "forward_entry_time",
                    "prefill_finished_time",
                    "diff_realtime_monotonic",
                },
            )
            api = rts.APIServerReqTimeStats(first_token_time=105.0)
            api.record_frontend_first_output(scheduler, 103.0, 104.0)
            self.assertFalse(api._frontend_diagnostics_recorded)
            self.assertEqual(api.frontend_recv_time, 0.0)
            self.assertEqual(
                set(api.convert_to_output_meta_info()), {"first_output_ts"}
            )

    def test_enabled_serializes_without_metrics(self):
        scheduler = rts.SchedulerReqTimeStats(first_output_send_time=100.0)
        with mock.patch.object(rts, "SGLANG_FRONTEND_DIAGNOSTICS", True):
            received = pickle.loads(pickle.dumps(scheduler))
        self.assertFalse(scheduler.enable_metrics)
        self.assertTrue(received.has_timing_data)
        self.assertEqual(received.first_output_send_time, 100.0)
        self.assertEqual(received.detokenizer_recv_time, 0.0)
        self.assertEqual(received.detokenizer_send_time, 0.0)

    def test_two_hops_rebase_stamped_fields_and_preserve_zero(self):
        scheduler = rts.SchedulerReqTimeStats(first_output_send_time=101.0)
        with mock.patch.object(rts, "SGLANG_FRONTEND_DIAGNOSTICS", True):
            with mock.patch.object(rts, "global_diff_realtime_monotonic", 1_000_000.0):
                blob = pickle.dumps(scheduler)
            with mock.patch.object(rts, "global_diff_realtime_monotonic", 1_000_005.0):
                detokenizer = pickle.loads(blob)
                self.assertEqual(detokenizer.first_output_send_time, 96.0)
                self.assertEqual(detokenizer.detokenizer_recv_time, 0.0)
                self.assertEqual(detokenizer.detokenizer_send_time, 0.0)
                detokenizer.detokenizer_recv_time = 98.0
                # Keep send unset to exercise its sentinel through the second hop.
                blob = pickle.dumps(detokenizer)
            with mock.patch.object(rts, "global_diff_realtime_monotonic", 1_000_009.0):
                frontend = pickle.loads(blob)
                self.assertEqual(frontend.first_output_send_time, 92.0)
                self.assertEqual(frontend.detokenizer_recv_time, 94.0)
                self.assertEqual(frontend.detokenizer_send_time, 0.0)
                api = rts.APIServerReqTimeStats(first_token_time=98.0)
                api.record_frontend_first_output(frontend, 96.0, 97.0)
                metadata = api.convert_to_output_meta_info()
            self.assertEqual(metadata["first_output_send_ts"], 1_000_101.0)
            self.assertEqual(metadata["detokenizer_recv_ts"], 1_000_103.0)
            self.assertNotIn("detokenizer_send_ts", metadata)
            self.assertEqual(metadata["frontend_recv_ts"], 1_000_105.0)
            self.assertEqual(metadata["frontend_handle_ts"], 1_000_106.0)
            self.assertEqual(metadata["first_output_ts"], 1_000_107.0)

    def test_first_packet_is_latched_independently_of_later_snapshots(self):
        scheduler = rts.SchedulerReqTimeStats(
            first_output_send_time=101.0,
            detokenizer_recv_time=102.0,
            detokenizer_send_time=103.0,
        )
        api = rts.APIServerReqTimeStats(first_token_time=106.0)
        with mock.patch.object(rts, "SGLANG_FRONTEND_DIAGNOSTICS", True):
            api.record_frontend_first_output(scheduler, 104.0, 105.0)
            scheduler.detokenizer_recv_time = 999.0
            later = rts.SchedulerReqTimeStats(first_output_send_time=201.0)
            api.record_frontend_first_output(later, 204.0, 205.0)
            with mock.patch.object(rts, "global_diff_realtime_monotonic", 1_000_000.0):
                metadata = api.convert_to_output_meta_info(later)
        self.assertEqual(metadata["first_output_send_ts"], 1_000_101.0)
        self.assertEqual(metadata["detokenizer_recv_ts"], 1_000_102.0)
        self.assertEqual(metadata["detokenizer_send_ts"], 1_000_103.0)
        self.assertEqual(metadata["frontend_recv_ts"], 1_000_104.0)
        self.assertEqual(metadata["frontend_handle_ts"], 1_000_105.0)
        self.assertEqual(metadata["first_output_ts"], 1_000_106.0)
        self.assertEqual(api.first_token_time, 106.0)

    def test_missing_first_packet_timing_is_not_filled_from_later_packet(self):
        api = rts.APIServerReqTimeStats()
        later = rts.SchedulerReqTimeStats(
            first_output_send_time=201.0,
            detokenizer_recv_time=202.0,
            detokenizer_send_time=203.0,
        )
        with mock.patch.object(rts, "SGLANG_FRONTEND_DIAGNOSTICS", True):
            api.record_frontend_first_output(None, 104.0, 105.0)
            api.record_frontend_first_output(later, 204.0, 205.0)
            metadata = api.convert_to_output_meta_info(later)
        self.assertEqual(set(metadata), {"frontend_recv_ts", "frontend_handle_ts"})
        self.assertEqual(api.first_output_send_time, 0.0)

    def test_enabled_unset_fields_are_absent_from_output(self):
        with mock.patch.object(rts, "SGLANG_FRONTEND_DIAGNOSTICS", True):
            self.assertEqual(
                rts.APIServerReqTimeStats().convert_to_output_meta_info(), {}
            )


if __name__ == "__main__":
    unittest.main()
