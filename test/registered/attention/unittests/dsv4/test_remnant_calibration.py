"""CPU math/ownership gates plus model-free CUDA packing checks.

Author: Winston Cai.
"""

import importlib
import importlib.util
import json
import math as python_math
import tempfile
import unittest
import copy
import time
import threading
import tarfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch

ROOT = Path(__file__).resolve().parents[5]
SPEC = importlib.util.spec_from_file_location(
    "calibration_under_test", ROOT / "python/sglang/benchmark/remnant/calibrate.py"
)
math = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(math)
ref = math.reference()
capture = importlib.import_module(ref.__package__ + ".capture")
packed = importlib.import_module(ref.__package__ + ".packed")


class Plan:
    def __init__(self, words, decode=False):
        self.is_decode = decode
        self.plan_c = torch.tensor(words, dtype=torch.int32).view(torch.uint8)
        self.plan_w = torch.tensor([[7, 91]], dtype=torch.int32).view(torch.uint8)

    def __getitem__(self, index):
        return self.plan_c if index == 1 else self.plan_w


class CalibrationTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        self.c = torch.randn(4, 512, dtype=torch.float64) * 0.2
        self.weight = torch.rand(512, dtype=torch.float64) + 0.5
        angles = torch.randn(4, 32, dtype=torch.float64)
        self.freq = torch.polar(torch.ones_like(angles), angles)
        self.q = torch.randn(2, 512, dtype=torch.float64) * 0.1
        self.sink = torch.tensor([0.7, -2.0], dtype=torch.float64)

    def test_fresh_hundred_issue_contract_and_no_training_leak(self):
        items = [
            {
                "instance_id": f"issue-{i}",
                "repo": f"repository-{i}",
                "split": "held_out",
            }
            for i in range(100)
        ]
        manifest = {
            "status": "complete",
            "requested_count": 100,
            "held_out_only": True,
            "selected": items,
        }
        math.cohort_contract(items, manifest)
        with self.assertRaisesRegex(ValueError, "complete frozen"):
            math.cohort_contract(items[:-1], manifest)
        items[0]["split"] = "calibration"
        with self.assertRaisesRegex(ValueError, "remain held-out"):
            math.cohort_contract(items, manifest)
        items[0]["split"] = "held_out"
        items[0]["instance_id"] = "../escape"
        with self.assertRaisesRegex(ValueError, "unsafe"):
            math.cohort_contract(items, manifest)

    def checkpoint_fixture(self, root, rid="issue"):
        config = {
            "layers": [2, 4],
            "concurrency": 10,
            "requests": {
                rid: {"prompt_tokens": 128000, "quota": 10, "split": "held_out"}
            },
        }
        math.atomic_json(root / "capture-config.json", config)
        for rank in range(4):
            base = root / f"rank-{rank}" / rid
            math.atomic_json(base / "reservoir.json", {"0": 7})
            for layer in config["layers"]:
                torch.save({"ordinal": 7}, base / f"layer-{layer:02d}-slot-00.pt")
        result = {
            "rid": rid,
            "response_id": rid,
            "finish_reason": "tool_calls",
            "usage": {"prompt_tokens": 128000, "completion_tokens": 8},
            "wall_seconds": 1.0,
        }
        return config, result

    def test_completed_checkpoint_hashes_all_ranks_and_layers(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config, result = self.checkpoint_fixture(root)
            self.assertIsNone(math.completed_issue(root, config, "issue"))
            receipt = math.checkpoint_issue(root, config, result)
            self.assertEqual(len(receipt["files_sha256"]), 12)
            self.assertEqual(math.completed_issue(root, config, "issue"), receipt)
            payload = root / "rank-3/issue/layer-04-slot-00.pt"
            payload.write_bytes(payload.read_bytes() + b"modified")
            with self.assertRaisesRegex(ValueError, "payload changed"):
                math.completed_issue(root, config, "issue")

    def test_incomplete_generation_and_partial_rank_never_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config, result = self.checkpoint_fixture(root)
            with self.assertRaisesRegex(ValueError, "natural"):
                math.checkpoint_issue(
                    root, config, {**result, "finish_reason": "length"}
                )
            (root / "rank-3/issue/layer-04-slot-00.pt").unlink()
            with self.assertRaises(FileNotFoundError):
                math.checkpoint_issue(root, config, result)
            self.assertFalse((root / "done/issue.json").exists())

    def test_resume_checkpoint_rejects_different_frozen_capture(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config, result = self.checkpoint_fixture(root)
            math.checkpoint_issue(root, config, result)
            math.atomic_json(
                root / "capture-config.json", {**config, "different": True}
            )
            with self.assertRaisesRegex(ValueError, "another frozen capture"):
                math.completed_issue(root, config, "issue")

    def test_client_resume_skips_already_completed_issue(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config, result = self.checkpoint_fixture(root)
            math.checkpoint_issue(root, config, result)
            (root / "replay.jsonl").write_text(
                json.dumps({"rid": "issue", "body": {}}) + "\n"
            )
            fake_requests = SimpleNamespace(
                post=lambda *a, **k: self.fail("resubmitted completed issue")
            )
            with patch.dict("sys.modules", {"requests": fake_requests}):
                math.capture_client(root, "http://unused", time.monotonic() + 10)
            progress = json.loads((root / "capture-progress.json").read_text())
            self.assertEqual(progress["status"], "complete")
            self.assertEqual(progress["completed"], 1)

    def test_client_replenishes_before_slow_request_finishes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            ids = ["pilot-a", "pilot-b", "slow", "fast", "replacement"]
            config = {
                "layers": [2],
                "concurrency": 2,
                "requests": {
                    rid: {"split": "held_out", "prompt_tokens": 128000} for rid in ids
                },
            }
            math.atomic_json(root / "capture-config.json", config)
            (root / "replay.jsonl").write_text(
                "\n".join(json.dumps({"rid": rid, "body": {}}) for rid in ids)
            )
            replacement_started = threading.Event()
            checkpointed = []

            class Response:
                def __init__(self, rid):
                    self.rid = rid

                def __enter__(self):
                    return self

                def __exit__(self, *args):
                    pass

                def raise_for_status(self):
                    pass

                def iter_lines(self):
                    if self.rid == "slow" and not replacement_started.wait(3):
                        raise RuntimeError("queue waited for a slow wave tail")
                    if self.rid == "replacement":
                        if "fast" not in checkpointed:
                            raise RuntimeError(
                                "replacement admitted before predecessor checkpoint"
                            )
                        replacement_started.set()
                    yield b"data: " + json.dumps(
                        {
                            "id": self.rid,
                            "usage": {"prompt_tokens": 128000, "completion_tokens": 1},
                            "choices": [{"finish_reason": "stop"}],
                        }
                    ).encode()

            def checkpoint(root, config, result):
                checkpointed.append(result["rid"])
                math.atomic_json(root / "done" / (result["rid"] + ".json"), result)
                return result

            requests = SimpleNamespace(
                post=lambda *a, **k: Response(k["headers"]["x-override-rid"])
            )
            with patch.dict("sys.modules", {"requests": requests}), patch.object(
                math, "state_paths", return_value=[[Path("unused")]]
            ), patch.object(math, "load_state", return_value={}), patch.object(
                math, "validate_state"
            ), patch.object(
                math, "checkpoint_issue", side_effect=checkpoint
            ):
                math.capture_client(root, "http://unused", time.monotonic() + 10)
            self.assertTrue(replacement_started.is_set())
            self.assertLess(checkpointed.index("fast"), checkpointed.index("slow"))
            self.assertEqual(len(list((root / "done").glob("*.json"))), 5)

    def test_snapshot_cache_preserves_owned_tensors_and_memory_bound(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "state.pt"
            torch.save({"x": self.c}, path)
            try:
                total = math.cache_snapshots([path], time.monotonic() + 10)
                self.assertEqual(total, path.stat().st_size)
                first = math.read_snapshot(path)
                torch.testing.assert_close(first["x"], self.c, atol=0, rtol=0)
                first["x"].zero_()
                torch.testing.assert_close(
                    math.read_snapshot(path)["x"], self.c, atol=0, rtol=0
                )
                with self.assertRaisesRegex(ValueError, "memory budget"):
                    math.cache_snapshots([path], time.monotonic() + 10, max_bytes=1)
            finally:
                math._snapshot_bytes.clear()

    def test_sensitivity_checkpoint_validates_capture_and_coverage(self):
        data = {
            "capture_config_sha256": "frozen",
            "weights": [1.0] * 512,
            "floor": 1e-6,
            "output_floors": {"continuous": 1e-6, "packed": 1e-6},
            "empirical": {"issue": {"r": [2.0] * 512, "row_instances": 16}},
            "diagnostics": {},
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "checkpoint.json"
            path.write_text(json.dumps(data))
            restored, h = math.restore_sensitivity(path, "frozen", {"issue": 2})
            self.assertEqual(restored, data)
            torch.testing.assert_close(h, torch.ones(512), atol=0, rtol=0)
            with self.assertRaisesRegex(ValueError, "another capture"):
                math.restore_sensitivity(path, "different", {"issue": 2})
            with self.assertRaisesRegex(ValueError, "coverage changed"):
                math.restore_sensitivity(path, "frozen", {"issue": 3})
            data["weights"][0] = float("nan")
            path.write_text(json.dumps(data))
            with self.assertRaisesRegex(ValueError, "invalid checkpoint weights"):
                math.restore_sensitivity(path, "frozen", {"issue": 2})

    def test_archive_is_complete_only_bounded_and_excludes_mutable_files(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "run"
            root.mkdir()
            (root / "allocation.json").write_text(json.dumps({"status": "running"}))
            with self.assertRaisesRegex(ValueError, "incomplete"):
                math.archive_capture(root)
            (root / "allocation.json").write_text(json.dumps({"status": "complete"}))
            (root / "state.pt").write_bytes(b"immutable capture")
            (root / "link").symlink_to(root / "state.pt")
            (root / "offline-allocation.json").write_text("mutable")
            for name in ("analysis", "jit-cache", "offline-attempts"):
                (root / name).mkdir()
                (root / name / "ignored").write_text("not capture data")
            archive = math.archive_capture(root)
            with tarfile.open(archive) as bundle:
                self.assertEqual(
                    set(bundle.getnames()), {"run/allocation.json", "run/state.pt"}
                )
                self.assertEqual(
                    bundle.extractfile("run/state.pt").read(), b"immutable capture"
                )
            self.assertEqual(math.archive_capture(root), archive)

    def test_replay_body_keeps_frozen_payload_and_explicit_id(self):
        item = {
            "instance_id": "frozen-issue",
            "request": {
                "messages": [{"role": "user", "content": "x"}],
                "tools": [{"type": "function", "function": {"name": "t"}}],
                "max_tokens": 128,
                "max_completion_tokens": 128,
                "seed": 42,
            },
        }
        original = copy.deepcopy(item)
        body = math.replay_body(item)
        self.assertEqual(body["rid"], "frozen-issue")
        self.assertNotIn("max_tokens", body)
        self.assertNotIn("max_completion_tokens", body)
        self.assertEqual(body["messages"], item["request"]["messages"])
        self.assertEqual(body["tools"], item["request"]["tools"])
        body["messages"][0]["content"] = "changed"
        self.assertEqual(item, original)

    def test_render_uses_protocol_defaults_without_mutating_request(self):
        body = {
            "messages": [
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [{"function": {"arguments": '{"x": 1}'}}],
                }
            ],
            "tools": [{"function": {"name": "test"}}],
        }
        original = copy.deepcopy(body)
        dumped_tool = {
            "type": "function",
            "function": {
                "description": None,
                "name": "test",
                "parameters": None,
                "strict": False,
            },
            "defer_loading": None,
        }

        def request_type(**kwargs):
            return SimpleNamespace(
                messages=[
                    SimpleNamespace(
                        model_dump=lambda: copy.deepcopy(body["messages"][0])
                    )
                ],
                tools=[SimpleNamespace(model_dump=lambda: copy.deepcopy(dumped_tool))],
                chat_template_kwargs={"thinking": False},
                continue_final_message=False,
                task=None,
                reasoning_effort="low",
            )

        encoder = SimpleNamespace(
            encode_messages=lambda messages, **kwargs: (messages, kwargs)
        )
        messages, options = math.render_serving_request(body, encoder, request_type)
        self.assertEqual(messages[0]["tools"], [dumped_tool])
        self.assertEqual(messages[1]["content"], "")
        self.assertEqual(
            messages[1]["tool_calls"][0]["function"]["arguments"], {"x": 1}
        )
        self.assertEqual(options["thinking_mode"], "chat")
        self.assertEqual(body, original)

    def test_native_prefill_plan_mapping(self):
        plan = Plan([[4, 3, 99, 101], [8, 7, 88, 90], [-1, 65535, -1, -1]])
        loc = torch.arange(8) + 20
        self.assertEqual(capture.plan_writes(plan, loc), [(0, 3, 23, 0), (1, 7, 27, 4)])
        self.assertEqual(packed._plan_rows(plan).shape, (3, 4))
        self.assertNotEqual(plan.plan_c.shape[0], plan.plan_w.shape[0])

    def test_decode_boundaries(self):
        plan = Plan([[4, 99, 0, 0], [5, 0, 0, 0], [8, 0, 0, 0]], True)
        self.assertEqual(
            capture.plan_writes(plan, torch.tensor([17, 18, -1])), [(0, 0, 17, 0)]
        )

    def test_sink_empty_and_partial(self):
        out, p, lse = math.attention(self.q, self.c[:0], self.sink, 0.5)
        self.assertEqual(p.shape, (2, 0))
        self.assertTrue(torch.equal(out, torch.zeros_like(out)))
        self.assertTrue(torch.isneginf(lse).all())
        out, p, _ = math.attention(
            self.q, self.c[:2], torch.full_like(self.sink, 100), 0.5
        )
        self.assertLess(float(out.abs().max()), 1e-30)
        self.assertTrue((p.sum(1) < 1).all())

    def test_rope_inverse_and_fp32_fp64(self):
        torch.testing.assert_close(
            math.rotate(math.rotate(self.c, self.freq), self.freq, True), self.c
        )
        rows = math.normalize(self.c, self.weight, 1e-6, self.freq)
        expected = math.attention(self.q, rows, self.sink, 0.5)[0]
        actual = math.attention(self.q.float(), rows.float(), self.sink.float(), 0.5)[0]
        torch.testing.assert_close(actual.double(), expected, atol=1e-6, rtol=1e-5)

    def test_row_update_duplicates(self):
        rows = math.normalize(self.c, self.weight, 1e-6, self.freq)
        duplicate = torch.cat((rows, rows[:1]))
        baseline, p, _ = math.attention(self.q, duplicate, self.sink, 0.5)
        changed = self.c[0].clone()
        changed[11] = 0
        new = math.normalize(changed, self.weight, 1e-6, self.freq[0])
        actual = math.row_update(
            baseline, self.q, rows[0], new[None], p[:, 0] + p[:, -1], 0.5
        )[0]
        duplicate[0] = duplicate[-1] = new
        expected = math.attention(self.q, duplicate, self.sink, 0.5)[0]
        torch.testing.assert_close(actual, expected, atol=1e-10, rtol=1e-9)

    def test_analytic_against_finite_difference(self):
        rows = math.normalize(self.c, self.weight, 1e-6, self.freq)
        output, p, _ = math.attention(self.q, rows, self.sink, 0.5)
        diagonal = math.analytic_diagonal(
            self.c[0], self.weight, 1e-6, self.freq[0], self.q, output, p[:, 0], 0.5
        )
        for index in (0, 17, 447, 448, 511):
            for step in (1e-4, 1e-5, 1e-6):
                self._check_difference(rows, diagonal, index, step)

    def _check_difference(self, rows, diagonal, index, step):
        perturb = torch.zeros(512, dtype=torch.float64)
        perturb[index] = step
        variants = []
        for sign in (-1, 1):
            changed = rows.clone()
            changed[0] = math.normalize(
                self.c[0] + sign * perturb, self.weight, 1e-6, self.freq[0]
            )
            variants.append(math.attention(self.q, changed, self.sink, 0.5)[0])
        derivative = (variants[1] - variants[0]) / (2 * step)
        torch.testing.assert_close(
            diagonal[index], derivative.square().sum(), atol=1e-8, rtol=1e-5
        )

    def test_packing_all_ones_weights(self):
        c = self.c.float()
        c[0, :100] = 0
        mask = ref.topmag_keep_mask(c, 0.5)
        torch.testing.assert_close(
            mask, ref.topmag_keep_mask(c.abs() * torch.ones(512), 0.5)
        )
        codes, bitmap, scales = ref.pack_rows_ref(c, mask, self.weight.float(), 1e-6)
        self.assertEqual(codes.shape, (4, 256))
        reconstructed = ref.unpack_rows_ref(codes, bitmap, scales)
        self.assertTrue(
            torch.equal(reconstructed[~mask], torch.zeros_like(reconstructed[~mask]))
        )

    def test_reservoir_layer_and_rank_consistency(self):
        with tempfile.TemporaryDirectory() as directory:
            config = {
                "layers": [2, 22],
                "heads_per_rank": 16,
                "seed": 42,
                "requests": {"a": {"quota": 3}, "b": {"quota": 2}},
            }
            Path(directory, "capture-config.json").write_text(json.dumps(config))
            obs = capture.Observer(directory)
            for ordinal in range(100):
                chosen = obs.consider("a", ordinal)
                self.assertEqual(chosen, obs.consider("a", ordinal))
                obs.consider("b", ordinal)
            selected = sorted(x[1] for x in obs.banks["a"].values())
            expected = sorted(
                sorted(range(100), key=lambda n: capture.sample_priority("a", n))[:3]
            )
            self.assertEqual(selected, expected)
            self.assertEqual(len(obs.banks["b"]), 2)

    def test_native_page_layout(self):
        page_size = 16
        pool = (
            torch.arange(2 * 584 * page_size, dtype=torch.int64)
            .remainder(256)
            .to(torch.uint8)
            .reshape(2, -1)
        )
        records = capture.native_records(pool, torch.tensor([15, 16, 15]), page_size)
        torch.testing.assert_close(records[0, :576], pool[0, 15 * 576 : 16 * 576])
        torch.testing.assert_close(records[1, 576:], pool[1, 16 * 576 : 16 * 576 + 8])
        torch.testing.assert_close(records[0], records[2])

    def test_disabled_capture(self):
        if not capture.enabled():
            self.assertIsNone(capture.observer())

    def test_startup_dummy_and_untracked_batches_do_not_capture(self):
        with tempfile.TemporaryDirectory() as directory:
            Path(directory, "capture-config.json").write_text(
                json.dumps({"layers": [2], "requests": {"a": {}}})
            )
            obs = capture.Observer(directory)
            for rids in (None, [], ["health-check"]):
                batch = SimpleNamespace(rids=rids, sampling_info=None)
                # None operands expose any accidental tensor access/copy.
                obs.observe_raw(2, batch, None, None, None, None, None, None)
                obs.observe_attention(2, batch, *([None] * 14))
                obs.observe_output(2, batch, None, None, None, None, None)
            obs.flush()
            self.assertEqual(obs.generation, 0)
            for name in ("shadow", "layer_data", "banks", "pending"):
                self.assertEqual(getattr(obs, name), {})
            self.assertEqual(list(Path(directory).rglob("*.pt")), [])

    def test_real_batch_without_ids_fails_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            Path(directory, "capture-config.json").write_text(
                json.dumps({"layers": [2], "requests": {"a": {}}})
            )
            obs = capture.Observer(directory)
            batch = SimpleNamespace(rids=None, sampling_info=object())
            with self.assertRaisesRegex(RuntimeError, "request IDs"):
                obs.observe_raw(2, batch, None, None, None, None, None, None)
            with self.assertRaisesRegex(RuntimeError, "request IDs"):
                obs.observe_attention(2, batch, *([None] * 14))
            with self.assertRaisesRegex(RuntimeError, "request IDs"):
                obs.observe_output(2, batch, None, None, None, None, None)
            obs.flush()

    def test_shadow_ownership_reuse_and_aliasing(self):
        with tempfile.TemporaryDirectory() as directory:
            Path(directory, "capture-config.json").write_text(
                json.dumps({"layers": [2], "requests": {"a": {}, "b": {}}})
            )
            obs = capture.Observer(directory)
            batch = SimpleNamespace(
                rids=["a"],
                extend_seq_lens_cpu=[4],
                forward_mode=SimpleNamespace(is_decode=lambda: False),
            )
            raw = self.c[:1].float()
            original = raw.clone()
            plan = Plan([[4, 3, 0, 0]])
            locations = torch.tensor([-1, -1, -1, 17])
            obs.observe_raw(
                2, batch, raw, plan, locations, self.weight, 1e-6, self.freq
            )
            raw.zero_()
            stored = obs.shadow[("a", 2)][17]
            torch.testing.assert_close(stored[2].wait()[stored[3]], original[0])
            batch.rids = ["b"]
            obs.observe_raw(
                2,
                batch,
                raw,
                Plan([[8, 3, 0, 0]]),
                locations,
                self.weight,
                1e-6,
                self.freq,
            )
            self.assertEqual(obs.shadow[("b", 2)][17][0], 4)
            self.assertGreater(obs.shadow[("b", 2)][17][1], stored[1])
            self.assertEqual(
                obs.physical_owners[(2, 17)], ("b", obs.shadow[("b", 2)][17][1])
            )
            self.assertNotEqual(obs.physical_owners[(2, 17)], ("a", stored[1]))
            batch.rids = ["a"]
            batch.positions = torch.tensor([8])
            batch.forward_mode = SimpleNamespace(is_decode=lambda: True)
            obs.config["seed"] = 42
            obs.config["requests"]["a"].update(prompt_tokens=4, quota=1)
            with self.assertRaisesRegex(RuntimeError, "stale raw row"):
                obs.observe_attention(
                    2,
                    batch,
                    None,
                    torch.zeros(2, 16 * 584, dtype=torch.uint8),
                    torch.zeros(2, 16 * 584, dtype=torch.uint8),
                    torch.tensor([[-1]]),
                    torch.tensor([[17]]),
                    torch.tensor([0]),
                    torch.tensor([1]),
                    None,
                    1,
                    None,
                    16,
                    16,
                    None,
                )
            obs.flush()

    def test_zero_magnitude_floor_and_ties(self):
        errors = torch.ones(2, 512)
        magnitude = torch.ones_like(errors)
        magnitude[:, :8] = 0
        result, valid = math.empirical_ratio(errors, magnitude, 1e-6)
        self.assertEqual(int((~valid).sum()), 16)
        self.assertTrue(torch.isfinite(result).all())
        self.assertTrue(torch.equal(result[:8], torch.zeros(8)))
        c = torch.ones(2, 512)
        c[:, :100] = 0
        top = ref.topmag_keep_mask(c, 0.5)
        weighted = ref.topmag_keep_mask(c.abs() * torch.ones(512), 0.5)
        self.assertTrue(torch.equal(top, weighted))
        self.assertTrue(torch.equal(top.sum(-1), torch.full((2,), 256)))

    def test_rank_assembly_rejects_shared_data_mismatch(self):
        with tempfile.TemporaryDirectory() as directory:
            paths = []
            for rank in range(4):
                data = {
                    "rank": rank,
                    "rid": "a",
                    "layer": 2,
                    "ordinal": 3,
                    "query_position": 131075,
                    "norm_eps": 1e-6,
                    "softmax_scale": 0.1,
                }
                for key in (
                    "raw_c",
                    "raw_positions",
                    "csa_bytes",
                    "swa_bytes",
                    "csa_indices",
                    "swa_indices",
                    "csa_valid",
                    "swa_valid",
                    "norm_weight",
                    "csa_freqs",
                    "query_freqs",
                ):
                    data[key] = torch.zeros(1)
                data["csa_indices"] = torch.zeros(1, dtype=torch.int32)
                data["csa_valid"] = torch.ones(1, dtype=torch.bool)
                data["q"] = torch.full((16, 512), rank, dtype=torch.bfloat16)
                paths.append(Path(directory, f"rank-{rank}.pt"))
                torch.save(data, paths[-1])
            self.assertEqual(math.load_state(paths, "cpu")["q"].shape, (64, 512))
            data["raw_c"] = torch.ones(1)
            torch.save(data, paths[-1])
            with self.assertRaisesRegex(ValueError, "shared-data mismatch"):
                math.load_state(paths, "cpu")

    def shared_fixture(self, root):
        paths = []
        ids = torch.tensor([7, 3, 7, -1], dtype=torch.int32)
        for rank in range(4):
            order = torch.tensor([3, 2, 1, 0]) if rank % 2 else torch.arange(4)
            valid = ids != -1
            compact = torch.arange(4)[order]
            compact = compact[valid[compact]]
            values = ids[compact, None].repeat(1, 512).float()
            data = {
                "rank": rank,
                "rid": "issue",
                "layer": 2,
                "ordinal": 3,
                "query_position": 103,
                "norm_eps": 1e-6,
                "softmax_scale": 0.1,
                "csa_indices": ids[order],
                "csa_valid": valid[order],
                "swa_indices": torch.tensor([1]),
                "swa_valid": torch.tensor([True]),
                "raw_c": values,
                "raw_positions": ids[compact].long(),
                "csa_bytes": ids[compact, None].repeat(1, 584).to(torch.uint8),
                "swa_bytes": torch.zeros(1, 584, dtype=torch.uint8),
                "norm_weight": torch.ones(512),
                "csa_freqs": torch.ones(3, 32, dtype=torch.complex64),
                "query_freqs": torch.ones(32, dtype=torch.complex64),
                "q": torch.full((16, 512), rank, dtype=torch.bfloat16),
                "sink": torch.zeros(16),
                "bar_o": torch.zeros(16, 512),
                "o": torch.zeros(16, 512),
                "lse": torch.zeros(16),
                "scheduler": {},
            }
            path = root / f"rank-{rank}" / "issue" / "layer-02-slot-00.pt"
            capture.atomic_tensor(path, data)
            math.atomic_json(path.parent / "reservoir.json", {"0": 3})
            paths.append(path)
        config = {
            "layers": [2],
            "requests": {"issue": {"prompt_tokens": 100, "quota": 1}},
        }
        math.atomic_json(root / "capture-config.json", config)
        return paths, config

    def test_shared_roundtrip_partial_padding_duplicates_and_constants(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths, _ = self.shared_fixture(root / "old")
            converted = math.convert_shared_group(paths, root / "new")
            self.assertEqual(len(list((root / "new/shared").glob("*constants.pt"))), 1)
            state = math.load_state(converted, "cpu")
            self.assertEqual(state["q"].shape, (64, 512))
            self.assertEqual(state["raw_c"].shape, (3, 512))
            self.assertEqual(state["csa_indices"].tolist(), [3, 7, 7, -1])
            self.assertNotIn("raw_c", math._read_payload(converted[1]))
            converted[3].unlink()
            with self.assertRaises(FileNotFoundError):
                math.load_state(converted, "cpu")

    def test_native_validation_uses_compact_rows_once(self):
        # Zero KV gives zero output. Native LSE excludes the sink, so it is
        # log(admitted rows), with FlashMLA's +inf sentinel for no rows.
        for shared in (False, True):
            for csa_count, swa_count in ((3, 2), (0, 2), (3, 0), (0, 0)):
                with self.subTest(
                    shared=shared, csa=csa_count, swa=swa_count
                ), tempfile.TemporaryDirectory() as directory:
                    root = Path(directory)
                    paths, _ = self.shared_fixture(root / "old")
                    for path in paths:
                        data = math.read_snapshot(path)
                        if not csa_count:
                            data["csa_valid"].zero_()
                            data["csa_indices"].fill_(-1)
                        for key in ("raw_c", "raw_positions", "csa_freqs"):
                            data[key] = data[key][:csa_count]
                        data["csa_bytes"] = torch.zeros(
                            csa_count, 584, dtype=torch.uint8
                        )
                        data["swa_indices"] = torch.tensor(
                            [1, -1, 1, 4], dtype=torch.int32
                        )
                        data["swa_valid"] = torch.tensor([True, False, True, False])
                        if not swa_count:
                            data["swa_valid"].zero_()
                        data["swa_bytes"] = torch.zeros(
                            swa_count, 584, dtype=torch.uint8
                        )
                        count = csa_count + swa_count
                        data["lse"] = torch.full(
                            (16,), python_math.log(count) if count else float("inf")
                        )
                        capture.atomic_tensor(path, data)
                    if shared:
                        paths = math.convert_shared_group(paths, root / "new")
                    state = math.load_state(paths, "cpu")
                    result = math.validate_state(state)
                    self.assertEqual(result["max_output_error"], 0)
                    self.assertLessEqual(result["max_lse_error"], 1e-6)
                    if not csa_count and not swa_count:
                        wrong = {
                            **state,
                            "lse": torch.full_like(state["lse"], -float("inf")),
                        }
                        with self.assertRaises(AssertionError):
                            math.validate_state(wrong)
                    for prefix in ("csa", "swa"):
                        wrong = dict(state)
                        wrong[prefix + "_bytes"] = torch.cat(
                            (
                                state[prefix + "_bytes"],
                                torch.zeros(1, 584, dtype=torch.uint8),
                            )
                        )
                        with self.assertRaisesRegex(
                            ValueError, prefix + " compact row count"
                        ):
                            math.validate_state(wrong)

    def test_owner_capture_allows_empty_csa_selection(self):
        for schema in (1, 2):
            with self.subTest(
                schema=schema
            ), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                math.atomic_json(
                    root / "capture-config.json",
                    {
                        "storage_schema_version": schema,
                        "audit_shared": False,
                        "layers": [2],
                        "heads_per_rank": 16,
                        "seed": 42,
                        "requests": {"issue": {"quota": 1, "prompt_tokens": 100}},
                    },
                )
                with patch.object(
                    torch.distributed, "is_initialized", return_value=False
                ):
                    obs = capture.Observer(root)
                self.addCleanup(obs.flush)
                freq = torch.ones(104, 32, dtype=torch.complex64)
                obs.layer_data[2] = (torch.ones(512), 1e-6, freq)
                batch = SimpleNamespace(
                    rids=["issue"],
                    positions=torch.tensor([103]),
                    forward_mode=SimpleNamespace(is_decode=lambda: True),
                )
                q = torch.zeros(1, 16, 512)
                cache = torch.zeros(1, 16 * 584, dtype=torch.uint8)
                obs.observe_attention(
                    2,
                    batch,
                    q,
                    cache,
                    cache,
                    torch.tensor([[0, -1]]),
                    torch.tensor([[-1, -1]]),
                    torch.tensor([1]),
                    torch.tensor([0]),
                    torch.zeros(16),
                    0.1,
                    q,
                    16,
                    16,
                    SimpleNamespace(),
                )
                obs.observe_output(2, batch, q, freq, batch.positions, 0, 16)
                obs.flush()
                record = math.read_snapshot(root / "rank-0/issue/layer-02-slot-00.pt")
                self.assertEqual(record["raw_c"].shape, (0, 512))
                self.assertEqual(record["csa_bytes"].shape, (0, 584))
                self.assertEqual(record["csa_freqs"].shape, (0, 32))
                self.assertEqual(record["raw_positions"].dtype, torch.long)
                self.assertEqual(record["swa_bytes"].shape, (1, 584))

    def test_shared_hash_corruption_and_audit_disagreement_fail(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths, _ = self.shared_fixture(root / "old")
            converted = math.convert_shared_group(paths, root / "new")
            record = math.read_snapshot(paths[1])
            record["raw_c"] = record["raw_c"] + 1
            capture.write_shared_snapshot(
                root / "new", converted[1], record, 0, audit=True
            )
            with self.assertRaisesRegex(ValueError, "audit mismatch"):
                math.read_snapshot(converted[1])
            common = root / "new/shared/issue/layer-02-slot-00.pt"
            common.write_bytes(common.read_bytes() + b"corrupt")
            with self.assertRaisesRegex(ValueError, "hash mismatch"):
                math.read_snapshot(converted[0])

    def test_shared_publication_roundtrip_cleanup_resume_and_corruption(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths, config = self.shared_fixture(root / "old")
            converted = math.convert_shared_group(paths, root / "stage")
            math.atomic_json(root / "stage/capture-config.json", config)
            for rank in range(4):
                math.atomic_json(
                    root / f"stage/rank-{rank}/issue/reservoir.json", {"0": 3}
                )
            result = {
                "rid": "issue",
                "response_id": "issue",
                "finish_reason": "stop",
                "usage": {"prompt_tokens": 100, "completion_tokens": 4},
            }
            math.checkpoint_issue(root / "stage", config, result)
            record = math.publish_issue(
                root / "stage", root / "persistent", "issue", cleanup=True
            )
            self.assertFalse(converted[0].exists())
            math.materialize_issue(root / "persistent", "issue", root / "read")
            self.assertIsNotNone(math.completed_issue(root / "read", config, "issue"))
            math.materialize_issue(
                root / "persistent", "issue", root / "selected", layers=[2]
            )
            self.assertTrue(
                (root / "selected/shared/issue/layer-02-slot-00.pt").exists()
            )
            self.assertEqual(math.published_issue(root / "persistent", "issue"), record)
            archive = root / "persistent" / record["bundle"]
            archive.write_bytes(archive.read_bytes() + b"corrupt")
            with self.assertRaisesRegex(ValueError, "corrupt"):
                math.published_issue(root / "persistent", "issue")

    def test_resume_publishes_completed_local_issue_without_generation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config, result = self.checkpoint_fixture(root / "stage")
            math.checkpoint_issue(root / "stage", config, result)
            (root / "stage/replay.jsonl").write_text(
                json.dumps({"rid": "issue", "body": {}}) + "\n"
            )
            fake_requests = SimpleNamespace(
                post=lambda *a, **k: self.fail("resubmitted completed issue")
            )
            with patch.dict("sys.modules", {"requests": fake_requests}):
                math.capture_client(
                    root / "stage",
                    "http://unused",
                    time.monotonic() + 10,
                    root / "persistent",
                )
            self.assertIsNotNone(math.published_issue(root / "persistent", "issue"))

    def test_selected_layer_extraction_skips_other_layer_payloads(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths, config = self.shared_fixture(root / "old")
            other = []
            for path in paths:
                data = math.read_snapshot(path)
                data["layer"] = 4
                target = path.with_name("layer-04-slot-00.pt")
                capture.atomic_tensor(target, data)
                other.append(target)
            math.convert_shared_group(paths, root / "stage")
            math.convert_shared_group(other, root / "stage")
            config["layers"] = [2, 4]
            math.atomic_json(root / "stage/capture-config.json", config)
            for rank in range(4):
                math.atomic_json(
                    root / f"stage/rank-{rank}/issue/reservoir.json", {"0": 3}
                )
            result = {
                "rid": "issue",
                "response_id": "issue",
                "finish_reason": "stop",
                "usage": {"prompt_tokens": 100},
            }
            math.checkpoint_issue(root / "stage", config, result)
            math.publish_issue(root / "stage", root / "persistent", "issue")
            math.materialize_issue(
                root / "persistent", "issue", root / "selected", layers=[2]
            )
            self.assertTrue(
                (root / "selected/rank-0/issue/layer-02-slot-00.pt").exists()
            )
            self.assertFalse(
                (root / "selected/rank-0/issue/layer-04-slot-00.pt").exists()
            )
            self.assertFalse((root / "selected/shared/layer-04-constants.pt").exists())

    def test_capture_disabled_has_no_observer_or_copies(self):
        with patch.object(capture, "_DIRECTORY", None), patch.object(
            capture, "HostCopy", side_effect=AssertionError("copy")
        ):
            self.assertFalse(capture.enabled())
            self.assertIsNone(capture.observer())

    def test_nonowner_optimized_raw_hook_does_not_copy(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            math.atomic_json(
                root / "capture-config.json",
                {
                    "storage_schema_version": 2,
                    "audit_shared": False,
                    "layers": [2],
                    "requests": {"issue": {"quota": 1}},
                },
            )
            with patch.object(
                torch.distributed, "is_initialized", return_value=True
            ), patch.object(torch.distributed, "get_rank", return_value=1):
                obs = capture.Observer(root)
            batch = SimpleNamespace(rids=["issue"])
            config_module = importlib.import_module(ref.__package__ + ".config")
            with patch.object(
                config_module, "packed_enabled", return_value=False
            ), patch.object(
                capture, "HostCopy", side_effect=AssertionError("shared copy")
            ):
                obs.observe_raw(2, batch, None, None, None, None, None, None)
            self.assertEqual(obs.shadow, {})
            obs.flush()

    def test_nonowner_attention_copies_only_head_fields(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            math.atomic_json(
                root / "capture-config.json",
                {
                    "storage_schema_version": 2,
                    "audit_shared": False,
                    "layers": [2],
                    "heads_per_rank": 16,
                    "seed": 42,
                    "requests": {"issue": {"quota": 1, "prompt_tokens": 100}},
                },
            )
            with patch.object(
                torch.distributed, "is_initialized", return_value=True
            ), patch.object(torch.distributed, "get_rank", return_value=1):
                obs = capture.Observer(root)
            batch = SimpleNamespace(
                rids=["issue"],
                positions=torch.tensor([103]),
                forward_mode=SimpleNamespace(is_decode=lambda: True),
            )
            q = torch.zeros(1, 16, 512)
            cache = torch.zeros(1, 16 * 584, dtype=torch.uint8)

            class CountingCopy(capture.HostCopy):
                calls = 0

                def __init__(self, tensor):
                    type(self).calls += 1
                    super().__init__(tensor)

            with patch.object(
                capture, "native_records", side_effect=AssertionError("shared gather")
            ), patch.object(capture, "HostCopy", CountingCopy):
                obs.observe_attention(
                    2,
                    batch,
                    q,
                    cache,
                    cache,
                    torch.tensor([[1]]),
                    torch.tensor([[0, 1, 0]]),
                    torch.tensor([1]),
                    torch.tensor([3]),
                    torch.zeros(16),
                    0.1,
                    q,
                    16,
                    16,
                    SimpleNamespace(),
                )
                obs.observe_output(2, batch, q, None, batch.positions, 1, 16)
                obs.flush()
                self.assertEqual(
                    CountingCopy.calls, 4
                )  # q, sink, pre/post-RoPE outputs
            record = math._read_payload(root / "rank-1/issue/layer-02-slot-00.pt")
            self.assertNotIn("raw_c", record)
            self.assertNotIn("csa_bytes", record)
            self.assertNotIn("query_freqs", record)

    def test_publication_failure_keeps_local_payloads(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config, result = self.checkpoint_fixture(root / "stage")
            math.checkpoint_issue(root / "stage", config, result)
            with patch.object(
                math.shutil, "copyfile", side_effect=OSError("transfer failed")
            ):
                with self.assertRaisesRegex(OSError, "transfer failed"):
                    math.publish_issue(
                        root / "stage", root / "persistent", "issue", cleanup=True
                    )
            self.assertTrue((root / "stage/rank-0/issue/layer-02-slot-00.pt").exists())
            self.assertFalse((root / "persistent/published/issue.json").exists())

    def test_optimized_rollout_rejects_conversion_and_changed_sources(self):
        config = {"layers": [2], "input_manifest": {"model_revision": "model"}}
        proof = {"status": "pass", "kind": "cpu-conversion", "issues": 20}
        with self.assertRaisesRegex(ValueError, "live-capture proof"):
            math.verify_shared_rollout(config, proof)
        proof.update(
            kind="audited-live-shared-capture",
            layers=[2],
            model_revision="model",
            collector_sha256="changed",
            runner_sha256=math.file_sha256(Path(math.__file__)),
        )
        with self.assertRaisesRegex(ValueError, "live-capture proof"):
            math.verify_shared_rollout(config, proof)

    def test_publication_insufficient_space_keeps_local_receipt(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config, result = self.checkpoint_fixture(root / "stage")
            math.checkpoint_issue(root / "stage", config, result)
            with patch.object(
                math.shutil, "disk_usage", return_value=SimpleNamespace(free=0)
            ):
                with self.assertRaisesRegex(ValueError, "insufficient persistent"):
                    math.publish_issue(
                        root / "stage", root / "persistent", "issue", cleanup=True
                    )
            self.assertIsNotNone(math.completed_issue(root / "stage", config, "issue"))

    def test_rank_assembly_aligns_selected_rows_and_preserves_duplicates(self):
        with tempfile.TemporaryDirectory() as directory:
            paths = []
            ids = torch.tensor([7, 3, 7, 9])
            rows = ids[:, None].repeat(1, 512).float()
            for rank in range(4):
                order = torch.roll(torch.arange(4), rank)
                data = {
                    "rank": rank,
                    "rid": "a",
                    "layer": 2,
                    "ordinal": 3,
                    "query_position": 131075,
                    "norm_eps": 1e-6,
                    "softmax_scale": 0.1,
                    "csa_indices": ids[order],
                    "raw_c": rows[order],
                    "raw_positions": (ids * 4)[order],
                    "csa_bytes": rows[order].to(torch.uint8),
                    "csa_freqs": rows[order, :32].to(torch.complex64),
                    "csa_valid": torch.ones(4, dtype=torch.bool),
                    "q": torch.full((16, 512), rank, dtype=torch.bfloat16),
                }
                for key in (
                    "swa_bytes",
                    "swa_indices",
                    "swa_valid",
                    "norm_weight",
                    "query_freqs",
                ):
                    data[key] = torch.zeros(1)
                paths.append(Path(directory, f"rank-{rank}.pt"))
                torch.save(data, paths[-1])
            state = math.load_state(paths, "cpu")
            self.assertEqual(state["csa_indices"].tolist(), [3, 7, 7, 9])
            self.assertTrue(
                torch.equal(state["raw_c"][:, 0], state["csa_indices"].float())
            )
            data["csa_indices"][0] = 12
            torch.save(data, paths[-1])
            with self.assertRaisesRegex(ValueError, "shared-data mismatch"):
                math.load_state(paths, "cpu")

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
    def test_cuda_real_planner_and_stream(self):
        from sglang.test.kernels.deepseek_v4.common import (
            make_paged_context,
            to_seq_extend,
        )

        ctx = make_paged_context(bs=1, compress_ratio=4, ring_size=8)
        seq, extend, num_q = to_seq_extend([(20, 20)])
        plan = ctx.make_prefill_plan(seq, extend, num_q)
        self.assertNotEqual(plan.plan_c.shape[0], plan.plan_w.shape[0])
        device = plan.plan_c.device
        raw = torch.randn(plan.plan_c.shape[0], 512, device=device).to(torch.bfloat16)
        weight = torch.ones(512, device=device)
        mask = ref.topmag_keep_mask(raw, 0.5)
        locations = torch.arange(num_q, device=device, dtype=torch.int64) + 16
        buffers = packed.PackedBuffers(
            torch.zeros(64, 256, dtype=torch.uint8, device=device),
            torch.zeros(64, 8, dtype=torch.uint64, device=device),
            torch.zeros(64, 8, dtype=torch.uint8, device=device),
        )
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            packed.pack_rows(raw, mask, weight, 1e-6, plan, locations, buffers)
            copy = capture.HostCopy(buffers.values)
        copy.wait()
        torch.cuda.current_stream().wait_stream(stream)
        codes, bitmaps, scales = ref.pack_rows_ref(raw, mask, weight, 1e-6)
        for row, _, loc, _ in capture.plan_writes(plan, locations):
            torch.testing.assert_close(buffers.values[loc], codes[row], atol=0, rtol=0)
            torch.testing.assert_close(
                buffers.bitmaps[loc].view(torch.int64),
                bitmaps[row].view(torch.int64),
                atol=0,
                rtol=0,
            )
            torch.testing.assert_close(buffers.scales[loc], scales[row], atol=0, rtol=0)


if __name__ == "__main__":
    torch.set_num_threads(4)
    unittest.main()
