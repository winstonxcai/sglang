"""Unit tests for bench_serving streaming with reasoning chunks.
Author: Winston Cai. (recorded-request and tool-output coverage)

Reasoning models (DeepSeek-R1, MiMo, Qwen3 reasoning, Kimi-K2, ...) stream their
chain-of-thought via fields such as OpenAI's `delta.reasoning_content` and
vLLM Kimi's `delta.reasoning`. Without explicit support, bench_serving only
inspects `delta.content` and silently reports zero TTFT / ITL and an empty
`generated_text`, which then retokenizes to 0 tokens even though the backend
completed real work.
"""

import asyncio
import json
import socket
import threading
import tempfile
import time
import unittest
from argparse import Namespace
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from unittest.mock import Mock

from sglang.benchmark.datasets.openai_dataset import sample_openai_requests

from sglang.benchmark.serving import (
    RequestFuncInput,
    async_request_openai_chat_completions,
    calculate_metrics,
    set_global_args,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=16, suite="base-a-test-cpu")


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class _SSEHandler(BaseHTTPRequestHandler):
    """Streams a fixed sequence of OpenAI-compatible SSE chunks per test."""

    chunks: list = []
    chunk_delay_s: float = 0.02

    def do_POST(self):  # noqa: N802 (BaseHTTPRequestHandler interface)
        length = int(self.headers.get("Content-Length", "0"))
        if length:
            self.rfile.read(length)
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        for chunk in self.chunks:
            self.wfile.write(b"data: " + json.dumps(chunk).encode() + b"\n\n")
            self.wfile.flush()
            time.sleep(self.chunk_delay_s)
        self.wfile.write(b"data: [DONE]\n\n")
        self.wfile.flush()

    def log_message(self, fmt, *args):  # silence access logs
        return


class _JSONHandler(BaseHTTPRequestHandler):
    response_body: dict = {}
    request_bodies: list = []

    def do_POST(self):  # noqa: N802 (BaseHTTPRequestHandler interface)
        length = int(self.headers.get("Content-Length", "0"))
        if length:
            self.request_bodies.append(json.loads(self.rfile.read(length)))
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps(self.response_body).encode())
        self.wfile.flush()

    def log_message(self, fmt, *args):
        return


def _make_chunk(
    content=None, reasoning_content=None, reasoning=None, completion_tokens=None
):
    delta = {}
    if content is not None:
        delta["content"] = content
    if reasoning_content is not None:
        delta["reasoning_content"] = reasoning_content
    if reasoning is not None:
        delta["reasoning"] = reasoning
    chunk = {"choices": [{"index": 0, "delta": delta}]}
    if completion_tokens is not None:
        chunk["usage"] = {"completion_tokens": completion_tokens}
    return chunk


def _make_response(content=None, reasoning_content=None, completion_tokens=1):
    message = {"role": "assistant", "content": content}
    if reasoning_content is not None:
        message["reasoning_content"] = reasoning_content
    return {
        "choices": [{"index": 0, "message": message, "finish_reason": "length"}],
        "usage": {"completion_tokens": completion_tokens},
    }


class _StrictStringTokenizer:
    def encode(self, text, add_special_tokens=False):
        if not isinstance(text, str):
            raise ValueError("text input must be of type `str`")
        return text.split()


class TestBenchServingReasoningStream(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        set_global_args(
            Namespace(
                disable_stream=False,
                disable_ignore_eos=True,
                print_requests=False,
                tokenizer="",
                header=None,
            )
        )

    def _run(self, chunks, extra_request_body=None):
        port = _free_port()

        class Handler(_SSEHandler):
            pass

        Handler.chunks = list(chunks)
        server = HTTPServer(("127.0.0.1", port), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            req = RequestFuncInput(
                prompt="hello",
                api_url=f"http://127.0.0.1:{port}/v1/chat/completions",
                prompt_len=1,
                output_len=64,
                model="dummy-model",
                lora_name="",
                image_data=None,
                extra_request_body=extra_request_body or {},
            )
            return asyncio.run(async_request_openai_chat_completions(req))
        finally:
            server.shutdown()
            server.server_close()

    def test_tool_call_only_stream_records_first_output_and_usage(self):
        call = [{"index": 0, "function": {"name": "run", "arguments": ""}}]
        chunks = [
            {"choices": [{"delta": {"tool_calls": call}}]},
            {"choices": [], "usage": {"completion_tokens": 7}},
        ]
        out = self._run(chunks, {"stream_options": {"include_usage": True}})
        self.assertTrue(out.success, out.error)
        self.assertGreater(out.ttft, 0)
        self.assertEqual(out.output_len, 7)
        self.assertEqual(out.generated_text, json.dumps(call))

    def test_requested_stream_usage_must_be_present(self):
        out = self._run(
            [_make_chunk(content="answer")], {"stream_options": {"include_usage": True}}
        )
        self.assertFalse(out.success)
        self.assertIn("omitted requested", out.error)

    def test_recorded_prompt_length_preserves_messages_and_tools(self):
        messages = [{"role": "user", "content": [{"type": "text", "text": "code"}]}]
        tools = [{"type": "function", "function": {"name": "run"}}]
        tokenizer = Mock()
        tokenizer.apply_chat_template.side_effect = AssertionError("no HF template")
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "requests.jsonl"
            path.write_text(
                json.dumps(
                    {
                        "messages": messages,
                        "tools": tools,
                        "prompt_len": 12345,
                        "max_tokens": 39,
                    }
                )
                + "\n"
            )
            rows = sample_openai_requests(str(path), 1, tokenizer)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0].prompt, messages)
        self.assertEqual(rows[0].prompt_len, 12345)
        self.assertEqual(rows[0].output_len, 39)
        self.assertEqual(rows[0].extra_request_body, {"tools": tools})
        tokenizer.apply_chat_template.assert_not_called()

    def test_invalid_recorded_prompt_lengths_fail(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "requests.jsonl"
            for length in [-1, True, "123"]:
                with self.subTest(length=length):
                    path.write_text(
                        json.dumps(
                            {
                                "messages": [{"role": "user", "content": "x"}],
                                "prompt_len": length,
                            }
                        )
                        + "\n"
                    )
                    with self.assertRaisesRegex(ValueError, "nonnegative integer"):
                        sample_openai_requests(str(path), 1, Mock())

    def test_unknown_recorded_prompt_length_is_explicit_zero(self):
        tokenizer = Mock()
        tokenizer.apply_chat_template.side_effect = AssertionError("no HF template")
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "requests.jsonl"
            path.write_text(
                json.dumps(
                    {
                        "messages": [{"role": "user", "content": "code"}],
                        "prompt_len": 0,
                        "max_tokens": 2,
                    }
                )
                + "\n"
            )
            rows = sample_openai_requests(str(path), 1, tokenizer)
        self.assertEqual(rows[0].prompt_len, 0)
        self.assertNotIn("prompt_len", rows[0].extra_request_body)

    def test_reasoning_only_stream_populates_metrics(self):
        chunks = [
            _make_chunk(reasoning_content="Let "),
            _make_chunk(reasoning_content="me "),
            _make_chunk(reasoning_content="think."),
            _make_chunk(completion_tokens=3),
        ]
        out = self._run(chunks)

        self.assertTrue(out.success, msg=f"request failed: {out.error}")
        self.assertEqual(out.generated_text, "Let me think.")
        self.assertGreater(out.ttft, 0.0)
        self.assertEqual(len(out.itl), 2, msg="should record ITL for chunks 2..N")
        for v in out.itl:
            self.assertGreater(v, 0.0)
        self.assertEqual(out.text_chunks, ["me ", "think."])
        self.assertEqual(out.output_len, 3)

    def test_vllm_kimi_reasoning_stream_populates_metrics(self):
        chunks = [
            _make_chunk(reasoning="Let "),
            _make_chunk(reasoning="me "),
            _make_chunk(reasoning="think."),
            _make_chunk(completion_tokens=3),
        ]
        out = self._run(chunks)

        self.assertTrue(out.success, msg=f"request failed: {out.error}")
        self.assertEqual(out.generated_text, "Let me think.")
        self.assertGreater(out.ttft, 0.0)
        self.assertEqual(len(out.itl), 2, msg="should record ITL for chunks 2..N")
        self.assertEqual(out.output_len, 3)

    def test_reasoning_then_content_accounts_both(self):
        chunks = [
            _make_chunk(reasoning_content="step1 "),
            _make_chunk(reasoning_content="step2 "),
            _make_chunk(content="answer "),
            _make_chunk(content="here"),
            _make_chunk(completion_tokens=4),
        ]
        out = self._run(chunks)

        self.assertTrue(out.success, msg=f"request failed: {out.error}")
        self.assertEqual(out.generated_text, "step1 step2 answer here")
        self.assertGreater(out.ttft, 0.0)
        self.assertEqual(len(out.itl), 3)
        self.assertEqual(out.text_chunks, ["step2 ", "answer ", "here"])
        self.assertEqual(out.output_len, 4)

    def test_single_delta_preserves_reasoning_before_content(self):
        chunks = [
            _make_chunk(content="answer", reasoning_content="thought "),
            _make_chunk(completion_tokens=2),
        ]
        out = self._run(chunks)

        self.assertTrue(out.success, msg=f"request failed: {out.error}")
        self.assertEqual(out.generated_text, "thought answer")
        self.assertGreater(out.ttft, 0.0)
        self.assertEqual(out.output_len, 2)

    def test_usage_only_stream_chunk_does_not_break(self):
        chunks = [
            _make_chunk(reasoning_content="thinking"),
            {"choices": [], "usage": {"completion_tokens": 1}},
        ]
        out = self._run(chunks)

        self.assertTrue(out.success, msg=f"request failed: {out.error}")
        self.assertEqual(out.generated_text, "thinking")
        self.assertGreater(out.ttft, 0.0)
        self.assertEqual(out.output_len, 1)

    def test_content_only_stream_unchanged(self):
        chunks = [
            _make_chunk(content="hi "),
            _make_chunk(content="there"),
            _make_chunk(completion_tokens=2),
        ]
        out = self._run(chunks)

        self.assertTrue(out.success, msg=f"request failed: {out.error}")
        self.assertEqual(out.generated_text, "hi there")
        self.assertGreater(out.ttft, 0.0)
        self.assertEqual(len(out.itl), 1)
        self.assertEqual(out.text_chunks, ["there"])
        self.assertEqual(out.output_len, 2)

    def test_null_reasoning_field_does_not_break(self):
        # Mirrors sglang's _StreamDelta: reasoning_content is always emitted,
        # serialized as null when only content is present.
        chunks = [
            {
                "choices": [
                    {"index": 0, "delta": {"content": "ok", "reasoning_content": None}}
                ]
            },
            {
                "choices": [
                    {"index": 0, "delta": {"content": None, "reasoning_content": None}}
                ],
                "usage": {"completion_tokens": 1},
            },
        ]
        out = self._run(chunks)

        self.assertTrue(out.success, msg=f"request failed: {out.error}")
        self.assertEqual(out.generated_text, "ok")
        self.assertGreater(out.ttft, 0.0)


class TestBenchServingReasoningNonStream(CustomTestCase):
    def _run(self, response_body):
        set_global_args(
            Namespace(
                disable_stream=True,
                disable_ignore_eos=False,
                print_requests=False,
                tokenizer="",
                header=None,
            )
        )
        port = _free_port()

        class Handler(_JSONHandler):
            pass

        Handler.response_body = response_body
        Handler.request_bodies = []
        server = HTTPServer(("127.0.0.1", port), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            req = RequestFuncInput(
                prompt="hello",
                api_url=f"http://127.0.0.1:{port}/v1/chat/completions",
                prompt_len=1,
                output_len=64,
                model="dummy-model",
                lora_name="",
                image_data=None,
                extra_request_body={},
            )
            return (
                asyncio.run(async_request_openai_chat_completions(req)),
                Handler.request_bodies,
            )
        finally:
            server.shutdown()
            server.server_close()

    def test_reasoning_only_non_stream_metrics_retokenize_text(self):
        out, request_bodies = self._run(
            _make_response(
                content=None,
                reasoning_content="Let me think.",
                completion_tokens=3,
            )
        )

        self.assertTrue(out.success, msg=f"request failed: {out.error}")
        self.assertEqual(out.generated_text, "Let me think.")
        self.assertEqual(out.output_len, 3)
        self.assertFalse(request_bodies[0]["stream"])

        metrics, output_lens = calculate_metrics(
            input_requests=None,
            outputs=[out],
            dur_s=1.0,
            tokenizer=_StrictStringTokenizer(),
            backend="sglang-oai-chat",
        )
        self.assertEqual(metrics.completed, 1)
        self.assertEqual(output_lens, [3])
        self.assertEqual(metrics.total_output_retokenized, 3)

    def test_reasoning_then_content_non_stream_accounts_both(self):
        out, _ = self._run(
            _make_response(
                content="answer",
                reasoning_content="thought ",
                completion_tokens=2,
            )
        )

        self.assertTrue(out.success, msg=f"request failed: {out.error}")
        self.assertEqual(out.generated_text, "thought answer")
        self.assertGreater(out.ttft, 0.0)
        self.assertEqual(out.output_len, 2)

    def test_content_only_non_stream_unchanged(self):
        out, _ = self._run(_make_response(content="answer", completion_tokens=1))

        self.assertTrue(out.success, msg=f"request failed: {out.error}")
        self.assertEqual(out.generated_text, "answer")
        self.assertGreater(out.ttft, 0.0)
        self.assertEqual(out.output_len, 1)


if __name__ == "__main__":
    unittest.main()
