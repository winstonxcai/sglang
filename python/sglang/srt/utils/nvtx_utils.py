# Copyright 2023-2024 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Profiler span helpers for hot SGLang code paths.

Author: Winston Cai. (Remnant batch metadata annotations.)

A span has two independent emitters:

* ``record_function`` -- emitted whenever a torch profiler is active, so spans
  show up in torch/Perfetto traces for free (no env, no extra package).
* ``nvtx`` range -- emitted only when the caller opts in via ``nvtx_enabled``
  (wired to a per-subsystem ``SGLANG_ENABLE_NVTX_*`` gate) and the ``nvtx``
  package is importable, for Nsight Systems timelines.

Decoupling the two lets every annotation site -- scheduler stages, batch-overlap
ops, and the speculative-decoding / forward spans -- share one primitive.
"""

import json
import logging
from contextlib import AbstractContextManager, ExitStack, contextmanager, nullcontext
from functools import partial, wraps
from typing import Any, Callable, Optional

import torch

from sglang.srt.environ import envs

logger = logging.getLogger(__name__)

_SCHEDULER_NVTX = envs.SGLANG_ENABLE_NVTX_SCHEDULER.get()
_OPERATIONS_NVTX = envs.SGLANG_ENABLE_NVTX_OPERATIONS.get()

_nvtx_module = None
if _SCHEDULER_NVTX or _OPERATIONS_NVTX:
    try:
        import nvtx as _nvtx_module  # type: ignore
    except ImportError:
        logger.warning(
            "An SGLANG_ENABLE_NVTX_* flag is set, but the `nvtx` package is "
            "missing. NVTX markers are disabled; torch profiler spans still emit."
        )

NVTX_AVAILABLE = _nvtx_module is not None
# Per-subsystem nvtx gates: emit nvtx ranges only when the flag is set AND the
# package is importable. The record_function path is independent of both.
NVTX_SCHEDULER_ENABLED = _SCHEDULER_NVTX and NVTX_AVAILABLE
NVTX_OPERATIONS_ENABLED = _OPERATIONS_NVTX and NVTX_AVAILABLE

# Default nvtx colors for statically-named spans (only used on the nvtx path).
_NVTX_COLOR_MAP = {
    "scheduler.recv_requests": "blue",
    "scheduler.process_input_requests": "purple",
    "scheduler.get_next_batch_to_run": "green",
    "scheduler.run_batch": "red",
    "scheduler.process_batch_result": "cyan",
}

_NULL_CONTEXT = nullcontext()


@contextmanager
def _profile_range_impl(
    debug_name: str, color: Optional[str], record: bool, nvtx_enabled: bool
):
    with ExitStack() as stack:
        if record:
            stack.enter_context(torch.profiler.record_function(debug_name))
        if nvtx_enabled:
            if color is None:
                color = _NVTX_COLOR_MAP.get(debug_name)
            stack.enter_context(_nvtx_module.annotate(debug_name, color=color))
        yield


def profile_range(
    debug_name: str, *, color: Optional[str] = None, nvtx_enabled: bool = False
):
    """Context manager emitting a profiler span for ``debug_name``.

    A torch ``record_function`` is emitted whenever a torch profiler is active;
    an nvtx range is emitted additionally when ``nvtx_enabled`` is true. Returns a
    shared no-op when neither applies, so off-profile hot paths pay only one
    ``_profiler_enabled()`` check.
    """
    record = torch.autograd._profiler_enabled()
    if not record and not nvtx_enabled:
        return _NULL_CONTEXT
    return _profile_range_impl(debug_name, color, record, nvtx_enabled)


def profile_method(
    debug_name: str, *, color: Optional[str] = None, nvtx_enabled: bool = False
):
    """Decorator form of ``profile_range``."""

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            with profile_range(debug_name, color=color, nvtx_enabled=nvtx_enabled):
                return func(*args, **kwargs)

        return wrapper

    return decorator


# Pre-bound per-subsystem helpers: torch spans always (under a profiler), nvtx
# ranges only when that subsystem's gate is on.
scheduler_nvtx_method = partial(profile_method, nvtx_enabled=NVTX_SCHEDULER_ENABLED)
operations_nvtx_range = partial(profile_range, nvtx_enabled=NVTX_OPERATIONS_ENABLED)


def batch_profile_range(
    name: str,
    batch: Any,
    *,
    step: Optional[int] = None,
    rank: Optional[int] = None,
    padded_bs: Optional[int] = None,
) -> AbstractContextManager:
    """Annotate a launch with CPU-only metadata; never read a CUDA scalar.

    Scheduler ranges describe scheduled batches. Nested graph-replay ranges
    describe the resolved forward batch and its actual padded bucket. Join them
    through the CPU range nesting and CUDA launch correlation, not range wall
    time: overlap scheduling remains asynchronous.
    """
    if not NVTX_SCHEDULER_ENABLED:
        return _NULL_CONTEXT

    size = batch.batch_size
    size = size() if callable(size) else size
    mode = batch.forward_mode.name
    decode_requests = None
    prefill_requests = None
    prefill_tokens = None
    if mode == "DECODE":
        decode_requests, prefill_requests, prefill_tokens = size, 0, 0
    elif mode in ("EXTEND", "SPLIT_PREFILL", "MIXED"):
        decoding_reqs = getattr(batch, "decoding_reqs", None)
        if mode != "MIXED" or decoding_reqs is not None:
            decode_requests = len(decoding_reqs) if decoding_reqs is not None else 0
            prefill_requests = size - decode_requests
            extend_tokens = getattr(batch, "extend_num_tokens", None)
            if extend_tokens is not None:
                prefill_tokens = extend_tokens - decode_requests

    metadata = {
        "step": step,
        "rank": rank,
        "mode": mode,
        "requests": size,
        "decode_requests": decode_requests,
        "prefill_requests": prefill_requests,
        "prefill_tokens": prefill_tokens,
        "padded_bs": padded_bs,
        "context_source": "unavailable",
    }
    lengths = getattr(batch, "seq_lens_cpu", None)
    if lengths is not None and lengths.device.type == "cpu":
        # No .cpu(), .item() or reads of seq_lens on the GPU. In the scheduler
        # this CPU mirror may precede overlap resolution; the replay annotation
        # uses the forward batch after load_batch has resolved its metadata.
        values = lengths.tolist()
        metadata.update(
            context_source="seq_lens_cpu",
            context_count=len(values),
            context_min=min(values) if values else None,
            context_max=max(values) if values else None,
            context_sum=sum(values),
        )
    return profile_range(
        name + " " + json.dumps(metadata, separators=(",", ":")),
        nvtx_enabled=True,
    )


def scheduler_batch_nvtx_method(func: Callable) -> Callable:
    """Add step metadata while preserving the existing static scheduler span."""

    @wraps(func)
    def wrapper(self: Any, batch: Any, *args: Any, **kwargs: Any) -> Any:
        if not NVTX_SCHEDULER_ENABLED:
            return func(self, batch, *args, **kwargs)
        with batch_profile_range(
            "scheduler.batch",
            batch,
            step=self.forward_ct + 1,
            rank=self.ps.tp_rank,
        ):
            return func(self, batch, *args, **kwargs)

    return wrapper
