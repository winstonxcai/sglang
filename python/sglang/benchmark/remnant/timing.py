"""Shared event timing and paired scheduling; no model/runtime configuration.

Author: Winston Cai.
"""

from __future__ import annotations

import math
import random
import statistics
from collections.abc import Callable, Mapping


def round_orders(
    keys: tuple[str, ...], rounds: int, seed: int, *, balanced: bool = False
) -> list[list[str]]:
    if rounds < 1 or not keys:
        raise ValueError("Timing needs at least one path and one round")
    rng = random.Random(seed)
    if balanced:
        if len(keys) != 2 or rounds % 2:
            raise ValueError(
                "Balanced timing requires two paths and an even round count"
            )
        orders = [list(keys), list(reversed(keys))] * (rounds // 2)
        rng.shuffle(orders)
        return orders
    orders = []
    for _ in range(rounds):
        order = list(keys)
        rng.shuffle(order)
        orders.append(order)
    return orders


def measure(call: Callable[[], object], repeats: int) -> float:
    """Milliseconds per call, including GPU gaps within the event interval."""
    import torch

    if repeats < 1:
        raise ValueError("repeats must be positive")
    start, end = (
        torch.cuda.Event(enable_timing=True),
        torch.cuda.Event(enable_timing=True),
    )
    start.record()
    for _ in range(repeats):
        call()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) / repeats


def paired_samples(
    calls: Mapping[str, Callable[[], object]],
    rounds: int,
    seed: int,
    repeats: int,
    *,
    before: Callable[[str], None] | None = None,
    balanced: bool = False,
    sample: Callable[[Callable, int], float] = measure,
    check: Callable[[], None] | None = None,
) -> tuple[dict[str, list[float]], list[list[str]]]:
    orders = round_orders(tuple(calls), rounds, seed, balanced=balanced)
    samples: dict[str, list[float]] = {key: [] for key in calls}
    for order in orders:
        for key in order:
            if check:
                check()
            if before:
                before(key)
            samples[key].append(sample(calls[key], repeats))
    return samples, orders


def summary(samples: list[float]) -> dict:
    if len(samples) < 2 or any(not math.isfinite(x) or x <= 0 for x in samples):
        raise ValueError("timing requires at least two positive finite samples")
    ordered = sorted(samples)
    position = (len(samples) - 1) * 0.95
    lo, hi = math.floor(position), math.ceil(position)
    return {
        "samples": samples,
        "mean": statistics.mean(samples),
        "median": statistics.median(samples),
        "stdev": statistics.stdev(samples),
        "p95": ordered[lo] + (ordered[hi] - ordered[lo]) * (position - lo),
    }


def paired_ratio_summary(numerator: list[float], denominator: list[float]) -> dict:
    """Summarize ratios from corresponding complete-call timing rounds."""
    if len(numerator) != len(denominator):
        raise ValueError("paired timings must have equal round counts")
    summary(numerator)
    summary(denominator)
    return summary([a / b for a, b in zip(numerator, denominator)])
