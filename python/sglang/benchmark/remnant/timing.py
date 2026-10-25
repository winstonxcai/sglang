"""Shared event timing and paired scheduling; no model/runtime configuration.

Author: Winston Cai.
"""

from __future__ import annotations

import random
from collections.abc import Callable, Mapping


def round_orders(
    keys: tuple[str, ...], rounds: int, seed: int, *, balanced: bool = False
) -> list[list[str]]:
    if rounds < 2 or len(keys) < 2:
        raise ValueError("Paired timing needs two paths and at least two rounds")
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
