"""Shared CPU preflight, TP4 capture client, and model-free attention replay.

Author: Winston Cai.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import copy
import csv
import hashlib
import io
import json
import math
import os
import shutil
import stat
import sys
import tarfile
import threading
import time
import types
from collections import defaultdict
from pathlib import Path
from uuid import uuid4

import numpy as np
import torch

# Offline-only cache: parallel sequential reads avoid thousands of cold-volume
# seeks. Capture uses the ordinary uncached loader. The 64-GiB L4 worker keeps
# at most 24 GiB of immutable snapshot bytes; tensors still load on CPU.
_snapshot_bytes = {}


def _read_payload(path):
    data = _snapshot_bytes.get(Path(path))
    return torch.load(
        io.BytesIO(data) if data is not None else path,
        map_location="cpu",
        weights_only=True,
    )


def _checked_reference(root, reference):
    path = (root / reference["path"]).resolve()
    if not path.is_relative_to(root.resolve()) or path == root.resolve():
        raise ValueError("capture reference escapes root")
    payload = path.read_bytes()
    if hashlib.sha256(payload).hexdigest() != reference["sha256"]:
        raise ValueError(f"capture reference hash mismatch: {reference['path']}")
    return torch.load(io.BytesIO(payload), map_location="cpu", weights_only=True)


def _selection_permutation(source, target):
    """Stable alignment preserves duplicate selections and their multiplicity."""
    a, b = torch.argsort(source, stable=True), torch.argsort(target, stable=True)
    if not torch.equal(source[a], target[b]):
        raise ValueError("rank selected-row multiset mismatch")
    permutation = torch.empty_like(a)
    permutation[b] = a
    return permutation


def read_snapshot(path):
    record = _read_payload(path)
    version = record.get("storage_schema_version", 1)
    if version == 1:
        return record
    if version != 2:
        raise ValueError("unsupported capture snapshot schema")
    root = Path(path).parents[2]
    owner = _read_payload(root / "rank-0" / record["rid"] / Path(path).name)
    if owner.get("storage_schema_version") != 2 or owner.get("rank") != 0:
        raise ValueError("missing schema-2 shared owner")
    common = _checked_reference(
        root, {"path": owner["shared_path"], "sha256": owner["shared_sha256"]}
    )
    if record["shared_path"] != owner["shared_path"]:
        raise ValueError("rank shared reference differs")
    for key in ("rid", "layer", "ordinal", "query_position"):
        if record[key] != common[key]:
            raise ValueError(f"shared identity mismatch: {key}")
    cperm = _selection_permutation(
        common["csa_indices"][common["csa_valid"]],
        record["csa_indices"][record["csa_valid"]],
    )
    sperm = _selection_permutation(
        common["swa_indices"][common["swa_valid"]],
        record["swa_indices"][record["swa_valid"]],
    )
    shared = {
        k: common[k]
        for k in (
            "raw_c",
            "raw_positions",
            "csa_bytes",
            "swa_bytes",
            "csa_freqs",
            "query_freqs",
        )
    }
    for key in ("raw_c", "raw_positions", "csa_bytes", "csa_freqs"):
        shared[key] = shared[key][cperm]
    shared["swa_bytes"] = shared["swa_bytes"][sperm]
    shared.update(_checked_reference(root, common["constants"]))
    audit = record.get("audit_shared")
    if record.get("audit_required") and record["rank"] != 0 and audit is None:
        raise ValueError("missing required rank shared audit")
    if audit is not None:
        for key, value in shared.items():
            same = (
                torch.equal(value, audit[key])
                if isinstance(value, torch.Tensor)
                else value == audit[key]
            )
            if not same:
                raise ValueError(f"rank shared audit mismatch: {key}")
    metadata = {
        k: v
        for k, v in record.items()
        if k
        not in {
            "audit_shared",
            "audit_required",
            "shared_path",
            "shared_sha256",
            "storage_schema_version",
        }
    }
    return {**metadata, **shared}


def snapshot_dependencies(paths):
    result = set(map(Path, paths))
    for path in paths:
        record = _read_payload(path)
        if record.get("storage_schema_version", 1) == 2:
            root = Path(path).parents[2]
            owner = _read_payload(root / "rank-0" / record["rid"] / Path(path).name)
            ref = {"path": owner["shared_path"], "sha256": owner["shared_sha256"]}
            common = _checked_reference(root, ref)
            _checked_reference(root, common["constants"])
            result.update((root / ref["path"], root / common["constants"]["path"]))
    return sorted(result)


def convert_shared_group(paths, output, verify=True):
    """Offline schema-1 conversion; the model and source files are untouched."""
    from importlib import import_module

    capture = import_module(reference().__package__ + ".capture")
    load_state(paths, "cpu")  # Reject disagreement before deleting duplicates.
    converted = []
    for rank, source in enumerate(paths):
        data = read_snapshot(source)
        destination = Path(output) / f"rank-{rank}" / data["rid"] / Path(source).name
        slot = int(destination.stem.rsplit("-", 1)[1])
        capture.write_shared_snapshot(Path(output), destination, data, slot)
        converted.append(destination)
    if verify:
        for old_path, new_path in zip(paths, converted):
            old, new = read_snapshot(old_path), read_snapshot(new_path)
            if set(old) != set(new):
                raise ValueError("conversion changed snapshot fields")

            def equal(a, b):
                if isinstance(a, torch.Tensor):
                    return isinstance(b, torch.Tensor) and torch.equal(a, b)
                if isinstance(a, dict):
                    return set(a) == set(b) and all(equal(a[k], b[k]) for k in a)
                if isinstance(a, (list, tuple)):
                    return len(a) == len(b) and all(equal(x, y) for x, y in zip(a, b))
                return a == b

            if not equal(old, new):
                raise ValueError(f"conversion changed tensors/metadata: {old_path}")
    return converted


def cache_snapshots(paths, deadline, max_bytes=24 * 1024**3):
    total = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=32) as pool:
        for path, data in pool.map(lambda p: (p, p.read_bytes()), paths):
            if time.monotonic() >= deadline:
                raise TimeoutError("snapshot preload exceeded allocation deadline")
            total += len(data)
            if total > max_bytes:
                raise ValueError("snapshot cache exceeds bounded memory budget")
            _snapshot_bytes[path] = data
    return total


def archive_capture(root):
    """CPU-only, bounded parallel reads into one atomically committed archive."""
    if json.loads((root / "allocation.json").read_text())["status"] != "complete":
        raise ValueError("refuse to archive an incomplete capture")
    archive = root.with_suffix(".tar.gz")
    if archive.exists():
        return archive
    paths = []
    for directory, subdirs, files in os.walk(root, followlinks=False):
        subdirs[:] = sorted(
            name
            for name in subdirs
            if name not in {"analysis", "jit-cache", "offline-attempts"}
        )
        paths.extend(
            Path(directory) / name
            for name in sorted(files)
            if name != "offline-allocation.json"
        )

    def read_member(path):
        metadata = path.stat(follow_symlinks=False)
        if not stat.S_ISREG(metadata.st_mode):
            return None
        payload = path.read_bytes()
        member = tarfile.TarInfo(str(Path(root.name) / path.relative_to(root)))
        member.size, member.mtime = len(payload), metadata.st_mtime
        member.mode = metadata.st_mode & 0o777
        return member, payload

    temporary = archive.with_name(archive.name + "." + uuid4().hex + ".partial")
    with tarfile.open(temporary, "w:gz", compresslevel=1) as bundle:
        with concurrent.futures.ThreadPoolExecutor(max_workers=32) as pool:
            # Limit completed read buffers to one batch, not the whole capture.
            for begin in range(0, len(paths), 32):
                for result in pool.map(read_member, paths[begin : begin + 32]):
                    if result is not None:
                        member, payload = result
                        bundle.addfile(member, io.BytesIO(payload))
    temporary.replace(archive)
    return archive


def reference():
    # Load the real pure-Torch primitives without importing the serving stack.
    name = "_remnant_calibration_reference"
    if name not in sys.modules:
        package = types.ModuleType(name)
        package.__path__ = [str(Path(__file__).parents[2] / "srt/remnant")]
        sys.modules[name] = package
    from importlib import import_module

    return import_module(name + ".reference")


def rotate(x, frequencies, inverse=False):
    x = x.clone()
    tail = x[..., -64:].reshape(*x.shape[:-1], 32, 2)
    freq = torch.view_as_real(frequencies).to(x.dtype)
    real, imag = tail[..., 0].clone(), tail[..., 1].clone()
    cos, sin = freq[..., 0], freq[..., 1]
    tail[..., 0] = real * cos + imag * sin if inverse else real * cos - imag * sin
    tail[..., 1] = imag * cos - real * sin if inverse else imag * cos + real * sin
    return x


def normalize(c, weight, eps, frequencies):
    value = c * torch.rsqrt(c.square().mean(-1, keepdim=True) + eps) * weight
    return rotate(value, frequencies)


def decode_native(records):
    nope = records[:, :448].contiguous().view(torch.float8_e4m3fn).float()
    scales = torch.exp2(records[:, 576:583].float() - 127).repeat_interleave(64, -1)
    tail = records[:, 448:576].contiguous().view(torch.bfloat16).float()
    return torch.cat((nope * scales, tail), -1)


def attention(q, rows, sink, scale):
    logits = q @ rows.T * scale
    joined = torch.cat((logits, sink[:, None]), -1)
    probabilities = torch.softmax(joined, -1)[:, :-1]
    # Pinned FlashMLA reports natural-log LSE before adding the sink.
    lse = torch.logsumexp(logits, -1)
    return probabilities @ rows, probabilities, lse


def row_update(output, q, old, new, mass, scale):
    delta = (new - old) @ q.T * scale
    log_target = mass.log()[None, :] + delta
    log_d = torch.logaddexp(torch.log1p(-mass)[None, :], log_target)
    rest = output - mass[:, None] * old
    return (
        rest[None, :, :] * torch.exp(-log_d)[:, :, None]
        + new[:, None, :] * torch.exp(log_target - log_d)[:, :, None]
    )


def analytic_diagonal(c, weight, eps, freq, q, output, mass, scale):
    dimension = c.numel()
    a = torch.rsqrt(c.square().mean() + eps)
    u = normalize(c, weight, eps, freq)
    v = u[None, :] - output
    qt, vt = rotate(q, freq, True), rotate(v, freq, True)
    wc = weight * c
    qb = a * weight * qt - a**3 / dimension * c * (qt @ wc)[:, None]
    vb = a * weight * vt - a**3 / dimension * c * (vt @ wc)[:, None]
    bnorm = (a * weight).square() - 2 * a**4 / dimension * weight.square() * c.square()
    bnorm += a**6 / dimension**2 * c.square() * wc.square().sum()
    diagonal = (
        mass[:, None].square()
        * (
            bnorm
            + 2 * scale * qb * vb
            + scale**2 * qb.square() * v.square().sum(-1)[:, None]
        )
    ).sum(0)
    bound = diagonal.abs().max().clamp_min(1e-30) * 1e-5
    if bool((diagonal < -bound).any()) or not bool(torch.isfinite(diagonal).all()):
        raise ValueError("invalid analytic diagonal")
    return diagonal.clamp_min(0)


def bootstrap(values, seed=42):
    values = np.asarray(values, dtype=float)
    if not len(values):
        raise ValueError("no independent issues")
    rng = np.random.default_rng(seed)
    means = values[rng.integers(len(values), size=(2000, len(values)))].mean(1)
    return [float(x) for x in np.quantile(means, [0.025, 0.975])]


def correlation(x, y, ranks=False):
    if ranks:

        def rank(a):
            order = np.argsort(a, kind="stable")
            unique, starts, counts = np.unique(
                a[order], return_index=True, return_counts=True
            )
            del unique
            result = np.empty(len(a), float)
            for start, count in zip(starts, counts):
                result[order[start : start + count]] = start + (count - 1) / 2
            return result

        x, y = rank(x), rank(y)
    if np.std(x) == 0 or np.std(y) == 0:
        return None
    return float(np.corrcoef(x, y)[0, 1])


def sensitivity_summary(issues, requests, weights):
    """Uncertainty unit is an issue, not a head/coordinate/query."""
    summary = {}
    for split in ("calibration", "held_out"):
        vectors = np.asarray(
            [x["r"] for rid, x in issues.items() if requests[rid]["split"] == split]
        )
        mean = vectors.mean(0)
        correlations = {}
        for name, ranks in (("spearman", True), ("pearson", False)):
            values = [correlation(weights, v, ranks) for v in vectors]
            valid = [v for v in values if v is not None]
            correlations[name] = {
                "mean": float(np.mean(valid)) if valid else None,
                "ci": bootstrap(valid) if valid else None,
            }
        regions = {}
        for name, bounds in (
            ("all", (0, 512)),
            ("nope", (0, 448)),
            ("rope", (448, 512)),
        ):
            value = mean[slice(*bounds)]
            spreads = [
                np.quantile(v[slice(*bounds)], 0.9)
                / max(np.quantile(v[slice(*bounds)], 0.1), 1e-30)
                for v in vectors
            ]
            regions[name] = {
                "p90_p10": float(
                    np.quantile(value, 0.9) / max(np.quantile(value, 0.1), 1e-30)
                ),
                "cv": float(value.std() / max(value.mean(), 1e-30)),
                "issue_mean_spread_ci": bootstrap(spreads),
            }
        summary[split] = {
            "issues": len(vectors),
            "correlations": correlations,
            "regions": regions,
        }
    return summary


def empirical_ratio(errors, magnitudes, floor):
    valid = magnitudes > floor
    return (
        torch.where(valid, errors / (magnitudes + floor), 0).sum(0)
        / valid.sum(0).clamp_min(1)
    ), valid


def load_state(paths, device):
    shards = [read_snapshot(p) for p in paths]
    # Top-K ordering can differ across TP ranks even when their selected
    # multiset is identical. Align every row-associated tensor by physical ID;
    # never align raw rows separately from the cache bytes or RoPE positions.
    for shard in shards:
        valid = shard["csa_valid"]
        compact_order = torch.argsort(shard["csa_indices"][valid], stable=True)
        valid_slots = torch.arange(len(valid))[valid][compact_order]
        invalid_slots = torch.arange(len(valid))[~valid]
        order = torch.cat((valid_slots, invalid_slots))
        for key in ("raw_c", "raw_positions", "csa_bytes", "csa_freqs"):
            shard[key] = shard[key][compact_order]
        for key in ("csa_indices", "csa_valid"):
            shard[key] = shard[key][order]
    first = shards[0]
    for rank, shard in enumerate(shards):
        if shard["rank"] != rank:
            raise ValueError("missing or reordered TP rank")
        for key in (
            "rid",
            "layer",
            "ordinal",
            "query_position",
            "norm_eps",
            "softmax_scale",
        ):
            if shard[key] != first[key]:
                raise ValueError(f"rank identity mismatch: {key}")
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
            if not torch.equal(shard[key], first[key]):
                raise ValueError(f"rank shared-data mismatch: {key}")
    state = dict(first)
    for key in ("q", "sink", "bar_o", "o", "lse"):
        if key in first:
            state[key] = torch.cat([shard[key] for shard in shards], 0)
    if state["q"].shape != (64, 512):
        raise ValueError("expected all 64 real query heads")
    return {
        k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in state.items()
    }


def validate_state(state):
    q = state["q"].float()
    # Capture already compacts each byte array using its full selection mask.
    # Preserve duplicate admitted rows; applying the mask again misaddresses
    # padded/partial selections and fails on empty compact arrays.
    for prefix in ("swa", "csa"):
        records = state[prefix + "_bytes"]
        expected = int(state[prefix + "_valid"].sum())
        if records.shape != (expected, 584):
            raise ValueError(
                f"{prefix} compact row count/layout mismatch: expected {(expected, 584)}, got {tuple(records.shape)}"
            )
    rows = torch.cat(
        (decode_native(state["swa_bytes"]), decode_native(state["csa_bytes"]))
    )
    output, _, lse = attention(q, rows, state["sink"].float(), state["softmax_scale"])
    if rows.shape[0] == 0:
        # FlashMLA's final (combined or no-split) output uses +inf for an
        # empty domain, unlike the mathematical logsumexp result of -inf.
        lse = torch.full_like(lse, float("inf"))
    torch.testing.assert_close(output, state["bar_o"].float(), atol=0.02, rtol=0.02)
    inverse = rotate(state["bar_o"].float(), state["query_freqs"], True).to(
        torch.bfloat16
    )
    torch.testing.assert_close(
        inverse.float(), state["o"].float(), atol=0.02, rtol=0.02
    )
    if "lse" in state:
        torch.testing.assert_close(lse, state["lse"].float(), atol=0.01, rtol=0)
        # Equal +inf values are the legitimate Native LSE for an empty domain.
        lse_error = torch.where(
            lse == state["lse"].float(), 0, (lse - state["lse"].float()).abs()
        )
    return {
        "max_output_error": float((output - state["bar_o"].float()).abs().max()),
        "max_lse_error": float(lse_error.max()) if "lse" in state else None,
    }


def state_paths(root, config, rid, layer):
    banks = [
        json.loads((root / f"rank-{r}" / rid / "reservoir.json").read_text())
        for r in range(4)
    ]
    if any(bank != banks[0] for bank in banks):
        raise ValueError("rank reservoir mismatch")
    result = []
    for slot, ordinal in sorted(banks[0].items(), key=lambda item: item[1]):
        paths = [
            root / f"rank-{r}" / rid / f"layer-{layer:02d}-slot-{int(slot):02d}.pt"
            for r in range(4)
        ]
        for path in paths:
            data = read_snapshot(path)
            if data["ordinal"] != ordinal:
                raise ValueError("capture writer has not committed the final reservoir")
        result.append(paths)
    return result


def render_serving_request(body, encoder, request_type):
    """Use the serving protocol's defaults and field ordering, not raw JSON."""
    request = request_type(**body)
    messages = [message.model_dump() for message in request.messages]
    for message in messages:
        if not isinstance(message.get("content"), (str, type(None))):
            raise ValueError("capture cohort must contain text-only messages")
        if message.get("content") is None:
            message["content"] = ""
        for call in message.get("tool_calls") or []:
            arguments = call["function"].get("arguments")
            if isinstance(arguments, str):
                call["function"]["arguments"] = json.loads(arguments)
    if not messages or messages[0]["role"] != "system":
        messages.insert(0, {"role": "system", "content": ""})
    if request.tools:
        messages[0]["tools"] = [tool.model_dump() for tool in request.tools]
    if (request.chat_template_kwargs or {}).get("thinking") is not False:
        raise ValueError("frozen cohort must explicitly disable thinking")
    if request.continue_final_message or request.task is not None:
        raise ValueError("unsupported continuation/task in frozen cohort")
    return encoder.encode_messages(
        messages,
        thinking_mode="chat",
        reasoning_effort=request.reasoning_effort,
        reasoning_effort_profile="official",
    )


def audit_serving_prompts(items, model):
    from transformers import AutoTokenizer

    from sglang.srt.entrypoints.openai import encoding_dsv4 as encoder
    from sglang.srt.entrypoints.openai.protocol import ChatCompletionRequest

    tokenizer = AutoTokenizer.from_pretrained(
        str(model), trust_remote_code=True, local_files_only=True
    )
    audit = {}
    for item in items:
        prompt = render_serving_request(item["request"], encoder, ChatCompletionRequest)
        audit[item["instance_id"]] = {
            "prompt_tokens": len(tokenizer.encode(prompt)),
            "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
            "prepared_prompt_tokens": item["prompt_tokens"],
            "prepared_prompt_sha256": item["prompt_sha256"],
            "serving_encoder_sha256": hashlib.sha256(
                Path(encoder.__file__).read_bytes()
            ).hexdigest(),
        }
    return audit


def replay_body(item):
    body = copy.deepcopy(item["request"])
    body.pop("max_tokens", None)
    body.pop("max_completion_tokens", None)
    body.update(
        rid=item["instance_id"], stream=True, stream_options={"include_usage": True}
    )
    return body


def atomic_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + "." + uuid4().hex + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def cohort_contract(items, manifest):
    expected = manifest.get("requested_count", len(manifest["selected"]))
    if manifest["status"] != "complete" or len(items) != expected or not items:
        raise ValueError("need the complete frozen cohort")
    ids = [x["instance_id"] for x in items]
    if len(set(ids)) != len(ids) or any(
        not rid
        or any(
            c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_."
            for c in rid
        )
        for rid in ids
    ):
        raise ValueError("duplicate or unsafe request identities")
    if len({x["repo"] for x in items}) != len(items):
        raise ValueError("one independent issue per repository required")
    if any(x["split"] not in {"calibration", "held_out"} for x in items):
        raise ValueError("invalid split")
    if manifest.get("held_out_only") and any(x["split"] != "held_out" for x in items):
        raise ValueError("validation cohort must remain held-out")


def completed_issue(root, config, rid, verify_hashes=True):
    """Only immutable, fully drained natural completions can be resumed/exported."""
    marker = root / "done" / f"{rid}.json"
    if not marker.exists():
        return None
    receipt = json.loads(marker.read_text())
    fingerprint = hashlib.sha256(
        (root / "capture-config.json").read_bytes()
    ).hexdigest()
    if receipt.get("capture_config_sha256") != fingerprint or receipt.get("rid") != rid:
        raise ValueError("completion checkpoint belongs to another frozen capture")
    if receipt.get("response_id") != rid or receipt.get("finish_reason") not in {
        "stop",
        "tool_calls",
    }:
        raise ValueError("invalid natural completion checkpoint")
    if receipt["usage"]["prompt_tokens"] != config["requests"][rid]["prompt_tokens"]:
        raise ValueError("checkpoint prompt mismatch")
    expected = set()
    for layer in config["layers"]:
        paths = state_paths(root, config, rid, layer)
        if not paths or len(paths) > config["requests"][rid]["quota"]:
            raise ValueError("invalid checkpoint coverage")
        expected.update(
            str(p.relative_to(root))
            for group in paths
            for p in snapshot_dependencies(group)
        )
    expected.update(f"rank-{rank}/{rid}/reservoir.json" for rank in range(4))
    if set(receipt.get("files_sha256", {})) != expected:
        raise ValueError("incomplete checkpoint file manifest")
    if verify_hashes:
        for name, digest in receipt["files_sha256"].items():
            if hashlib.sha256((root / name).read_bytes()).hexdigest() != digest:
                raise ValueError("completed capture payload changed")
    return receipt


def checkpoint_issue(root, config, result):
    rid = result["rid"]
    if result["response_id"] != rid or result["finish_reason"] not in {
        "stop",
        "tool_calls",
    }:
        raise ValueError("only natural completions can be checkpointed")
    if result["usage"]["prompt_tokens"] != config["requests"][rid]["prompt_tokens"]:
        raise ValueError("checkpoint prompt mismatch")
    paths = []
    for layer in config["layers"]:
        for group in state_paths(root, config, rid, layer):
            if _read_payload(group[0]).get("storage_schema_version") == 2:
                load_state(group, "cpu")  # Audit and alignment precede publication.
            paths.extend(snapshot_dependencies(group))
    if not paths:
        raise ValueError("empty checkpoint")
    paths.extend(root / f"rank-{rank}" / rid / "reservoir.json" for rank in range(4))
    receipt = {
        **result,
        "capture_config_sha256": hashlib.sha256(
            (root / "capture-config.json").read_bytes()
        ).hexdigest(),
        "files_sha256": {
            str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in paths
        },
    }
    atomic_json(root / "done" / f"{rid}.json", receipt)
    return receipt


def preflight(cohort, model, output, storage_schema=1, audit_shared=False):
    config = json.loads((model / "config.json").read_text())
    layers = [i for i, ratio in enumerate(config["compress_ratios"]) if ratio == 4]
    if layers != list(range(2, 43, 2)) or config["num_attention_heads"] != 64:
        raise ValueError("unexpected pinned model geometry")
    index = model / "model.safetensors.index.json"
    shards = set(json.loads(index.read_text())["weight_map"].values())
    if len(shards) != 48 or any(not (model / name).is_file() for name in shards):
        raise ValueError("expected all 48 pinned checkpoint shards")
    if output.exists():
        raise ValueError("new capture prefix required")
    items = [
        json.loads(line)
        for line in (cohort / "requests.jsonl").read_text().splitlines()
    ]
    manifest = json.loads((cohort / "manifest.json").read_text())
    audit = json.loads((cohort / "audit.json").read_text())
    cohort_contract(items, manifest)
    revision = "7872f01b1d1fe23eabc4c98b48bffcef5a386062"
    if audit["status"] != "pass" or manifest["model_revision"] != revision:
        raise ValueError("cohort audit or revision mismatch")
    for name, key in (
        ("requests.jsonl", "requests_jsonl_sha256"),
        ("manifest.json", "manifest_sha256"),
    ):
        if hashlib.sha256((cohort / name).read_bytes()).hexdigest() != audit[key]:
            raise ValueError(f"cohort changed: {name}")
    checkpoint_hashes = {}
    for name, expected in manifest["checkpoint_files_sha256"].items():
        actual = hashlib.sha256((model / name).read_bytes()).hexdigest()
        if actual != expected:
            raise ValueError(f"checkpoint/tokenizer mismatch: {name}")
        checkpoint_hashes[name] = actual
    # HF download receipts pin every shard without rereading 160 GB on CPU.
    for name in sorted(shards | {"config.json", "model.safetensors.index.json"}):
        receipt = model / ".cache/huggingface/download" / (name + ".metadata")
        if not receipt.is_file() or receipt.read_text().splitlines()[0] != revision:
            raise ValueError(f"missing pinned checkpoint receipt: {name}")
    prompt_audit = audit_serving_prompts(items, model)
    output.mkdir(parents=True)
    omit = set()
    if not manifest.get("held_out_only"):
        for split in ("calibration", "held_out"):
            omit.update(
                x["instance_id"]
                for x in sorted(
                    (x for x in items if x["split"] == split),
                    key=lambda x: x["instance_id"],
                )[:2]
            )
    requests = {}
    replay = []
    for item in items:
        rid = item["instance_id"]
        requests[rid] = {
            k: item[k]
            for k in (
                "instance_id",
                "trajectory_id",
                "repo",
                "split",
                "prompt_tokens",
                "prompt_sha256",
            )
        }
        requests[rid].update(prompt_audit[rid])
        requests[rid]["quota"] = 9 if rid in omit else 10
        replay.append({"rid": rid, "body": replay_body(item)})
    capture = {
        "schema_version": 1,
        "layers": layers,
        "heads_per_rank": 16,
        "seed": 42,
        "requests": requests,
        "input_manifest": manifest,
        "generation": "natural EOS; no completion token cap",
        "concurrency": 10,
        "checkpoint_sha256": checkpoint_hashes,
        "checkpoint_receipts": revision,
        "shards": sorted(shards),
    }
    if storage_schema == 2:
        capture.update(storage_schema_version=2, audit_shared=bool(audit_shared))
        capture["concurrency"] = 2
        for request in requests.values():
            request["quota"] = 10
    (output / "capture-config.json").write_text(json.dumps(capture, indent=2) + "\n")
    (output / "serving-tokenization-audit.json").write_text(
        json.dumps(prompt_audit, indent=2) + "\n"
    )
    (output / "replay.jsonl").write_text("".join(json.dumps(x) + "\n" for x in replay))
    print(
        json.dumps(
            {
                "layers": layers,
                "issues": len(items),
                "target_queries_per_layer": sum(x["quota"] for x in requests.values()),
            }
        ),
        flush=True,
    )


_publication_lock = threading.Lock()


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024**2), b""):
            digest.update(block)
    return digest.hexdigest()


def published_issue(destination, rid):
    marker = Path(destination) / "published" / f"{rid}.json"
    if not marker.exists():
        return None
    record = json.loads(marker.read_text())
    if record.get("rid") != rid or record.get("receipt", {}).get("rid") != rid:
        raise ValueError("published issue identity mismatch")
    archive = (Path(destination) / record["bundle"]).resolve()
    if (
        not archive.is_relative_to(Path(destination).resolve())
        or file_sha256(archive) != record["bundle_sha256"]
    ):
        raise ValueError("published capture bundle missing or corrupt")
    return record


def publish_issue(root, destination, rid, cleanup=False):
    """Single bounded publisher; canonical receipt is written last.

    No local payload is removed until rereading the persistent bundle succeeds.
    Prefix and production caches are not touched.
    """
    root, destination = Path(root), Path(destination)
    if root.resolve() == destination.resolve():
        raise ValueError("publication must be outside local staging")
    with _publication_lock:
        config = json.loads((root / "capture-config.json").read_text())
        receipt = completed_issue(root, config, rid)
        if receipt is None:
            raise ValueError("cannot publish an incomplete issue")
        destination.mkdir(parents=True, exist_ok=True)
        old = published_issue(destination, rid)
        if old is not None:
            if old["receipt"] != receipt:
                raise ValueError("published issue belongs to a different capture")
            return old
        for name in ("capture-config.json", "replay.jsonl", "source-manifest.json"):
            source = root / name
            if not source.exists():
                continue
            target = destination / name
            if target.exists():
                if file_sha256(source) != file_sha256(target):
                    raise ValueError(f"publication configuration differs: {name}")
            else:
                temp = target.with_name(target.name + "." + uuid4().hex + ".tmp")
                shutil.copyfile(source, temp)
                temp.replace(target)
        names = sorted(receipt["files_sha256"])
        required = sum((root / name).stat().st_size for name in names)
        if shutil.disk_usage(destination).free < required * 1.25:
            raise ValueError("insufficient persistent space with 25% headroom")
        bundles = destination / "bundles"
        bundles.mkdir(exist_ok=True)
        target = bundles / f"{rid}.tar.gz"
        if target.exists():
            raise ValueError(
                "uncommitted bundle exists; preserve and diagnose before retry"
            )
        temporary = bundles / f".{rid}.{uuid4().hex}.partial"
        try:
            with tarfile.open(temporary, "w:gz", compresslevel=1) as archive:
                for name in names + [f"done/{rid}.json"]:
                    archive.add(root / name, arcname=name, recursive=False)
            digest = file_sha256(temporary)
            temporary.replace(target)
            if file_sha256(target) != digest:
                raise ValueError("published bundle verification failed")
            record = {
                "rid": rid,
                "bundle": str(target.relative_to(destination)),
                "bundle_sha256": digest,
                "bundle_bytes": target.stat().st_size,
                "expanded_bytes": required + (root / f"done/{rid}.json").stat().st_size,
                "receipt": receipt,
            }
            atomic_json(destination / "published" / f"{rid}.json", record)
        finally:
            if temporary.exists():
                temporary.unlink()
        if cleanup:
            for name in names:
                # Per-layer constants are shared by other requests.
                if Path(name).parent == Path("shared"):
                    continue
                (root / name).unlink()
        return record


def materialize_issue(destination, rid, output, layers=None):
    """Bounded caller-owned extraction, rejecting links and unlisted members."""
    destination, output = Path(destination), Path(output)
    record = published_issue(destination, rid)
    if record is None or output.exists():
        raise ValueError("verified bundle and fresh extraction directory required")
    expected = set(record["receipt"]["files_sha256"]) | {f"done/{rid}.json"}
    selected = None if layers is None else {int(layer) for layer in layers}

    def wanted(name):
        if selected is None or not Path(name).name.startswith("layer-"):
            return True
        return int(Path(name).name.split("-")[1]) in selected

    expanded_bytes = record.get("expanded_bytes")
    if expanded_bytes is None:
        with tarfile.open(destination / record["bundle"], "r:gz") as archive:
            expanded_bytes = sum(member.size for member in archive)
    output.parent.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(output.parent).free < expanded_bytes * 1.25:
        raise ValueError("insufficient extraction space with 25% headroom")
    output.mkdir(parents=True)
    with tarfile.open(destination / record["bundle"], "r:gz") as archive:
        seen = set()
        for member in archive:
            path = (output / member.name).resolve()
            if (
                not member.isfile()
                or member.name not in expected
                or member.name in seen
                or not path.is_relative_to(output.resolve())
            ):
                raise ValueError("unsafe or unexpected capture bundle member")
            seen.add(member.name)
            if not wanted(member.name):
                continue
            path.parent.mkdir(parents=True, exist_ok=True)
            with archive.extractfile(member) as source, path.open("wb") as target:
                shutil.copyfileobj(source, target, length=1024**2)
        if seen != expected:
            raise ValueError("capture bundle file coverage incomplete")
    for name, digest in record["receipt"]["files_sha256"].items():
        if not wanted(name):
            continue
        if file_sha256(output / name) != digest:
            raise ValueError("extracted capture payload hash mismatch")
    for name in ("capture-config.json", "source-manifest.json"):
        if (destination / name).exists():
            shutil.copyfile(destination / name, output / name)
    config = json.loads((output / "capture-config.json").read_text())
    if selected is None:
        completed_issue(output, config, rid)
    else:
        if not selected or not selected <= set(config["layers"]):
            raise ValueError("invalid requested capture layers")
        if (
            hashlib.sha256((output / "capture-config.json").read_bytes()).hexdigest()
            != record["receipt"]["capture_config_sha256"]
        ):
            raise ValueError("extracted capture configuration differs")
        for layer in selected:
            state_paths(output, config, rid, layer)
    return output


def verify_shared_rollout(config, proof):
    """Fail closed: a conversion report is not an audited live-capture proof."""
    collector = Path(__file__).parents[2] / "srt/remnant/capture.py"
    revision = config["input_manifest"]["model_revision"]
    if (
        proof.get("status") != "pass"
        or proof.get("kind") != "audited-live-shared-capture"
        or proof.get("issues", 0) < 2
        or proof.get("layers") != config["layers"]
        or proof.get("model_revision") != revision
        or proof.get("collector_sha256") != file_sha256(collector)
        or proof.get("runner_sha256") != file_sha256(Path(__file__))
    ):
        raise ValueError(
            "shared-only rollout requires matching audited live-capture proof"
        )


def audit_published_capture(root, scratch):
    """Independent CPU proof for a fully published audited pilot, no fitting."""
    root, scratch = Path(root), Path(scratch)
    config = json.loads((root / "capture-config.json").read_text())
    if (
        config.get("storage_schema_version") != 2
        or config.get("audit_shared") is not True
    ):
        raise ValueError("an explicitly audited live schema-2 run is required")
    source = json.loads((root / "source-manifest.json").read_text())["source_sha256"]
    collector = Path(__file__).parents[2] / "srt/remnant/capture.py"
    for name, path in (
        ("srt/remnant/capture.py", collector),
        ("benchmark/remnant/calibrate.py", Path(__file__)),
    ):
        if source[name] != file_sha256(path):
            raise ValueError("audited pilot sources differ from current rollout")
    issues = sorted(p.stem for p in (root / "published").glob("*.json"))
    if len(issues) < 2:
        raise ValueError("at least two fully audited issues required")
    scratch.mkdir(parents=True, exist_ok=True)
    import tempfile

    checks = 0
    bundles = {}
    for rid in issues:
        with tempfile.TemporaryDirectory(prefix="shared-audit-", dir=scratch) as owned:
            extracted = materialize_issue(root, rid, Path(owned) / "capture")
            bundles[rid] = published_issue(root, rid)["bundle_sha256"]
            for layer in config["layers"]:
                for group in state_paths(extracted, config, rid, layer):
                    if any(
                        _read_payload(path).get("audit_required") is not True
                        for path in group
                    ):
                        raise ValueError(
                            "optimized/converted sample cannot supply live audit proof"
                        )
                    validate_state(load_state(group, "cpu"))
                    checks += 1
    proof = {
        "status": "pass",
        "kind": "audited-live-shared-capture",
        "issues": len(issues),
        "layers": config["layers"],
        "native_checks": checks,
        "bundles": bundles,
        "model_revision": config["input_manifest"]["model_revision"],
        "collector_sha256": file_sha256(collector),
        "runner_sha256": file_sha256(Path(__file__)),
        "audit_config_sha256": file_sha256(root / "capture-config.json"),
    }
    atomic_json(root / "shared-audit-proof.json", proof)
    return proof


def capture_preflight(root: Path, published: Path, model: Path, port: int) -> None:
    """Check prepared capture inputs and actual mounts, without loading a model.

    Server lifecycle, deadlines and container ownership belong to the shell
    launcher. This command is also safe in its GPU-free preflight container.
    """
    import socket
    import tempfile

    if not 1 <= port <= 65535:
        raise ValueError("port in 1..65535 required")
    if any(
        not p.is_absolute() or p.resolve() == Path("/")
        for p in (root, published, model)
    ):
        raise ValueError(
            "explicit absolute, non-root capture/model/publication directories required"
        )
    if root.resolve() == published.resolve():
        raise ValueError("scratch and persistent destinations must differ")
    if not model.is_dir() or not (root / "replay.jsonl").is_file():
        raise ValueError(
            "model directory and prepared capture root/replay.jsonl required"
        )
    config = json.loads((root / "capture-config.json").read_text())
    if config.get("storage_schema_version") != 2 or not isinstance(
        config.get("audit_shared"), bool
    ):
        raise ValueError("schema 2 and an explicit audit_shared policy are required")
    if config.get("concurrency") != 2:
        raise ValueError("capture pilot requires concurrency two")
    if not config["audit_shared"]:
        verify_shared_rollout(
            config, json.loads((root / "shared-audit-proof.json").read_text())
        )
    published.mkdir(parents=True, exist_ok=True)
    for destination in (root, published):
        # Access bits alone do not detect read-only mounts.
        with tempfile.TemporaryFile(dir=destination) as probe:
            probe.write(b"capture mount probe")
            probe.flush()
    if shutil.disk_usage(root).free < 50 * 1024**3:
        raise ValueError(f"local scratch {root} must leave 50 GiB free")
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", port))


def capture_client(root, endpoint, deadline, publish_root=None):
    import requests

    config = json.loads((root / "capture-config.json").read_text())
    if (
        config.get("storage_schema_version") == 2
        and config.get("audit_shared") is not True
    ):
        proof_path = root / "shared-audit-proof.json"
        verify_shared_rollout(config, json.loads(proof_path.read_text()))
    rows = [
        json.loads(line) for line in (root / "replay.jsonl").read_text().splitlines()
    ]
    results = []
    for item in rows:
        published = (
            published_issue(publish_root, item["rid"])
            if publish_root is not None
            else None
        )
        if published is not None:
            if (
                published["receipt"]["capture_config_sha256"]
                != hashlib.sha256(
                    (root / "capture-config.json").read_bytes()
                ).hexdigest()
            ):
                raise ValueError("published resume configuration differs")
            results.append(published["receipt"])
            continue
        receipt = completed_issue(root, config, item["rid"])
        if receipt is not None:
            if publish_root is not None:
                publish_issue(root, publish_root, item["rid"], cleanup=True)
            results.append(receipt)
    completed = {x["rid"] for x in results}
    remaining = [x for x in rows if x["rid"] not in completed]
    started = time.monotonic()
    stop = threading.Event()

    def progress(status):
        elapsed = time.monotonic() - started
        newly_completed = len(results) - len(completed)
        atomic_json(root / "capture-results.json", results)
        atomic_json(
            root / "capture-progress.json",
            {
                "status": status,
                "target": len(rows),
                "completed": len(results),
                "completed_ids": sorted(x["rid"] for x in results),
                "client_seconds": elapsed,
                "deadline_remaining_seconds": max(0, deadline - time.monotonic()),
                "projected_remaining_seconds": (
                    (len(rows) - len(results)) * elapsed / newly_completed
                    if newly_completed
                    else None
                ),
                "projection_caveat": "Observed completion rate only; outstanding EOS tails may dominate.",
                "concurrency_limit": config["concurrency"],
                "generations_truncated_by_budget_are_not_complete": True,
            },
        )
        if publish_root is not None:
            for name in ("capture-results.json", "capture-progress.json"):
                atomic_json(
                    Path(publish_root) / name, json.loads((root / name).read_text())
                )

    def run(item, pilot=False):
        if stop.is_set() or time.monotonic() >= deadline:
            raise TimeoutError("do not admit another request after allocation deadline")
        if publish_root is not None:
            if shutil.disk_usage(root).free < 50 * 1024**3:
                raise ValueError("local staging free space fell below 50 GiB")
            used = sum(p.stat().st_size for p in root.rglob("*") if p.is_file())
            if used > 80 * 1024**3:
                raise ValueError("local staging exceeded 80 GiB; stop admissions")
        rid = item["rid"]
        begun = time.monotonic()
        finish, usage, response_id = None, None, None
        with requests.post(
            endpoint + "/v1/chat/completions",
            json=item["body"],
            headers={"x-override-rid": rid},
            stream=True,
            # C10 long-context prefill can exceed 60 seconds
            # before the first SSE event. Parent enforces the
            # allocation deadline independently of socket reads.
            timeout=(30, min(180, max(1, deadline - time.monotonic()))),
        ) as response:
            response.raise_for_status()
            for line in response.iter_lines():
                if stop.is_set() or time.monotonic() >= deadline:
                    raise TimeoutError("capture allocation deadline")
                if not line.startswith(b"data: ") or line == b"data: [DONE]":
                    continue
                data = json.loads(line[6:])
                if "error" in data:
                    raise RuntimeError(str(data["error"]))
                if data.get("id"):
                    response_id = data["id"]
                    if response_id != rid:
                        raise ValueError(
                            f"server request ID changed: {rid}: {response_id}"
                        )
                if data.get("usage"):
                    usage = data["usage"]
                for choice in data.get("choices", []):
                    if choice.get("finish_reason"):
                        finish = choice["finish_reason"]
        result = {
            "rid": rid,
            "response_id": response_id,
            "finish_reason": finish,
            "usage": usage,
            "wall_seconds": time.monotonic() - begun,
        }
        (root / "receipts").mkdir(exist_ok=True)
        (root / "receipts" / f"{rid}.json").write_text(json.dumps(result))
        if finish not in {"stop", "tool_calls"} or usage is None:
            raise ValueError(f"request did not naturally complete: {rid}: {finish}")
        if usage["prompt_tokens"] != config["requests"][rid]["prompt_tokens"]:
            raise ValueError(
                f"server prompt count changed: {rid}: expected "
                f"{config['requests'][rid]['prompt_tokens']}, got {usage['prompt_tokens']}"
            )
        # Writer may still be draining pinned copies after SSE completes.
        while True:
            try:
                for layer in config["layers"]:
                    state_paths(root, config, rid, layer)
                break
            except (FileNotFoundError, ValueError, EOFError, RuntimeError):
                if time.monotonic() >= deadline:
                    raise
                time.sleep(0.5)
        receipt = checkpoint_issue(root, config, result)
        if publish_root is not None and not pilot:
            publish_issue(root, publish_root, rid, cleanup=True)
        return receipt

    pool = concurrent.futures.ThreadPoolExecutor(max_workers=config["concurrency"])
    try:
        progress("running")
        if remaining:
            pilot = []
            for split in ("calibration", "held_out"):
                entry = next(
                    (
                        x
                        for x in remaining
                        if config["requests"][x["rid"]]["split"] == split
                    ),
                    None,
                )
                if entry is not None:
                    pilot.append(entry)
            for item in remaining:
                if len(pilot) >= 2:
                    break
                if item not in pilot:
                    pilot.append(item)
            pilot = pilot[:2]
            for future in concurrent.futures.as_completed(
                [pool.submit(run, x, True) for x in pilot]
            ):
                results.append(future.result())
                progress("pilot")
            for item in pilot:
                for layer in config["layers"]:
                    paths = state_paths(root, config, item["rid"], layer)[0]
                    validate_state(load_state(paths, "cuda"))
                if publish_root is not None:
                    publish_issue(root, publish_root, item["rid"], cleanup=True)
            print("C2 pilot: all 21 layers, all 64 heads, oracle PASS", flush=True)
            # A worker immediately takes the next queued issue after its
            # immutable completion is checkpointed. Never wait for a wave tail.
            for future in concurrent.futures.as_completed(
                [pool.submit(run, x) for x in remaining if x not in pilot]
            ):
                results.append(future.result())
                progress("running")
                print("EOS", results[-1]["rid"], results[-1]["usage"], flush=True)
        progress("complete")
    except BaseException:
        stop.set()
        progress("incomplete")
        raise
    finally:
        pool.shutdown(wait=True, cancel_futures=True)


def restore_sensitivity(path, config_hash, query_counts):
    """Explicit resume only; legacy checkpoints are re-oracled by the caller."""
    data = json.loads(path.read_text())
    if data.get("capture_config_sha256", config_hash) != config_hash:
        raise ValueError("sensitivity checkpoint belongs to another capture")
    if set(data["empirical"]) != set(query_counts):
        raise ValueError("checkpoint issue coverage changed")
    h = torch.tensor(data["weights"], dtype=torch.float32)
    if h.shape != (512,) or not bool(torch.isfinite(h).all()) or bool((h < 0).any()):
        raise ValueError("invalid checkpoint weights")
    torch.testing.assert_close(h.mean(), torch.tensor(1.0), atol=1e-5, rtol=1e-5)
    if not math.isfinite(data["floor"]) or data["floor"] <= 0:
        raise ValueError("invalid checkpoint magnitude floor")
    for rid, count in query_counts.items():
        item = data["empirical"][rid]
        values = np.asarray(item["r"])
        if (
            values.shape != (512,)
            or not np.isfinite(values).all()
            or (values < 0).any()
        ):
            raise ValueError("invalid checkpoint sensitivity")
        if item["row_instances"] != count * 8:
            raise ValueError("checkpoint ablation coverage changed")
    for space in ("continuous", "packed"):
        if (
            not math.isfinite(data["output_floors"][space])
            or data["output_floors"][space] <= 0
        ):
            raise ValueError("invalid checkpoint output floor")
    return data, h


def replay(root, device, deadline, *, resume=False, preload=False):
    config = json.loads((root / "capture-config.json").read_text())
    ref = reference()
    output_dir = root / "analysis"
    output_dir.mkdir(exist_ok=True)
    config_hash = hashlib.sha256(
        json.dumps(config, sort_keys=True).encode()
    ).hexdigest()
    execution = {
        "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "capture_config_sha256": config_hash,
        "resume": resume,
        "preload": preload,
        "resumed_layers": [],
    }
    if preload:
        paths = []
        for rid in config["requests"]:
            banks = [
                json.loads((root / f"rank-{rank}" / rid / "reservoir.json").read_text())
                for rank in range(4)
            ]
            if any(bank != banks[0] for bank in banks):
                raise ValueError("rank reservoir mismatch during preload")
            for layer in config["layers"]:
                for slot in banks[0]:
                    paths.extend(
                        root
                        / f"rank-{rank}"
                        / rid
                        / f"layer-{layer:02d}-slot-{int(slot):02d}.pt"
                        for rank in range(4)
                    )
        began = time.monotonic()
        execution["snapshot_cache_bytes"] = cache_snapshots(paths, deadline)
        execution["snapshot_preload_seconds"] = time.monotonic() - began
        print(
            "Snapshot preload complete:",
            len(paths),
            "files in",
            execution["snapshot_preload_seconds"],
            "seconds",
            flush=True,
        )
    (output_dir / "execution.json").write_text(json.dumps(execution, indent=2) + "\n")
    empirical, analytic, floors, checks, diagnostics, output_floors = (
        {},
        {},
        {},
        [],
        {},
        {},
    )
    train = [
        rid
        for rid in config["requests"]
        if config["requests"][rid]["split"] == "calibration"
    ]
    # Each issue contributes equally regardless of its natural output length.
    for layer in config["layers"]:
        checkpoint = output_dir / f"sensitivity-layer-{layer:02d}.json"
        if resume and checkpoint.exists():
            query_counts = {}
            for rid in config["requests"]:
                paths_by_query = state_paths(root, config, rid, layer)
                query_counts[rid] = len(paths_by_query)
                for paths in paths_by_query:
                    if time.monotonic() >= deadline:
                        raise TimeoutError("offline allocation deadline")
                    checks.append(validate_state(load_state(paths, device)))
            data, h = restore_sensitivity(checkpoint, config_hash, query_counts)
            analytic[layer], empirical[layer] = h, data["empirical"]
            floors[layer], output_floors[layer] = data["floor"], data["output_floors"]
            diagnostics[layer] = data["diagnostics"]
            execution["resumed_layers"].append(layer)
            (output_dir / "execution.json").write_text(
                json.dumps(execution, indent=2) + "\n"
            )
            print(
                "Sensitivity restored and native oracle rechecked: layer",
                layer,
                flush=True,
            )
            continue
        per_issue = {}
        squared, norms = [], {"continuous": [], "packed": []}
        for rid in config["requests"]:
            if time.monotonic() >= deadline:
                raise TimeoutError("offline allocation deadline")
            es, hs = [], []
            for paths in state_paths(root, config, rid, layer):
                if time.monotonic() >= deadline:
                    raise TimeoutError("offline allocation deadline")
                state = load_state(paths, device)
                checks.append(validate_state(state))
                raw, weight = state["raw_c"].float(), state["norm_weight"].float()
                if config["requests"][rid]["split"] == "calibration":
                    squared.append(raw.square().flatten().detach().cpu())
                rows = normalize(raw, weight, state["norm_eps"], state["csa_freqs"])
                swa = decode_native(state["swa_bytes"])
                q, sink, scale = (
                    state["q"].float(),
                    state["sink"].float(),
                    state["softmax_scale"],
                )
                baseline, p, _ = attention(q, torch.cat((swa, rows)), sink, scale)
                if rid in train:
                    norms["continuous"].append(float(baseline.norm()))
                    norms["packed"].append(float(state["o"].float().norm()))
                ids = state["csa_indices"][state["csa_valid"]].cpu().numpy()
                unique = np.unique(ids)
                rng = np.random.default_rng(42 + layer + state["ordinal"])
                targets = rng.choice(unique, min(8, len(unique)), replace=False)
                for target in targets:
                    indices = np.flatnonzero(ids == target)
                    index = int(indices[0])
                    c, freq = raw[index], state["csa_freqs"][index]
                    mass = p[:, len(swa) + torch.tensor(indices, device=device)].sum(-1)
                    hs.append(
                        analytic_diagonal(
                            c, weight, state["norm_eps"], freq, q, baseline, mass, scale
                        ).cpu()
                    )
                    errors = []
                    for begin in range(0, 512, 32):
                        if time.monotonic() >= deadline:
                            raise TimeoutError("offline allocation deadline")
                        coordinates = torch.arange(begin, begin + 32, device=device)
                        changed = c.repeat(32, 1)
                        changed[torch.arange(32, device=device), coordinates] = 0
                        transformed = normalize(
                            changed, weight, state["norm_eps"], freq
                        )
                        updated = row_update(
                            baseline, q, rows[index], transformed, mass, scale
                        )
                        errors.append((updated - baseline).square().sum((1, 2)).cpu())
                    es.append((torch.cat(errors), c.square().cpu()))
            if not es or not hs:
                raise ValueError(f"no valid sensitivity samples: {rid}, {layer}")
            per_issue[rid] = (es, torch.stack(hs).mean(0))
        positive = torch.cat(squared)
        positive = positive[positive > 0]
        floor = float(positive.median()) * 1e-6
        floors[layer] = max(floor, 1e-30)
        h = torch.stack([per_issue[rid][1] for rid in train]).mean(0)
        h = h / h.mean().clamp_min(1e-30)
        analytic[layer] = h
        empirical[layer] = {}
        output_floors[layer] = {
            space: max(float(np.median(values)) * 1e-6, 1e-30)
            for space, values in norms.items()
        }
        edges = torch.quantile(positive, torch.tensor([0.0, 0.25, 0.5, 0.75, 1.0]))
        bins, sweeps = {}, {}
        for rid, (samples, _) in per_issue.items():
            errors = torch.stack([x[0] for x in samples])
            magnitudes = torch.stack([x[1] for x in samples])
            sensitivity, valid = empirical_ratio(errors, magnitudes, floors[layer])
            empirical[layer][rid] = {
                "r": sensitivity.tolist(),
                "excluded": int((~valid).sum()),
                "row_instances": len(samples),
            }
            sweeps[rid] = {}
            for factor in (0.1, 1.0, 10.0):
                alternate, admitted = empirical_ratio(
                    errors, magnitudes, floors[layer] * factor
                )
                sweeps[rid][str(factor)] = {
                    "excluded": int((~admitted).sum()),
                    "rank_correlation_with_default": correlation(
                        sensitivity.numpy(), alternate.numpy(), True
                    ),
                }
            bins[rid] = []
            for lower, upper in zip(edges[:-1], edges[1:]):
                admitted = valid & (magnitudes >= lower) & (magnitudes <= upper)
                bins[rid].append(
                    {
                        "samples": int(admitted.sum()),
                        "nope_samples": int(admitted[:, :448].sum()),
                        "rope_samples": int(admitted[:, 448:].sum()),
                        "mean_nope_ratio": (
                            float(
                                (
                                    errors[:, :448]
                                    / (magnitudes[:, :448] + floors[layer])
                                )[admitted[:, :448]].mean()
                            )
                            if bool(admitted[:, :448].any())
                            else None
                        ),
                        "mean_rope_ratio": (
                            float(
                                (
                                    errors[:, 448:]
                                    / (magnitudes[:, 448:] + floors[layer])
                                )[admitted[:, 448:]].mean()
                            )
                            if bool(admitted[:, 448:].any())
                            else None
                        ),
                    }
                )
        diagnostics[layer] = {
            "summary": sensitivity_summary(
                empirical[layer], config["requests"], h.numpy()
            ),
            "magnitude_squared_edges": edges.tolist(),
            "matched_magnitude": bins,
            "floor_sweep": sweeps,
        }
        torch.save(h, output_dir / f"h-layer-{layer:02d}.pt")
        (output_dir / f"sensitivity-layer-{layer:02d}.json").write_text(
            json.dumps(
                {
                    "capture_config_sha256": config_hash,
                    "empirical": empirical[layer],
                    "weights": h.tolist(),
                    "floor": floors[layer],
                    "output_floors": output_floors[layer],
                    "diagnostics": diagnostics[layer],
                }
            )
            + "\n"
        )
        print("Sensitivity complete: layer", layer, flush=True)
    metrics = []
    for layer in config["layers"]:
        for rid, request in config["requests"].items():
            if request["split"] != "held_out":
                continue
            for paths in state_paths(root, config, rid, layer):
                if time.monotonic() >= deadline:
                    raise TimeoutError("offline allocation deadline")
                s = load_state(paths, device)
                raw, weight = s["raw_c"].float(), s["norm_weight"].float()
                q, sink, scale = s["q"].float(), s["sink"].float(), s["softmax_scale"]
                swa = decode_native(s["swa_bytes"])
                continuous = normalize(raw, weight, s["norm_eps"], s["csa_freqs"])
                baselines = {
                    "continuous": attention(
                        q, torch.cat((swa, continuous)), sink, scale
                    )[0],
                    "packed": attention(
                        q, torch.cat((swa, decode_native(s["csa_bytes"]))), sink, scale
                    )[0],
                }
                h = analytic[layer].to(device)
                for policy, scores in (
                    ("topmag", raw.abs()),
                    ("h_topmag", raw.abs() * h.sqrt()),
                ):
                    mask = ref.topmag_keep_mask(scores, 0.5)
                    masked = raw.masked_fill(~mask, 0)
                    crows = normalize(masked, weight, s["norm_eps"], s["csa_freqs"])
                    codes, bitmap, scales = ref.pack_rows_ref(
                        raw, mask, weight, s["norm_eps"]
                    )
                    prows = (
                        rotate(
                            ref.unpack_rows_ref(codes, bitmap, scales).float(),
                            s["csa_freqs"],
                        )
                        .to(torch.bfloat16)
                        .float()
                    )
                    for space, rows in (("continuous", crows), ("packed", prows)):
                        output = attention(q, torch.cat((swa, rows)), sink, scale)[0]
                        baseline = baselines[space]
                        if space == "packed":
                            # Consumer emits BF16, then inverse RoPE rounds again.
                            output = output.to(torch.bfloat16).float()
                            baseline = baseline.to(torch.bfloat16).float()
                        output = rotate(output, s["query_freqs"], True)
                        baseline = rotate(baseline, s["query_freqs"], True)
                        if space == "packed":
                            output, baseline = (
                                output.to(torch.bfloat16).float(),
                                baseline.to(torch.bfloat16).float(),
                            )
                        error = float((output - baseline).norm())
                        epsilon_o = output_floors[layer][space]
                        relative = error / max(float(baseline.norm()), epsilon_o)
                        cosine = float((output * baseline).sum()) / max(
                            float(output.norm() * baseline.norm()), epsilon_o**2
                        )
                        metrics.append(
                            {
                                "layer": layer,
                                "rid": rid,
                                "ordinal": s["ordinal"],
                                "space": space,
                                "policy": policy,
                                "absolute_l2": error,
                                "relative_l2": relative,
                                "cosine": cosine,
                                "near_zero_reference": float(baseline.norm())
                                < epsilon_o,
                            }
                        )
    results = {
        "status": "complete",
        "config": config,
        "sensitivity": empirical,
        "weights": {str(k): v.tolist() for k, v in analytic.items()},
        "floors": floors,
        "oracle_checks": checks,
        "metrics": metrics,
        "diagnostics": diagnostics,
        "output_floors": output_floors,
        "finite_nonnegative_diagonal_fraction": 1.0,
    }
    (output_dir / "results.json").write_text(json.dumps(results) + "\n")
    report(root)


def report(root):
    result = json.loads((root / "analysis/results.json").read_text())
    lines = [
        "# Attention-Aware Pruning on DeepSeek-V4-Flash-0731",
        "",
        "## Overview",
        "",
        "Frozen OpenHands histories were replayed through untouched TP4 native attention. "
        "All 21 CSA layers and 64 real heads were analyzed using six calibration and four held-out issues. "
        "Both policies retain 256 coordinates in the same 328-byte format. This is a local attention-error "
        "experiment, not an end-to-end coding-quality or TPOT result.",
        "",
        "## Held-out output error",
        "",
        "| Reference | TopMag mean | H-TopMag mean | Paired Δ [95% CI] | TopMag / H p95 |",
        "|---|---:|---:|---:|---:|",
    ]
    summaries = {}
    layer_rows = []
    for space in ("continuous", "packed"):
        records = [x for x in result["metrics"] if x["space"] == space]
        groups = defaultdict(list)
        for x in records:
            groups[(x["rid"], x["layer"], x["policy"])].append(x["relative_l2"])
        issue = defaultdict(lambda: defaultdict(list))
        for (rid, layer, policy), values in groups.items():
            issue[rid][policy].append(float(np.mean(values)))
        top = [np.mean(x["topmag"]) for x in issue.values()]
        weighted = [np.mean(x["h_topmag"]) for x in issue.values()]
        delta = np.subtract(weighted, top)
        ci = bootstrap(delta)
        p95 = [
            np.quantile(
                [x["relative_l2"] for x in records if x["policy"] == policy], 0.95
            )
            for policy in ("topmag", "h_topmag")
        ]
        summaries[space] = {
            "topmag": float(np.mean(top)),
            "h_topmag": float(np.mean(weighted)),
            "delta": float(delta.mean()),
            "ci": ci,
            "p95": [float(x) for x in p95],
            "issue_deltas": dict(zip(issue, delta.tolist())),
        }
        for layer in result["config"]["layers"]:
            a = [np.mean(groups[(rid, layer, "topmag")]) for rid in issue]
            b = [np.mean(groups[(rid, layer, "h_topmag")]) for rid in issue]
            layer_delta = np.subtract(b, a)
            layer_rows.append(
                {
                    "space": space,
                    "layer": layer,
                    "topmag": float(np.mean(a)),
                    "h_topmag": float(np.mean(b)),
                    "delta": float(layer_delta.mean()),
                    "ci_low": bootstrap(layer_delta)[0],
                    "ci_high": bootstrap(layer_delta)[1],
                }
            )
        lines.append(
            f"| {space} | {np.mean(top):.6f} | {np.mean(weighted):.6f} | {delta.mean():+.6f} [{ci[0]:+.6f}, {ci[1]:+.6f}] | {p95[0]:.6f} / {p95[1]:.6f} |"
        )
    with (root / "analysis/layers.csv").open("w") as file:
        writer = csv.DictWriter(file, fieldnames=list(layer_rows[0]))
        writer.writeheader()
        writer.writerows(layer_rows)
    with (root / "analysis/queries.csv").open("w") as file:
        writer = csv.DictWriter(file, fieldnames=list(result["metrics"][0]))
        writer.writeheader()
        writer.writerows(result["metrics"])
    lines.extend(
        [
            "",
            "The means first balance queries within a layer, then layers within an issue, then the four issues. "
            "The p95 values pool held-out query/layer states and describe tails; they are not independent samples. "
            "Continuous comparisons isolate pruning. Packed comparisons include masked RMSNorm, E4M3/UE8M0 "
            "rounding, BF16 reconstruction and inverse-RoPE rounding against native-byte attention.",
            "",
            "## Sensitivity",
            "",
            "| CSA layer | Held-out p90/p10 | Spearman mean [95% CI] |",
            "|---|---:|---:|",
        ]
    )
    for layer in (2, 22, 42):
        diagnostic = result["diagnostics"][str(layer)]["summary"]["held_out"]
        spread = diagnostic["regions"]["all"]["p90_p10"]
        corr = diagnostic["correlations"]["spearman"]
        text = (
            f"{corr['mean']:.3f} [{corr['ci'][0]:.3f}, {corr['ci'][1]:.3f}]"
            if corr["mean"] is not None
            else "undefined"
        )
        lines.append(f"| {layer} | {spread:.2f}× | {text} |")
    packed = summaries["packed"]
    regressions = sum(x["delta"] > 0 for x in layer_rows if x["space"] == "packed")
    tails_regress = packed["p95"][1] > packed["p95"][0]
    verdict = (
        "promising"
        if all(summaries[s]["ci"][1] < 0 for s in summaries)
        and not regressions
        and not tails_regress
        else "inconclusive"
        if packed["delta"] < 0
        else "not beneficial"
    )
    summaries.update(
        assessment=verdict,
        packed_layer_regressions=regressions,
        packed_tail_regression=bool(tails_regress),
    )
    lines.extend(
        [
            "",
            "## Assessment",
            "",
            f"**{verdict.capitalize()}.** H-TopMag raises the packed mean error in {regressions}/21 layers; "
            f"the packed p95 {'increases' if tails_regress else 'does not increase'}. "
            "The paired change is H-TopMag minus TopMag; negative favors attention-aware pruning. "
            "Four held-out issues give limited statistical strength. The bootstrap resamples issues, "
            "not correlated heads, layers, or adjacent decode tokens. Layer and tail regressions must "
            "be reviewed before promotion; no speedup or closed-loop task-success claim follows from this replay.",
            "",
            "Weights were fitted only on calibration issues, using the full K=V derivative through learned "
            "RMSNorm, positional rotation, logits and values. Coordinate deletions recompute RMSNorm and "
            "attention probabilities, including every duplicate occurrence. Analytic derivatives passed "
            "central finite-difference step sweeps before fitting. Calibration-defined denominator floors, "
            "matched-magnitude quartiles and floor-sensitivity checks are retained with the per-layer results. "
            "The table shows representative early/middle/late layers; all 21 were tested.",
            "",
        ]
    )
    coverage = {}
    for rid, request in result["config"]["requests"].items():
        bank = json.loads((root / "rank-0" / rid / "reservoir.json").read_text())
        coverage[rid] = {
            "queries": len(bank),
            "quota": request["quota"],
            "ordinals": sorted(bank.values()),
        }
    summaries["coverage"] = coverage
    lines.extend(
        [
            "## Execution and limitations",
            "",
            f"Captured {sum(x['queries'] for x in coverage.values())} query states per layer across ten natural-EOS "
            "continuations, using a deterministic reservoir over each complete continuation. A C2 pilot "
            "preceded the eight remaining requests at up to C8 within a C10 server limit. Capture disabled "
            "graphs, overlap scheduling and radix reuse; its elapsed time is not a serving benchmark. "
            "No historical tool actions were executed. Queries, selections, SWA rows and future states "
            "remain frozen, so this experiment cannot measure feedback from changed model outputs.",
            "",
            "A favorable local result would justify more independent trajectories followed by end-to-end "
            "quality validation. A directional result with confidence intervals overlapping zero is not "
            "grounds to replace TopMag50. Both policies store exactly 256 retained values per row, so any "
            "quality gain here has no storage or throughput gain by itself.",
        ]
    )
    allocation = (
        json.loads((root / "allocation.json").read_text())
        if (root / "allocation.json").exists()
        else {}
    )
    offline = (
        json.loads((root / "offline-allocation.json").read_text())
        if (root / "offline-allocation.json").exists()
        else {}
    )
    if allocation and offline:
        lines.extend(
            [
                "",
                f"Observed worker cost: capture {allocation['worker_wall_seconds']:.1f} s / "
                f"{allocation['observed_h100_hours']:.3f} H100-hours; offline {offline['worker_wall_seconds']:.1f} s / "
                f"{offline['observed_l4_hours']:.3f} L4-hours. Startup is included; platform provisioning and "
                "CPU-image build are billed separately.",
            ]
        )
    (root / "analysis/summary.json").write_text(json.dumps(summaries, indent=2) + "\n")
    (root / "analysis/report.md").write_text("\n".join(lines) + "\n")
    # Small, dependency-free scientific figure; full coordinates remain in JSON.
    svg = [
        '<svg xmlns="http://www.w3.org/2000/svg" width="900" height="570" viewBox="0 0 900 570">',
        '<rect width="900" height="570" fill="white"/>',
    ]
    for panel, layer in enumerate((2, 22, 42)):
        y0 = 35 + panel * 180
        issues = result["sensitivity"][str(layer)]
        empirical = np.mean(
            [
                v["r"]
                for rid, v in issues.items()
                if result["config"]["requests"][rid]["split"] == "calibration"
            ],
            0,
        )
        empirical /= max(empirical.mean(), 1e-30)
        predicted = np.asarray(result["weights"][str(layer)])
        upper = max(float(empirical.max()), float(predicted.max()), 1)
        svg.extend(
            [
                f'<text x="50" y="{y0}" font-size="15">CSA {layer}: calibration sensitivity, mean normalized</text>',
                f'<path d="M50 {y0 + 140}H850 M50 {y0 + 15}V{y0 + 140}" stroke="black" fill="none"/>',
                f'<path d="M{50 + 448 / 511 * 800:.1f} {y0 + 15}V{y0 + 140}" stroke="gray" stroke-dasharray="4 4"/>',
            ]
        )
        for vector, color in ((empirical, "#137b51"), (predicted, "#315ac3")):
            points = " ".join(
                f"{50 + i / 511 * 800:.1f},{y0 + 140 - v / upper * 120:.1f}"
                for i, v in enumerate(vector)
            )
            svg.append(
                f'<polyline points="{points}" fill="none" stroke="{color}" stroke-width="1"/>'
            )
        svg.append(
            f'<text x="50" y="{y0 + 158}" font-size="12">0 — NoPE — 447 | RoPE — 511. Green: deletion; blue: analytic; y max {upper:.2f}.</text>'
        )
    svg.append("</svg>")
    (root / "analysis/sensitivity.svg").write_text("\n".join(svg))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command",
        choices=(
            "preflight",
            "capture-preflight",
            "capture-client",
            "replay",
            "report",
            "archive",
            "audit-shared",
        ),
    )
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--cohort", type=Path)
    parser.add_argument("--model", type=Path)
    parser.add_argument("--endpoint", default="http://127.0.0.1:30000")
    parser.add_argument("--port", type=int, default=30312)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seconds", type=int, default=1800)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--preload", action="store_true")
    parser.add_argument("--storage-schema", type=int, choices=(1, 2), default=1)
    parser.add_argument("--audit-shared", action="store_true")
    parser.add_argument("--publish-root", type=Path)
    parser.add_argument("--scratch-root", type=Path)
    args = parser.parse_args()
    torch.set_num_threads(4)
    deadline = time.monotonic() + args.seconds
    if args.command == "audit-shared":
        if args.scratch_root is None:
            parser.error("audit-shared requires an explicit --scratch-root")
        print(json.dumps(audit_published_capture(args.root, args.scratch_root)))
    elif args.command == "preflight":
        preflight(
            args.cohort, args.model, args.root, args.storage_schema, args.audit_shared
        )
    elif args.command == "capture-preflight":
        if args.model is None or args.publish_root is None:
            parser.error("capture-preflight requires --model and --publish-root")
        capture_preflight(args.root, args.publish_root, args.model, args.port)
    elif args.command == "capture-client":
        capture_client(args.root, args.endpoint, deadline, args.publish_root)
    elif args.command == "replay":
        replay(
            args.root, args.device, deadline, resume=args.resume, preload=args.preload
        )
    elif args.command == "archive":
        print(archive_capture(args.root), flush=True)
    else:
        report(args.root)


if __name__ == "__main__":
    main()
