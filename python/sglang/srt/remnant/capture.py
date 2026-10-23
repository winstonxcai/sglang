"""Opt-in, CPU-backed native attention observations. Never changes model tensors.

Author: Winston Cai.
"""

from __future__ import annotations

import atexit
import hashlib
import json
import os
import queue
import threading
from pathlib import Path

import torch

_DIRECTORY = os.environ.get("REMNANT_CAPTURE_DIR")
_OBSERVER = None

SHARED_FIELDS = (
    "raw_c",
    "raw_positions",
    "csa_bytes",
    "swa_bytes",
    "norm_weight",
    "norm_eps",
    "csa_freqs",
    "query_freqs",
)


def atomic_tensor(path, data):
    """Publish a complete CPU payload; caller owns the writer thread."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    torch.save(data, temporary)
    temporary.replace(path)


def file_reference(root, path):
    return {
        "path": str(path.relative_to(root)),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def write_shared_snapshot(root, path, data, slot, audit=False):
    """Schema 2: owner writes common data, each rank writes its own metadata.

    Relative references are resolved only inside the capture root. Shared row
    ordering belongs to rank 0; readers align each rank's admitted selections.
    """
    root = Path(root)
    common_path = (
        root / "shared" / data["rid"] / f"layer-{data['layer']:02d}-slot-{slot:02d}.pt"
    )
    constants_path = root / "shared" / f"layer-{data['layer']:02d}-constants.pt"
    record = {k: v for k, v in data.items() if k not in SHARED_FIELDS}
    if data["rank"] == 0:
        constants = {k: data[k] for k in ("norm_weight", "norm_eps")}
        if constants_path.exists():
            old = torch.load(constants_path, map_location="cpu", weights_only=True)
            if old["norm_eps"] != constants["norm_eps"] or not torch.equal(
                old["norm_weight"], constants["norm_weight"]
            ):
                raise ValueError("capture layer constants changed")
        else:
            atomic_tensor(constants_path, constants)
        common = {k: data[k] for k in SHARED_FIELDS if k not in constants}
        common.update(
            {
                k: data[k]
                for k in (
                    "rid",
                    "layer",
                    "ordinal",
                    "query_position",
                    "csa_indices",
                    "swa_indices",
                    "csa_valid",
                    "swa_valid",
                )
            }
        )
        common["constants"] = file_reference(root, constants_path)
        atomic_tensor(common_path, common)
        record["shared_sha256"] = file_reference(root, common_path)["sha256"]
    elif audit:
        record["audit_shared"] = {k: data[k] for k in SHARED_FIELDS}
    record.update(
        storage_schema_version=2,
        shared_path=str(common_path.relative_to(root)),
        audit_required=bool(audit),
    )
    atomic_tensor(path, record)


def enabled():
    return bool(_DIRECTORY)


def plan_writes(plan, locations):
    """Native norm/store mapping, not the compressor state-ring WritePlan."""
    words = plan[1].detach().cpu().contiguous().view(torch.int32).reshape(-1, 4)
    locations = locations.detach().cpu().long()
    result = []
    for row, words_row in enumerate(words.tolist()):
        length, word1 = words_row[:2]
        if length < 0 or length % 4:
            continue
        token = row if plan.is_decode else word1 & 0xFFFF
        loc = int(locations[token])
        if loc >= 0:
            result.append((row, token, loc, length - 4))
    return result


def sample_priority(rid, ordinal, seed=42):
    return int.from_bytes(
        hashlib.blake2b(f"{seed}:{rid}:{ordinal}".encode(), digest_size=8).digest(),
        "big",
    )


def native_records(cache, locations, page_size):
    """Gather actual page data and scale blocks into logical 584-byte records."""
    cache = cache.view(torch.uint8)
    loc = locations.long()
    page, offset = loc // page_size, loc % page_size
    payload = cache[
        page[:, None], offset[:, None] * 576 + torch.arange(576, device=cache.device)
    ]
    scales = cache[
        page[:, None],
        page_size * 576 + offset[:, None] * 8 + torch.arange(8, device=cache.device),
    ]
    return torch.cat((payload, scales), -1)


class HostCopy:
    def __init__(self, tensor):
        if tensor.is_cuda:
            # Retain the source until the stream-ordered D2H copy finishes.
            self.source = tensor.detach().contiguous()
            self.value = torch.empty(
                self.source.shape, dtype=self.source.dtype, pin_memory=True
            )
            self.value.copy_(self.source, non_blocking=True)
            self.event = torch.cuda.Event()
            self.event.record()
        else:
            self.source = None
            self.value = tensor.detach().clone()
            self.event = None

    def wait(self):
        if self.event is not None:
            self.event.synchronize()
        self.source = None
        return self.value


def _atomic_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(data, indent=2) + "\n")
    temporary.replace(path)


class Observer:
    def __init__(self, root):
        self.root = Path(root)
        self.config = json.loads((self.root / "capture-config.json").read_text())
        self.schema = self.config.get("storage_schema_version", 1)
        if self.schema not in (1, 2):
            raise ValueError("unsupported capture storage schema")
        if self.schema == 2 and not isinstance(self.config.get("audit_shared"), bool):
            raise ValueError("schema 2 requires an explicit audit_shared policy")
        self.layers = set(self.config["layers"])
        self.rank = (
            torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
        )
        self.keep_shared = (
            self.schema == 1 or self.rank == 0 or self.config.get("audit_shared", False)
        )
        self.base = self.root / f"rank-{self.rank}"
        self.base.mkdir(parents=True, exist_ok=True)
        self.shadow = {}
        self.physical_owners = {}
        self.layer_data = {}
        self.banks = {}
        self.decisions = {}
        self.pending = {}
        self.generation = 0
        self.error = None
        self.jobs = queue.Queue(maxsize=2)
        self.writer = threading.Thread(target=self._write, daemon=True)
        self.writer.start()
        atexit.register(self.flush)

    def check(self):
        if self.error is not None:
            raise RuntimeError("attention capture writer failed") from self.error

    def flush(self):
        self.jobs.join()
        self.check()

    def _write(self):
        while True:
            job = self.jobs.get()
            try:
                path, data, raw = job
                if path is None:
                    data.wait()
                    continue
                for key, value in list(data.items()):
                    if isinstance(value, HostCopy):
                        data[key] = value.wait()
                if raw is not None:
                    data["raw_c"] = (
                        torch.stack([copy.wait()[row] for copy, row in raw])
                        if raw
                        else torch.empty(
                            (0, 512),
                            dtype=torch.bfloat16 if self.schema == 1 else torch.float32,
                        )
                    )
                if self.schema == 2:
                    slot = int(path.stem.rsplit("-", 1)[1])
                    write_shared_snapshot(
                        self.root, path, data, slot, self.config["audit_shared"]
                    )
                else:
                    atomic_tensor(path, data)
            except BaseException as error:
                self.error = error
            finally:
                self.jobs.task_done()

    def _owners(self, batch):
        if batch.rids is None:
            raise RuntimeError("capture needs explicit ForwardBatch request IDs")
        if batch.forward_mode.is_decode():
            return batch.rids
        lengths = batch.extend_seq_lens_cpu
        if lengths is None:
            raise RuntimeError("capture needs chunked-prefill ragged lengths")
        return [rid for rid, length in zip(batch.rids, lengths) for _ in range(length)]

    def _is_capture_batch(self, batch):
        # BaseRunner's autotune/dummy batches have neither request IDs nor
        # sampling metadata. Real ScheduleBatch conversions populate both.
        # Reject missing IDs on a real batch rather than hiding lost capture.
        if batch.rids is None:
            if getattr(batch, "sampling_info", None) is not None:
                raise RuntimeError("capture needs explicit ForwardBatch request IDs")
            return False
        return any(rid in self.config["requests"] for rid in batch.rids)

    def observe_raw(self, layer, batch, raw, plan, locations, weight, eps, frequencies):
        self.check()
        if layer not in self.layers or not self._is_capture_batch(batch):
            return
        from .config import packed_enabled

        if packed_enabled():
            raise RuntimeError("calibration capture requires untouched native storage")
        if not self.keep_shared:
            return
        # Finished requests may disappear from a batch without finishing peers.
        done = self.root / "done"
        if done.exists():
            for marker in done.glob("*.json"):
                rid = marker.stem
                self.shadow.pop((rid, layer), None)
        owners = self._owners(batch)
        writes = plan_writes(plan, locations)
        if not writes:
            return
        copy = HostCopy(raw)
        self.jobs.put((None, copy, None))
        if layer not in self.layer_data:
            self.layer_data[layer] = (HostCopy(weight).wait(), float(eps), frequencies)
        for row, token, physical, position in writes:
            rid = owners[token]
            if rid not in self.config["requests"]:
                continue
            self.generation += 1
            self.shadow.setdefault((rid, layer), {})[physical] = (
                position,
                self.generation,
                copy,
                row,
            )
            self.physical_owners[(layer, physical)] = (rid, self.generation)

    def consider(self, rid, ordinal):
        previous = self.decisions.get(rid)
        if previous is not None and previous[0] == ordinal:
            return previous[1]
        quota = self.config["requests"][rid]["quota"]
        priority = sample_priority(rid, ordinal, self.config["seed"])
        bank = self.banks.setdefault(rid, {})
        slot = None
        if len(bank) < quota:
            slot = len(bank)
        else:
            worst = max(bank, key=lambda key: bank[key][0])
            if priority < bank[worst][0]:
                slot = worst
        if slot is not None:
            bank[slot] = (priority, ordinal)
            _atomic_json(
                self.base / rid / "reservoir.json",
                {str(k): v[1] for k, v in bank.items()},
            )
        self.decisions[rid] = (ordinal, slot)
        return slot

    def observe_attention(
        self,
        layer,
        batch,
        q,
        swa_cache,
        csa_cache,
        swa_indices,
        csa_indices,
        swa_lengths,
        csa_lengths,
        sink,
        scale,
        output,
        swa_page_size,
        csa_page_size,
        scheduler,
        lse=None,
    ):
        self.check()
        if (
            layer not in self.layers
            or not self._is_capture_batch(batch)
            or not batch.forward_mode.is_decode()
        ):
            return
        positions = batch.positions.detach().cpu().tolist()
        for index, rid in enumerate(batch.rids):
            if rid not in self.config["requests"]:
                continue
            ordinal = positions[index] - self.config["requests"][rid]["prompt_tokens"]
            if ordinal < 0:
                raise RuntimeError("invalid capture decode ordinal")
            slot = self.consider(rid, ordinal)
            if slot is None:
                continue
            cidx = csa_indices[index].reshape(-1).detach().cpu()
            sidx = swa_indices[index].reshape(-1).detach().cpu()
            clen, slen = int(csa_lengths[index]), int(swa_lengths[index])
            cvalid = (torch.arange(cidx.numel()) < clen) & (cidx != -1)
            svalid = (torch.arange(sidx.numel()) < slen) & (sidx != -1)
            if bool(
                (
                    (cidx[cvalid] < 0)
                    | (cidx[cvalid] >= csa_cache.shape[0] * csa_page_size)
                ).any()
            ):
                raise RuntimeError("out-of-range admitted native CSA location")
            if bool(
                (
                    (sidx[svalid] < 0)
                    | (sidx[svalid] >= swa_cache.shape[0] * swa_page_size)
                ).any()
            ):
                raise RuntimeError("out-of-range admitted native SWA location")
            records = self.shadow.get((rid, layer), {})
            entries = []
            for physical in cidx[cvalid].tolist() if self.keep_shared else []:
                if physical not in records:
                    raise RuntimeError(
                        f"missing raw row: {rid}, layer {layer}, physical {physical}"
                    )
                if self.physical_owners.get((layer, physical)) != (
                    rid,
                    records[physical][1],
                ):
                    raise RuntimeError(
                        f"stale raw row: {rid}, layer {layer}, physical {physical}"
                    )
                entries.append(records[physical])
            rope_positions = torch.tensor(
                [entry[0] for entry in entries], device=q.device, dtype=torch.long
            )
            heads = self.config["heads_per_rank"]
            data = {
                "rid": rid,
                "layer": layer,
                "rank": self.rank,
                "ordinal": ordinal,
                "query_position": positions[index],
                "heads": heads,
                "softmax_scale": float(scale),
                "csa_indices": cidx,
                "swa_indices": sidx,
                "csa_valid": cvalid,
                "swa_valid": svalid,
                "csa_length": clen,
                "swa_length": slen,
                "csa_page_size": csa_page_size,
                "swa_page_size": swa_page_size,
                "csa_page_stride_bytes": csa_cache.stride(0) * csa_cache.element_size(),
                "swa_page_stride_bytes": swa_cache.stride(0) * swa_cache.element_size(),
                "row_generations": [entry[1] for entry in entries],
                "q": HostCopy(q[index].reshape(-1, q.shape[-1])[:heads].clone()),
                "sink": HostCopy(sink[:heads].clone()),
                "bar_o": HostCopy(output[index, :heads].clone()),
            }
            if self.keep_shared:
                weight, eps, frequencies = self.layer_data[layer]
                data.update(
                    raw_positions=rope_positions.cpu(),
                    norm_weight=weight,
                    norm_eps=eps,
                    csa_bytes=HostCopy(
                        native_records(
                            csa_cache.reshape(csa_cache.shape[0], -1),
                            cidx[cvalid].to(q.device),
                            csa_page_size,
                        )
                    ),
                    swa_bytes=HostCopy(
                        native_records(
                            swa_cache.reshape(swa_cache.shape[0], -1),
                            sidx[svalid].to(q.device),
                            swa_page_size,
                        )
                    ),
                    csa_freqs=HostCopy(frequencies[rope_positions].clone()),
                )
            # Retain tensor metadata without serializing mutable scheduler objects.
            data["scheduler"] = {
                key: HostCopy(value.clone())
                for key, value in vars(scheduler).items()
                if isinstance(value, torch.Tensor)
            }
            if lse is not None:
                data["lse"] = HostCopy(lse[index].reshape(-1)[:heads].clone())
            self.pending[(rid, layer)] = (
                slot,
                data,
                (
                    [(entry[2], entry[3]) for entry in entries]
                    if self.keep_shared
                    else None
                ),
            )

    def observe_output(self, layer, batch, output, frequencies, positions, rank, heads):
        self.check()
        if (
            layer not in self.layers
            or not self._is_capture_batch(batch)
            or not batch.forward_mode.is_decode()
        ):
            return
        if rank != self.rank or heads != self.config["heads_per_rank"]:
            raise RuntimeError("unexpected attention head partition")
        for index, rid in enumerate(batch.rids):
            pending = self.pending.pop((rid, layer), None)
            if pending is None:
                continue
            slot, data, raw = pending
            data["o"] = HostCopy(output[index].clone())
            if self.keep_shared:
                data["query_freqs"] = HostCopy(frequencies[positions[index]].clone())
            path = self.base / rid / f"layer-{layer:02d}-slot-{slot:02d}.pt"
            path.parent.mkdir(parents=True, exist_ok=True)
            # Scheduler metadata is descriptive; materialize it before enqueue.
            data["scheduler"] = {
                key: value.wait() for key, value in data["scheduler"].items()
            }
            self.jobs.put((path, data, raw))


def observer():
    global _OBSERVER
    if not enabled():
        return None
    if _OBSERVER is None:
        _OBSERVER = Observer(_DIRECTORY)
    return _OBSERVER
