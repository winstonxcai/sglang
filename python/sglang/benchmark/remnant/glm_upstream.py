"""Pinned, unedited upstream GLM attention baseline and CPU artifacts.

Author: Winston Cai.
Consumers: glm_attention fixtures and the maintained Modal CPU preparation.
This module changes no production dispatch and imports CUDA only when loading
or compiling the baseline inside the verified image.
"""

from __future__ import annotations

import hashlib
import importlib.util
import inspect
import json
import os
import re
import subprocess
import sys
from functools import cache
from pathlib import Path

COMMIT = "baa9befa381511cd4d56f83980951e5a45fffc6d"
MODULE = "kernels/ops/attention/dsa/tilelang_kernel.py"
FILES = {
    MODULE: "3d288dc04676b304e1a4d1a44703a8281762ebe789ca7cdd745b0c80fcac591e",
    "kernels/ops/quantization/fp8_kernel.py": "f5b428f2cb51b7707b19dd7b55e9ef61818a5f99b1cb382cd5e811429dd46806",
    "srt/utils/__init__.py": "d4c887ce6d68fb14d2f870c50dabc621da1c67a85823e8c8cf704392865963c8",
    "srt/utils/common.py": "5423d1bb153e72852306934ab56ee0b3eccab8b3ec6395dc9da0fea79f7e74da",
}
SOURCE_ENV = "REMNANT_GLM_UPSTREAM_SOURCE"
IMAGE_SOURCE = "/opt/remnant-glm-upstream-v0522"
WGMMA_INNER = {8: 1, 16: 2, 24: 3, 32: 4, 64: 8}


def identity() -> dict:
    return {"commit": COMMIT, "files": dict(FILES), "scope": "attention-module"}


def prepare_source(checkout: Path, destination: Path) -> dict:
    """Export exact Git blobs locally; the image needs neither Git nor network."""
    destination.mkdir(parents=True, exist_ok=True)
    for name, expected in FILES.items():
        data = subprocess.check_output(
            ["git", "-C", str(checkout), "show", f"{COMMIT}:python/sglang/{name}"],
            timeout=30,
        )
        if hashlib.sha256(data).hexdigest() != expected:
            raise RuntimeError(f"Pinned upstream source mismatch: {COMMIT}/{name}")
        if name == MODULE:
            (destination / "tilelang_kernel.py").write_bytes(data)
    (destination / "source.json").write_text(json.dumps(identity(), indent=2) + "\n")
    verify_source(destination, checkout / "python/sglang")
    return identity()


def verify_source(directory: Path, package: Path) -> dict:
    """Check source and shared helper bytes without importing Torch or CUDA."""
    if json.loads((directory / "source.json").read_text()) != identity():
        raise RuntimeError("Upstream baseline receipt changed")
    for name, expected in FILES.items():
        path = directory / "tilelang_kernel.py" if name == MODULE else package / name
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise RuntimeError(f"Upstream baseline/helper changed: {path}")
    return identity()


@cache
def load_baseline():
    """Load an independently named upstream module, never the fork Native arm."""
    package = Path(__file__).resolve().parents[2]
    directory = Path(os.environ[SOURCE_ENV])
    verify_source(directory, package)
    path = directory / "tilelang_kernel.py"
    name = "_remnant_pristine_glm_tilelang_v0522"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
        for helper, relative in (
            (module.is_fp8_fnuz, "kernels/ops/quantization/fp8_kernel.py"),
            (module.is_hip, "srt/utils/common.py"),
            (module.is_gfx95_supported, "srt/utils/common.py"),
        ):
            if (
                Path(inspect.getsourcefile(inspect.unwrap(helper))).resolve()
                != (package / relative).resolve()
            ):
                raise RuntimeError(f"Upstream helper import was redirected: {relative}")
        if (
            Path(inspect.getsourcefile(module.tilelang_sparse_fwd)).resolve()
            != path.resolve()
        ):
            raise RuntimeError("Upstream attention import was redirected")
    except BaseException:
        sys.modules.pop(name, None)
        raise
    return module


def wgmma_shared_ranges(source: str) -> dict[str, tuple[int, int]]:
    """Read the generated allocation offsets and reject asynchronous aliases."""
    sizes = dict(q=16 * 512, kv=64 * 512, v_t=512 * 64, p=16 * 64, scratch=32 * 272)
    offsets = dict(
        re.findall(
            r"void\* (q|kv|v_t|p|scratch) = .*buf_dyn_shmem \+ (\d+)\)\);",
            source,
        )
    )
    if offsets.keys() != sizes.keys():
        raise RuntimeError("Cannot verify WGMMA shared-memory offsets")
    ranges = {
        name: (int(offset), int(offset) + sizes[name])
        for name, offset in offsets.items()
    }
    ordered = sorted(ranges.values())
    if any(end > start for (_, end), (start, _) in zip(ordered, ordered[1:])):
        raise RuntimeError(f"Aliased WGMMA shared-memory operands: {ranges}")
    return ranges


def precompile() -> dict:
    """Compile only heads16 decode partial/combine variants on CPU/SM90a."""
    from sglang.benchmark.remnant.glm_artifacts import (
        bind_decode_library,
        bind_partial_launch_metadata,
    )
    from sglang.kernels.ops.attention.dsa.tilelang_kernel import (
        glm_packed_decode_combine,
        glm_packed_wgmma_partial,
    )

    required = ("TRITON_CACHE_DIR", "TILELANG_CACHE_DIR", SOURCE_ENV)
    if any(not os.environ.get(name) for name in required):
        raise RuntimeError(f"Upstream CPU preparation requires {required}")
    os.environ.setdefault(
        "TILELANG_DEFAULT_TARGET", json.dumps({"kind": "cuda", "arch": "sm_90a"})
    )
    module = load_baseline()
    variants = []
    for inner in (1, 2):
        partial = module.sparse_mla_fwd_decode_partial_fp8(
            16,
            512,
            0,
            2112,
            sm_scale=256**-0.5,
            block_I=32,
            inner_iter=inner,
            threads=128,
        )
        variants.append(
            dict(
                heads=16,
                inner=inner,
                packed=False,
                block_i=32,
                source_sha256=hashlib.sha256(
                    partial.get_kernel_source().encode()
                ).hexdigest(),
            )
        )
        module.sparse_mla_fwd_decode_combine(
            16,
            512,
            (66 // inner) * 32,
            head_per_block=4,
            block_I=32,
            threads=128,
        )
    wgmma = {}
    for batch, inner in WGMMA_INNER.items():
        groups = (33 + inner - 1) // inner
        partial = glm_packed_wgmma_partial(inner, 256**-0.5)
        combine = glm_packed_decode_combine(16, 512, groups, 512)
        wgmma[str(batch)] = partial, combine
    compiled = dict(
        source=identity(),
        tilelang=variants,
        cache_files={
            str(path): hashlib.sha256(path.read_bytes()).hexdigest()
            for root in (Path(os.environ[name]) for name in required[:2])
            for path in sorted(root.rglob("*"))
            if path.is_file()
        },
    )
    bind_partial_launch_metadata(compiled)
    artifacts = compiled["cache_files"]
    compiled["decode_wgmma"] = {}
    for batch, (partial_kernel, combine) in wgmma.items():
        partial = partial_kernel.get_kernel_source()
        variant = bind_decode_library(partial_kernel, artifacts)
        shared_ranges = wgmma_shared_ranges(partial)
        loader = next(
            (line.strip() for line in partial.splitlines()
             if "remnant_glm_expand_wide_tile<64, true>(" in line),
            None,
        )
        if loader is None:
            raise RuntimeError(f"Missing Packed vector expansion: {batch=}")
        sass = subprocess.check_output(
            ["cuobjdump", "--dump-sass", variant["cubin_path"]], text=True, timeout=30
        )
        # Hopper encodes FP8 warpgroup MMA as QGMMA.
        qgmma = len(re.findall(r"\bQGMMA\.", sass))
        if qgmma < 32:
            raise RuntimeError(f"WGMMA partial has only {qgmma} QGMMA sites: {batch=}")
        barriers = len(re.findall(r"\bBAR\.SYNC\b", sass))
        if barriers > 24:
            raise RuntimeError(f"WGMMA partial emitted {barriers} CTA barriers: {batch=}")
        compiled["decode_wgmma"][batch] = dict(
            partial=variant,
            combine=bind_decode_library(combine, artifacts),
            qgmma_sites=qgmma,
            cta_barrier_sites=barriers,
            local_sites=len(re.findall(r"\b(?:LDL|STL)\b", sass)),
            loader_call=loader,
            shared_ranges=shared_ranges,
            resource_usage=subprocess.check_output(
                ["cuobjdump", "--dump-resource-usage", variant["cubin_path"]],
                text=True, timeout=30,
            ).strip(),
        )
    return compiled


def verify_compilation(compiled: dict) -> None:
    """Fail before timing on absent, changed or incomplete cached artifacts."""
    if compiled.get("source") != identity():
        raise RuntimeError("Upstream compiled source identity changed")
    variants = compiled.get("tilelang", [])
    if {(v["heads"], v["inner"], v["packed"]) for v in variants} != {
        (16, 1, False),
        (16, 2, False),
    } or len(variants) != 2:
        raise RuntimeError("Missing upstream heads16 SM90 variants")
    artifacts = compiled.get("cache_files", {})
    for variant in variants:
        path, digest = variant.get("cubin_path"), variant.get("cubin_sha256")
        if (
            not isinstance(path, str)
            or not path
            or not isinstance(digest, str)
            or len(digest) != 64
            or any(c not in "0123456789abcdef" for c in digest)
            or artifacts.get(path) != digest
        ):
            raise RuntimeError("Unbound upstream cubin")
    if not artifacts:
        raise RuntimeError("Missing upstream compiled artifacts")
    if wgmma := compiled.get("decode_wgmma"):
        if set(wgmma) != {str(batch) for batch in WGMMA_INNER}:
            raise RuntimeError("Missing WGMMA decode specializations")
        for batch, record in wgmma.items():
            for kind in ("partial", "combine"):
                variant = record[kind]
                if any(
                    not variant.get(f"{artifact}_sha256")
                    or artifacts.get(variant.get(f"{artifact}_path"))
                    != variant[f"{artifact}_sha256"]
                    for artifact in ("cubin", "library")
                ):
                    raise RuntimeError(f"Unbound WGMMA decode {kind}: {batch=}")
            if record["qgmma_sites"] < 32:
                raise RuntimeError(f"Missing CPU-verified WGMMA instructions: {batch=}")
    for name, expected in artifacts.items():
        path = Path(name)
        if (
            not path.is_file()
            or hashlib.sha256(path.read_bytes()).hexdigest() != expected
        ):
            raise RuntimeError(f"Upstream compiled artifact changed: {path}")


if __name__ == "__main__":
    record = precompile()
    Path("/opt/remnant-glm-upstream-compile.json").write_text(
        json.dumps(record, indent=2) + "\n"
    )
