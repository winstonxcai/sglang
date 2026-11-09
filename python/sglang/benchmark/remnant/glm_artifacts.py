"""CPU-only SM90 artifact preparation shared by the GLM tests and build runner.

Author: Winston Cai.

Run with explicit TRITON_CACHE_DIR, TILELANG_CACHE_DIR and
REMNANT_GLM_COMPILE_MANIFEST paths:
    python -m sglang.benchmark.remnant.glm_artifacts
No CUDA tensors or device context are created. The resulting manifest includes
verified cubin paths, hashes, generated launch ABI and shared-memory sizes.
"""

import hashlib
import json
import os
import re
from pathlib import Path
from typing import TypedDict

from typing_extensions import NotRequired

ATTENTION_SCALE = 256**-0.5


class CudaParameter(TypedDict):
    name: str
    pointer: bool
    type: str


class PartialVariant(TypedDict):
    heads: int
    inner: int
    packed: bool
    block_i: NotRequired[int]
    source_sha256: str
    cubin_path: NotRequired[str]
    cubin_sha256: NotRequired[str]
    symbol: NotRequired[str]
    parameters: NotRequired[list[CudaParameter]]
    dynamic_shared: NotRequired[int]
    libraries: NotRequired[dict[str, str]]


class CompileManifest(TypedDict):
    attention_scale: float
    precompile_workers: int
    triton: list[str]
    tilelang: list[PartialVariant]
    decode_combine: NotRequired[dict[str, dict[str, str]]]
    cache_files: dict[str, str]


def packed_loader_probe(*, use_rank_table: bool = False):
    """Invoke the production helper and compare every raw code independently.

    Synthetic masks include >256 survivors, empty words, all fragment/word
    boundaries, and ranks at the final window. Every FP8 byte pattern occurs.
    The large sparse backing tests a real address beyond 2 GiB; only selected
    rows are initialized, and no allocation is made during graph replay.
    """
    import tilelang
    import tilelang.language as T

    source = (
        Path(__file__).resolve().parents[2]
        / "kernels/jit/include/sgl_kernel/glm_packed_decode.cuh"
    ).read_text()
    loader = (
        "remnant_glm_expand_tile<true, true, true>"
        if use_rank_table
        else "remnant_glm_expand_tile"
    )
    symbol = "glm_packed_bytes_rank_table" if use_rank_table else "glm_packed_bytes"

    @tilelang.jit(out_idx=[-1])
    def probe(slots: int):
        @T.prim_func
        def main(
            values: T.Tensor([slots, 256], T.uint8),
            bitmap: T.Tensor([slots, 8], T.uint64),
            ids: T.Tensor([32], T.int32),
            out: T.Tensor([32, 512], T.uint8),
        ):
            with T.Kernel(1, threads=128):
                T.import_source(source)
                scratch = T.alloc_shared([32, 272], T.uint8)
                rank_table = T.alloc_shared([16], T.uint32)
                a = T.alloc_shared([32, 128], T.uint8)
                b = T.alloc_shared([32, 128], T.uint8)
                c = T.alloc_shared([32, 128], T.uint8)
                d = T.alloc_shared([32, 128], T.uint8)
                T.annotate_layout(
                    {
                        a: T.Layout([32, 128], lambda i, j: (i, j ^ ((i & 7) << 4))),
                        b: T.Layout([32, 128], lambda i, j: (i, j ^ ((i & 7) << 4))),
                        c: T.Layout([32, 128], lambda i, j: (i, j ^ ((i & 7) << 4))),
                        d: T.Layout([32, 128], lambda i, j: (i, j ^ ((i & 7) << 4))),
                    }
                )
                T.evaluate(
                    T.call_extern(
                        "int32",
                        "remnant_glm_init_rank_table",
                        T.access_ptr(rank_table, "w"),
                    )
                )
                T.sync_threads()
                T.evaluate(
                    T.call_extern(
                        "int32",
                        loader,
                        T.access_ptr(values, "r", extent=T.int64(slots) * 256),
                        T.access_ptr(bitmap, "r", extent=T.int64(slots) * 8),
                        T.access_ptr(ids, "r"),
                        T.access_ptr(scratch, "rw"),
                        T.access_ptr(a, "w"),
                        T.access_ptr(b, "w"),
                        T.access_ptr(c, "w"),
                        T.access_ptr(d, "w"),
                        T.int64(slots),
                        T.int64(0),
                        T.access_ptr(rank_table, "r"),
                    )
                )
                T.sync_threads()
                for i, j in T.Parallel(32, 128):
                    out[i, j] = a[i, j]
                    out[i, j + 128] = b[i, j]
                    out[i, j + 256] = c[i, j]
                    out[i, j + 384] = d[i, j]

        return main.with_attr("global_symbol", symbol)

    return probe


def compile_sm90_variant(case: tuple[int, int, int]) -> list[PartialVariant]:
    """One independent head/split pair, owned by a CPU compiler process."""
    from sglang.kernels.ops.attention.dsa.tilelang_kernel import (
        sparse_mla_fwd_decode_combine,
        sparse_mla_fwd_decode_partial_fp8,
    )

    heads, inner, block_i = case
    variants = []
    for packed in (False, True) if block_i == 32 else (True,):
        kernel = sparse_mla_fwd_decode_partial_fp8(
            heads,
            512,
            0,
            2112,
            packed=packed,
            sm_scale=ATTENTION_SCALE,
            block_I=block_i,
            inner_iter=inner,
            threads=128,
        )
        variants.append(
            dict(
                heads=heads,
                inner=inner,
                packed=packed,
                block_i=block_i,
                source_sha256=hashlib.sha256(
                    kernel.get_kernel_source().encode()
                ).hexdigest(),
            )
        )
    sparse_mla_fwd_decode_combine(
        heads,
        512,
        ((2112 + block_i * inner - 1) // (block_i * inner)) * block_i,
        head_per_block=4,
        block_I=block_i,
        threads=128,
    )
    return variants


def precompile_sm90() -> None:
    """Build exact CPU artifacts without allocating a CUDA tensor or context."""
    required = (
        "TRITON_CACHE_DIR",
        "TILELANG_CACHE_DIR",
        "REMNANT_GLM_COMPILE_MANIFEST",
    )
    missing = [name for name in required if not os.environ.get(name)]
    if missing:
        raise RuntimeError(
            "GLM CPU preparation requires explicit paths: " + ", ".join(missing)
        )
    os.environ.setdefault(
        "TILELANG_DEFAULT_TARGET", json.dumps({"kind": "cuda", "arch": "sm_90a"})
    )
    import multiprocessing
    from concurrent.futures import ProcessPoolExecutor

    from sglang.srt.remnant.glm import compile_kernels
    from sglang.srt.remnant.glm_config import (
        H200_PACKED_DECODE_COMBINE,
        H200_PACKED_DECODE_INNER,
    )

    store, unpack = compile_kernels()
    artifacts: CompileManifest = {
        "attention_scale": ATTENTION_SCALE,
        "precompile_workers": 4,
        "triton": [store.hash, unpack.hash],
        "tilelang": [],
    }
    cases = [
        (heads, inner, 32) for heads in (8, 16, 32, 64) for inner in (1, 2, 33, 66)
    ]
    cases.extend([(8, 17, 64), (16, 17, 64), (16, 9, 128)])
    cases.extend(
        (16, inner, 32)
        for inner in sorted(set(H200_PACKED_DECODE_INNER.values()) - {1, 2})
    )
    # Separate processes isolate TVM compiler state; each pair owns its combine
    # variant. The maintained runner supplies the overall bounded subprocess.
    with ProcessPoolExecutor(
        max_workers=4, mp_context=multiprocessing.get_context("spawn")
    ) as executor:
        for variants in executor.map(compile_sm90_variant, cases):
            artifacts["tilelang"].extend(variants)
            print(
                "Compiled GLM variants",
                [(v["heads"], v["inner"], v["block_i"], v["packed"]) for v in variants],
                flush=True,
            )
    packed_loader_probe()(8388610)
    if os.environ.get("REMNANT_GLM_UPSTREAM_DECODE") == "1":
        packed_loader_probe(use_rank_table=True)(8388610)
    from sglang.kernels.ops.attention.dsa.tilelang_kernel import (
        glm_packed_decode_combine,
    )

    combines = {
        f"{groups}x{columns}": glm_packed_decode_combine(16, 512, groups, columns)
        for groups, columns in H200_PACKED_DECODE_COMBINE
    }
    artifacts["cache_files"] = {
        str(path): hashlib.sha256(path.read_bytes()).hexdigest()
        for root in (
            Path(os.environ["TRITON_CACHE_DIR"]),
            Path(os.environ["TILELANG_CACHE_DIR"]),
        )
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }
    artifacts["decode_combine"] = {
        geometry: bind_decode_library(kernel, artifacts["cache_files"])
        for geometry, kernel in combines.items()
    }
    bind_partial_launch_metadata(artifacts)
    Path(os.environ["REMNANT_GLM_COMPILE_MANIFEST"]).write_text(
        json.dumps(artifacts, indent=2) + "\n"
    )


def bind_decode_library(kernel, cache: dict[str, str]) -> dict[str, str]:
    """Record the CPU cubin and TVM FFI library for a generated decode kernel."""
    source = kernel.get_kernel_source()
    digest = hashlib.sha256(source.encode()).hexdigest()
    # TileLang v0.1.12 tags the persistent cache entry on a cold compile,
    # when the adapter may only hold an in-memory runtime module.
    # Distinct wrappers can legitimately share the same generated device code.
    cache_path = getattr(kernel, "_tilelang_cache_path", None)
    if not cache_path:
        raise RuntimeError(f"Missing GLM decode cache entry: {digest}")
    entry = Path(cache_path)
    library = str(entry / "executable.so")
    host = kernel.adapter.get_host_source()
    if (
        cache.get(str(entry / "device_kernel.cu")) != digest
        or not host
        or cache.get(str(entry / "host_kernel.cu")) != hashlib.sha256(host.encode()).hexdigest()
        or library not in cache
    ):
        raise RuntimeError(f"GLM decode cache entry does not match the compilation: {entry}")
    cubin = str(_verified_sm90_cubin(source, cache))
    return dict(
        source_sha256=digest,
        cubin_path=cubin,
        cubin_sha256=cache[cubin],
        library_path=library,
        library_sha256=cache[library],
    )


def verify_decode_library(kernel, variant: dict[str, str]) -> None:
    """Require the TVM FFI adapter to load the library prepared on CPU."""
    library = getattr(kernel.adapter, "libpath", None)
    if library is None or Path(library) != Path(variant["library_path"]):
        raise RuntimeError(f"GLM decode did not load the prepared library: {library}")
    if (
        hashlib.sha256(Path(library).read_bytes()).hexdigest()
        != variant["library_sha256"]
        or hashlib.sha256(kernel.get_kernel_source().encode()).hexdigest()
        != variant["source_sha256"]
    ):
        raise RuntimeError("Prepared GLM decode library or device source changed")


def verify_decode_partial_library(kernel, variant: PartialVariant) -> None:
    """Check the Packed partial's loaded TVM FFI library and device source."""
    library = getattr(kernel.adapter, "libpath", None)
    prepared = variant.get("libraries", {})
    if library is None or str(library) not in prepared:
        raise RuntimeError(f"GLM partial did not load a prepared library: {library}")
    if (
        hashlib.sha256(Path(library).read_bytes()).hexdigest() != prepared[str(library)]
        or hashlib.sha256(kernel.get_kernel_source().encode()).hexdigest()
        != variant["source_sha256"]
    ):
        raise RuntimeError("Prepared GLM partial library or device source changed")


def _verified_sm90_cubin(source: str, cache: dict[str, str]) -> Path:
    """Resolve a generated SM90 cubin against its CPU preparation cache hashes."""
    import tilelang
    from tilelang.cache.cuda_binary_cache import CUDABinaryCache

    package = Path(tilelang.__file__).parent
    options = [
        "-std=c++20",
        f"-I{package}/3rdparty/../src",
        f"-I{package}/3rdparty/cutlass/include",
    ]
    key = CUDABinaryCache.make_key(
        code=source,
        target_kind="cuda",
        target_arch="90a",
        target_code=[],
        compile_format="cubin",
        options=options,
    )
    cubin = Path(CUDABinaryCache.get_path(key, "cubin"))
    if (
        str(cubin) not in cache
        or hashlib.sha256(cubin.read_bytes()).hexdigest() != cache[str(cubin)]
    ):
        raise RuntimeError(f"Unverified production GLM cubin: {cubin}")
    return cubin


def dynamic_shared_bytes(host_source: str) -> int:
    """Read TileLang's generated launch argument for dynamic shared memory."""
    sizes = re.findall(r"v_int64\) = \(\(int64_t\)(\d+)\);", host_source)
    if not sizes:
        raise RuntimeError("Missing generated dynamic shared-memory size")
    return int(sizes[-1])


def partial_parameters(
    source: str, symbol: str, *, packed: bool
) -> list[CudaParameter]:
    """Bind the generated CUDA arguments used by the exact-CTA graph probe."""
    match = re.search(r"\b" + re.escape(symbol) + r"\(([^)]*)\)\s*\{", source)
    if not match:
        raise RuntimeError(f"Missing production CUDA definition: {symbol}")
    parameters = []
    for declaration in match[1].split(","):
        declaration = declaration.strip()
        name = declaration.split()[-1]
        pointer = "*" in declaration
        if not pointer and declaration != f"int {name}":
            raise RuntimeError(f"Unsupported GLM CUDA ABI: {declaration}")
        parameters.append(
            dict(name=name, pointer=pointer, type="pointer" if pointer else "int")
        )
    expected = (["bitmap"] if packed else []) + [
        "indices",
        "kv_fp8",
        "partial_lse",
        "partial_o",
        "q_fp8",
        "num_pages",
        "seq_len",
    ]
    if [p["name"] for p in parameters] != expected:
        raise RuntimeError(f"GLM partial ABI changed: {parameters}")
    return parameters


def bind_partial_launch_metadata(compiled: CompileManifest) -> None:
    """Bind exact production cubins, generated ABI and dynamic shared bytes.

    The qualified TileLang version uses this public CUDA cache key API. Resolve
    it on CPU, require a manifest hash, and fail before H100 on any mismatch.
    """
    cache = compiled["cache_files"]
    for variant in compiled["tilelang"]:
        matches = [
            Path(p)
            for p, h in cache.items()
            if p.endswith("/device_kernel.cu") and h == variant["source_sha256"]
        ]
        if not matches:
            raise RuntimeError(f"Missing generated GLM source: {variant}")
        source = matches[0].read_text()
        if hashlib.sha256(source.encode()).hexdigest() != variant["source_sha256"]:
            raise RuntimeError(f"GLM generated source changed: {matches[0]}")
        cubin = _verified_sm90_cubin(source, cache)
        if (
            variant["packed"]
            and variant["heads"] == 16
            and variant.get("block_i", 32) == 32
        ):
            libraries = [str(path.with_name("executable.so")) for path in matches]
            variant["libraries"] = {
                path: cache[path] for path in libraries if path in cache
            }
            if not variant["libraries"]:
                raise RuntimeError("Missing CPU-prepared GLM partial runtime library")
        symbol = "glm_packed_partial_kernel" if variant["packed"] else "main_kernel"
        host_path = matches[0].parent / "host_kernel.cu"
        host = host_path.read_text()
        if hashlib.sha256(host.encode()).hexdigest() != cache.get(str(host_path)):
            raise RuntimeError(f"GLM generated host changed: {host_path}")
        variant.update(
            cubin_path=str(cubin),
            cubin_sha256=cache[str(cubin)],
            symbol=symbol,
            parameters=partial_parameters(source, symbol, packed=variant["packed"]),
            dynamic_shared=dynamic_shared_bytes(host),
        )


if __name__ == "__main__":
    precompile_sm90()
