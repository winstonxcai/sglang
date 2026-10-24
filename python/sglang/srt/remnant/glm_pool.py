"""GLM Packed MLA subpool; indexer, KDA mapping and tails retain Native ownership.

Author: Winston Cai.
"""

from contextlib import nullcontext

import torch

from sglang.srt.runtime_context import get_model
from sglang.srt.mem_cache.memory_pool import (
    DSATokenToKVPool,
    MLATokenToKVPool,
    GPU_MEMORY_TYPE_KV_CACHE,
)
from .glm import PackedCache, normalize_and_store
from .glm_config import MAX_QUERY_CHUNK, SELECTION_COLUMNS


class GLMPackedTokenToKVPool(DSATokenToKVPool):
    glm_packed = True

    def _create_buffers(self):
        if (self.dtype, self.kv_lora_rank, self.qk_rope_head_dim) != (
            torch.float8_e4m3fn,
            512,
            0,
        ):
            raise ValueError("GLM Packed pool requires raw E4M3 512-dimensional MLA")
        if torch.cuda.get_device_capability(self.device) != (9, 0):
            raise ValueError("GLM Packed is qualified only for H100/SM90")
        if self.dcp_replicated:
            raise ValueError("GLM Packed does not support DCP")
        self.implementation = get_model().glm5_mla_cache_implementation
        with self.memory_saver_adapter.region(GPU_MEMORY_TYPE_KV_CACHE):
            with (
                torch.cuda.use_mem_pool(self.custom_mem_pool)
                if self.custom_mem_pool
                else nullcontext()
            ):
                self.kv_buffer = [
                    torch.zeros(
                        (self.size + self.page_size, 256),
                        dtype=torch.uint8,
                        device=self.device,
                    )
                    for _ in range(self.layer_num)
                ]
                self.bitmap_buffer = [
                    torch.zeros(
                        (self.size + self.page_size, 8),
                        dtype=torch.uint64,
                        device=self.device,
                    )
                    for _ in range(self.layer_num)
                ]
                # One shared Adapter scratch, reused serially by all MLA layers.
                self.adapter_workspace = (
                    torch.empty(
                        (MAX_QUERY_CHUNK * SELECTION_COLUMNS, 512),
                        dtype=torch.uint8,
                        device=self.device,
                    )
                    if self.implementation == "adapter"
                    else None
                )

    def get_glm_packed_buffer(self, layer_id):
        i = layer_id - self.start_layer
        return PackedCache(
            self.kv_buffer[i],
            self.bitmap_buffer[i],
            self.adapter_workspace,
            self.implementation,
        )

    def normalize_and_store_glm(self, layer_id, raw, norm, loc):
        cache = self.get_glm_packed_buffer(layer_id)
        return normalize_and_store(raw, norm, loc, cache.values, cache.bitmaps)

    def get_key_buffer(self, layer_id):
        raise RuntimeError(
            "GLM Packed requires get_glm_packed_buffer; no dense mirror exists"
        )

    get_value_buffer = get_key_buffer

    def set_mla_kv_buffer(self, *args, **kwargs):
        raise RuntimeError("GLM Packed is written once by the raw-latent producer")

    set_kv_buffer = set_mla_kv_buffer

    def get_kv_size_bytes(self):
        return super().get_kv_size_bytes() + sum(x.nbytes for x in self.bitmap_buffer)

    def move_kv_cache(self, tgt_loc, src_loc):
        # Copy complete logical pages: pooled indexer entries belong to pages,
        # whereas the two latent planes are indexed by physical token slots.
        if tgt_loc.numel() % self.page_size or src_loc.numel() != tgt_loc.numel():
            raise ValueError(
                "GLM Packed cache relocation requires matched complete pages"
            )
        # Lifecycle operation, outside the measured attention path. Validate
        # complete page geometry before modifying either latent or indexer data.
        for loc in (tgt_loc, src_loc):
            pages = loc.reshape(-1, self.page_size)
            expected = pages[:, :1] + torch.arange(self.page_size, device=loc.device)
            if (
                not torch.equal(pages, expected)
                or bool((pages[:, 0] % self.page_size != 0).any())
                or bool(((loc < 0) | (loc >= self.size + self.page_size)).any())
            ):
                raise ValueError(
                    "GLM Packed relocation needs valid aligned complete pages"
                )
        MLATokenToKVPool.move_kv_cache(self, tgt_loc, src_loc)
        self.index_key_cache.move(
            tgt_loc[:: self.page_size] // self.page_size,
            src_loc[:: self.page_size] // self.page_size,
        )
        for bitmap in self.bitmap_buffer:
            signed = bitmap.view(torch.int64)
            signed[tgt_loc.long()] = signed[src_loc.long()]

    def get_cpu_copy(self, indices, mamba_indices=None, req_pool_index=None):
        saved = super().get_cpu_copy(indices, mamba_indices, req_pool_index)
        saved["bitmap"] = [
            b.view(torch.int64)[indices].cpu() for b in self.bitmap_buffer
        ]
        return saved

    def load_cpu_copy(self, saved, indices, mamba_indices=None, req_pool_index=None):
        super().load_cpu_copy(saved, indices, mamba_indices, req_pool_index)
        for bitmap, cpu in zip(self.bitmap_buffer, saved["bitmap"]):
            bitmap.view(torch.int64)[indices] = cpu.to(bitmap.device)

    def host_pool_decls(self):
        raise ValueError("GLM Packed HiCache is not qualified")

    def get_contiguous_buf_infos(self):
        raise ValueError("GLM Packed disaggregation is not qualified")

    def _clear_buffers(self):
        super()._clear_buffers()
        del self.bitmap_buffer
        self.adapter_workspace = None
