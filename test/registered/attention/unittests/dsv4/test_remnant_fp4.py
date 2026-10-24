"""Model-free SM90 FP4 contracts using the upstream independent CPU oracle.

Author: Winston Cai.
These fixtures qualify storage/attention engineering, never model/task quality.
"""
from __future__ import annotations

import gc
import json
import math
import os
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from sglang.kernels.ops.attention.dsv4.kv_layout import KVLayout
from sglang.srt.remnant.config import validate_v41_cache_args
from sglang.test.kernels.deepseek_v4.dsv41_kv_quant_reference import (
    dequantize_k_cache_v41_fp4,
    quantize_k_cache_v41_fp4,
)

GPU = pytest.mark.skipif(not torch.cuda.is_available(), reason='requires Modal H100')
RATIOS = [0]*2 + [2]*18 + [1]*20
SOURCES = (2,8,14,20)


def _oracle(x: torch.Tensor, packed: bool) -> tuple[torch.Tensor, torch.Tensor]:
    """CPU-only oracle: dense quantization precedes independent stable selection."""
    assert x.device.type == 'cpu' and x.ndim == 3
    x = x.detach()
    dense = quantize_k_cache_v41_fp4(x)
    p = x.shape[1]
    values = dequantize_k_cache_v41_fp4(dense,p)
    if not packed:
        return dense,values
    out = torch.zeros((x.shape[0],p*224),dtype=torch.uint8)
    data = dense[:,:p*256].reshape(-1,256)
    scales = dense[:,p*256:p*288].reshape(-1,32)
    dst = out[:,:p*192].reshape(x.shape[0],p,192)
    for page in range(x.shape[0]):
        for row in range(p):
            codes = torch.empty(512,dtype=torch.uint8)
            codes[0::2] = data[page*p+row] & 15
            codes[1::2] = data[page*p+row] >> 4
            keep = torch.argsort(values[page,row].abs(),descending=True,stable=True)[:256].sort().values
            selected = codes[keep]
            dst[page,row,:128] = selected[0::2] | (selected[1::2] << 4)
            mask = torch.zeros(512,dtype=torch.bool)
            mask[keep] = True
            for word in range(8):
                bits = sum(int(mask[word*64+i]) << (63-i) for i in range(64))
                dst[page,row,128+word*8:136+word*8] = torch.tensor(list(bits.to_bytes(8,'little')),dtype=torch.uint8)
            values[page,row,~mask] = 0
    out[:,p*192:] = scales.reshape(x.shape[0],p*32)
    return out,values


def test_fp4_startup_contract():
    validate_v41_cache_args(SimpleNamespace())
    for mode in ('fp4','remnant'):
        validate_v41_cache_args(SimpleNamespace(dsv41_main_kv_cache_format=mode))
        for key,value in [('enable_hierarchical_cache',True),('speculative_algorithm','DSpark'),
                          ('dsv4_c4_cache_format','remnant'),('dsv4_c4_cache_keep_count',192),
                          ('disaggregation_mode','decode'),('enable_encoder_swa_bounded_replay',True)]:
            with pytest.raises(ValueError,match=key):
                validate_v41_cache_args(SimpleNamespace(dsv41_main_kv_cache_format=mode,**{key:value}))
    with pytest.raises(ValueError):
        validate_v41_cache_args(SimpleNamespace(dsv41_main_kv_cache_format='fp8'))
    import argparse
    from sglang.srt.server_args import ServerArgs
    from sglang.srt.runtime_context import get_context,get_model
    parser = argparse.ArgumentParser()
    ServerArgs.add_cli_args(parser)
    args = ServerArgs.from_cli_args(parser.parse_args(['--model','dummy','--dsv41-main-kv-cache-format','remnant']))
    assert args.dsv41_main_kv_cache_format == 'remnant'
    with get_context().override_server_args(dsv41_main_kv_cache_format='remnant'):
        assert get_model().dsv41_main_kv_cache_format == 'remnant'
    assert KVLayout.V41_FP4.bytes_per_token == 288
    assert KVLayout.V41_FP4_REMNANT.bytes_per_token == 224
    for p in (128,256):
        assert KVLayout.V41_FP4_REMNANT.page_bytes(p) == p*224


@GPU
@pytest.mark.parametrize('packed',[False,True])
@pytest.mark.parametrize('page',[128,256])
@pytest.mark.parametrize('pattern',['random','ties','zeros','boundaries'])
def test_fp4_store_and_reader(packed,page,pattern):
    from sglang.kernels.ops.attention.dsv4.fp4_cache import read_fp4_cache,store_fp4_cache
    g = torch.Generator().manual_seed(73)
    x = torch.randn((page,512),generator=g).bfloat16()
    if pattern == 'ties':
        x[:] = 1
        x[:,448:] = 2  # selected tail coordinates exercise bitmap word 7
    elif pattern == 'zeros':
        x.zero_()
        x[:,1::2] = -0.0
    elif pattern == 'boundaries':
        x[:] = torch.tensor([.25,.75,1.25,1.75,2.5,3.5,5.,6.]*64).bfloat16()
    # Per-row RoPE, BF16 tail rounding, then quantize exactly once.
    angle = torch.arange(page*32,dtype=torch.float32).reshape(page,32)*.003
    freq = torch.polar(torch.ones_like(angle),angle)
    rotated = x.clone()
    pairs = x[:,448:].float().reshape(page,32,2)
    rot = torch.view_as_complex(pairs)*freq
    rotated[:,448:] = torch.view_as_real(rot).flatten(1).bfloat16()
    source = torch.zeros((2,page,512),dtype=torch.bfloat16)
    source[1] = rotated
    expected,values = _oracle(source,packed)
    # Page zero is reserved and remains byte-zero (not quantized zero scales).
    expected[0].zero_()
    cache = torch.zeros_like(expected,device='cuda')
    locations = torch.arange(page,page*2,device='cuda',dtype=torch.int64)
    x_gpu,f_gpu = x.cuda(),freq.cuda()
    store_fp4_cache(x_gpu,cache,locations,page,packed=packed,freqs_cis=f_gpu)
    torch.testing.assert_close(cache.cpu(),expected,rtol=0,atol=0)
    before = cache.clone()
    bad = torch.tensor([0,-1,page*2],device='cuda',dtype=torch.int64)
    store_fp4_cache(x_gpu[:3],cache,bad,page,packed=packed,freqs_cis=f_gpu[:3])
    assert torch.equal(cache,before)
    ids = torch.tensor([page,page+1,2*page-1,page,-1,2*page,2**31-1],device='cuda',dtype=torch.int32)
    out = torch.empty((ids.numel()+2,1,512),device='cuda',dtype=torch.bfloat16)[1:-1]
    assert read_fp4_cache(cache,ids,page,packed=packed,out=out) is out
    target = torch.cat((values[1,[0,1,page-1,0]],torch.zeros((3,512),dtype=torch.bfloat16)))
    torch.testing.assert_close(out[:,0].cpu(),target,rtol=0,atol=0)


def _attention_case(packed: bool,heads: int,page: int):
    from test_remnant_flashmla_direct import _make_swa_cache
    from sglang.kernels.ops.attention.dsv4.fp4_cache import store_fp4_cache
    torch.manual_seed(37)
    x = torch.randn((2,page,512)).bfloat16()
    x[0].zero_()
    expected,values = _oracle(x,packed)
    expected[0].zero_()
    cache = torch.zeros_like(expected,device='cuda')
    loc = torch.arange(page,2*page,device='cuda',dtype=torch.int64)
    x_gpu = x[1].cuda()
    store_fp4_cache(x_gpu,cache,loc,page,packed=packed)
    swa = _make_swa_cache(2,128,torch.device('cuda'))
    swa_raw = swa.view(torch.uint8).as_strided((2,swa.stride(0)),(swa.stride(0),1))
    swa_values = torch.cat((swa_raw[:,:128*576].reshape(2,128,576)[:,:,:448].contiguous().view(torch.float8_e4m3fn).bfloat16(),
        swa_raw[:,:128*576].reshape(2,128,576)[:,:,448:].contiguous().view(torch.bfloat16)),dim=-1).reshape(-1,512)
    b = 8
    q = (torch.randn((b,1,heads,512),device='cuda')*.02).bfloat16()
    indices = torch.arange(64,device='cuda',dtype=torch.int32).reshape(1,1,64).expand(b,1,64).contiguous()
    extra = (torch.arange(512,device='cuda',dtype=torch.int32)%page+page).reshape(1,1,512).expand(b,1,512).contiguous()
    lengths = torch.tensor([0,1,63,64,65,317,511,512],device='cuda',dtype=torch.int32)
    valid = torch.arange(512,device='cuda')[None,None,:] < lengths[:,None,None]
    extra[~valid] = 2**31-1
    extra[:,0,2] = -1
    extra[-1,0,5] = 2**31-1  # invalid physical selection inside the live length
    sink = torch.linspace(-1,1,heads,device='cuda')
    return SimpleNamespace(q=q,swa=swa,indices=indices,cache=cache,
        cache_view=cache.view(2,page,1,224 if packed else 288),extra=extra,lengths=lengths,
        swa_values=swa_values,values=values.reshape(-1,512).cuda(),sink=sink,
        x=x_gpu,loc=loc,page=page,packed=packed)


def _attention_reference(t):
    idx = t.extra.long()
    mask = (idx >= 0) & (idx < t.values.shape[0]) & (torch.arange(512,device='cuda')[None,None,:] < t.lengths[:,None,None])
    gathered = t.values[idx.clamp(0,t.values.shape[0]-1)]
    kv = torch.cat((t.swa_values[t.indices.long()],gathered),dim=2).float()
    valid = torch.cat((torch.ones_like(t.indices,dtype=torch.bool),mask),dim=-1)
    scores = torch.einsum('bshd,bskd->bshk',t.q.float(),kv)*512**-.5
    scores.masked_fill_(~valid[:,:,None,:],-float('inf'))
    lse = scores.logsumexp(-1)
    weights = torch.exp(scores-torch.logaddexp(lse,t.sink[None,None,:])[:,:,:,None])
    out = torch.einsum('bshk,bskd->bshd',weights,kv).bfloat16()
    return out,lse.transpose(1,2)


@GPU
@pytest.mark.parametrize('packed',[False,True])
@pytest.mark.parametrize('heads',[64,128])
@pytest.mark.parametrize('page',[128,256])
def test_fp4_direct(packed,heads,page):
    from sgl_kernel.flash_mla import flash_mla_with_fp4_kvcache
    t = _attention_case(packed,heads,page)
    actual = flash_mla_with_fp4_kvcache(t.q,t.swa,t.indices,t.cache_view,t.extra,
        packed=packed,extra_topk_length=t.lengths,attn_sink=t.sink)
    for a,e in zip(actual,_attention_reference(t)):
        torch.testing.assert_close(a,e,atol=.02,rtol=.02)
    # Adapter reconstructs the same pruned values, then uses stock BF16 attention.
    from sglang.kernels.ops.attention.dsv4.fp4_cache import read_fp4_cache
    from sgl_kernel.flash_mla import flash_mla_sparse_fwd
    ids = torch.arange(t.values.shape[0],device='cuda',dtype=torch.int32)
    reconstructed = torch.empty((ids.numel(),1,512),device='cuda',dtype=torch.bfloat16)
    read_fp4_cache(t.cache,ids,page,packed=packed,out=reconstructed)
    valid = (t.extra >= 0) & (t.extra < ids.numel()) & (torch.arange(512,device='cuda')[None,None,:] < t.lengths[:,None,None])
    combined = torch.cat((t.indices,torch.where(valid,t.extra+t.swa_values.shape[0],-1)),dim=-1)
    adapter = flash_mla_sparse_fwd(t.q[:,0],torch.cat((t.swa_values[:,None,:],reconstructed)),
        combined,512**-.5,attn_sink=t.sink)[0]
    torch.testing.assert_close(actual[0][:,0],adapter,atol=.02,rtol=.02)


@GPU
@pytest.mark.parametrize('packed',[False,True])
@pytest.mark.parametrize('heads',[64,128])
def test_fp4_graph_and_stream(packed,heads):
    from sgl_kernel.flash_mla import flash_mla_with_fp4_kvcache
    from sglang.kernels.ops.attention.dsv4.fp4_cache import read_fp4_cache,store_fp4_cache
    t = _attention_case(packed,heads,128)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    out = torch.empty((t.loc.numel(),1,512),device='cuda',dtype=torch.bfloat16)
    with torch.cuda.stream(stream):
        def run():
            store_fp4_cache(t.x,t.cache,t.loc,t.page,packed=packed)
            read_fp4_cache(t.cache,t.loc,t.page,packed=packed,out=out)
            return flash_mla_with_fp4_kvcache(t.q,t.swa,t.indices,t.cache_view,t.extra,
                packed=packed,extra_topk_length=t.lengths,attn_sink=t.sink)
        for _ in range(2):
            run()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph,stream=stream):
            captured = run()
        for step in range(2):
            t.x.copy_(t.x.roll(17,1)*-.7)
            t.extra[-1,0,:4] = torch.tensor([128+step,-1,129,128],device='cuda',dtype=torch.int32)
            t.lengths[-1] = 65+step
            graph.replay()
            expected_pages,expected_values = _oracle(torch.cat((torch.zeros_like(t.x.cpu())[None],t.x.cpu()[None])),packed)
            t.values = expected_values.reshape(-1,512).cuda()
            for a,e in zip(captured,_attention_reference(t)):
                torch.testing.assert_close(a,e,atol=.02,rtol=.02)
            torch.testing.assert_close(out[:,0],t.values[128:],rtol=0,atol=0)
    torch.cuda.current_stream().wait_stream(stream)


@GPU
def test_fp4_pool_allocations(tmp_path):
    from sglang.srt.mem_cache.deepseek_v4_memory_pool import DeepSeekV4SingleKVPool,DeepSeekV4TokenToKVPool
    from sglang.srt.runtime_context import get_context
    from test_remnant_pool import _gpu_storages
    from sglang.srt.model_executor.pool_configurator import DSV4PoolConfigurator
    from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
    report = {'status':'running','modes':[],'edges':[], 'scope':'model-free actual pool storage; no serving capacity claim'}
    with get_context().override_server_args(device='cuda',dsv4_attn_backend='flashmla',page_size=256,max_running_requests=8):
        for layout in (KVLayout.V41_FP4,KVLayout.V41_FP4_REMNANT):
            for p in (128,256):
                for rows in (0,1,127,128,129,255,256,257):
                    pool = DeepSeekV4SingleKVPool(rows,p,torch.float8_e4m3fn,448,64,1,'cuda',False,
                        global_page_size=256,kv_layout=layout)
                    expected = math.ceil((rows+256)/p)*p*layout.bytes_per_token
                    assert sum(_gpu_storages(pool).values()) == expected
                    report['edges'].append(dict(layout=layout.value,page=p,rows=rows,storage_bytes=expected))
                    del pool
    for mode in ('fp4','remnant'):
        with get_context().override_server_args(device='cuda',dsv4_attn_backend='flashmla',page_size=256,
                max_running_requests=8,chunked_prefill_size=512,dsv41_main_kv_cache_format=mode):
            metadata = SimpleNamespace(qk_nope_head_dim=448,qk_rope_head_dim=64,index_head_dim=128,
                context_len=131072,compress_ratios=RATIOS,window_size=128,
                hf_config=SimpleNamespace(kv_source_layer_ids=SOURCES))
            sizing = DSV4PoolConfigurator(SimpleNamespace(kv_cache_dtype_str='fp8_e4m3',model_config=metadata,
                layer_info=SimpleNamespace(start_layer=0,end_layer=40),pp_size=1,attn_dp_size=1,
                page_size=256,sliding_window_size=128,spec_algorithm=SpeculativeAlgorithm.NONE))
            record_bytes = 288 if mode == 'fp4' else 224
            assert sizing._compressed_bytes_per_full_token(1) == record_bytes+68
            assert sizing._compressed_bytes_per_full_token(2) == (record_bytes+68)/2
            workspace_bytes = 1024*512*2
            byte_budget = 1 << 30
            sizes = sizing.calculate_pool_sizes(byte_budget-workspace_bytes,256)
            full = sizes.full_max_total_num_tokens
            assert sizing.swa_cap_tokens is not None
            pool = DeepSeekV4TokenToKVPool(max_num_reqs=8,swa_size=sizes.swa_max_total_num_tokens,c4_size=0,c128_size=0,
                c4_state_pool_size=0,c128_state_pool_size=0,page_size=256,swa_page_size=256,
                dtype=torch.float8_e4m3fn,c4_state_dtype=torch.float32,c128_state_dtype=torch.float32,
                qk_nope_head_dim=448,qk_rope_head_dim=64,indexer_head_dim=128,layer_num=40,
                device='cuda',enable_memory_saver=False,compression_ratios=RATIOS,
                kv_source_layers=SOURCES,full_size=full,kv_layout=KVLayout.V4,
                compressed_kv_layout='sm90_fp4' if mode == 'fp4' else 'sm90_remnant')
            assert pool.kv_pools[2].layer_num == 3 and pool.kv_pools[1].layer_num == 1
            assert pool.get_extra_key_page_size(2) == 128 and pool.get_extra_key_page_size(20) == 256
            assert pool.get_extra_key_buffer(2).data_ptr() == pool.get_extra_key_buffer(3).data_ptr()
            groups = {name:sum(_gpu_storages(obj).values()) for name,obj in dict(
                swa=pool.swa_kv_pool,global_kv=pool.kv_pools,indexer=pool.index_pools,
                compressor_state=pool.compress_state_pools).items()}
            expected_global = (math.ceil((full/2+256)/128)*128*3 + math.ceil((full+256)/256)*256)*record_bytes
            assert groups['global_kv'] == expected_global
            from sglang.srt.mem_cache.deepseek_v4_memory_pool import DSV41_INDEX_PAGE_SIZE
            ip = DSV41_INDEX_PAGE_SIZE
            expected_index = (math.ceil(((full+256)//2+ip)/ip)*ip*3 + math.ceil((full+256+ip)/ip)*ip)*68
            assert groups['indexer'] == expected_index
            # Explicit prefill workspace is identical in both configurations.
            workspace = torch.empty((1024,1,512),device='cuda',dtype=torch.bfloat16)
            total_bytes = sum(groups.values())+workspace.nbytes
            assert total_bytes <= byte_budget
            assert total_bytes + 256*sizing.bytes_per_full_token > byte_budget
            report['modes'].append(dict(mode='Native' if mode == 'fp4' else 'Packed',
                groups=groups,prefill_workspace_bytes=workspace.nbytes,total_bytes=total_bytes,
                byte_budget=byte_budget,full_tokens=full,swa_tokens=sizes.swa_max_total_num_tokens,
                reserved_page_bytes=sizing._get_v41_reserved_page_bytes()))
            del pool,workspace
            gc.collect()
    a,b = report['modes']
    for group in ('swa','compressor_state'):
        assert a['groups'][group] == b['groups'][group]
    report['capacity_gain_percent'] = (b['full_tokens']/a['full_tokens']-1)*100
    report['status']='passed'
    Path(os.environ.get('REMNANT_ALLOCATION_REPORT',str(tmp_path/'allocation.json'))).write_text(json.dumps(report,indent=2)+'\n')


@GPU
@pytest.mark.parametrize('ratio',[1,2])
@pytest.mark.parametrize('packed',[False,True])
def test_fp4_compressor_boundary(ratio,packed):
    from sglang.srt.layers.attention.dsv4.dsv41_sparse import DeepseekV41Compressor
    from sglang.srt.layers.attention.deepseek_v4_backend import DeepseekV4AttnBackend
    from sglang.kernels.ops.attention.dsv4.fp4_cache import store_fp4_cache
    torch.manual_seed(71)
    compressor = DeepseekV41Compressor(512,512,ratio,1e-6,fused_compress=False).cuda()
    with torch.no_grad():
        compressor.norm.weight.copy_(torch.linspace(.5,1.5,512,device='cuda'))
    x = torch.randn((8,512),device='cuda',dtype=torch.bfloat16)
    kv,score = compressor.project(x)
    pooled = kv if ratio == 1 else compressor.pool_pairs(kv.reshape(4,2,512),score.reshape(4,2,512))
    n = pooled.shape[0]
    captured = {}
    def index_keys(latent,freq):
        captured['index_latent'] = latent.clone()
        return torch.zeros((n,128),device='cuda',dtype=torch.bfloat16)
    layout = KVLayout.V41_FP4_REMNANT if packed else KVLayout.V41_FP4
    cache = torch.zeros((2,128*layout.bytes_per_token),device='cuda',dtype=torch.uint8)
    def store(**kwargs):
        store_fp4_cache(kwargs['cache_k'],cache,kwargs['loc'],128,packed=packed,freqs_cis=kwargs['freqs_cis'])
    def index_store(**kwargs):
        captured['index_written'] = kwargs['cache_k'].clone()
    pool = SimpleNamespace(get_extra_key_layout=lambda _:layout,set_extra_key_buffer_fused=store,set_index_k_fp4=index_store)
    angle = torch.arange(n*32,device='cuda',dtype=torch.float32).reshape(n,32)*.021
    freq = torch.polar(torch.ones_like(angle),angle)
    layer = SimpleNamespace(layer_id=20 if ratio == 1 else 2,compressor=compressor,
        freqs_cis=freq,rope_head_dim=64,indexer=SimpleNamespace(owns_k=True,index_keys=index_keys))
    backend = SimpleNamespace(token_to_kv_pool=pool)
    slots = torch.arange(128,128+n,device='cuda',dtype=torch.int64)
    DeepseekV4AttnBackend._low_ratio_write_group(backend,layer,pooled,slots,
        torch.arange(n,device='cuda',dtype=torch.int64))
    latent = captured['index_latent']
    assert latent.dtype == torch.bfloat16 and captured['index_written'].shape == (n,128)
    # Independent normalization check also catches pruning before normalization.
    v = pooled.bfloat16().float()
    expected_norm = (v*torch.rsqrt(v.square().mean(-1,keepdim=True)+1e-6)*compressor.norm.weight.float()).bfloat16()
    torch.testing.assert_close(latent,expected_norm,rtol=2**-6,atol=2**-20)
    from sglang.srt.layers.attention.dsv4.dsv41_sparse import rope_tail
    rotated = rope_tail(latent,freq,64).cpu()
    source = torch.zeros((2,128,512),dtype=torch.bfloat16)
    source[1,:n] = rotated
    expected,_ = _oracle(source,packed)
    expected[0].zero_()
    # Untouched rows stay byte-zero rather than receiving quantized zero scales.
    expected[1,n*layout.data_bytes:128*layout.data_bytes].zero_()
    expected[1,128*layout.data_bytes+n*32:].zero_()
    torch.testing.assert_close(cache.cpu(),expected,rtol=0,atol=0)
