"""Compact V162 HiF4 solution.

This module contains only the active implementations and their transitive
dependencies. Historical definitions superseded by later versions were removed.
"""
from __future__ import annotations

import math

from typing import Optional, Tuple

import torch

import heapq

_HIF4_BLOCK = 64

_NVFP4_BLOCK = 16

_SEARCH_CHUNK_BLOCKS = 16384

_E6_ANCHOR_OFFSETS = (-1, 0, 1, 2, 3, 4)

_K_QUOTIENT_TOTAL_ROUNDS = 8

_K_RELAX_GAMMAS = (1.0, 1.5, 2.0, 2.5)

_K_MEDIAN_EXTRA_ROUNDS = 3

_E6_TABLE_CACHE: dict[str, torch.Tensor] = {}

def dequantize_nvfp4(
    quant_float: torch.Tensor,
    scale_float: torch.Tensor,
    blk_size: int = _NVFP4_BLOCK,
) -> torch.Tensor:
    c = int(quant_float.shape[-1])
    if c % blk_size != 0:
        raise ValueError(
            f"Last dimension {c} is not divisible by NVFP4 block size {blk_size}"
        )
    x = quant_float.unflatten(-1, (-1, blk_size))
    result = x * scale_float.unsqueeze(-1)
    # Match the supplied checker domain: NVFP4 dequantization is rounded to BF16.
    return result.flatten(-2, -1).to(torch.bfloat16)

def _build_e6m2_table(device: torch.device) -> torch.Tensor:
    key = str(device)
    cached = _E6_TABLE_CACHE.get(key)
    if cached is not None and cached.device == device:
        return cached
    values = []
    for e in range(-48, 16):
        for m in (1.0, 1.25, 1.5, 1.75):
            v = math.ldexp(m, e)
            if v <= 49152.0:
                values.append(v)
    table = torch.tensor(values, dtype=torch.float32, device=device)
    _E6_TABLE_CACHE[key] = table
    return table

def _nearest_e6m2_index(target: torch.Tensor, table: torch.Tensor) -> torch.Tensor:
    # Avoid table[0].item()/table[-1].item() device synchronizations on accelerators.
    t = target.clamp(min=2.0 ** -48, max=49152.0)
    hi = torch.searchsorted(table, t).clamp(0, table.numel() - 1)
    lo = (hi - 1).clamp(0, table.numel() - 1)
    vlo = table[lo]
    vhi = table[hi]
    choose_hi = (vhi - t).abs() < (t - vlo).abs()
    return torch.where(choose_hi, hi, lo)

def _fixed_scale_self_sse(abs_x: torch.Tensor, sf: torch.Tensor) -> torch.Tensor:
    """Exact minimum unweighted SSE for a fixed E6M2 level-1 scale."""
    errs = []
    for mult in (1.0, 2.0, 4.0):
        denom = sf * mult
        mant = torch.round((abs_x / denom) * 4.0) * 0.25
        mant = mant.clamp_(0.0, 1.75)
        errs.append((mant * denom - abs_x).square().sum(dim=-1, keepdim=True))
    e1, e2, e4 = errs
    # lv2=1 -> each 4-vector may use effective multiplier 1 or 2.
    err_l2_1 = torch.minimum(e1, e2).sum(dim=-2, keepdim=True)
    # lv2=2 -> each 4-vector may use effective multiplier 2 or 4.
    err_l2_2 = torch.minimum(e2, e4).sum(dim=-2, keepdim=True)
    return torch.minimum(err_l2_1, err_l2_2).sum(dim=(-3, -2, -1))

def _materialize_fixed_scale_self(
    x: torch.Tensor,
    sf: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    abs_x = x.abs()
    mant_by_mult = []
    err_by_mult = []
    for mult in (1.0, 2.0, 4.0):
        denom = sf * mult
        mant = torch.round((abs_x / denom) * 4.0) * 0.25
        mant = mant.clamp(0.0, 1.75)
        rec = mant * denom
        mant_by_mult.append(mant)
        err_by_mult.append((rec - abs_x).square().sum(dim=-1, keepdim=True))

    e1, e2, e4 = err_by_mult
    l3_if_l2_1 = torch.where(e2 < e1, 2.0, 1.0)
    l3_if_l2_2 = torch.where(e4 < e2, 2.0, 1.0)
    err_l2_1 = torch.minimum(e1, e2).sum(dim=-2, keepdim=True)
    err_l2_2 = torch.minimum(e2, e4).sum(dim=-2, keepdim=True)
    l2 = torch.where(err_l2_2 < err_l2_1, 2.0, 1.0)
    l3 = torch.where(l2 == 1.0, l3_if_l2_1, l3_if_l2_2)

    mult = l2 * l3
    mant = torch.where(
        mult == 1.0,
        mant_by_mult[0],
        torch.where(mult == 2.0, mant_by_mult[1], mant_by_mult[2]),
    )
    sign = torch.sign(x)
    sign = torch.where(mant == 0.0, torch.zeros_like(sign), sign)
    return sign, mant, l2, l3

def _quantize_tensor_self_mse(
    x: torch.Tensor,
    *,
    return_dequant: bool = False,
) -> Tuple[dict, Optional[torch.Tensor]]:
    """Strict per-tensor HiF4 quantizer; objective is only this tensor's SSE."""
    shape = tuple(int(s) for s in x.shape)
    if not shape:
        raise ValueError("Input must have at least one dimension")
    c = shape[-1]
    if c % _HIF4_BLOCK != 0:
        raise ValueError(f"Last dimension {c} not divisible by HiF4 block size 64")

    x = x.float()
    nblocks = c // _HIF4_BLOCK
    rows = x.numel() // c
    blocks = x.reshape(rows, nblocks, 8, 2, 4).reshape(-1, 8, 2, 4)
    total = int(blocks.shape[0])

    table = _build_e6m2_table(x.device)
    last = int(table.numel() - 1)

    sf_out = torch.empty((total, 1, 1, 1), dtype=torch.bfloat16, device=x.device)
    l2_out = torch.empty((total, 8, 1, 1), dtype=torch.bfloat16, device=x.device)
    l3_out = torch.empty((total, 8, 2, 1), dtype=torch.bfloat16, device=x.device)
    sign_out = torch.empty((total, 8, 2, 4), dtype=torch.bfloat16, device=x.device)
    mant_out = torch.empty((total, 8, 2, 4), dtype=torch.bfloat16, device=x.device)
    dq_out = torch.empty_like(blocks) if return_dequant else None

    for start in range(0, total, _SEARCH_CHUNK_BLOCKS):
        end = min(start + _SEARCH_CHUNK_BLOCKS, total)
        xb = blocks[start:end]
        ax = xb.abs()
        bsz = int(xb.shape[0])

        # Natural anchor: the maximum value just fits 1.75 * 2 * 2 = 7 times sf.
        block_max = ax.amax(dim=(1, 2, 3))
        anchor = _nearest_e6m2_index(block_max / 7.0, table)

        best_err = torch.full((bsz,), float("inf"), dtype=torch.float32, device=x.device)
        best_idx = anchor.clone()
        for off in _E6_ANCHOR_OFFSETS:
            idx = (anchor + off).clamp(0, last)
            sf = table[idx].view(bsz, 1, 1, 1)
            err = _fixed_scale_self_sse(ax, sf)
            better = err < best_err
            best_err = torch.where(better, err, best_err)
            best_idx = torch.where(better, idx, best_idx)

        sf = table[best_idx].view(bsz, 1, 1, 1)
        sign, mant, l2, l3 = _materialize_fixed_scale_self(xb, sf)

        sf_out[start:end] = sf.to(torch.bfloat16)
        l2_out[start:end] = l2.to(torch.bfloat16)
        l3_out[start:end] = l3.to(torch.bfloat16)
        sign_out[start:end] = sign.to(torch.bfloat16)
        mant_out[start:end] = mant.to(torch.bfloat16)
        if dq_out is not None:
            dq_out[start:end] = sign * mant * l2 * l3 * sf

    prefix = shape[:-1]
    params = {
        "scale_factor": sf_out.reshape(*prefix, nblocks, 1, 1, 1),
        "scale_lv2": l2_out.reshape(*prefix, nblocks, 8, 1, 1),
        "scale_lv3": l3_out.reshape(*prefix, nblocks, 8, 2, 1),
        "sign": sign_out.reshape(*prefix, nblocks, 8, 2, 4),
        "mant": mant_out.reshape(*prefix, nblocks, 8, 2, 4),
    }
    dq = None
    if dq_out is not None:
        # Match checker dequantization output precision.
        dq = dq_out.reshape(shape).to(torch.bfloat16).float()
    return params, dq

def _merge_key_candidates_per_feature_block(
    x: torch.Tensor,
    params_list: list[dict],
    dq_list: list[torch.Tensor],
) -> dict:
    """
    Select the best K representative modulo sequence-wise translation.

    For each batch-prefix x 64-feature block independently, measure
      min_c ||Kq - (K-c)||_F^2,
    whose closed form is the centered residual SSE. Each selected block may use
    a different translation vector; their concatenation is still one vector
    common to all key positions, hence it remains an exact softmax nullspace.
    """
    if len(params_list) == 1 or x.dim() < 2 or int(x.shape[-2]) <= 1:
        return params_list[0]

    shape = tuple(int(s) for s in x.shape)
    seq = shape[-2]
    hidden = shape[-1]
    nblocks = hidden // 64
    batch_prefix = shape[:-2]
    groups = int(math.prod(batch_prefix)) if batch_prefix else 1
    x4 = x.float().reshape(groups, seq, nblocks, 64)

    scores = []
    for dq in dq_list:
        e = dq.float().reshape(groups, seq, nblocks, 64) - x4
        # Common translation across keys is invisible to softmax; project it out.
        e = e - e.mean(dim=1, keepdim=True)
        scores.append(e.square().sum(dim=(1, 3)))  # [groups, nblocks]
    score_stack = torch.stack(scores, dim=0)  # [candidate, groups, nblocks]
    best = score_stack.argmin(dim=0)  # [groups, nblocks]

    out: dict[str, torch.Tensor] = {}
    for name in params_list[0]:
        base = params_list[0][name]
        # Original parameter shape is [batch_prefix..., seq, nblocks, tail...].
        tail = tuple(int(v) for v in base.shape[len(batch_prefix) + 2 :])
        y = base.reshape(groups, seq, nblocks, *tail).clone()
        for ci in range(1, len(params_list)):
            cand = params_list[ci][name].reshape(groups, seq, nblocks, *tail)
            mask_shape = (groups, 1, nblocks) + (1,) * len(tail)
            mask = (best == ci).reshape(groups, 1, nblocks, *([1] * len(tail)))
            y = torch.where(mask, cand, y)
        out[name] = y.reshape(base.shape)
    return out

def _select_best_key_reconstruction_per_block(
    x: torch.Tensor,
    dq_list: list[torch.Tensor],
) -> torch.Tensor:
    """Return the history-best dequantized K independently per 64-feature block."""
    if len(dq_list) == 1 or x.dim() < 2 or int(x.shape[-2]) <= 1:
        return dq_list[0]
    shape = tuple(int(s) for s in x.shape)
    seq = shape[-2]
    hidden = shape[-1]
    nblocks = hidden // 64
    batch_prefix = shape[:-2]
    groups = int(math.prod(batch_prefix)) if batch_prefix else 1
    x4 = x.float().reshape(groups, seq, nblocks, 64)
    scores = []
    qs = []
    for dq in dq_list:
        q4 = dq.float().reshape(groups, seq, nblocks, 64)
        e = q4 - x4
        e = e - e.mean(dim=1, keepdim=True)
        scores.append(e.square().sum(dim=(1, 3)))
        qs.append(q4)
    best = torch.stack(scores, dim=0).argmin(dim=0)
    out = qs[0].clone()
    for ci in range(1, len(qs)):
        mask = (best == ci).reshape(groups, 1, nblocks, 1)
        out = torch.where(mask, qs[ci], out)
    return out.reshape(shape)

def _fast_walsh_hadamard_64(x):
    c=x.shape[-1]
    if c%64: return x.float()
    orig=x.shape; y=x.float().reshape(-1,c//64,64).clone(); h=1
    while h<64:
        z=y.reshape(*y.shape[:-1],-1,2*h); a=z[...,:h].clone(); b=z[...,h:2*h].clone(); z[...,:h]=a+b; z[...,h:2*h]=a-b; y=z.reshape(-1,c//64,64); h*=2
    return (y*0.125).reshape(orig)

def _compute_smooth_scale(amax,wmax,beta):
    if beta<=0:return torch.ones_like(wmax)
    ls=float(beta)*(torch.log(wmax.clamp_min(2**-24))-torch.log(amax.clamp_min(2**-24))); ls-=ls.median(); return torch.exp(ls).clamp_(2**-6,2**6)

def _make_hadamard_sign_vector(c, pattern, device):
    if pattern <= 0:
        return torch.ones(c, dtype=torch.float32, device=device)
    i = torch.arange(c, dtype=torch.int64, device=device)
    h = i * 1103515245 + 12345
    bit = (h ^ (h >> 16)) & 1
    return torch.where(bit == 0, 1.0, -1.0).float()

def _rotate_feature_blocks_64(x, pattern):
    if pattern < 0:
        return x.float()
    y = x.float()
    if pattern > 0:
        y = y * _make_hadamard_sign_vector(y.shape[-1], pattern, y.device)
    return _fast_walsh_hadamard_64(y)

def _rotate_attention_heads(x, num_heads, head_dim, pattern):
    if pattern < 0 or head_dim % 64 != 0:
        return x.float()
    shape = x.shape
    y = x.float().reshape(-1, num_heads, head_dim).reshape(-1, head_dim)
    y = _rotate_feature_blocks_64(y, pattern)
    return y.reshape(shape)

def _quantize_key_with_translation_search(x):
    x = x.float()
    if x.dim() < 2 or int(x.shape[-2]) <= 1:
        return _quantize_tensor_self_mse(x, return_dequant=False)[0]
    params_list, dq_list = [], []
    p, q = _quantize_tensor_self_mse(x, return_dequant=True)
    params_list.append(p); dq_list.append(q); q_best = q
    c_prev = torch.zeros_like(x.mean(dim=-2, keepdim=True))
    for _ in range(1, _K_QUOTIENT_TOTAL_ROUNDS):
        c_star = (x - q_best).mean(dim=-2, keepdim=True)
        delta = c_star - c_prev
        for gamma in _K_RELAX_GAMMAS:
            p, q = _quantize_tensor_self_mse(
                x - (c_prev + float(gamma) * delta), return_dequant=True)
            params_list.append(p); dq_list.append(q)
        q_best = _select_best_key_reconstruction_per_block(x, dq_list)
        c_prev = c_star
    c_prev = x.median(dim=-2, keepdim=True).values
    p, q = _quantize_tensor_self_mse(x - c_prev, return_dequant=True)
    params_list.append(p); dq_list.append(q)
    q_best = _select_best_key_reconstruction_per_block(x, dq_list)
    for _ in range(_K_MEDIAN_EXTRA_ROUNDS):
        c_star = (x - q_best).mean(dim=-2, keepdim=True)
        delta = c_star - c_prev
        for gamma in _K_RELAX_GAMMAS:
            p, q = _quantize_tensor_self_mse(
                x - (c_prev + float(gamma) * delta), return_dequant=True)
            params_list.append(p); dq_list.append(q)
        q_best = _select_best_key_reconstruction_per_block(x, dq_list)
        c_prev = c_star
    return _merge_key_candidates_per_feature_block(x, params_list, dq_list)

def _apply_per_head_permutation(x: torch.Tensor, num_heads: int, head_dim: int,
                         perm: Optional[torch.Tensor]) -> torch.Tensor:
    if not isinstance(perm, torch.Tensor) or tuple(perm.shape) != (num_heads, head_dim):
        return x.float()
    shape = x.shape
    y = x.float().reshape(-1, num_heads, head_dim)
    p = perm.to(y.device, dtype=torch.long).unsqueeze(0).expand(y.shape[0], -1, -1)
    y = torch.gather(y, 2, p)
    return y.reshape(shape)

def _build_balanced_feature_permutation(key: torch.Tensor) -> torch.Tensor:
    c = int(key.numel())
    if c % 64 != 0 or c < 128:
        return torch.argsort(torch.log2(key.clamp_min(2.0 ** -40)), stable=True)
    nb = c // 64
    order = torch.argsort(key, descending=True, stable=True)
    # Round-robin assignment makes every 64-block receive a comparable spectrum
    # of heavy/light channels before the block-local Hadamard mixes them.
    grid = torch.empty((nb, 64), dtype=torch.long, device=key.device)
    t = torch.arange(c, device=key.device)
    block = t.remainder(nb)
    slot = torch.div(t, nb, rounding_mode='floor')
    grid[block, slot] = order
    return grid.reshape(-1)

def _apply_per_head_matrix(x: torch.Tensor, num_heads: int, head_dim: int,
                           mats: torch.Tensor) -> torch.Tensor:
    if x.shape[-1] != num_heads * head_dim:
        return x.float()
    if not isinstance(mats, torch.Tensor) or tuple(mats.shape) != (num_heads, head_dim, head_dim):
        return x.float()
    y = x.float().reshape(x.shape[0], num_heads, head_dim)
    return torch.einsum("lhd,hde->lhe", y, mats.to(y.device, dtype=torch.float32)).reshape_as(x)

def _apply_query_transform_components(x, state, q_num_heads, head_dim):
    y=x.float()
    if not (isinstance(state,dict) and state.get("enabled",False)):return y
    s=state.get("scale")
    if isinstance(s,torch.Tensor) and s.numel()==y.shape[-1]:y=y*s.to(y.device)
    if state.get("packed",False):
        p=state.get("perm")
        if isinstance(p,torch.Tensor):y=_apply_per_head_permutation(y,q_num_heads,head_dim,p)
    else:
        r=int(state.get("rotation",-1))
        if r>=0:y=_rotate_attention_heads(y,q_num_heads,head_dim,r)
    return y

def _apply_key_transform_components(x, state, kv_num_heads, head_dim):
    y=x.float()
    if not (isinstance(state,dict) and state.get("enabled",False)):return y
    s=state.get("scale")
    if isinstance(s,torch.Tensor) and s.numel()==y.shape[-1]:y=y/s.to(y.device)
    if state.get("packed",False):
        p=state.get("perm")
        if isinstance(p,torch.Tensor):y=_apply_per_head_permutation(y,kv_num_heads,head_dim,p)
    else:
        r=int(state.get("rotation",-1))
        if r>=0:y=_rotate_attention_heads(y,kv_num_heads,head_dim,r)
    return y

_ATTENTION_TRANSFORM_VERSION='v44_adaptive'

def _apply_query_transform(q, state, q_num_heads, head_dim):
    # ---------------------------------------------------------
    # V162: per-head rotation
    # ---------------------------------------------------------
    patterns = state.get("head_rotation_patterns")

    if isinstance(patterns, torch.Tensor):
        s = state.get("scale")

        if isinstance(s, torch.Tensor) and s.numel() == q.shape[-1]:
            q = q * s.to(q.device)

        q = _rotate_heads_by_pattern(
            q,
            q_num_heads,
            head_dim,
            patterns,
        )

        return q

    # ---------------------------------------------------------
    # Original path: unchanged
    # ---------------------------------------------------------
    if state.get('version') == _ATTENTION_TRANSFORM_VERSION:
        if state.get('transform_kind') == 'rot':
            s = state.get('scale')

            if isinstance(s, torch.Tensor):
                q = q * s.to(q.device)

            q = _rotate_attention_heads(
                q,
                q_num_heads,
                head_dim,
                int(state.get('rotation', 0)),
            )
        else:
            q = _apply_per_head_matrix(
                q,
                q_num_heads,
                head_dim,
                state.get('matrix'),
            )
    else:
        q = _apply_query_transform_components(
            q,
            state,
            q_num_heads,
            head_dim,
        )

    return q

def _apply_key_transform(k, state, kv_num_heads, head_dim):
    # ---------------------------------------------------------
    # V162: per-head rotation
    # ---------------------------------------------------------
    patterns = state.get("head_rotation_patterns")

    if isinstance(patterns, torch.Tensor):
        s = state.get("scale")

        if isinstance(s, torch.Tensor) and s.numel() == k.shape[-1]:
            k = k / s.to(k.device)

        k = _rotate_heads_by_pattern(
            k,
            kv_num_heads,
            head_dim,
            patterns,
        )

        return k

    # ---------------------------------------------------------
    # Original path: unchanged
    # ---------------------------------------------------------
    if state.get('version') == _ATTENTION_TRANSFORM_VERSION:
        if state.get('transform_kind') == 'rot':
            s = state.get('scale')

            if isinstance(s, torch.Tensor):
                k = k / s.to(k.device)

            k = _rotate_attention_heads(
                k,
                kv_num_heads,
                head_dim,
                int(state.get('rotation', 0)),
            )
        else:
            k = _apply_per_head_matrix(
                k,
                kv_num_heads,
                head_dim,
                state.get('matrix'),
            )
    else:
        k = _apply_key_transform_components(
            k,
            state,
            kv_num_heads,
            head_dim,
        )

    return k

_KEY_TRANSLATION_STEP_FACTORS=(1.0,1.75,2.5)

_KEY_MEAN_REFINEMENT_ROUNDS=2

_KEY_MEDIAN_REFINEMENT_ROUNDS=1

def _select_key_candidate_by_centered_error(x, params_list, dq_list):
    return _select_best_key_reconstruction_per_block(x,dq_list)

def _dequantize_hif4_params(params, shape):
    return (params['sign'] * params['mant'] * params['scale_lv2'] * params['scale_lv3'] * params['scale_factor']).reshape(shape).float()

def _clone_hif4_params(params):
    return {k: v.clone() for k, v in params.items()}

def _quantize_with_block_hessian(x: torch.Tensor, hblocks: torch.Tensor, *, return_dequant=False):
    """Fast-exact V37 Hessian E6 selection without per-row Hessian replication."""
    shape=tuple(int(s) for s in x.shape); c=shape[-1]
    if c%64!=0 or hblocks.dim()!=3 or tuple(hblocks.shape[1:])!=(64,64) or hblocks.shape[0]!=c//64:
        return _quantize_tensor_self_mse(x,return_dequant=return_dequant)
    x=x.float(); nb=c//64; rows=x.numel()//c
    blocks=x.reshape(rows,nb,8,2,4)
    table=_build_e6m2_table(x.device); last=int(table.numel()-1)
    sf_out=torch.empty((rows,nb,1,1,1),dtype=torch.bfloat16,device=x.device)
    l2_out=torch.empty((rows,nb,8,1,1),dtype=torch.bfloat16,device=x.device)
    l3_out=torch.empty((rows,nb,8,2,1),dtype=torch.bfloat16,device=x.device)
    sg_out=torch.empty((rows,nb,8,2,4),dtype=torch.bfloat16,device=x.device)
    ma_out=torch.empty((rows,nb,8,2,4),dtype=torch.bfloat16,device=x.device)
    dq_out=torch.empty_like(blocks) if return_dequant else None
    H=hblocks.to(x.device,torch.float32)
    # Match the old ~4096-block working-set, but chunk by complete rows so H can broadcast.
    chunk_rows=max(1,4096//max(nb,1))
    for rs in range(0,rows,chunk_rows):
        re=min(rows,rs+chunk_rows); z=blocks[rs:re]; nr=re-rs
        anchor=_nearest_e6m2_index(z.abs().amax((2,3,4))/7.0,table)
        best=torch.full((nr,nb),float('inf'),dtype=torch.float32,device=x.device)
        best_pack=None;best_q=None
        for off in _E6_ANCHOR_OFFSETS:
            idx=(anchor+off).clamp(0,last);sf=table[idx].view(nr,nb,1,1,1)
            sg,ma,l2,l3=_materialize_fixed_scale_self(z,sf)
            q=(sg*ma*l2*l3*sf).reshape(nr,nb,64);e=q-z.reshape(nr,nb,64)
            err=torch.einsum('rbi,bij,rbj->rb',e,H,e)
            better=err<best
            if best_pack is None:
                best=err;best_pack=[sf.clone(),l2.clone(),l3.clone(),sg.clone(),ma.clone()];best_q=q.clone()
            else:
                best=torch.where(better,err,best)
                vals=(sf,l2,l3,sg,ma)
                for jj,v in enumerate(vals):
                    mask=better.view(nr,nb,*([1]*(v.dim()-2)))
                    best_pack[jj]=torch.where(mask,v,best_pack[jj])
                best_q=torch.where(better[:,:,None],q,best_q)
        sf,l2,l3,sg,ma=best_pack
        sf_out[rs:re]=sf.to(torch.bfloat16);l2_out[rs:re]=l2.to(torch.bfloat16);l3_out[rs:re]=l3.to(torch.bfloat16)
        sg_out[rs:re]=sg.to(torch.bfloat16);ma_out[rs:re]=ma.to(torch.bfloat16)
        if dq_out is not None:dq_out[rs:re]=best_q.reshape(nr,nb,8,2,4)
    prefix=shape[:-1]
    params={'scale_factor':sf_out.reshape(*prefix,nb,1,1,1),'scale_lv2':l2_out.reshape(*prefix,nb,8,1,1),
            'scale_lv3':l3_out.reshape(*prefix,nb,8,2,1),'sign':sg_out.reshape(*prefix,nb,8,2,4),'mant':ma_out.reshape(*prefix,nb,8,2,4)}
    dq=dq_out.reshape(shape).to(torch.bfloat16).float() if dq_out is not None else None
    return params,dq

_FULL_KEY_SEARCH_MAX_SEQUENCE = 64

def _quantize_key_fast_translation_search(x):
    x = x.float()
    if x.dim() >= 2 and int(x.shape[-2]) <= _FULL_KEY_SEARCH_MAX_SEQUENCE:
        return _quantize_key_with_translation_search(x)
    if x.dim()<2 or int(x.shape[-2])<=1:
        return _quantize_tensor_self_mse(x,return_dequant=False)[0]
    params=[]; dqs=[]
    p,q=_quantize_tensor_self_mse(x,return_dequant=True); params.append(p); dqs.append(q)
    qbest=q; cprev=torch.zeros_like(x.mean(dim=-2,keepdim=True))
    for _ in range(_KEY_MEAN_REFINEMENT_ROUNDS):
        cstar=(x-qbest).mean(dim=-2,keepdim=True); delta=cstar-cprev
        for g in _KEY_TRANSLATION_STEP_FACTORS:
            p,q=_quantize_tensor_self_mse(x-(cprev+float(g)*delta),return_dequant=True)
            params.append(p); dqs.append(q)
        qbest=_select_key_candidate_by_centered_error(x,params,dqs); cprev=cstar
    cprev=x.median(dim=-2,keepdim=True).values
    p,q=_quantize_tensor_self_mse(x-cprev,return_dequant=True);params.append(p);dqs.append(q)
    qbest=_select_key_candidate_by_centered_error(x,params,dqs)
    for _ in range(_KEY_MEDIAN_REFINEMENT_ROUNDS):
        cstar=(x-qbest).mean(dim=-2,keepdim=True);delta=cstar-cprev
        for g in _KEY_TRANSLATION_STEP_FACTORS:
            p,q=_quantize_tensor_self_mse(x-(cprev+float(g)*delta),return_dequant=True)
            params.append(p);dqs.append(q)
        qbest=_select_key_candidate_by_centered_error(x,params,dqs);cprev=cstar
    return _merge_key_candidates_per_feature_block(x,params,dqs)

def _decode_calibration_activations(calib_activation_list, k, device):
    out=[]
    for pair in calib_activation_list:
        if not isinstance(pair,(list,tuple)) or len(pair)!=2:
            continue
        a=dequantize_nvfp4(pair[0],pair[1]).float().to(device).reshape(-1,k)
        if a.numel() and a.shape[-1]==k:
            out.append(a)
    return out

def _expand_block_values(v):
    return v.reshape(-1,1).expand(-1,64).reshape(-1)

def _apply_linear_transform(x,smooth,perm,phases,had,weight_side=False):
    y=x.float()
    y=y/smooth if weight_side else y*smooth
    y=y.index_select(-1,perm.to(y.device,dtype=torch.long))
    pv=_expand_block_values(phases.to(y.device,torch.float32))
    y=y/pv if weight_side else y*pv
    if had:
        y=_fast_walsh_hadamard_64(y)
    return y

def _compute_weight_hessian_64(wq):
    if wq.dim()!=2 or wq.shape[-1]%64!=0:return None
    m,k=map(int,wq.shape);nb=k//64
    z=wq.float().reshape(m,nb,64)
    H=torch.einsum('mbi,mbj->bij',z,z)
    sc=H.diagonal(dim1=-2,dim2=-1).mean(-1).abs().clamp_min(1e-12)
    return H/sc[:,None,None]

def _compute_weight_hessian_256(wq):
    if wq.dim()!=2 or wq.shape[-1]%256!=0:return None
    m,k=map(int,wq.shape);ng=k//256
    z=wq.float().reshape(m,ng,256)
    H=torch.einsum('mgi,mgj->gij',z,z)
    sc=H.diagonal(dim1=-2,dim2=-1).mean(-1).abs().clamp_min(1e-12)
    return H/sc[:,None,None]

def _refine_mantissas_hessian_256(y,p,H,iters=4):
    """Hessian-only Super256 OMCD: minimize (Q(A)-A)^T H (Q(A)-A)."""
    shape=tuple(int(s) for s in y.shape);k=shape[-1]
    if k%256!=0 or not isinstance(H,torch.Tensor):return p
    rows=y.numel()//k;ng=k//256;nb=k//64
    if tuple(H.shape)!=(ng,256,256):return p

    pp=_clone_hif4_params(p)
    sf=pp['scale_factor'].float().reshape(rows,nb,1,1,1)
    l2=pp['scale_lv2'].float().reshape(rows,nb,8,1,1)
    l3=pp['scale_lv3'].float().reshape(rows,nb,8,2,1)
    eff=(sf*l2*l3).expand(rows,nb,8,2,4).reshape(rows,ng,256)
    u=(pp['sign'].float()*pp['mant'].float()).reshape(rows,ng,256)
    yy=y.float().reshape(rows,ng,256)
    H=H.to(y.device,torch.float32)

    q=u*eff
    e=q-yy
    g=torch.einsum('gij,rgi->rgj',H,e)
    diag=H.diagonal(dim1=-2,dim2=-1).unsqueeze(0)
    step=0.25*eff
    step2=step.square()*diag
    gidx=torch.arange(ng,device=y.device).view(1,ng).expand(rows,ng)

    for _ in range(int(iters)):
        any_good=False
        for sub in range(4):
            lo=sub*64;hi=lo+64
            base=2.0*step[:,:,lo:hi]*g[:,:,lo:hi]
            dp=base+step2[:,:,lo:hi]
            dm=-base+step2[:,:,lo:hi]
            us=u[:,:,lo:hi]
            dp.masked_fill_(us>=1.75-1e-6,float('inf'))
            dm.masked_fill_(us<=-1.75+1e-6,float('inf'))
            choose=dp<dm
            move=torch.minimum(dp,dm)
            best,j0=move.min(dim=2)
            good=best<-1e-8
            if not bool(good.any().item()):continue
            any_good=True
            plus=choose.gather(2,j0.unsqueeze(-1)).squeeze(-1)
            direction=torch.where(plus,torch.ones_like(best),-torch.ones_like(best))
            du=0.25*direction*good
            j=j0+lo
            u.scatter_add_(2,j.unsqueeze(-1),du.unsqueeze(-1))
            de=du*eff.gather(2,j.unsqueeze(-1)).squeeze(-1)
            col=H[gidx,:,j]
            g.add_(col*de.unsqueeze(-1))
        if not any_good:break

    ma=u.abs().reshape(rows,nb,64)
    sg=torch.sign(u).reshape(rows,nb,64)
    sg=torch.where(ma==0.0,torch.zeros_like(sg),sg)
    pp['mant']=ma.reshape_as(pp['mant']).to(torch.bfloat16)
    pp['sign']=sg.reshape_as(pp['sign']).to(torch.bfloat16)
    return pp

def _make_activation_transform_state(version,smooth,perm,had,phases,**extra):
    st={
        'version':version,
        'rule_safe_no_AW':True,
        'beta':0.50,
        'smooth':smooth.detach().cpu().float(),
        'perm':perm.detach().cpu().to(torch.int32),
        'hadamard64':bool(had),
        'block_phase':phases.detach().cpu().float(),
    }
    st.update(extra)
    return st

def _decode_and_transform_activation(activation_quant,activation_scale,st):
    a=dequantize_nvfp4(activation_quant,activation_scale).float()
    s=st.get('smooth');perm=st.get('perm');ph=st.get('block_phase')
    if not all(isinstance(z,torch.Tensor) for z in (s,perm,ph)):
        return a
    return _apply_linear_transform(a,s.to(a.device),perm.to(a.device),ph.to(a.device),
                         bool(st.get('hadamard64',False)),False)

def _slice_hif4_params(p, st, en):
    return {k: v[st:en].clone() for k,v in p.items()}

def _write_hif4_param_slice(dst, src, st, en):
    for k in dst:
        dst[k][st:en] = src[k]
    return dst

def _refine_weight_hessian_256_chunked(wt, p, HA256, iters=1, chunk_rows=256):
    """Memory-bounded H256 mantissa refinement for offline Weight."""
    if not isinstance(HA256, torch.Tensor) or wt.dim()!=2 or wt.shape[-1]%256:
        return p
    out=_clone_hif4_params(p)
    m=int(wt.shape[0])
    for st in range(0,m,int(chunk_rows)):
        en=min(st+int(chunk_rows),m)
        pc=_slice_hif4_params(out,st,en)
        pc=_refine_mantissas_hessian_256(wt[st:en],pc,HA256,int(iters))
        _write_hif4_param_slice(out,pc,st,en)
    return out

def _make_quantized_weight_state(version,s,perm,had,ph,wq,extra=None):
    H64=_compute_weight_hessian_64(wq)
    H256=_compute_weight_hessian_256(wq)
    st=_make_activation_transform_state(version,s,perm,had,ph,
                          transform_kind='safe_v40_marginal',
                          super256_iters=4)
    if extra:
        st.update(extra)
    if isinstance(H64,torch.Tensor):
        st['weight_hessian_blocks']=H64.detach().cpu().to(torch.bfloat16)
    if isinstance(H256,torch.Tensor):
        st['super256_hessian_blocks']=H256.detach().cpu().to(torch.bfloat16)
    return st

def _quantize_value_tensor(v_quant,v_scale,kv_num_heads,head_dim,v_state):
    # Strict tensor-self V path.
    v=dequantize_nvfp4(v_quant,v_scale).float()
    return _quantize_tensor_self_mse(v,return_dequant=False)[0]

def _dequantize_params_for_tensor(y,p):
    return _dequantize_hif4_params(p,tuple(int(s) for s in y.shape)).float()

def _refine_scales_hessian_256(y,p,H256,sweeps=1):
    """
    Exact coordinate update for E6M2 scale under the full H256 quadratic.

    With current q_b = s*c and gradient g_b = dL/dq_b / 2,
        Delta(s) = 2 (s-s0) c^T g_b + (s-s0)^2 c^T H_bb c.
    Thus the continuous optimum for fixed coefficient c is available in closed
    form; we snap around that optimum to legal E6M2 and re-round mantissas while
    keeping lv2/lv3 fixed. No output target is needed.
    """
    shape=tuple(int(s) for s in y.shape); k=shape[-1]
    if y.dim()!=2 or k%256!=0 or not isinstance(H256,torch.Tensor):
        return p
    rows=int(y.shape[0]); ng=k//256; nb=k//64
    if tuple(H256.shape)!=(ng,256,256):
        return p

    pp=_clone_hif4_params(p)
    yy=y.float().reshape(rows,ng,256)
    H=H256.to(y.device,torch.float32)
    table=_build_e6m2_table(y.device)
    last=int(table.numel()-1)

    for _ in range(int(sweeps)):
        q=_dequantize_params_for_tensor(y,pp).reshape(rows,ng,256)
        e=q-yy
        grad=torch.einsum('gij,rgi->rgj',H,e)

        sfv=pp['scale_factor'].float().reshape(rows,nb,1,1,1)
        l2v=pp['scale_lv2'].float().reshape(rows,nb,8,1,1)
        l3v=pp['scale_lv3'].float().reshape(rows,nb,8,2,1)
        sgv=pp['sign'].float().reshape(rows,nb,8,2,4)
        mav=pp['mant'].float().reshape(rows,nb,8,2,4)

        for sub in range(4):
            lo=sub*64; hi=lo+64
            bidx=torch.arange(ng,device=y.device)*4+sub
            z=yy[:,:,lo:hi]
            qcur=q[:,:,lo:hi]
            gcur=grad[:,:,lo:hi]
            Hss=H[:,lo:hi,lo:hi]

            sfcur=sfv[:,bidx,0,0,0].clamp_min(2.0**-48)
            c=qcur/sfcur.unsqueeze(-1)
            den=torch.einsum('rgi,gij,rgj->rg',c,Hss,c).clamp_min(1e-20)
            num=(c*gcur).sum(-1)
            sfstar=(sfcur-num/den).clamp(min=2.0**-48,max=49152.0)

            curidx=_nearest_e6m2_index(sfcur,table)
            optidx=_nearest_e6m2_index(sfstar,table)
            idxs=[
                curidx,
                (curidx-1).clamp(0,last),(curidx+1).clamp(0,last),
                (optidx-1).clamp(0,last),optidx,(optidx+1).clamp(0,last),
            ]

            l2=l2v[:,bidx].reshape(rows,ng,8,1,1)
            l3=l3v[:,bidx].reshape(rows,ng,8,2,1)
            za=z.reshape(rows,ng,8,2,4)
            sg=torch.sign(za)

            costs=[torch.zeros((rows,ng),device=y.device)]
            qlist=[qcur]
            packs=[None]
            for idx in idxs:
                sf=table[idx].reshape(rows,ng,1,1,1)
                eff=sf*l2*l3
                ma=(torch.round((za.abs()/eff.clamp_min(2.0**-48))*4.0)*0.25).clamp(0.0,1.75)
                sgc=torch.where(ma==0.0,torch.zeros_like(sg),sg)
                qc=(sgc*ma*eff).reshape(rows,ng,64)
                d=qc-qcur
                dc=2.0*(d*gcur).sum(-1)+torch.einsum('rgi,gij,rgj->rg',d,Hss,d)
                costs.append(dc); qlist.append(qc); packs.append((sf,sgc,ma))

            choice=torch.stack(costs,0).argmin(0)
            if not bool((choice>0).any().item()):
                continue

            qstack=torch.stack(qlist,0)  # [C,R,G,64]
            x=qstack.permute(1,2,0,3)
            qi=choice[:,:,None,None].expand(rows,ng,1,64)
            qnew=x.gather(2,qi).squeeze(2)
            d=qnew-qcur
            q[:,:,lo:hi]=qnew
            grad.add_(torch.einsum('gij,rgj->rgi',H[:,:,lo:hi],d))

            for ci in range(1,len(packs)):
                mask=(choice==ci)
                if not bool(mask.any().item()):
                    continue
                sf,sgc,ma=packs[ci]
                m5=mask[:,:,None,None,None]
                sfv[:,bidx]=torch.where(m5,sf,sfv[:,bidx])
                sgv[:,bidx]=torch.where(m5,sgc,sgv[:,bidx])
                mav[:,bidx]=torch.where(m5,ma,mav[:,bidx])

        pp['scale_factor']=sfv.reshape_as(pp['scale_factor']).to(torch.bfloat16)
        pp['scale_lv2']=l2v.reshape_as(pp['scale_lv2']).to(torch.bfloat16)
        pp['scale_lv3']=l3v.reshape_as(pp['scale_lv3']).to(torch.bfloat16)
        pp['sign']=sgv.reshape_as(pp['sign']).to(torch.bfloat16)
        pp['mant']=mav.reshape_as(pp['mant']).to(torch.bfloat16)
    return pp

def _mean_sample_rms(ats,k):
    acc=torch.zeros(k,device=ats[0].device,dtype=torch.float32)
    ns=0
    for a in ats:
        aa=a.float().reshape(-1,k)
        acc.add_(aa.square().mean(0))
        ns+=1
    return torch.sqrt((acc/float(max(ns,1))).clamp_min(1e-24))

def _mean_sample_absolute_value(ats,k):
    acc=torch.zeros(k,device=ats[0].device,dtype=torch.float32)
    ns=0
    for a in ats:
        aa=a.float().reshape(-1,k)
        acc.add_(aa.abs().mean(0))
        ns+=1
    return acc/float(max(ns,1))

def _build_mass_balanced_permutation(mass,block=64):
    c=int(mass.numel())
    if c%int(block):
        return torch.argsort(mass,descending=True,stable=True)
    nb=c//int(block)
    vals=mass.detach().float().cpu().tolist()
    order=sorted(range(c),key=lambda i:(-vals[i],i))
    heap=[(0.0,j,0) for j in range(nb)]
    heapq.heapify(heap)
    buckets=[[] for _ in range(nb)]
    for i in order:
        load,j,count=heapq.heappop(heap)
        buckets[j].append(i)
        count+=1
        load+=float(vals[i])
        if count<int(block):
            heapq.heappush(heap,(load,j,count))
    flat=[i for b in buckets for i in b]
    return torch.tensor(flat,dtype=torch.long,device=mass.device)

def _prepare_linear_quantization_geometry(w,acts,perm_kind='v40',post_pressure='max',robust_phase_mix=0.0):
    k=int(w.shape[-1]); nb=k//64
    if not acts or k%64:
        s=torch.ones(k,device=w.device)
        perm=torch.arange(k,device=w.device)
        ph=torch.ones(nb,device=w.device)
        post=torch.arange(k,device=w.device)
        return s,perm,True,ph,post,w.float(),[a.float() for a in acts]

    amax=torch.stack([a.abs().amax(0) for a in acts],0).mean(0)
    wmax=w.abs().amax(0)
    smooth=_compute_smooth_scale(amax,wmax,0.50)

    if perm_kind=='mass_act':
        apre=[a.float()*smooth for a in acts]
        mass=_mean_sample_absolute_value(apre,k)
        perm=_build_mass_balanced_permutation(mass,64)
    elif perm_kind=='mass_joint':
        apre=[a.float()*smooth for a in acts]
        amass=_mean_sample_absolute_value(apre,k)
        wmass=(w.float()/smooth).abs().mean(0)
        mass=torch.sqrt((amass*wmass).clamp_min(1e-24))
        perm=_build_mass_balanced_permutation(mass,64)
    else:
        aeff=amax*smooth
        weff=wmax/smooth.clamp_min(2.0**-24)
        perm=_build_balanced_feature_permutation(torch.maximum(aeff,weff))

    wt0=(w.float()/smooth).index_select(-1,perm)
    wt0=_fast_walsh_hadamard_64(wt0)
    wb_max=wt0.abs().reshape(-1,nb,64).amax((0,2))
    wb_rms=torch.sqrt(wt0.square().reshape(-1,nb,64).mean((0,2)).clamp_min(1e-24))

    ab_max=torch.zeros(nb,dtype=torch.float32,device=w.device)
    ab_rms_acc=torch.zeros(nb,dtype=torch.float32,device=w.device)
    ats_pre=[]
    for a in acts:
        at=(a.float()*smooth).index_select(-1,perm)
        at=_fast_walsh_hadamard_64(at)
        ats_pre.append(at)
        z=at.reshape(-1,nb,64)
        ab_max=torch.maximum(ab_max,z.abs().amax((0,2)))
        ab_rms_acc.add_(z.square().mean((0,2)))
    ab_rms=torch.sqrt((ab_rms_acc/float(max(len(acts),1))).clamp_min(1e-24))

    pmax=torch.sqrt(wb_max.clamp_min(2.0**-24)/ab_max.clamp_min(2.0**-24))
    prms=torch.sqrt(wb_rms.clamp_min(2.0**-24)/ab_rms.clamp_min(2.0**-24))
    mix=float(robust_phase_mix)
    phases=torch.exp((1.0-mix)*torch.log(pmax.clamp_min(1e-24))+
                     mix*torch.log(prms.clamp_min(1e-24)))
    phases=phases.clamp(0.50,2.00)
    phases=phases/torch.exp(torch.log(phases).median())
    phases=phases.clamp(0.50,2.00)

    pv=_expand_block_values(phases)
    wt=wt0/pv
    ats=[a*pv for a in ats_pre]

    wrms=torch.sqrt(wt.square().mean(0).clamp_min(1e-24))
    arms=_mean_sample_rms(ats,k)
    pressure=torch.sqrt((wrms*arms).clamp_min(1e-24)) if post_pressure=='geom' else torch.maximum(wrms,arms)
    local=torch.argsort(pressure.reshape(nb,64),dim=1,stable=True)
    offs=(torch.arange(nb,device=w.device)*64)[:,None]
    post=(local+offs).reshape(-1)

    wt=wt.index_select(-1,post)
    ats=[a.index_select(-1,post) for a in ats]
    return smooth,perm,True,phases,post,wt,ats

def _estimate_adaptive_covariance(acts_t,k,device,group=64,alpha=.5,rmin=.40,rmax=.80,noise_gain=1.0):
    if k%group:return None,None
    ng=k//group;Cs=[];ws=[]
    for a in acts_t:
      z=a.float().reshape(-1,ng,group);n=max(int(z.shape[0]),1);C=torch.einsum('rgi,rgj->gij',z,z)/float(n);Cs.append(C);ws.append(float(n)**(1-alpha))
    if not Cs:return None,None
    W=sum(ws);H=sum(C*w for C,w in zip(Cs,ws))/W
    sc=H.diagonal(dim1=-2,dim2=-1).mean(-1).abs().clamp_min(1e-12);Hn=H/sc[:,None,None]
    # Normalize each sample by aggregate block scale so variability is comparable.
    var=torch.zeros(ng,device=device)
    for C,w in zip(Cs,ws):
      Cn=C/sc[:,None,None];d=Cn-Hn;diag=torch.diag_embed(d.diagonal(dim1=-2,dim2=-1));off=d-diag
      var += float(w)*(off.square().mean((1,2)))
    var/=W
    D=torch.diag_embed(Hn.diagonal(dim1=-2,dim2=-1));off=Hn-D;signal=off.square().mean((1,2)).clamp_min(1e-12)
    rho=(signal/(signal+float(noise_gain)*var)).clamp(float(rmin),float(rmax))
    Hr=rho[:,None,None]*Hn+(1-rho[:,None,None])*D
    return Hr,rho

def _calibrate_and_quantize_linear_weight(weight_quant,weight_scale,calib_activation_list):
    w=dequantize_nvfp4(weight_quant,weight_scale).float();acts=_decode_calibration_activations(calib_activation_list,w.shape[-1],w.device)
    s,perm,had,ph,post,wt,acts_t=_prepare_linear_quantization_geometry(w,acts,perm_kind='mass_act',post_pressure='max',robust_phase_mix=0.0);k=int(w.shape[-1])
    H64,r64=_estimate_adaptive_covariance(acts_t,k,w.device,64,.5,.40,.80,1.0);H256,r256=_estimate_adaptive_covariance(acts_t,k,w.device,256,.5,.40,.80,1.0)
    wp,wq=_quantize_with_block_hessian(wt,H64,return_dequant=True) if isinstance(H64,torch.Tensor) else _quantize_tensor_self_mse(wt,return_dequant=True)
    if isinstance(H256,torch.Tensor):wp=_refine_weight_hessian_256_chunked(wt,wp,H256,iters=6,chunk_rows=256)
    wq=_dequantize_hif4_params(wp,tuple(w.shape)).float();st=_make_quantized_weight_state('v156',s,perm,had,ph,wq,{'post_perm':post.cpu().to(torch.int32),'post_perm_enabled':True,'rho64_mean':float(r64.mean()),'rho256_mean':float(r256.mean())})
    return {'weight_params':wp,'activation_state':st}

def _refine_level3_scales_greedy(y,p,H256,iters=2):
    shape=tuple(int(s) for s in y.shape); k=shape[-1]
    if y.dim()!=2 or k%256 or not isinstance(H256,torch.Tensor):
        return p
    rows=int(y.shape[0]); ng=k//256; nb=k//64
    H=H256.to(y.device,torch.float32); pp=_clone_hif4_params(p)
    yy=y.float().reshape(rows,ng,256)
    q=_dequantize_hif4_params(pp,shape).float().reshape(rows,ng,256)
    g=torch.einsum('gij,rgi->rgj',H,q-yy)

    sf=pp['scale_factor'].float().reshape(rows,nb)
    l2=pp['scale_lv2'].float().reshape(rows,nb,8)
    l3=pp['scale_lv3'].float().reshape(rows,nb,8,2)
    sgv=pp['sign'].float().reshape(rows,nb,8,2,4)
    mav=pp['mant'].float().reshape(rows,nb,8,2,4)

    # 64 lv3 groups in each 256-group; each lv3 controls 4 contiguous values.
    sf4=sf.reshape(rows,ng,4,1,1).expand(rows,ng,4,8,2).reshape(rows,ng,64)
    l24=l2.reshape(rows,ng,4,8,1).expand(rows,ng,4,8,2).reshape(rows,ng,64)
    l34=l3.reshape(rows,ng,4,8,2).reshape(rows,ng,64)
    z4=yy.reshape(rows,ng,64,4)
    q4=q.reshape(rows,ng,64,4)
    # Exact 4x4 principal submatrices for each lv3-controlled quartet.
    H4=torch.stack([H[:,i*4:(i+1)*4,i*4:(i+1)*4] for i in range(64)],dim=1)

    for _ in range(int(iters)):
        # Toggle each E1 lv3 bit, then re-round its four mantissas exactly.
        new=torch.where(l34>1.5,torch.ones_like(l34),torch.full_like(l34,2.0))
        eff=sf4*l24*new
        ma=(torch.round((z4.abs()/eff[:,:,:,None].clamp_min(2**-48))*4.0)*0.25).clamp(0,1.75)
        sg=torch.where(ma==0,torch.zeros_like(z4),torch.sign(z4))
        qn=sg*ma*eff[:,:,:,None]
        d=qn-q4
        g4=g.reshape(rows,ng,64,4)
        delta=2*(d*g4).sum(-1)+torch.einsum('rgbi,gbij,rgbj->rgb',d,H4,d)
        best,idx=delta.min(-1)
        good=best < -1e-8
        if not bool(good.any().item()):
            break

        sel=idx[:,:,None,None].expand(rows,ng,1,4)
        dsel=d.gather(2,sel).squeeze(2)*good[:,:,None]
        qnsel=qn.gather(2,sel).squeeze(2)
        masel=ma.gather(2,sel).squeeze(2)
        sgsel=sg.gather(2,sel).squeeze(2)
        newsel=new.gather(2,idx[:,:,None]).squeeze(2)

        # Rank-4 exact gradient update against the full H256.
        de=torch.zeros((rows,ng,256),device=y.device,dtype=torch.float32)
        coord=(idx*4)[:,:,None]+torch.arange(4,device=y.device).view(1,1,4)
        de.scatter_(2,coord,dsel)
        g.add_(torch.einsum('gij,rgj->rgi',H,de))

        oldq=q4.gather(2,sel)
        q4.scatter_(2,sel,torch.where(good[:,:,None,None],qnsel[:,:,None,:],oldq))
        oldl=l34.gather(2,idx[:,:,None]).squeeze(2)
        l34.scatter_(2,idx[:,:,None],torch.where(good,newsel,oldl)[:,:,None])

        mf=mav.reshape(rows,ng,64,4)
        sfv=sgv.reshape(rows,ng,64,4)
        oldm=mf.gather(2,sel); olds=sfv.gather(2,sel)
        mf.scatter_(2,sel,torch.where(good[:,:,None,None],masel[:,:,None,:],oldm))
        sfv.scatter_(2,sel,torch.where(good[:,:,None,None],sgsel[:,:,None,:],olds))
        mav=mf.reshape(rows,nb,8,2,4)
        sgv=sfv.reshape(rows,nb,8,2,4)
        l3=l34.reshape(rows,nb,8,2)

    pp['scale_lv3']=l3.reshape_as(pp['scale_lv3']).to(torch.bfloat16)
    pp['sign']=sgv.reshape_as(pp['sign']).to(torch.bfloat16)
    pp['mant']=mav.reshape_as(pp['mant']).to(torch.bfloat16)
    return pp

def _quantize_dynamic_tensor_hessian(y,H64,H256,lv3_iters=2,base_mant_iters=4):
    if not isinstance(H64,torch.Tensor):
        return _quantize_tensor_self_mse(y,return_dequant=False)[0]
    p,_=_quantize_with_block_hessian(y,H64.to(y.device),return_dequant=False)
    if isinstance(H256,torch.Tensor):
        H=H256.to(y.device,torch.float32)
        # Proven V156 baseline first.
        p=_refine_mantissas_hessian_256(y,p,H,int(base_mant_iters))
        # Re-open the frozen level-1 E6 scale under the full H256 metric.
        p=_refine_scales_hessian_256(y,p,H,1)
        p=_refine_mantissas_hessian_256(y,p,H,1)
        # Re-open the most useful HiF4 hierarchy bit (lv3, one bit / 4 values).
        p=_refine_level3_scales_greedy(y,p,H,int(lv3_iters))
        p=_refine_mantissas_hessian_256(y,p,H,1)
    return p

def _quantize_dynamic_key(k_quant,k_scale,kv_num_heads,head_dim,k_state,lv3_iters=1):
    k=dequantize_nvfp4(k_quant,k_scale).float()
    if isinstance(k_state,dict):
        k=_apply_key_transform(k,k_state,kv_num_heads,head_dim)
    return _quantize_key_fast_translation_search(k)

def _quantize_dynamic_query(q_quant,q_scale,q_num_heads,head_dim,q_state,lv3_iters=1):
    q=dequantize_nvfp4(q_quant,q_scale).float()
    if isinstance(q_state,dict):
        q=_apply_query_transform(q,q_state,q_num_heads,head_dim)
    return _quantize_tensor_self_mse(q,return_dequant=False)[0]

def _refine_level2_scales_once(y, p, H256):
    shape = tuple(int(s) for s in y.shape)
    k = shape[-1]
    if y.dim() != 2 or k % 256 or not isinstance(H256, torch.Tensor):
        return p

    rows = int(y.shape[0])
    ng = k // 256
    nb = k // 64
    H = H256.to(y.device, torch.float32)
    pp = _clone_hif4_params(p)

    yy = y.float().reshape(rows, ng, 256)
    q = _dequantize_hif4_params(pp, shape).float().reshape(rows, ng, 256)
    g = torch.einsum('gij,rgi->rgj', H, q - yy)

    sf = pp['scale_factor'].float().reshape(rows, nb)
    l2 = pp['scale_lv2'].float().reshape(rows, nb, 8)
    l3 = pp['scale_lv3'].float().reshape(rows, nb, 8, 2)
    sgv = pp['sign'].float().reshape(rows, nb, 8, 2, 4)
    mav = pp['mant'].float().reshape(rows, nb, 8, 2, 4)

    # 32 lv2 groups per H256 group; each controls 8 values.
    sf8 = sf.reshape(rows, ng, 4, 1).expand(rows, ng, 4, 8).reshape(rows, ng, 32)
    l28 = l2.reshape(rows, ng, 32)
    l38 = l3.reshape(rows, ng, 32, 2)
    z8 = yy.reshape(rows, ng, 32, 8)
    q8 = q.reshape(rows, ng, 32, 8)

    new = torch.where(
        l28 > 1.5,
        torch.ones_like(l28),
        torch.full_like(l28, 2.0),
    )
    eff = (sf8 * new)[:, :, :, None, None] * l38[:, :, :, :, None]
    zz = z8.reshape(rows, ng, 32, 2, 4)
    ma = (
        torch.round((zz.abs() / eff.clamp_min(2.0 ** -48)) * 4.0) * 0.25
    ).clamp(0.0, 1.75)
    sg = torch.where(ma == 0.0, torch.zeros_like(zz), torch.sign(zz))
    qn = (sg * ma * eff).reshape(rows, ng, 32, 8)

    d = qn - q8
    g8 = g.reshape(rows, ng, 32, 8)
    H8 = torch.stack(
        [H[:, i * 8:(i + 1) * 8, i * 8:(i + 1) * 8] for i in range(32)],
        dim=1,
    )
    delta = (
        2.0 * (d * g8).sum(-1)
        + torch.einsum('rgbi,gbij,rgbj->rgb', d, H8, d)
    )

    # One exact best lv2 move per row x H256 group.
    best, idx = delta.min(-1)
    good = best < -1.0e-8
    if not bool(good.any().item()):
        return pp

    sel = idx[:, :, None, None].expand(rows, ng, 1, 8)

    newsel = new.gather(2, idx[:, :, None]).squeeze(2)
    oldl = l28.gather(2, idx[:, :, None]).squeeze(2)
    l28.scatter_(
        2,
        idx[:, :, None],
        torch.where(good, newsel, oldl)[:, :, None],
    )

    mf = mav.reshape(rows, ng, 32, 8)
    sfv = sgv.reshape(rows, ng, 32, 8)
    masel = ma.reshape(rows, ng, 32, 8).gather(2, sel).squeeze(2)
    sgsel = sg.reshape(rows, ng, 32, 8).gather(2, sel).squeeze(2)
    oldm = mf.gather(2, sel)
    olds = sfv.gather(2, sel)
    mf.scatter_(
        2, sel,
        torch.where(good[:, :, None, None], masel[:, :, None, :], oldm),
    )
    sfv.scatter_(
        2, sel,
        torch.where(good[:, :, None, None], sgsel[:, :, None, :], olds),
    )

    pp['scale_lv2'] = l28.reshape_as(pp['scale_lv2']).to(torch.bfloat16)
    pp['sign'] = sfv.reshape_as(pp['sign']).to(torch.bfloat16)
    pp['mant'] = mf.reshape_as(pp['mant']).to(torch.bfloat16)
    return pp

def _quantize_dynamic_activation(aq, asc, st):
    st = st if isinstance(st, dict) else {}
    y = _decode_and_transform_activation(aq, asc, st)
    post = st.get('post_perm')
    if isinstance(post, torch.Tensor):
        y = y.index_select(-1, post.to(y.device, dtype=torch.long))

    H64 = st.get('weight_hessian_blocks')
    H256 = st.get('super256_hessian_blocks')

    # Exact V158 baseline first.
    p = _quantize_dynamic_tensor_hessian(
        y, H64, H256, lv3_iters=2, base_mant_iters=4
    )

    if isinstance(H256, torch.Tensor):
        H = H256.to(y.device, torch.float32)
        # Re-open one lv2 group, then locally repair lv3/mantissa.
        p = _refine_level2_scales_once(y, p, H)
        p = _refine_mantissas_hessian_256(y, p, H, 1)
        p = _refine_level3_scales_greedy(y, p, H, 1)
        p = _refine_mantissas_hessian_256(y, p, H, 1)
    return p

def hif4_calibration_and_quantize_weight(
    weight_quant, weight_scale, calib_activation_list
):
    return _calibrate_and_quantize_linear_weight(weight_quant, weight_scale, calib_activation_list)

def hif4_dynamic_quantize_activation(
    activation_quant, activation_scale, activation_state
):
    return _quantize_dynamic_activation(
        activation_quant, activation_scale, activation_state
    )

_ATTENTION_STATE_VERSION = "v168_v111_direct_per_head"

def _rotate_heads_by_pattern(
    x: torch.Tensor,
    num_heads: int,
    head_dim: int,
    patterns: torch.Tensor,
) -> torch.Tensor:
    """
    Apply independently selected rotation to each attention head.

    patterns[h]:
        -1 -> identity
         0 -> H64
         1 -> signed H64

    Input:
        x: [..., num_heads * head_dim]

    Output:
        same shape as x
    """
    x = x.float()

    if x.shape[-1] != num_heads * head_dim:
        return x

    if not isinstance(patterns, torch.Tensor):
        return x

    if patterns.numel() != num_heads:
        return x

    # H64 requires each head to contain complete 64-value blocks.
    if head_dim % 64 != 0:
        return x

    orig_shape = x.shape

    # [..., H * D]
    # ->
    # [rows, H, D]
    y = x.reshape(-1, num_heads, head_dim)

    patterns = patterns.to(
        device=y.device,
        dtype=torch.long,
    ).reshape(num_heads)

    # Compute the two transformed candidates vectorized over all heads.
    #
    # The block rotation treats the last dimension independently, therefore
    # [rows, heads, head_dim] is safe: heads never mix.
    h64 = _rotate_feature_blocks_64(y, 0)
    sh64 = _rotate_feature_blocks_64(y, 1)

    out = y

    mask_h = (patterns == 0).reshape(1, num_heads, 1)
    mask_sh = (patterns == 1).reshape(1, num_heads, 1)

    out = torch.where(mask_h, h64, out)
    out = torch.where(mask_sh, sh64, out)

    return out.reshape(orig_shape)

def hif4_calibration_attention(
    calib_qkv_list,
    q_num_heads,
    kv_num_heads,
    head_dim,
):
    """Low-latency v111-style direct per-head attention state.

    A fixed signed-H64 basis preserves every Q/K head product. Direct tensor
    quantization generalized better than calibration partner-Hessian repair,
    so no request-specific covariance state or rotation search is retained.
    """
    if q_num_heads % kv_num_heads != 0:
        raise ValueError("q_num_heads must be divisible by kv_num_heads")
    if head_dim % _HIF4_BLOCK != 0:
        raise ValueError(
            f"per-head HiF4 requires head_dim divisible by {_HIF4_BLOCK}, "
            f"got {head_dim}"
        )

    common = {
        "version": _ATTENTION_STATE_VERSION,
        "enabled": True,
        "head_dim": int(head_dim),
        "transform_kind": "per_head_rot",
        "per_head_rotation": True,
        "beta": 0.0,
    }
    return {
        "q_state": {
            **common,
            "role": "q",
            "scale": torch.ones(q_num_heads * head_dim, dtype=torch.float32),
            "head_rotation_patterns": torch.ones(q_num_heads, dtype=torch.int8),
        },
        "k_state": {
            **common,
            "role": "k",
            "scale": torch.ones(kv_num_heads * head_dim, dtype=torch.float32),
            "head_rotation_patterns": torch.ones(kv_num_heads, dtype=torch.int8),
        },
        "v_state": {
            "version": _ATTENTION_STATE_VERSION,
            "enabled": False,
            "role": "v",
        },
    }

def hif4_dynamic_quantize_q(
    q_quant,
    q_scale,
    q_num_heads,
    head_dim,
    q_state,
):
    return _quantize_dynamic_query(
        q_quant,
        q_scale,
        q_num_heads,
        head_dim,
        q_state,
        lv3_iters=1,
    )

def hif4_dynamic_quantize_k(
    k_quant,
    k_scale,
    kv_num_heads,
    head_dim,
    k_state,
):
    return _quantize_dynamic_key(
        k_quant,
        k_scale,
        kv_num_heads,
        head_dim,
        k_state,
        lv3_iters=1,
    )

def hif4_dynamic_quantize_v(
    v_quant,
    v_scale,
    kv_num_heads,
    head_dim,
    v_state,
):
    return _quantize_value_tensor(
        v_quant,
        v_scale,
        kv_num_heads,
        head_dim,
        v_state,
    )
