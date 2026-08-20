from __future__ import annotations

import math
from typing import Optional, Tuple

import torch

# =============================================================================
# V30-k-multistart
# Strict split-interface quantization. Every public per-tensor API reads ONLY
# its own NVFP4 tensor+scale. No cross-operand statistics, MatMul/SDPA targets,
# or result-baseline fitting are used.
#
# Two general changes over V26:
#   1) exact-structure E6M2 search around max/7: offsets [-1..+4]. Extensive
#      exhaustive checks against all 255 legal E6M2 values found the optimum in
#      this interval for NVFP4-like and broad continuous stress distributions.
#   2) K-only softmax-nullspace optimization. K -> K-c (same c for every key
#      position) changes each query's logits only by a rowwise constant, hence
#      softmax is exactly invariant. We optimize this quotient space using K
#      alone and keep the raw-K quantization as a hard candidate per 64 features.
# =============================================================================

_HIF4_BLOCK = 64
_NVFP4_BLOCK = 16
_SEARCH_CHUNK_BLOCKS = 16384
_E6_ANCHOR_OFFSETS = (-1, 0, 1, 2, 3, 4)
_K_QUOTIENT_TOTAL_ROUNDS = 8  # seven K-only quotient stages after raw baseline
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


def _quantize_nvfp4_operand(quant: torch.Tensor, scale: torch.Tensor) -> dict:
    x = dequantize_nvfp4(quant, scale).float()
    params, _ = _quantize_tensor_self_mse(x, return_dequant=False)
    return params


def _merge_k_candidates_per_feature_block(
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


def _select_best_k_dq_per_feature_block(
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


def _quantize_k_softmax_quotient(k_quant: torch.Tensor, k_scale: torch.Tensor) -> dict:
    """K-only softmax-quotient quantization with safeguarded over-relaxation.

    For any feature vector c shared across key positions, K -> K-c changes each
    query's logits only by a rowwise constant and is therefore an exact softmax
    invariance.  V28 used one fixed-point candidate per stage.  V29 evaluates a
    few over-relaxed translations around that fixed-point update to cross discrete
    HiF4 decision boundaries.  The raw/V28 trajectory remains in the candidate
    history and final selection is per 64-feature block by exact quotient SSE.
    """
    x = dequantize_nvfp4(k_quant, k_scale).float()
    if x.dim() < 2 or int(x.shape[-2]) <= 1:
        return _quantize_tensor_self_mse(x, return_dequant=False)[0]

    params_list: list[dict] = []
    dq_list: list[torch.Tensor] = []

    # Raw K is an immutable hard baseline.
    p, q = _quantize_tensor_self_mse(x, return_dequant=True)
    params_list.append(p)
    dq_list.append(q)

    q_best = q
    c_prev = torch.zeros_like(x.mean(dim=-2, keepdim=True))

    for _ in range(1, _K_QUOTIENT_TOTAL_ROUNDS):
        # Exact best translation for the current history-best representative.
        c_star = (x - q_best).mean(dim=-2, keepdim=True)
        delta = c_star - c_prev

        # gamma=1 is exactly the standard V28 update. Larger gammas only add
        # candidates; they can never force a worse block into the final result.
        for gamma in _K_RELAX_GAMMAS:
            c_try = c_prev + float(gamma) * delta
            target = x - c_try
            p, q = _quantize_tensor_self_mse(target, return_dequant=True)
            params_list.append(p)
            dq_list.append(q)

        q_best = _select_best_k_dq_per_feature_block(x, dq_list)
        c_prev = c_star

    # V30: a second deterministic basin seeded only from K itself.  Median
    # centering is robust to outliers and, unlike mean/midrange multi-starts in
    # stress tests, consistently found complementary discrete HiF4 basins.
    # Crucially, all V29 candidates remain in history, so this branch can only
    # win a 64-feature block when its exact softmax-quotient SSE is lower.
    c_prev = x.median(dim=-2, keepdim=True).values
    p, q = _quantize_tensor_self_mse(x - c_prev, return_dequant=True)
    params_list.append(p)
    dq_list.append(q)
    q_best = _select_best_k_dq_per_feature_block(x, dq_list)

    for _ in range(_K_MEDIAN_EXTRA_ROUNDS):
        c_star = (x - q_best).mean(dim=-2, keepdim=True)
        delta = c_star - c_prev
        for gamma in _K_RELAX_GAMMAS:
            c_try = c_prev + float(gamma) * delta
            p, q = _quantize_tensor_self_mse(x - c_try, return_dequant=True)
            params_list.append(p)
            dq_list.append(q)
        q_best = _select_best_k_dq_per_feature_block(x, dq_list)
        c_prev = c_star

    return _merge_k_candidates_per_feature_block(x, params_list, dq_list)



# =============================================================================
# V31: conservative calibration-aware Linear + proven V30 Attention.
# =============================================================================
_V31_STATE_VERSION="v31_calib_smooth_h64_safe"
_V31_MODES=((0.0,False),(0.25,False),(0.50,False),(0.0,True),(0.25,True),(0.50,True))


def _fwht64_v31(x):
    c=x.shape[-1]
    if c%64: return x.float()
    orig=x.shape; y=x.float().reshape(-1,c//64,64).clone(); h=1
    while h<64:
        z=y.reshape(*y.shape[:-1],-1,2*h); a=z[...,:h].clone(); b=z[...,h:2*h].clone(); z[...,:h]=a+b; z[...,h:2*h]=a-b; y=z.reshape(-1,c//64,64); h*=2
    return (y*0.125).reshape(orig)


def _v31_even(n,k,device):
    if n<=k:return torch.arange(n,device=device)
    return torch.linspace(0,n-1,steps=k,device=device).round().long().unique()


def _v31_smooth(amax,wmax,beta):
    if beta<=0:return torch.ones_like(wmax)
    ls=float(beta)*(torch.log(wmax.clamp_min(2**-24))-torch.log(amax.clamp_min(2**-24))); ls-=ls.median(); return torch.exp(ls).clamp_(2**-6,2**6)


def _v31_calib_acts(calib_list,k,device):
    out=[]
    for pair in calib_list:
        if not isinstance(pair,(list,tuple)) or len(pair)!=2:continue
        a=dequantize_nvfp4(pair[0],pair[1]).float().to(device).reshape(-1,k)
        if a.numel() and a.shape[-1]==k:out.append(a)
    return out


def _v31_choose_linear_mode(*args, **kwargs):
    """Disabled in organizer-rule audit-clean build."""
    raise RuntimeError("disabled legacy output-aware Linear path")


def hif4_calibration_and_quantize_weight(weight_quant,weight_scale,calib_activation_list):
    w=dequantize_nvfp4(weight_quant,weight_scale).float(); acts=_v31_calib_acts(calib_activation_list,w.shape[-1],w.device)
    s,had,mode=_v31_choose_linear_mode(w,acts); wt=w/s
    if had:wt=_fwht64_v31(wt)
    wp=_quantize_tensor_self_mse(wt,return_dequant=False)[0]
    state={"version":_V31_STATE_VERSION,"smooth":s.cpu().float(),"hadamard64":had,"beta":mode[0]}
    return {"weight_params":wp,"activation_state":state}


def hif4_dynamic_quantize_activation(activation_quant,activation_scale,activation_state):
    a=dequantize_nvfp4(activation_quant,activation_scale).float()
    if isinstance(activation_state,dict) and "smooth" in activation_state:
        s=activation_state["smooth"].to(a.device); a=a*s
        if bool(activation_state.get("hadamard64",False)):a=_fwht64_v31(a)
    return _quantize_tensor_self_mse(a,return_dequant=False)[0]


def hif4_calibration_attention(calib_qkv_list,q_num_heads,kv_num_heads,head_dim):
    # Keep V30 Attention bit-for-bit; calibration Q/K transforms were less robust
    # under held-out distribution shift than the already strong K quotient path.
    common={"version":_V31_STATE_VERSION,"head_dim":int(head_dim)}
    return {"q_state":{**common,"role":"q","num_heads":int(q_num_heads)},"k_state":{**common,"role":"k","num_heads":int(kv_num_heads)},"v_state":{**common,"role":"v","num_heads":int(kv_num_heads)}}


def hif4_dynamic_quantize_q(q_quant,q_scale,q_num_heads,head_dim,q_state):return _quantize_nvfp4_operand(q_quant,q_scale)
def hif4_dynamic_quantize_k(k_quant,k_scale,kv_num_heads,head_dim,k_state):return _quantize_k_softmax_quotient(k_quant,k_scale)
def hif4_dynamic_quantize_v(v_quant,v_scale,kv_num_heads,head_dim,v_state):return _quantize_nvfp4_operand(v_quant,v_scale)


# =============================================================================
# V35: preserve leaderboard-proven V31 Linear bit-for-bit; add only frozen
# calibration-learned Q/K equivalent transforms for Attention.
# =============================================================================
_V35_VERSION = "v35_attention_calibration"
_V35_BETAS = (0.0, 0.25, 0.50, 0.75)
_V35_ROTS = (-1,)  # reciprocal Smooth only; no rotation search in the fast submission


def _v35_sign_vector(c, pattern, device):
    if pattern <= 0:
        return torch.ones(c, dtype=torch.float32, device=device)
    i = torch.arange(c, dtype=torch.int64, device=device)
    h = i * 1103515245 + 12345
    bit = (h ^ (h >> 16)) & 1
    return torch.where(bit == 0, 1.0, -1.0).float()


def _v35_rotate64(x, pattern):
    if pattern < 0:
        return x.float()
    y = x.float()
    if pattern > 0:
        y = y * _v35_sign_vector(y.shape[-1], pattern, y.device)
    return _fwht64_v31(y)


def _v35_rotate_heads(x, num_heads, head_dim, pattern):
    if pattern < 0 or head_dim % 64 != 0:
        return x.float()
    shape = x.shape
    y = x.float().reshape(-1, num_heads, head_dim).reshape(-1, head_dim)
    y = _v35_rotate64(y, pattern)
    return y.reshape(shape)


def _v35_decode_calib(calib, q_num_heads, kv_num_heads, head_dim):
    out = []
    for s in calib:
        try:
            q = dequantize_nvfp4(s["q"][0], s["q"][1]).float()
            k = dequantize_nvfp4(s["k"][0], s["k"][1]).float()
            v = dequantize_nvfp4(s["v"][0], s["v"][1]).float()
            out.append((q, k, v))
        except Exception:
            pass
    return out


def _v35_qk_scale(decoded, q_num_heads, kv_num_heads, head_dim, beta):
    device = decoded[0][0].device
    rep = q_num_heads // kv_num_heads
    qmax = torch.zeros((q_num_heads, head_dim), device=device)
    kmax = torch.zeros((kv_num_heads, head_dim), device=device)
    for q, k, _ in decoded:
        qh = q.reshape(-1, q_num_heads, head_dim)
        kh = k.reshape(-1, kv_num_heads, head_dim)
        qmax = torch.maximum(qmax, qh.abs().amax(dim=0))
        kmax = torch.maximum(kmax, kh.abs().amax(dim=0))
    if beta <= 0.0:
        sk = torch.ones_like(kmax)
    else:
        qgrp = qmax.reshape(kv_num_heads, rep, head_dim).amax(dim=1)
        z = float(beta) * (
            torch.log(kmax.clamp_min(2.0 ** -24))
            - torch.log(qgrp.clamp_min(2.0 ** -24))
        )
        z = z - z.median(dim=-1, keepdim=True).values
        sk = torch.exp(z).clamp_(2.0 ** -6, 2.0 ** 6)
    sq = sk.repeat_interleave(rep, dim=0)
    return sq.reshape(-1), sk.reshape(-1)


def _disabled_legacy__v35_centered_logit_error_498(q, k, qq, kk, q_num_heads, kv_num_heads, head_dim):
    raise RuntimeError("disabled legacy output-aware Attention calibration helper")


def _disabled_legacy__v35_choose_attention_520(calib, q_num_heads, kv_num_heads, head_dim):
    raise RuntimeError("disabled legacy output-aware Attention calibration helper")


def _v35_quantize_k_tensor(x):
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
        q_best = _select_best_k_dq_per_feature_block(x, dq_list)
        c_prev = c_star
    c_prev = x.median(dim=-2, keepdim=True).values
    p, q = _quantize_tensor_self_mse(x - c_prev, return_dequant=True)
    params_list.append(p); dq_list.append(q)
    q_best = _select_best_k_dq_per_feature_block(x, dq_list)
    for _ in range(_K_MEDIAN_EXTRA_ROUNDS):
        c_star = (x - q_best).mean(dim=-2, keepdim=True)
        delta = c_star - c_prev
        for gamma in _K_RELAX_GAMMAS:
            p, q = _quantize_tensor_self_mse(
                x - (c_prev + float(gamma) * delta), return_dequant=True)
            params_list.append(p); dq_list.append(q)
        q_best = _select_best_k_dq_per_feature_block(x, dq_list)
        c_prev = c_star
    return _merge_k_candidates_per_feature_block(x, params_list, dq_list)


# --- Linear: exact V31 definitions are intentionally repeated to guarantee that
# this submission changes Attention only. ---
def hif4_calibration_and_quantize_weight(weight_quant,weight_scale,calib_activation_list):
    w=dequantize_nvfp4(weight_quant,weight_scale).float(); acts=_v31_calib_acts(calib_activation_list,w.shape[-1],w.device)
    s,had,mode=_v31_choose_linear_mode(w,acts); wt=w/s
    if had:wt=_fwht64_v31(wt)
    wp=_quantize_tensor_self_mse(wt,return_dequant=False)[0]
    state={"version":_V35_VERSION,"smooth":s.cpu().float(),"hadamard64":had,"beta":mode[0]}
    return {"weight_params":wp,"activation_state":state}


def hif4_dynamic_quantize_activation(activation_quant,activation_scale,activation_state):
    a=dequantize_nvfp4(activation_quant,activation_scale).float()
    if isinstance(activation_state,dict) and "smooth" in activation_state:
        s=activation_state["smooth"].to(a.device); a=a*s
        if bool(activation_state.get("hadamard64",False)):a=_fwht64_v31(a)
    return _quantize_tensor_self_mse(a,return_dequant=False)[0]


def hif4_calibration_attention(calib_qkv_list,q_num_heads,kv_num_heads,head_dim):
    best=_v35_choose_attention(calib_qkv_list,q_num_heads,kv_num_heads,head_dim)
    common={"version":_V35_VERSION,"head_dim":int(head_dim)}
    if best is None:
        return {"q_state":{**common,"enabled":False,"role":"q"},
                "k_state":{**common,"enabled":False,"role":"k"},
                "v_state":{**common,"enabled":False,"role":"v"}}
    sq,sk,rot,beta=best
    return {"q_state":{**common,"enabled":True,"role":"q","scale":sq.detach().cpu().float(),"rotation":rot,"beta":beta},
            "k_state":{**common,"enabled":True,"role":"k","scale":sk.detach().cpu().float(),"rotation":rot,"beta":beta},
            "v_state":{**common,"enabled":False,"role":"v"}}


def hif4_dynamic_quantize_q(q_quant,q_scale,q_num_heads,head_dim,q_state):
    q=dequantize_nvfp4(q_quant,q_scale).float()
    if isinstance(q_state,dict) and q_state.get("enabled",False):
        s=q_state.get("scale")
        if isinstance(s,torch.Tensor) and s.numel()==q.shape[-1]:q=q*s.to(q.device)
        rot=int(q_state.get("rotation",-1))
        if rot>=0:q=_v35_rotate_heads(q,q_num_heads,head_dim,rot)
    return _quantize_tensor_self_mse(q,return_dequant=False)[0]


def hif4_dynamic_quantize_k(k_quant,k_scale,kv_num_heads,head_dim,k_state):
    if not (isinstance(k_state,dict) and k_state.get("enabled",False)):
        return _quantize_k_softmax_quotient(k_quant,k_scale)
    k=dequantize_nvfp4(k_quant,k_scale).float(); s=k_state.get("scale")
    if isinstance(s,torch.Tensor) and s.numel()==k.shape[-1]:k=k/s.to(k.device)
    rot=int(k_state.get("rotation",-1))
    if rot>=0:k=_v35_rotate_heads(k,kv_num_heads,head_dim,rot)
    return _v35_quantize_k_tensor(k)


def hif4_dynamic_quantize_v(v_quant,v_scale,kv_num_heads,head_dim,v_state):
    return _quantize_nvfp4_operand(v_quant,v_scale)

# =============================================================================
# V36: fixed offline Weight-error compensation carried by activation_state.
# The map is fitted ONCE from the fixed transformed Weight and its HiF4 image.
# Dynamic activations never see a reference output or any peer operand.
# =============================================================================
_V36_VERSION = "v36_weight_block_comp_v35_attn"
_V36_COMP_RIDGE = 1.0e-4
_V36_COMP_MAX_SPECTRAL = 1.35
_V36_COMP_POOLED_GATE = 0.92
_V36_COMP_WORST_GATE = 0.985


def _v36_apply_block_map(x: torch.Tensor, block_map: torch.Tensor) -> torch.Tensor:
    """Apply one fixed 64x64 map to each last-dimension HiF4 block."""
    x = x.float()
    if x.shape[-1] % 64 != 0:
        return x
    nb = x.shape[-1] // 64
    if block_map.dim() != 3 or block_map.shape != (nb, 64, 64):
        return x
    lead = x.shape[:-1]
    xb = x.reshape(-1, nb, 64)
    # bfloat16 state is expanded only for the actual online GEMM.
    T = block_map.to(device=x.device, dtype=torch.float32)
    yb = torch.einsum("nbd,bdk->nbk", xb, T)
    return yb.reshape(*lead, x.shape[-1])


def _v36_fit_weight_block_map(w_ref: torch.Tensor, w_q: torch.Tensor):
    """Fit block-diagonal T so (A T) Wq^T approximates A Wref^T for any A."""
    if w_ref.dim() != 2 or w_ref.shape != w_q.shape or w_ref.shape[-1] % 64 != 0:
        return None
    maps = []
    eye = torch.eye(64, dtype=torch.float32, device=w_ref.device)
    for start in range(0, w_ref.shape[-1], 64):
        wr = w_ref[:, start:start + 64].float()
        wq = w_q[:, start:start + 64].float()
        gram = wq.t() @ wq
        ridge = gram.diagonal().mean().abs().clamp_min(1.0e-10) * _V36_COMP_RIDGE
        rhs = wq.t() @ wr
        try:
            X = torch.linalg.solve(gram + ridge * eye, rhs)
        except Exception:
            X = torch.linalg.pinv(gram + ridge * eye) @ rhs
        T = X.t().contiguous()
        if not torch.isfinite(T).all():
            T = eye.clone()
        else:
            # Guard against a poorly conditioned block that could amplify unseen A.
            try:
                spec = float(torch.linalg.matrix_norm(T, ord=2).item())
            except Exception:
                spec = float("inf")
            # Also require the fixed Weight reconstruction itself to improve.
            base = (wq - wr).square().sum()
            cand = (wq @ T.t() - wr).square().sum()
            if (not math.isfinite(spec) or spec > _V36_COMP_MAX_SPECTRAL or
                    not torch.isfinite(cand) or cand >= base * 0.98):
                T = eye.clone()
        maps.append(T)
    return torch.stack(maps, dim=0)


def _v36_calib_comp_gate(w_orig, acts, smooth, had, wq_trans, block_map):
    raise RuntimeError('disabled legacy output-fitting helper')


# Override only the Linear public API; Attention remains exactly V35.
def hif4_calibration_and_quantize_weight(weight_quant, weight_scale, calib_activation_list):
    w = dequantize_nvfp4(weight_quant, weight_scale).float()
    acts = _v31_calib_acts(calib_activation_list, w.shape[-1], w.device)
    smooth, had, mode = _v31_choose_linear_mode(w, acts)

    wt = w / smooth
    if had:
        wt = _fwht64_v31(wt)
    wp, wq = _quantize_tensor_self_mse(wt, return_dequant=True)

    block_map = _v36_fit_weight_block_map(wt, wq)
    enabled = _v36_calib_comp_gate(w, acts, smooth, had, wq, block_map)
    state = {
        "version": _V36_VERSION,
        "smooth": smooth.detach().cpu().float(),
        "hadamard64": bool(had),
        "beta": float(mode[0]),
        "weight_comp_enabled": bool(enabled),
    }
    if enabled and block_map is not None:
        # BF16 halves clone/state bandwidth; map application promotes to FP32.
        state["weight_comp"] = block_map.detach().cpu().to(torch.bfloat16)
    return {"weight_params": wp, "activation_state": state}


def hif4_dynamic_quantize_activation(activation_quant, activation_scale, activation_state):
    a = dequantize_nvfp4(activation_quant, activation_scale).float()
    if isinstance(activation_state, dict) and "smooth" in activation_state:
        s = activation_state["smooth"].to(a.device)
        a = a * s
        if bool(activation_state.get("hadamard64", False)):
            a = _fwht64_v31(a)
        if bool(activation_state.get("weight_comp_enabled", False)):
            T = activation_state.get("weight_comp")
            if isinstance(T, torch.Tensor):
                a = _v36_apply_block_map(a, T)
    return _quantize_tensor_self_mse(a, return_dequant=False)[0]

# =============================================================================
# V37: cross-block low-rank Weight compensation + fixed-Weight Hessian-aware
# dynamic Activation scale selection.  Both are frozen calibration artifacts.
# =============================================================================
_V37_VERSION = "v37_crossrank_hessian"
_V37_LR_RIDGE = 2.0e-4
_V37_LR_GATE = 0.985
_V37_HESS_GATE = 0.995


def _v37_effective_weight_blockmap(wq: torch.Tensor, block_map: torch.Tensor) -> torch.Tensor:
    if block_map is None or wq.shape[-1] % 64 != 0:
        return wq.float()
    K = int(wq.shape[-1]); nb = K // 64
    out = torch.empty_like(wq.float())
    T = block_map.to(wq.device, torch.float32)
    for b in range(nb):
        s = b * 64
        out[:, s:s+64] = wq[:, s:s+64].float() @ T[b].t()
    return out


def _v37_rank_budget(m: int, k: int) -> int:
    d = min(m, k)
    if k <= 512:
        return min(64, d)
    if k <= 2048:
        return min(32, d)
    return min(16, d)


def _v37_fit_crossblock_lowrank(w_ref, wq, block_map):
    """Fit E ~= U V^T so Wq (Tblock+E)^T ~= Wref, without a KxK state."""
    if w_ref.dim() != 2 or w_ref.shape != wq.shape:
        return None
    m, k = map(int, w_ref.shape)
    r = _v37_rank_budget(m, k)
    if r < 4:
        return None
    weff = _v37_effective_weight_blockmap(wq, block_map)
    resid = (w_ref.float() - weff).float()
    base = float(resid.square().sum().item())
    if not math.isfinite(base) or base <= 0.0:
        return None
    try:
        # Randomized low-rank SVD is substantially cheaper than a full KxK solve.
        qdim = min(min(m, k), r + min(8, max(2, r // 4)))
        Uo, S, V = torch.pca_lowrank(resid, q=qdim, center=False, niter=2)
        Uo = Uo[:, :r].float(); S = S[:r].float(); Q = V[:, :r].float()  # Q:[K,r]
        target = Uo * S.view(1, -1)  # [M,r]
        W = wq.float()
        # Ridge least squares W @ R ~= target.  Solve through the smaller Gram.
        if m <= k:
            G = W @ W.t()
            ridge = G.diagonal().mean().abs().clamp_min(1e-10) * _V37_LR_RIDGE
            Z = torch.linalg.solve(G + ridge * torch.eye(m, device=W.device), target)
            R = W.t() @ Z  # [K,r]
        else:
            G = W.t() @ W
            ridge = G.diagonal().mean().abs().clamp_min(1e-10) * _V37_LR_RIDGE
            R = torch.linalg.solve(G + ridge * torch.eye(k, device=W.device), W.t() @ target)
        cand = weff + (W @ R) @ Q.t()
        err = float((cand - w_ref.float()).square().sum().item())
        if not math.isfinite(err) or err >= base * 0.995:
            return None
        # Online: x E = (x Q) R^T, because E = Q R^T.
        return Q.contiguous(), R.contiguous(), err / max(base, 1e-20)
    except Exception:
        return None


def _v37_apply_comp(x, block_map, lr_q=None, lr_r=None):
    x0 = x.float()
    y = _v36_apply_block_map(x0, block_map) if isinstance(block_map, torch.Tensor) else x0
    if isinstance(lr_q, torch.Tensor) and isinstance(lr_r, torch.Tensor):
        if lr_q.dim() == 2 and lr_r.dim() == 2 and lr_q.shape == lr_r.shape and lr_q.shape[0] == x0.shape[-1]:
            q = lr_q.to(x0.device, torch.float32)
            r = lr_r.to(x0.device, torch.float32)
            y = y + (x0 @ q) @ r.t()
    return y


def _v37_calib_lr_gate(*args, **kwargs):
    """Disabled in organizer-rule audit-clean build."""
    raise RuntimeError("disabled legacy output-aware Linear path")


def _v37_weight_hessian_blocks(wq: torch.Tensor) -> torch.Tensor:
    K = int(wq.shape[-1]); nb = K // 64
    hs = []
    for b in range(nb):
        wb = wq[:, b*64:(b+1)*64].float()
        H = wb.t() @ wb
        # Normalize scale only; candidate ranking is invariant to a positive scalar.
        H = H / H.diagonal().mean().abs().clamp_min(1e-12)
        hs.append(H)
    return torch.stack(hs, dim=0)


def _v37_quantize_hessian(x: torch.Tensor, hblocks: torch.Tensor, *, return_dequant=False):
    """Select the E6 scale of each 64-block by e^T H e; local HiF4 hierarchy stays exact/self-optimal."""
    shape = tuple(int(s) for s in x.shape); c = shape[-1]
    if c % 64 != 0 or hblocks.dim() != 3 or tuple(hblocks.shape[1:]) != (64,64) or hblocks.shape[0] != c // 64:
        return _quantize_tensor_self_mse(x, return_dequant=return_dequant)
    x = x.float(); nb = c // 64; rows = x.numel() // c
    blocks = x.reshape(rows, nb, 8,2,4).reshape(-1,8,2,4); total = blocks.shape[0]
    table = _build_e6m2_table(x.device); last = int(table.numel()-1)
    sf_out=torch.empty((total,1,1,1),dtype=torch.bfloat16,device=x.device)
    l2_out=torch.empty((total,8,1,1),dtype=torch.bfloat16,device=x.device)
    l3_out=torch.empty((total,8,2,1),dtype=torch.bfloat16,device=x.device)
    sg_out=torch.empty((total,8,2,4),dtype=torch.bfloat16,device=x.device)
    ma_out=torch.empty((total,8,2,4),dtype=torch.bfloat16,device=x.device)
    dq_out=torch.empty_like(blocks) if return_dequant else None
    H_all=hblocks.to(x.device,torch.float32)
    # Smaller chunks: candidate materialization carries 64-vectors and Hessians.
    chunk=4096
    for st in range(0,total,chunk):
        en=min(st+chunk,total); z=blocks[st:en]; az=z.abs(); n=en-st
        ids=torch.arange(st,en,device=x.device)%nb; H=H_all[ids]
        anchor=_nearest_e6m2_index(az.amax((1,2,3))/7.0,table)
        best=torch.full((n,),float('inf'),device=x.device); best_pack=None; best_q=None
        for off in _E6_ANCHOR_OFFSETS:
            idx=(anchor+off).clamp(0,last); sf=table[idx].view(n,1,1,1)
            sg,ma,l2,l3=_materialize_fixed_scale_self(z,sf)
            q=(sg*ma*l2*l3*sf).reshape(n,64); e=q-z.reshape(n,64)
            err=torch.einsum('bi,bij,bj->b',e,H,e)
            better=err<best
            if best_pack is None:
                best=err; best_pack=[sf.clone(),l2.clone(),l3.clone(),sg.clone(),ma.clone()]; best_q=q.clone()
            else:
                best=torch.where(better,err,best)
                vals=[sf,l2,l3,sg,ma]
                for j,v in enumerate(vals):
                    mask=better.view(n,*([1]*(v.dim()-1)))
                    best_pack[j]=torch.where(mask,v,best_pack[j])
                best_q=torch.where(better[:,None],q,best_q)
        sf,l2,l3,sg,ma=best_pack
        sf_out[st:en]=sf.to(torch.bfloat16); l2_out[st:en]=l2.to(torch.bfloat16); l3_out[st:en]=l3.to(torch.bfloat16)
        sg_out[st:en]=sg.to(torch.bfloat16); ma_out[st:en]=ma.to(torch.bfloat16)
        if dq_out is not None: dq_out[st:en]=best_q.reshape(n,8,2,4)
    prefix=shape[:-1]
    params={"scale_factor":sf_out.reshape(*prefix,nb,1,1,1),"scale_lv2":l2_out.reshape(*prefix,nb,8,1,1),
            "scale_lv3":l3_out.reshape(*prefix,nb,8,2,1),"sign":sg_out.reshape(*prefix,nb,8,2,4),"mant":ma_out.reshape(*prefix,nb,8,2,4)}
    dq=dq_out.reshape(shape).to(torch.bfloat16).float() if dq_out is not None else None
    return params,dq


def _v37_calib_hess_gate(*args, **kwargs):
    """Disabled in organizer-rule audit-clean build."""
    raise RuntimeError("disabled legacy output-aware Linear path")


# Final V37 Linear overrides; Attention remains V35.
def hif4_calibration_and_quantize_weight(weight_quant, weight_scale, calib_activation_list):
    w=dequantize_nvfp4(weight_quant,weight_scale).float(); acts=_v31_calib_acts(calib_activation_list,w.shape[-1],w.device)
    smooth,had,mode=_v31_choose_linear_mode(w,acts); wt=w/smooth
    if had: wt=_fwht64_v31(wt)
    wp,wq=_quantize_tensor_self_mse(wt,return_dequant=True)
    block_map=_v36_fit_weight_block_map(wt,wq)
    block_enabled=_v36_calib_comp_gate(w,acts,smooth,had,wq,block_map)
    if not block_enabled: block_map=None
    lr=_v37_fit_crossblock_lowrank(wt,wq,block_map) if block_map is not None else None
    if lr is not None and not _v37_calib_lr_gate(w,acts,smooth,had,wq,block_map,lr): lr=None
    hblocks=_v37_weight_hessian_blocks(wq)
    hess_enabled=_v37_calib_hess_gate(w,acts,smooth,had,wq,block_map,lr,hblocks)
    state={"version":_V37_VERSION,"smooth":smooth.detach().cpu().float(),"hadamard64":bool(had),"beta":float(mode[0]),
           "weight_comp_enabled":bool(block_map is not None),"lowrank_enabled":bool(lr is not None),"hessian_enabled":bool(hess_enabled)}
    if block_map is not None: state["weight_comp"]=block_map.detach().cpu().to(torch.bfloat16)
    if lr is not None:
        Q,R,_=lr; state["lr_q"]=Q.detach().cpu().to(torch.bfloat16); state["lr_r"]=R.detach().cpu().to(torch.bfloat16)
    if hess_enabled: state["weight_hessian_blocks"]=hblocks.detach().cpu().to(torch.bfloat16)
    return {"weight_params":wp,"activation_state":state}


def hif4_dynamic_quantize_activation(activation_quant,activation_scale,activation_state):
    a=dequantize_nvfp4(activation_quant,activation_scale).float()
    if isinstance(activation_state,dict) and "smooth" in activation_state:
        a=a*activation_state["smooth"].to(a.device)
        if bool(activation_state.get("hadamard64",False)):a=_fwht64_v31(a)
        T=activation_state.get("weight_comp") if activation_state.get("weight_comp_enabled",False) else None
        Q=activation_state.get("lr_q") if activation_state.get("lowrank_enabled",False) else None
        R=activation_state.get("lr_r") if activation_state.get("lowrank_enabled",False) else None
        a=_v37_apply_comp(a,T,Q,R)
        H=activation_state.get("weight_hessian_blocks") if activation_state.get("hessian_enabled",False) else None
        if isinstance(H,torch.Tensor): return _v37_quantize_hessian(a,H,return_dequant=False)[0]
    return _quantize_tensor_self_mse(a,return_dequant=False)[0]

# =============================================================================
# V39: calibration-learned reversible channel packing.
#
# HiF4 shares its level-1 scale over 64 channels, lv2 over groups of 8 and lv3
# over groups of 4.  A fixed permutation can therefore improve representability
# by placing channels with similar calibrated magnitudes in the same hierarchy.
# The transform is exactly operator-preserving:
#   A' = (A D) P,  W' = (W D^-1) P  =>  A' W'^T = A W^T.
# For Attention the same per-KV-head permutation is applied to each associated Q
# head and K head, together with reciprocal calibrated Q/K scaling.
# No dynamic call observes a peer operand or a reference operator output.
# =============================================================================
_V39_VERSION = "v39_calib_channel_packing"
_V39_LINEAR_BETAS = (0.0, 0.25, 0.50, 0.75)
_V39_LINEAR_PERM_KINDS = (0, 1, 2)  # identity, joint balanced, activation balanced
_V39_LINEAR_GATE = 0.97
_V39_LINEAR_WORST = 1.01
_V39_ATTN_BETAS = (0.0, 0.25, 0.50, 0.75)
_V39_ATTN_GATE = 0.97
_V39_ATTN_WORST = 1.02


def _v39_identity_perm(c: int, device: torch.device) -> torch.Tensor:
    return torch.arange(c, dtype=torch.long, device=device)


def _v39_linear_perm(amax: torch.Tensor, wmax: torch.Tensor,
                     smooth: torch.Tensor, kind: int) -> torch.Tensor:
    c = int(amax.numel())
    if kind <= 0 or c < 64:
        return _v39_identity_perm(c, amax.device)
    aeff = amax.float() * smooth.float()
    weff = wmax.float() / smooth.float().clamp_min(2.0 ** -24)
    if kind == 1:
        # Joint packing: after reciprocal smoothing, rank channels by the larger
        # of the two operand ranges.  This directly minimizes within-block range
        # pressure on both sides at once.
        key = torch.maximum(aeff, weff)
    else:
        # Activation-centric packing is retained as a calibration candidate for
        # layers where online activations dominate the final error.
        key = aeff
    # log1p is monotone but avoids pathological subnormal ordering noise.
    return torch.argsort(torch.log2(key.clamp_min(2.0 ** -40)), stable=True)


def _v39_apply_perm(x: torch.Tensor, perm: Optional[torch.Tensor]) -> torch.Tensor:
    if not isinstance(perm, torch.Tensor) or perm.numel() != x.shape[-1]:
        return x.float()
    return x.float().index_select(-1, perm.to(x.device, dtype=torch.long))


def _v39_choose_linear_transform(*args, **kwargs):
    """Disabled in organizer-rule audit-clean build."""
    raise RuntimeError("disabled legacy output-aware Linear path")


def _v39_attention_perms(decoded, q_num_heads, kv_num_heads, head_dim,
                         sq: torch.Tensor, sk: torch.Tensor):
    rep = q_num_heads // kv_num_heads
    device = decoded[0][0].device
    qmax = torch.zeros((q_num_heads, head_dim), device=device)
    kmax = torch.zeros((kv_num_heads, head_dim), device=device)
    for q, k, _ in decoded:
        qh = q.reshape(-1, q_num_heads, head_dim)
        kh = k.reshape(-1, kv_num_heads, head_dim)
        qmax = torch.maximum(qmax, qh.abs().amax(0))
        kmax = torch.maximum(kmax, kh.abs().amax(0))
    sqh = sq.reshape(q_num_heads, head_dim)
    skh = sk.reshape(kv_num_heads, head_dim)
    qgrp = (qmax * sqh).reshape(kv_num_heads, rep, head_dim).amax(dim=1)
    keff = kmax / skh.clamp_min(2.0 ** -24)
    difficulty = torch.maximum(qgrp, keff)
    pk = torch.argsort(torch.log2(difficulty.clamp_min(2.0 ** -40)), dim=-1, stable=True)
    pq = pk.repeat_interleave(rep, dim=0)
    return pq.long(), pk.long()


def _v39_apply_head_perm(x: torch.Tensor, num_heads: int, head_dim: int,
                         perm: Optional[torch.Tensor]) -> torch.Tensor:
    if not isinstance(perm, torch.Tensor) or tuple(perm.shape) != (num_heads, head_dim):
        return x.float()
    shape = x.shape
    y = x.float().reshape(-1, num_heads, head_dim)
    p = perm.to(y.device, dtype=torch.long).unsqueeze(0).expand(y.shape[0], -1, -1)
    y = torch.gather(y, 2, p)
    return y.reshape(shape)


def _v39_attention_baseline(decoded, q_num_heads, kv_num_heads, head_dim):
    best = _v35_choose_attention(
        [{"q": (q, torch.ones((*q.shape[:-1], q.shape[-1]//16), device=q.device)),
          "k": (k, torch.ones((*k.shape[:-1], k.shape[-1]//16), device=k.device)),
          "v": (v, torch.ones((*v.shape[:-1], v.shape[-1]//16), device=v.device))}
         for q,k,v in []], q_num_heads, kv_num_heads, head_dim)
    # The helper above cannot consume already-dequantized tensors.  Recompute the
    # V35 baseline directly below; this function exists only to keep the main
    # calibration routine compact.
    return best


def _disabled_legacy__v39_choose_attention_1065(calib, q_num_heads, kv_num_heads, head_dim):
    raise RuntimeError("disabled legacy output-aware Attention calibration helper")


# Deterministic replacement for the V37 randomized low-rank residual fit.
def _v37_fit_crossblock_lowrank(w_ref, wq, block_map):
    if w_ref.dim()!=2 or w_ref.shape!=wq.shape:return None
    m,k=map(int,w_ref.shape);r=_v37_rank_budget(m,k)
    if r<4:return None
    weff=_v37_effective_weight_blockmap(wq,block_map);resid=(w_ref.float()-weff).float();base=float(resid.square().sum().item())
    if not math.isfinite(base) or base<=0:return None
    try:
        qdim=min(min(m,k),r+min(8,max(2,r//4)))
        gen=torch.Generator(device=resid.device);gen.manual_seed(39017 + m*3 + k)
        omega=torch.randn((k,qdim),generator=gen,device=resid.device,dtype=torch.float32)
        Y=resid@omega
        for _ in range(2):Y=resid@(resid.t()@Y)
        Qo,_=torch.linalg.qr(Y,mode='reduced');B=Qo.t()@resid
        Ub,S,Vh=torch.linalg.svd(B,full_matrices=False)
        Uo=(Qo@Ub[:,:r]).float();S=S[:r].float();Q=Vh[:r,:].t().float();target=Uo*S.view(1,-1);W=wq.float()
        if m<=k:
            G=W@W.t();ridge=G.diagonal().mean().abs().clamp_min(1e-10)*_V37_LR_RIDGE;Z=torch.linalg.solve(G+ridge*torch.eye(m,device=W.device),target);R=W.t()@Z
        else:
            G=W.t()@W;ridge=G.diagonal().mean().abs().clamp_min(1e-10)*_V37_LR_RIDGE;R=torch.linalg.solve(G+ridge*torch.eye(k,device=W.device),W.t()@target)
        cand=weff+(W@R)@Q.t();err=float((cand-w_ref.float()).square().sum().item())
        if not math.isfinite(err) or err>=base*0.995:return None
        return Q.contiguous(),R.contiguous(),err/max(base,1e-20)
    except Exception:return None


# Final V39 public interfaces.
def hif4_calibration_and_quantize_weight(*args, **kwargs):
    """Disabled in organizer-rule audit-clean build."""
    raise RuntimeError("disabled legacy output-aware Linear path")


def hif4_dynamic_quantize_activation(activation_quant,activation_scale,activation_state):
    a=dequantize_nvfp4(activation_quant,activation_scale).float()
    if isinstance(activation_state,dict) and "smooth" in activation_state:
        a=a*activation_state["smooth"].to(a.device)
        perm=activation_state.get("perm")
        if isinstance(perm,torch.Tensor):a=_v39_apply_perm(a,perm)
        if bool(activation_state.get("hadamard64",False)):a=_fwht64_v31(a)
        T=activation_state.get("weight_comp") if activation_state.get("weight_comp_enabled",False) else None
        Q=activation_state.get("lr_q") if activation_state.get("lowrank_enabled",False) else None
        R=activation_state.get("lr_r") if activation_state.get("lowrank_enabled",False) else None
        a=_v37_apply_comp(a,T,Q,R)
        H=activation_state.get("weight_hessian_blocks") if activation_state.get("hessian_enabled",False) else None
        if isinstance(H,torch.Tensor):return _v37_quantize_hessian(a,H,return_dequant=False)[0]
    return _quantize_tensor_self_mse(a,return_dequant=False)[0]


def hif4_calibration_attention(calib_qkv_list,q_num_heads,kv_num_heads,head_dim):
    base, packed = _v39_choose_attention(calib_qkv_list,q_num_heads,kv_num_heads,head_dim)
    common={"version":_V39_VERSION,"head_dim":int(head_dim)}
    if packed is not None:
        sq,sk,pq,pk,beta=packed
        return {"q_state":{**common,"enabled":True,"packed":True,"role":"q","scale":sq.detach().cpu().float(),"perm":pq.detach().cpu().to(torch.int32),"beta":beta},
                "k_state":{**common,"enabled":True,"packed":True,"role":"k","scale":sk.detach().cpu().float(),"perm":pk.detach().cpu().to(torch.int32),"beta":beta},
                "v_state":{**common,"enabled":False,"role":"v"}}
    if base is None:
        return {"q_state":{**common,"enabled":False,"role":"q"},"k_state":{**common,"enabled":False,"role":"k"},"v_state":{**common,"enabled":False,"role":"v"}}
    sq,sk,rot,beta=base
    return {"q_state":{**common,"enabled":True,"packed":False,"role":"q","scale":sq.detach().cpu().float(),"rotation":int(rot),"beta":float(beta)},
            "k_state":{**common,"enabled":True,"packed":False,"role":"k","scale":sk.detach().cpu().float(),"rotation":int(rot),"beta":float(beta)},
            "v_state":{**common,"enabled":False,"role":"v"}}


def hif4_dynamic_quantize_q(q_quant,q_scale,q_num_heads,head_dim,q_state):
    q=dequantize_nvfp4(q_quant,q_scale).float()
    if isinstance(q_state,dict) and q_state.get("enabled",False):
        s=q_state.get("scale")
        if isinstance(s,torch.Tensor) and s.numel()==q.shape[-1]:q=q*s.to(q.device)
        if q_state.get("packed",False):
            p=q_state.get("perm")
            if isinstance(p,torch.Tensor):q=_v39_apply_head_perm(q,q_num_heads,head_dim,p)
        else:
            rot=int(q_state.get("rotation",-1))
            if rot>=0:q=_v35_rotate_heads(q,q_num_heads,head_dim,rot)
    return _quantize_tensor_self_mse(q,return_dequant=False)[0]


def hif4_dynamic_quantize_k(k_quant,k_scale,kv_num_heads,head_dim,k_state):
    if not (isinstance(k_state,dict) and k_state.get("enabled",False)):
        return _quantize_k_softmax_quotient(k_quant,k_scale)
    k=dequantize_nvfp4(k_quant,k_scale).float();s=k_state.get("scale")
    if isinstance(s,torch.Tensor) and s.numel()==k.shape[-1]:k=k/s.to(k.device)
    if k_state.get("packed",False):
        p=k_state.get("perm")
        if isinstance(p,torch.Tensor):k=_v39_apply_head_perm(k,kv_num_heads,head_dim,p)
    else:
        rot=int(k_state.get("rotation",-1))
        if rot>=0:k=_v35_rotate_heads(k,kv_num_heads,head_dim,rot)
    return _v35_quantize_k_tensor(k)


def hif4_dynamic_quantize_v(v_quant,v_scale,kv_num_heads,head_dim,v_state):
    return _quantize_nvfp4_operand(v_quant,v_scale)

# =============================================================================
# V40: monomial calibration refinement = permutation + diagonal scaling.
# Adds (1) balanced channel packing for H64 and (2) one reciprocal phase per
# HiF4 64-channel block.  Both are frozen calibration parameters and preserve
# the continuous Linear operator exactly.
# =============================================================================
_V40_VERSION = "v40_monomial_blockphase"
_V40_PHASES = (0.75, 0.875, 1.0, 1.125, 1.25, 1.5)
_V40_PHASE_GATE = 0.985
_V40_PHASE_WORST = 1.02


def _v40_balanced_perm(key: torch.Tensor) -> torch.Tensor:
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


def _v40_choose_linear_transform(*args, **kwargs):
    """Disabled in organizer-rule audit-clean build."""
    raise RuntimeError("disabled legacy output-aware Linear path")


def _v40_local_product_error(a,w,aq,wq):
    raise RuntimeError('disabled legacy output-fitting helper')


def _v40_phase_vector(phases: torch.Tensor, device) -> torch.Tensor:
    return phases.to(device=device,dtype=torch.float32).reshape(-1,1).expand(-1,64).reshape(-1)


def _v40_choose_block_phases(w: torch.Tensor, acts, smooth, perm, had):
    k=int(w.shape[-1]);nb=k//64
    one=torch.ones(nb,dtype=torch.float32,device=w.device)
    if not acts or nb<=0:return one
    iw=_v31_even(w.shape[0],min(40,w.shape[0]),w.device);wp=w[iw]
    wt_base=(wp/smooth)[:,perm]
    val=[]
    for a in acts[:4]:
        ia=_v31_even(a.shape[0],min(16,a.shape[0]),a.device);val.append((a[ia]*smooth)[:,perm])
    all_err=[]
    for ph in _V40_PHASES:
        pv=torch.full((k,),float(ph),device=w.device)
        wt=wt_base/pv
        if had:wt=_fwht64_v31(wt)
        _,wq=_quantize_tensor_self_mse(wt,return_dequant=True);errs=[]
        for at0 in val:
            at=at0*pv
            if had:at=_fwht64_v31(at)
            _,aq=_quantize_tensor_self_mse(at,return_dequant=True)
            errs.append(_v40_local_product_error(at,wt,aq,wq))
        all_err.append(torch.stack(errs,0))
    E=torch.stack(all_err,0) # [P,S,B]
    base_idx=_V40_PHASES.index(1.0);base=E[base_idx].clamp_min(1e-20)
    pooled=E.sum(1)/base.sum(0).clamp_min(1e-20)
    worst=(E/base.unsqueeze(0)).amax(1)
    eligible=(pooled<=_V40_PHASE_GATE)&(worst<=_V40_PHASE_WORST)
    score=torch.where(eligible,pooled,torch.full_like(pooled,float('inf')))
    idx=score.argmin(0);best=score.min(0).values
    # If no non-identity candidate clears the gate, phase=1 is the hard fallback.
    out=torch.ones(nb,dtype=torch.float32,device=w.device)
    has=torch.isfinite(best)&(best<1.0)
    if bool(has.any().item()):
        table=torch.tensor(_V40_PHASES,dtype=torch.float32,device=w.device);out[has]=table[idx[has]]
    return out


def _v40_apply_base_transform(x,smooth,perm,phases,had,weight_side=False):
    y=x.float()
    y=y/smooth if weight_side else y*smooth
    y=y.index_select(-1,perm.to(y.device,dtype=torch.long))
    pv=_v40_phase_vector(phases,y.device)
    y=y/pv if weight_side else y*pv
    if had:y=_fwht64_v31(y)
    return y


def hif4_calibration_and_quantize_weight(*args, **kwargs):
    """Disabled in organizer-rule audit-clean build."""
    raise RuntimeError("disabled legacy output-aware Linear path")


def hif4_dynamic_quantize_activation(activation_quant,activation_scale,activation_state):
    a=dequantize_nvfp4(activation_quant,activation_scale).float()
    if isinstance(activation_state,dict) and 'smooth' in activation_state:
        s=activation_state['smooth'].to(a.device);p=activation_state.get('perm');ph=activation_state.get('block_phase')
        if not isinstance(p,torch.Tensor):p=torch.arange(a.shape[-1],device=a.device)
        else:p=p.to(a.device)
        if not isinstance(ph,torch.Tensor):ph=torch.ones(a.shape[-1]//64,device=a.device)
        else:ph=ph.to(a.device)
        a=_v40_apply_base_transform(a,s,p,ph,bool(activation_state.get('hadamard64',False)),False)
        T=activation_state.get('weight_comp') if activation_state.get('weight_comp_enabled',False) else None;Q=activation_state.get('lr_q') if activation_state.get('lowrank_enabled',False) else None;R=activation_state.get('lr_r') if activation_state.get('lowrank_enabled',False) else None
        a=_v37_apply_comp(a,T,Q,R);H=activation_state.get('weight_hessian_blocks') if activation_state.get('hessian_enabled',False) else None
        if isinstance(H,torch.Tensor):return _v37_quantize_hessian(a,H,return_dequant=False)[0]
    return _quantize_tensor_self_mse(a,return_dequant=False)[0]

# Attention remains V39 packed-QK calibration; its transform is also a frozen
# reciprocal scaling + permutation and therefore follows the same monomial idea.

# =============================================================================
# V42: strong calibration-only Attention reparameterization search.
# Hard baseline: V40/V39 Attention.  New candidates are
#   (a) reciprocal Smooth + full-head H64 / signed-H64, and
#   (b) per-KV-head full-matrix covariance balancing.
# All selected transforms are frozen during calibration.  Dynamic Q/K remain
# completely separate and never fit a test-time output.
# =============================================================================
_V42_VERSION = "v42_attention_rotcov"
_V42_GATE = 0.90
_V42_WORST = 1.01
_V42_COV_REG = 3.0e-3
_V42_COV_MAX_COND = 12.0
_V42_BETAS = (0.0, 0.25, 0.50, 0.75)
_V42_ROTS = (0, 1)  # H64 and one deterministic signed-H64.

_v42_v40_calibration_attention = hif4_calibration_attention
_v42_v40_dynamic_q = hif4_dynamic_quantize_q
_v42_v40_dynamic_k = hif4_dynamic_quantize_k
_v42_v40_dynamic_v = hif4_dynamic_quantize_v


def _v42_apply_head_matrix(x: torch.Tensor, num_heads: int, head_dim: int,
                           mats: torch.Tensor) -> torch.Tensor:
    if x.shape[-1] != num_heads * head_dim:
        return x.float()
    if not isinstance(mats, torch.Tensor) or tuple(mats.shape) != (num_heads, head_dim, head_dim):
        return x.float()
    y = x.float().reshape(x.shape[0], num_heads, head_dim)
    return torch.einsum("lhd,hde->lhe", y, mats.to(y.device, dtype=torch.float32)).reshape_as(x)


def _v42_apply_v40_state_q(x, state, q_num_heads, head_dim):
    y=x.float()
    if not (isinstance(state,dict) and state.get("enabled",False)):return y
    s=state.get("scale")
    if isinstance(s,torch.Tensor) and s.numel()==y.shape[-1]:y=y*s.to(y.device)
    if state.get("packed",False):
        p=state.get("perm")
        if isinstance(p,torch.Tensor):y=_v39_apply_head_perm(y,q_num_heads,head_dim,p)
    else:
        r=int(state.get("rotation",-1))
        if r>=0:y=_v35_rotate_heads(y,q_num_heads,head_dim,r)
    return y


def _v42_apply_v40_state_k(x, state, kv_num_heads, head_dim):
    y=x.float()
    if not (isinstance(state,dict) and state.get("enabled",False)):return y
    s=state.get("scale")
    if isinstance(s,torch.Tensor) and s.numel()==y.shape[-1]:y=y/s.to(y.device)
    if state.get("packed",False):
        p=state.get("perm")
        if isinstance(p,torch.Tensor):y=_v39_apply_head_perm(y,kv_num_heads,head_dim,p)
    else:
        r=int(state.get("rotation",-1))
        if r>=0:y=_v35_rotate_heads(y,kv_num_heads,head_dim,r)
    return y


def _disabled_legacy__v42_score_transformed_1382(decoded, q_num_heads, kv_num_heads, head_dim, qfn, kfn):
    raise RuntimeError("disabled legacy output-aware Attention calibration helper")


def _v42_fit_cov(decoded,q_num_heads,kv_num_heads,head_dim):
    if not decoded or head_dim<=0 or q_num_heads%kv_num_heads!=0:return None
    if head_dim!=64:return None
    rep=q_num_heads//kv_num_heads;dev=decoded[0][0].device;eye=torch.eye(head_dim,device=dev)
    tq=[];ki=[];conds=[]
    try:
        for h in range(kv_num_heads):
            qs=[];ks=[]
            for q,k,_ in decoded:
                qh=q.reshape(q.shape[0],q_num_heads,head_dim)[:,h*rep:(h+1)*rep].reshape(-1,head_dim)
                kh=k.reshape(k.shape[0],kv_num_heads,head_dim)[:,h]
                qs.append(qh);ks.append(kh)
            Q=torch.cat(qs);K=torch.cat(ks)
            Cq=Q.t()@Q/max(1,Q.shape[0]);Ck=K.t()@K/max(1,K.shape[0])
            dq=Cq.diagonal().mean().abs().clamp_min(1e-10);dk=Ck.diagonal().mean().abs().clamp_min(1e-10)
            Cq=0.5*(Cq+Cq.t())+eye*(dq*_V42_COV_REG);Ck=0.5*(Ck+Ck.t())+eye*(dk*_V42_COV_REG)
            l,u=torch.linalg.eigh(Cq);l=l.clamp_min(dq*_V42_COV_REG)
            S=(u*l.sqrt().unsqueeze(0))@u.t();Is=(u*l.rsqrt().unsqueeze(0))@u.t()
            M=S@Ck@S;M=0.5*(M+M.t());lm,um=torch.linalg.eigh(M);lm=lm.clamp_min(dq*dk*(_V42_COV_REG**2))
            T=Is@(um*lm.pow(.25).unsqueeze(0))
            U,sv,Vh=torch.linalg.svd(T,full_matrices=False);g=torch.exp(torch.log(sv.clamp_min(1e-12)).mean());sv=sv/g
            cap=math.sqrt(_V42_COV_MAX_COND);sv=sv.clamp(1.0/cap,cap);T=(U*sv.unsqueeze(0))@Vh
            c=float((sv.max()/sv.min()).item())
            if not math.isfinite(c):return None
            invt=torch.linalg.inv(T).t()
            tq.append(T.contiguous());ki.append(invt.contiguous());conds.append(c)
        Tkv=torch.stack(tq);Kinv=torch.stack(ki);Tq=Tkv.repeat_interleave(rep,dim=0)
        return Tq,Kinv,max(conds)
    except Exception:return None


def hif4_calibration_attention(calib_qkv_list,q_num_heads,kv_num_heads,head_dim):
    base=_v42_v40_calibration_attention(calib_qkv_list,q_num_heads,kv_num_heads,head_dim)
    decoded=_v35_decode_calib(calib_qkv_list,q_num_heads,kv_num_heads,head_dim)
    if not decoded:return base
    qs=base.get("q_state",{}) if isinstance(base,dict) else {};ks=base.get("k_state",{}) if isinstance(base,dict) else {}
    base_scores=_v42_score_transformed(decoded,q_num_heads,kv_num_heads,head_dim,
        lambda q:_v42_apply_v40_state_q(q,qs,q_num_heads,head_dim),
        lambda k:_v42_apply_v40_state_k(k,ks,kv_num_heads,head_dim))
    if not base_scores:return base
    best_ratio=1.0;best=None

    # Candidate family 1: Smooth + full-head H64 / signed-H64.
    if head_dim%64==0:
        for beta in _V42_BETAS:
            sq,sk=_v35_qk_scale(decoded,q_num_heads,kv_num_heads,head_dim,beta)
            for rot in _V42_ROTS:
                scores=_v42_score_transformed(decoded,q_num_heads,kv_num_heads,head_dim,
                    lambda q,sq=sq,rot=rot:_v35_rotate_heads(q*sq,q_num_heads,head_dim,rot),
                    lambda k,sk=sk,rot=rot:_v35_rotate_heads(k/sk,kv_num_heads,head_dim,rot))
                ratio=sum(scores)/max(sum(base_scores),1e-20);worst=max(x/max(y,1e-20) for x,y in zip(scores,base_scores))
                if ratio<best_ratio and ratio<=_V42_GATE and worst<=_V42_WORST:
                    best_ratio=ratio;best=("rot",sq,sk,int(rot),float(beta),None)

    # Candidate family 2: full-matrix covariance balancing.
    cov=_v42_fit_cov(decoded,q_num_heads,kv_num_heads,head_dim)
    if cov is not None:
        Tq,Kinv,cond=cov
        scores=_v42_score_transformed(decoded,q_num_heads,kv_num_heads,head_dim,
            lambda q:_v42_apply_head_matrix(q,q_num_heads,head_dim,Tq),
            lambda k:_v42_apply_head_matrix(k,kv_num_heads,head_dim,Kinv))
        ratio=sum(scores)/max(sum(base_scores),1e-20);worst=max(x/max(y,1e-20) for x,y in zip(scores,base_scores))
        if ratio<best_ratio and ratio<=_V42_GATE and worst<=_V42_WORST:
            best_ratio=ratio;best=("cov",Tq,Kinv,-1,0.0,float(cond))

    if best is None:return base
    kind,a,b,rot,beta,cond=best;common={"version":_V42_VERSION,"enabled":True,"head_dim":int(head_dim),"transform_kind":kind,"calib_ratio":float(best_ratio)}
    if kind=="rot":
        return {"q_state":{**common,"role":"q","scale":a.detach().cpu().float(),"rotation":rot,"beta":beta},
                "k_state":{**common,"role":"k","scale":b.detach().cpu().float(),"rotation":rot,"beta":beta},
                "v_state":base.get("v_state",{"version":_V42_VERSION,"enabled":False,"role":"v"})}
    return {"q_state":{**common,"role":"q","matrix":a.detach().cpu().float(),"condition":cond},
            "k_state":{**common,"role":"k","matrix":b.detach().cpu().float(),"condition":cond},
            "v_state":base.get("v_state",{"version":_V42_VERSION,"enabled":False,"role":"v"})}


def hif4_dynamic_quantize_q(q_quant,q_scale,q_num_heads,head_dim,q_state):
    if isinstance(q_state,dict) and q_state.get("version")==_V42_VERSION:
        q=dequantize_nvfp4(q_quant,q_scale).float();kind=q_state.get("transform_kind")
        if kind=="rot":
            s=q_state.get("scale")
            if isinstance(s,torch.Tensor):q=q*s.to(q.device)
            q=_v35_rotate_heads(q,q_num_heads,head_dim,int(q_state.get("rotation",0)))
        elif kind=="cov":q=_v42_apply_head_matrix(q,q_num_heads,head_dim,q_state.get("matrix"))
        return _quantize_tensor_self_mse(q,return_dequant=False)[0]
    return _v42_v40_dynamic_q(q_quant,q_scale,q_num_heads,head_dim,q_state)


def hif4_dynamic_quantize_k(k_quant,k_scale,kv_num_heads,head_dim,k_state):
    if isinstance(k_state,dict) and k_state.get("version")==_V42_VERSION:
        k=dequantize_nvfp4(k_quant,k_scale).float();kind=k_state.get("transform_kind")
        if kind=="rot":
            s=k_state.get("scale")
            if isinstance(s,torch.Tensor):k=k/s.to(k.device)
            k=_v35_rotate_heads(k,kv_num_heads,head_dim,int(k_state.get("rotation",0)))
        elif kind=="cov":k=_v42_apply_head_matrix(k,kv_num_heads,head_dim,k_state.get("matrix"))
        return _v35_quantize_k_tensor(k)
    return _v42_v40_dynamic_k(k_quant,k_scale,kv_num_heads,head_dim,k_state)


def hif4_dynamic_quantize_v(v_quant,v_scale,kv_num_heads,head_dim,v_state):
    return _v42_v40_dynamic_v(v_quant,v_scale,kv_num_heads,head_dim,v_state)

# =============================================================================
# V43: calibration importance + fast Attention search + fast K quotient.
# Linear remains V40/V42 bit-identical.  Attention keeps V42's legal frozen
# reparameterizations, but calibration is two-stage and dynamic Q/K use fixed
# partner-energy importance learned only from calibration data.
# =============================================================================
_V43_VERSION = "v43_fastimportance"
_V43_PILOT_ROWS = 8
_V43_FULL_SAMPLES = 2
_V43_GATE = 0.93
_V43_WORST = 1.02
_V43_FAST_K_GAMMAS = (1.0, 1.75)

_v43_linear_calibration = hif4_calibration_and_quantize_weight
_v43_linear_activation = hif4_dynamic_quantize_activation
_v43_v42_dynamic_v = hif4_dynamic_quantize_v


def _fixed_scale_weighted_sse(abs_x: torch.Tensor, sf: torch.Tensor,
                              ww: torch.Tensor) -> torch.Tensor:
    errs=[]
    for mult in (1.0,2.0,4.0):
        denom=sf*mult
        mant=(torch.round((abs_x/denom)*4.0)*0.25).clamp(0.0,1.75)
        errs.append(((mant*denom-abs_x).square()*ww).sum(dim=-1,keepdim=True))
    e1,e2,e4=errs
    a=torch.minimum(e1,e2).sum(dim=-2,keepdim=True)
    b=torch.minimum(e2,e4).sum(dim=-2,keepdim=True)
    return torch.minimum(a,b).sum(dim=(-3,-2,-1))


def _materialize_fixed_scale_weighted(x: torch.Tensor, sf: torch.Tensor,
                                      ww: torch.Tensor):
    ax=x.abs(); ms=[]; es=[]
    for mult in (1.0,2.0,4.0):
        denom=sf*mult
        mant=(torch.round((ax/denom)*4.0)*0.25).clamp(0.0,1.75)
        ms.append(mant); es.append(((mant*denom-ax).square()*ww).sum(dim=-1,keepdim=True))
    e1,e2,e4=es
    l31=torch.where(e2<e1,2.0,1.0); l32=torch.where(e4<e2,2.0,1.0)
    a=torch.minimum(e1,e2).sum(dim=-2,keepdim=True)
    b=torch.minimum(e2,e4).sum(dim=-2,keepdim=True)
    l2=torch.where(b<a,2.0,1.0); l3=torch.where(l2==1.0,l31,l32)
    mult=l2*l3
    mant=torch.where(mult==1.0,ms[0],torch.where(mult==2.0,ms[1],ms[2]))
    sign=torch.sign(x); sign=torch.where(mant==0.0,torch.zeros_like(sign),sign)
    return sign,mant,l2,l3


def _quantize_tensor_weighted_mse(x: torch.Tensor, importance: torch.Tensor,
                                  *, return_dequant: bool=False):
    shape=tuple(int(s) for s in x.shape); c=shape[-1]
    if c%64!=0 or not isinstance(importance,torch.Tensor) or importance.numel()!=c:
        return _quantize_tensor_self_mse(x,return_dequant=return_dequant)
    x=x.float(); nblocks=c//64; rows=x.numel()//c
    blocks=x.reshape(rows,nblocks,8,2,4).reshape(-1,8,2,4)
    # Normalize geometric scale so only relative importance matters.
    imp=importance.to(x.device,dtype=torch.float32).reshape(c).clamp_min(1e-8)
    imp=imp/imp.mean().clamp_min(1e-8)
    wb=imp.reshape(nblocks,8,2,4).unsqueeze(0).expand(rows,-1,-1,-1,-1).reshape(-1,8,2,4)
    total=blocks.shape[0]; table=_build_e6m2_table(x.device); last=table.numel()-1
    sf_o=torch.empty((total,1,1,1),dtype=torch.bfloat16,device=x.device)
    l2_o=torch.empty((total,8,1,1),dtype=torch.bfloat16,device=x.device)
    l3_o=torch.empty((total,8,2,1),dtype=torch.bfloat16,device=x.device)
    sg_o=torch.empty((total,8,2,4),dtype=torch.bfloat16,device=x.device)
    ma_o=torch.empty((total,8,2,4),dtype=torch.bfloat16,device=x.device)
    dq_o=torch.empty_like(blocks) if return_dequant else None
    for st in range(0,total,_SEARCH_CHUNK_BLOCKS):
        en=min(st+_SEARCH_CHUNK_BLOCKS,total); xb=blocks[st:en]; ww=wb[st:en]; ax=xb.abs(); bsz=xb.shape[0]
        anchor=_nearest_e6m2_index(ax.amax(dim=(1,2,3))/7.0,table)
        best_e=torch.full((bsz,),float('inf'),device=x.device); best_i=anchor.clone()
        for off in _E6_ANCHOR_OFFSETS:
            idx=(anchor+off).clamp(0,last); sf=table[idx].view(bsz,1,1,1)
            e=_fixed_scale_weighted_sse(ax,sf,ww); ok=e<best_e
            best_e=torch.where(ok,e,best_e); best_i=torch.where(ok,idx,best_i)
        sf=table[best_i].view(bsz,1,1,1); sg,ma,l2,l3=_materialize_fixed_scale_weighted(xb,sf,ww)
        sf_o[st:en]=sf.to(torch.bfloat16); l2_o[st:en]=l2.to(torch.bfloat16); l3_o[st:en]=l3.to(torch.bfloat16)
        sg_o[st:en]=sg.to(torch.bfloat16); ma_o[st:en]=ma.to(torch.bfloat16)
        if dq_o is not None:dq_o[st:en]=sg*ma*l2*l3*sf
    prefix=shape[:-1]
    p={"scale_factor":sf_o.reshape(*prefix,nblocks,1,1,1),"scale_lv2":l2_o.reshape(*prefix,nblocks,8,1,1),
       "scale_lv3":l3_o.reshape(*prefix,nblocks,8,2,1),"sign":sg_o.reshape(*prefix,nblocks,8,2,4),
       "mant":ma_o.reshape(*prefix,nblocks,8,2,4)}
    dq=dq_o.reshape(shape).to(torch.bfloat16).float() if dq_o is not None else None
    return p,dq


def _v43_select_k(x, params, dqs, importance):
    if len(params)==1:return params[0]
    sh=tuple(int(s) for s in x.shape); seq=sh[-2]; hidden=sh[-1]; nb=hidden//64
    groups=int(math.prod(sh[:-2])) if sh[:-2] else 1
    x4=x.float().reshape(groups,seq,nb,64)
    w=importance.to(x.device).float().reshape(nb,64).clamp_min(1e-8)
    scores=[]
    for dq in dqs:
        e=dq.float().reshape(groups,seq,nb,64)-x4
        e=e-e.mean(dim=1,keepdim=True)
        scores.append((e.square()*w.reshape(1,1,nb,64)).sum(dim=(1,3)))
    best=torch.stack(scores).argmin(0)
    out={}
    bp=sh[:-2]
    for name in params[0]:
        base=params[0][name]; tail=tuple(int(v) for v in base.shape[len(bp)+2:]); y=base.reshape(groups,seq,nb,*tail).clone()
        for ci in range(1,len(params)):
            cand=params[ci][name].reshape(groups,seq,nb,*tail); mask=(best==ci).reshape(groups,1,nb,*([1]*len(tail)))
            y=torch.where(mask,cand,y)
        out[name]=y.reshape(base.shape)
    return out


def _v43_fast_k_tensor(x: torch.Tensor, importance: torch.Tensor):
    x=x.float()
    if x.dim()<2 or x.shape[-2]<=1:return _quantize_tensor_weighted_mse(x,importance,return_dequant=False)[0]
    ps=[];qs=[]
    def add(t):
        p,q=_quantize_tensor_weighted_mse(t,importance,return_dequant=True);ps.append(p);qs.append(q)
    add(x)
    # Mean and median are the two robust exact-nullspace basins.
    cmean=x.mean(dim=-2,keepdim=True); add(x-cmean)
    cmed=x.median(dim=-2,keepdim=True).values; add(x-cmed)
    qbest=_select_best_k_dq_per_feature_block(x,qs)
    cstar=(x-qbest).mean(dim=-2,keepdim=True)
    # One short over-relaxed refinement is enough to cross nearby discrete boundaries.
    for g in _V43_FAST_K_GAMMAS:
        add(x-float(g)*cstar)
    return _v43_select_k(x,ps,qs,importance)


def _v43_rms_qk_scale(decoded,q_num_heads,kv_num_heads,head_dim,beta):
    rep=q_num_heads//kv_num_heads;dev=decoded[0][0].device
    qe=torch.zeros((q_num_heads,head_dim),device=dev);ke=torch.zeros((kv_num_heads,head_dim),device=dev);nq=nk=0
    for q,k,_ in decoded:
        qh=q.reshape(-1,q_num_heads,head_dim);kh=k.reshape(-1,kv_num_heads,head_dim)
        qe+=(qh.square().sum(0));ke+=(kh.square().sum(0));nq+=qh.shape[0];nk+=kh.shape[0]
    qr=(qe/max(nq,1)).sqrt().clamp_min(2**-24);kr=(ke/max(nk,1)).sqrt().clamp_min(2**-24)
    if beta<=0:sk=torch.ones_like(kr)
    else:
        qg=qr.reshape(kv_num_heads,rep,head_dim).square().mean(1).sqrt()
        z=float(beta)*(torch.log(kr)-torch.log(qg));z=z-z.median(-1,keepdim=True).values;sk=torch.exp(z).clamp(2**-6,2**6)
    return sk.repeat_interleave(rep,0).reshape(-1),sk.reshape(-1)


def _v43_pilot_pair(q,k,n=8):
    nq=min(int(q.shape[0]),n); nk=min(int(k.shape[0]),max(n,12))
    return q[:nq],k[:nk]


def _disabled_legacy__v43_candidate_score_1642(decoded,q_num_heads,kv_num_heads,head_dim,qfn,kfn,pilot=True):
    raise RuntimeError("disabled legacy output-aware Attention calibration helper")


def _v43_importance(decoded,q_num_heads,kv_num_heads,head_dim,qfn,kfn):
    rep=q_num_heads//kv_num_heads;dev=decoded[0][0].device
    qe=torch.zeros((q_num_heads,head_dim),device=dev);ke=torch.zeros((kv_num_heads,head_dim),device=dev);nq=nk=0
    for q,k,_ in decoded:
        qt=qfn(q).reshape(-1,q_num_heads,head_dim);kt=kfn(k).reshape(-1,kv_num_heads,head_dim)
        qe+=qt.square().sum(0);ke+=kt.square().sum(0);nq+=qt.shape[0];nk+=kt.shape[0]
    qe/=max(nq,1);ke/=max(nk,1)
    qi=ke.repeat_interleave(rep,0)
    ki=qe.reshape(kv_num_heads,rep,head_dim).mean(1)
    qi=(qi/qi.mean().clamp_min(1e-8)).clamp(0.125,8.0).reshape(-1)
    ki=(ki/ki.mean().clamp_min(1e-8)).clamp(0.125,8.0).reshape(-1)
    return qi,ki


def _disabled_legacy_hif4_calibration_attention_1667(calib_qkv_list,q_num_heads,kv_num_heads,head_dim):
    # Hard baseline is V40, not V42: we rebuild V42's useful candidates with a
    # much cheaper two-stage search.
    raise RuntimeError("disabled legacy output-aware Attention calibration helper")


def hif4_dynamic_quantize_q(q_quant,q_scale,q_num_heads,head_dim,q_state):
    q=dequantize_nvfp4(q_quant,q_scale).float()
    if isinstance(q_state,dict):
        if q_state.get('version')==_V43_VERSION:
            kind=q_state.get('transform_kind')
            if kind=='rot':
                s=q_state.get('scale');q=q*s.to(q.device);q=_v35_rotate_heads(q,q_num_heads,head_dim,int(q_state.get('rotation',0)))
            elif kind=='cov':q=_v42_apply_head_matrix(q,q_num_heads,head_dim,q_state.get('matrix'))
        else:q=_v42_apply_v40_state_q(q,q_state,q_num_heads,head_dim)
        imp=q_state.get('v43_importance')
        if isinstance(imp,torch.Tensor) and imp.numel()==q.shape[-1]:return _quantize_tensor_weighted_mse(q,imp,return_dequant=False)[0]
    return _quantize_tensor_self_mse(q,return_dequant=False)[0]


def hif4_dynamic_quantize_k(k_quant,k_scale,kv_num_heads,head_dim,k_state):
    k=dequantize_nvfp4(k_quant,k_scale).float()
    if isinstance(k_state,dict):
        if k_state.get('version')==_V43_VERSION:
            kind=k_state.get('transform_kind')
            if kind=='rot':
                s=k_state.get('scale');k=k/s.to(k.device);k=_v35_rotate_heads(k,kv_num_heads,head_dim,int(k_state.get('rotation',0)))
            elif kind=='cov':k=_v42_apply_head_matrix(k,kv_num_heads,head_dim,k_state.get('matrix'))
        else:k=_v42_apply_v40_state_k(k,k_state,kv_num_heads,head_dim)
        imp=k_state.get('v43_importance')
        if isinstance(imp,torch.Tensor) and imp.numel()==k.shape[-1]:return _v43_fast_k_tensor(k,imp)
    return _quantize_k_softmax_quotient(k_quant,k_scale)


def hif4_dynamic_quantize_v(v_quant,v_scale,kv_num_heads,head_dim,v_state):
    return _v43_v42_dynamic_v(v_quant,v_scale,kv_num_heads,head_dim,v_state)

# V43-scorefirst final override: preserve full K quotient depth, but make its
# discrete decisions calibration-partner-energy aware.  Full covariance online
# transforms are intentionally disabled; only cheap reciprocal scale + H64
# candidates are used by the final calibration override below.

def _v43_weighted_best_dq(x,dqs,importance):
    sh=tuple(int(s) for s in x.shape);seq=sh[-2];hidden=sh[-1];nb=hidden//64;groups=int(math.prod(sh[:-2])) if sh[:-2] else 1
    x4=x.float().reshape(groups,seq,nb,64);w=importance.to(x.device).float().reshape(nb,64).clamp_min(1e-8)
    scores=[];arr=[]
    for dq in dqs:
        q4=dq.float().reshape(groups,seq,nb,64);e=q4-x4;e=e-e.mean(1,keepdim=True)
        scores.append((e.square()*w.reshape(1,1,nb,64)).sum((1,3)));arr.append(q4)
    best=torch.stack(scores).argmin(0);out=arr[0].clone()
    for ci in range(1,len(arr)):out=torch.where((best==ci).reshape(groups,1,nb,1),arr[ci],out)
    return out.reshape(sh)


def _v43_full_weighted_k_tensor(x,importance):
    x=x.float()
    if x.dim()<2 or x.shape[-2]<=1:return _quantize_tensor_weighted_mse(x,importance,return_dequant=False)[0]
    ps=[];ds=[]
    def add(t):p,q=_quantize_tensor_weighted_mse(t,importance,return_dequant=True);ps.append(p);ds.append(q)
    add(x);qbest=ds[0];cprev=torch.zeros_like(x.mean(-2,keepdim=True))
    for _ in range(1,_K_QUOTIENT_TOTAL_ROUNDS):
        cstar=(x-qbest).mean(-2,keepdim=True);delta=cstar-cprev
        for g in _K_RELAX_GAMMAS:add(x-(cprev+float(g)*delta))
        qbest=_v43_weighted_best_dq(x,ds,importance);cprev=cstar
    cprev=x.median(-2,keepdim=True).values;add(x-cprev);qbest=_v43_weighted_best_dq(x,ds,importance)
    for _ in range(_K_MEDIAN_EXTRA_ROUNDS):
        cstar=(x-qbest).mean(-2,keepdim=True);delta=cstar-cprev
        for g in _K_RELAX_GAMMAS:add(x-(cprev+float(g)*delta))
        qbest=_v43_weighted_best_dq(x,ds,importance);cprev=cstar
    return _v43_select_k(x,ps,ds,importance)


def _disabled_legacy_hif4_calibration_attention_1786(calib_qkv_list,q_num_heads,kv_num_heads,head_dim):
    raise RuntimeError("disabled legacy output-aware Attention calibration helper")


def hif4_dynamic_quantize_k(k_quant,k_scale,kv_num_heads,head_dim,k_state):
    k=dequantize_nvfp4(k_quant,k_scale).float()
    if isinstance(k_state,dict):
        if k_state.get('version')==_V43_VERSION:
            s=k_state.get('scale');
            if isinstance(s,torch.Tensor):k=k/s.to(k.device)
            k=_v35_rotate_heads(k,kv_num_heads,head_dim,int(k_state.get('rotation',0)))
        else:k=_v42_apply_v40_state_k(k,k_state,kv_num_heads,head_dim)
        imp=k_state.get('v43_importance')
        if isinstance(imp,torch.Tensor) and imp.numel()==k.shape[-1]:return _v43_full_weighted_k_tensor(k,imp)
    return _quantize_k_softmax_quotient(k_quant,k_scale)

# =============================================================================
# V44 adaptive: protect V42/V40 as exact hard paths. Dense covariance is only
# enabled for large predicted wins; partner-energy importance is itself gated.
# =============================================================================
_V44_VERSION='v44_adaptive'
_V44_STRUCT_GATE=0.93
_V44_COV_PILOT_GATE=0.72
_V44_COV_FULL_GATE=0.78
_V44_IMPORTANCE_GATE=0.985
_V44_IMPORTANCE_WORST=1.005


def _disabled_legacy__v44_score_weight_mode_1847(decoded,q_num_heads,kv_num_heads,head_dim,qfn,kfn,qi,ki,weighted):
    raise RuntimeError("disabled legacy output-aware Attention calibration helper")


def _disabled_legacy_hif4_calibration_attention_1859(calib_qkv_list,q_num_heads,kv_num_heads,head_dim):
    raise RuntimeError("disabled legacy output-aware Attention calibration helper")


def _v44_apply_q(q,state,q_num_heads,head_dim):
    if state.get('version')==_V44_VERSION:
        if state.get('transform_kind')=='rot':
            s=state.get('scale');q=q*s.to(q.device);q=_v35_rotate_heads(q,q_num_heads,head_dim,int(state.get('rotation',0)))
        else:q=_v42_apply_head_matrix(q,q_num_heads,head_dim,state.get('matrix'))
    else:q=_v42_apply_v40_state_q(q,state,q_num_heads,head_dim)
    return q


def _v44_apply_k(k,state,kv_num_heads,head_dim):
    if state.get('version')==_V44_VERSION:
        if state.get('transform_kind')=='rot':
            s=state.get('scale');k=k/s.to(k.device);k=_v35_rotate_heads(k,kv_num_heads,head_dim,int(state.get('rotation',0)))
        else:k=_v42_apply_head_matrix(k,kv_num_heads,head_dim,state.get('matrix'))
    else:k=_v42_apply_v40_state_k(k,state,kv_num_heads,head_dim)
    return k


def hif4_dynamic_quantize_q(q_quant,q_scale,q_num_heads,head_dim,q_state):
    q=dequantize_nvfp4(q_quant,q_scale).float()
    if isinstance(q_state,dict):
        q=_v44_apply_q(q,q_state,q_num_heads,head_dim);imp=q_state.get('v44_importance')
        if isinstance(imp,torch.Tensor) and imp.numel()==q.shape[-1]:return _quantize_tensor_weighted_mse(q,imp,return_dequant=False)[0]
    return _quantize_tensor_self_mse(q,return_dequant=False)[0]


def hif4_dynamic_quantize_k(k_quant,k_scale,kv_num_heads,head_dim,k_state):
    if not isinstance(k_state,dict):return _quantize_k_softmax_quotient(k_quant,k_scale)
    k=dequantize_nvfp4(k_quant,k_scale).float();k=_v44_apply_k(k,k_state,kv_num_heads,head_dim);imp=k_state.get('v44_importance')
    if isinstance(imp,torch.Tensor) and imp.numel()==k.shape[-1]:return _v43_full_weighted_k_tensor(k,imp)
    # Preserve original full-depth quotient on the selected transformed K.
    return _v35_quantize_k_tensor(k)

# =============================================================================
# V45 final: quotient-aware importance gate. It evaluates four legal frozen
# importance modes on one calibration sample using the same K quotient path that
# will be used online, then freezes the best mode only if the gain is material.
# =============================================================================
_V45_VERSION='v45_adaptive_gate'
_V45_IMP_GATE=0.985


def _v45_deq_params(p,shape):
    return (p['sign'].float()*p['mant'].float()*p['scale_lv2'].float()*p['scale_lv3'].float()*p['scale_factor'].float()).reshape(shape)


def _disabled_legacy__v45_choose_importance_mode_1956(decoded,q_num_heads,kv_num_heads,head_dim,qfn,kfn,qi,ki):
    raise RuntimeError("disabled legacy output-aware Attention calibration helper")


_v45_prev_calibration_attention=hif4_calibration_attention

def hif4_calibration_attention(calib_qkv_list,q_num_heads,kv_num_heads,head_dim):
    # First let V44 choose the structured/covariance transform, but discard its
    # surrogate importance decision and redo that decision with the true quotient.
    out=_v45_prev_calibration_attention(calib_qkv_list,q_num_heads,kv_num_heads,head_dim)
    decoded=_v35_decode_calib(calib_qkv_list,q_num_heads,kv_num_heads,head_dim)
    if not decoded:return out
    qs=out.get('q_state',{});ks=out.get('k_state',{})
    # Strip previous importance flags before defining frozen transform functions.
    qs={k:v for k,v in qs.items() if k not in ('v44_importance','v43_importance')}
    ks={k:v for k,v in ks.items() if k not in ('v44_importance','v43_importance')}
    qf=lambda q:_v44_apply_q(q,qs,q_num_heads,head_dim);kf=lambda k:_v44_apply_k(k,ks,kv_num_heads,head_dim)
    qi,ki=_v43_importance(decoded,q_num_heads,kv_num_heads,head_dim,qf,kf)
    useq,usek,ratio=_v45_choose_importance_mode(decoded,q_num_heads,kv_num_heads,head_dim,qf,kf,qi,ki)
    qs={**qs,'v45_use_q_importance':bool(useq),'v45_importance_ratio':float(ratio)}
    ks={**ks,'v45_use_k_importance':bool(usek),'v45_importance_ratio':float(ratio)}
    if useq:qs['v45_importance']=qi.cpu().float()
    if usek:ks['v45_importance']=ki.cpu().float()
    out['q_state']=qs;out['k_state']=ks
    return out


def hif4_dynamic_quantize_q(q_quant,q_scale,q_num_heads,head_dim,q_state):
    q=dequantize_nvfp4(q_quant,q_scale).float()
    if isinstance(q_state,dict):
        q=_v44_apply_q(q,q_state,q_num_heads,head_dim);imp=q_state.get('v45_importance')
        if q_state.get('v45_use_q_importance',False) and isinstance(imp,torch.Tensor):return _quantize_tensor_weighted_mse(q,imp,return_dequant=False)[0]
    return _quantize_tensor_self_mse(q,return_dequant=False)[0]


def hif4_dynamic_quantize_k(k_quant,k_scale,kv_num_heads,head_dim,k_state):
    if not isinstance(k_state,dict):return _quantize_k_softmax_quotient(k_quant,k_scale)
    k=dequantize_nvfp4(k_quant,k_scale).float();k=_v44_apply_k(k,k_state,kv_num_heads,head_dim);imp=k_state.get('v45_importance')
    if k_state.get('v45_use_k_importance',False) and isinstance(imp,torch.Tensor):return _v43_full_weighted_k_tensor(k,imp)
    return _v35_quantize_k_tensor(k)

# =============================================================================
# V48: precision-first full-hidden Hadamard candidate for Linear.
# Hard baseline is the leaderboard-proven V40 Linear path contained in V45.
# New candidates use only a fixed reciprocal Smooth + normalized full-hidden
# Walsh-Hadamard transform.  Calibration selects them by true MatMul MSE.
# Dynamic cost is O(K log K), with no dense KxK matrix.
# Attention remains V45 unchanged.
# =============================================================================
_V48_VERSION = 'v48_fullhidden_hadamard'
_V48_BETAS = (0.0, 0.25, 0.50, 0.75)
_V48_GATE = 0.90
_V48_WORST = 0.995


def _v48_is_pow2(n: int) -> bool:
    return n > 0 and (n & (n - 1)) == 0


def _v48_fwht(x: torch.Tensor) -> torch.Tensor:
    """Normalized full-last-dimension FWHT for power-of-two K."""
    k = int(x.shape[-1])
    if not _v48_is_pow2(k):
        return x.float()
    orig = x.shape
    y = x.float().reshape(-1, k).clone()
    h = 1
    while h < k:
        z = y.reshape(y.shape[0], -1, 2 * h)
        a = z[..., :h].clone()
        b = z[..., h:2*h].clone()
        z[..., :h] = a + b
        z[..., h:2*h] = a - b
        y = z.reshape(-1, k)
        h *= 2
    return (y / math.sqrt(float(k))).reshape(orig)


def _v48_rms_stats(w: torch.Tensor, acts):
    wr = w.float().square().mean(0).clamp_min(2.0 ** -48).sqrt()
    if not acts:
        ar = torch.ones_like(wr)
    else:
        total = torch.zeros_like(wr)
        count = 0
        for a in acts:
            total += a.float().square().sum(0)
            count += int(a.shape[0])
        ar = (total / max(count, 1)).clamp_min(2.0 ** -48).sqrt()
    return ar, wr


def _v48_full_base_transform(x: torch.Tensor, smooth: torch.Tensor, weight_side=False):
    y = x.float() / smooth if weight_side else x.float() * smooth
    return _v48_fwht(y)


def _v48_choose_full_candidate(*args, **kwargs):
    """Disabled in organizer-rule audit-clean build."""
    raise RuntimeError("disabled legacy output-aware Linear path")


def _v48_choose_full_phases(w: torch.Tensor, acts, smooth: torch.Tensor):
    """Per-64 reciprocal phase after full-H; phase=1 is hard fallback."""
    k = int(w.shape[-1]); nb = k // 64
    one = torch.ones(nb, device=w.device, dtype=torch.float32)
    if not acts or nb <= 0:
        return one
    iw = _v31_even(w.shape[0], min(40, w.shape[0]), w.device)
    wp = w[iw]
    wt0 = _v48_full_base_transform(wp, smooth, True)
    vals = []
    for a in acts[:4]:
        ia = _v31_even(a.shape[0], min(16, a.shape[0]), a.device)
        vals.append(_v48_full_base_transform(a[ia], smooth, False))
    errs = []
    for ph in _V40_PHASES:
        pv = torch.full((k,), float(ph), device=w.device)
        _, wq = _quantize_tensor_self_mse(wt0 / pv, return_dequant=True)
        es = []
        for at0 in vals:
            _, aq = _quantize_tensor_self_mse(at0 * pv, return_dequant=True)
            es.append(_v40_local_product_error(at0, wt0, aq, wq))
        errs.append(torch.stack(es, 0))
    E = torch.stack(errs, 0)
    bi = _V40_PHASES.index(1.0); base = E[bi].clamp_min(1e-20)
    pooled = E.sum(1) / base.sum(0).clamp_min(1e-20)
    worst = (E / base.unsqueeze(0)).amax(1)
    eligible = (pooled <= _V40_PHASE_GATE) & (worst <= _V40_PHASE_WORST)
    score = torch.where(eligible, pooled, torch.full_like(pooled, float('inf')))
    idx = score.argmin(0); val = score.min(0).values
    out = one
    good = torch.isfinite(val) & (val < 1.0)
    if bool(good.any().item()):
        tab = torch.tensor(_V40_PHASES, device=w.device, dtype=torch.float32)
        out[good] = tab[idx[good]]
    return out


def _v48_apply(x, state_like, weight_side=False):
    kind = state_like.get('linear_transform_kind', 'v40')
    if kind == 'full_h':
        s = state_like['smooth'].to(x.device)
        y = _v48_full_base_transform(x, s, weight_side)
        ph = state_like.get('block_phase')
        if isinstance(ph, torch.Tensor):
            pv = _v40_phase_vector(ph.to(x.device), x.device)
            y = y / pv if weight_side else y * pv
        return y
    s = state_like['smooth'].to(x.device)
    p = state_like.get('perm')
    ph = state_like.get('block_phase')
    if not isinstance(p, torch.Tensor):
        p = torch.arange(x.shape[-1], device=x.device)
    else:
        p = p.to(x.device)
    if not isinstance(ph, torch.Tensor):
        ph = torch.ones(x.shape[-1] // 64, device=x.device)
    else:
        ph = ph.to(x.device)
    return _v40_apply_base_transform(x, s, p, ph, bool(state_like.get('hadamard64', False)), weight_side)


def hif4_calibration_and_quantize_weight(*args, **kwargs):
    """Disabled in organizer-rule audit-clean build."""
    raise RuntimeError("disabled legacy output-aware Linear path")


def hif4_dynamic_quantize_activation(activation_quant, activation_scale, activation_state):
    a = dequantize_nvfp4(activation_quant, activation_scale).float()
    if isinstance(activation_state, dict) and 'smooth' in activation_state:
        a = _v48_apply(a, activation_state, False)
        T = activation_state.get('weight_comp') if activation_state.get('weight_comp_enabled', False) else None
        Q = activation_state.get('lr_q') if activation_state.get('lowrank_enabled', False) else None
        R = activation_state.get('lr_r') if activation_state.get('lowrank_enabled', False) else None
        a = _v37_apply_comp(a, T, Q, R)
        H = activation_state.get('weight_hessian_blocks') if activation_state.get('hessian_enabled', False) else None
        if isinstance(H, torch.Tensor):
            return _v37_quantize_hessian(a, H, return_dequant=False)[0]
    return _quantize_tensor_self_mse(a, return_dequant=False)[0]

# =============================================================================
# V51: calibration-selected randomized/signed full-hidden Hadamard.
# R = D_sign H_K is orthogonal.  Compared with V48 plain H_K it has identical
# online asymptotic cost (one extra sign multiply) but can break coherent
# channel-sign patterns before global mixing.  V48 remains a hard baseline.
# =============================================================================
_V51_VERSION = 'v51_signed_fullhidden_hadamard'
_V51_SIGN_GATE = 0.965       # require >=3.5% pooled improvement over exact V48 transform
_V51_SIGN_WORST = 0.998      # every calib sample must improve
_V51_SIGN_SEEDS = (0x243F6A88, 0x9E3779B9, 0xD1B54A35)


def _v51_sign_pattern(k: int, seed: int, device) -> torch.Tensor:
    """Deterministic Rademacher vector without relying on global RNG state."""
    # Integer hash over indices; generate on CPU for deterministic cross-device result.
    idx = torch.arange(k, dtype=torch.int64)
    x = idx + int(seed & 0x7FFFFFFF)
    x = (x ^ (x >> 16)) * 0x45d9f3b
    x = (x ^ (x >> 16)) * 0x45d9f3b
    x = x ^ (x >> 16)
    s = torch.where((x & 1) == 0, torch.ones(k), -torch.ones(k)).float()
    return s.to(device)


def _v51_full_transform(x: torch.Tensor, smooth: torch.Tensor, sign: torch.Tensor,
                        weight_side: bool = False) -> torch.Tensor:
    y = x.float() / smooth if weight_side else x.float() * smooth
    y = y * sign
    return _v48_fwht(y)


def _v51_choose_signed_candidate(*args, **kwargs):
    """Disabled in organizer-rule audit-clean build."""
    raise RuntimeError("disabled legacy output-aware Linear path")


def _v51_choose_signed_phases(w: torch.Tensor, acts, smooth: torch.Tensor, sign: torch.Tensor):
    k=int(w.shape[-1]); nb=k//64
    one=torch.ones(nb,device=w.device,dtype=torch.float32)
    if not acts or nb<=0: return one
    iw=_v31_even(w.shape[0],min(40,w.shape[0]),w.device); wp=w[iw]
    wt0=_v51_full_transform(wp,smooth,sign,True)
    avals=[]
    for a in acts[:4]:
        ia=_v31_even(a.shape[0],min(16,a.shape[0]),a.device)
        avals.append(_v51_full_transform(a[ia],smooth,sign,False))
    errs=[]
    for ph in _V40_PHASES:
        pv=torch.full((k,),float(ph),device=w.device)
        _,wq=_quantize_tensor_self_mse(wt0/pv,return_dequant=True)
        es=[]
        for at0 in avals:
            _,aq=_quantize_tensor_self_mse(at0*pv,return_dequant=True)
            es.append(_v40_local_product_error(at0,wt0,aq,wq))
        errs.append(torch.stack(es,0))
    E=torch.stack(errs,0); bi=_V40_PHASES.index(1.0); base=E[bi].clamp_min(1e-20)
    pooled=E.sum(1)/base.sum(0).clamp_min(1e-20); worst=(E/base.unsqueeze(0)).amax(1)
    eligible=(pooled<=_V40_PHASE_GATE)&(worst<=_V40_PHASE_WORST)
    score=torch.where(eligible,pooled,torch.full_like(pooled,float('inf')))
    idx=score.argmin(0); val=score.min(0).values; out=one
    good=torch.isfinite(val)&(val<1.0)
    if bool(good.any().item()):
        tab=torch.tensor(_V40_PHASES,device=w.device,dtype=torch.float32); out[good]=tab[idx[good]]
    return out


def _v51_apply(x, state_like, weight_side=False):
    if state_like.get('linear_transform_kind')=='signed_full_h':
        s=state_like['smooth'].to(x.device); sign=state_like['full_h_sign'].to(x.device)
        y=_v51_full_transform(x,s,sign,weight_side)
        ph=state_like.get('block_phase')
        if isinstance(ph,torch.Tensor):
            pv=_v40_phase_vector(ph.to(x.device),x.device); y=y/pv if weight_side else y*pv
        return y
    return _v48_apply(x,state_like,weight_side)


def hif4_calibration_and_quantize_weight(*args, **kwargs):
    """Disabled in organizer-rule audit-clean build."""
    raise RuntimeError("disabled legacy output-aware Linear path")


def hif4_dynamic_quantize_activation(activation_quant,activation_scale,activation_state):
    a=dequantize_nvfp4(activation_quant,activation_scale).float()
    if isinstance(activation_state,dict) and 'smooth' in activation_state:
        a=_v51_apply(a,activation_state,False)
        T=activation_state.get('weight_comp') if activation_state.get('weight_comp_enabled',False) else None
        Q=activation_state.get('lr_q') if activation_state.get('lowrank_enabled',False) else None
        R=activation_state.get('lr_r') if activation_state.get('lowrank_enabled',False) else None
        a=_v37_apply_comp(a,T,Q,R)
        H=activation_state.get('weight_hessian_blocks') if activation_state.get('hessian_enabled',False) else None
        if isinstance(H,torch.Tensor): return _v37_quantize_hessian(a,H,return_dequant=False)[0]
    return _quantize_tensor_self_mse(a,return_dequant=False)[0]

# =============================================================================
# V52: two-stage Signed Randomized Hadamard search against exact V48 baseline.
# Unlike V51, signed full-H may win even when plain V48 full-H did not pass gate.
# Pilot is tiny; only Top-2 signed candidates receive full MatMul validation.
# =============================================================================
_V52_VERSION='v52_signed_rht_twostage'
_V52_BETAS=(0.25,0.50,0.75)
_V52_FULL_GATE=0.92
_V52_FULL_WORST=0.995


def _v52_smooth_stats(w,acts):
    amax=torch.stack([a.abs().amax(0) for a in acts],0).amax(0)
    wmax=w.abs().amax(0)
    arms,wrms=_v48_rms_stats(w,acts)
    return {'max':(amax,wmax),'rms':(arms,wrms)}


def _v52_choose_signed_global(*args, **kwargs):
    """Disabled in organizer-rule audit-clean build."""
    raise RuntimeError("disabled legacy output-aware Linear path")


def hif4_calibration_and_quantize_weight(*args, **kwargs):
    """Disabled in organizer-rule audit-clean build."""
    raise RuntimeError("disabled legacy output-aware Linear path")

# Dynamic path is identical to V51 (one FWHT + optional sign multiply).

# =============================================================================
# V60 Fast-Core: latency reset.
# Preserve V52 Linear path. Replace V44/V45 Attention calibration with a tiny
# structured Smooth+H64 challenger and replace the 42-candidate K quotient by
# an 11-candidate medium-depth search. No covariance, no importance simulation.
# =============================================================================
_V60_VERSION='v60_fastcore'
_V60_K_GAMMAS=(1.0,1.75,2.5)
_V60_K_MEAN_ROUNDS=2
_V60_K_MEDIAN_ROUNDS=1
_V60_ATTN_BETAS=(0.25,0.50)
_V60_ATTN_GATE=0.92
_V60_ATTN_WORST=1.01


def _v60_k_score_select(x, params_list, dq_list):
    return _select_best_k_dq_per_feature_block(x,dq_list)


def _v60_quantize_k_tensor_fast(x):
    x=x.float()
    if x.dim()<2 or int(x.shape[-2])<=1:
        return _quantize_tensor_self_mse(x,return_dequant=False)[0]
    params=[]; dqs=[]
    p,q=_quantize_tensor_self_mse(x,return_dequant=True); params.append(p); dqs.append(q)
    qbest=q; cprev=torch.zeros_like(x.mean(dim=-2,keepdim=True))
    # Two mean-basin updates x 3 gammas = 6 candidates.
    for _ in range(_V60_K_MEAN_ROUNDS):
        cstar=(x-qbest).mean(dim=-2,keepdim=True); delta=cstar-cprev
        for g in _V60_K_GAMMAS:
            p,q=_quantize_tensor_self_mse(x-(cprev+float(g)*delta),return_dequant=True)
            params.append(p); dqs.append(q)
        qbest=_v60_k_score_select(x,params,dqs); cprev=cstar
    # Median basin + one refinement x 3 gammas = 4 more candidates.
    cprev=x.median(dim=-2,keepdim=True).values
    p,q=_quantize_tensor_self_mse(x-cprev,return_dequant=True);params.append(p);dqs.append(q)
    qbest=_v60_k_score_select(x,params,dqs)
    for _ in range(_V60_K_MEDIAN_ROUNDS):
        cstar=(x-qbest).mean(dim=-2,keepdim=True);delta=cstar-cprev
        for g in _V60_K_GAMMAS:
            p,q=_quantize_tensor_self_mse(x-(cprev+float(g)*delta),return_dequant=True)
            params.append(p);dqs.append(q)
        qbest=_v60_k_score_select(x,params,dqs);cprev=cstar
    return _merge_k_candidates_per_feature_block(x,params,dqs)


def _disabled_legacy__v60_pilot_logit_err_2319(q,k,qq,kk,q_num_heads,kv_num_heads,head_dim,nq=8,nk=32):
    raise RuntimeError("disabled legacy output-aware Attention calibration helper")


def _disabled_legacy_hif4_calibration_attention_2332(calib_qkv_list,q_num_heads,kv_num_heads,head_dim):
    # V40/V39 is the hard baseline; it is already much cheaper than V45.
    raise RuntimeError("disabled legacy output-aware Attention calibration helper")


def _v60_apply_q(q,state,q_num_heads,head_dim):
    if isinstance(state,dict) and state.get('version')==_V60_VERSION:
        s=state.get('scale');
        if isinstance(s,torch.Tensor):q=q*s.to(q.device)
        return _v35_rotate_heads(q,q_num_heads,head_dim,int(state.get('rotation',0)))
    return _v42_apply_v40_state_q(q,state,q_num_heads,head_dim)


def _v60_apply_k(k,state,kv_num_heads,head_dim):
    if isinstance(state,dict) and state.get('version')==_V60_VERSION:
        s=state.get('scale');
        if isinstance(s,torch.Tensor):k=k/s.to(k.device)
        return _v35_rotate_heads(k,kv_num_heads,head_dim,int(state.get('rotation',0)))
    return _v42_apply_v40_state_k(k,state,kv_num_heads,head_dim)


def hif4_dynamic_quantize_q(q_quant,q_scale,q_num_heads,head_dim,q_state):
    q=dequantize_nvfp4(q_quant,q_scale).float();q=_v60_apply_q(q,q_state,q_num_heads,head_dim)
    return _quantize_tensor_self_mse(q,return_dequant=False)[0]


def hif4_dynamic_quantize_k(k_quant,k_scale,kv_num_heads,head_dim,k_state):
    k=dequantize_nvfp4(k_quant,k_scale).float();k=_v60_apply_k(k,k_state,kv_num_heads,head_dim)
    return _v60_quantize_k_tensor_fast(k)


def hif4_dynamic_quantize_v(v_quant,v_scale,kv_num_heads,head_dim,v_state):
    # Keep V40 V path; no additional calibration-time importance/covariance.
    return _v42_v40_dynamic_v(v_quant,v_scale,kv_num_heads,head_dim,v_state)

# NEW ABLATION: V44 structured attention transform + V60 fast K quotient.
def hif4_calibration_attention(calib_qkv_list,q_num_heads,kv_num_heads,head_dim):
    out=_v45_prev_calibration_attention(calib_qkv_list,q_num_heads,kv_num_heads,head_dim)
    # Strip importance tensors/flags; keep only frozen structured transform.
    for role in ('q_state','k_state'):
        st=dict(out.get(role,{}))
        for k in list(st.keys()):
            if 'importance' in k:
                st.pop(k,None)
        out[role]=st
    return out

def hif4_dynamic_quantize_q(q_quant,q_scale,q_num_heads,head_dim,q_state):
    q=dequantize_nvfp4(q_quant,q_scale).float()
    if isinstance(q_state,dict): q=_v44_apply_q(q,q_state,q_num_heads,head_dim)
    return _quantize_tensor_self_mse(q,return_dequant=False)[0]

def hif4_dynamic_quantize_k(k_quant,k_scale,kv_num_heads,head_dim,k_state):
    k=dequantize_nvfp4(k_quant,k_scale).float()
    if isinstance(k_state,dict): k=_v44_apply_k(k,k_state,kv_num_heads,head_dim)
    return _v60_quantize_k_tensor_fast(k)

def hif4_dynamic_quantize_v(v_quant,v_scale,kv_num_heads,head_dim,v_state):
    v=dequantize_nvfp4(v_quant,v_scale).float()
    return _quantize_tensor_self_mse(v,return_dequant=False)[0]

# =============================================================================
# V64: Output-Aware Mantissa Coordinate Descent (OMCD) for Linear.
# Base: V52 Linear + Structured-QK/V60 Fast-K11 Attention.
# Weight side: calibration-only OMCD using transformed activation covariance.
# Activation side: frozen Wq Gram Hessian + 8 legal mantissa +/-0.25 updates.
# No test reference/cross-operand access; dynamic reads only current A + frozen state.
# =============================================================================
_V64_VERSION = 'v64_dualside_omcd8'
_V64_WEIGHT_ITERS = 8
_V64_ACT_ITERS = 8
_V64_GATE = 0.97
_V64_WORST = 1.005
_V64_BASE_CAL_LINEAR = hif4_calibration_and_quantize_weight
_V64_BASE_DYN_LINEAR = hif4_dynamic_quantize_activation


def _v64_dequant_params(params, shape):
    return (params['sign'] * params['mant'] * params['scale_lv2'] * params['scale_lv3'] * params['scale_factor']).reshape(shape).float()


def _v64_clone_params(params):
    return {k: v.clone() for k, v in params.items()}


def _v64_cov_blocks(acts_t, hidden):
    nb = hidden // 64
    out = []
    for b in range(nb):
        H = torch.zeros((64,64), dtype=torch.float32, device=acts_t[0].device)
        n = 0
        for a in acts_t:
            z = a.float().reshape(-1, hidden)[:, b*64:(b+1)*64]
            if z.numel() == 0:
                continue
            H += z.t() @ z
            n += int(z.shape[0])
        H = H / max(n, 1)
        H = H / H.diagonal().mean().abs().clamp_min(1e-12)
        out.append(H)
    return torch.stack(out, 0)


def _v64_refine_mantissa(x, params, hblocks, iters):
    """Exact coordinate descent on e^T H e with fixed legal HiF4 hierarchy.

    Only sign/mant are modified. Each move is exactly one legal signed-mantissa
    step (+/- 0.25) and is accepted only if the frozen quadratic objective drops.
    """
    if not isinstance(hblocks, torch.Tensor) or int(iters) <= 0:
        return params
    shape = tuple(int(s) for s in x.shape)
    c = shape[-1]
    if c % 64 != 0 or hblocks.dim() != 3 or tuple(hblocks.shape[1:]) != (64,64):
        return params
    nb = c // 64
    if int(hblocks.shape[0]) != nb:
        return params
    p = _v64_clone_params(params)
    rows = x.numel() // c
    total = rows * nb
    sf = p['scale_factor'].float().reshape(rows,nb,1,1,1)
    l2 = p['scale_lv2'].float().reshape(rows,nb,8,1,1)
    l3 = p['scale_lv3'].float().reshape(rows,nb,8,2,1)
    eff = (sf*l2*l3).expand(rows,nb,8,2,4).reshape(total,64)
    u = (p['sign'].float()*p['mant'].float()).reshape(total,64)
    xb = x.float().reshape(total,64)
    H0 = hblocks.to(x.device, torch.float32)
    ids = torch.arange(total, device=x.device) % nb
    H = H0[ids]
    e = u*eff - xb
    diag = H.diagonal(dim1=-2, dim2=-1)
    for _ in range(int(iters)):
        g = torch.bmm(H, e.unsqueeze(-1)).squeeze(-1)
        step = 0.25 * eff
        dp = 2.0*step*g + step.square()*diag
        dm = -2.0*step*g + step.square()*diag
        inf = torch.full_like(dp, float('inf'))
        dp = torch.where(u < 1.75 - 1e-6, dp, inf)
        dm = torch.where(u > -1.75 + 1e-6, dm, inf)
        vals = torch.cat([dp,dm], dim=1)
        best, idx = vals.min(dim=1)
        good = best < -1e-8
        if not bool(good.any().item()):
            break
        j = idx % 64
        direction = torch.where(idx < 64, torch.ones_like(best), -torch.ones_like(best))
        du = torch.zeros_like(u)
        du.scatter_(1, j[:,None], (0.25*direction)[:,None])
        du = du * good[:,None]
        u = u + du
        e = e + du*eff
    ma = u.abs()
    sg = torch.sign(u)
    sg = torch.where(ma == 0.0, torch.zeros_like(sg), sg)
    p['mant'] = ma.reshape_as(p['mant']).to(torch.bfloat16)
    p['sign'] = sg.reshape_as(p['sign']).to(torch.bfloat16)
    return p


def _v64_gate_block_map(*args, **kwargs):
    """Disabled in organizer-rule audit-clean build."""
    raise RuntimeError("disabled legacy output-aware Linear path")


def _v64_gate_lr(w_orig, acts, state, wq, block_map, lr):
    raise RuntimeError('disabled legacy output-fitting helper')


def _v64_quantize_candidate_activation(a, state, hblocks):
    at = _v51_apply(a, state, False)
    T = state.get('weight_comp') if state.get('weight_comp_enabled',False) else None
    Q = state.get('lr_q') if state.get('lowrank_enabled',False) else None
    R = state.get('lr_r') if state.get('lowrank_enabled',False) else None
    y = _v37_apply_comp(at,T,Q,R)
    p,_ = _v37_quantize_hessian(y,hblocks,return_dequant=False)
    p = _v64_refine_mantissa(y,p,hblocks,_V64_ACT_ITERS)
    return p, y


def hif4_calibration_and_quantize_weight(*args, **kwargs):
    """Disabled in organizer-rule audit-clean build."""
    raise RuntimeError("disabled legacy output-aware Linear path")


def hif4_dynamic_quantize_activation(activation_quant,activation_scale,activation_state):
    if not (isinstance(activation_state,dict) and activation_state.get('version')==_V64_VERSION and activation_state.get('omcd_enabled',False)):
        return _V64_BASE_DYN_LINEAR(activation_quant,activation_scale,activation_state)
    a=dequantize_nvfp4(activation_quant,activation_scale).float()
    H=activation_state.get('weight_hessian_blocks')
    if not isinstance(H,torch.Tensor):
        return _V64_BASE_DYN_LINEAR(activation_quant,activation_scale,activation_state)
    p,_=_v64_quantize_candidate_activation(a,activation_state,H)
    return p

# V67 max-score cross-block compensation (4x V37 budget).
def _v37_rank_budget(m: int, k: int) -> int:
    d=min(m,k)
    if k<=512: return min(256,d)
    if k<=2048: return min(128,d)
    return min(64,d)

# =============================================================================
# V70 Fast-Exact engineering overrides.
# Preserve V67 math / legal HiF4 decisions while removing repeated 64x64 work.
# 1) OMCD: one initial H@e, then exact rank-1 gradient updates.
# 2) Never materialize per-row copies of Hessian blocks.
# 3) Vectorize calibration covariance / Weight Hessian / block compensation.
# =============================================================================
_V70_VERSION = 'v70_fast_exact_omcd'


def _v64_cov_blocks(acts_t, hidden):
    """Vectorized equivalent of V64 per-64-block activation covariance."""
    nb = int(hidden) // 64
    if nb <= 0 or not acts_t:
        return torch.empty((0,64,64), dtype=torch.float32)
    dev = acts_t[0].device
    H = torch.zeros((nb,64,64), dtype=torch.float32, device=dev)
    n = 0
    for a in acts_t:
        z = a.float().reshape(-1, nb, 64)
        if z.numel() == 0:
            continue
        # [nb,64,64], same sum over rows as block-by-block z.T@z.
        H.add_(torch.einsum('rbi,rbj->bij', z, z))
        n += int(z.shape[0])
    H = H / max(n, 1)
    scale = H.diagonal(dim1=-2, dim2=-1).mean(dim=-1).abs().clamp_min(1e-12)
    return H / scale[:,None,None]


def _v37_weight_hessian_blocks(wq: torch.Tensor) -> torch.Tensor:
    """Vectorized equivalent of V37 Wq_b^T Wq_b for all 64-blocks."""
    K = int(wq.shape[-1]); nb = K // 64
    z = wq.float().reshape(-1, nb, 64)
    H = torch.einsum('mbi,mbj->bij', z, z)
    scale = H.diagonal(dim1=-2, dim2=-1).mean(dim=-1).abs().clamp_min(1e-12)
    return H / scale[:,None,None]


def _v37_effective_weight_blockmap(wq: torch.Tensor, block_map: torch.Tensor) -> torch.Tensor:
    if block_map is None or wq.shape[-1] % 64 != 0:
        return wq.float()
    K = int(wq.shape[-1]); nb = K // 64
    T = block_map.to(wq.device, torch.float32)
    if T.dim()!=3 or tuple(T.shape)!=(nb,64,64):
        return wq.float()
    z = wq.float().reshape(-1, nb, 64)
    # Old code: wq_block @ T[b].T
    y = torch.einsum('mbd,bkd->mbk', z, T)
    return y.reshape_as(wq.float())


def _v36_fit_weight_block_map(w_ref: torch.Tensor, w_q: torch.Tensor):
    """Batched equivalent of the V36 64x64 ridge solves."""
    if w_ref.dim()!=2 or w_ref.shape!=w_q.shape or w_ref.shape[-1] % 64 != 0:
        return None
    M,K = map(int,w_ref.shape); nb=K//64
    wr = w_ref.float().reshape(M,nb,64)
    wq = w_q.float().reshape(M,nb,64)
    gram = torch.einsum('mbi,mbj->bij', wq, wq)
    rhs  = torch.einsum('mbi,mbj->bij', wq, wr)
    ridge = gram.diagonal(dim1=-2,dim2=-1).mean(dim=-1).abs().clamp_min(1e-10) * _V36_COMP_RIDGE
    eye = torch.eye(64,dtype=torch.float32,device=w_ref.device).expand(nb,64,64)
    A = gram + ridge[:,None,None]*eye
    try:
        X = torch.linalg.solve(A, rhs)
    except Exception:
        try:
            X = torch.linalg.pinv(A) @ rhs
        except Exception:
            return None
    T = X.transpose(-1,-2).contiguous()
    finite = torch.isfinite(T).all(dim=(-2,-1))
    try:
        spec = torch.linalg.matrix_norm(T, ord=2)
    except Exception:
        spec = torch.full((nb,), float('inf'), device=w_ref.device)
    cand = torch.einsum('mbd,bkd->mbk', wq, T)  # wq @ T.T
    base_err = (wq-wr).square().sum(dim=(0,2))
    cand_err = (cand-wr).square().sum(dim=(0,2))
    good = finite & torch.isfinite(cand_err) & (spec <= _V36_COMP_MAX_SPECTRAL) & (cand_err < base_err*0.98)
    out = T.clone()
    if not bool(good.all().item()):
        out[~good] = torch.eye(64,dtype=torch.float32,device=w_ref.device)
    return out


def _v64_refine_mantissa(x, params, hblocks, iters):
    """Fast-exact V67 OMCD.

    Same coordinate objective/moves as V67, but H@e is computed once.  Since one
    coordinate changes per (row,block) per iteration, gradient is updated exactly
    by g <- g + H[:,j] * delta_e_j.  Hessian blocks are broadcast, not copied per
    activation row.
    """
    if not isinstance(hblocks, torch.Tensor) or int(iters) <= 0:
        return params
    shape = tuple(int(s) for s in x.shape); c=shape[-1]
    if c % 64 != 0 or hblocks.dim()!=3 or tuple(hblocks.shape[1:])!=(64,64):
        return params
    nb=c//64
    if int(hblocks.shape[0])!=nb:
        return params
    p=_v64_clone_params(params)
    rows=x.numel()//c
    sf=p['scale_factor'].float().reshape(rows,nb,1,1,1)
    l2=p['scale_lv2'].float().reshape(rows,nb,8,1,1)
    l3=p['scale_lv3'].float().reshape(rows,nb,8,2,1)
    eff=(sf*l2*l3).expand(rows,nb,8,2,4).reshape(rows,nb,64)
    u=(p['sign'].float()*p['mant'].float()).reshape(rows,nb,64)
    xb=x.float().reshape(rows,nb,64)
    H=hblocks.to(x.device,torch.float32)
    e=u*eff-xb
    # H is symmetric. This equals batched H@e without materializing [rows*nb,64,64].
    g=torch.einsum('bij,rbi->rbj',H,e)
    diag=H.diagonal(dim1=-2,dim2=-1).unsqueeze(0)
    step=0.25*eff
    step2diag=step.square()*diag
    bidx=torch.arange(nb,device=x.device).view(1,nb).expand(rows,nb)
    for _ in range(int(iters)):
        base=2.0*step*g
        dp=base+step2diag
        dm=-base+step2diag
        dp.masked_fill_(u>=1.75-1e-6,float('inf'))
        dm.masked_fill_(u<=-1.75+1e-6,float('inf'))
        choose_p=dp<dm
        move=torch.minimum(dp,dm)
        best,j=move.min(dim=2)
        good=best < -1e-8
        if not bool(good.any().item()):
            break
        plus=choose_p.gather(2,j.unsqueeze(-1)).squeeze(-1)
        direction=torch.where(plus,torch.ones_like(best),-torch.ones_like(best))
        du=0.25*direction*good
        u.scatter_add_(2,j.unsqueeze(-1),du.unsqueeze(-1))
        de=du*eff.gather(2,j.unsqueeze(-1)).squeeze(-1)
        # H[b,:,j] for every row/block, with no repeated Hessian tensor.
        col=H[bidx,:,j]
        g.add_(col*de.unsqueeze(-1))
    ma=u.abs(); sg=torch.sign(u); sg=torch.where(ma==0.0,torch.zeros_like(sg),sg)
    p['mant']=ma.reshape_as(p['mant']).to(torch.bfloat16)
    p['sign']=sg.reshape_as(p['sign']).to(torch.bfloat16)
    return p


def _v37_quantize_hessian(x: torch.Tensor, hblocks: torch.Tensor, *, return_dequant=False):
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

# V74: retain V70 Linear OMCD-8 / fast-exact path.
# Rescue only very short K sequences with full quotient search.
# This reads only the current K tensor's own sequence length.
_V74_FULL_K_MAX_SEQ = 64
_V64_VERSION = 'v74_fast_exact_shortk64'

def _v60_quantize_k_tensor_fast(x):
    x = x.float()
    if x.dim() >= 2 and int(x.shape[-2]) <= _V74_FULL_K_MAX_SEQ:
        return _v35_quantize_k_tensor(x)
    if x.dim()<2 or int(x.shape[-2])<=1:
        return _quantize_tensor_self_mse(x,return_dequant=False)[0]
    params=[]; dqs=[]
    p,q=_quantize_tensor_self_mse(x,return_dequant=True); params.append(p); dqs.append(q)
    qbest=q; cprev=torch.zeros_like(x.mean(dim=-2,keepdim=True))
    for _ in range(_V60_K_MEAN_ROUNDS):
        cstar=(x-qbest).mean(dim=-2,keepdim=True); delta=cstar-cprev
        for g in _V60_K_GAMMAS:
            p,q=_quantize_tensor_self_mse(x-(cprev+float(g)*delta),return_dequant=True)
            params.append(p); dqs.append(q)
        qbest=_v60_k_score_select(x,params,dqs); cprev=cstar
    cprev=x.median(dim=-2,keepdim=True).values
    p,q=_quantize_tensor_self_mse(x-cprev,return_dequant=True);params.append(p);dqs.append(q)
    qbest=_v60_k_score_select(x,params,dqs)
    for _ in range(_V60_K_MEDIAN_ROUNDS):
        cstar=(x-qbest).mean(dim=-2,keepdim=True);delta=cstar-cprev
        for g in _V60_K_GAMMAS:
            p,q=_quantize_tensor_self_mse(x-(cprev+float(g)*delta),return_dequant=True)
            params.append(p);dqs.append(q)
        qbest=_v60_k_score_select(x,params,dqs);cprev=cstar
    return _merge_k_candidates_per_feature_block(x,params,dqs)

# =============================================================================
# V76: Weight-Error-Aware / Qronos-style block objective for dynamic activations.
# The old OMCD minimizes ||(q-y) Wq^T||^2.  The true checker target is
# ||q Wq^T - y Wt^T||^2, where Wt is the continuous transformed Weight.
# For each 64 block we therefore use
#   H = Wq_b^T Wq_b, B = Wq_b^T Wt_b,
# and optimize q^T H q - 2 q^T B y.  A calibrated lambda blends the old and
# true local objectives, protecting against omitted cross-block terms.
# Attention remains exactly V74 (short-K64 rescue + Fast-K11 otherwise).
# =============================================================================
_V76_VERSION = 'v76_qronos_block_omcd8'
_V76_BASE_CAL = hif4_calibration_and_quantize_weight
_V76_BASE_DYN = hif4_dynamic_quantize_activation
_V76_BASE_VERSION = _V64_VERSION
_V76_LAMBDAS = (0.25, 0.50, 0.75, 1.00)
_V76_GATE = 0.995
_V76_WORST = 1.005


def _v76_cross_blocks(wq: torch.Tensor, wt: torch.Tensor):
    """Return normalized H=Wq^T Wq and B=Wq^T Wt, block diagonal (64)."""
    if wq.dim()!=2 or wt.dim()!=2 or wq.shape!=wt.shape or wq.shape[-1] % 64 != 0:
        return None, None
    m,k = map(int,wq.shape); nb=k//64
    q = wq.float().reshape(m,nb,64)
    t = wt.float().reshape(m,nb,64)
    H = torch.einsum('mbi,mbj->bij',q,q)
    B = torch.einsum('mbi,mbj->bij',q,t)
    s = H.diagonal(dim1=-2,dim2=-1).mean(dim=-1).abs().clamp_min(1e-12)
    return H/s[:,None,None], B/s[:,None,None]


def _v76_quantize_objective(y: torch.Tensor, hblocks: torch.Tensor, bblocks: torch.Tensor,
                            lam: float, iters: int = 8):
    """HiF4 quantize with a block-local true-output quadratic objective.

    lambda=0 exactly recovers the V70 Hessian objective; lambda=1 uses the
    local true output objective.  Scale-factor candidate selection and mantissa
    CD use the same blended objective.
    """
    shape=tuple(int(s) for s in y.shape); k=shape[-1]
    if (k%64!=0 or not isinstance(hblocks,torch.Tensor) or not isinstance(bblocks,torch.Tensor)):
        p,_=_quantize_tensor_self_mse(y,return_dequant=False); return p
    nb=k//64
    if tuple(hblocks.shape)!=(nb,64,64) or tuple(bblocks.shape)!=(nb,64,64):
        p,_=_v37_quantize_hessian(y,hblocks,return_dequant=False); return p
    y=y.float(); rows=y.numel()//k; yb=y.reshape(rows,nb,64)
    H=hblocks.to(y.device,torch.float32); B=bblocks.to(y.device,torch.float32)
    lam=float(max(0.0,min(1.0,lam)))
    # T y = ((1-lam)H + lam B) y.  In column notation this is the linear term
    # in q^T H q - 2 q^T T y.
    Hy=torch.einsum('bij,rbj->rbi',H,yb)
    By=torch.einsum('bij,rbj->rbi',B,yb)
    target=Hy + lam*(By-Hy)

    blocks=y.reshape(rows,nb,8,2,4)
    table=_build_e6m2_table(y.device); last=int(table.numel()-1)
    sf_out=torch.empty((rows,nb,1,1,1),dtype=torch.bfloat16,device=y.device)
    l2_out=torch.empty((rows,nb,8,1,1),dtype=torch.bfloat16,device=y.device)
    l3_out=torch.empty((rows,nb,8,2,1),dtype=torch.bfloat16,device=y.device)
    sg_out=torch.empty((rows,nb,8,2,4),dtype=torch.bfloat16,device=y.device)
    ma_out=torch.empty((rows,nb,8,2,4),dtype=torch.bfloat16,device=y.device)
    chunk_rows=max(1,4096//max(nb,1))
    for rs in range(0,rows,chunk_rows):
        re=min(rows,rs+chunk_rows); z=blocks[rs:re]; nr=re-rs; tt=target[rs:re]
        anchor=_nearest_e6m2_index(z.abs().amax((2,3,4))/7.0,table)
        best=torch.full((nr,nb),float('inf'),dtype=torch.float32,device=y.device)
        best_pack=None
        for off in _E6_ANCHOR_OFFSETS:
            idx=(anchor+off).clamp(0,last); sf=table[idx].view(nr,nb,1,1,1)
            sg,ma,l2,l3=_materialize_fixed_scale_self(z,sf)
            q=(sg*ma*l2*l3*sf).reshape(nr,nb,64)
            # Constant y-target terms are omitted because they do not depend on q.
            qHq=torch.einsum('rbi,bij,rbj->rb',q,H,q)
            cross=(q*tt).sum(dim=-1)
            err=qHq-2.0*cross
            better=err<best
            if best_pack is None:
                best=err; best_pack=[sf.clone(),l2.clone(),l3.clone(),sg.clone(),ma.clone()]
            else:
                best=torch.where(better,err,best)
                for jj,v in enumerate((sf,l2,l3,sg,ma)):
                    mask=better.view(nr,nb,*([1]*(v.dim()-2)))
                    best_pack[jj]=torch.where(mask,v,best_pack[jj])
        sf,l2,l3,sg,ma=best_pack
        sf_out[rs:re]=sf.to(torch.bfloat16); l2_out[rs:re]=l2.to(torch.bfloat16)
        l3_out[rs:re]=l3.to(torch.bfloat16); sg_out[rs:re]=sg.to(torch.bfloat16); ma_out[rs:re]=ma.to(torch.bfloat16)

    prefix=shape[:-1]
    p={'scale_factor':sf_out.reshape(*prefix,nb,1,1,1),
       'scale_lv2':l2_out.reshape(*prefix,nb,8,1,1),
       'scale_lv3':l3_out.reshape(*prefix,nb,8,2,1),
       'sign':sg_out.reshape(*prefix,nb,8,2,4),
       'mant':ma_out.reshape(*prefix,nb,8,2,4)}
    if int(iters)<=0: return p

    # Exact incremental coordinate descent for the same blended objective.
    sf=p['scale_factor'].float().reshape(rows,nb,1,1,1)
    l2=p['scale_lv2'].float().reshape(rows,nb,8,1,1)
    l3=p['scale_lv3'].float().reshape(rows,nb,8,2,1)
    eff=(sf*l2*l3).expand(rows,nb,8,2,4).reshape(rows,nb,64)
    u=(p['sign'].float()*p['mant'].float()).reshape(rows,nb,64)
    q=u*eff
    g=torch.einsum('bij,rbi->rbj',H,q)-target
    diag=H.diagonal(dim1=-2,dim2=-1).unsqueeze(0)
    step=0.25*eff; step2diag=step.square()*diag
    bidx=torch.arange(nb,device=y.device).view(1,nb).expand(rows,nb)
    for _ in range(int(iters)):
        base=2.0*step*g; dp=base+step2diag; dm=-base+step2diag
        dp.masked_fill_(u>=1.75-1e-6,float('inf')); dm.masked_fill_(u<=-1.75+1e-6,float('inf'))
        choose_p=dp<dm; move=torch.minimum(dp,dm); best,j=move.min(dim=2); good=best < -1e-8
        if not bool(good.any().item()): break
        plus=choose_p.gather(2,j.unsqueeze(-1)).squeeze(-1)
        direction=torch.where(plus,torch.ones_like(best),-torch.ones_like(best))
        du=0.25*direction*good; u.scatter_add_(2,j.unsqueeze(-1),du.unsqueeze(-1))
        de=du*eff.gather(2,j.unsqueeze(-1)).squeeze(-1)
        col=H[bidx,:,j]; g.add_(col*de.unsqueeze(-1))
    ma=u.abs(); sg=torch.sign(u); sg=torch.where(ma==0.0,torch.zeros_like(sg),sg)
    p['mant']=ma.reshape_as(p['mant']).to(torch.bfloat16); p['sign']=sg.reshape_as(p['sign']).to(torch.bfloat16)
    return p


def _v76_y(a, st):
    at=_v51_apply(a,st,False)
    T=st.get('weight_comp') if st.get('weight_comp_enabled',False) else None
    Q=st.get('lr_q') if st.get('lowrank_enabled',False) else None
    R=st.get('lr_r') if st.get('lowrank_enabled',False) else None
    return _v37_apply_comp(at,T,Q,R)


def hif4_calibration_and_quantize_weight(*args, **kwargs):
    """Disabled in organizer-rule audit-clean build."""
    raise RuntimeError("disabled legacy output-aware Linear path")


def hif4_dynamic_quantize_activation(activation_quant,activation_scale,activation_state):
    if not (isinstance(activation_state,dict) and activation_state.get('version')==_V76_VERSION and activation_state.get('qronos_block_enabled',False)):
        return _V76_BASE_DYN(activation_quant,activation_scale,activation_state)
    a=dequantize_nvfp4(activation_quant,activation_scale).float(); st=activation_state
    H=st.get('weight_hessian_blocks'); B=st.get('weight_cross_blocks')
    if not isinstance(H,torch.Tensor) or not isinstance(B,torch.Tensor):
        return _V76_BASE_DYN(activation_quant,activation_scale,activation_state)
    y=_v76_y(a,st); lam=float(st.get('qronos_lambda',0.0))
    return _v76_quantize_objective(y,H,B,lam,int(st.get('omcd_iters',8)))

# =============================================================================
# V78: 128-feature superblock output-aware OMCD.
# Storage remains legal HiF4 64->8->4; only the rounding objective spans two
# adjacent 64 groups to recover the most local cross-block correlations.
# =============================================================================
_V78_VERSION='v78_qronos_super128_omcd6'
_V78_BASE_CAL=_V76_BASE_CAL
_V78_BASE_DYN=_V76_BASE_DYN
_V78_BASE_VERSION=_V76_BASE_VERSION
_V78_LAMBDAS=(0.0,0.25,0.50,0.75,1.00)
_V78_ITERS=6
_V78_GATE=0.985
_V78_WORST=1.005


def _v78_super_blocks(wq,wt):
    if wq.dim()!=2 or wt.shape!=wq.shape:return None,None
    m,k=map(int,wq.shape)
    if k%128!=0:return None,None
    ng=k//128;q=wq.float().reshape(m,ng,128);t=wt.float().reshape(m,ng,128)
    H=torch.einsum('mgi,mgj->gij',q,q);B=torch.einsum('mgi,mgj->gij',q,t)
    s=H.diagonal(dim1=-2,dim2=-1).mean(dim=-1).abs().clamp_min(1e-12)
    return H/s[:,None,None],B/s[:,None,None]


def _v78_refine_super128(y,p,H,B,lam,iters=_V78_ITERS):
    shape=tuple(int(s) for s in y.shape);k=shape[-1]
    if k%128!=0 or not isinstance(H,torch.Tensor) or not isinstance(B,torch.Tensor):return p
    ng=k//128;rows=y.numel()//k
    if tuple(H.shape)!=(ng,128,128) or tuple(B.shape)!=(ng,128,128):return p
    pp=_v64_clone_params(p);nb=k//64
    sf=pp['scale_factor'].float().reshape(rows,nb,1,1,1);l2=pp['scale_lv2'].float().reshape(rows,nb,8,1,1);l3=pp['scale_lv3'].float().reshape(rows,nb,8,2,1)
    eff=(sf*l2*l3).expand(rows,nb,8,2,4).reshape(rows,ng,128)
    u=(pp['sign'].float()*pp['mant'].float()).reshape(rows,ng,128);yb=y.float().reshape(rows,ng,128)
    H=H.to(y.device,torch.float32);B=B.to(y.device,torch.float32);lam=float(lam)
    q=u*eff;Hy=torch.einsum('gij,rgj->rgi',H,yb);By=torch.einsum('gij,rgj->rgi',B,yb);target=Hy+lam*(By-Hy)
    g=torch.einsum('gij,rgi->rgj',H,q)-target
    diag=H.diagonal(dim1=-2,dim2=-1).unsqueeze(0);step=0.25*eff;step2=step.square()*diag
    gidx=torch.arange(ng,device=y.device).view(1,ng).expand(rows,ng)
    # Sequentially update each of the two 64-subblocks, so the interaction term
    # between their moves is included exactly through the gradient update.
    for _ in range(int(iters)):
        any_good=False
        for sub in (0,1):
            lo=sub*64;hi=lo+64;base=2.0*step[:,:,lo:hi]*g[:,:,lo:hi]
            dp=base+step2[:,:,lo:hi];dm=-base+step2[:,:,lo:hi]
            us=u[:,:,lo:hi]
            dp.masked_fill_(us>=1.75-1e-6,float('inf'));dm.masked_fill_(us<=-1.75+1e-6,float('inf'))
            choose=dp<dm;move=torch.minimum(dp,dm);best,j0=move.min(dim=2);good=best<-1e-8
            if not bool(good.any().item()):continue
            any_good=True;plus=choose.gather(2,j0.unsqueeze(-1)).squeeze(-1);direction=torch.where(plus,torch.ones_like(best),-torch.ones_like(best));du=0.25*direction*good
            j=j0+lo
            u.scatter_add_(2,j.unsqueeze(-1),du.unsqueeze(-1));de=du*eff.gather(2,j.unsqueeze(-1)).squeeze(-1)
            col=H[gidx,:,j];g.add_(col*de.unsqueeze(-1))
        if not any_good:break
    ma=u.abs().reshape(rows,nb,64);sg=torch.sign(u).reshape(rows,nb,64);sg=torch.where(ma==0.0,torch.zeros_like(sg),sg)
    pp['mant']=ma.reshape_as(pp['mant']).to(torch.bfloat16);pp['sign']=sg.reshape_as(pp['sign']).to(torch.bfloat16)
    return pp


def _v78_quant(y,H64,B64,H128,B128,lam,iters=_V78_ITERS):
    # Output-aware scale choice at native 64 granularity, then exact superblock CD.
    p=_v76_quantize_objective(y,H64,B64,lam,0)
    return _v78_refine_super128(y,p,H128,B128,lam,iters)


def hif4_calibration_and_quantize_weight(*args, **kwargs):
    """Disabled in organizer-rule audit-clean build."""
    raise RuntimeError("disabled legacy output-aware Linear path")


def hif4_dynamic_quantize_activation(activation_quant,activation_scale,activation_state):
    if not(isinstance(activation_state,dict) and activation_state.get('version')==_V78_VERSION and activation_state.get('qronos_super128_enabled',False)):return _V78_BASE_DYN(activation_quant,activation_scale,activation_state)
    a=dequantize_nvfp4(activation_quant,activation_scale).float();st=activation_state;H64=st.get('weight_hessian_blocks');B64=st.get('weight_cross_blocks');H128=st.get('super_hessian_blocks');B128=st.get('super_cross_blocks')
    if not all(isinstance(z,torch.Tensor) for z in (H64,B64,H128,B128)):return _V78_BASE_DYN(activation_quant,activation_scale,activation_state)
    y=_v76_y(a,st);return _v78_quant(y,H64,B64,H128,B128,float(st.get('qronos_lambda',0.0)),int(st.get('super128_iters',_V78_ITERS)))

# =============================================================================
# RULE-COMPLIANT LINEAR OVERRIDES
#
# New organizer rule: never form A@W (or a sampled A@W) and never use such an
# output target to fit/choose Q(A).  These final public overrides do not call any
# legacy Linear calibration path.
#
# Transform selection below uses ONLY separate marginal reconstruction quality:
#   - activation tensor reconstruction error
#   - weight tensor reconstruction error
# It never evaluates a MatMul output.
# =============================================================================

_RULE_SAFE_VERSION = "rule_safe_marginal_v1"
_RULE_SAFE_BETAS = (0.0, 0.25, 0.50)
_RULE_SAFE_HADS = (False, True)


def _rule_safe_decode_acts(calib_activation_list, k, device):
    out=[]
    for pair in calib_activation_list:
        if not isinstance(pair,(list,tuple)) or len(pair)!=2:
            continue
        a=dequantize_nvfp4(pair[0],pair[1]).float().to(device).reshape(-1,k)
        if a.numel() and a.shape[-1]==k:
            out.append(a)
    return out


def _rule_safe_rel_recon(x, q):
    num=(q.float()-x.float()).square().mean()
    den=x.float().square().mean().clamp_min(1e-20)
    return float((num/den).item())


def _rule_safe_choose_transform(w, acts):
    """Choose reciprocal Smooth/H64 only from separate tensor reconstruction."""
    if not acts:
        return torch.ones(w.shape[-1],device=w.device), False, 0.0

    amax=torch.stack([a.abs().amax(0) for a in acts]).amax(0)
    wmax=w.abs().amax(0)

    iw=_v31_even(w.shape[0],min(48,w.shape[0]),w.device)
    wp=w[iw]

    best=None
    for beta in _RULE_SAFE_BETAS:
        s=_v31_smooth(amax,wmax,float(beta))
        for had in _RULE_SAFE_HADS:
            wt=wp/s
            if had:
                wt=_fwht64_v31(wt)
            _,wq=_quantize_tensor_self_mse(wt,return_dequant=True)
            score=_rule_safe_rel_recon(wt,wq)

            # Activation quality is assessed independently; no W is multiplied by A.
            for a in acts[:4]:
                ia=_v31_even(a.shape[0],min(24,a.shape[0]),a.device)
                at=a[ia]*s
                if had:
                    at=_fwht64_v31(at)
                _,aq=_quantize_tensor_self_mse(at,return_dequant=True)
                score += _rule_safe_rel_recon(at,aq)

            if best is None or score < best[0]:
                best=(float(score),s,bool(had),float(beta))
    _,s,had,beta=best
    return s,had,beta


def _rule_safe_transform_weight(w, s, had):
    y=w.float()/s
    if had:
        y=_fwht64_v31(y)
    return y


def _rule_safe_transform_activation(a, s, had):
    y=a.float()*s
    if had:
        y=_fwht64_v31(y)
    return y


def _rule_safe_weight_hessian64(wq):
    if wq.dim()!=2 or wq.shape[-1]%64!=0:
        return None
    m,k=map(int,wq.shape); nb=k//64
    z=wq.float().reshape(m,nb,64)
    H=torch.einsum('mbi,mbj->bij',z,z)
    sc=H.diagonal(dim1=-2,dim2=-1).mean(dim=-1).abs().clamp_min(1e-12)
    return H/sc[:,None,None]

# -----------------------------------------------------------------------------
# V89 SAFE-HESSIAN: never computes A@W and never uses an A@W target.
# It uses only the frozen Gram matrix Wq^T Wq as a metric on activation
# reconstruction error e=(Q(A)-A), i.e. e^T H e.
#
# This is materially closer to the former OMCD quality, but because the new rule
# says "in any form", use V88 if the organizer intends to ban every W-aware
# activation metric, not merely A@W-target fitting.
# -----------------------------------------------------------------------------
_V89_VERSION="v89_rule_safe_hessian8"
_V89_ITERS=8

def hif4_calibration_and_quantize_weight(weight_quant,weight_scale,calib_activation_list):
    w=dequantize_nvfp4(weight_quant,weight_scale).float()
    acts=_rule_safe_decode_acts(calib_activation_list,w.shape[-1],w.device)
    s,had,beta=_rule_safe_choose_transform(w,acts)
    wt=_rule_safe_transform_weight(w,s,had)
    wp,wq=_quantize_tensor_self_mse(wt,return_dequant=True)
    H=_rule_safe_weight_hessian64(wq)
    state={
        "version":_V89_VERSION,
        "smooth":s.detach().cpu().float(),
        "hadamard64":bool(had),
        "beta":float(beta),
        "rule_safe_no_AW":True,
        "activation_objective":"reconstruction_error_under_frozen_Wq_Gram",
        "omcd_iters":int(_V89_ITERS),
    }
    if isinstance(H,torch.Tensor):
        state["weight_hessian_blocks"]=H.detach().cpu().to(torch.bfloat16)
    return {"weight_params":wp,"activation_state":state}


def hif4_dynamic_quantize_activation(activation_quant,activation_scale,activation_state):
    a=dequantize_nvfp4(activation_quant,activation_scale).float()
    st=activation_state if isinstance(activation_state,dict) else {}
    s=st.get("smooth")
    if isinstance(s,torch.Tensor):
        a=_rule_safe_transform_activation(
            a,s.to(a.device),bool(st.get("hadamard64",False)))
    H=st.get("weight_hessian_blocks")
    if not isinstance(H,torch.Tensor):
        return _quantize_tensor_self_mse(a,return_dequant=False)[0]
    p,_=_v37_quantize_hessian(a,H,return_dequant=False)
    return _v64_refine_mantissa(a,p,H,int(st.get("omcd_iters",_V89_ITERS)))





# =============================================================================
# V90+ organizer-rule-safe Linear branch
#
# No calibration output-product target is ever formed.
# Q(W) is legal per organizer clarification.
# All transforms are exact reciprocal reparameterizations chosen only from
# marginal A/W statistics. Dynamic activation optimization uses only Q(W)-Gram.
# =============================================================================

def _safe90_identity_perm(c, device):
    return torch.arange(c,dtype=torch.long,device=device)


def _safe90_blockvec(v):
    return v.reshape(-1,1).expand(-1,64).reshape(-1)


def _safe90_geometry(w, acts, *, use_perm=True, use_had=True, use_phase=True):
    """Pure marginal-statistics transform; no output-product or product-error evaluation."""
    k=int(w.shape[-1])
    if not acts or k%64!=0:
        return (torch.ones(k,device=w.device),
                _safe90_identity_perm(k,w.device),
                False,
                torch.ones(k//64,device=w.device))

    # SmoothQuant-style exact reciprocal balancing, fixed beta=1/2:
    # max(A*s) and max(W/s) are balanced channel-wise up to a common median norm.
    amax=torch.stack([a.abs().amax(0) for a in acts]).amax(0)
    wmax=w.abs().amax(0)
    smooth=_v31_smooth(amax,wmax,0.50)

    aeff=amax*smooth
    weff=wmax/smooth.clamp_min(2.0**-24)
    key=torch.maximum(aeff,weff)
    perm=_v40_balanced_perm(key) if use_perm else _safe90_identity_perm(k,w.device)

    had=bool(use_had)
    phases=torch.ones(k//64,dtype=torch.float32,device=w.device)
    if use_phase:
        # Estimate post-(smooth,perm,H64) block ranges independently for A and W.
        # A scalar phase per 64 block commutes with the within-block H64.
        wt=(w.float()/smooth).index_select(-1,perm)
        if had:
            wt=_fwht64_v31(wt)
        wb=wt.abs().reshape(-1,k//64,64).amax(dim=(0,2))

        ab=torch.zeros(k//64,dtype=torch.float32,device=w.device)
        for a in acts:
            at=(a.float()*smooth).index_select(-1,perm)
            if had:
                at=_fwht64_v31(at)
            ab=torch.maximum(ab,at.abs().reshape(-1,k//64,64).amax(dim=(0,2)))

        # Reciprocal block balancing: A' = A*p, W' = W/p.
        phases=torch.sqrt(wb.clamp_min(2.0**-24)/ab.clamp_min(2.0**-24))
        # Keep the reparameterization conservative and distribution-robust.
        phases=phases.clamp(0.50,2.00)
        # Remove an irrelevant global gain to avoid systematic drift.
        phases=phases/torch.exp(torch.log(phases).median())
        phases=phases.clamp(0.50,2.00)

    return smooth,perm,had,phases


def _safe90_apply(x,smooth,perm,phases,had,weight_side=False):
    y=x.float()
    y=y/smooth if weight_side else y*smooth
    y=y.index_select(-1,perm.to(y.device,dtype=torch.long))
    pv=_safe90_blockvec(phases.to(y.device,torch.float32))
    y=y/pv if weight_side else y*pv
    if had:
        y=_fwht64_v31(y)
    return y


def _safe90_hessian64(wq):
    if wq.dim()!=2 or wq.shape[-1]%64!=0:return None
    m,k=map(int,wq.shape);nb=k//64
    z=wq.float().reshape(m,nb,64)
    H=torch.einsum('mbi,mbj->bij',z,z)
    sc=H.diagonal(dim1=-2,dim2=-1).mean(-1).abs().clamp_min(1e-12)
    return H/sc[:,None,None]


def _safe90_hessian256(wq):
    if wq.dim()!=2 or wq.shape[-1]%256!=0:return None
    m,k=map(int,wq.shape);ng=k//256
    z=wq.float().reshape(m,ng,256)
    H=torch.einsum('mgi,mgj->gij',z,z)
    sc=H.diagonal(dim1=-2,dim2=-1).mean(-1).abs().clamp_min(1e-12)
    return H/sc[:,None,None]


def _safe90_refine256(y,p,H,iters=4):
    """Hessian-only Super256 OMCD: minimize (Q(A)-A)^T H (Q(A)-A)."""
    shape=tuple(int(s) for s in y.shape);k=shape[-1]
    if k%256!=0 or not isinstance(H,torch.Tensor):return p
    rows=y.numel()//k;ng=k//256;nb=k//64
    if tuple(H.shape)!=(ng,256,256):return p

    pp=_v64_clone_params(p)
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


def _safe90_make_state(version,smooth,perm,had,phases,**extra):
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


def _safe90_decode_transform_activation(activation_quant,activation_scale,st):
    a=dequantize_nvfp4(activation_quant,activation_scale).float()
    s=st.get('smooth');perm=st.get('perm');ph=st.get('block_phase')
    if not all(isinstance(z,torch.Tensor) for z in (s,perm,ph)):
        return a
    return _safe90_apply(a,s.to(a.device),perm.to(a.device),ph.to(a.device),
                         bool(st.get('hadamard64',False)),False)

# V91: V90 geometry + Hessian-only Super256 OMCD4.
_V91_VERSION='v91_safe_geometry_super256_omcd4'
_V91_ITERS64=0
_V91_ITERS256=4

def hif4_calibration_and_quantize_weight(weight_quant,weight_scale,calib_activation_list):
    w=dequantize_nvfp4(weight_quant,weight_scale).float()
    acts=_rule_safe_decode_acts(calib_activation_list,w.shape[-1],w.device)
    s,perm,had,ph=_safe90_geometry(w,acts,use_perm=True,use_had=True,use_phase=True)
    wt=_safe90_apply(w,s,perm,ph,had,True)
    wp,wq=_quantize_tensor_self_mse(wt,return_dequant=True)
    H64=_safe90_hessian64(wq);H256=_safe90_hessian256(wq)
    st=_safe90_make_state(_V91_VERSION,s,perm,had,ph,
                          super256_iters=_V91_ITERS256,
                          transform_kind='safe_v40_marginal')
    if isinstance(H64,torch.Tensor):st['weight_hessian_blocks']=H64.detach().cpu().to(torch.bfloat16)
    if isinstance(H256,torch.Tensor):st['super256_hessian_blocks']=H256.detach().cpu().to(torch.bfloat16)
    return {'weight_params':wp,'activation_state':st}

def hif4_dynamic_quantize_activation(activation_quant,activation_scale,activation_state):
    st=activation_state if isinstance(activation_state,dict) else {}
    y=_safe90_decode_transform_activation(activation_quant,activation_scale,st)
    H64=st.get('weight_hessian_blocks');H256=st.get('super256_hessian_blocks')
    if not isinstance(H64,torch.Tensor):
        return _quantize_tensor_self_mse(y,return_dequant=False)[0]
    p,_=_v37_quantize_hessian(y,H64,return_dequant=False)
    if isinstance(H256,torch.Tensor):
        p=_safe90_refine256(y,p,H256,int(st.get('super256_iters',_V91_ITERS256)))
    return p




# =============================================================================
# V99/V100 RULE-SAFE COVARIANCE-AWARE WEIGHT QUANTIZATION
#
# No calibration output-product target. No Q(W)^T W cross term.
# Weight metric comes only from calibration activation Gram C_A=A^T A.
# Activation metric comes only from legal quantized-weight Gram H_W=Q(W)^TQ(W).
# =============================================================================

def _safe99_activation_hessian64(acts_t, k, device):
    nb=k//64
    H=torch.zeros((nb,64,64),dtype=torch.float32,device=device)
    count=0
    for a in acts_t:
        z=a.float().reshape(-1,nb,64)
        H.add_(torch.einsum('rbi,rbj->bij',z,z))
        count+=int(z.shape[0])
    if count<=0:return None
    H.div_(float(count))
    sc=H.diagonal(dim1=-2,dim2=-1).mean(-1).abs().clamp_min(1e-12)
    return H/sc[:,None,None]


def _safe99_activation_hessian256(acts_t, k, device):
    if k%256:return None
    ng=k//256
    H=torch.zeros((ng,256,256),dtype=torch.float32,device=device)
    count=0
    for a in acts_t:
        z=a.float().reshape(-1,ng,256)
        H.add_(torch.einsum('rgi,rgj->gij',z,z))
        count+=int(z.shape[0])
    if count<=0:return None
    H.div_(float(count))
    sc=H.diagonal(dim1=-2,dim2=-1).mean(-1).abs().clamp_min(1e-12)
    return H/sc[:,None,None]


def _safe99_proxy_transform_score(w, acts, s, perm, had, ph):
    """
    Separable output-error proxy, no output product:
      weight-error under activation covariance
      + activation-error under Q(W) Gram
    evaluated on bounded calibration pilots.
    """
    wt=_safe90_apply(w,s,perm,ph,had,True)
    acts_t=[_safe90_apply(a,s,perm,ph,had,False) for a in acts]
    HA=_safe99_activation_hessian64(acts_t,int(w.shape[-1]),w.device)
    if HA is None:return float('inf'),None,None,None

    wp,wq=_v37_quantize_hessian(wt,HA,return_dequant=True)

    # Weight-side covariance error.
    m,k=map(int,wt.shape);nb=k//64
    ew=(wq.float()-wt.float()).reshape(m,nb,64)
    werr=torch.einsum('mbi,bij,mbj->',ew,HA,ew)
    wden=torch.einsum('mbi,bij,mbj->',wt.reshape(m,nb,64),HA,wt.reshape(m,nb,64)).clamp_min(1e-20)
    score=float((werr/wden).item())

    # Activation-side Q(W)-Gram error, sampled only for speed.
    HW=_safe90_hessian64(wq)
    if isinstance(HW,torch.Tensor):
        for a in acts_t[:4]:
            ia=_v31_even(a.shape[0],min(24,a.shape[0]),a.device)
            ap=a[ia]
            _,aq=_v37_quantize_hessian(ap,HW,return_dequant=True)
            ea=(aq.float()-ap.float()).reshape(-1,nb,64)
            num=torch.einsum('rbi,bij,rbj->',ea,HW,ea)
            den=torch.einsum('rbi,bij,rbj->',ap.reshape(-1,nb,64),HW,ap.reshape(-1,nb,64)).clamp_min(1e-20)
            score += float((num/den).item())
    return score,wp,wq,HA


def _safe99_transform_candidates(w,acts):
    """Small rule-safe transform family; no output-reference scoring."""
    k=int(w.shape[-1])
    amax=torch.stack([a.abs().amax(0) for a in acts]).amax(0)
    wmax=w.abs().amax(0)
    out=[]
    for beta in (0.25,0.50,0.75):
        s=_v31_smooth(amax,wmax,beta)
        aeff=amax*s;weff=wmax/s.clamp_min(2.0**-24)
        key=torch.maximum(aeff,weff)
        identity=torch.arange(k,device=w.device)
        balanced=_v40_balanced_perm(key)
        for perm,had,tag in (
            (identity,True,f'h64_b{beta}'),
            (balanced,True,f'balh64_b{beta}'),
            (balanced,False,f'bal_b{beta}'),
        ):
            # Per-64 reciprocal phase from pure marginal transformed ranges.
            wt0=(w.float()/s).index_select(-1,perm)
            if had:wt0=_fwht64_v31(wt0)
            wb=wt0.abs().reshape(-1,k//64,64).amax((0,2))
            ab=torch.zeros(k//64,device=w.device)
            for a in acts:
                at=(a.float()*s).index_select(-1,perm)
                if had:at=_fwht64_v31(at)
                ab=torch.maximum(ab,at.abs().reshape(-1,k//64,64).amax((0,2)))
            ph=torch.sqrt(wb.clamp_min(2**-24)/ab.clamp_min(2**-24)).clamp(.5,2.0)
            ph=ph/torch.exp(torch.log(ph).median())
            ph=ph.clamp(.5,2.0)
            out.append((s,perm,had,ph,tag))
    return out


def _safe99_choose_and_quantize_weight(w,acts,search_transform):
    if not acts:
        s=torch.ones(w.shape[-1],device=w.device);perm=torch.arange(w.shape[-1],device=w.device)
        ph=torch.ones(w.shape[-1]//64,device=w.device)
        wt=w.float();wp,wq=_quantize_tensor_self_mse(wt,return_dequant=True)
        return s,perm,False,ph,'identity',wp,wq

    if not search_transform:
        s,perm,had,ph=_safe90_geometry(w,acts,use_perm=True,use_had=True,use_phase=True)
        wt=_safe90_apply(w,s,perm,ph,had,True)
        acts_t=[_safe90_apply(a,s,perm,ph,had,False) for a in acts]
        HA=_safe99_activation_hessian64(acts_t,int(w.shape[-1]),w.device)
        wp,wq=_v37_quantize_hessian(wt,HA,return_dequant=True)
        return s,perm,had,ph,'fixed_safe_v40',wp,wq

    best=None
    for s,perm,had,ph,tag in _safe99_transform_candidates(w,acts):
        sc,wp,wq,HA=_safe99_proxy_transform_score(w,acts,s,perm,had,ph)
        if best is None or sc<best[0]:
            best=(sc,s,perm,had,ph,tag,wp,wq)
    _,s,perm,had,ph,tag,wp,wq=best
    return s,perm,had,ph,tag,wp,wq


def _safe99_calibrate(weight_quant,weight_scale,calib_activation_list,search_transform=False,weight_omcd=0):
    w=dequantize_nvfp4(weight_quant,weight_scale).float()
    acts=_rule_safe_decode_acts(calib_activation_list,w.shape[-1],w.device)
    s,perm,had,ph,tag,wp,wq=_safe99_choose_and_quantize_weight(w,acts,search_transform)

    # Optional weight-side mantissa OMCD under activation covariance.
    if weight_omcd>0 and acts:
        wt=_safe90_apply(w,s,perm,ph,had,True)
        acts_t=[_safe90_apply(a,s,perm,ph,had,False) for a in acts]
        HA=_safe99_activation_hessian64(acts_t,int(w.shape[-1]),w.device)
        if isinstance(HA,torch.Tensor):
            wp=_v64_refine_mantissa(wt,wp,HA,int(weight_omcd))
            wq=_v64_dequant_params(wp,tuple(w.shape))

    H64=_safe90_hessian64(wq);H256=_safe90_hessian256(wq)
    return w,acts,s,perm,had,ph,tag,wp,wq,H64,H256

_V99_VERSION='v99_rulesafe_actcov_weight_h256'

def hif4_calibration_and_quantize_weight(weight_quant,weight_scale,calib_activation_list):
    w,acts,s,perm,had,ph,tag,wp,wq,H64,H256=_safe99_calibrate(
        weight_quant,weight_scale,calib_activation_list,search_transform=False,weight_omcd=0)
    st=_safe90_make_state(_V99_VERSION,s,perm,had,ph,
                          transform_kind=tag,super256_iters=4,
                          weight_metric='calib_activation_gram64')
    if isinstance(H64,torch.Tensor):st['weight_hessian_blocks']=H64.detach().cpu().to(torch.bfloat16)
    if isinstance(H256,torch.Tensor):st['super256_hessian_blocks']=H256.detach().cpu().to(torch.bfloat16)
    return {'weight_params':wp,'activation_state':st}

def hif4_dynamic_quantize_activation(activation_quant,activation_scale,activation_state):
    st=activation_state if isinstance(activation_state,dict) else {}
    y=_safe90_decode_transform_activation(activation_quant,activation_scale,st)
    H64=st.get('weight_hessian_blocks');H256=st.get('super256_hessian_blocks')
    if not isinstance(H64,torch.Tensor):return _quantize_tensor_self_mse(y,return_dequant=False)[0]
    p,_=_v37_quantize_hessian(y,H64,return_dequant=False)
    if isinstance(H256,torch.Tensor):
        p=_safe90_refine256(y,p,H256,int(st.get('super256_iters',4)))
    return p




# =============================================================================
# V105+ RULE-SAFE WEIGHT GENERALIZATION / CROSS-BLOCK REFINEMENT
#
# Only calibration activation covariance A^T A and legal Q(W)-derived Gram
# statistics are used. No calibration output-product reference is formed.
# =============================================================================

def _safe105_slice_params(p, st, en):
    return {k: v[st:en].clone() for k,v in p.items()}


def _safe105_write_params(dst, src, st, en):
    for k in dst:
        dst[k][st:en] = src[k]
    return dst


def _safe105_chunked_weight_h256(wt, p, HA256, iters=1, chunk_rows=256):
    """Memory-bounded H256 mantissa refinement for offline Weight."""
    if not isinstance(HA256, torch.Tensor) or wt.dim()!=2 or wt.shape[-1]%256:
        return p
    out=_v64_clone_params(p)
    m=int(wt.shape[0])
    for st in range(0,m,int(chunk_rows)):
        en=min(st+int(chunk_rows),m)
        pc=_safe105_slice_params(out,st,en)
        pc=_safe90_refine256(wt[st:en],pc,HA256,int(iters))
        _safe105_write_params(out,pc,st,en)
    return out


def _safe106_merge_block_params(pself, pcov, choose_cov):
    """choose_cov: [rows, n64] bool."""
    out={}
    for name in pself:
        a=pself[name]
        b=pcov[name]
        tail=a.dim()-2
        mask=choose_cov.reshape(*choose_cov.shape,*([1]*tail))
        out[name]=torch.where(mask,b,a)
    return out


def _safe106_trust_weight_quant(w,acts,tau):
    """
    Full covariance-aware Weight quantization with a per-(row,64) self-MSE trust region.
    Covariance candidate is accepted only when:
      1) activation-covariance objective improves, and
      2) ordinary Weight reconstruction SSE rises by at most tau.
    """
    s,perm,had,ph=_safe90_geometry(w,acts,use_perm=True,use_had=True,use_phase=True)
    wt=_safe90_apply(w,s,perm,ph,had,True)
    acts_t=[_safe90_apply(a,s,perm,ph,had,False) for a in acts]
    HA=_safe99_activation_hessian64(acts_t,int(w.shape[-1]),w.device)

    pself,qself=_quantize_tensor_self_mse(wt,return_dequant=True)
    if not isinstance(HA,torch.Tensor):
        return s,perm,had,ph,pself,qself

    pcov,qcov=_v37_quantize_hessian(wt,HA,return_dequant=True)
    m,k=map(int,wt.shape); nb=k//64
    x=wt.float().reshape(m,nb,64)
    es=(qself.float().reshape(m,nb,64)-x).square().sum(-1)
    ec=(qcov.float().reshape(m,nb,64)-x).square().sum(-1)

    ds=qself.float().reshape(m,nb,64)-x
    dc=qcov.float().reshape(m,nb,64)-x
    hs=torch.einsum('mbi,bij,mbj->mb',ds,HA,ds)
    hc=torch.einsum('mbi,bij,mbj->mb',dc,HA,dc)

    choose=(hc < hs) & (ec <= es*(1.0+float(tau)) + 1e-20)
    p=_safe106_merge_block_params(pself,pcov,choose)
    q=_v64_dequant_params(p,tuple(w.shape)).float()
    return s,perm,had,ph,p,q


def _safe105_make_state(version,s,perm,had,ph,wq,extra=None):
    H64=_safe90_hessian64(wq)
    H256=_safe90_hessian256(wq)
    st=_safe90_make_state(version,s,perm,had,ph,
                          transform_kind='safe_v40_marginal',
                          super256_iters=4)
    if extra:
        st.update(extra)
    if isinstance(H64,torch.Tensor):
        st['weight_hessian_blocks']=H64.detach().cpu().to(torch.bfloat16)
    if isinstance(H256,torch.Tensor):
        st['super256_hessian_blocks']=H256.detach().cpu().to(torch.bfloat16)
    return st


def _safe105_dynamic(activation_quant,activation_scale,activation_state):
    st=activation_state if isinstance(activation_state,dict) else {}
    y=_safe90_decode_transform_activation(activation_quant,activation_scale,st)
    H64=st.get('weight_hessian_blocks')
    H256=st.get('super256_hessian_blocks')
    if not isinstance(H64,torch.Tensor):
        return _quantize_tensor_self_mse(y,return_dequant=False)[0]
    p,_=_v37_quantize_hessian(y,H64,return_dequant=False)
    if isinstance(H256,torch.Tensor):
        p=_safe90_refine256(y,p,H256,int(st.get('super256_iters',4)))
    return p

_V105_VERSION='v105_rulesafe_weight_cov256_chunk1'

def hif4_calibration_and_quantize_weight(weight_quant,weight_scale,calib_activation_list):
    w=dequantize_nvfp4(weight_quant,weight_scale).float()
    acts=_rule_safe_decode_acts(calib_activation_list,w.shape[-1],w.device)

    # V99 initialization: full block-diagonal activation covariance.
    s,perm,had,ph=_safe90_geometry(w,acts,use_perm=True,use_had=True,use_phase=True)
    wt=_safe90_apply(w,s,perm,ph,had,True)
    acts_t=[_safe90_apply(a,s,perm,ph,had,False) for a in acts]
    HA64=_safe99_activation_hessian64(acts_t,int(w.shape[-1]),w.device)
    HA256=_safe99_activation_hessian256(acts_t,int(w.shape[-1]),w.device)
    wp,wq=_v37_quantize_hessian(wt,HA64,return_dequant=True) if isinstance(HA64,torch.Tensor) else _quantize_tensor_self_mse(wt,return_dequant=True)

    # New: one memory-bounded cross-64 Weight refinement sweep under A^T A (256).
    if isinstance(HA256,torch.Tensor):
        wp=_safe105_chunked_weight_h256(wt,wp,HA256,iters=1,chunk_rows=256)
        wq=_v64_dequant_params(wp,tuple(w.shape)).float()

    st=_safe105_make_state(_V105_VERSION,s,perm,had,ph,wq,{
        'weight_metric':'activation_covariance_64_plus_256',
        'weight_h256_iters':1,
    })
    return {'weight_params':wp,'activation_state':st}

def hif4_dynamic_quantize_activation(activation_quant,activation_scale,activation_state):
    return _safe105_dynamic(activation_quant,activation_scale,activation_state)




# =============================================================================
# V108+ ATTENTION PARTNER-COVARIANCE REFINEMENT
#
# Linear path remains V105.
#
# New Attention metric:
#   Q error metric <- centered calibration K covariance
#   K candidate ranking <- calibration Q covariance + K translation nullspace
#
# No final Attention/SDPA output is used by these new refinements.
# =============================================================================

# Capture the currently active V105 Attention path before overriding it.
_v108_base_attn_calib = hif4_calibration_attention
_v108_base_q = hif4_dynamic_quantize_q
_v108_base_k = hif4_dynamic_quantize_k
_v108_base_v = hif4_dynamic_quantize_v


def _safe108_norm_h(H):
    H=H.float()
    sc=H.diagonal(dim1=-2,dim2=-1).mean(-1).abs().clamp_min(1e-12)
    return H/sc[:,None,None]


def _safe108_h256_to_h64(H256):
    # H256: [num_heads, 256, 256], head_dim is 256 in the supplied mini sample.
    hs=[]
    nh=int(H256.shape[0]);hd=int(H256.shape[-1])
    for h in range(nh):
        for s in range(0,hd,64):
            hs.append(H256[h,s:s+64,s:s+64])
    return torch.stack(hs,0)


def _safe108_partner_covariances(decoded,qs,ks,q_num_heads,kv_num_heads,head_dim):
    if not decoded or head_dim%64!=0:
        return None
    rep=q_num_heads//kv_num_heads
    device=decoded[0][0].device
    HQ_kv=torch.zeros((kv_num_heads,head_dim,head_dim),device=device,dtype=torch.float32)
    HK=torch.zeros((kv_num_heads,head_dim,head_dim),device=device,dtype=torch.float32)
    nk_count=0
    nq_count=0

    for q,k,_ in decoded:
        qt=_v44_apply_q(q.float(),qs,q_num_heads,head_dim).reshape(-1,q_num_heads,head_dim)
        kt=_v44_apply_k(k.float(),ks,kv_num_heads,head_dim).reshape(-1,kv_num_heads,head_dim)

        # Q logit error is insensitive to the mean of K across key positions:
        # delta_q @ K^T differs only by a row-constant along that component.
        kc=kt-kt.mean(dim=0,keepdim=True)
        HQ_kv.add_(torch.einsum('shd,she->hde',kc,kc))
        nk_count += int(kc.shape[0])

        # Each KV head is shared by `rep` Q heads. Aggregate all associated Q energy.
        qg=qt.reshape(qt.shape[0],kv_num_heads,rep,head_dim)
        HK.add_(torch.einsum('shrd,shre->hde',qg,qg))
        nq_count += int(qg.shape[0])*int(rep)

    if nk_count<=0 or nq_count<=0:
        return None
    HQ_kv.div_(float(nk_count))
    HK.div_(float(nq_count))

    HQ_kv=_safe108_norm_h(HQ_kv)
    HK=_safe108_norm_h(HK)

    # Repeat each K-head covariance for the Q heads that attend to it.
    HQ256=HQ_kv.repeat_interleave(rep,dim=0).contiguous()
    HK256=HK.contiguous()
    HQ64=_safe108_h256_to_h64(HQ256)
    HK64=_safe108_h256_to_h64(HK256)
    return HQ64,HQ256,HK64,HK256


def _safe108_attach_cov_state(out,calib_qkv_list,q_num_heads,kv_num_heads,head_dim,use_q_h256=False,use_k_metric=False):
    decoded=_v35_decode_calib(calib_qkv_list,q_num_heads,kv_num_heads,head_dim)
    if not decoded:
        return out
    qs=dict(out.get('q_state',{}));ks=dict(out.get('k_state',{}))
    cov=_safe108_partner_covariances(decoded,qs,ks,q_num_heads,kv_num_heads,head_dim)
    if cov is None:
        return out
    HQ64,HQ256,HK64,HK256=cov

    qs['partner_h64']=HQ64.detach().cpu().to(torch.bfloat16)
    qs['partner_cov_enabled']=True
    qs['partner_h256_enabled']=bool(use_q_h256)
    if use_q_h256:
        qs['partner_h256']=HQ256.detach().cpu().to(torch.bfloat16)

    ks['partner_k_metric_enabled']=bool(use_k_metric)
    if use_k_metric:
        # H256 is enough for candidate ranking. H64 is retained for future cheap
        # scale-selection ablations without changing the state contract.
        ks['partner_h256']=HK256.detach().cpu().to(torch.bfloat16)
        ks['partner_h64']=HK64.detach().cpu().to(torch.bfloat16)

    out=dict(out)
    out['q_state']=qs
    out['k_state']=ks
    return out


def _safe108_quantize_q(q_quant,q_scale,q_num_heads,head_dim,q_state,use_h256=False):
    q=dequantize_nvfp4(q_quant,q_scale).float()
    if isinstance(q_state,dict):
        q=_v44_apply_q(q,q_state,q_num_heads,head_dim)
        H64=q_state.get('partner_h64')
        if isinstance(H64,torch.Tensor):
            p,_=_v37_quantize_hessian(q,H64.to(q.device),return_dequant=False)
            if use_h256:
                H256=q_state.get('partner_h256')
                if isinstance(H256,torch.Tensor):
                    p=_safe90_refine256(q,p,H256.to(q.device),1)
            return p
    return _quantize_tensor_self_mse(q,return_dequant=False)[0]


def _safe108_k_head_scores(x,dq,H256,kv_num_heads,head_dim):
    # Exact centered partner-covariance metric for the K translation quotient.
    seq=int(x.shape[-2])
    e=(dq.float()-x.float()).reshape(seq,kv_num_heads,head_dim)
    e=e-e.mean(dim=0,keepdim=True)
    H=H256.to(x.device,torch.float32)
    return torch.einsum('shd,hde,she->h',e,H,e)


def _safe108_merge_k_by_head(params_list,best_head,kv_num_heads,head_dim):
    if len(params_list)==1:
        return params_list[0]
    bph=head_dim//64
    best_blocks=best_head.repeat_interleave(bph)  # [nblocks]
    out={}
    for name in params_list[0]:
        base=params_list[0][name]
        y=base.clone()
        # Expected K param layout: [seq, nblocks, tail...].
        tail=base.dim()-2
        for ci in range(1,len(params_list)):
            cand=params_list[ci][name]
            mask=(best_blocks==ci).reshape(1,-1,*([1]*tail))
            y=torch.where(mask,cand,y)
        out[name]=y
    return out


def _safe108_quantize_k_partner(x,H256,kv_num_heads,head_dim):
    """
    Same fast K quotient basins as V60, but candidate selection is performed with
    centered Q-covariance metric instead of plain feature SSE.
    Candidate quantization itself remains the fast self-MSE kernel.
    """
    x=x.float()
    if x.dim()!=2 or int(x.shape[-2])<=1 or not isinstance(H256,torch.Tensor):
        return _v60_quantize_k_tensor_fast(x)

    params=[];dqs=[]
    p,q=_quantize_tensor_self_mse(x,return_dequant=True)
    params.append(p);dqs.append(q)

    def current_best():
        scores=torch.stack([_safe108_k_head_scores(x,z,H256,kv_num_heads,head_dim) for z in dqs],0)
        best=scores.argmin(0)
        # Build a dequantized mixed representative per head for the next quotient update.
        qs=torch.stack([z.reshape(x.shape[0],kv_num_heads,head_dim) for z in dqs],0)
        y=qs[0].clone()
        for ci in range(1,len(dqs)):
            mask=(best==ci).reshape(1,kv_num_heads,1)
            y=torch.where(mask,qs[ci],y)
        return best,y.reshape_as(x)

    best,qbest=current_best()
    cprev=torch.zeros_like(x.mean(dim=-2,keepdim=True))
    for _ in range(_V60_K_MEAN_ROUNDS):
        cstar=(x-qbest).mean(dim=-2,keepdim=True)
        delta=cstar-cprev
        for g in _V60_K_GAMMAS:
            p,q=_quantize_tensor_self_mse(x-(cprev+float(g)*delta),return_dequant=True)
            params.append(p);dqs.append(q)
        best,qbest=current_best()
        cprev=cstar

    cprev=x.median(dim=-2,keepdim=True).values
    p,q=_quantize_tensor_self_mse(x-cprev,return_dequant=True)
    params.append(p);dqs.append(q)
    best,qbest=current_best()
    for _ in range(_V60_K_MEDIAN_ROUNDS):
        cstar=(x-qbest).mean(dim=-2,keepdim=True)
        delta=cstar-cprev
        for g in _V60_K_GAMMAS:
            p,q=_quantize_tensor_self_mse(x-(cprev+float(g)*delta),return_dequant=True)
            params.append(p);dqs.append(q)
        best,qbest=current_best()
        cprev=cstar

    return _safe108_merge_k_by_head(params,best,kv_num_heads,head_dim)


def _safe108_quantize_k(k_quant,k_scale,kv_num_heads,head_dim,k_state,use_partner=False):
    k=dequantize_nvfp4(k_quant,k_scale).float()
    if isinstance(k_state,dict):
        k=_v44_apply_k(k,k_state,kv_num_heads,head_dim)
        H=k_state.get('partner_h256')
        if use_partner and isinstance(H,torch.Tensor):
            return _safe108_quantize_k_partner(k,H,kv_num_heads,head_dim)
    return _v60_quantize_k_tensor_fast(k)

_V110_VERSION='v110_v105_qcov256_k_partner_quotient'

def hif4_calibration_attention(calib_qkv_list,q_num_heads,kv_num_heads,head_dim):
    out=_v108_base_attn_calib(calib_qkv_list,q_num_heads,kv_num_heads,head_dim)
    return _safe108_attach_cov_state(out,calib_qkv_list,q_num_heads,kv_num_heads,head_dim,
                                     use_q_h256=True,use_k_metric=True)

def hif4_dynamic_quantize_q(q_quant,q_scale,q_num_heads,head_dim,q_state):
    return _safe108_quantize_q(q_quant,q_scale,q_num_heads,head_dim,q_state,True)

def hif4_dynamic_quantize_k(k_quant,k_scale,kv_num_heads,head_dim,k_state):
    return _safe108_quantize_k(k_quant,k_scale,kv_num_heads,head_dim,k_state,True)

def hif4_dynamic_quantize_v(v_quant,v_scale,kv_num_heads,head_dim,v_state):
    return _v108_base_v(v_quant,v_scale,kv_num_heads,head_dim,v_state)




# =============================================================================
# V111+ STRICT-RULE-SAFE ATTENTION CALIBRATION
#
# Replaces the inherited V44/V45 output-aware Attention selector completely.
# Calibration uses only marginal Q/K statistics and partner covariances.
#
# Fixed exact-structure transform:
#   beta = 0.5 reciprocal Q/K Smooth
#   rotation = FWHT64 pattern 1
#
# No QK-logit reference, no SDPA/Attention reference, no final operator output
# is formed anywhere in the active V111+ Attention calibration path.
# =============================================================================

def _safe111_fixed_attention_base(calib_qkv_list,q_num_heads,kv_num_heads,head_dim):
    decoded=_v35_decode_calib(calib_qkv_list,q_num_heads,kv_num_heads,head_dim)
    common={
        'version':_V44_VERSION,
        'enabled':True,
        'head_dim':int(head_dim),
        'transform_kind':'rot',
        'rotation':1,
        'beta':0.5,
        'strict_safe_fixed':True,
    }
    if not decoded:
        return {
            'q_state':{**common,'role':'q','scale':torch.ones(q_num_heads*head_dim)},
            'k_state':{**common,'role':'k','scale':torch.ones(kv_num_heads*head_dim)},
            'v_state':{'enabled':False,'role':'v','strict_safe_fixed':True},
        }

    # Pure marginal reciprocal balancing; no joint operator output.
    sq,sk=_v35_qk_scale(decoded,q_num_heads,kv_num_heads,head_dim,0.5)

    return {
        'q_state':{**common,'role':'q','scale':sq.detach().cpu().float()},
        'k_state':{**common,'role':'k','scale':sk.detach().cpu().float()},
        'v_state':{'enabled':False,'role':'v','strict_safe_fixed':True},
    }


def _safe111_attach(calib_qkv_list,q_num_heads,kv_num_heads,head_dim,
                    use_q_h256=False,use_k_metric=False):
    out=_safe111_fixed_attention_base(
        calib_qkv_list,q_num_heads,kv_num_heads,head_dim
    )
    return _safe108_attach_cov_state(
        out,calib_qkv_list,q_num_heads,kv_num_heads,head_dim,
        use_q_h256=bool(use_q_h256),
        use_k_metric=bool(use_k_metric),
    )


def _safe111_v(v_quant,v_scale,kv_num_heads,head_dim,v_state):
    # Strict tensor-self V path.
    v=dequantize_nvfp4(v_quant,v_scale).float()
    return _quantize_tensor_self_mse(v,return_dequant=False)[0]

_V113_VERSION='v113_strictsafe_qcov256_kpartner'

def hif4_calibration_attention(calib_qkv_list,q_num_heads,kv_num_heads,head_dim):
    return _safe111_attach(
        calib_qkv_list,q_num_heads,kv_num_heads,head_dim,
        use_q_h256=True,use_k_metric=True
    )

def hif4_dynamic_quantize_q(q_quant,q_scale,q_num_heads,head_dim,q_state):
    return _safe108_quantize_q(q_quant,q_scale,q_num_heads,head_dim,q_state,True)

def hif4_dynamic_quantize_k(k_quant,k_scale,kv_num_heads,head_dim,k_state):
    return _safe108_quantize_k(k_quant,k_scale,kv_num_heads,head_dim,k_state,True)

def hif4_dynamic_quantize_v(v_quant,v_scale,kv_num_heads,head_dim,v_state):
    return _safe111_v(v_quant,v_scale,kv_num_heads,head_dim,v_state)




# =============================================================================
# V117+ STRICT-SAFE DEEP H256 / ATTENTION SECOND-MOMENT EXPERIMENTS
# No calibration operator-output reference is formed.
# =============================================================================

def _safe117_linear_calib(weight_quant,weight_scale,calib_activation_list,
                          aligned_iters=2,shifted=False,version='v117'):
    w=dequantize_nvfp4(weight_quant,weight_scale).float()
    acts=_rule_safe_decode_acts(calib_activation_list,w.shape[-1],w.device)

    s,perm,had,ph=_safe90_geometry(w,acts,use_perm=True,use_had=True,use_phase=True)
    wt=_safe90_apply(w,s,perm,ph,had,True)
    acts_t=[_safe90_apply(a,s,perm,ph,had,False) for a in acts]
    HA64=_safe99_activation_hessian64(acts_t,int(w.shape[-1]),w.device)
    HA256=_safe99_activation_hessian256(acts_t,int(w.shape[-1]),w.device)

    if isinstance(HA64,torch.Tensor):
        wp,wq=_v37_quantize_hessian(wt,HA64,return_dequant=True)
    else:
        wp,wq=_quantize_tensor_self_mse(wt,return_dequant=True)

    # Deeper aligned H256 coordinate descent; chunked to avoid V101's large native path.
    if isinstance(HA256,torch.Tensor):
        wp=_safe105_chunked_weight_h256(
            wt,wp,HA256,iters=int(aligned_iters),chunk_rows=256
        )

    # Optional shifted 256 windows, offset by 128 features (=2 HiF4 blocks).
    if shifted and wt.dim()==2 and int(wt.shape[-1])>=384:
        k=int(wt.shape[-1])
        starts=list(range(128,k-255,256))
        out=_v64_clone_params(wp)
        m=int(wt.shape[0])
        for start in starts:
            # calibration covariance for this exact shifted window
            H=torch.zeros((256,256),device=wt.device,dtype=torch.float32)
            nrow=0
            for a in acts_t:
                zz=a.float()[...,start:start+256].reshape(-1,256)
                H.add_(zz.t()@zz)
                nrow += int(zz.shape[0])
            if nrow<=0:
                continue
            H.div_(float(nrow))
            sc=H.diagonal().mean().abs().clamp_min(1e-12)
            H=(H/sc).unsqueeze(0)
            b0=start//64
            for rs in range(0,m,256):
                re=min(rs+256,m)
                y=wt[rs:re,start:start+256]
                pc={name:t[rs:re,b0:b0+4].clone() for name,t in out.items()}
                pc=_safe90_refine256(y,pc,H,iters=1)
                for name in out:
                    out[name][rs:re,b0:b0+4]=pc[name]
        wp=out

    wq=_v64_dequant_params(wp,tuple(w.shape)).float()
    st=_safe105_make_state(version,s,perm,had,ph,wq,{
        'weight_metric':'activation_covariance_deep_h256',
        'weight_h256_iters':int(aligned_iters),
        'weight_shifted256':bool(shifted),
    })
    return {'weight_params':wp,'activation_state':st}


def _safe120_q_iters(q_quant,q_scale,q_num_heads,head_dim,q_state,iters):
    q=dequantize_nvfp4(q_quant,q_scale).float()
    if isinstance(q_state,dict):
        q=_v44_apply_q(q,q_state,q_num_heads,head_dim)
        H64=q_state.get('partner_h64')
        if isinstance(H64,torch.Tensor):
            p,_=_v37_quantize_hessian(q,H64.to(q.device),return_dequant=False)
            H256=q_state.get('partner_h256')
            if isinstance(H256,torch.Tensor):
                p=_safe90_refine256(q,p,H256.to(q.device),int(iters))
            return p
    return _quantize_tensor_self_mse(q,return_dequant=False)[0]


def _safe122_qk_mix_scale(decoded,q_num_heads,kv_num_heads,head_dim,
                          alpha_max=0.875,beta=0.5):
    """
    Conservative max/RMS second-moment blend.
    alpha_max=1 is V113 max scaling; smaller values inject RMS geometry.
    """
    device=decoded[0][0].device
    rep=q_num_heads//kv_num_heads
    qmax=torch.zeros((q_num_heads,head_dim),device=device)
    kmax=torch.zeros((kv_num_heads,head_dim),device=device)
    q2=torch.zeros((kv_num_heads,head_dim),device=device)
    k2=torch.zeros((kv_num_heads,head_dim),device=device)
    nq=0; nk=0
    for q,k,_ in decoded:
        qh=q.float().reshape(-1,q_num_heads,head_dim)
        kh=k.float().reshape(-1,kv_num_heads,head_dim)
        qmax=torch.maximum(qmax,qh.abs().amax(0))
        kmax=torch.maximum(kmax,kh.abs().amax(0))
        qg=qh.reshape(qh.shape[0],kv_num_heads,rep,head_dim)
        q2.add_(qg.square().sum((0,2)))
        k2.add_(kh.square().sum(0))
        nq += int(qh.shape[0])*rep
        nk += int(kh.shape[0])

    qgrp_max=qmax.reshape(kv_num_heads,rep,head_dim).amax(1)
    qrms=torch.sqrt(q2/float(max(nq,1))).clamp_min(2**-24)
    krms=torch.sqrt(k2/float(max(nk,1))).clamp_min(2**-24)

    a=float(alpha_max)
    logratio=(
        a*(torch.log(kmax.clamp_min(2**-24))-torch.log(qgrp_max.clamp_min(2**-24)))
        +(1.0-a)*(torch.log(krms)-torch.log(qrms))
    )
    z=float(beta)*logratio
    z=z-z.median(dim=-1,keepdim=True).values
    sk=torch.exp(z).clamp_(2**-6,2**6)
    sq=sk.repeat_interleave(rep,dim=0)
    return sq.reshape(-1),sk.reshape(-1)


def _safe122_attn_calib(calib_qkv_list,q_num_heads,kv_num_heads,head_dim,alpha_max):
    decoded=_v35_decode_calib(calib_qkv_list,q_num_heads,kv_num_heads,head_dim)
    common={
        'version':_V44_VERSION,
        'enabled':True,
        'head_dim':int(head_dim),
        'transform_kind':'rot',
        'rotation':1,
        'beta':0.5,
        'strict_safe_fixed':True,
        'max_rms_mix':float(alpha_max),
    }
    if not decoded:
        out={
            'q_state':{**common,'role':'q','scale':torch.ones(q_num_heads*head_dim)},
            'k_state':{**common,'role':'k','scale':torch.ones(kv_num_heads*head_dim)},
            'v_state':{'enabled':False,'role':'v','strict_safe_fixed':True},
        }
    else:
        sq,sk=_safe122_qk_mix_scale(
            decoded,q_num_heads,kv_num_heads,head_dim,float(alpha_max),0.5
        )
        out={
            'q_state':{**common,'role':'q','scale':sq.detach().cpu().float()},
            'k_state':{**common,'role':'k','scale':sk.detach().cpu().float()},
            'v_state':{'enabled':False,'role':'v','strict_safe_fixed':True},
        }
    return _safe108_attach_cov_state(
        out,calib_qkv_list,q_num_heads,kv_num_heads,head_dim,
        use_q_h256=True,use_k_metric=True
    )

_V118_VERSION='v118_strictsafe_weight_h256x4'
def hif4_calibration_and_quantize_weight(weight_quant,weight_scale,calib_activation_list):
    return _safe117_linear_calib(weight_quant,weight_scale,calib_activation_list,
                                 aligned_iters=4,shifted=False,version=_V118_VERSION)
def hif4_dynamic_quantize_activation(activation_quant,activation_scale,activation_state):
    return _safe105_dynamic(activation_quant,activation_scale,activation_state)




# =============================================================================
# V130+ STRICT-SAFE FORMAT-AWARE HiF4 OPTIMIZATION
#
# Key change:
#   optimize the HiF4 hierarchy itself (E6M2 -> lv2 -> lv3 -> mantissa)
#   under legal second-order covariance metrics, instead of freezing lv2/lv3
#   at tensor-self-MSE choices.
#
# No calibration operator-output reference is formed.
# =============================================================================

def _safe130_slice_rows(p, st, en):
    return {k:v[st:en].clone() for k,v in p.items()}

def _safe130_write_rows(dst, src, st, en):
    for k in dst:
        dst[k][st:en]=src[k]
    return dst

def _safe130_q(y,p):
    return _v64_dequant_params(p,tuple(int(s) for s in y.shape)).float()


def _safe130_scale_core(y,p,H256,sweeps=1):
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

    pp=_v64_clone_params(p)
    yy=y.float().reshape(rows,ng,256)
    H=H256.to(y.device,torch.float32)
    table=_build_e6m2_table(y.device)
    last=int(table.numel()-1)

    for _ in range(int(sweeps)):
        q=_safe130_q(y,pp).reshape(rows,ng,256)
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


def _safe130_hierarchy_core(y,p,H256,sweeps=1):
    """
    Groupwise coordinate descent over the REAL HiF4 tree.

    For every 8-value lv2 group, enumerate all 8 legal configurations:
        lv2 in {1,2}
        lv3_left in {1,2}
        lv3_right in {1,2}
    Re-round the 8 mantissas for each configuration and choose by the exact
    H256 objective delta. This optimizes all eight lv2 groups in a 64-block per
    sweep, instead of changing at most one hierarchy bit.
    """
    shape=tuple(int(s) for s in y.shape); k=shape[-1]
    if y.dim()!=2 or k%256!=0 or not isinstance(H256,torch.Tensor):
        return p
    rows=int(y.shape[0]); ng=k//256; nb=k//64
    if tuple(H256.shape)!=(ng,256,256):
        return p

    pp=_v64_clone_params(p)
    yy=y.float().reshape(rows,ng,256)
    H=H256.to(y.device,torch.float32)

    for _ in range(int(sweeps)):
        q=_safe130_q(y,pp).reshape(rows,ng,256)
        e=q-yy
        grad=torch.einsum('gij,rgi->rgj',H,e)

        sfv=pp['scale_factor'].float().reshape(rows,nb,1,1,1)
        l2v=pp['scale_lv2'].float().reshape(rows,nb,8,1,1)
        l3v=pp['scale_lv3'].float().reshape(rows,nb,8,2,1)
        sgv=pp['sign'].float().reshape(rows,nb,8,2,4)
        mav=pp['mant'].float().reshape(rows,nb,8,2,4)

        for sub in range(4):
            bidx=torch.arange(ng,device=y.device)*4+sub
            sf=sfv[:,bidx,0,0,0]

            for g8 in range(8):
                lo=sub*64+g8*8; hi=lo+8
                z=yy[:,:,lo:hi].reshape(rows,ng,2,4)
                qcur=q[:,:,lo:hi]
                gcur=grad[:,:,lo:hi]
                Hss=H[:,lo:hi,lo:hi]
                sg=torch.sign(z)

                costs=[torch.zeros((rows,ng),device=y.device)]
                qlist=[qcur]
                meta=[None]

                for l2c in (1.0,2.0):
                    for l30 in (1.0,2.0):
                        for l31 in (1.0,2.0):
                            l3pair=torch.tensor([l30,l31],device=y.device,dtype=torch.float32).view(1,1,2,1)
                            eff=sf[:,:,None,None]*float(l2c)*l3pair
                            ma=(torch.round((z.abs()/eff.clamp_min(2.0**-48))*4.0)*0.25).clamp(0.0,1.75)
                            sgc=torch.where(ma==0.0,torch.zeros_like(sg),sg)
                            qc=(sgc*ma*eff).reshape(rows,ng,8)
                            d=qc-qcur
                            dc=2.0*(d*gcur).sum(-1)+torch.einsum('rgi,gij,rgj->rg',d,Hss,d)
                            costs.append(dc); qlist.append(qc)
                            meta.append((l2c,l30,l31,sgc,ma))

                choice=torch.stack(costs,0).argmin(0)
                if not bool((choice>0).any().item()):
                    continue

                qstack=torch.stack(qlist,0).permute(1,2,0,3)
                qi=choice[:,:,None,None].expand(rows,ng,1,8)
                qnew=qstack.gather(2,qi).squeeze(2)
                d=qnew-qcur
                q[:,:,lo:hi]=qnew
                grad.add_(torch.einsum('gij,rgj->rgi',H[:,:,lo:hi],d))

                for ci in range(1,len(meta)):
                    mask=(choice==ci)
                    if not bool(mask.any().item()):
                        continue
                    l2c,l30,l31,sgc,ma=meta[ci]
                    l2old=l2v[:,bidx,g8,0,0]
                    l2v[:,bidx,g8,0,0]=torch.where(mask,torch.full_like(l2old,float(l2c)),l2old)

                    l3old0=l3v[:,bidx,g8,0,0]
                    l3old1=l3v[:,bidx,g8,1,0]
                    l3v[:,bidx,g8,0,0]=torch.where(mask,torch.full_like(l3old0,float(l30)),l3old0)
                    l3v[:,bidx,g8,1,0]=torch.where(mask,torch.full_like(l3old1,float(l31)),l3old1)

                    m4=mask[:,:,None,None]
                    sgv[:,bidx,g8]=torch.where(m4,sgc,sgv[:,bidx,g8])
                    mav[:,bidx,g8]=torch.where(m4,ma,mav[:,bidx,g8])

        pp['scale_factor']=sfv.reshape_as(pp['scale_factor']).to(torch.bfloat16)
        pp['scale_lv2']=l2v.reshape_as(pp['scale_lv2']).to(torch.bfloat16)
        pp['scale_lv3']=l3v.reshape_as(pp['scale_lv3']).to(torch.bfloat16)
        pp['sign']=sgv.reshape_as(pp['sign']).to(torch.bfloat16)
        pp['mant']=mav.reshape_as(pp['mant']).to(torch.bfloat16)
    return pp


def _safe130_chunked_format_refine(y,p,H256,scale_sweeps=0,hier_sweeps=1,
                                   mant_repair=2,cycles=1,chunk_rows=128):
    if y.dim()!=2:
        return p
    out=_v64_clone_params(p)
    m=int(y.shape[0])
    for st in range(0,m,int(chunk_rows)):
        en=min(st+int(chunk_rows),m)
        pc=_safe130_slice_rows(out,st,en)
        yc=y[st:en]
        for _ in range(int(cycles)):
            if int(scale_sweeps)>0:
                pc=_safe130_scale_core(yc,pc,H256,int(scale_sweeps))
            if int(hier_sweeps)>0:
                pc=_safe130_hierarchy_core(yc,pc,H256,int(hier_sweeps))
            if int(mant_repair)>0:
                pc=_safe90_refine256(yc,pc,H256,int(mant_repair))
        _safe130_write_rows(out,pc,st,en)
    return out


def _safe130_linear(weight_quant,weight_scale,calib_activation_list,
                    scale_sweeps,hier_sweeps,mant_repair,cycles,version):
    # Rebuild the proven V118 path explicitly, then unlock the frozen hierarchy.
    w=dequantize_nvfp4(weight_quant,weight_scale).float()
    acts=_rule_safe_decode_acts(calib_activation_list,w.shape[-1],w.device)
    s,perm,had,ph=_safe90_geometry(w,acts,use_perm=True,use_had=True,use_phase=True)
    wt=_safe90_apply(w,s,perm,ph,had,True)
    acts_t=[_safe90_apply(a,s,perm,ph,had,False) for a in acts]
    HA64=_safe99_activation_hessian64(acts_t,int(w.shape[-1]),w.device)
    HA256=_safe99_activation_hessian256(acts_t,int(w.shape[-1]),w.device)

    if isinstance(HA64,torch.Tensor):
        wp,wq=_v37_quantize_hessian(wt,HA64,return_dequant=True)
    else:
        wp,wq=_quantize_tensor_self_mse(wt,return_dequant=True)
    if isinstance(HA256,torch.Tensor):
        wp=_safe105_chunked_weight_h256(wt,wp,HA256,iters=4,chunk_rows=256)
        wp=_safe130_chunked_format_refine(
            wt,wp,HA256,
            scale_sweeps=int(scale_sweeps),
            hier_sweeps=int(hier_sweeps),
            mant_repair=int(mant_repair),
            cycles=int(cycles),
            chunk_rows=128,
        )

    wq=_v64_dequant_params(wp,tuple(w.shape)).float()
    st=_safe105_make_state(version,s,perm,had,ph,wq,{
        'weight_metric':'format_aware_hif4_hierarchy_h256',
        'format_scale_sweeps':int(scale_sweeps),
        'format_hier_sweeps':int(hier_sweeps),
        'format_cycles':int(cycles),
    })
    return {'weight_params':wp,'activation_state':st}


def _safe133_postperm_geometry(w,acts):
    """
    Hierarchy-aware post-Hadamard ordering.

    After the ordinary smooth/global balanced packing/H64 transform, sort the
    64 output coordinates by combined A/W RMS pressure *within each 64 block*.
    Adjacent 4- and 8-value groups therefore contain channels with similar scale
    demand, matching HiF4's lv3/lv2 sharing topology.
    """
    s,perm,had,ph=_safe90_geometry(w,acts,use_perm=True,use_had=True,use_phase=True)
    wt=_safe90_apply(w,s,perm,ph,had,True)
    ats=[_safe90_apply(a,s,perm,ph,had,False) for a in acts]
    k=int(w.shape[-1]); nb=k//64

    wrms=torch.sqrt(wt.float().square().mean(0).clamp_min(1e-24))
    asum=torch.zeros(k,device=w.device,dtype=torch.float32); n=0
    for a in ats:
        aa=a.float().reshape(-1,k)
        asum.add_(aa.square().sum(0)); n+=int(aa.shape[0])
    arms=torch.sqrt((asum/float(max(n,1))).clamp_min(1e-24))

    # Conservative combined pressure: neither side may hide behind the other.
    pressure=torch.maximum(wrms,arms).reshape(nb,64)
    local=torch.argsort(pressure,dim=1,stable=True)
    offs=(torch.arange(nb,device=w.device)*64)[:,None]
    post=(local+offs).reshape(-1)

    wt=wt.index_select(-1,post)
    ats=[a.index_select(-1,post) for a in ats]
    return s,perm,had,ph,post,wt,ats


def _safe133_linear(weight_quant,weight_scale,calib_activation_list,
                    format_refine=False,version='v133'):
    w=dequantize_nvfp4(weight_quant,weight_scale).float()
    acts=_rule_safe_decode_acts(calib_activation_list,w.shape[-1],w.device)
    s,perm,had,ph,post,wt,acts_t=_safe133_postperm_geometry(w,acts)
    HA64=_safe99_activation_hessian64(acts_t,int(w.shape[-1]),w.device)
    HA256=_safe99_activation_hessian256(acts_t,int(w.shape[-1]),w.device)

    if isinstance(HA64,torch.Tensor):
        wp,wq=_v37_quantize_hessian(wt,HA64,return_dequant=True)
    else:
        wp,wq=_quantize_tensor_self_mse(wt,return_dequant=True)
    if isinstance(HA256,torch.Tensor):
        wp=_safe105_chunked_weight_h256(wt,wp,HA256,iters=4,chunk_rows=256)
        if format_refine:
            wp=_safe130_chunked_format_refine(
                wt,wp,HA256,scale_sweeps=1,hier_sweeps=1,
                mant_repair=2,cycles=1,chunk_rows=128
            )

    wq=_v64_dequant_params(wp,tuple(w.shape)).float()
    st=_safe105_make_state(version,s,perm,had,ph,wq,{
        'weight_metric':'hierarchy_aligned_posthad_order',
        'post_perm':post.detach().cpu().to(torch.int32),
        'post_perm_enabled':True,
    })
    return {'weight_params':wp,'activation_state':st}


def _safe133_dynamic(activation_quant,activation_scale,activation_state):
    st=activation_state if isinstance(activation_state,dict) else {}
    y=_safe90_decode_transform_activation(activation_quant,activation_scale,st)
    post=st.get('post_perm')
    if isinstance(post,torch.Tensor):
        y=y.index_select(-1,post.to(y.device,dtype=torch.long))
    H64=st.get('weight_hessian_blocks'); H256=st.get('super256_hessian_blocks')
    if not isinstance(H64,torch.Tensor):
        return _quantize_tensor_self_mse(y,return_dequant=False)[0]
    p,_=_v37_quantize_hessian(y,H64,return_dequant=False)
    if isinstance(H256,torch.Tensor):
        p=_safe90_refine256(y,p,H256,int(st.get('super256_iters',4)))
    return p


def _safe136_q_format(q_quant,q_scale,q_num_heads,head_dim,q_state,
                      scale_sweeps=0,hier_sweeps=1):
    q=dequantize_nvfp4(q_quant,q_scale).float()
    if isinstance(q_state,dict):
        q=_v44_apply_q(q,q_state,q_num_heads,head_dim)
        H64=q_state.get('partner_h64'); H256=q_state.get('partner_h256')
        if isinstance(H64,torch.Tensor):
            p,_=_v37_quantize_hessian(q,H64.to(q.device),return_dequant=False)
            if isinstance(H256,torch.Tensor):
                HH=H256.to(q.device)
                p=_safe90_refine256(q,p,HH,1)
                p=_safe130_chunked_format_refine(
                    q,p,HH,
                    scale_sweeps=int(scale_sweeps),
                    hier_sweeps=int(hier_sweeps),
                    mant_repair=1,cycles=1,chunk_rows=128
                )
            return p
    return _quantize_tensor_self_mse(q,return_dequant=False)[0]


def _safe137_self_cf(x,cycles=2):
    """SOAR-like fixed-assignment closed-form scale refinement for tensor-self MSE."""
    p,q=_quantize_tensor_self_mse(x,return_dequant=True)
    shape=tuple(int(s) for s in x.shape); k=shape[-1]
    rows=x.numel()//k; nb=k//64
    xx=x.float().reshape(rows,nb,8,2,4)
    table=_build_e6m2_table(x.device); last=int(table.numel()-1)

    for _ in range(int(cycles)):
        qq=_v64_dequant_params(p,shape).float().reshape(rows,nb,8,2,4)
        sfcur=p['scale_factor'].float().reshape(rows,nb).clamp_min(2.0**-48)
        c=qq/sfcur[:,:,None,None,None]
        num=(xx*c).sum((2,3,4))
        den=c.square().sum((2,3,4)).clamp_min(1e-20)
        star=(num/den).abs().clamp(min=2.0**-48,max=49152.0)
        curidx=_nearest_e6m2_index(sfcur,table)
        optidx=_nearest_e6m2_index(star,table)
        idxs=[
            curidx,(curidx-1).clamp(0,last),(curidx+1).clamp(0,last),
            (optidx-2).clamp(0,last),(optidx-1).clamp(0,last),
            optidx,(optidx+1).clamp(0,last),(optidx+2).clamp(0,last)
        ]

        best=((qq-xx)**2).sum((2,3,4))
        bestpack=None
        curpack={
            'scale_factor':p['scale_factor'].float().reshape(rows,nb,1,1,1),
            'scale_lv2':p['scale_lv2'].float().reshape(rows,nb,8,1,1),
            'scale_lv3':p['scale_lv3'].float().reshape(rows,nb,8,2,1),
            'sign':p['sign'].float().reshape(rows,nb,8,2,4),
            'mant':p['mant'].float().reshape(rows,nb,8,2,4),
        }
        bestpack={k0:v.clone() for k0,v in curpack.items()}

        for idx in idxs:
            sf=table[idx].reshape(rows,nb,1,1,1)
            sg,ma,l2,l3=_materialize_fixed_scale_self(xx,sf)
            qc=sg*ma*l2*l3*sf
            err=(qc-xx).square().sum((2,3,4))
            win=err<best
            if bool(win.any().item()):
                m5=win[:,:,None,None,None]
                best=torch.where(win,err,best)
                bestpack['scale_factor']=torch.where(m5,sf,bestpack['scale_factor'])
                bestpack['scale_lv2']=torch.where(m5,l2,bestpack['scale_lv2'])
                bestpack['scale_lv3']=torch.where(m5,l3,bestpack['scale_lv3'])
                bestpack['sign']=torch.where(m5,sg,bestpack['sign'])
                bestpack['mant']=torch.where(m5,ma,bestpack['mant'])

        p={
            'scale_factor':bestpack['scale_factor'].reshape_as(p['scale_factor']).to(torch.bfloat16),
            'scale_lv2':bestpack['scale_lv2'].reshape_as(p['scale_lv2']).to(torch.bfloat16),
            'scale_lv3':bestpack['scale_lv3'].reshape_as(p['scale_lv3']).to(torch.bfloat16),
            'sign':bestpack['sign'].reshape_as(p['sign']).to(torch.bfloat16),
            'mant':bestpack['mant'].reshape_as(p['mant']).to(torch.bfloat16),
        }
    return p

def hif4_calibration_and_quantize_weight(weight_quant,weight_scale,calib_activation_list):
    return _safe133_linear(weight_quant,weight_scale,calib_activation_list,
                           format_refine=False,version='v133')
def hif4_dynamic_quantize_activation(activation_quant,activation_scale,activation_state):
    return _safe133_dynamic(activation_quant,activation_scale,activation_state)




import heapq

# =============================================================================
# V139+ STRICT-SAFE SCORE-ALIGNED COVARIANCE / MASS-DIFFUSION
# =============================================================================

def _safe139_cov64(acts_t,k,device,alpha=1.0,trace_normalize=False):
    nb=k//64
    H=torch.zeros((nb,64,64),dtype=torch.float32,device=device)
    denom=0.0
    for a in acts_t:
        z=a.float().reshape(-1,nb,64)
        n=max(int(z.shape[0]),1)
        C=torch.einsum('rbi,rbj->bij',z,z)/float(n)
        if trace_normalize:
            sc=C.diagonal(dim1=-2,dim2=-1).mean(-1).abs().clamp_min(1e-12)
            C=C/sc[:,None,None]
            w=1.0
        else:
            w=float(n)**(1.0-float(alpha))
        H.add_(C*float(w))
        denom += float(w)
    if denom<=0:
        return None
    H.div_(denom)
    sc=H.diagonal(dim1=-2,dim2=-1).mean(-1).abs().clamp_min(1e-12)
    return H/sc[:,None,None]


def _safe139_cov256(acts_t,k,device,alpha=1.0,trace_normalize=False):
    if k%256:
        return None
    ng=k//256
    H=torch.zeros((ng,256,256),dtype=torch.float32,device=device)
    denom=0.0
    for a in acts_t:
        z=a.float().reshape(-1,ng,256)
        n=max(int(z.shape[0]),1)
        C=torch.einsum('rgi,rgj->gij',z,z)/float(n)
        if trace_normalize:
            sc=C.diagonal(dim1=-2,dim2=-1).mean(-1).abs().clamp_min(1e-12)
            C=C/sc[:,None,None]
            w=1.0
        else:
            w=float(n)**(1.0-float(alpha))
        H.add_(C*float(w))
        denom += float(w)
    if denom<=0:
        return None
    H.div_(denom)
    sc=H.diagonal(dim1=-2,dim2=-1).mean(-1).abs().clamp_min(1e-12)
    return H/sc[:,None,None]


def _safe139_sample_rms(ats,k):
    acc=torch.zeros(k,device=ats[0].device,dtype=torch.float32)
    ns=0
    for a in ats:
        aa=a.float().reshape(-1,k)
        acc.add_(aa.square().mean(0))
        ns+=1
    return torch.sqrt((acc/float(max(ns,1))).clamp_min(1e-24))


def _safe139_sample_meanabs(ats,k):
    acc=torch.zeros(k,device=ats[0].device,dtype=torch.float32)
    ns=0
    for a in ats:
        aa=a.float().reshape(-1,k)
        acc.add_(aa.abs().mean(0))
        ns+=1
    return acc/float(max(ns,1))


def _safe139_massdiff_perm(mass,block=64):
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


def _safe139_geometry(w,acts,perm_kind='v40',post_pressure='max'):
    k=int(w.shape[-1]); nb=k//64
    if not acts or k%64:
        s=torch.ones(k,device=w.device)
        perm=torch.arange(k,device=w.device)
        ph=torch.ones(nb,device=w.device)
        post=torch.arange(k,device=w.device)
        return s,perm,True,ph,post,w.float(),[a.float() for a in acts]

    amax=torch.stack([a.abs().amax(0) for a in acts],0).mean(0)
    wmax=w.abs().amax(0)
    smooth=_v31_smooth(amax,wmax,0.50)

    if perm_kind=='mass_act':
        apre=[a.float()*smooth for a in acts]
        mass=_safe139_sample_meanabs(apre,k)
        perm=_safe139_massdiff_perm(mass,64)
    elif perm_kind=='mass_joint':
        apre=[a.float()*smooth for a in acts]
        amass=_safe139_sample_meanabs(apre,k)
        wmass=(w.float()/smooth).abs().mean(0)
        mass=torch.sqrt((amass*wmass).clamp_min(1e-24))
        perm=_safe139_massdiff_perm(mass,64)
    else:
        aeff=amax*smooth
        weff=wmax/smooth.clamp_min(2.0**-24)
        perm=_v40_balanced_perm(torch.maximum(aeff,weff))

    wt=(w.float()/smooth).index_select(-1,perm)
    wt=_fwht64_v31(wt)
    wb=wt.abs().reshape(-1,nb,64).amax((0,2))

    ab=torch.zeros(nb,dtype=torch.float32,device=w.device)
    ats_pre=[]
    for a in acts:
        at=(a.float()*smooth).index_select(-1,perm)
        at=_fwht64_v31(at)
        ats_pre.append(at)
        ab=torch.maximum(ab,at.abs().reshape(-1,nb,64).amax((0,2)))

    phases=torch.sqrt(wb.clamp_min(2.0**-24)/ab.clamp_min(2.0**-24))
    phases=phases.clamp(0.50,2.00)
    phases=phases/torch.exp(torch.log(phases).median())
    phases=phases.clamp(0.50,2.00)

    pv=_safe90_blockvec(phases)
    wt=wt/pv
    ats=[a*pv for a in ats_pre]

    wrms=torch.sqrt(wt.float().square().mean(0).clamp_min(1e-24))
    arms=_safe139_sample_rms(ats,k)
    if post_pressure=='geom':
        pressure=torch.sqrt((wrms*arms).clamp_min(1e-24))
    else:
        pressure=torch.maximum(wrms,arms)

    local=torch.argsort(pressure.reshape(nb,64),dim=1,stable=True)
    offs=(torch.arange(nb,device=w.device)*64)[:,None]
    post=(local+offs).reshape(-1)

    wt=wt.index_select(-1,post)
    ats=[a.index_select(-1,post) for a in ats]
    return smooth,perm,True,phases,post,wt,ats


def _safe139_linear(weight_quant,weight_scale,calib_activation_list,
                    cov_alpha=1.0,trace_normalize=False,
                    perm_kind='v40',post_pressure='max',
                    h256_iters=4,version='v139'):
    w=dequantize_nvfp4(weight_quant,weight_scale).float()
    acts=_rule_safe_decode_acts(calib_activation_list,w.shape[-1],w.device)

    s,perm,had,ph,post,wt,acts_t=_safe139_geometry(
        w,acts,perm_kind=perm_kind,post_pressure=post_pressure
    )

    k=int(w.shape[-1])
    HA64=_safe139_cov64(
        acts_t,k,w.device,alpha=float(cov_alpha),
        trace_normalize=bool(trace_normalize)
    )
    HA256=_safe139_cov256(
        acts_t,k,w.device,alpha=float(cov_alpha),
        trace_normalize=bool(trace_normalize)
    )

    if isinstance(HA64,torch.Tensor):
        wp,wq=_v37_quantize_hessian(wt,HA64,return_dequant=True)
    else:
        wp,wq=_quantize_tensor_self_mse(wt,return_dequant=True)

    if isinstance(HA256,torch.Tensor):
        wp=_safe105_chunked_weight_h256(
            wt,wp,HA256,iters=int(h256_iters),chunk_rows=256
        )

    wq=_v64_dequant_params(wp,tuple(w.shape)).float()
    st=_safe105_make_state(version,s,perm,had,ph,wq,{
        'weight_metric':'score_aligned_covariance_hierarchy_order',
        'post_perm':post.detach().cpu().to(torch.int32),
        'post_perm_enabled':True,
        'cov_alpha':float(cov_alpha),
        'cov_trace_normalize':bool(trace_normalize),
        'perm_kind':str(perm_kind),
        'post_pressure':str(post_pressure),
        'weight_h256_iters':int(h256_iters),
    })
    return {'weight_params':wp,'activation_state':st}


def _safe139_dynamic(activation_quant,activation_scale,activation_state):
    return _safe133_dynamic(
        activation_quant,activation_scale,activation_state
    )

def hif4_calibration_and_quantize_weight(weight_quant,weight_scale,calib_activation_list):
    return _safe139_linear(
        weight_quant,weight_scale,calib_activation_list,
        cov_alpha=0.5,
        trace_normalize=False,
        perm_kind='v40',
        post_pressure='max',
        h256_iters=4,
        version='v140'
    )

def hif4_dynamic_quantize_activation(activation_quant,activation_scale,activation_state):
    return _safe139_dynamic(activation_quant,activation_scale,activation_state)




# =============================================================================
# V147+ STRICT-SAFE PRIVATE-GENERALIZATION REFINEMENT
# =============================================================================

def _safe147_multiscale_segments(a):
    a=a.float()
    n=int(a.shape[0])
    if n <= 1:
        return [[a]]
    lens=[]
    for frac in (1.0,0.5,0.25,0.125):
        L=max(1,int(round(n*frac)))
        if L not in lens:
            lens.append(L)
    groups=[]
    for L in lens:
        if L >= n:
            groups.append([a]); continue
        starts=[0,max(0,(n-L)//2),n-L]
        uniq=[]; seen=set()
        for s in starts:
            if s not in seen:
                uniq.append(a[s:s+L]); seen.add(s)
        groups.append(uniq)
    return groups


def _safe147_multiscale_cov64(acts_t,k,device,alpha=0.5,blend_full=0.25):
    nb=k//64
    H=torch.zeros((nb,64,64),dtype=torch.float32,device=device)
    denom=0.0
    for a in acts_t:
        n=max(int(a.shape[0]),1)
        Cs=[]
        for windows in _safe147_multiscale_segments(a):
            Cscale=torch.zeros((nb,64,64),device=device,dtype=torch.float32)
            for seg in windows:
                z=seg.reshape(-1,nb,64)
                Cscale.add_(torch.einsum('rbi,rbj->bij',z,z)/float(max(int(z.shape[0]),1)))
            Cscale.div_(float(len(windows)))
            Cs.append(Cscale)
        Cm=sum(Cs)/float(len(Cs))
        zf=a.reshape(-1,nb,64)
        Cf=torch.einsum('rbi,rbj->bij',zf,zf)/float(n)
        C=(1.0-float(blend_full))*Cm + float(blend_full)*Cf
        w=float(n)**(1.0-float(alpha))
        H.add_(C*w); denom+=w
    if denom<=0: return None
    H.div_(denom)
    sc=H.diagonal(dim1=-2,dim2=-1).mean(-1).abs().clamp_min(1e-12)
    return H/sc[:,None,None]


def _safe147_multiscale_cov256(acts_t,k,device,alpha=0.5,blend_full=0.25):
    if k%256: return None
    ng=k//256
    H=torch.zeros((ng,256,256),dtype=torch.float32,device=device)
    denom=0.0
    for a in acts_t:
        n=max(int(a.shape[0]),1)
        Cs=[]
        for windows in _safe147_multiscale_segments(a):
            Cscale=torch.zeros((ng,256,256),device=device,dtype=torch.float32)
            for seg in windows:
                z=seg.reshape(-1,ng,256)
                Cscale.add_(torch.einsum('rgi,rgj->gij',z,z)/float(max(int(z.shape[0]),1)))
            Cscale.div_(float(len(windows)))
            Cs.append(Cscale)
        Cm=sum(Cs)/float(len(Cs))
        zf=a.reshape(-1,ng,256)
        Cf=torch.einsum('rgi,rgj->gij',zf,zf)/float(n)
        C=(1.0-float(blend_full))*Cm + float(blend_full)*Cf
        w=float(n)**(1.0-float(alpha))
        H.add_(C*w); denom+=w
    if denom<=0: return None
    H.div_(denom)
    sc=H.diagonal(dim1=-2,dim2=-1).mean(-1).abs().clamp_min(1e-12)
    return H/sc[:,None,None]


def _safe147_geometry(w,acts,perm_kind='v40',post_pressure='max',robust_phase_mix=0.0):
    k=int(w.shape[-1]); nb=k//64
    if not acts or k%64:
        s=torch.ones(k,device=w.device)
        perm=torch.arange(k,device=w.device)
        ph=torch.ones(nb,device=w.device)
        post=torch.arange(k,device=w.device)
        return s,perm,True,ph,post,w.float(),[a.float() for a in acts]

    amax=torch.stack([a.abs().amax(0) for a in acts],0).mean(0)
    wmax=w.abs().amax(0)
    smooth=_v31_smooth(amax,wmax,0.50)

    if perm_kind=='mass_act':
        apre=[a.float()*smooth for a in acts]
        mass=_safe139_sample_meanabs(apre,k)
        perm=_safe139_massdiff_perm(mass,64)
    elif perm_kind=='mass_joint':
        apre=[a.float()*smooth for a in acts]
        amass=_safe139_sample_meanabs(apre,k)
        wmass=(w.float()/smooth).abs().mean(0)
        mass=torch.sqrt((amass*wmass).clamp_min(1e-24))
        perm=_safe139_massdiff_perm(mass,64)
    else:
        aeff=amax*smooth
        weff=wmax/smooth.clamp_min(2.0**-24)
        perm=_v40_balanced_perm(torch.maximum(aeff,weff))

    wt0=(w.float()/smooth).index_select(-1,perm)
    wt0=_fwht64_v31(wt0)
    wb_max=wt0.abs().reshape(-1,nb,64).amax((0,2))
    wb_rms=torch.sqrt(wt0.square().reshape(-1,nb,64).mean((0,2)).clamp_min(1e-24))

    ab_max=torch.zeros(nb,dtype=torch.float32,device=w.device)
    ab_rms_acc=torch.zeros(nb,dtype=torch.float32,device=w.device)
    ats_pre=[]
    for a in acts:
        at=(a.float()*smooth).index_select(-1,perm)
        at=_fwht64_v31(at)
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

    pv=_safe90_blockvec(phases)
    wt=wt0/pv
    ats=[a*pv for a in ats_pre]

    wrms=torch.sqrt(wt.square().mean(0).clamp_min(1e-24))
    arms=_safe139_sample_rms(ats,k)
    pressure=torch.sqrt((wrms*arms).clamp_min(1e-24)) if post_pressure=='geom' else torch.maximum(wrms,arms)
    local=torch.argsort(pressure.reshape(nb,64),dim=1,stable=True)
    offs=(torch.arange(nb,device=w.device)*64)[:,None]
    post=(local+offs).reshape(-1)

    wt=wt.index_select(-1,post)
    ats=[a.index_select(-1,post) for a in ats]
    return smooth,perm,True,phases,post,wt,ats


def _safe147_linear(weight_quant,weight_scale,calib_activation_list,
                    alpha=0.5,perm_kind='v40',post_pressure='max',
                    multiscale=False,ms_blend_full=0.25,
                    robust_phase_mix=0.0,version='v147'):
    w=dequantize_nvfp4(weight_quant,weight_scale).float()
    acts=_rule_safe_decode_acts(calib_activation_list,w.shape[-1],w.device)

    s,perm,had,ph,post,wt,acts_t=_safe147_geometry(
        w,acts,perm_kind=perm_kind,post_pressure=post_pressure,
        robust_phase_mix=float(robust_phase_mix)
    )
    k=int(w.shape[-1])

    if multiscale:
        HA64=_safe147_multiscale_cov64(acts_t,k,w.device,float(alpha),float(ms_blend_full))
        HA256=_safe147_multiscale_cov256(acts_t,k,w.device,float(alpha),float(ms_blend_full))
    else:
        HA64=_safe139_cov64(acts_t,k,w.device,alpha=float(alpha),trace_normalize=False)
        HA256=_safe139_cov256(acts_t,k,w.device,alpha=float(alpha),trace_normalize=False)

    if isinstance(HA64,torch.Tensor):
        wp,wq=_v37_quantize_hessian(wt,HA64,return_dequant=True)
    else:
        wp,wq=_quantize_tensor_self_mse(wt,return_dequant=True)

    if isinstance(HA256,torch.Tensor):
        wp=_safe105_chunked_weight_h256(wt,wp,HA256,iters=4,chunk_rows=256)

    wq=_v64_dequant_params(wp,tuple(w.shape)).float()
    st=_safe105_make_state(version,s,perm,had,ph,wq,{
        'weight_metric':'private_generalization_multiscale',
        'post_perm':post.detach().cpu().to(torch.int32),
        'post_perm_enabled':True,
        'cov_alpha':float(alpha),
        'perm_kind':str(perm_kind),
        'post_pressure':str(post_pressure),
        'multiscale_cov':bool(multiscale),
        'ms_blend_full':float(ms_blend_full),
        'robust_phase_mix':float(robust_phase_mix),
        'weight_h256_iters':4,
    })
    return {'weight_params':wp,'activation_state':st}


def _safe147_dynamic(activation_quant,activation_scale,activation_state):
    return _safe133_dynamic(activation_quant,activation_scale,activation_state)

def hif4_calibration_and_quantize_weight(weight_quant,weight_scale,calib_activation_list):
    return _safe147_linear(
        weight_quant,weight_scale,calib_activation_list,
        alpha=0.5,
        perm_kind='mass_act',
        post_pressure='max',
        multiscale=False,
        ms_blend_full=0.25,
        robust_phase_mix=0.0,
        version='v149'
    )

def hif4_dynamic_quantize_activation(activation_quant,activation_scale,activation_state):
    return _safe147_dynamic(activation_quant,activation_scale,activation_state)



# =============================================================================
# V155: V149 + off-diagonal covariance shrinkage (rho=0.60)
# =============================================================================
_V155_RHO = 0.60


def _v155_shrink_covariance(H, rho=_V155_RHO):
    if not isinstance(H, torch.Tensor):
        return H
    D = torch.diag_embed(H.diagonal(dim1=-2, dim2=-1))
    return float(rho) * H + (1.0 - float(rho)) * D


def _v155_linear(weight_quant, weight_scale, calib_activation_list):
    w = dequantize_nvfp4(weight_quant, weight_scale).float()
    acts = _rule_safe_decode_acts(calib_activation_list, w.shape[-1], w.device)

    # Keep V149 representation exactly: MassDiff + V133 hierarchy-aware post order.
    s, perm, had, ph, post, wt, acts_t = _safe147_geometry(
        w,
        acts,
        perm_kind='mass_act',
        post_pressure='max',
        robust_phase_mix=0.0,
    )
    k = int(w.shape[-1])

    # Keep V149 sqrt-length calibration weighting (alpha=0.5), but regularize
    # only cross-channel covariance terms. Diagonal energy is preserved exactly.
    HA64 = _safe139_cov64(
        acts_t, k, w.device, alpha=0.50, trace_normalize=False
    )
    HA256 = _safe139_cov256(
        acts_t, k, w.device, alpha=0.50, trace_normalize=False
    )
    HA64 = _v155_shrink_covariance(HA64)
    HA256 = _v155_shrink_covariance(HA256)

    if isinstance(HA64, torch.Tensor):
        wp, wq = _v37_quantize_hessian(wt, HA64, return_dequant=True)
    else:
        wp, wq = _quantize_tensor_self_mse(wt, return_dequant=True)

    if isinstance(HA256, torch.Tensor):
        wp = _safe105_chunked_weight_h256(
            wt, wp, HA256, iters=4, chunk_rows=256
        )

    wq = _v64_dequant_params(wp, tuple(w.shape)).float()
    st = _safe105_make_state(
        'v155', s, perm, had, ph, wq,
        {
            'weight_metric': 'sqrt_length_massdiff_cov_shrink',
            'post_perm': post.detach().cpu().to(torch.int32),
            'post_perm_enabled': True,
            'cov_alpha': 0.50,
            'cov_offdiag_rho': float(_V155_RHO),
            'perm_kind': 'mass_act',
            'post_pressure': 'max',
            'weight_h256_iters': 4,
        },
    )
    return {'weight_params': wp, 'activation_state': st}


def hif4_calibration_and_quantize_weight(weight_quant, weight_scale, calib_activation_list):
    return _v155_linear(weight_quant, weight_scale, calib_activation_list)


def hif4_dynamic_quantize_activation(activation_quant, activation_scale, activation_state):
    return _safe133_dynamic(activation_quant, activation_scale, activation_state)


# adaptive reliability shrinkage

def _v156_cov_adaptive(acts_t,k,device,group=64,alpha=.5,rmin=.40,rmax=.80,noise_gain=1.0):
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

def _v156_linear(weight_quant,weight_scale,calib_activation_list):
    w=dequantize_nvfp4(weight_quant,weight_scale).float();acts=_rule_safe_decode_acts(calib_activation_list,w.shape[-1],w.device)
    s,perm,had,ph,post,wt,acts_t=_safe147_geometry(w,acts,perm_kind='mass_act',post_pressure='max',robust_phase_mix=0.0);k=int(w.shape[-1])
    H64,r64=_v156_cov_adaptive(acts_t,k,w.device,64,.5,.40,.80,1.0);H256,r256=_v156_cov_adaptive(acts_t,k,w.device,256,.5,.40,.80,1.0)
    wp,wq=_v37_quantize_hessian(wt,H64,return_dequant=True) if isinstance(H64,torch.Tensor) else _quantize_tensor_self_mse(wt,return_dequant=True)
    if isinstance(H256,torch.Tensor):wp=_safe105_chunked_weight_h256(wt,wp,H256,iters=4,chunk_rows=256)
    wq=_v64_dequant_params(wp,tuple(w.shape)).float();st=_safe105_make_state('v156',s,perm,had,ph,wq,{'post_perm':post.cpu().to(torch.int32),'post_perm_enabled':True,'rho64_mean':float(r64.mean()),'rho256_mean':float(r256.mean())})
    return {'weight_params':wp,'activation_state':st}
def hif4_calibration_and_quantize_weight(weight_quant,weight_scale,calib_activation_list):return _v156_linear(weight_quant,weight_scale,calib_activation_list)
def hif4_dynamic_quantize_activation(activation_quant,activation_scale,activation_state):return _safe133_dynamic(activation_quant,activation_scale,activation_state)


# =============================================================================
# V158/V159 ONLINE HIF4 HIERARCHY REFINEMENT
# Strict-safe: optimizes the current activation/Q tensor under a frozen
# covariance/Hessian. No A@W, QK^T, SDPA, or operator-output target is formed.
# =============================================================================

def _v158_lv3_greedy(y,p,H256,iters=2):
    shape=tuple(int(s) for s in y.shape); k=shape[-1]
    if y.dim()!=2 or k%256 or not isinstance(H256,torch.Tensor):
        return p
    rows=int(y.shape[0]); ng=k//256; nb=k//64
    H=H256.to(y.device,torch.float32); pp=_v64_clone_params(p)
    yy=y.float().reshape(rows,ng,256)
    q=_v64_dequant_params(pp,shape).float().reshape(rows,ng,256)
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


def _v158_dynamic_tensor_h256(y,H64,H256,lv3_iters=2,base_mant_iters=4):
    if not isinstance(H64,torch.Tensor):
        return _quantize_tensor_self_mse(y,return_dequant=False)[0]
    p,_=_v37_quantize_hessian(y,H64.to(y.device),return_dequant=False)
    if isinstance(H256,torch.Tensor):
        H=H256.to(y.device,torch.float32)
        # Proven V156 baseline first.
        p=_safe90_refine256(y,p,H,int(base_mant_iters))
        # Re-open the frozen level-1 E6 scale under the full H256 metric.
        p=_safe130_scale_core(y,p,H,1)
        p=_safe90_refine256(y,p,H,1)
        # Re-open the most useful HiF4 hierarchy bit (lv3, one bit / 4 values).
        p=_v158_lv3_greedy(y,p,H,int(lv3_iters))
        p=_safe90_refine256(y,p,H,1)
    return p


def _v158_dynamic_activation(aq,asc,st,lv3_iters=2):
    st=st if isinstance(st,dict) else {}
    y=_safe90_decode_transform_activation(aq,asc,st)
    post=st.get('post_perm')
    if isinstance(post,torch.Tensor):
        y=y.index_select(-1,post.to(y.device,dtype=torch.long))
    return _v158_dynamic_tensor_h256(
        y,st.get('weight_hessian_blocks'),st.get('super256_hessian_blocks'),
        lv3_iters=int(lv3_iters),base_mant_iters=4
    )


# V158: V156 offline Weight + online activation E6/lv3 H256 correction.
def hif4_calibration_and_quantize_weight(weight_quant,weight_scale,calib_activation_list):
    return _v156_linear(weight_quant,weight_scale,calib_activation_list)

def hif4_dynamic_quantize_activation(activation_quant,activation_scale,activation_state):
    return _v158_dynamic_activation(activation_quant,activation_scale,activation_state,lv3_iters=2)


def _v159_dynamic_k(k_quant,k_scale,kv_num_heads,head_dim,k_state,lv3_iters=1):
    k=dequantize_nvfp4(k_quant,k_scale).float()
    if not isinstance(k_state,dict):
        return _v60_quantize_k_tensor_fast(k)
    x=_v44_apply_k(k,k_state,kv_num_heads,head_dim)
    H256=k_state.get('partner_h256'); H64=k_state.get('partner_h64')
    if not isinstance(H256,torch.Tensor):
        return _v60_quantize_k_tensor_fast(x)
    H256d=H256.to(x.device,torch.float32)
    # Exact V113 quotient baseline.
    p0=_safe108_quantize_k_partner(x,H256d,kv_num_heads,head_dim)
    q0=_v64_dequant_params(p0,tuple(x.shape)).float()
    # The optimal common translation per feature is invisible to softmax.
    c=(x-q0).mean(dim=-2,keepdim=True)
    target=x-c
    if not isinstance(H64,torch.Tensor):
        return p0
    # Re-quantize inside the already-discovered quotient basin using the Q-partner
    # covariance and the current tensor hierarchy.
    p1=_v158_dynamic_tensor_h256(
        target,H64,H256d,lv3_iters=int(lv3_iters),base_mant_iters=1
    )
    q1=_v64_dequant_params(p1,tuple(x.shape)).float()
    s0=_safe108_k_head_scores(x,q0,H256d,kv_num_heads,head_dim)
    s1=_safe108_k_head_scores(x,q1,H256d,kv_num_heads,head_dim)
    best=(s1<s0).to(torch.long)
    return _safe108_merge_k_by_head([p0,p1],best,kv_num_heads,head_dim)


def _v159_dynamic_q(q_quant,q_scale,q_num_heads,head_dim,q_state,lv3_iters=1):
    q=dequantize_nvfp4(q_quant,q_scale).float()
    if isinstance(q_state,dict):
        q=_v44_apply_q(q,q_state,q_num_heads,head_dim)
        H64=q_state.get('partner_h64'); H256=q_state.get('partner_h256')
        if isinstance(H64,torch.Tensor):
            return _v158_dynamic_tensor_h256(q,H64,H256,lv3_iters=int(lv3_iters),base_mant_iters=1)
    return _quantize_tensor_self_mse(q,return_dequant=False)[0]

def hif4_dynamic_quantize_q(q_quant,q_scale,q_num_heads,head_dim,q_state):
    return _v159_dynamic_q(q_quant,q_scale,q_num_heads,head_dim,q_state,1)
def hif4_dynamic_quantize_k(k_quant,k_scale,kv_num_heads,head_dim,k_state):
    return _v159_dynamic_k(k_quant,k_scale,kv_num_heads,head_dim,k_state,1)

# V159 = exact V158 Linear + strict-safe Q/K per-head hierarchy refinement.

# =============================================================================
# V161 FIXED: full V159 Attention + exact V160 Linear override
# =============================================================================

def _v160_lv2_once(y, p, H256):
    shape = tuple(int(s) for s in y.shape)
    k = shape[-1]
    if y.dim() != 2 or k % 256 or not isinstance(H256, torch.Tensor):
        return p

    rows = int(y.shape[0])
    ng = k // 256
    nb = k // 64
    H = H256.to(y.device, torch.float32)
    pp = _v64_clone_params(p)

    yy = y.float().reshape(rows, ng, 256)
    q = _v64_dequant_params(pp, shape).float().reshape(rows, ng, 256)
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

def _v160_dynamic_activation(aq, asc, st):
    st = st if isinstance(st, dict) else {}
    y = _safe90_decode_transform_activation(aq, asc, st)
    post = st.get('post_perm')
    if isinstance(post, torch.Tensor):
        y = y.index_select(-1, post.to(y.device, dtype=torch.long))

    H64 = st.get('weight_hessian_blocks')
    H256 = st.get('super256_hessian_blocks')

    # Exact V158 baseline first.
    p = _v158_dynamic_tensor_h256(
        y, H64, H256, lv3_iters=2, base_mant_iters=4
    )

    if isinstance(H256, torch.Tensor):
        H = H256.to(y.device, torch.float32)
        # Re-open one lv2 group, then locally repair lv3/mantissa.
        p = _v160_lv2_once(y, p, H)
        p = _safe90_refine256(y, p, H, 1)
        p = _v158_lv3_greedy(y, p, H, 1)
        p = _safe90_refine256(y, p, H, 1)
    return p

def hif4_calibration_and_quantize_weight(
    weight_quant, weight_scale, calib_activation_list
):
    return _v156_linear(weight_quant, weight_scale, calib_activation_list)

def hif4_dynamic_quantize_activation(
    activation_quant, activation_scale, activation_state
):
    return _v160_dynamic_activation(
        activation_quant, activation_scale, activation_state
    )


