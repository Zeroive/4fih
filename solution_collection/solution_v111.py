from __future__ import annotations

import math
from typing import Optional, Tuple

import torch

# =============================================================================
# 全局常量与缓存
# =============================================================================
_HIF4_BLOCK = 64
_NVFP4_BLOCK = 16
_SEARCH_CHUNK_BLOCKS = 16384
_E6_ANCHOR_OFFSETS = (-1, 0, 1, 2, 3, 4)
# Per-64-block alternating least-squares refinement.  Each round derives the
# continuous optimum from the block's current HiF4 codes, projects it back to
# E6M2, requantizes lv2/lv3/mantissa, and keeps it only when the objective drops.
_LSQ_SCALE_ITERS = 3
# Latency-bounded K quotient: 6 candidates instead of the previous 42.
_K_QUOTIENT_TOTAL_ROUNDS = 2
_K_RELAX_GAMMAS = (1.0, 2.0)
_K_MEDIAN_EXTRA_ROUNDS = 1
_ATTN_HEAD_ROTATION_CANDIDATES = (-1, 0, 1)  # identity, H64, signed H64

_E6_TABLE_CACHE: dict[str, torch.Tensor] = {}

_V31_STATE_VERSION = "v31_calib_smooth_h64_safe"
_V42_VERSION = "v42_attention_rotcov"
_V105_VERSION = 'v105_rulesafe_weight_cov256_chunk1'
_V111_ATTN_HEAD_VERSION = "v111_attention_per_head"


# =============================================================================
# 核心量化与反量化底层算子
# =============================================================================
def dequantize_nvfp4(
        quant_float: torch.Tensor,
        scale_float: torch.Tensor,
        blk_size: int = _NVFP4_BLOCK,
) -> torch.Tensor:
    c = int(quant_float.shape[-1])
    if c % blk_size != 0:
        raise ValueError(f"Last dimension {c} is not divisible by NVFP4 block size {blk_size}")
    x = quant_float.unflatten(-1, (-1, blk_size))
    result = x * scale_float.unsqueeze(-1)
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
    t = target.clamp(min=2.0 ** -48, max=49152.0)
    hi = torch.searchsorted(table, t).clamp(0, table.numel() - 1)
    lo = (hi - 1).clamp(0, table.numel() - 1)
    vlo = table[lo]
    vhi = table[hi]
    choose_hi = (vhi - t).abs() < (t - vlo).abs()
    return torch.where(choose_hi, hi, lo)


def _fixed_scale_self_sse(abs_x: torch.Tensor, sf: torch.Tensor) -> torch.Tensor:
    errs = []
    for mult in (1.0, 2.0, 4.0):
        denom = sf * mult
        mant = torch.round((abs_x / denom) * 4.0) * 0.25
        mant = mant.clamp_(0.0, 1.75)
        errs.append((mant * denom - abs_x).square().sum(dim=-1, keepdim=True))
    e1, e2, e4 = errs
    err_l2_1 = torch.minimum(e1, e2).sum(dim=-2, keepdim=True)
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


def _least_squares_scale_self(
        x: torch.Tensor,
        sign: torch.Tensor,
        mant: torch.Tensor,
        l2: torch.Tensor,
        l3: torch.Tensor,
) -> torch.Tensor:
    """Continuous MSE-optimal base scale for fixed per-element HiF4 codes."""
    code = sign * mant * l2 * l3
    numerator = (x * code).sum(dim=(-1, -2, -3), keepdim=True)
    denominator = code.square().sum(dim=(-1, -2, -3), keepdim=True)
    return torch.where(
        denominator > 0.0,
        numerator / denominator.clamp_min(2.0 ** -48),
        torch.zeros_like(numerator),
    ).clamp_min(2.0 ** -48)


def _least_squares_scale_hessian(
        x: torch.Tensor,
        sign: torch.Tensor,
        mant: torch.Tensor,
        l2: torch.Tensor,
        l3: torch.Tensor,
        hessian: torch.Tensor,
) -> torch.Tensor:
    """Hessian-weighted continuous optimum for fixed per-element HiF4 codes."""
    code = (sign * mant * l2 * l3).reshape(*x.shape[:2], 64)
    target = x.reshape(*x.shape[:2], 64)
    numerator = torch.einsum('rbi,bij,rbj->rb', code, hessian, target)
    denominator = torch.einsum('rbi,bij,rbj->rb', code, hessian, code)
    scale = torch.where(
        denominator > 0.0,
        numerator / denominator.clamp_min(2.0 ** -48),
        torch.zeros_like(numerator),
    )
    return scale.clamp_min(2.0 ** -48).view(*x.shape[:2], 1, 1, 1)


def _quantize_tensor_self_mse(
        x: torch.Tensor,
        *,
        return_dequant: bool = False,
) -> Tuple[dict, Optional[torch.Tensor]]:
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

        # Alternate between the discrete HiF4 codes and their exact continuous
        # least-squares scale.  E6M2 projection preserves the output contract;
        # retaining only strict improvements prevents refinement regressions.
        for _ in range(_LSQ_SCALE_ITERS):
            sf_ls = _least_squares_scale_self(xb, sign, mant, l2, l3)
            idx = _nearest_e6m2_index(sf_ls.reshape(-1), table)
            sf_try = table[idx].view(bsz, 1, 1, 1)
            sign_try, mant_try, l2_try, l3_try = _materialize_fixed_scale_self(
                xb, sf_try
            )
            dq_try = sign_try * mant_try * l2_try * l3_try * sf_try
            err_try = (dq_try - xb).square().sum(dim=(-1, -2, -3))
            better = err_try < best_err
            mask = better.view(bsz, 1, 1, 1)
            best_err = torch.where(better, err_try, best_err)
            sf = torch.where(mask, sf_try, sf)
            sign = torch.where(mask, sign_try, sign)
            mant = torch.where(mask, mant_try, mant)
            l2 = torch.where(mask, l2_try, l2)
            l3 = torch.where(mask, l3_try, l3)

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
        dq = dq_out.reshape(shape).to(torch.bfloat16).float()
    return params, dq


# =============================================================================
# K 向量商空间优化算法 (Softmax 零空间不变性)
# =============================================================================
def _merge_k_candidates_per_feature_block(
        x: torch.Tensor,
        params_list: list[dict],
        dq_list: list[torch.Tensor],
) -> dict:
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
        e = e - e.mean(dim=1, keepdim=True)
        scores.append(e.square().sum(dim=(1, 3)))
    score_stack = torch.stack(scores, dim=0)
    best = score_stack.argmin(dim=0)

    out: dict[str, torch.Tensor] = {}
    for name in params_list[0]:
        base = params_list[0][name]
        tail = tuple(int(v) for v in base.shape[len(batch_prefix) + 2:])
        y = base.reshape(groups, seq, nblocks, *tail).clone()
        for ci in range(1, len(params_list)):
            cand = params_list[ci][name].reshape(groups, seq, nblocks, *tail)
            mask = (best == ci).reshape(groups, 1, nblocks, *([1] * len(tail)))
            y = torch.where(mask, cand, y)
        out[name] = y.reshape(base.shape)
    return out


def _select_best_k_dq_per_feature_block(
        x: torch.Tensor,
        dq_list: list[torch.Tensor],
) -> torch.Tensor:
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


def _quantize_k_softmax_quotient_tensor(x: torch.Tensor) -> dict:
    """Search the softmax-invariant K quotient in each leading group separately."""
    x = x.float()
    if x.dim() < 2 or int(x.shape[-2]) <= 1:
        return _quantize_tensor_self_mse(x, return_dequant=False)[0]

    params_list: list[dict] = []
    dq_list: list[torch.Tensor] = []

    p, q = _quantize_tensor_self_mse(x, return_dequant=True)
    params_list.append(p)
    dq_list.append(q)

    q_best = q
    c_prev = torch.zeros_like(x.mean(dim=-2, keepdim=True))

    for _ in range(1, _K_QUOTIENT_TOTAL_ROUNDS):
        c_star = (x - q_best).mean(dim=-2, keepdim=True)
        delta = c_star - c_prev
        for gamma in _K_RELAX_GAMMAS:
            c_try = c_prev + float(gamma) * delta
            target = x - c_try
            p, q = _quantize_tensor_self_mse(target, return_dequant=True)
            params_list.append(p)
            dq_list.append(q)
        q_best = _select_best_k_dq_per_feature_block(x, dq_list)
        c_prev = c_star

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


def _quantize_k_softmax_quotient(k_quant: torch.Tensor, k_scale: torch.Tensor) -> dict:
    x = dequantize_nvfp4(k_quant, k_scale).float()
    return _quantize_k_softmax_quotient_tensor(x)


# =============================================================================
# 变换算子 (FWHT, Smooth, 旋转矩阵)
# =============================================================================
def _fwht64_v31(x):
    c = x.shape[-1]
    if c % 64: return x.float()
    orig = x.shape
    y = x.float().reshape(-1, c // 64, 64).clone()
    h = 1
    while h < 64:
        z = y.reshape(*y.shape[:-1], -1, 2 * h)
        a = z[..., :h].clone()
        b = z[..., h:2 * h].clone()
        z[..., :h] = a + b
        z[..., h:2 * h] = a - b
        y = z.reshape(-1, c // 64, 64)
        h *= 2
    return (y * 0.125).reshape(orig)


def _v31_smooth(amax, wmax, beta):
    if beta <= 0: return torch.ones_like(wmax)
    ls = float(beta) * (torch.log(wmax.clamp_min(2 ** -24)) - torch.log(amax.clamp_min(2 ** -24)))
    ls -= ls.median()
    return torch.exp(ls).clamp_(2 ** -6, 2 ** 6)


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


def _attention_head_view(x: torch.Tensor, num_heads: int, head_dim: int) -> torch.Tensor:
    expected = int(num_heads) * int(head_dim)
    if x.dim() < 2 or int(x.shape[-1]) != expected:
        raise ValueError(
            f"attention tensor must end in num_heads * head_dim = {expected}, "
            f"got shape {tuple(x.shape)}"
        )
    if head_dim % _HIF4_BLOCK != 0:
        raise ValueError(
            f"per-head HiF4 requires head_dim divisible by {_HIF4_BLOCK}, got {head_dim}"
        )
    return x.float().reshape(*x.shape[:-1], int(num_heads), int(head_dim))


def _apply_per_head_rotation(
        x: torch.Tensor,
        num_heads: int,
        head_dim: int,
        patterns,
) -> torch.Tensor:
    """Apply an independently selected block-Hadamard transform to every head."""
    y = _attention_head_view(x, num_heads, head_dim)
    if not isinstance(patterns, torch.Tensor) or patterns.numel() != num_heads:
        return y
    # Group heads by the three possible patterns. This avoids one CUDA launch and
    # one synchronising Tensor.item() call per head.
    patterns_cpu = patterns.reshape(num_heads).to(device="cpu", dtype=torch.int64)
    out = torch.empty_like(y)
    for pattern in _ATTN_HEAD_ROTATION_CANDIDATES:
        index_cpu = torch.nonzero(patterns_cpu == pattern, as_tuple=False).flatten()
        if int(index_cpu.numel()) == 0:
            continue
        index = index_cpu.to(y.device)
        selected = y.index_select(-2, index)
        transformed = _v35_rotate64(selected, int(pattern))
        out.index_copy_(-2, index, transformed)
    return out


def _flatten_per_head_params(
        params: dict,
        input_shape: tuple[int, ...],
        num_heads: int,
        head_dim: int,
) -> dict:
    """Restore per-head quantizer output to the public flattened-head layout."""
    prefix = input_shape[:-1]
    head_blocks = head_dim // _HIF4_BLOCK
    out = {}
    for name, value in params.items():
        tail = tuple(int(v) for v in value.shape[len(prefix) + 2:])
        out[name] = value.reshape(*prefix, num_heads * head_blocks, *tail)
    return out


def _quantize_attention_per_head(
        x: torch.Tensor,
        num_heads: int,
        head_dim: int,
        *,
        return_dequant: bool = False,
):
    """Run scale search with head_dim as the quantizer's feature dimension."""
    shape = tuple(int(v) for v in x.shape)
    xh = _attention_head_view(x, num_heads, head_dim)
    params, dq = _quantize_tensor_self_mse(xh, return_dequant=return_dequant)
    params = _flatten_per_head_params(params, shape, num_heads, head_dim)
    if dq is not None:
        dq = dq.reshape(shape)
    return params, dq


def _qk_logit_sse(
        q: torch.Tensor,
        k: torch.Tensor,
        qdq: torch.Tensor,
        kdq: torch.Tensor,
) -> torch.Tensor:
    """Exact per-Q-head SSE between QK^T and dequantized HiF4 QK^T.

    The Gram formulation is algebraically identical to materialising both
    token-by-token logit matrices, but its temporary storage is O(head_dim^2)
    instead of O(seq_len^2).
    """
    q_gram = torch.einsum("tgi,tgj->gij", q, q)
    k_gram = torch.einsum("ti,tj->ij", k, k)
    qdq_gram = torch.einsum("tgi,tgj->gij", qdq, qdq)
    kdq_gram = torch.einsum("ti,tj->ij", kdq, kdq)
    q_cross = torch.einsum("tgi,tgj->gij", qdq, q)
    k_cross = torch.einsum("ti,tj->ij", kdq, k)

    ref_sq = torch.einsum("gij,ij->g", q_gram, k_gram)
    dq_sq = torch.einsum("gij,ij->g", qdq_gram, kdq_gram)
    cross = torch.einsum("gij,ij->g", q_cross, k_cross)
    return (ref_sq + dq_sq - 2.0 * cross).clamp_min_(0.0)


def _quantize_k_per_head(
        x: torch.Tensor,
        num_heads: int,
        head_dim: int,
) -> dict:
    """Run quotient search independently for every K head."""
    shape = tuple(int(v) for v in x.shape)
    xh = _attention_head_view(x, num_heads, head_dim)
    batch_prefix = shape[:-2]
    # The public interface is 2-D. Head-major layout makes every head a leading
    # group while evaluating all heads in the same quotient-search kernels.
    x_head_major = xh.transpose(-3, -2)
    params = _quantize_k_softmax_quotient_tensor(x_head_major)

    head_axis = len(batch_prefix)
    out = {}
    for name, value in params.items():
        value = value.movedim(head_axis, head_axis + 1)
        tail = tuple(int(v) for v in value.shape[len(batch_prefix) + 3:])
        out[name] = value.reshape(
            *batch_prefix,
            shape[-2],
            num_heads * (head_dim // _HIF4_BLOCK),
            *tail,
        )
    return out


def _v42_apply_head_matrix(x: torch.Tensor, num_heads: int, head_dim: int,
                           mats: torch.Tensor) -> torch.Tensor:
    if x.shape[-1] != num_heads * head_dim:
        return x.float()
    if not isinstance(mats, torch.Tensor) or tuple(mats.shape) != (num_heads, head_dim, head_dim):
        return x.float()
    y = x.float().reshape(x.shape[0], num_heads, head_dim)
    return torch.einsum("lhd,hde->lhe", y, mats.to(y.device, dtype=torch.float32)).reshape_as(x)


def _v40_balanced_perm(key: torch.Tensor) -> torch.Tensor:
    c = int(key.numel())
    if c % 64 != 0 or c < 128:
        return torch.argsort(torch.log2(key.clamp_min(2.0 ** -40)), stable=True)
    nb = c // 64
    order = torch.argsort(key, descending=True, stable=True)
    grid = torch.empty((nb, 64), dtype=torch.long, device=key.device)
    t = torch.arange(c, device=key.device)
    block = t.remainder(nb)
    slot = torch.div(t, nb, rounding_mode='floor')
    grid[block, slot] = order
    return grid.reshape(-1)


# =============================================================================
# 海森矩阵与 OMCD 优化辅助函数
# =============================================================================
def _v64_clone_params(params):
    return {k: v.clone() for k, v in params.items()}


def _v64_dequant_params(params, shape):
    return (params['sign'] * params['mant'] * params['scale_lv2'] * params['scale_lv3'] * params[
        'scale_factor']).reshape(shape).float()


def _v37_quantize_hessian(x: torch.Tensor, hblocks: torch.Tensor, *, return_dequant=False):
    shape = tuple(int(s) for s in x.shape)
    c = shape[-1]
    if c % 64 != 0 or hblocks.dim() != 3 or tuple(hblocks.shape[1:]) != (64, 64) or hblocks.shape[0] != c // 64:
        return _quantize_tensor_self_mse(x, return_dequant=return_dequant)
    x = x.float()
    nb = c // 64
    rows = x.numel() // c
    blocks = x.reshape(rows, nb, 8, 2, 4)
    table = _build_e6m2_table(x.device)
    last = int(table.numel() - 1)
    sf_out = torch.empty((rows, nb, 1, 1, 1), dtype=torch.bfloat16, device=x.device)
    l2_out = torch.empty((rows, nb, 8, 1, 1), dtype=torch.bfloat16, device=x.device)
    l3_out = torch.empty((rows, nb, 8, 2, 1), dtype=torch.bfloat16, device=x.device)
    sg_out = torch.empty((rows, nb, 8, 2, 4), dtype=torch.bfloat16, device=x.device)
    ma_out = torch.empty((rows, nb, 8, 2, 4), dtype=torch.bfloat16, device=x.device)
    dq_out = torch.empty_like(blocks) if return_dequant else None
    H = hblocks.to(x.device, torch.float32)
    chunk_rows = max(1, 4096 // max(nb, 1))
    for rs in range(0, rows, chunk_rows):
        re = min(rows, rs + chunk_rows)
        z = blocks[rs:re]
        nr = re - rs
        anchor = _nearest_e6m2_index(z.abs().amax((2, 3, 4)) / 7.0, table)
        best = torch.full((nr, nb), float('inf'), dtype=torch.float32, device=x.device)
        best_pack = None
        best_q = None
        for off in _E6_ANCHOR_OFFSETS:
            idx = (anchor + off).clamp(0, last)
            sf = table[idx].view(nr, nb, 1, 1, 1)
            sg, ma, l2, l3 = _materialize_fixed_scale_self(z, sf)
            q = (sg * ma * l2 * l3 * sf).reshape(nr, nb, 64)
            e = q - z.reshape(nr, nb, 64)
            err = torch.einsum('rbi,bij,rbj->rb', e, H, e)
            better = err < best
            if best_pack is None:
                best = err
                best_pack = [sf.clone(), l2.clone(), l3.clone(), sg.clone(), ma.clone()]
                best_q = q.clone()
            else:
                best = torch.where(better, err, best)
                vals = (sf, l2, l3, sg, ma)
                for jj, v in enumerate(vals):
                    mask = better.view(nr, nb, *([1] * (v.dim() - 2)))
                    best_pack[jj] = torch.where(mask, v, best_pack[jj])
                best_q = torch.where(better[:, :, None], q, best_q)

        # The Linear weight/activation paths use a Hessian-weighted objective,
        # so refine their base scales with the corresponding generalized
        # least-squares closed form rather than unweighted elementwise MSE.
        for _ in range(_LSQ_SCALE_ITERS):
            sf, l2, l3, sg, ma = best_pack
            sf_ls = _least_squares_scale_hessian(z, sg, ma, l2, l3, H)
            idx = _nearest_e6m2_index(sf_ls.reshape(-1), table)
            sf_try = table[idx].view(nr, nb, 1, 1, 1)
            sg_try, ma_try, l2_try, l3_try = _materialize_fixed_scale_self(
                z, sf_try
            )
            q_try = (sg_try * ma_try * l2_try * l3_try * sf_try).reshape(
                nr, nb, 64
            )
            e_try = q_try - z.reshape(nr, nb, 64)
            err_try = torch.einsum('rbi,bij,rbj->rb', e_try, H, e_try)
            better = err_try < best
            best = torch.where(better, err_try, best)
            vals = (sf_try, l2_try, l3_try, sg_try, ma_try)
            for jj, v in enumerate(vals):
                mask = better.view(nr, nb, *([1] * (v.dim() - 2)))
                best_pack[jj] = torch.where(mask, v, best_pack[jj])
            best_q = torch.where(better[:, :, None], q_try, best_q)
        sf, l2, l3, sg, ma = best_pack
        sf_out[rs:re] = sf.to(torch.bfloat16)
        l2_out[rs:re] = l2.to(torch.bfloat16)
        l3_out[rs:re] = l3.to(torch.bfloat16)
        sg_out[rs:re] = sg.to(torch.bfloat16)
        ma_out[rs:re] = ma.to(torch.bfloat16)
        if dq_out is not None: dq_out[rs:re] = best_q.reshape(nr, nb, 8, 2, 4)
    prefix = shape[:-1]
    params = {'scale_factor': sf_out.reshape(*prefix, nb, 1, 1, 1), 'scale_lv2': l2_out.reshape(*prefix, nb, 8, 1, 1),
              'scale_lv3': l3_out.reshape(*prefix, nb, 8, 2, 1), 'sign': sg_out.reshape(*prefix, nb, 8, 2, 4),
              'mant': ma_out.reshape(*prefix, nb, 8, 2, 4)}
    dq = dq_out.reshape(shape).to(torch.bfloat16).float() if dq_out is not None else None
    return params, dq


# =============================================================================
# 安全几何变换与校准状态生成 (V90/V105)
# =============================================================================
def _rule_safe_decode_acts(calib_activation_list, k, device):
    out = []
    for pair in calib_activation_list:
        if not isinstance(pair, (list, tuple)) or len(pair) != 2:
            continue
        a = dequantize_nvfp4(pair[0], pair[1]).float().to(device).reshape(-1, k)
        if a.numel() and a.shape[-1] == k:
            out.append(a)
    return out


def _safe90_identity_perm(c, device):
    return torch.arange(c, dtype=torch.long, device=device)


def _safe90_blockvec(v):
    return v.reshape(-1, 1).expand(-1, 64).reshape(-1)


def _safe90_geometry(w, acts, *, use_perm=True, use_had=True, use_phase=True):
    k = int(w.shape[-1])
    if not acts or k % 64 != 0:
        return (torch.ones(k, device=w.device),
                _safe90_identity_perm(k, w.device),
                False,
                torch.ones(k // 64, device=w.device))

    # First take a per-channel token maximum inside each calibration sample,
    # then average those maxima across samples to reduce single-sample outliers.
    amax = torch.stack([a.abs().amax(dim=0) for a in acts], dim=0).mean(dim=0)
    wmax = w.abs().amax(0)
    smooth = _v31_smooth(amax, wmax, 0.50)

    aeff = amax * smooth
    weff = wmax / smooth.clamp_min(2.0 ** -24)
    key = torch.maximum(aeff, weff)
    perm = _v40_balanced_perm(key) if use_perm else _safe90_identity_perm(k, w.device)

    had = bool(use_had)
    phases = torch.ones(k // 64, dtype=torch.float32, device=w.device)
    if use_phase:
        wt = (w.float() / smooth).index_select(-1, perm)
        if had:
            wt = _fwht64_v31(wt)
        wb = wt.abs().reshape(-1, k // 64, 64).amax(dim=(0, 2))

        ab = torch.zeros(k // 64, dtype=torch.float32, device=w.device)
        for a in acts:
            at = (a.float() * smooth).index_select(-1, perm)
            if had:
                at = _fwht64_v31(at)
            ab = torch.maximum(ab, at.abs().reshape(-1, k // 64, 64).amax(dim=(0, 2)))

        phases = torch.sqrt(wb.clamp_min(2.0 ** -24) / ab.clamp_min(2.0 ** -24))
        phases = phases.clamp(0.50, 2.00)
        phases = phases / torch.exp(torch.log(phases).median())
        phases = phases.clamp(0.50, 2.00)

    return smooth, perm, had, phases


def _safe90_apply(x, smooth, perm, phases, had, weight_side=False):
    y = x.float()
    y = y / smooth if weight_side else y * smooth
    y = y.index_select(-1, perm.to(y.device, dtype=torch.long))
    pv = _safe90_blockvec(phases.to(y.device, torch.float32))
    y = y / pv if weight_side else y * pv
    if had:
        y = _fwht64_v31(y)
    return y


def _safe90_hessian64(wq):
    if wq.dim() != 2 or wq.shape[-1] % 64 != 0: return None
    m, k = map(int, wq.shape)
    nb = k // 64
    z = wq.float().reshape(m, nb, 64)
    H = torch.einsum('mbi,mbj->bij', z, z)
    sc = H.diagonal(dim1=-2, dim2=-1).mean(-1).abs().clamp_min(1e-12)
    return H / sc[:, None, None]


def _safe90_hessian256(wq):
    if wq.dim() != 2 or wq.shape[-1] % 256 != 0: return None
    m, k = map(int, wq.shape)
    ng = k // 256
    z = wq.float().reshape(m, ng, 256)
    H = torch.einsum('mgi,mgj->gij', z, z)
    sc = H.diagonal(dim1=-2, dim2=-1).mean(-1).abs().clamp_min(1e-12)
    return H / sc[:, None, None]


def _safe90_refine256(y, p, H, iters=4):
    shape = tuple(int(s) for s in y.shape)
    k = shape[-1]
    if k % 256 != 0 or not isinstance(H, torch.Tensor): return p
    rows = y.numel() // k
    ng = k // 256
    nb = k // 64
    if tuple(H.shape) != (ng, 256, 256): return p

    pp = _v64_clone_params(p)
    sf = pp['scale_factor'].float().reshape(rows, nb, 1, 1, 1)
    l2 = pp['scale_lv2'].float().reshape(rows, nb, 8, 1, 1)
    l3 = pp['scale_lv3'].float().reshape(rows, nb, 8, 2, 1)
    eff = (sf * l2 * l3).expand(rows, nb, 8, 2, 4).reshape(rows, ng, 256)
    u = (pp['sign'].float() * pp['mant'].float()).reshape(rows, ng, 256)
    yy = y.float().reshape(rows, ng, 256)
    H = H.to(y.device, torch.float32)

    q = u * eff
    e = q - yy
    g = torch.einsum('gij,rgi->rgj', H, e)
    diag = H.diagonal(dim1=-2, dim2=-1).unsqueeze(0)
    step = 0.25 * eff
    step2 = step.square() * diag
    gidx = torch.arange(ng, device=y.device).view(1, ng).expand(rows, ng)

    for _ in range(int(iters)):
        any_good = False
        for sub in range(4):
            lo = sub * 64
            hi = lo + 64
            base = 2.0 * step[:, :, lo:hi] * g[:, :, lo:hi]
            dp = base + step2[:, :, lo:hi]
            dm = -base + step2[:, :, lo:hi]
            us = u[:, :, lo:hi]
            dp.masked_fill_(us >= 1.75 - 1e-6, float('inf'))
            dm.masked_fill_(us <= -1.75 + 1e-6, float('inf'))
            choose = dp < dm
            move = torch.minimum(dp, dm)
            best, j0 = move.min(dim=2)
            good = best < -1e-8
            if not bool(good.any().item()): continue
            any_good = True
            plus = choose.gather(2, j0.unsqueeze(-1)).squeeze(-1)
            direction = torch.where(plus, torch.ones_like(best), -torch.ones_like(best))
            du = 0.25 * direction * good
            j = j0 + lo
            u.scatter_add_(2, j.unsqueeze(-1), du.unsqueeze(-1))
            de = du * eff.gather(2, j.unsqueeze(-1)).squeeze(-1)
            col = H[gidx, :, j]
            g.add_(col * de.unsqueeze(-1))
        if not any_good: break

    ma = u.abs().reshape(rows, nb, 64)
    sg = torch.sign(u).reshape(rows, nb, 64)
    sg = torch.where(ma == 0.0, torch.zeros_like(sg), sg)
    pp['mant'] = ma.reshape_as(pp['mant']).to(torch.bfloat16)
    pp['sign'] = sg.reshape_as(pp['sign']).to(torch.bfloat16)
    return pp


def _safe90_make_state(version, smooth, perm, had, phases, **extra):
    st = {
        'version': version,
        'rule_safe_no_AW': True,
        'beta': 0.50,
        'smooth': smooth.detach().cpu().float(),
        'perm': perm.detach().cpu().to(torch.int32),
        'hadamard64': bool(had),
        'block_phase': phases.detach().cpu().float(),
    }
    st.update(extra)
    return st


def _safe90_decode_transform_activation(activation_quant, activation_scale, st):
    a = dequantize_nvfp4(activation_quant, activation_scale).float()
    s = st.get('smooth')
    perm = st.get('perm')
    ph = st.get('block_phase')
    if not all(isinstance(z, torch.Tensor) for z in (s, perm, ph)):
        return a
    return _safe90_apply(a, s.to(a.device), perm.to(a.device), ph.to(a.device),
                         bool(st.get('hadamard64', False)), False)


def _safe99_activation_hessian64(acts_t, k, device):
    nb = k // 64
    H = torch.zeros((nb, 64, 64), dtype=torch.float32, device=device)
    count = 0
    for a in acts_t:
        z = a.float().reshape(-1, nb, 64)
        H.add_(torch.einsum('rbi,rbj->bij', z, z))
        count += int(z.shape[0])
    if count <= 0: return None
    H.div_(float(count))
    sc = H.diagonal(dim1=-2, dim2=-1).mean(-1).abs().clamp_min(1e-12)
    return H / sc[:, None, None]


def _safe99_activation_hessian256(acts_t, k, device):
    if k % 256: return None
    ng = k // 256
    H = torch.zeros((ng, 256, 256), dtype=torch.float32, device=device)
    count = 0
    for a in acts_t:
        z = a.float().reshape(-1, ng, 256)
        H.add_(torch.einsum('rgi,rgj->gij', z, z))
        count += int(z.shape[0])
    if count <= 0: return None
    H.div_(float(count))
    sc = H.diagonal(dim1=-2, dim2=-1).mean(-1).abs().clamp_min(1e-12)
    return H / sc[:, None, None]


def _safe105_slice_params(p, st, en):
    return {k: v[st:en].clone() for k, v in p.items()}


def _safe105_write_params(dst, src, st, en):
    for k in dst:
        dst[k][st:en] = src[k]
    return dst


def _safe105_chunked_weight_h256(wt, p, HA256, iters=1, chunk_rows=256):
    if not isinstance(HA256, torch.Tensor) or wt.dim() != 2 or wt.shape[-1] % 256:
        return p
    out = _v64_clone_params(p)
    m = int(wt.shape[0])
    for st in range(0, m, int(chunk_rows)):
        en = min(st + int(chunk_rows), m)
        pc = _safe105_slice_params(out, st, en)
        pc = _safe90_refine256(wt[st:en], pc, HA256, int(iters))
        _safe105_write_params(out, pc, st, en)
    return out


def _safe105_make_state(version, s, perm, had, ph, wq, extra=None):
    H64 = _safe90_hessian64(wq)
    H256 = _safe90_hessian256(wq)
    st = _safe90_make_state(version, s, perm, had, ph,
                            transform_kind='safe_v40_marginal',
                            super256_iters=4)
    if extra:
        st.update(extra)
    if isinstance(H64, torch.Tensor):
        st['weight_hessian_blocks'] = H64.detach().cpu().to(torch.bfloat16)
    if isinstance(H256, torch.Tensor):
        st['super256_hessian_blocks'] = H256.detach().cpu().to(torch.bfloat16)
    return st


def _safe105_dynamic(activation_quant, activation_scale, activation_state):
    st = activation_state if isinstance(activation_state, dict) else {}
    y = _safe90_decode_transform_activation(activation_quant, activation_scale, st)
    H64 = st.get('weight_hessian_blocks')
    H256 = st.get('super256_hessian_blocks')
    if not isinstance(H64, torch.Tensor):
        return _quantize_tensor_self_mse(y, return_dequant=False)[0]
    p, _ = _v37_quantize_hessian(y, H64, return_dequant=False)
    if isinstance(H256, torch.Tensor):
        p = _safe90_refine256(y, p, H256, int(st.get('super256_iters', 4)))
    return p


def _safe111_v(v_quant, v_scale, kv_num_heads, head_dim, v_state):
    v = dequantize_nvfp4(v_quant, v_scale).float()
    return _quantize_attention_per_head(
        v, kv_num_heads, head_dim, return_dequant=False
    )[0]


# =============================================================================
# 最终公共 API (Public API Endpoints)
# =============================================================================
def hif4_calibration_and_quantize_weight(weight_quant, weight_scale, calib_activation_list):
    w = dequantize_nvfp4(weight_quant, weight_scale).float()
    acts = _rule_safe_decode_acts(calib_activation_list, w.shape[-1], w.device)

    s, perm, had, ph = _safe90_geometry(w, acts, use_perm=True, use_had=True, use_phase=True)
    wt = _safe90_apply(w, s, perm, ph, had, True)
    acts_t = [_safe90_apply(a, s, perm, ph, had, False) for a in acts]
    HA64 = _safe99_activation_hessian64(acts_t, int(w.shape[-1]), w.device)
    HA256 = _safe99_activation_hessian256(acts_t, int(w.shape[-1]), w.device)

    wp, wq = _v37_quantize_hessian(wt, HA64, return_dequant=True) if isinstance(HA64,
                                                                                torch.Tensor) else _quantize_tensor_self_mse(
        wt, return_dequant=True)

    if isinstance(HA256, torch.Tensor):
        wp = _safe105_chunked_weight_h256(wt, wp, HA256, iters=1, chunk_rows=256)
        wq = _v64_dequant_params(wp, tuple(w.shape)).float()

    st = _safe105_make_state(_V105_VERSION, s, perm, had, ph, wq, {
        'weight_metric': 'activation_covariance_64_plus_256',
        'weight_h256_iters': 1,
    })
    return {'weight_params': wp, 'activation_state': st}


def hif4_dynamic_quantize_activation(activation_quant, activation_scale, activation_state):
    return _safe105_dynamic(activation_quant, activation_scale, activation_state)


def hif4_calibration_attention(calib_qkv_list, q_num_heads, kv_num_heads, head_dim):
    if not calib_qkv_list:
        raise ValueError("calib_qkv_list must contain at least one sample")
    if q_num_heads % kv_num_heads != 0:
        raise ValueError("q_num_heads must be divisible by kv_num_heads")
    if head_dim % _HIF4_BLOCK != 0:
        raise ValueError(
            f"per-head HiF4 requires head_dim divisible by {_HIF4_BLOCK}, got {head_dim}"
        )

    q_per_kv = q_num_heads // kv_num_heads
    first_device = calib_qkv_list[0]["q"][0].device
    nc = len(_ATTN_HEAD_ROTATION_CANDIDATES)
    qk_logit_sse = torch.zeros((kv_num_heads, nc), device=first_device)
    qk_logit_elements = torch.zeros(kv_num_heads, device=first_device)

    for sample in calib_qkv_list:
        q = dequantize_nvfp4(*sample["q"]).float().to(first_device)
        k = dequantize_nvfp4(*sample["k"]).float().to(first_device)
        qh = _attention_head_view(q, q_num_heads, head_dim)
        kh = _attention_head_view(k, kv_num_heads, head_dim)

        q_flat = qh.reshape(-1, q_num_heads, head_dim)
        k_flat = kh.reshape(-1, kv_num_heads, head_dim)

        for kv_head in range(kv_num_heads):
            q_begin = kv_head * q_per_kv
            q_end = q_begin + q_per_kv
            q_group = q_flat[:, q_begin:q_end, :]
            k_head = k_flat[:, kv_head, :]
            qk_logit_elements[kv_head] += (
                int(q_group.shape[0]) * int(k_head.shape[0]) * q_per_kv
            )
            for ci, pattern in enumerate(_ATTN_HEAD_ROTATION_CANDIDATES):
                qt = _v35_rotate64(q_group, pattern)
                kt = _v35_rotate64(k_head, pattern)
                _, qdq = _quantize_tensor_self_mse(qt, return_dequant=True)
                _, kdq = _quantize_tensor_self_mse(kt, return_dequant=True)
                qk_logit_sse[kv_head, ci] += _qk_logit_sse(
                    qt, kt, qdq, kdq
                ).sum()

    # Select the rotation that minimises calibration QK-logit MSE for each KV
    # head and all of its associated GQA query heads.
    qk_score = qk_logit_sse / qk_logit_elements[:, None].clamp_min(1.0)
    best = qk_score.argmin(dim=1)
    candidate_tensor = torch.tensor(
        _ATTN_HEAD_ROTATION_CANDIDATES, device=first_device, dtype=torch.int64
    )
    k_patterns = candidate_tensor[best]
    q_patterns = k_patterns.repeat_interleave(q_per_kv)

    common = {"version": _V111_ATTN_HEAD_VERSION, "head_dim": int(head_dim)}
    return {
        "q_state": {
            **common,
            "role": "q",
            "num_heads": int(q_num_heads),
            "rotation_per_head": q_patterns.cpu(),
        },
        "k_state": {
            **common,
            "role": "k",
            "num_heads": int(kv_num_heads),
            "rotation_per_head": k_patterns.cpu(),
        },
        "v_state": {
            **common,
            "role": "v",
            "num_heads": int(kv_num_heads),
        },
    }


def hif4_dynamic_quantize_q(q_quant, q_scale, q_num_heads, head_dim, q_state):
    if isinstance(q_state, dict) and q_state.get("version") == _V111_ATTN_HEAD_VERSION:
        q = dequantize_nvfp4(q_quant, q_scale).float()
        qh = _apply_per_head_rotation(
            q, q_num_heads, head_dim, q_state.get("rotation_per_head")
        )
        return _quantize_attention_per_head(
            qh.reshape_as(q), q_num_heads, head_dim, return_dequant=False
        )[0]

    if isinstance(q_state, dict) and q_state.get("version") == _V42_VERSION:
        q = dequantize_nvfp4(q_quant, q_scale).float()
        kind = q_state.get("transform_kind")
        if kind == "rot":
            s = q_state.get("scale")
            if isinstance(s, torch.Tensor): q = q * s.to(q.device)
            q = _v35_rotate_heads(q, q_num_heads, head_dim, int(q_state.get("rotation", 0)))
        elif kind == "cov":
            q = _v42_apply_head_matrix(q, q_num_heads, head_dim, q_state.get("matrix"))
        return _quantize_tensor_self_mse(q, return_dequant=False)[0]

    q = dequantize_nvfp4(q_quant, q_scale).float()
    if isinstance(q_state, dict) and q_state.get("enabled", False):
        s = q_state.get("scale")
        if isinstance(s, torch.Tensor) and s.numel() == q.shape[-1]: q = q * s.to(q.device)
        rot = int(q_state.get("rotation", -1))
        if rot >= 0: q = _v35_rotate_heads(q, q_num_heads, head_dim, rot)
    return _quantize_tensor_self_mse(q, return_dequant=False)[0]


def hif4_dynamic_quantize_k(k_quant, k_scale, kv_num_heads, head_dim, k_state):
    k = dequantize_nvfp4(k_quant, k_scale).float()
    if isinstance(k_state, dict) and k_state.get("version") == _V111_ATTN_HEAD_VERSION:
        kh = _apply_per_head_rotation(
            k, kv_num_heads, head_dim, k_state.get("rotation_per_head")
        )
        k = kh.reshape_as(k)
        return _quantize_k_per_head(k, kv_num_heads, head_dim)
    return _quantize_k_per_head(k, kv_num_heads, head_dim)


def hif4_dynamic_quantize_v(v_quant, v_scale, kv_num_heads, head_dim, v_state):
    return _safe111_v(v_quant, v_scale, kv_num_heads, head_dim, v_state)
