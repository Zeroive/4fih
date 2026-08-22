# Clean submission build.
# Contains only the dependency closure of the six active public APIs.
# Linear calibration uses marginal transforms and covariance statistics.

from __future__ import annotations
import math
import heapq
from typing import Optional, Tuple
import torch
_HIF4_BLOCK = 64
_NVFP4_BLOCK = 16
_SEARCH_CHUNK_BLOCKS = 16384
_E6_ANCHOR_OFFSETS = (-1, 0, 1, 2, 3, 4)
_E6_TABLE_CACHE: dict[str, torch.Tensor] = {}

# =============================================================================
# ============================= SHARED / COMMON ================================
# =============================================================================

# 功能：将 NVFP4 的量化值按 block scale 反量化回浮点张量。
# 说明：最后一维按 blk_size 分块；每块乘对应 scale，最终返回 bfloat16。
def dequantize_nvfp4(quant_float: torch.Tensor, scale_float: torch.Tensor, blk_size: int=_NVFP4_BLOCK) -> torch.Tensor:
    c = int(quant_float.shape[-1])
    if c % blk_size != 0:
        raise ValueError(f'Last dimension {c} is not divisible by NVFP4 block size {blk_size}')
    x = quant_float.unflatten(-1, (-1, blk_size))
    result = x * scale_float.unsqueeze(-1)
    return result.flatten(-2, -1).to(torch.bfloat16)

# 功能：构造并缓存当前 device 上可用的正 E6M2 scale 查找表。
# 说明：枚举 exponent/mantissa 组合，并过滤超过 49152 的非法值。
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

# 功能：为目标 scale 查找 E6M2 表中数值最近的合法档位索引。
# 说明：先 clamp 到合法范围，再比较 searchsorted 左右两个候选。
def _nearest_e6m2_index(target: torch.Tensor, table: torch.Tensor) -> torch.Tensor:
    t = target.clamp(min=2.0 ** (-48), max=49152.0)
    hi = torch.searchsorted(table, t).clamp(0, table.numel() - 1)
    lo = (hi - 1).clamp(0, table.numel() - 1)
    vlo = table[lo]
    vhi = table[hi]
    choose_hi = (vhi - t).abs() < (t - vlo).abs()
    return torch.where(choose_hi, hi, lo)

# 功能：在给定一级 E6M2 scale_factor 时，计算该 64 元素块可达到的最小 Self-MSE/SSE。
# 说明：同时枚举有效倍率 1/2/4，并隐式选择最优 lv2/lv3 组合。
def _fixed_scale_self_sse(abs_x: torch.Tensor, sf: torch.Tensor) -> torch.Tensor:
    """Exact minimum unweighted SSE for a fixed E6M2 level-1 scale."""
    errs = []
    for mult in (1.0, 2.0, 4.0):
        denom = sf * mult
        mant = torch.round(abs_x / denom * 4.0) * 0.25
        mant = mant.clamp_(0.0, 1.75)
        errs.append((mant * denom - abs_x).square().sum(dim=-1, keepdim=True))
    e1, e2, e4 = errs
    err_l2_1 = torch.minimum(e1, e2).sum(dim=-2, keepdim=True)
    err_l2_2 = torch.minimum(e2, e4).sum(dim=-2, keepdim=True)
    return torch.minimum(err_l2_1, err_l2_2).sum(dim=(-3, -2, -1))

# 功能：在固定一级 scale_factor 下，真正生成 sign、mantissa、lv2、lv3 参数。
# 说明：根据局部 SSE 选择层级倍率，并将零 mantissa 的 sign 归零。
def _materialize_fixed_scale_self(x: torch.Tensor, sf: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    abs_x = x.abs()
    mant_by_mult = []
    err_by_mult = []
    for mult in (1.0, 2.0, 4.0):
        denom = sf * mult
        mant = torch.round(abs_x / denom * 4.0) * 0.25
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
    mant = torch.where(mult == 1.0, mant_by_mult[0], torch.where(mult == 2.0, mant_by_mult[1], mant_by_mult[2]))
    sign = torch.sign(x)
    sign = torch.where(mant == 0.0, torch.zeros_like(sign), sign)
    return (sign, mant, l2, l3)

# 功能：HiF4 的基础 Self-MSE 量化器：逐 64 元素块独立最小化当前张量自身的平方误差。
# 说明：搜索邻近 E6M2 scale，再物化 hierarchy 参数；可选返回反量化结果。
def _quantize_tensor_self_mse(x: torch.Tensor, *, return_dequant: bool=False) -> Tuple[dict, Optional[torch.Tensor]]:
    """Strict per-tensor HiF4 quantizer; objective is only this tensor's SSE."""
    shape = tuple((int(s) for s in x.shape))
    if not shape:
        raise ValueError('Input must have at least one dimension')
    c = shape[-1]
    if c % _HIF4_BLOCK != 0:
        raise ValueError(f'Last dimension {c} not divisible by HiF4 block size 64')
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
        best_err = torch.full((bsz,), float('inf'), dtype=torch.float32, device=x.device)
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
    params = {'scale_factor': sf_out.reshape(*prefix, nblocks, 1, 1, 1), 'scale_lv2': l2_out.reshape(*prefix, nblocks, 8, 1, 1), 'scale_lv3': l3_out.reshape(*prefix, nblocks, 8, 2, 1), 'sign': sign_out.reshape(*prefix, nblocks, 8, 2, 4), 'mant': mant_out.reshape(*prefix, nblocks, 8, 2, 4)}
    dq = None
    if dq_out is not None:
        dq = dq_out.reshape(shape).to(torch.bfloat16).float()
    return (params, dq)

# 功能：对最后一维每 64 个元素执行归一化 Fast Walsh-Hadamard Transform。
# 说明：若维度不能被 64 整除则不变换，仅转为 float。
def _fwht64_v31(x):
    c = x.shape[-1]
    if c % 64:
        return x.float()
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

# 功能：根据 activation max 与 weight max 计算逐通道 SmoothQuant 风格缩放。
# 说明：beta 控制平衡强度，并用中位数归一化后限制在 [2^-6, 2^6]。
def _v31_smooth(amax, wmax, beta):
    if beta <= 0:
        return torch.ones_like(wmax)
    ls = float(beta) * (torch.log(wmax.clamp_min(2 ** (-24))) - torch.log(amax.clamp_min(2 ** (-24))))
    ls -= ls.median()
    return torch.exp(ls).clamp_(2 ** (-6), 2 ** 6)

# 功能：根据 pattern 生成确定性的 ±1 符号向量，用于 Hadamard 前的随机化符号旋转。
# 说明：pattern<=0 时直接返回全 1，不引入符号翻转。
def _v35_sign_vector(c, pattern, device):
    if pattern <= 0:
        return torch.ones(c, dtype=torch.float32, device=device)
    i = torch.arange(c, dtype=torch.int64, device=device)
    h = i * 1103515245 + int(pattern) * 12345
    bit = (h ^ h >> 16) & 1
    return torch.where(bit == 0, 1.0, -1.0).float()

# 功能：执行 V35 旋转：可选逐通道符号翻转，然后做 64 维 Hadamard 变换。
# 说明：pattern<0 表示关闭旋转。
def _v35_rotate64(x, pattern):
    if pattern < 0:
        return x.float()
    y = x.float()
    if pattern > 0:
        y = y * _v35_sign_vector(y.shape[-1], pattern, y.device)
    return _fwht64_v31(y)

# 功能：按 attention head 重排后，对每个 head 的 head_dim 分别应用 V35 旋转。
# 说明：只有 head_dim 能被 64 整除时才执行 Hadamard 路径。
def _v35_rotate_heads(x, num_heads, head_dim, pattern):
    if pattern < 0 or head_dim % 64 != 0:
        return x.float()
    shape = x.shape
    y = x.float().reshape(-1, num_heads, head_dim).reshape(-1, head_dim)
    y = _v35_rotate64(y, pattern)
    return y.reshape(shape)

# 功能：解码 attention calibration 样本中的 Q/K/V NVFP4 张量。
# 说明：无法解码的样本会被跳过，输出可用的浮点 (q, k, v) 列表。
# =============================================================================
# =============================== ATTN PART ===================================
# =============================================================================

def _v35_decode_calib(calib, q_num_heads, kv_num_heads, head_dim):
    out = []
    for s in calib:
        try:
            q = dequantize_nvfp4(s['q'][0], s['q'][1]).float()
            k = dequantize_nvfp4(s['k'][0], s['k'][1]).float()
            v = dequantize_nvfp4(s['v'][0], s['v'][1]).float()
            out.append((q, k, v))
        except Exception:
            pass
    return out

# 功能：根据校准数据的 Q/K 通道极值计算成对缩放，使 Q 乘 scale、K 除 scale。
# 说明：在不改变理论 QK 点积的前提下平衡两侧动态范围。
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
        z = float(beta) * (torch.log(kmax.clamp_min(2.0 ** (-24))) - torch.log(qgrp.clamp_min(2.0 ** (-24))))
        z = z - z.median(dim=-1, keepdim=True).values
        sk = torch.exp(z).clamp_(2.0 ** (-6), 2.0 ** 6)
    sq = sk.repeat_interleave(rep, dim=0)
    return (sq.reshape(-1), sk.reshape(-1))

_V44_VERSION = 'v44_adaptive'

# 功能：应用 Q 侧在线变换：先乘 calibration 得到的逐通道 scale，再做按 head 旋转。
# 说明：与 K 侧的逆向 scale 配合，保持 QK 乘积结构。
def _apply_q_transform(q, state, q_num_heads, head_dim):
    q = q.float()
    scale = state['scale']
    q = q * scale.to(q.device)
    return _v35_rotate_heads(
        q, q_num_heads, head_dim, int(state.get('rotation', 0)),
    )


# 功能：应用 K 侧在线变换：先除以与 Q 配对的 scale，再做相同的按 head 旋转。
# 说明：该变换与 Q 侧成对设计，用于降低量化动态范围压力。
def _apply_k_transform(k, state, kv_num_heads, head_dim):
    k = k.float()
    scale = state['scale']
    k = k / scale.to(k.device)
    return _v35_rotate_heads(
        k, kv_num_heads, head_dim, int(state.get('rotation', 0)),
    )
_V60_K_GAMMAS = (1.0, 1.75, 2.5)
_V60_K_MEAN_ROUNDS = 2
_V60_K_MEDIAN_ROUNDS = 1

# 功能：将 HiF4 参数字典 sign×mant×lv2×lv3×scale_factor 重建为浮点张量。
# 说明：shape 用于恢复原始张量形状。
def _v64_dequant_params(params, shape):
    return (params['sign'] * params['mant'] * params['scale_lv2'] * params['scale_lv3'] * params['scale_factor']).reshape(shape).float()

# 功能：深拷贝 HiF4 参数字典中的所有 tensor，避免 refinement 原地修改输入参数。
def _v64_clone_params(params):
    return {k: v.clone() for k, v in params.items()}

# 功能：解码 Linear calibration activation 列表，并过滤格式或 hidden width 不合法的样本。
# 说明：返回统一 reshape 为 [-1, k] 的 float activation。
# =============================================================================
# ============================== LINEAR PART ==================================
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

# 功能：把每个 64-block 的 phase 标量扩展成逐通道向量，供 weight/activation 对称缩放使用。
def _safe90_blockvec(v):
    return v.reshape(-1, 1).expand(-1, 64).reshape(-1)

# 功能：应用 Linear 的安全几何变换：smooth、通道 permutation、block phase，以及可选 Hadamard。
# 说明：weight_side=True 时使用 activation 侧的逆变换，保证线性算子整体等价。
def _safe90_apply(x, smooth, perm, phases, had, weight_side=False):
    y = x.float()
    y = y / smooth if weight_side else y * smooth
    y = y.index_select(-1, perm.to(y.device, dtype=torch.long))
    pv = _safe90_blockvec(phases.to(y.device, torch.float32))
    y = y / pv if weight_side else y * pv
    if had:
        y = _fwht64_v31(y)
    return y




# 功能：打包并保存 Linear activation 在线变换需要的 calibration state。
# 说明：tensor 被转移到 CPU，额外元数据通过 **extra 一并写入。
def _safe90_make_state(version, smooth, perm, had, phases, **extra):
    st = {'version': version, 'rule_safe_no_AW': True, 'beta': 0.5, 'smooth': smooth.detach().cpu().float(), 'perm': perm.detach().cpu().to(torch.int32), 'hadamard64': bool(had), 'block_phase': phases.detach().cpu().float()}
    st.update(extra)
    return st

# 功能：将动态 activation 从 NVFP4 解码，并按 calibration state 重放 smooth/perm/phase/Hadamard。
# 说明：若 state 缺少必要 tensor，则只返回原始反量化 activation。
def _safe90_decode_transform_activation(activation_quant, activation_scale, st):
    a = dequantize_nvfp4(activation_quant, activation_scale).float()
    s = st.get('smooth')
    perm = st.get('perm')
    ph = st.get('block_phase')
    if not all((isinstance(z, torch.Tensor) for z in (s, perm, ph))):
        return a
    return _safe90_apply(a, s.to(a.device), perm.to(a.device), ph.to(a.device), bool(st.get('hadamard64', False)), False)





# 功能：按每个 Hessian/协方差矩阵的平均对角线值归一化，消除整体尺度对比较的影响。
# =============================================================================
# =============================== ATTN PART ===================================
# =============================================================================

def _safe108_norm_h(H):
    H = H.float()
    sc = H.diagonal(dim1=-2, dim2=-1).mean(-1).abs().clamp_min(1e-12)
    return H / sc[:, None, None]

# 功能：从校准 Q/K 构造 partner-aware 协方差：Q 的量化误差用 K covariance 衡量，K 反之。
# 说明：直接生成真实 head_dim×head_dim Hessian，不再构造 H64 初始化 state。
def _safe108_partner_covariances(decoded, qs, ks, q_num_heads, kv_num_heads, head_dim):
    if not decoded or head_dim % 64 != 0:
        return None
    rep = q_num_heads // kv_num_heads
    device = decoded[0][0].device
    HQ_kv = torch.zeros((kv_num_heads, head_dim, head_dim), device=device, dtype=torch.float32)
    HK = torch.zeros((kv_num_heads, head_dim, head_dim), device=device, dtype=torch.float32)
    nk_count = 0
    nq_count = 0
    for q, k, _ in decoded:
        qt = _apply_q_transform(q, qs, q_num_heads, head_dim).reshape(-1, q_num_heads, head_dim)
        kt = _apply_k_transform(k, ks, kv_num_heads, head_dim).reshape(-1, kv_num_heads, head_dim)
        kc = kt - kt.mean(dim=0, keepdim=True)
        HQ_kv.add_(torch.einsum('shd,she->hde', kc, kc))
        nk_count += int(kc.shape[0])
        qg = qt.reshape(qt.shape[0], kv_num_heads, rep, head_dim)
        HK.add_(torch.einsum('shrd,shre->hde', qg, qg))
        nq_count += int(qg.shape[0]) * int(rep)
    if nk_count <= 0 or nq_count <= 0:
        return None
    HQ_kv.div_(float(nk_count))
    HK.div_(float(nq_count))
    HQ_kv = _safe108_norm_h(HQ_kv)
    HK = _safe108_norm_h(HK)
    q_head_hessian = HQ_kv.repeat_interleave(rep, dim=0).contiguous()
    k_head_hessian = HK.contiguous()
    return (q_head_hessian, k_head_hessian)

# 功能：按 KV head 计算 K 量化误差在 partner head Hessian 下的二次型分数。
# 说明：误差会先沿序列维去均值，以匹配 K quotient/centering 搜索目标。
def _safe108_k_head_scores(x, dq, head_hessian, kv_num_heads, head_dim):
    seq = int(x.shape[-2])
    e = (dq.float() - x.float()).reshape(seq, kv_num_heads, head_dim)
    e = e - e.mean(dim=0, keepdim=True)
    H = head_hessian.to(x.device, torch.float32)
    return torch.einsum('shd,hde,she->h', e, H, e)

# 功能：根据每个 KV head 的最佳候选编号，从多组 HiF4 参数中逐 head 合并最终参数。
# 说明：head_dim 内的所有 64-block 共享该 head 的候选选择。
def _safe108_merge_k_by_head(params_list, best_head, kv_num_heads, head_dim):
    if len(params_list) == 1:
        return params_list[0]
    bph = head_dim // 64
    best_blocks = best_head.repeat_interleave(bph)
    out = {}
    for name in params_list[0]:
        base = params_list[0][name]
        y = base.clone()
        tail = base.dim() - 2
        for ci in range(1, len(params_list)):
            cand = params_list[ci][name]
            mask = (best_blocks == ci).reshape(1, -1, *[1] * tail)
            y = torch.where(mask, cand, y)
        out[name] = y
    return out

# 功能：为 K 构造多组 quotient/centering 候选，并使用 Q-partner covariance 指标逐 head 选优。
# 说明：候选本身仍由快速 Self-MSE kernel 量化，评分使用真实 head_dim Hessian。
def _safe108_quantize_k_partner(x, head_hessian, kv_num_heads, head_dim):
    """
    Same fast K quotient basins as V60, but candidate selection is performed with
    centered Q-covariance metric instead of plain feature SSE.
    Candidate quantization itself remains the fast self-MSE kernel.
    """
    x = x.float()
    params = []
    dqs = []
    p, q = _quantize_tensor_self_mse(x, return_dequant=True)
    params.append(p)
    dqs.append(q)

    # 功能：内部辅助：对当前所有 K 候选计算 partner Hessian 分数，并逐 KV head 取最优反量化结果。
    def current_best():
        scores = torch.stack([_safe108_k_head_scores(x, z, head_hessian, kv_num_heads, head_dim) for z in dqs], 0)
        best = scores.argmin(0)
        qs = torch.stack([z.reshape(x.shape[0], kv_num_heads, head_dim) for z in dqs], 0)
        y = qs[0].clone()
        for ci in range(1, len(dqs)):
            mask = (best == ci).reshape(1, kv_num_heads, 1)
            y = torch.where(mask, qs[ci], y)
        return (best, y.reshape_as(x))
    best, qbest = current_best()
    cprev = torch.zeros_like(x.mean(dim=-2, keepdim=True))
    for _ in range(_V60_K_MEAN_ROUNDS):
        cstar = (x - qbest).mean(dim=-2, keepdim=True)
        delta = cstar - cprev
        for g in _V60_K_GAMMAS:
            p, q = _quantize_tensor_self_mse(x - (cprev + float(g) * delta), return_dequant=True)
            params.append(p)
            dqs.append(q)
        best, qbest = current_best()
        cprev = cstar
    cprev = x.median(dim=-2, keepdim=True).values
    p, q = _quantize_tensor_self_mse(x - cprev, return_dequant=True)
    params.append(p)
    dqs.append(q)
    best, qbest = current_best()
    for _ in range(_V60_K_MEDIAN_ROUNDS):
        cstar = (x - qbest).mean(dim=-2, keepdim=True)
        delta = cstar - cprev
        for g in _V60_K_GAMMAS:
            p, q = _quantize_tensor_self_mse(x - (cprev + float(g) * delta), return_dequant=True)
            params.append(p)
            dqs.append(q)
        best, qbest = current_best()
        cprev = cstar
    return _safe108_merge_k_by_head(params, best, kv_num_heads, head_dim)

# 功能：Attention 校准入口：解码 Q/K/V 样本，生成 Q/K 平衡 scale、固定旋转及 partner Hessian。
# 说明：返回 q_state、k_state、v_state，供动态 Q/K/V 量化路径使用。
def hif4_calibration_attention(calib_qkv_list, q_num_heads, kv_num_heads, head_dim):
    decoded = _v35_decode_calib(
        calib_qkv_list, q_num_heads, kv_num_heads, head_dim,
    )
    q_scale, k_scale = _v35_qk_scale(
        decoded, q_num_heads, kv_num_heads, head_dim, beta=0.5,
    )
    common = {
        'version': _V44_VERSION,
        'enabled': True,
        'head_dim': int(head_dim),
        'transform_kind': 'rot',
        'rotation': 2,
        'beta': 0.5,
        'strict_safe_fixed': True,
    }
    q_state = {
        **common,
        'role': 'q',
        'scale': q_scale.detach().cpu().float(),
    }
    k_state = {
        **common,
        'role': 'k',
        'scale': k_scale.detach().cpu().float(),
    }
    q_head_hessian, k_head_hessian = _safe108_partner_covariances(
        decoded,
        q_state,
        k_state,
        q_num_heads,
        kv_num_heads,
        head_dim,
    )
    q_state.update({
        'partner_hessian': q_head_hessian.detach().cpu().to(torch.bfloat16),
        'partner_cov_enabled': True,
        'partner_hessian_enabled': True,
    })
    k_state.update({
        'partner_hessian': k_head_hessian.detach().cpu().to(torch.bfloat16),
        'partner_k_metric_enabled': True,
    })
    return {
        'q_state': q_state,
        'k_state': k_state,
        'v_state': {
            'enabled': False,
            'role': 'v',
            'strict_safe_fixed': True,
        },
    }

# 功能：动态量化 V：NVFP4 反量化后直接使用 HiF4 Self-MSE 量化。
# 说明：当前 V 不使用 Q/K 的 partner Hessian 或旋转状态。
def hif4_dynamic_quantize_v(v_quant, v_scale, kv_num_heads, head_dim, v_state):
    v = dequantize_nvfp4(v_quant, v_scale).float()
    return _quantize_tensor_self_mse(v, return_dequant=False)[0]

# 功能：在真实 head_dim Hessian 目标下，对每个 64-block 的一级 E6M2 scale_factor 做坐标更新。
# 说明：head_dim 可为任意 64 的倍数；利用连续最优 scale 并更新对应 head 的梯度。
def _refine_attention_scales(y, p, head_hessian, sweeps=1):
    """
    Exact coordinate update for E6M2 scale under each full head Hessian.

    With current q_b = s*c and gradient g_b = dL/dq_b / 2,
        Delta(s) = 2 (s-s0) c^T g_b + (s-s0)^2 c^T H_bb c.
    Thus the continuous optimum for fixed coefficient c is available in closed
    form; we snap around that optimum to legal E6M2 and re-round mantissas while
    keeping lv2/lv3 fixed. No output target is needed.
    """
    shape = tuple((int(s) for s in y.shape))
    k = shape[-1]
    if y.dim() != 2 or head_hessian.dim() != 3:
        return p
    head_width = int(head_hessian.shape[-1])
    if (
        head_width % 64 != 0
        or int(head_hessian.shape[-2]) != head_width
        or k % head_width != 0
    ):
        return p
    rows = int(y.shape[0])
    ng = k // head_width
    nb = k // 64
    blocks_per_head = head_width // 64
    if tuple(head_hessian.shape) != (ng, head_width, head_width):
        return p
    pp = _v64_clone_params(p)
    yy = y.float().reshape(rows, ng, head_width)
    H = head_hessian.to(y.device, torch.float32)
    table = _build_e6m2_table(y.device)
    last = int(table.numel() - 1)
    for _ in range(int(sweeps)):
        q = _v64_dequant_params(pp, shape).reshape(rows, ng, head_width)
        e = q - yy
        grad = torch.einsum('gij,rgi->rgj', H, e)
        sfv = pp['scale_factor'].float().reshape(rows, nb, 1, 1, 1)
        l2v = pp['scale_lv2'].float().reshape(rows, nb, 8, 1, 1)
        l3v = pp['scale_lv3'].float().reshape(rows, nb, 8, 2, 1)
        sgv = pp['sign'].float().reshape(rows, nb, 8, 2, 4)
        mav = pp['mant'].float().reshape(rows, nb, 8, 2, 4)
        for sub in range(blocks_per_head):
            lo = sub * 64
            hi = lo + 64
            bidx = torch.arange(ng, device=y.device) * blocks_per_head + sub
            z = yy[:, :, lo:hi]
            qcur = q[:, :, lo:hi]
            gcur = grad[:, :, lo:hi]
            Hss = H[:, lo:hi, lo:hi]
            sfcur = sfv[:, bidx, 0, 0, 0].clamp_min(2.0 ** (-48))
            c = qcur / sfcur.unsqueeze(-1)
            den = torch.einsum('rgi,gij,rgj->rg', c, Hss, c).clamp_min(1e-20)
            num = (c * gcur).sum(-1)
            sfstar = (sfcur - num / den).clamp(min=2.0 ** (-48), max=49152.0)
            curidx = _nearest_e6m2_index(sfcur, table)
            optidx = _nearest_e6m2_index(sfstar, table)
            idxs = [curidx, (curidx - 1).clamp(0, last), (curidx + 1).clamp(0, last), (optidx - 1).clamp(0, last), optidx, (optidx + 1).clamp(0, last)]
            l2 = l2v[:, bidx].reshape(rows, ng, 8, 1, 1)
            l3 = l3v[:, bidx].reshape(rows, ng, 8, 2, 1)
            za = z.reshape(rows, ng, 8, 2, 4)
            sg = torch.sign(za)
            costs = [torch.zeros((rows, ng), device=y.device)]
            qlist = [qcur]
            packs = [None]
            for idx in idxs:
                sf = table[idx].reshape(rows, ng, 1, 1, 1)
                eff = sf * l2 * l3
                ma = (torch.round(za.abs() / eff.clamp_min(2.0 ** (-48)) * 4.0) * 0.25).clamp(0.0, 1.75)
                sgc = torch.where(ma == 0.0, torch.zeros_like(sg), sg)
                qc = (sgc * ma * eff).reshape(rows, ng, 64)
                d = qc - qcur
                dc = 2.0 * (d * gcur).sum(-1) + torch.einsum('rgi,gij,rgj->rg', d, Hss, d)
                costs.append(dc)
                qlist.append(qc)
                packs.append((sf, sgc, ma))
            choice = torch.stack(costs, 0).argmin(0)
            qstack = torch.stack(qlist, 0)
            x = qstack.permute(1, 2, 0, 3)
            qi = choice[:, :, None, None].expand(rows, ng, 1, 64)
            qnew = x.gather(2, qi).squeeze(2)
            d = qnew - qcur
            q[:, :, lo:hi] = qnew
            grad.add_(torch.einsum('gij,rgj->rgi', H[:, :, lo:hi], d))
            for ci in range(1, len(packs)):
                mask = choice == ci
                sf, sgc, ma = packs[ci]
                m5 = mask[:, :, None, None, None]
                sfv[:, bidx] = torch.where(m5, sf, sfv[:, bidx])
                sgv[:, bidx] = torch.where(m5, sgc, sgv[:, bidx])
                mav[:, bidx] = torch.where(m5, ma, mav[:, bidx])
        pp['scale_factor'] = sfv.reshape_as(pp['scale_factor']).to(torch.bfloat16)
        pp['scale_lv2'] = l2v.reshape_as(pp['scale_lv2']).to(torch.bfloat16)
        pp['scale_lv3'] = l3v.reshape_as(pp['scale_lv3']).to(torch.bfloat16)
        pp['sign'] = sgv.reshape_as(pp['sign']).to(torch.bfloat16)
        pp['mant'] = mav.reshape_as(pp['mant']).to(torch.bfloat16)
    return pp

# 功能：汇总多组 activation 样本的逐通道 RMS，用于后续通道压力/排序估计。
# =============================================================================
# ============================== LINEAR PART ==================================
# =============================================================================

def _safe139_sample_rms(ats, k):
    acc = torch.zeros(k, device=ats[0].device, dtype=torch.float32)
    ns = 0
    for a in ats:
        aa = a.float().reshape(-1, k)
        acc.add_(aa.square().mean(0))
        ns += 1
    return torch.sqrt((acc / float(max(ns, 1))).clamp_min(1e-24))

# 功能：汇总多组 activation 样本的逐通道平均绝对值，用于构造质量均衡 permutation。
def _safe139_sample_meanabs(ats, k):
    acc = torch.zeros(k, device=ats[0].device, dtype=torch.float32)
    ns = 0
    for a in ats:
        aa = a.float().reshape(-1, k)
        acc.add_(aa.abs().mean(0))
        ns += 1
    return acc / float(max(ns, 1))

# 功能：按通道 mass 从大到小分配到多个 64-block，使各 block 的总质量尽量均衡。
# 说明：返回新的全局通道 permutation；不能整除 block 时退化为全局降序排序。
def _safe139_massdiff_perm(mass, block=64):
    c = int(mass.numel())
    if c % int(block):
        return torch.argsort(mass, descending=True, stable=True)
    nb = c // int(block)
    vals = mass.detach().float().cpu().tolist()
    order = sorted(range(c), key=lambda i: (-vals[i], i))
    heap = [(0.0, j, 0) for j in range(nb)]
    heapq.heapify(heap)
    buckets = [[] for _ in range(nb)]
    for i in order:
        load, j, count = heapq.heappop(heap)
        buckets[j].append(i)
        count += 1
        load += float(vals[i])
        if count < int(block):
            heapq.heappush(heap, (load, j, count))
    flat = [i for b in buckets for i in b]
    return torch.tensor(flat, dtype=torch.long, device=mass.device)

# 功能：构造 Linear 的离线几何预处理：smooth → mass-balanced permutation → Hadamard → block phase → block 内排序。
# 说明：同时对 weight 与 calibration activation 应用互为逆向的等价变换，返回变换后的 wt/acts。
def _safe147_geometry(w, acts):
    k = int(w.shape[-1])
    nb = k // 64
    if not acts or k % 64:
        s = torch.ones(k, device=w.device)
        perm = torch.arange(k, device=w.device)
        ph = torch.ones(nb, device=w.device)
        post = torch.arange(k, device=w.device)
        return (s, perm, True, ph, post, w.float(), [a.float() for a in acts])
    amax = torch.stack([a.abs().amax(0) for a in acts], 0).mean(0)
    wmax = w.abs().amax(0)
    smooth = _v31_smooth(amax, wmax, 0.5)
    apre = [a.float() * smooth for a in acts]
    mass = _safe139_sample_rms(apre, k)
    perm = _safe139_massdiff_perm(mass, 64)
    wt0 = (w.float() / smooth).index_select(-1, perm)
    wt0 = _fwht64_v31(wt0)
    wb_max = wt0.abs().reshape(-1, nb, 64).amax((0, 2))
    ab_max = torch.zeros(nb, dtype=torch.float32, device=w.device)
    ats_pre = []
    for a in acts:
        at = (a.float() * smooth).index_select(-1, perm)
        at = _fwht64_v31(at)
        ats_pre.append(at)
        z = at.reshape(-1, nb, 64)
        ab_max = torch.maximum(ab_max, z.abs().amax((0, 2)))
    pmax = torch.sqrt(wb_max.clamp_min(2.0 ** (-24)) / ab_max.clamp_min(2.0 ** (-24)))
    phases = torch.exp(torch.log(pmax.clamp_min(1e-24)))
    phases = phases.clamp(0.5, 2.0)
    phases = phases / torch.exp(torch.log(phases).median())
    phases = phases.clamp(0.5, 2.0)
    pv = _safe90_blockvec(phases)
    wt = wt0 / pv
    ats = [a * pv for a in ats_pre]
    wrms = torch.sqrt(wt.square().mean(0).clamp_min(1e-24))
    arms = _safe139_sample_rms(ats, k)
    pressure = torch.maximum(wrms, arms)
    local = torch.argsort(pressure.reshape(nb, 64), dim=1, stable=True)
    offs = (torch.arange(nb, device=w.device) * 64)[:, None]
    post = (local + offs).reshape(-1)
    wt = wt.index_select(-1, post)
    ats = [a.index_select(-1, post) for a in ats]
    return (smooth, perm, True, phases, post, wt, ats)

# 功能：从多组 activation 估计分组协方差/Hessian，并做数据自适应的对角 shrinkage。
# 说明：rho 由跨样本 off-diagonal 信号与噪声比决定，降低小样本 covariance 估计噪声。
def _v156_cov_adaptive(acts_t, k, device, group=64, alpha=0.5, rmin=0.4, rmax=0.8, noise_gain=1.0):
    if k % group:
        return (None, None)
    ng = k // group
    Cs = []
    ws = []
    for a in acts_t:
        z = a.float().reshape(-1, ng, group)
        n = max(int(z.shape[0]), 1)
        C = torch.einsum('rgi,rgj->gij', z, z) / float(n)
        Cs.append(C)
        ws.append(float(n) ** (1 - alpha))
    if not Cs:
        return (None, None)
    W = sum(ws)
    H = sum((C * w for C, w in zip(Cs, ws))) / W
    sc = H.diagonal(dim1=-2, dim2=-1).mean(-1).abs().clamp_min(1e-12)
    Hn = H / sc[:, None, None]
    var = torch.zeros(ng, device=device)
    for C, w in zip(Cs, ws):
        Cn = C / sc[:, None, None]
        d = Cn - Hn
        diag = torch.diag_embed(d.diagonal(dim1=-2, dim2=-1))
        off = d - diag
        var += float(w) * off.square().mean((1, 2))
    var /= W
    D = torch.diag_embed(Hn.diagonal(dim1=-2, dim2=-1))
    off = Hn - D
    signal = off.square().mean((1, 2)).clamp_min(1e-12)
    rho = (signal / (signal + float(noise_gain) * var)).clamp(float(rmin), float(rmax))
    Hr = rho[:, None, None] * Hn + (1 - rho[:, None, None]) * D
    return (Hr, rho)


# 功能：Linear 离线权重量化主流程：先做安全几何变换，再用完整 activation covariance 作为 Full-H 目标优化权重。
# 说明：Self-MSE 仅作为初始化，随后每个 row chunk 都使用同一个完整 H[k,k] refinement。
def _v156_linear(weight_quant, weight_scale, calib_activation_list):
    """
    Offline Linear weight quantization with a single full-width Hessian.
    """
    w = dequantize_nvfp4(weight_quant, weight_scale).float()
    acts = _rule_safe_decode_acts(
        calib_activation_list, w.shape[-1], w.device,
    )
    s, perm, had, ph, post, wt, acts_t = _safe147_geometry(w, acts)

    k = int(w.shape[-1])

    # Full activation covariance: [1, k, k].
    Hfull, rfull = _v156_cov_adaptive(
        acts_t, k, w.device,
        group=k,
        alpha=0.5,
        rmin=0.4,
        rmax=0.8,
        noise_gain=1.0,
    )

    # Common Self-MSE initialization.
    wp, _ = _quantize_tensor_self_mse(wt, return_dequant=True)

    # Every row chunk uses the same full H[k, k].
    if isinstance(Hfull, torch.Tensor) and tuple(Hfull.shape) == (1, k, k):
        HH = Hfull[0].to(w.device, torch.float32)
        out = _v64_clone_params(wp)
        chunk_rows = 1024

        for rs in range(0, int(wt.shape[0]), chunk_rows):
            re = min(rs + chunk_rows, int(wt.shape[0]))
            pc = {
                name: value[rs:re].clone()
                for name, value in out.items()
            }
            pc = _refine_activation_full_hessian(
                wt[rs:re],
                pc,
                HH,
                mantissa_iters=4,
                block_batch=4,
            )
            for name in out:
                out[name][rs:re] = pc[name]

        wp = out

    st = _safe90_make_state(
        'v156_full_hessian_weight',
        s,
        perm,
        had,
        ph,
        transform_kind='safe_v40_marginal',
        post_perm=post.detach().cpu().to(torch.int32),
        post_perm_enabled=True,
        full_weight_hessian=True,
        rho_full_mean=(
            float(rfull.mean())
            if isinstance(rfull, torch.Tensor)
            else 0.0
        ),
    )

    return {
        'weight_params': wp,
        'activation_state': st,
    }



# 功能：使用逐 head 的完整 Hessian 对 Attention tensor 的 mantissa 做坐标下降初始化。
# 说明：每次更新一个 64-block 内的一个坐标，但梯度包含同一 head 内全部通道的相关性。
# =============================================================================
# =============================== ATTN PART ===================================
# =============================================================================

def _refine_attention_mantissas_full_hessian(y, p, head_hessian, sweeps=1):
    shape = tuple(int(s) for s in y.shape)
    if y.dim() != 2 or head_hessian.dim() != 3:
        return p
    rows, k = shape
    head_width = int(head_hessian.shape[-1])
    if head_width % 64 != 0 or k % head_width != 0:
        return p
    num_heads = k // head_width
    if tuple(head_hessian.shape) != (num_heads, head_width, head_width):
        return p

    pp = _v64_clone_params(p)
    blocks_per_head = head_width // 64
    num_blocks = k // 64
    sf = pp['scale_factor'].float().reshape(rows, num_heads, blocks_per_head, 1, 1, 1)
    l2 = pp['scale_lv2'].float().reshape(rows, num_heads, blocks_per_head, 8, 1, 1)
    l3 = pp['scale_lv3'].float().reshape(rows, num_heads, blocks_per_head, 8, 2, 1)
    eff = (sf * l2 * l3).expand(rows, num_heads, blocks_per_head, 8, 2, 4).reshape(rows, num_heads, head_width)
    u = (pp['sign'].float() * pp['mant'].float()).reshape(rows, num_heads, head_width)
    yy = y.float().reshape(rows, num_heads, head_width)
    H = head_hessian.to(y.device, torch.float32)
    grad = torch.einsum('rgi,gij->rgj', u * eff - yy, H)
    step = 0.25 * eff
    step2 = step.square() * H.diagonal(dim1=-2, dim2=-1).unsqueeze(0)
    head_index = torch.arange(num_heads, device=y.device).reshape(1, num_heads)

    for _ in range(int(sweeps)):
        for block in range(blocks_per_head):
            lo = block * 64
            hi = lo + 64
            base = 2.0 * step[:, :, lo:hi] * grad[:, :, lo:hi]
            plus_delta = base + step2[:, :, lo:hi]
            minus_delta = -base + step2[:, :, lo:hi]
            ub = u[:, :, lo:hi]
            plus_delta.masked_fill_(ub >= 1.75 - 1e-6, float('inf'))
            minus_delta.masked_fill_(ub <= -1.75 + 1e-6, float('inf'))
            choose_plus = plus_delta < minus_delta
            best, local_index = torch.minimum(plus_delta, minus_delta).min(dim=-1)
            good = best < -1e-8
            direction = torch.where(
                choose_plus.gather(2, local_index.unsqueeze(-1)).squeeze(-1),
                torch.ones_like(best),
                -torch.ones_like(best),
            )
            du = 0.25 * direction * good
            channel_index = local_index + lo
            u.scatter_add_(2, channel_index.unsqueeze(-1), du.unsqueeze(-1))
            de = du * eff.gather(2, channel_index.unsqueeze(-1)).squeeze(-1)
            grad.add_(H[head_index, channel_index] * de.unsqueeze(-1))

    mant = u.abs().reshape(rows, num_blocks, 64)
    sign = torch.sign(u).reshape(rows, num_blocks, 64)
    sign = torch.where(mant == 0.0, torch.zeros_like(sign), sign)
    pp['mant'] = mant.reshape_as(pp['mant']).to(torch.bfloat16)
    pp['sign'] = sign.reshape_as(pp['sign']).to(torch.bfloat16)
    return pp


# 功能：Attention tensor 的完整逐 head Hessian 量化封装。
# 说明：Self-MSE 仅负责物化合法参数，随后用完整 head Hessian 初始化 mantissa 并优化 scale。
def _quantize_attention_tensor_hessian(y, head_hessian):
    """Initialize and refine against each full head Hessian."""
    H = head_hessian.to(y.device, torch.float32)
    p, _ = _quantize_tensor_self_mse(y, return_dequant=False)
    p = _refine_attention_mantissas_full_hessian(y, p, H, sweeps=1)
    return _refine_attention_scales(
        y, p, H, sweeps=1,
    )


# 功能：动态 K 量化：应用 K transform 后，同时比较 partner-aware quotient 候选与中心化 Hessian 候选。
# 说明：最终按 KV head 的真实 head_dim Hessian 误差分数选择并合并参数。
def hif4_dynamic_quantize_k(
    k_quant, k_scale, kv_num_heads, head_dim, k_state,
):
    k = dequantize_nvfp4(k_quant, k_scale).float()
    x = _apply_k_transform(k, k_state, kv_num_heads, head_dim)
    head_hessian = k_state['partner_hessian']
    head_hessian_d = head_hessian.to(x.device, torch.float32)
    p0 = _safe108_quantize_k_partner(
        x, head_hessian_d, kv_num_heads, head_dim,
    )
    q0 = _v64_dequant_params(p0, tuple(x.shape)).float()
    c = (x - q0).mean(dim=-2, keepdim=True)
    target = x - c
    p1 = _quantize_attention_tensor_hessian(target, head_hessian_d)
    q1 = _v64_dequant_params(p1, tuple(x.shape)).float()
    s0 = _safe108_k_head_scores(
        x, q0, head_hessian_d, kv_num_heads, head_dim,
    )
    s1 = _safe108_k_head_scores(
        x, q1, head_hessian_d, kv_num_heads, head_dim,
    )
    best = (s1 < s0).to(torch.long)
    return _safe108_merge_k_by_head([p0, p1], best, kv_num_heads, head_dim)


# 功能：动态 Q 量化：应用 Q transform 后，使用 K-partner full-head Hessian 完成量化。
def hif4_dynamic_quantize_q(
    q_quant, q_scale, q_num_heads, head_dim, q_state,
):
    q = dequantize_nvfp4(q_quant, q_scale).float()
    q = _apply_q_transform(q, q_state, q_num_heads, head_dim)
    return _quantize_attention_tensor_hessian(
        q, q_state['partner_hessian'],
    )












# 功能：Linear 总校准入口：完成 Full-H 权重量化，并从量化后的 Wq 构造动态 activation 使用的完整 Wq^T Wq Hessian。
# 说明：清理旧版分块 Hessian 字段，只保留统一 full-H state。
# =============================================================================
# ============================== LINEAR PART ==================================
# =============================================================================

def hif4_calibration_and_quantize_weight(
    weight_quant,
    weight_scale,
    calib_activation_list,
):
    """
    Full-H-only Linear calibration.

    Offline weight optimization uses the full activation covariance.
    Dynamic activation optimization uses the full Wq^T Wq metric.
    """
    base = _v156_linear(
        weight_quant,
        weight_scale,
        calib_activation_list,
    )
    p = base['weight_params']
    st = base['activation_state']

    wq = _v64_dequant_params(
        p,
        tuple(int(s) for s in weight_quant.shape),
    ).float()

    for key in (
        'weight_hessian_blocks',
        'super256_hessian_blocks',
        'super512_hessian_blocks',
        'super1024_hessian_blocks',
        'cross1024_hessian_blocks',
        'full_hessian',
        'hessian_blocks',
    ):
        st.pop(key, None)

    k = int(wq.shape[-1])
    Hblocks = None
    if k % 64 == 0:
        H = wq.t().matmul(wq)
        scale = H.diagonal().mean().abs().clamp_min(1e-12)
        Hblocks = (H / scale).unsqueeze(0)

    if isinstance(Hblocks, torch.Tensor):
        st['hessian_blocks'] = Hblocks.cpu().to(torch.bfloat16)
        st['full_hessian'] = Hblocks[0].cpu().to(torch.bfloat16)

    st['version'] = 'unified_full_hessian'
    return {
        'weight_params': p,
        'activation_state': st,
    }


# 功能：在完整 H[k,k] 二次型下，对每个 64-block 内的 mantissa/sign 做批量坐标下降。
# 说明：每次只尝试 ±0.25 mantissa 步长，并用全局 gradient 增量更新避免反复重算完整目标。
def _refine_full_hessian_batched(y, p, H, iters=20, block_batch=4):
    if y.dim() != 2 or int(y.shape[-1]) % 64 != 0:
        return p
    rows = int(y.shape[0])
    k = int(y.shape[-1])
    nb = k // 64
    if not isinstance(H, torch.Tensor) or H.numel() != k * k:
        return p
    pp = _v64_clone_params(p)
    sf = pp['scale_factor'].float().reshape(rows, nb, 1, 1, 1)
    l2 = pp['scale_lv2'].float().reshape(rows, nb, 8, 1, 1)
    l3 = pp['scale_lv3'].float().reshape(rows, nb, 8, 2, 1)
    eff = (sf * l2 * l3).expand(rows, nb, 8, 2, 4).reshape(rows, k)
    u = (pp['sign'].float() * pp['mant'].float()).reshape(rows, k)
    yy = y.float().reshape(rows, k)
    HH = H.to(y.device, torch.float32).reshape(k, k)
    g = (u * eff - yy).matmul(HH)
    step = 0.25 * eff
    step2 = step.square() * HH.diagonal().reshape(1, k)
    batch = max(1, int(block_batch))
    for _ in range(int(iters)):
        for block0 in range(0, nb, batch):
            block_count = min(batch, nb - block0)
            lo = block0 * 64
            hi = lo + block_count * 64
            base = 2.0 * step[:, lo:hi] * g[:, lo:hi]
            dp = (base + step2[:, lo:hi]).reshape(rows, block_count, 64)
            dm = (-base + step2[:, lo:hi]).reshape(rows, block_count, 64)
            ub = u[:, lo:hi].reshape(rows, block_count, 64)
            dp.masked_fill_(ub >= 1.75 - 1e-06, float('inf'))
            dm.masked_fill_(ub <= -1.75 + 1e-06, float('inf'))
            plus = dp < dm
            best, j0 = torch.minimum(dp, dm).min(dim=-1)
            good = best < -1e-08
            direction = torch.where(plus.gather(2, j0.unsqueeze(-1)).squeeze(-1), torch.ones_like(best), -torch.ones_like(best))
            du = 0.25 * direction * good
            offsets = (lo + torch.arange(block_count, device=y.device) * 64).reshape(1, block_count)
            j = j0 + offsets
            u.scatter_add_(1, j, du)
            de = du * eff.gather(1, j)
            g.add_((HH[j] * de[:, :, None]).sum(dim=1))
    ma = u.abs().reshape(rows, nb, 64)
    sg = torch.sign(u).reshape(rows, nb, 64)
    sg = torch.where(ma == 0.0, torch.zeros_like(sg), sg)
    pp['mant'] = ma.reshape_as(pp['mant']).to(torch.bfloat16)
    pp['sign'] = sg.reshape_as(pp['sign']).to(torch.bfloat16)
    return pp


# 功能：为 full-H hierarchy refinement 初始化共享状态：参数副本、当前反量化 q，以及全局梯度 g=(q-y)@H。
# 说明：后续 scale/lv3/lv2 sweep 复用同一 q/g，保证跨 block 相互作用被保留。
def _full_hierarchy_init(y, p, H):
    """Initialize the shared full-H hierarchy state q and g=(q-y)@H once."""
    if y.dim() != 2 or int(y.shape[-1]) % 64 != 0:
        return None
    rows = int(y.shape[0])
    k = int(y.shape[-1])
    if not isinstance(H, torch.Tensor) or H.numel() != k * k:
        return None
    pp = _v64_clone_params(p)
    HH = H.to(y.device, torch.float32).reshape(k, k)
    yy = y.float().reshape(rows, k)
    q = _v64_dequant_params(pp, tuple((int(s) for s in y.shape))).float().reshape(rows, k)
    g = (q - yy).matmul(HH)
    return pp, q, g


# 功能：在完整 H 下顺序扫描所有 64-block，对每个 block 的 scale_factor 搜索可改善目标的 E6M2 候选。
# 说明：接受候选后立即更新 q 和全局 gradient，再处理下一 block。
def _refine_full_scale_once(y, p, H, q=None, g=None):
    """
    One exact full-H scale_factor sweep.

    A scale candidate only changes one 64-value block S. Candidate scoring uses
        dL = 2 d^T g_S + d^T H_SS d,
    and an accepted candidate updates the shared global gradient with
        g <- g + d^T H_{S,:}.
    No full objective is recomputed per candidate.
    """
    if y.dim() != 2 or int(y.shape[-1]) % 64 != 0:
        return p, q, g
    rows = int(y.shape[0])
    k = int(y.shape[-1])
    nb = k // 64
    if not isinstance(H, torch.Tensor) or H.numel() != k * k:
        return p, q, g

    pp = _v64_clone_params(p)
    HH = H.to(y.device, torch.float32).reshape(k, k)
    yy = y.float().reshape(rows, k)
    if q is None:
        q = _v64_dequant_params(pp, tuple((int(s) for s in y.shape))).float().reshape(rows, k)
    if g is None:
        g = (q - yy).matmul(HH)

    table = _build_e6m2_table(y.device)
    last = int(table.numel() - 1)
    sfv = pp['scale_factor'].float().reshape(rows, nb)
    l2v = pp['scale_lv2'].float().reshape(rows, nb, 8)
    l3v = pp['scale_lv3'].float().reshape(rows, nb, 8, 2)
    sgv = pp['sign'].float().reshape(rows, nb, 8, 2, 4)
    mav = pp['mant'].float().reshape(rows, nb, 8, 2, 4)

    for b in range(nb):
        lo = b * 64
        hi = lo + 64
        z = yy[:, lo:hi].reshape(rows, 8, 2, 4)
        qcur = q[:, lo:hi]
        gcur = g[:, lo:hi]
        Hss = HH[lo:hi, lo:hi]

        sfcur = sfv[:, b].clamp_min(2.0 ** (-48))
        c = qcur / sfcur[:, None]
        den = torch.einsum('ri,ij,rj->r', c, Hss, c).clamp_min(1e-20)
        num = (c * gcur).sum(-1)
        sfstar = (sfcur - num / den).clamp(min=2.0 ** (-48), max=49152.0)
        curidx = _nearest_e6m2_index(sfcur, table)
        optidx = _nearest_e6m2_index(sfstar, table)
        idxs = (
            curidx,
            (curidx - 1).clamp(0, last),
            (curidx + 1).clamp(0, last),
            (optidx - 1).clamp(0, last),
            optidx,
            (optidx + 1).clamp(0, last),
        )

        l2 = l2v[:, b].reshape(rows, 8, 1, 1)
        l3 = l3v[:, b].reshape(rows, 8, 2, 1)
        sg0 = torch.sign(z)
        costs = [torch.zeros(rows, device=y.device, dtype=torch.float32)]
        qlist = [qcur]
        packs = [None]
        for idx in idxs:
            sfc = table[idx]
            eff = sfc[:, None, None, None] * l2 * l3
            ma = (torch.round(z.abs() / eff.clamp_min(2.0 ** (-48)) * 4.0) * 0.25).clamp(0.0, 1.75)
            sg = torch.where(ma == 0.0, torch.zeros_like(sg0), sg0)
            qc = (sg * ma * eff).reshape(rows, 64)
            d = qc - qcur
            dc = 2.0 * (d * gcur).sum(-1) + torch.einsum('ri,ij,rj->r', d, Hss, d)
            costs.append(dc)
            qlist.append(qc)
            packs.append((sfc, sg, ma))

        choice = torch.stack(costs, dim=0).argmin(0)
        qstack = torch.stack(qlist, dim=1)
        qnew = qstack.gather(1, choice[:, None, None].expand(rows, 1, 64)).squeeze(1)
        d = qnew - qcur
        good = choice != 0
        if bool(good.any()):
            q[:, lo:hi] = qnew
            g.add_(d.matmul(HH[lo:hi, :]))
            for ci in range(1, len(packs)):
                mask = choice == ci
                if not bool(mask.any()):
                    continue
                sfc, sg, ma = packs[ci]
                sfv[mask, b] = sfc[mask]
                sgv[mask, b] = sg[mask]
                mav[mask, b] = ma[mask]

    pp['scale_factor'] = sfv.reshape_as(pp['scale_factor']).to(torch.bfloat16)
    pp['scale_lv2'] = l2v.reshape_as(pp['scale_lv2']).to(torch.bfloat16)
    pp['scale_lv3'] = l3v.reshape_as(pp['scale_lv3']).to(torch.bfloat16)
    pp['sign'] = sgv.reshape_as(pp['sign']).to(torch.bfloat16)
    pp['mant'] = mav.reshape_as(pp['mant']).to(torch.bfloat16)
    return pp, q, g


# 功能：在完整 H 下对 lv3（每 4 元素共享一级倍率）做严格顺序 toggle refinement。
# 说明：每个 64-block 每行最多接受一个最佳 4-value toggle，并立即更新全局 gradient。
def _refine_full_lv3_once(y, p, H, q=None, g=None):
    """
    One strict-sequential full-H lv3 pass (64-block batch size = 1).

    For each 64-value block, evaluate its 16 possible lv3 toggles using only
    the exact local 4x4 H_SS terms. Accept at most one improving toggle per row
    in that block, immediately update
        g <- g + d^T H_{S,:},
    and only then move to the next 64 block. Therefore candidate decisions for
    different 64 blocks never share a stale gradient and no simultaneous
    cross-block term is omitted.
    """
    if y.dim() != 2 or int(y.shape[-1]) % 64 != 0:
        return p, q, g
    rows = int(y.shape[0])
    k = int(y.shape[-1])
    nb = k // 64
    if not isinstance(H, torch.Tensor) or H.numel() != k * k:
        return p, q, g

    pp = _v64_clone_params(p)
    HH = H.to(y.device, torch.float32).reshape(k, k)
    yy = y.float().reshape(rows, k)
    if q is None:
        q = _v64_dequant_params(pp, tuple((int(s) for s in y.shape))).float().reshape(rows, k)
    if g is None:
        g = (q - yy).matmul(HH)

    sfv = pp['scale_factor'].float().reshape(rows, nb)
    l2v = pp['scale_lv2'].float().reshape(rows, nb * 8)
    l3v = pp['scale_lv3'].float().reshape(rows, nb * 16)
    sgv = pp['sign'].float().reshape(rows, nb * 16, 4)
    mav = pp['mant'].float().reshape(rows, nb * 16, 4)
    sf4 = sfv.repeat_interleave(16, dim=1)
    l24 = l2v.repeat_interleave(2, dim=1)
    ridx = torch.arange(rows, device=y.device)

    # Strict block_batch=1: finish candidate selection and the global-gradient
    # update for this 64 block before evaluating any candidate in the next one.
    for b in range(nb):
        lo = b * 64
        hi = lo + 64
        c0 = b * 16
        c1 = c0 + 16
        z4 = yy[:, lo:hi].reshape(rows, 16, 4)
        q4 = q[:, lo:hi].reshape(rows, 16, 4)
        g4 = g[:, lo:hi].reshape(rows, 16, 4)
        cur = l3v[:, c0:c1]
        new = torch.where(cur > 1.5, torch.ones_like(cur), torch.full_like(cur, 2.0))
        eff = sf4[:, c0:c1] * l24[:, c0:c1] * new
        ma = (torch.round(z4.abs() / eff[:, :, None].clamp_min(2.0 ** (-48)) * 4.0) * 0.25).clamp(0.0, 1.75)
        sg = torch.where(ma == 0.0, torch.zeros_like(z4), torch.sign(z4))
        qn = sg * ma * eff[:, :, None]
        d = qn - q4

        H64 = HH[lo:hi, lo:hi]
        H4 = torch.stack([H64[i * 4:(i + 1) * 4, i * 4:(i + 1) * 4] for i in range(16)], dim=0)
        delta = 2.0 * (d * g4).sum(-1) + torch.einsum('rbi,bij,rbj->rb', d, H4, d)
        best, idx = delta.min(dim=1)
        good = best < -1e-08
        if not bool(good.any()):
            continue

        rr = ridx[good]
        ii = idx[good]
        glob = c0 + ii
        q4[rr, ii] = qn[rr, ii]
        l3v[rr, glob] = new[rr, ii]
        mav[rr, glob] = ma[rr, ii]
        sgv[rr, glob] = sg[rr, ii]

        # Rows are independent objectives; different rows may choose different
        # 4-value coordinates. Group only by coordinate for the H[S,:] update.
        for gi in range(16):
            mask = good & (idx == gi)
            if bool(mask.any()):
                s0 = lo + gi * 4
                g[mask] = g[mask] + d[mask, gi].matmul(HH[s0:s0 + 4, :])

    pp['scale_lv3'] = l3v.reshape_as(pp['scale_lv3']).to(torch.bfloat16)
    pp['sign'] = sgv.reshape_as(pp['sign']).to(torch.bfloat16)
    pp['mant'] = mav.reshape_as(pp['mant']).to(torch.bfloat16)
    return pp, q, g


# 功能：在完整 H 下对 lv2（每 8 元素共享一级倍率）做严格顺序 toggle refinement。
# 说明：每个 64-block 每行最多接受一个最佳 8-value toggle，并立即更新全局 gradient。
def _refine_full_lv2_once(y, p, H, q=None, g=None):
    """
    One strict-sequential full-H lv2 pass (64-block batch size = 1).

    For each 64-value block, evaluate its 8 possible lv2 toggles using only
    the exact local 8x8 H_SS terms. Accept at most one improving toggle per row
    in that block, immediately update the global gradient with H[S,:], and only
    then evaluate the next 64 block.
    """
    if y.dim() != 2 or int(y.shape[-1]) % 64 != 0:
        return p, q, g
    rows = int(y.shape[0])
    k = int(y.shape[-1])
    nb = k // 64
    if not isinstance(H, torch.Tensor) or H.numel() != k * k:
        return p, q, g

    pp = _v64_clone_params(p)
    HH = H.to(y.device, torch.float32).reshape(k, k)
    yy = y.float().reshape(rows, k)
    if q is None:
        q = _v64_dequant_params(pp, tuple((int(s) for s in y.shape))).float().reshape(rows, k)
    if g is None:
        g = (q - yy).matmul(HH)

    sfv = pp['scale_factor'].float().reshape(rows, nb)
    l2v = pp['scale_lv2'].float().reshape(rows, nb * 8)
    l3v = pp['scale_lv3'].float().reshape(rows, nb * 8, 2)
    sgv = pp['sign'].float().reshape(rows, nb * 8, 8)
    mav = pp['mant'].float().reshape(rows, nb * 8, 8)
    sf8 = sfv.repeat_interleave(8, dim=1)
    ridx = torch.arange(rows, device=y.device)

    # Strict block_batch=1 for hierarchy moves.
    for b in range(nb):
        lo = b * 64
        hi = lo + 64
        c0 = b * 8
        c1 = c0 + 8
        z8 = yy[:, lo:hi].reshape(rows, 8, 2, 4)
        q8 = q[:, lo:hi].reshape(rows, 8, 8)
        g8 = g[:, lo:hi].reshape(rows, 8, 8)
        cur = l2v[:, c0:c1]
        new = torch.where(cur > 1.5, torch.ones_like(cur), torch.full_like(cur, 2.0))
        eff = sf8[:, c0:c1, None, None] * new[:, :, None, None] * l3v[:, c0:c1, :, None]
        ma = (torch.round(z8.abs() / eff.clamp_min(2.0 ** (-48)) * 4.0) * 0.25).clamp(0.0, 1.75)
        sg = torch.where(ma == 0.0, torch.zeros_like(z8), torch.sign(z8))
        qn = (sg * ma * eff).reshape(rows, 8, 8)
        d = qn - q8

        H64 = HH[lo:hi, lo:hi]
        H8 = torch.stack([H64[i * 8:(i + 1) * 8, i * 8:(i + 1) * 8] for i in range(8)], dim=0)
        delta = 2.0 * (d * g8).sum(-1) + torch.einsum('rbi,bij,rbj->rb', d, H8, d)
        best, idx = delta.min(dim=1)
        good = best < -1e-08
        if not bool(good.any()):
            continue

        rr = ridx[good]
        ii = idx[good]
        glob = c0 + ii
        q8[rr, ii] = qn[rr, ii]
        l2v[rr, glob] = new[rr, ii]
        ma8 = ma.reshape(rows, 8, 8)
        sg8 = sg.reshape(rows, 8, 8)
        mav[rr, glob] = ma8[rr, ii]
        sgv[rr, glob] = sg8[rr, ii]

        for gi in range(8):
            mask = good & (idx == gi)
            if bool(mask.any()):
                s0 = lo + gi * 8
                g[mask] = g[mask] + d[mask, gi].matmul(HH[s0:s0 + 8, :])

    pp['scale_lv2'] = l2v.reshape_as(pp['scale_lv2']).to(torch.bfloat16)
    pp['sign'] = sgv.reshape_as(pp['sign']).to(torch.bfloat16)
    pp['mant'] = mav.reshape_as(pp['mant']).to(torch.bfloat16)
    return pp, q, g

# 功能：统一的动态 activation Full-H refinement：mantissa 坐标下降后，依次执行 scale、lv3、lv2 hierarchy 优化。
# 说明：所有阶段共享同一个完整 H[k,k]，不再按 hidden width 切换不同优化逻辑。
def _refine_activation_full_hessian(y, p, H, *, mantissa_iters=10, block_batch=4):
    """Run the complete activation refinement against one full Hessian.

    The public activation state stores the metric as [1, k, k] for every
    supported hidden width.  This helper keeps the dynamic path width-agnostic:
      1) exact full-H mantissa coordinate descent,
      2) one shared q/g hierarchy state,
      3) full-H scale, lv3 and lv2 refinement.
    """
    k = int(y.shape[-1])
    if y.dim() != 2 or k % 64 != 0:
        return p
    if not isinstance(H, torch.Tensor) or H.numel() != k * k:
        return p

    HH = H.to(y.device, torch.float32).reshape(k, k)
    p = _refine_full_hessian_batched(
        y, p, HH, int(mantissa_iters), block_batch=int(block_batch),
    )

    full_state = _full_hierarchy_init(y, p, HH)
    if full_state is None:
        return p

    p, q_full, g_full = full_state
    p, q_full, g_full = _refine_full_scale_once(
        y, p, HH, q_full, g_full,
    )
    p, q_full, g_full = _refine_full_lv3_once(
        y, p, HH, q_full, g_full,
    )
    p, q_full, g_full = _refine_full_lv2_once(
        y, p, HH, q_full, g_full,
    )
    return p


# 功能：动态 Linear activation 量化入口：解码并重放 calibration 几何变换，以 Self-MSE 初始化，再统一走 Full-H refinement。
# 说明：只有 state 中 hessian_blocks 形状严格为 [1,k,k] 时启用 Full-H，否则安全退回初始化结果。
def hif4_dynamic_quantize_activation(aq, asc, st):
    """Dynamic activation quantization using one unified full-H metric.

    All supported hidden widths (64/256/1024/2048/...) use exactly the same
    path when calibration provides hessian_blocks with shape [1, k, k].
    Invalid or legacy block-diagonal states simply fall back to the self-MSE
    initialization instead of taking a different optimization path.
    """
    st = st if isinstance(st, dict) else {}

    y = _safe90_decode_transform_activation(aq, asc, st)
    post = st.get('post_perm')
    if isinstance(post, torch.Tensor):
        y = y.index_select(-1, post.to(y.device, dtype=torch.long))

    # One common initialization for every width.
    p = _quantize_tensor_self_mse(y, return_dequant=False)[0]

    Hblocks = st.get('hessian_blocks')
    if not isinstance(Hblocks, torch.Tensor):
        return p

    k = int(y.shape[-1])
    if tuple(Hblocks.shape) != (1, k, k):
        return p

    return _refine_activation_full_hessian(
        y, p, Hblocks[0], mantissa_iters=10, block_batch=4,
    )
