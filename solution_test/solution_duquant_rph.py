"""Cheap DuQuant variant C: Smooth -> data-aware R1 -> zigzag P -> H64.

R1 follows the existing DuQuant solution's outlier-targeted Householder logic.
Standalone implementation for the six-function demo interface.
======================== Linear ========================
[Linear][Group 0] calibration: PASSED [241.57ms] (W=(8192, 2048), num_calib=5)
[Linear][Group 0][Test 0] activation: FAILED [2.40ms] (W=(8192, 2048), A=(10, 2048))
      MatMul MSE 5.7128e-03 exceeds threshold 0.001
[Linear][Group 0][Test 1] activation: FAILED [5.01ms] (W=(8192, 2048), A=(128, 2048))
      MatMul MSE 5.5278e-03 exceeds threshold 0.001
[Linear][Group 0][Test 2] activation: FAILED [9.48ms] (W=(8192, 2048), A=(512, 2048))
      MatMul MSE 4.4321e-03 exceeds threshold 0.001
[Linear][Group 0][Test 3] activation: FAILED [18.95ms] (W=(8192, 2048), A=(1024, 2048))
      MatMul MSE 4.2131e-03 exceeds threshold 0.001
[Linear][Group 0][Test 4] activation: FAILED [16.45ms] (W=(8192, 2048), A=(1024, 2048))
      MatMul MSE 4.1755e-03 exceeds threshold 0.001

====================== Attention ======================
[Attention][Group 0] calibration: PASSED [0.00ms] (q_heads=16, kv_heads=2, head_dim=256, num_calib=5)
[Attention][Group 0][Test 0] FAILED [6.65ms] (Q=(10, 4096), K=(10, 512), V=(10, 512))
      Attention MSE 1.0388e-03 exceeds threshold 0.001
[Attention][Group 0][Test 1] PASSED (MSE=4.0943e-04) [25.86ms] (Q=(128, 4096), K=(128, 512), V=(128, 512))
[Attention][Group 0][Test 2] PASSED (MSE=2.8481e-04) [50.14ms] (Q=(512, 4096), K=(512, 512), V=(512, 512))
[Attention][Group 0][Test 3] PASSED (MSE=2.1662e-04) [86.83ms] (Q=(1024, 4096), K=(1024, 512), V=(1024, 512))
[Attention][Group 0][Test 4] PASSED (MSE=2.3384e-04) [128.82ms] (Q=(1024, 4096), K=(1024, 512), V=(1024, 512))

"""
from __future__ import annotations

import torch


EPS = 1e-12
DUQUANT_ALPHA = 0.5


def dequantize_nvfp4(quant_float, scale_float, blk_size: int = 16):
    channels = int(quant_float.shape[-1])
    if channels % blk_size:
        raise ValueError(f"last dimension {channels} is not divisible by {blk_size}")
    x = quant_float.unflatten(-1, (-1, blk_size)) * scale_float.unsqueeze(-1)
    return x.flatten(-2, -1).to(torch.bfloat16)


def _round_to_e6m2(x):
    x = x.float().clamp(min=2.0**-48, max=(2.0**15) * 1.5)
    e = torch.floor(torch.log2(x)).clamp(-48.0, 15.0)
    base = torch.pow(torch.tensor(2.0, device=x.device), e)
    m = torch.round((x / base - 1.0) * 4.0)
    carry = m >= 4.0
    e, m = torch.where(carry, e + 1.0, e), torch.where(carry, torch.zeros_like(m), m)
    overflow = e > 15.0
    e, m = torch.where(overflow, torch.full_like(e, 15.0), e), torch.where(overflow, torch.full_like(m, 2.0), m)
    m = torch.where((e >= 15.0) & (m > 2.0), torch.full_like(m, 2.0), m).clamp(0.0, 3.0)
    return torch.pow(torch.tensor(2.0, device=x.device), e) * (1.0 + m * 0.25)


def _hif4_direct(x):
    if x.shape[-1] % 64:
        raise ValueError(f"HiF4 requires last dim divisible by 64, got {x.shape[-1]}")
    xg = x.float().unflatten(-1, (x.shape[-1] // 64, 8, 2, 4))
    ax = xg.abs()
    sf = _round_to_e6m2((ax.amax(dim=(-1, -2, -3), keepdim=True) / 7.0).clamp_min(2.0**-48))
    s2 = torch.where(ax.amax(dim=(-1, -2), keepdim=True) > 4.0 * sf, 2.0, 1.0)
    s3 = torch.where(ax.amax(dim=-1, keepdim=True) > 2.0 * sf * s2, 2.0, 1.0)
    mant = (torch.round(ax / (sf * s2 * s3).clamp_min(EPS) * 4.0) * 0.25).clamp(0.0, 1.75)
    sign = torch.where(mant > 0.0, torch.sign(xg), torch.zeros_like(xg))
    return {"scale_factor": sf.to(torch.bfloat16), "scale_lv2": s2.to(torch.bfloat16),
            "scale_lv3": s3.to(torch.bfloat16), "sign": sign.to(torch.bfloat16),
            "mant": mant.to(torch.bfloat16)}


def _householder64(position, device):
    source = torch.zeros(64, dtype=torch.float32, device=device)
    source[position] = 1.0
    target = torch.full_like(source, 1.0 / 8.0)
    v = source - target
    return torch.eye(64, device=device) - 2.0 * torch.outer(v, v) / v.dot(v).clamp_min(EPS)


def _block_rotate(x, rotation):
    shape = x.shape
    return torch.matmul(x.float().reshape(*shape[:-1], -1, 64), rotation).reshape(shape)


def _block_hadamard(x):
    """Normalized H64 using six butterfly stages, not a dense matmul."""
    shape = x.shape
    y = x.float().reshape(*shape[:-1], -1, 64)
    block_shape = y.shape
    for width in (1, 2, 4, 8, 16, 32):
        y = y.reshape(*block_shape[:-1], 64 // (2 * width), 2, width)
        left, right = y[..., 0, :], y[..., 1, :]
        y = torch.cat((left + right, left - right), dim=-1).reshape(block_shape)
    return y.reshape(shape) / 8.0


def _zigzag_permutation(score):
    order = torch.argsort(score, descending=True)
    groups = order.numel() // 64
    table = torch.empty(groups, 64, dtype=order.dtype, device=order.device)
    for rank in range(order.numel()):
        lane, offset = rank // groups, rank % groups
        group = offset if lane % 2 == 0 else groups - 1 - offset
        table[group, lane] = order[rank]
    return table.reshape(-1)


def _stats_and_samples(calib_activation_list, channels, device):
    amax = torch.zeros(channels, dtype=torch.float32, device=device)
    samples = []
    for aq, asc in calib_activation_list:
        a = dequantize_nvfp4(aq, asc).to(device=device, dtype=torch.float32).reshape(-1, channels)
        amax = torch.maximum(amax, a.abs().amax(dim=0))
        samples.append(a)
    return amax, samples


def hif4_calibration_and_quantize_weight(weight_quant, weight_scale, calib_activation_list):
    w = dequantize_nvfp4(weight_quant, weight_scale).float()
    amax, samples = _stats_and_samples(calib_activation_list, w.shape[-1], w.device)
    scale = amax.clamp_min(EPS).pow(DUQUANT_ALPHA) / w.abs().amax(dim=0).clamp_min(EPS).pow(1.0 - DUQUANT_ALPHA)
    local_score = torch.zeros(64, dtype=torch.float32, device=w.device)
    for a in samples:
        score = (a / scale).abs().reshape(-1, w.shape[-1] // 64, 64).amax(dim=(0, 1))
        local_score = torch.maximum(local_score, score)
    r1 = _householder64(int(local_score.argmax().item()), w.device)
    channel_score = torch.zeros(w.shape[-1], dtype=torch.float32, device=w.device)
    for a in samples:
        channel_score = torch.maximum(channel_score, _block_rotate(a / scale, r1).abs().amax(dim=0))
    perm = _zigzag_permutation(channel_score)
    wt = _block_hadamard(_block_rotate(w * scale, r1).index_select(-1, perm))
    state = {"smooth_scale": scale.detach().cpu(), "rotation1": r1.detach().cpu(),
             "perm": perm.detach().cpu()}
    return {"weight_params": _hif4_direct(wt), "activation_state": state}


def hif4_dynamic_quantize_activation(activation_quant, activation_scale, activation_state):
    a = dequantize_nvfp4(activation_quant, activation_scale).float()
    scale = activation_state["smooth_scale"].to(a.device)
    r1 = activation_state["rotation1"].to(a.device)
    perm = activation_state["perm"].to(a.device)
    at = _block_rotate(a / scale, r1).index_select(-1, perm)
    return _hif4_direct(_block_hadamard(at))


def hif4_calibration_attention(calib_qkv_list: list, q_num_heads: int, kv_num_heads: int, head_dim: int):
    return {"q_state": None, "k_state": None, "v_state": None}


def hif4_dynamic_quantize_q(q_quant, q_scale, q_num_heads, head_dim, q_state):
    return _hif4_direct(dequantize_nvfp4(q_quant, q_scale).float())


def hif4_dynamic_quantize_k(k_quant, k_scale, kv_num_heads, head_dim, k_state):
    return _hif4_direct(dequantize_nvfp4(k_quant, k_scale).float())


def hif4_dynamic_quantize_v(v_quant, v_scale, kv_num_heads, head_dim, v_state):
    return _hif4_direct(dequantize_nvfp4(v_quant, v_scale).float())
