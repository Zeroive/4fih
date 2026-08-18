from __future__ import annotations

from typing import Any
import math
import torch


# ============================================================
# Tunable switches for the first baseline
# ============================================================
USE_LINEAR_PERMUTE = True
USE_LINEAR_SMOOTH = True
LINEAR_SMOOTH_ALPHA = 0.50
LINEAR_SMOOTH_CLAMP = (1.0 / 16.0, 16.0)
USE_LINEAR_HADAMARD64 = False

USE_QK_SMOOTH = True
QK_SMOOTH_CLAMP = (1.0 / 16.0, 16.0)
USE_QK_HADAMARD64 = False

# Multi-start + alternating optimization of the HiF4 base scale.
# 1.0 corresponds to the paper's max/7 initialization.
HIF4_INIT_RATIOS = (1.10, 1.00, 0.95, 0.90, 0.85, 0.80, 0.72)
HIF4_ALT_ITERS = 2
EPS = 1e-12


# ============================================================
# NVFP4 helper
# ============================================================
def dequantize_nvfp4(
    quant_float: torch.Tensor,
    scale_float: torch.Tensor,
    blk_size: int = 16,
) -> torch.Tensor:
    channels = int(quant_float.shape[-1])
    if channels % blk_size != 0:
        raise ValueError(
            f"last dimension {channels} is not divisible by block size {blk_size}"
        )
    x = quant_float.unflatten(-1, (-1, blk_size))
    x = x * scale_float.unsqueeze(-1)
    return x.flatten(-2, -1).to(torch.bfloat16)


# ============================================================
# HiF4 numeric helpers
# ============================================================
def _round_to_e6m2(x: torch.Tensor) -> torch.Tensor:
    """Round positive FP32 values to numerical values representable by HiF4 E6M2.

    E6M2: unsigned, exponent bias 48, unbiased exponent [-48, 15],
    significand 1.{00,01,10,11}; the top code (E=15, M=3) is NaN,
    so the largest finite value is 2^15 * 1.5.
    """
    x = x.float().clamp_min(0.0)
    min_v = 2.0 ** -48
    max_v = (2.0 ** 15) * 1.5
    x = x.clamp(min=min_v, max=max_v)

    e = torch.floor(torch.log2(x))
    e = e.clamp(-48.0, 15.0)
    base = torch.pow(torch.tensor(2.0, device=x.device), e)
    norm = x / base

    # RNE is fine for the first baseline. torch.round uses bankers rounding.
    m = torch.round((norm - 1.0) * 4.0)

    carry = m >= 4.0
    e = torch.where(carry, e + 1.0, e)
    m = torch.where(carry, torch.zeros_like(m), m)

    # Saturate the finite range; exponent 15, mantissa 3 is reserved NaN.
    over = e > 15.0
    e = torch.where(over, torch.full_like(e, 15.0), e)
    m = torch.where(over, torch.full_like(m, 2.0), m)
    m = torch.where((e >= 15.0) & (m > 2.0), torch.full_like(m, 2.0), m)
    m = m.clamp(0.0, 3.0)

    return torch.pow(torch.tensor(2.0, device=x.device), e) * (1.0 + m * 0.25)


def _quantize_s1p2(abs_x: torch.Tensor, local_scale: torch.Tensor) -> torch.Tensor:
    """Return non-negative S1P2 mantissa values: {0, .25, ..., 1.75}."""
    z = abs_x / local_scale.clamp_min(EPS)
    return (torch.round(z * 4.0) * 0.25).clamp_(0.0, 1.75)


def _prepare_error_weight(x: torch.Tensor, error_weight: torch.Tensor | None) -> torch.Tensor:
    if error_weight is None:
        return torch.ones_like(x, dtype=torch.float32)
    w = error_weight.to(device=x.device, dtype=torch.float32)
    if w.ndim == 1:
        shape = [1] * (x.ndim - 1) + [x.shape[-1]]
        w = w.reshape(shape)
    return w.expand_as(x)


def _best_hierarchy_for_scale(
    xg: torch.Tensor,
    wg: torch.Tensor,
    sf: torch.Tensor,
):
    """Exact lv2/lv3 choice for a fixed E6M2 base scale.

    xg, wg: (..., G, 8, 2, 4)
    sf:     (..., G, 1, 1, 1)
    """
    ax = xg.abs()

    by_e2 = []
    for e2_factor in (1.0, 2.0):
        by_e3 = []
        for e3_factor in (1.0, 2.0):
            local_scale = sf * e2_factor * e3_factor
            mant = _quantize_s1p2(ax, local_scale)
            recon = mant * local_scale
            err4 = ((ax - recon) ** 2 * wg).sum(dim=-1)  # (..., G, 8, 2)
            by_e3.append((err4, mant))

        choose_e3_2 = by_e3[1][0] < by_e3[0][0]
        child_err = torch.where(choose_e3_2, by_e3[1][0], by_e3[0][0])
        mant = torch.where(choose_e3_2.unsqueeze(-1), by_e3[1][1], by_e3[0][1])
        lv3 = torch.where(
            choose_e3_2.unsqueeze(-1),
            torch.full_like(mant[..., :1], 2.0),
            torch.full_like(mant[..., :1], 1.0),
        )
        err8 = child_err.sum(dim=-1)  # (..., G, 8)
        by_e2.append((err8, mant, lv3))

    choose_e2_2 = by_e2[1][0] < by_e2[0][0]  # (..., G, 8)
    mant = torch.where(choose_e2_2.unsqueeze(-1).unsqueeze(-1), by_e2[1][1], by_e2[0][1])
    lv3 = torch.where(choose_e2_2.unsqueeze(-1).unsqueeze(-1), by_e2[1][2], by_e2[0][2])
    lv2 = torch.where(
        choose_e2_2.unsqueeze(-1).unsqueeze(-1),
        torch.full_like(mant[..., :1, :1], 2.0),
        torch.full_like(mant[..., :1, :1], 1.0),
    )

    coeff = mant * lv2 * lv3
    sign = torch.sign(xg)
    sign = torch.where(mant > 0, sign, torch.zeros_like(sign))
    err = (((ax - coeff * sf) ** 2) * wg).sum(dim=(-1, -2, -3))  # (..., G)
    return err, lv2, lv3, sign, mant, coeff


def _hif4_quantize(
    x: torch.Tensor,
    error_weight: torch.Tensor | None = None,
    init_ratios=HIF4_INIT_RATIOS,
    alt_iters: int = HIF4_ALT_ITERS,
) -> dict[str, torch.Tensor]:
    """Quantize the last dimension in groups of 64.

    Improvements over direct cast:
      1) multi-start E6M2 scale initialization around max/7;
      2) exact discrete search of lv2/lv3 for each fixed base scale;
      3) alternating least-squares update of the common base scale;
      4) optional output-sensitivity weights.
    """
    if x.shape[-1] % 64 != 0:
        raise ValueError(f"HiF4 requires last dim divisible by 64, got {x.shape[-1]}")

    xf = x.float()
    wf = _prepare_error_weight(xf, error_weight)
    G = x.shape[-1] // 64
    xg = xf.unflatten(-1, (G, 8, 2, 4))
    wg = wf.unflatten(-1, (G, 8, 2, 4))

    peak = xg.abs().amax(dim=(-1, -2, -3))  # (..., G)
    best = None

    for ratio in init_ratios:
        sf = _round_to_e6m2((peak * (float(ratio) / 7.0)).clamp_min(2.0 ** -48))
        sf = sf.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)

        for _ in range(alt_iters):
            _, lv2, lv3, sign, mant, coeff = _best_hierarchy_for_scale(xg, wg, sf)

            # Coordinate-descent / Lloyd-style base-scale update:
            # min_s sum_i w_i (|x_i| - s * coeff_i)^2
            num = (wg * xg.abs() * coeff).sum(dim=(-1, -2, -3))
            den = (wg * coeff.square()).sum(dim=(-1, -2, -3)).clamp_min(EPS)
            sf_ls = num / den

            # All-zero groups have coeff==0; keep minimum scale, mantissas stay zero.
            sf_ls = torch.where(peak > 0, sf_ls, torch.full_like(sf_ls, 2.0 ** -48))
            sf = _round_to_e6m2(sf_ls).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)

        err, lv2, lv3, sign, mant, _ = _best_hierarchy_for_scale(xg, wg, sf)

        if best is None:
            best = (err, sf, lv2, lv3, sign, mant)
        else:
            old_err, old_sf, old_lv2, old_lv3, old_sign, old_mant = best
            take = err < old_err
            best = (
                torch.where(take, err, old_err),
                torch.where(take[..., None, None, None], sf, old_sf),
                torch.where(take[..., None, None, None], lv2, old_lv2),
                torch.where(take[..., None, None, None], lv3, old_lv3),
                torch.where(take[..., None, None, None], sign, old_sign),
                torch.where(take[..., None, None, None], mant, old_mant),
            )

    _, sf, lv2, lv3, sign, mant = best
    out_dtype = torch.bfloat16
    return {
        "scale_factor": sf.to(out_dtype),
        "scale_lv2": lv2.to(out_dtype),
        "scale_lv3": lv3.to(out_dtype),
        "sign": sign.to(out_dtype),
        "mant": mant.to(out_dtype),
    }


def _dequantize_hif4_params(p: dict[str, torch.Tensor]) -> torch.Tensor:
    """Debug helper, not required by the submission interface."""
    y = (
        p["sign"].float()
        * p["mant"].float()
        * p["scale_lv3"].float()
        * p["scale_lv2"].float()
        * p["scale_factor"].float()
    )
    return y.flatten(-4, -1)


# ============================================================
# Exact linear-side equivalent transformations
# ============================================================
def _hadamard64(x: torch.Tensor) -> torch.Tensor:
    """Normalized block-diagonal Hadamard transform on every contiguous 64 channels."""
    if x.shape[-1] % 64 != 0:
        return x
    orig_shape = x.shape
    y = x.float().reshape(*orig_shape[:-1], -1, 64)
    h = 1
    while h < 64:
        y2 = y.reshape(*y.shape[:-1], 64 // (2 * h), 2, h)
        a = y2[..., 0, :]
        b = y2[..., 1, :]
        y = torch.cat((a + b, a - b), dim=-1).reshape(*y.shape[:-1], 64)
        h *= 2
    return (y / math.sqrt(64.0)).reshape(orig_shape)


def _linear_transform_activation(x: torch.Tensor, state: dict[str, Any]) -> torch.Tensor:
    perm = state.get("perm", None)
    if perm is not None:
        perm = perm.to(x.device)
        x = x.index_select(-1, perm)

    s = state.get("smooth_scale", None)
    if s is not None:
        s = s.to(device=x.device, dtype=torch.float32)
        x = x.float() / s

    if state.get("hadamard64", False):
        x = _hadamard64(x)
    return x


def _normalize_importance(v: torch.Tensor) -> torch.Tensor:
    v = v.float().clamp_min(EPS)
    return v / v.mean().clamp_min(EPS)


# ============================================================
# 1. Linear calibration + Weight quantization
# ============================================================
def hif4_calibration_and_quantize_weight(
    weight_quant: torch.Tensor,
    weight_scale: torch.Tensor,
    calib_activation_list: list,
) -> dict[str, Any]:
    W = dequantize_nvfp4(weight_quant, weight_scale).float()
    K = W.shape[-1]

    # First-pass activation statistics in the original channel basis.
    act_absmax = torch.zeros(K, device=W.device, dtype=torch.float32)
    act_sumsq = torch.zeros(K, device=W.device, dtype=torch.float32)
    n_tokens = 0
    for aq, asc in calib_activation_list:
        X = dequantize_nvfp4(aq, asc).to(device=W.device, dtype=torch.float32)
        act_absmax = torch.maximum(act_absmax, X.abs().amax(dim=0))
        act_sumsq += X.square().sum(dim=0)
        n_tokens += X.shape[0]
    act_rms = torch.sqrt(act_sumsq / max(n_tokens, 1) + EPS)

    w_absmax = W.abs().amax(dim=0)
    w_rms = torch.sqrt(W.square().mean(dim=0) + EPS)

    # A fixed channel permutation is mathematically free for X @ W^T
    # if the same permutation is applied to X and W.
    perm = None
    if USE_LINEAR_PERMUTE:
        # Group channels with similar joint dynamic-range difficulty.
        # Weight side is given slightly more emphasis.
        score = 0.70 * torch.log2(w_rms + EPS) + 0.30 * torch.log2(act_rms + EPS)
        perm = torch.argsort(score)
        W = W.index_select(-1, perm)
        act_absmax_p = act_absmax.index_select(0, perm)
        w_absmax_p = w_absmax.index_select(0, perm)
    else:
        act_absmax_p = act_absmax
        w_absmax_p = w_absmax

    smooth_scale = None
    if USE_LINEAR_SMOOTH:
        alpha = LINEAR_SMOOTH_ALPHA
        s = (act_absmax_p.clamp_min(EPS).pow(alpha) /
             w_absmax_p.clamp_min(EPS).pow(1.0 - alpha))
        s = s.clamp(*LINEAR_SMOOTH_CLAMP)
        # Remove an arbitrary global gain; X/s and W*s still exactly cancel.
        s = s / torch.exp(torch.log(s).mean())
        smooth_scale = s
        W = W * s

    if USE_LINEAR_HADAMARD64:
        W = _hadamard64(W)

    # Recompute activation second moment after the exact transform.
    act_t_sumsq = torch.zeros(K, device=W.device, dtype=torch.float32)
    n_tokens = 0
    tmp_state = {
        "perm": None if perm is None else perm.detach().cpu(),
        "smooth_scale": None if smooth_scale is None else smooth_scale.detach().cpu(),
        "hadamard64": bool(USE_LINEAR_HADAMARD64),
    }
    for aq, asc in calib_activation_list:
        X = dequantize_nvfp4(aq, asc).to(device=W.device, dtype=torch.float32)
        Xt = _linear_transform_activation(X, tmp_state)
        act_t_sumsq += Xt.square().sum(dim=0)
        n_tokens += Xt.shape[0]

    # Diagonal-Hessian proxy for Weight error: E[x_j^2].
    weight_error_weight = _normalize_importance(act_t_sumsq / max(n_tokens, 1))
    weight_params = _hif4_quantize(W, error_weight=weight_error_weight)

    # Diagonal sensitivity for Activation error: ||W[:, j]||_2^2.
    act_error_weight = _normalize_importance(W.square().mean(dim=0))

    activation_state = {
        "perm": None if perm is None else perm.detach().cpu(),
        "smooth_scale": None if smooth_scale is None else smooth_scale.detach().cpu(),
        "hadamard64": bool(USE_LINEAR_HADAMARD64),
        "act_error_weight": act_error_weight.detach().cpu(),
    }
    return {
        "weight_params": weight_params,
        "activation_state": activation_state,
    }


# ============================================================
# 2. Dynamic Activation quantization
# ============================================================
def hif4_dynamic_quantize_activation(
    activation_quant: torch.Tensor,
    activation_scale: torch.Tensor,
    activation_state: Any,
) -> dict[str, torch.Tensor]:
    X = dequantize_nvfp4(activation_quant, activation_scale).float()
    state = activation_state or {}
    X = _linear_transform_activation(X, state)
    ew = state.get("act_error_weight", None)
    if ew is not None:
        ew = ew.to(X.device)
    return _hif4_quantize(X, error_weight=ew)


# ============================================================
# Attention helpers
# ============================================================
def _apply_qk_transform(
    x: torch.Tensor,
    num_heads: int,
    head_dim: int,
    state: dict[str, Any],
) -> torch.Tensor:
    y = x.float().reshape(x.shape[0], num_heads, head_dim)
    pre_scale = state.get("pre_scale", None)
    if pre_scale is not None:
        pre_scale = pre_scale.to(device=y.device, dtype=torch.float32).reshape(num_heads, head_dim)
        y = y * pre_scale
    if state.get("hadamard64", False) and head_dim % 64 == 0:
        y = _hadamard64(y)
    return y.reshape(x.shape[0], num_heads * head_dim)


# ============================================================
# 3. Attention calibration
# ============================================================
def hif4_calibration_attention(
    calib_qkv_list: list,
    q_num_heads: int,
    kv_num_heads: int,
    head_dim: int,
) -> dict[str, Any]:
    if q_num_heads % kv_num_heads != 0:
        q_per_kv = None
    else:
        q_per_kv = q_num_heads // kv_num_heads

    device = calib_qkv_list[0]["q"][0].device
    q_abs = torch.zeros(q_num_heads, head_dim, device=device)
    k_abs = torch.zeros(kv_num_heads, head_dim, device=device)

    # Pass 1: calibration-static Q/K smoothing statistics.
    for sample in calib_qkv_list:
        qq, qs = sample["q"]
        kq, ks = sample["k"]
        Q = dequantize_nvfp4(qq, qs).to(device=device, dtype=torch.float32).reshape(-1, q_num_heads, head_dim)
        K = dequantize_nvfp4(kq, ks).to(device=device, dtype=torch.float32).reshape(-1, kv_num_heads, head_dim)
        q_abs = torch.maximum(q_abs, Q.abs().amax(dim=0))
        k_abs = torch.maximum(k_abs, K.abs().amax(dim=0))

    q_pre = torch.ones_like(q_abs)
    k_pre = torch.ones_like(k_abs)
    if USE_QK_SMOOTH and q_per_kv is not None:
        q_abs_g = q_abs.reshape(kv_num_heads, q_per_kv, head_dim).amax(dim=1)
        # q' = q / s, k' = k * s, preserving q·k exactly.
        s = torch.sqrt((q_abs_g + EPS) / (k_abs + EPS)).clamp(*QK_SMOOTH_CLAMP)
        k_pre = s
        q_pre = (1.0 / s).repeat_interleave(q_per_kv, dim=0)

    q_tmp = {
        "pre_scale": q_pre.detach().cpu(),
        "hadamard64": bool(USE_QK_HADAMARD64),
    }
    k_tmp = {
        "pre_scale": k_pre.detach().cpu(),
        "hadamard64": bool(USE_QK_HADAMARD64),
    }

    q_sumsq = torch.zeros(q_num_heads, head_dim, device=device)
    k_sumsq = torch.zeros(kv_num_heads, head_dim, device=device)
    nq = nk = 0

    # Pass 2: error sensitivities in the transformed basis.
    for sample in calib_qkv_list:
        qq, qs = sample["q"]
        kq, ks = sample["k"]
        Q = dequantize_nvfp4(qq, qs).to(device=device, dtype=torch.float32)
        K = dequantize_nvfp4(kq, ks).to(device=device, dtype=torch.float32)
        Qt = _apply_qk_transform(Q, q_num_heads, head_dim, q_tmp).reshape(-1, q_num_heads, head_dim)
        Kt = _apply_qk_transform(K, kv_num_heads, head_dim, k_tmp).reshape(-1, kv_num_heads, head_dim)
        q_sumsq += Qt.square().sum(dim=0)
        k_sumsq += Kt.square().sum(dim=0)
        nq += Qt.shape[0]
        nk += Kt.shape[0]

    q_second = q_sumsq / max(nq, 1)
    k_second = k_sumsq / max(nk, 1)

    if q_per_kv is not None:
        # Q quant error affects QK logits proportionally to K energy.
        q_error_weight = k_second.repeat_interleave(q_per_kv, dim=0).reshape(-1)
        # K is shared by q_per_kv Q heads, so sum their sensitivities.
        k_error_weight = q_second.reshape(kv_num_heads, q_per_kv, head_dim).sum(dim=1).reshape(-1)
    else:
        q_error_weight = torch.ones(q_num_heads * head_dim, device=device)
        k_error_weight = torch.ones(kv_num_heads * head_dim, device=device)

    q_state = {
        "pre_scale": q_pre.detach().cpu(),
        "hadamard64": bool(USE_QK_HADAMARD64),
        "error_weight": _normalize_importance(q_error_weight).detach().cpu(),
    }
    k_state = {
        "pre_scale": k_pre.detach().cpu(),
        "hadamard64": bool(USE_QK_HADAMARD64),
        "error_weight": _normalize_importance(k_error_weight).detach().cpu(),
    }
    v_state = {
        # No P/softmax hook exists in this interface, so start with direct V MSE.
        "error_weight": None,
    }
    return {"q_state": q_state, "k_state": k_state, "v_state": v_state}


# ============================================================
# 4. Dynamic Q quantization
# ============================================================
def hif4_dynamic_quantize_q(
    q_quant: torch.Tensor,
    q_scale: torch.Tensor,
    q_num_heads: int,
    head_dim: int,
    q_state: Any,
) -> dict[str, torch.Tensor]:
    Q = dequantize_nvfp4(q_quant, q_scale).float()
    state = q_state or {}
    Q = _apply_qk_transform(Q, q_num_heads, head_dim, state)
    ew = state.get("error_weight", None)
    if ew is not None:
        ew = ew.to(Q.device)
    return _hif4_quantize(Q, error_weight=ew)


# ============================================================
# 5. Dynamic K quantization
# ============================================================
def hif4_dynamic_quantize_k(
    k_quant: torch.Tensor,
    k_scale: torch.Tensor,
    kv_num_heads: int,
    head_dim: int,
    k_state: Any,
) -> dict[str, torch.Tensor]:
    K = dequantize_nvfp4(k_quant, k_scale).float()
    state = k_state or {}
    K = _apply_qk_transform(K, kv_num_heads, head_dim, state)
    ew = state.get("error_weight", None)
    if ew is not None:
        ew = ew.to(K.device)
    return _hif4_quantize(K, error_weight=ew)


# ============================================================
# 6. Dynamic V quantization
# ============================================================
def hif4_dynamic_quantize_v(
    v_quant: torch.Tensor,
    v_scale: torch.Tensor,
    kv_num_heads: int,
    head_dim: int,
    v_state: Any,
) -> dict[str, torch.Tensor]:
    V = dequantize_nvfp4(v_quant, v_scale).float()
    ew = None if not v_state else v_state.get("error_weight", None)
    if ew is not None:
        ew = ew.to(V.device)
    return _hif4_quantize(V, error_weight=ew)

# ============================================================
# V1: fast output-aware Linear search
# ============================================================
"""HiF4 v1: calibration-selected SmoothQuant scaling.

This module keeps the six public entry points from v0.  The only intentional
algorithmic change is on the Linear path: alpha is selected with an actual
quantized GEMM reconstruction objective instead of being fixed at 0.5.
"""

from typing import Any, Iterable




# None is the no-smoothing ablation.  Keeping it in the search prevents a
# poorly calibrated smooth transform from making a layer worse.
LINEAR_ALPHA_CANDIDATES = (None, 0.0, 0.2, 0.35, 0.5, 0.65, 0.8, 1.0)
SEARCH_MAX_TOKENS = 64
SEARCH_MAX_OUTPUTS = 64
SEARCH_MAX_WEIGHT_ROWS = 64
FAST_SEARCH_INIT_RATIOS = (1.0, 0.9, 0.8)
FAST_SEARCH_ALT_ITERS = 1


# Unchanged public entry points.


def _sample_calibration(
    calib_activation_list: list,
    device: torch.device,
    max_tokens: int = SEARCH_MAX_TOKENS,
) -> torch.Tensor:
    chunks = []
    remaining = max_tokens
    for aq, asc in calib_activation_list:
        if remaining <= 0:
            break
        x = dequantize_nvfp4(aq, asc).to(device=device, dtype=torch.float32)
        x = x.reshape(-1, x.shape[-1])[:remaining]
        chunks.append(x)
        remaining -= x.shape[0]
    if not chunks:
        raise ValueError("calib_activation_list must contain at least one sample")
    return torch.cat(chunks, dim=0)


def _candidate_perm(
    w_rms: torch.Tensor,
    act_rms: torch.Tensor,
) -> torch.Tensor | None:
    if not USE_LINEAR_PERMUTE:
        return None
    score = 0.70 * torch.log2(w_rms + EPS) + 0.30 * torch.log2(act_rms + EPS)
    return torch.argsort(score)


def _smooth_scale(
    act_absmax: torch.Tensor,
    w_absmax: torch.Tensor,
    alpha: float | None,
) -> torch.Tensor | None:
    if alpha is None or not USE_LINEAR_SMOOTH:
        return None
    s = (
        act_absmax.clamp_min(EPS).pow(alpha)
        / w_absmax.clamp_min(EPS).pow(1.0 - alpha)
    )
    s = s.clamp(*LINEAR_SMOOTH_CLAMP)
    return s / torch.exp(torch.log(s).mean())


def _evaluate_linear_candidate(
    W: torch.Tensor,
    X: torch.Tensor,
    perm: torch.Tensor | None,
    alpha: float | None,
    act_absmax: torch.Tensor,
    act_second: torch.Tensor,
    fast_search: bool = False,
) -> tuple[torch.Tensor, dict[str, torch.Tensor], dict[str, Any]]:
    if perm is None:
        Wp, Xp = W, X
        amax, asecond = act_absmax, act_second
    else:
        Wp = W.index_select(-1, perm)
        Xp = X.index_select(-1, perm)
        amax = act_absmax.index_select(0, perm)
        asecond = act_second.index_select(0, perm)

    s = _smooth_scale(amax, Wp.abs().amax(dim=0), alpha)
    if s is not None:
        Wt = Wp * s
        Xt = Xp / s
        asecond = asecond / s.square()
    else:
        Wt, Xt = Wp, Xp

    if USE_LINEAR_HADAMARD64:
        Wt = _hadamard64(Wt)
        Xt = _hadamard64(Xt)
        # Rotation mixes the diagonal statistics; sample statistics are exact
        # for the diagonal proxy and cheap at calibration time.
        asecond = Xt.square().mean(dim=0)

    w_importance = _normalize_importance(asecond)
    quant_kwargs = {}
    if fast_search:
        quant_kwargs = {
            "init_ratios": FAST_SEARCH_INIT_RATIOS,
            "alt_iters": FAST_SEARCH_ALT_ITERS,
        }
    wp = _hif4_quantize(Wt, error_weight=w_importance, **quant_kwargs)
    Wq = _dequantize_hif4_params(wp)

    a_importance = _normalize_importance(Wt.square().mean(dim=0))
    xp = _hif4_quantize(Xt, error_weight=a_importance, **quant_kwargs)
    Xq = _dequantize_hif4_params(xp)

    # Output-space loss directly measures the interaction between W and A
    # quantization.  Restricting output rows bounds calibration cost.
    rows = min(Wt.shape[0], SEARCH_MAX_OUTPUTS)
    ref = Xt @ Wt[:rows].T
    got = Xq @ Wq[:rows].T
    loss = (ref - got).square().mean() / ref.square().mean().clamp_min(EPS)

    state = {
        "perm": None if perm is None else perm.detach().cpu(),
        "smooth_scale": None if s is None else s.detach().cpu(),
        "smooth_alpha": alpha,
        "hadamard64": bool(USE_LINEAR_HADAMARD64),
        "act_error_weight": a_importance.detach().cpu(),
        "calibration_relative_output_mse": float(loss.item()),
    }
    return loss, wp, state


def _calibrate_with_permutations(
    W: torch.Tensor,
    X: torch.Tensor,
    permutations: Iterable[tuple[str, torch.Tensor | None]],
) -> dict[str, Any]:
    act_absmax = X.abs().amax(dim=0)
    act_second = X.square().mean(dim=0)
    act_rms = torch.sqrt(act_second + EPS)
    w_rms = torch.sqrt(W.square().mean(dim=0) + EPS)

    # Candidate ranking uses only a representative subset of output rows and
    # a shortened HiF4 scale search.  The selected transform is quantized once
    # with the full Weight and the normal high-quality quantizer below.
    W_search = W[: min(W.shape[0], SEARCH_MAX_WEIGHT_ROWS)]
    best = None
    for perm_name, perm in permutations:
        for alpha in LINEAR_ALPHA_CANDIDATES:
            loss, _, state = _evaluate_linear_candidate(
                W_search, X, perm, alpha, act_absmax, act_second,
                fast_search=True,
            )
            if best is None or bool(loss < best[0]):
                best = (loss, perm_name, perm, alpha)
    assert best is not None
    _, perm_name, perm, alpha = best
    _, params, state = _evaluate_linear_candidate(
        W, X, perm, alpha, act_absmax, act_second, fast_search=False
    )
    state["permutation_strategy"] = perm_name
    state["search_relative_output_mse"] = float(best[0].item())
    return {"weight_params": params, "activation_state": state}


def hif4_calibration_and_quantize_weight(
    weight_quant: torch.Tensor,
    weight_scale: torch.Tensor,
    calib_activation_list: list,
) -> dict[str, Any]:
    W = dequantize_nvfp4(weight_quant, weight_scale).float()
    X = _sample_calibration(calib_activation_list, W.device)
    w_rms = torch.sqrt(W.square().mean(dim=0) + EPS)
    act_rms = torch.sqrt(X.square().mean(dim=0) + EPS)
    perm = _candidate_perm(w_rms, act_rms)
    return _calibrate_with_permutations(W, X, (("magnitude_sort", perm),))

# Dynamic tensors are latency-sensitive.  The full Weight path above keeps the
# 7-start search, while A/Q/K/V use 3 starts and one alternating update.
DYNAMIC_INIT_RATIOS = (1.0, 0.9, 0.8)
DYNAMIC_ALT_ITERS = 1


def _hif4_quantize_dynamic(
    x: torch.Tensor,
    error_weight: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    return _hif4_quantize(
        x,
        error_weight=error_weight,
        init_ratios=DYNAMIC_INIT_RATIOS,
        alt_iters=DYNAMIC_ALT_ITERS,
    )


def hif4_dynamic_quantize_activation(
    activation_quant: torch.Tensor,
    activation_scale: torch.Tensor,
    activation_state: Any,
) -> dict[str, torch.Tensor]:
    X = dequantize_nvfp4(activation_quant, activation_scale).float()
    state = activation_state or {}
    X = _linear_transform_activation(X, state)
    ew = state.get("act_error_weight", None)
    if ew is not None:
        ew = ew.to(X.device)
    return _hif4_quantize_dynamic(X, ew)


def hif4_dynamic_quantize_q(
    q_quant: torch.Tensor,
    q_scale: torch.Tensor,
    q_num_heads: int,
    head_dim: int,
    q_state: Any,
) -> dict[str, torch.Tensor]:
    Q = dequantize_nvfp4(q_quant, q_scale).float()
    state = q_state or {}
    Q = _apply_qk_transform(Q, q_num_heads, head_dim, state)
    ew = state.get("error_weight", None)
    if ew is not None:
        ew = ew.to(Q.device)
    return _hif4_quantize_dynamic(Q, ew)


def hif4_dynamic_quantize_k(
    k_quant: torch.Tensor,
    k_scale: torch.Tensor,
    kv_num_heads: int,
    head_dim: int,
    k_state: Any,
) -> dict[str, torch.Tensor]:
    K = dequantize_nvfp4(k_quant, k_scale).float()
    state = k_state or {}
    K = _apply_qk_transform(K, kv_num_heads, head_dim, state)
    ew = state.get("error_weight", None)
    if ew is not None:
        ew = ew.to(K.device)
    return _hif4_quantize_dynamic(K, ew)


def hif4_dynamic_quantize_v(
    v_quant: torch.Tensor,
    v_scale: torch.Tensor,
    kv_num_heads: int,
    head_dim: int,
    v_state: Any,
) -> dict[str, torch.Tensor]:
    V = dequantize_nvfp4(v_quant, v_scale).float()
    ew = None if not v_state else v_state.get("error_weight", None)
    if ew is not None:
        ew = ew.to(V.device)
    return _hif4_quantize_dynamic(V, ew)



# ============================================================
# V2: permutation strategy search
# ============================================================
"""HiF4 v2: select channel regrouping with real quantized output loss.

In addition to v1's alpha search, v2 compares identity, magnitude sorting and
zigzag balancing.  The winner is selected after quantizing both calibration
activations and weights, not from an RMS/P99 proxy.
"""

from typing import Any






def _zigzag_permutation(order: torch.Tensor) -> torch.Tensor:
    """Distribute alternating low/high-score channels across 64-wide groups."""
    k = order.numel()
    if k % 64 != 0:
        return order
    groups = k // 64
    lo, hi = 0, k - 1
    table = torch.empty(groups, 64, dtype=order.dtype, device=order.device)
    # Fill the same within-group position across groups. Reversing every other
    # lane prevents one group from consistently receiving the extremes first.
    for lane in range(64):
        vals = []
        for _ in range(groups):
            if lane % 2 == 0:
                vals.append(order[hi])
                hi -= 1
            else:
                vals.append(order[lo])
                lo += 1
        lane_values = torch.stack(vals)
        if lane % 4 >= 2:
            lane_values = lane_values.flip(0)
        table[:, lane] = lane_values
    return table.reshape(-1)


def hif4_calibration_and_quantize_weight(
    weight_quant: torch.Tensor,
    weight_scale: torch.Tensor,
    calib_activation_list: list,
) -> dict[str, Any]:
    W = dequantize_nvfp4(weight_quant, weight_scale).float()
    X = _sample_calibration(calib_activation_list, W.device)
    score = (
        0.70 * torch.log2(torch.sqrt(W.square().mean(dim=0) + EPS))
        + 0.30 * torch.log2(torch.sqrt(X.square().mean(dim=0) + EPS))
    )
    magnitude = torch.argsort(score)
    if USE_LINEAR_PERMUTE:
        permutations = (
            ("identity", None),
            ("magnitude_sort", magnitude),
            ("zigzag_balance", _zigzag_permutation(magnitude)),
        )
    else:
        permutations = (("identity", None),)
    return _calibrate_with_permutations(W, X, permutations)
