"""HiF4 v3: v2 Linear search plus Q/K joint Smooth-QK calibration.

The attention smooth exponent is selected by quantized QK-logit error.  This
is safer than always using the equalization exponent 0.5 and preserves the
full-precision QK product exactly for every candidate.
"""
from __future__ import annotations

from typing import Any

import torch

import hif4_solution_v0 as _v0
import hif4_solution_v2 as _v2


QK_BETA_CANDIDATES = (None, 0.25, 0.5, 0.75, 1.0)
QK_SEARCH_MAX_TOKENS = 128


dequantize_nvfp4 = _v0.dequantize_nvfp4
hif4_calibration_and_quantize_weight = _v2.hif4_calibration_and_quantize_weight
hif4_dynamic_quantize_activation = _v0.hif4_dynamic_quantize_activation
hif4_dynamic_quantize_q = _v0.hif4_dynamic_quantize_q
hif4_dynamic_quantize_k = _v0.hif4_dynamic_quantize_k
hif4_dynamic_quantize_v = _v0.hif4_dynamic_quantize_v


def _collect_qk(
    calib_qkv_list: list,
    q_num_heads: int,
    kv_num_heads: int,
    head_dim: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    qs, ks = [], []
    remaining = QK_SEARCH_MAX_TOKENS
    for sample in calib_qkv_list:
        if remaining <= 0:
            break
        qq, qscale = sample["q"]
        kq, kscale = sample["k"]
        q = dequantize_nvfp4(qq, qscale).to(device=device, dtype=torch.float32)
        k = dequantize_nvfp4(kq, kscale).to(device=device, dtype=torch.float32)
        take = min(remaining, q.shape[0], k.shape[0])
        qs.append(q[:take].reshape(take, q_num_heads, head_dim))
        ks.append(k[:take].reshape(take, kv_num_heads, head_dim))
        remaining -= take
    if not qs:
        raise ValueError("calib_qkv_list must contain at least one sample")
    return torch.cat(qs), torch.cat(ks)


def _qk_candidate(
    Q: torch.Tensor,
    K: torch.Tensor,
    q_per_kv: int,
    beta: float | None,
) -> tuple[torch.Tensor, dict[str, Any], dict[str, Any]]:
    kv_heads, head_dim = K.shape[1:]
    q_abs = Q.abs().amax(dim=0).reshape(kv_heads, q_per_kv, head_dim).amax(dim=1)
    k_abs = K.abs().amax(dim=0)
    if beta is None or not _v0.USE_QK_SMOOTH:
        s = torch.ones_like(k_abs)
    else:
        # beta=0.5 is the v0 equalization rule.  Other candidates let the
        # calibration objective decide how aggressively to transfer outliers.
        ratio = (q_abs + _v0.EPS) / (k_abs + _v0.EPS)
        s = ratio.pow(beta).clamp(*_v0.QK_SMOOTH_CLAMP)

    q_pre = (1.0 / s).repeat_interleave(q_per_kv, dim=0)
    k_pre = s
    Qt = Q * q_pre.unsqueeze(0)
    Kt = K * k_pre.unsqueeze(0)
    if _v0.USE_QK_HADAMARD64:
        Qt = _v0._hadamard64(Qt)
        Kt = _v0._hadamard64(Kt)

    q_second = Qt.square().mean(dim=0)
    k_second = Kt.square().mean(dim=0)
    qw = _v0._normalize_importance(
        k_second.repeat_interleave(q_per_kv, dim=0).reshape(-1)
    )
    kw = _v0._normalize_importance(
        q_second.reshape(kv_heads, q_per_kv, head_dim).sum(dim=1).reshape(-1)
    )
    Qflat, Kflat = Qt.flatten(1), Kt.flatten(1)
    Qq = _v0._dequantize_hif4_params(_v0._hif4_quantize(Qflat, qw)).reshape_as(Qt)
    Kq = _v0._dequantize_hif4_params(_v0._hif4_quantize(Kflat, kw)).reshape_as(Kt)

    # Compare the full calibration-token QK matrix per KV head and its
    # associated Q heads.  This captures cross-token interactions as well as
    # the diagonal, while the token cap keeps memory bounded.
    qg = Qt.reshape(Qt.shape[0], kv_heads, q_per_kv, head_dim)
    qqg = Qq.reshape_as(qg)
    ref = torch.einsum("thgd,shd->hgts", qg, Kt)
    got = torch.einsum("thgd,shd->hgts", qqg, Kq)
    loss = (ref - got).square().mean() / ref.square().mean().clamp_min(_v0.EPS)
    q_state = {
        "pre_scale": q_pre.detach().cpu(),
        "hadamard64": bool(_v0.USE_QK_HADAMARD64),
        "error_weight": qw.detach().cpu(),
        "smooth_beta": beta,
    }
    k_state = {
        "pre_scale": k_pre.detach().cpu(),
        "hadamard64": bool(_v0.USE_QK_HADAMARD64),
        "error_weight": kw.detach().cpu(),
        "smooth_beta": beta,
        "calibration_relative_qk_mse": float(loss.item()),
    }
    return loss, q_state, k_state


def hif4_calibration_attention(
    calib_qkv_list: list,
    q_num_heads: int,
    kv_num_heads: int,
    head_dim: int,
) -> dict[str, Any]:
    if q_num_heads % kv_num_heads != 0:
        return _v0.hif4_calibration_attention(
            calib_qkv_list, q_num_heads, kv_num_heads, head_dim
        )
    device = calib_qkv_list[0]["q"][0].device
    Q, K = _collect_qk(
        calib_qkv_list, q_num_heads, kv_num_heads, head_dim, device
    )
    q_per_kv = q_num_heads // kv_num_heads
    best = None
    for beta in QK_BETA_CANDIDATES:
        candidate = _qk_candidate(Q, K, q_per_kv, beta)
        if best is None or bool(candidate[0] < best[0]):
            best = candidate
    assert best is not None
    return {
        "q_state": best[1],
        "k_state": best[2],
        "v_state": {"error_weight": None},
    }
