"""HiF4 permutation: six alpha candidates.

Selects six joint second-moment sorts with real activation-weighted HiF4 loss.
Only Linear channel permutation and direct HiF4 conversion are used.  There is
no smooth/AWQ transform and no HiF4 scale-factor search.

======================== Linear ========================
[Linear][Group 0] calibration: PASSED [1326.42ms] (W=(8192, 2048), num_calib=5)
[Linear][Group 0][Test 0] activation: FAILED [15.87ms] (W=(8192, 2048), A=(10, 2048))
      MatMul MSE 4.4571e-03 exceeds threshold 0.001
[Linear][Group 0][Test 1] activation: FAILED [19.36ms] (W=(8192, 2048), A=(128, 2048))
      MatMul MSE 4.2742e-03 exceeds threshold 0.001
[Linear][Group 0][Test 2] activation: FAILED [107.49ms] (W=(8192, 2048), A=(512, 2048))
      MatMul MSE 3.6569e-03 exceeds threshold 0.001
[Linear][Group 0][Test 3] activation: FAILED [90.49ms] (W=(8192, 2048), A=(1024, 2048))
      MatMul MSE 3.5108e-03 exceeds threshold 0.001
[Linear][Group 0][Test 4] activation: FAILED [85.43ms] (W=(8192, 2048), A=(1024, 2048))
      MatMul MSE 3.3962e-03 exceeds threshold 0.001

====================== Attention ======================
[Attention][Group 0] calibration: PASSED [0.01ms] (q_heads=16, kv_heads=2, head_dim=256, num_calib=5)
[Attention][Group 0][Test 0] PASSED (MSE=8.4295e-04) [337.67ms] (Q=(10, 4096), K=(10, 512), V=(10, 512))
[Attention][Group 0][Test 1] PASSED (MSE=2.6386e-04) [597.25ms] (Q=(128, 4096), K=(128, 512), V=(128, 512))
[Attention][Group 0][Test 2] PASSED (MSE=1.5081e-04) [937.96ms] (Q=(512, 4096), K=(512, 512), V=(512, 512))
[Attention][Group 0][Test 3] PASSED (MSE=1.2206e-04) [1464.30ms] (Q=(1024, 4096), K=(1024, 512), V=(1024, 512))
[Attention][Group 0][Test 4] PASSED (MSE=1.2945e-04) [1448.49ms] (Q=(1024, 4096), K=(1024, 512), V=(1024, 512))

"""
from __future__ import annotations

from typing import Any

import torch


EPS = 1e-12


# =============================================================================
# NVFP4 input decode
# =============================================================================
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


# =============================================================================
# Fast direct HiF4 conversion
# =============================================================================
def _round_to_e6m2(x: torch.Tensor) -> torch.Tensor:
    """Round positive FP32 values to finite unsigned E6M2 values."""
    x = x.float().clamp(min=2.0**-48, max=(2.0**15) * 1.5)
    exponent = torch.floor(torch.log2(x)).clamp(-48.0, 15.0)
    base = torch.pow(torch.tensor(2.0, device=x.device), exponent)
    mantissa = torch.round((x / base - 1.0) * 4.0)

    carry = mantissa >= 4.0
    exponent = torch.where(carry, exponent + 1.0, exponent)
    mantissa = torch.where(carry, torch.zeros_like(mantissa), mantissa)

    overflow = exponent > 15.0
    exponent = torch.where(overflow, torch.full_like(exponent, 15.0), exponent)
    mantissa = torch.where(overflow, torch.full_like(mantissa, 2.0), mantissa)
    # Exponent 15, mantissa code 3 is reserved; saturate to the largest finite.
    mantissa = torch.where(
        (exponent >= 15.0) & (mantissa > 2.0),
        torch.full_like(mantissa, 2.0),
        mantissa,
    ).clamp(0.0, 3.0)
    return torch.pow(torch.tensor(2.0, device=x.device), exponent) * (
        1.0 + mantissa * 0.25
    )


def _hif4_direct(x: torch.Tensor) -> dict[str, torch.Tensor]:
    """Convert contiguous groups of 64 with paper-style peak thresholds.

    The operation is fully vectorized and performs no candidate or iterative
    search.  A 64-value group is viewed as 8 x 2 x 4.
    """
    if x.shape[-1] % 64 != 0:
        raise ValueError(f"HiF4 requires last dim divisible by 64, got {x.shape[-1]}")

    xf = x.float()
    groups = x.shape[-1] // 64
    xg = xf.unflatten(-1, (groups, 8, 2, 4))
    ax = xg.abs()

    peak64 = ax.amax(dim=(-1, -2, -3), keepdim=True)
    scale_factor = _round_to_e6m2(
        (peak64 / 7.0).clamp_min(2.0**-48)
    )

    # One lv2 exponent is shared by 8 values.  lv2=2 is only needed when the
    # local peak exceeds the range normally handled by the lv3 exponent.
    peak8 = ax.amax(dim=(-1, -2), keepdim=True)
    scale_lv2 = torch.where(
        peak8 > 4.0 * scale_factor,
        torch.full_like(peak8, 2.0),
        torch.ones_like(peak8),
    )

    # One lv3 exponent is shared by 4 values.  Account for the already chosen
    # lv2 multiplier before applying the second threshold.
    peak4 = ax.amax(dim=-1, keepdim=True)
    scale_lv3 = torch.where(
        peak4 > 2.0 * scale_factor * scale_lv2,
        torch.full_like(peak4, 2.0),
        torch.ones_like(peak4),
    )

    local_scale = scale_factor * scale_lv2 * scale_lv3
    mant = (torch.round((ax / local_scale.clamp_min(EPS)) * 4.0) * 0.25).clamp(
        0.0, 1.75
    )
    sign = torch.where(mant > 0.0, torch.sign(xg), torch.zeros_like(xg))

    out_dtype = torch.bfloat16
    return {
        "scale_factor": scale_factor.to(out_dtype),
        "scale_lv2": scale_lv2.to(out_dtype),
        "scale_lv3": scale_lv3.to(out_dtype),
        "sign": sign.to(out_dtype),
        "mant": mant.to(out_dtype),
    }

# =============================================================================
# Permutation statistics and loss
# =============================================================================
def _second_moments(
    W: torch.Tensor,
    calib_activation_list: list,
) -> tuple[torch.Tensor, torch.Tensor]:
    channels = W.shape[-1]
    w2 = W.float().square().mean(dim=0).clamp_min(EPS)
    act_sumsq = torch.zeros(channels, dtype=torch.float32, device=W.device)
    count = 0
    for activation_quant, activation_scale in calib_activation_list:
        X = dequantize_nvfp4(activation_quant, activation_scale).to(
            device=W.device, dtype=torch.float32
        ).reshape(-1, channels)
        act_sumsq += X.square().sum(dim=0)
        count += X.shape[0]
    if count == 0:
        a2 = torch.ones_like(w2)
    else:
        a2 = (act_sumsq / count).clamp_min(EPS)
    return w2, a2


def _dequantize_hif4(params: dict[str, torch.Tensor]) -> torch.Tensor:
    x = (
        params["scale_factor"].float()
        * params["scale_lv2"].float()
        * params["scale_lv3"].float()
        * params["sign"].float()
        * params["mant"].float()
    )
    return x.flatten(-4, -1)


def _weighted_losses(
    Wp: torch.Tensor,
    a2p: torch.Tensor,
) -> tuple[dict[str, torch.Tensor], torch.Tensor, torch.Tensor]:
    params = _hif4_direct(Wp)
    Wq = _dequantize_hif4(params)
    # Keep the raw activation second moment so full-tensor and two-group local
    # evaluations use exactly the same additive objective.
    importance = a2p
    column_loss = importance * (Wp - Wq).square().mean(dim=0)
    group_loss = column_loss.reshape(-1, 64).sum(dim=-1)
    return params, group_loss, column_loss


def _score(w2: torch.Tensor, a2: torch.Tensor, alpha: float) -> torch.Tensor:
    # Log-domain form of a2**alpha * w2**(1-alpha), avoiding underflow.
    return alpha * torch.log(a2) + (1.0 - alpha) * torch.log(w2)


def _sorted_perm(w2: torch.Tensor, a2: torch.Tensor, alpha: float) -> torch.Tensor:
    return torch.argsort(_score(w2, a2, alpha), descending=True)


def _best_alpha_permutation(
    W: torch.Tensor,
    w2: torch.Tensor,
    a2: torch.Tensor,
) -> tuple[torch.Tensor, float | None, torch.Tensor]:
    # Identity is a safety candidate: permutation is accepted only if its real
    # activation-weighted HiF4 reconstruction loss is lower.
    identity = torch.arange(W.shape[-1], device=W.device)
    _, group_loss, _ = _weighted_losses(W, a2)
    best_loss = group_loss.sum()
    best_perm = identity
    best_alpha = None

    for alpha in (0.0, 0.2, 0.4, 0.6, 0.8, 1.0):
        perm = _sorted_perm(w2, a2, alpha)
        _, candidate_group_loss, _ = _weighted_losses(
            W.index_select(-1, perm), a2.index_select(0, perm)
        )
        loss = candidate_group_loss.sum()
        if bool(loss < best_loss):
            best_loss = loss
            best_perm = perm
            best_alpha = alpha
    return best_perm, best_alpha, best_loss


def _state(
    perm: torch.Tensor,
    strategy: str,
    alpha: float | None,
    weighted_loss: torch.Tensor,
) -> dict[str, Any]:
    return {
        "perm": perm.detach().cpu(),
        "strategy": strategy,
        "alpha": alpha,
        "weighted_hif4_loss": float(weighted_loss.item()),
    }


# =============================================================================
# Dynamic Linear Activation
# =============================================================================
def hif4_dynamic_quantize_activation(
    activation_quant: torch.Tensor,
    activation_scale: torch.Tensor,
    activation_state: Any,
) -> dict[str, torch.Tensor]:
    X = dequantize_nvfp4(activation_quant, activation_scale).float()
    state = activation_state or {}
    perm = state.get("perm", None)
    if perm is not None:
        X = X.index_select(-1, perm.to(X.device))
    return _hif4_direct(X)


# =============================================================================
# Attention: direct HiF4, no permutation or calibration search
# =============================================================================
def hif4_calibration_attention(
    calib_qkv_list: list,
    q_num_heads: int,
    kv_num_heads: int,
    head_dim: int,
) -> dict[str, Any]:
    return {"q_state": None, "k_state": None, "v_state": None}


def hif4_dynamic_quantize_q(
    q_quant: torch.Tensor,
    q_scale: torch.Tensor,
    q_num_heads: int,
    head_dim: int,
    q_state: Any,
) -> dict[str, torch.Tensor]:
    return _hif4_direct(dequantize_nvfp4(q_quant, q_scale).float())


def hif4_dynamic_quantize_k(
    k_quant: torch.Tensor,
    k_scale: torch.Tensor,
    kv_num_heads: int,
    head_dim: int,
    k_state: Any,
) -> dict[str, torch.Tensor]:
    return _hif4_direct(dequantize_nvfp4(k_quant, k_scale).float())


def hif4_dynamic_quantize_v(
    v_quant: torch.Tensor,
    v_scale: torch.Tensor,
    kv_num_heads: int,
    head_dim: int,
    v_state: Any,
) -> dict[str, torch.Tensor]:
    return _hif4_direct(dequantize_nvfp4(v_quant, v_scale).float())


# =============================================================================
# Linear calibration: six alpha candidates + identity fallback
# =============================================================================
def hif4_calibration_and_quantize_weight(
    weight_quant: torch.Tensor,
    weight_scale: torch.Tensor,
    calib_activation_list: list,
) -> dict[str, Any]:
    W = dequantize_nvfp4(weight_quant, weight_scale).float()
    w2, a2 = _second_moments(W, calib_activation_list)
    perm, alpha, best_loss = _best_alpha_permutation(W, w2, a2)
    params = _hif4_direct(W.index_select(-1, perm))
    strategy = "identity" if alpha is None else "six_alpha_second_moment_sort"
    return {
        "weight_params": params,
        "activation_state": _state(perm, strategy, alpha, best_loss),
    }
