"""ZeroQuant, adapted only to the six-function HiF4 demo interface.

Offline 64-group Weight quantization and online per-token 64-group Activation quantization.
No techniques from the other solution variants are included.

======================== Linear ========================
[Linear][Group 0] calibration: PASSED [152.53ms] (W=(8192, 2048), num_calib=5)
[Linear][Group 0][Test 0] activation: FAILED [2.91ms] (W=(8192, 2048), A=(10, 2048))
      MatMul MSE 2.1228e-02 exceeds threshold 0.001
[Linear][Group 0][Test 1] activation: FAILED [5.27ms] (W=(8192, 2048), A=(128, 2048))
      MatMul MSE 1.6638e-02 exceeds threshold 0.001
[Linear][Group 0][Test 2] activation: FAILED [43.64ms] (W=(8192, 2048), A=(512, 2048))
      MatMul MSE 1.6812e-02 exceeds threshold 0.001
[Linear][Group 0][Test 3] activation: FAILED [10.92ms] (W=(8192, 2048), A=(1024, 2048))
      MatMul MSE 1.5657e-02 exceeds threshold 0.001
[Linear][Group 0][Test 4] activation: FAILED [9.99ms] (W=(8192, 2048), A=(1024, 2048))
      MatMul MSE 1.6793e-02 exceeds threshold 0.001

====================== Attention ======================
[Attention][Group 0] calibration: PASSED [0.00ms] (q_heads=16, kv_heads=2, head_dim=256, num_calib=5)
[Attention][Group 0][Test 0] FAILED [7.03ms] (Q=(10, 4096), K=(10, 512), V=(10, 512))
      Attention MSE 1.0388e-03 exceeds threshold 0.001
[Attention][Group 0][Test 1] PASSED (MSE=4.0943e-04) [24.57ms] (Q=(128, 4096), K=(128, 512), V=(128, 512))
[Attention][Group 0][Test 2] PASSED (MSE=2.8481e-04) [48.82ms] (Q=(512, 4096), K=(512, 512), V=(512, 512))
[Attention][Group 0][Test 3] PASSED (MSE=2.1662e-04) [133.99ms] (Q=(1024, 4096), K=(1024, 512), V=(1024, 512))
[Attention][Group 0][Test 4] PASSED (MSE=2.3384e-04) [83.52ms] (Q=(1024, 4096), K=(1024, 512), V=(1024, 512))
"""
from __future__ import annotations

from typing import Any

import math
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

# ZeroQuant: offline group-wise W and online per-token group-wise A.
def hif4_calibration_and_quantize_weight(
    weight_quant: torch.Tensor,
    weight_scale: torch.Tensor,
    calib_activation_list: list,
) -> dict[str, Any]:
    W = dequantize_nvfp4(weight_quant, weight_scale).float()
    return {"weight_params": _hif4_direct(W), "activation_state": None}


def hif4_dynamic_quantize_activation(
    activation_quant: torch.Tensor,
    activation_scale: torch.Tensor,
    activation_state: Any,
) -> dict[str, torch.Tensor]:
    A = dequantize_nvfp4(activation_quant, activation_scale).float()
    return _hif4_direct(A)


# Attention is not changed by this paper-specific Linear ablation.
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

