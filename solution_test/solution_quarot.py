"""QuaRot, adapted only to the six-function HiF4 demo interface.

A fixed signed Hadamard64 rotation is applied equivalently to Linear W/A and Attention Q/K.
No techniques from the other solution variants are included.

======================== Linear ========================
[Linear][Group 0] calibration: PASSED [181.41ms] (W=(8192, 2048), num_calib=5)
[Linear][Group 0][Test 0] activation: FAILED [4.07ms] (W=(8192, 2048), A=(10, 2048))
      MatMul MSE 1.0345e-02 exceeds threshold 0.001
[Linear][Group 0][Test 1] activation: FAILED [6.88ms] (W=(8192, 2048), A=(128, 2048))
      MatMul MSE 9.6093e-03 exceeds threshold 0.001
[Linear][Group 0][Test 2] activation: FAILED [49.88ms] (W=(8192, 2048), A=(512, 2048))
      MatMul MSE 8.9642e-03 exceeds threshold 0.001
[Linear][Group 0][Test 3] activation: FAILED [18.13ms] (W=(8192, 2048), A=(1024, 2048))
      MatMul MSE 8.7056e-03 exceeds threshold 0.001
[Linear][Group 0][Test 4] activation: FAILED [19.18ms] (W=(8192, 2048), A=(1024, 2048))
      MatMul MSE 8.6983e-03 exceeds threshold 0.001

====================== Attention ======================
[Attention][Group 0] calibration: PASSED [0.00ms] (q_heads=16, kv_heads=2, head_dim=256, num_calib=5)
[Attention][Group 0][Test 0] FAILED [11.65ms] (Q=(10, 4096), K=(10, 512), V=(10, 512))
      Attention MSE 1.1494e-03 exceeds threshold 0.001
[Attention][Group 0][Test 1] PASSED (MSE=3.2892e-04) [34.63ms] (Q=(128, 4096), K=(128, 512), V=(128, 512))
[Attention][Group 0][Test 2] PASSED (MSE=1.9756e-04) [142.79ms] (Q=(512, 4096), K=(512, 512), V=(512, 512))
[Attention][Group 0][Test 3] PASSED (MSE=1.7040e-04) [137.48ms] (Q=(1024, 4096), K=(1024, 512), V=(1024, 512))
[Attention][Group 0][Test 4] PASSED (MSE=1.7489e-04) [136.19ms] (Q=(1024, 4096), K=(1024, 512), V=(1024, 512))

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

def _hadamard64(x: torch.Tensor) -> torch.Tensor:
    if x.shape[-1] % 64 != 0:
        raise ValueError("Hadamard64 requires the last dimension divisible by 64")
    shape = x.shape
    y = x.float().reshape(*shape[:-1], -1, 64)
    width = 1
    while width < 64:
        z = y.reshape(*y.shape[:-1], 64 // (2 * width), 2, width)
        left, right = z[..., 0, :], z[..., 1, :]
        y = torch.cat((left + right, left - right), dim=-1).reshape(*y.shape[:-1], 64)
        width *= 2
    return (y / 8.0).reshape(shape)


def _quarot(x: torch.Tensor) -> torch.Tensor:
    # Reproducible Rademacher-like diagonal followed by normalized Hadamard.
    # The same fixed orthogonal transform is applied to both GEMM operands.
    index = torch.arange(x.shape[-1], device=x.device)
    signs = torch.where(
        ((index * 13 + 7) % 17) < 8,
        torch.full_like(index, -1, dtype=torch.float32),
        torch.ones_like(index, dtype=torch.float32),
    )
    return _hadamard64(x.float() * signs)



def hif4_calibration_and_quantize_weight(
    weight_quant: torch.Tensor,
    weight_scale: torch.Tensor,
    calib_activation_list: list,
) -> dict[str, Any]:
    W = dequantize_nvfp4(weight_quant, weight_scale).float()
    return {
        "weight_params": _hif4_direct(_quarot(W)),
        "activation_state": {"quarot": True},
    }


def hif4_dynamic_quantize_activation(
    activation_quant: torch.Tensor,
    activation_scale: torch.Tensor,
    activation_state: Any,
) -> dict[str, torch.Tensor]:
    A = dequantize_nvfp4(activation_quant, activation_scale).float()
    return _hif4_direct(_quarot(A))


def hif4_calibration_attention(
    calib_qkv_list: list,
    q_num_heads: int,
    kv_num_heads: int,
    head_dim: int,
) -> dict[str, Any]:
    enabled = head_dim % 64 == 0
    return {
        "q_state": {"quarot": enabled},
        "k_state": {"quarot": enabled},
        "v_state": None,
    }


def _rotate_attention(x: torch.Tensor, heads: int, head_dim: int, state: Any) -> torch.Tensor:
    if not state or not state.get("quarot", False):
        return x.float()
    shape = x.shape
    y = x.float().reshape(shape[0], heads, head_dim)
    return _quarot(y).reshape(shape)


def hif4_dynamic_quantize_q(q_quant, q_scale, q_num_heads, head_dim, q_state):
    Q = dequantize_nvfp4(q_quant, q_scale)
    return _hif4_direct(_rotate_attention(Q, q_num_heads, head_dim, q_state))


def hif4_dynamic_quantize_k(k_quant, k_scale, kv_num_heads, head_dim, k_state):
    K = dequantize_nvfp4(k_quant, k_scale)
    return _hif4_direct(_rotate_attention(K, kv_num_heads, head_dim, k_state))


def hif4_dynamic_quantize_v(v_quant, v_scale, kv_num_heads, head_dim, v_state):
    return _hif4_direct(dequantize_nvfp4(v_quant, v_scale).float())
