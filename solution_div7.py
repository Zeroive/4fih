"""Standard HiF4 quantization using only absmax / 7 per 64 values.

This standalone solution contains no scale iteration, calibration search,
smoothing, permutation, or rotation.
"""

from __future__ import annotations

from typing import Any

import torch


EPS = 1e-12


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
    return (x * scale_float.unsqueeze(-1)).flatten(-2, -1).to(torch.bfloat16)


def _round_to_e6m2(x: torch.Tensor) -> torch.Tensor:
    """Round positive values to the finite E6M2 scale grid."""
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
    mantissa = torch.where(
        (exponent >= 15.0) & (mantissa > 2.0),
        torch.full_like(mantissa, 2.0),
        mantissa,
    ).clamp(0.0, 3.0)

    return torch.pow(torch.tensor(2.0, device=x.device), exponent) * (
        1.0 + 0.25 * mantissa
    )


def _quantize_div7(x: torch.Tensor) -> dict[str, torch.Tensor]:
    """Quantize each independent 64-value group with absmax / 7."""
    if x.shape[-1] % 64 != 0:
        raise ValueError(
            f"HiF4 requires last dimension divisible by 64, got {x.shape[-1]}"
        )

    grouped = x.float().unflatten(-1, (x.shape[-1] // 64, 8, 2, 4))
    absolute = grouped.abs()

    # HiF4's largest normalized value is 1.75 * 2 * 2 = 7.
    absmax64 = absolute.amax(dim=(-1, -2, -3), keepdim=True)
    scale_factor = _round_to_e6m2(
        (absmax64 / 7.0).clamp_min(2.0**-48)
    )

    # One lv2 value is shared by each group of 8 values.
    absmax8 = absolute.amax(dim=(-1, -2), keepdim=True)
    scale_lv2 = torch.where(
        absmax8 > 4.0 * scale_factor,
        torch.full_like(absmax8, 2.0),
        torch.ones_like(absmax8),
    )

    # One lv3 value is shared by each nested group of 4 values.
    absmax4 = absolute.amax(dim=-1, keepdim=True)
    scale_lv3 = torch.where(
        absmax4 > 2.0 * scale_factor * scale_lv2,
        torch.full_like(absmax4, 2.0),
        torch.ones_like(absmax4),
    )

    level = (scale_factor * scale_lv2 * scale_lv3).clamp_min(EPS)
    mantissa = (torch.round(absolute / level * 4.0) * 0.25).clamp(0.0, 1.75)
    sign = torch.where(
        mantissa > 0.0,
        torch.sign(grouped),
        torch.zeros_like(grouped),
    )

    return {
        "scale_factor": scale_factor.to(torch.bfloat16),
        "scale_lv2": scale_lv2.to(torch.bfloat16),
        "scale_lv3": scale_lv3.to(torch.bfloat16),
        "sign": sign.to(torch.bfloat16),
        "mant": mantissa.to(torch.bfloat16),
    }


def hif4_calibration_and_quantize_weight(
    weight_quant: torch.Tensor,
    weight_scale: torch.Tensor,
    calib_activation_list: list,
) -> dict[str, Any]:
    del calib_activation_list
    weight = dequantize_nvfp4(weight_quant, weight_scale).float()
    return {
        "weight_params": _quantize_div7(weight),
        "activation_state": None,
    }


def hif4_dynamic_quantize_activation(
    activation_quant: torch.Tensor,
    activation_scale: torch.Tensor,
    activation_state: Any,
) -> dict[str, torch.Tensor]:
    del activation_state
    activation = dequantize_nvfp4(activation_quant, activation_scale).float()
    return _quantize_div7(activation)


def hif4_calibration_attention(
    calib_qkv_list: list,
    q_num_heads: int,
    kv_num_heads: int,
    head_dim: int,
) -> dict[str, Any]:
    del calib_qkv_list, q_num_heads, kv_num_heads, head_dim
    return {"q_state": None, "k_state": None, "v_state": None}


def hif4_dynamic_quantize_q(
    q_quant: torch.Tensor,
    q_scale: torch.Tensor,
    q_num_heads: int,
    head_dim: int,
    q_state: Any,
) -> dict[str, torch.Tensor]:
    del q_num_heads, head_dim, q_state
    q = dequantize_nvfp4(q_quant, q_scale).float()
    return _quantize_div7(q)


def hif4_dynamic_quantize_k(
    k_quant: torch.Tensor,
    k_scale: torch.Tensor,
    kv_num_heads: int,
    head_dim: int,
    k_state: Any,
) -> dict[str, torch.Tensor]:
    del kv_num_heads, head_dim, k_state
    k = dequantize_nvfp4(k_quant, k_scale).float()
    return _quantize_div7(k)


def hif4_dynamic_quantize_v(
    v_quant: torch.Tensor,
    v_scale: torch.Tensor,
    kv_num_heads: int,
    head_dim: int,
    v_state: Any,
) -> dict[str, torch.Tensor]:
    del kv_num_heads, head_dim, v_state
    v = dequantize_nvfp4(v_quant, v_scale).float()
    return _quantize_div7(v)
