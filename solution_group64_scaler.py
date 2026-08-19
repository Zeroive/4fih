"""HiF4 with iterative least-squares scale fitting for every 64 values.

For each independent 64-value group:

    absmax / 7 -> initial E6M2 scale
    x / scale  -> nearest valid HiF4 hierarchical grid q
    scale      -> sum(x * q) / sum(q ** 2), projected to E6M2

The q/scale alternation is repeated ``GROUP64_SCALE_ITERATIONS`` times.  This
file is standalone and implements all six functions required by the demo.
"""

from __future__ import annotations

from typing import Any

import torch


GROUP64_SCALE_ITERATIONS = 5
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


def _mantissa_for_level(
    normalized_abs: torch.Tensor,
    level: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return nearest mantissa and elementwise squared error for one level."""
    mantissa = (
        torch.round(normalized_abs / level * 4.0) * 0.25
    ).clamp(0.0, 1.75)
    error = (normalized_abs - mantissa * level).square()
    return mantissa, error


def _nearest_hif4_grid(
    normalized: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Map normalized groups to the nearest valid hierarchical HiF4 grid.

    ``normalized`` has logical shape ``(..., groups, 8, 2, 4)``.  lv2 is
    shared by each 8-value block and lv3 by each nested 4-value block.  Both
    binary choices are selected by exact squared-error comparison.
    """
    absolute = normalized.abs()
    sign = torch.sign(normalized)

    lv2_candidates = []
    lv3_candidates = []
    mant_candidates = []
    error_candidates = []

    for lv2_value in (1.0, 2.0):
        mant_lv3_1, error_lv3_1 = _mantissa_for_level(absolute, lv2_value)
        mant_lv3_2, error_lv3_2 = _mantissa_for_level(
            absolute, lv2_value * 2.0
        )

        # lv3 is shared by the four values on the last axis.
        use_lv3_2 = error_lv3_2.sum(dim=-1, keepdim=True) < error_lv3_1.sum(
            dim=-1, keepdim=True
        )
        lv3 = torch.where(
            use_lv3_2,
            torch.full_like(absolute[..., :1], 2.0),
            torch.ones_like(absolute[..., :1]),
        )
        mantissa = torch.where(use_lv3_2, mant_lv3_2, mant_lv3_1)
        reconstruction = mantissa * lv2_value * lv3

        lv2_candidates.append(
            torch.full_like(absolute[..., :1, :1], lv2_value)
        )
        lv3_candidates.append(lv3)
        mant_candidates.append(mantissa)
        error_candidates.append(
            (absolute - reconstruction).square().sum(dim=(-1, -2), keepdim=True)
        )

    # lv2 is shared across both four-value children, i.e. all eight values.
    use_lv2_2 = error_candidates[1] < error_candidates[0]
    lv2 = torch.where(use_lv2_2, lv2_candidates[1], lv2_candidates[0])
    lv3 = torch.where(use_lv2_2, lv3_candidates[1], lv3_candidates[0])
    mantissa = torch.where(use_lv2_2, mant_candidates[1], mant_candidates[0])

    # HiF4 represents zero with sign=0 regardless of the source sign.
    quant_sign = torch.where(mantissa > 0.0, sign, torch.zeros_like(sign))
    q = quant_sign * mantissa * lv2 * lv3
    return lv2, lv3, quant_sign, mantissa, q


def _quantize_group64_iterative(x: torch.Tensor) -> dict[str, torch.Tensor]:
    if x.shape[-1] % 64 != 0:
        raise ValueError(
            f"HiF4 requires last dimension divisible by 64, got {x.shape[-1]}"
        )
    if GROUP64_SCALE_ITERATIONS < 1:
        raise ValueError("GROUP64_SCALE_ITERATIONS must be at least 1")

    grouped = x.float().unflatten(-1, (x.shape[-1] // 64, 8, 2, 4))
    absmax = grouped.abs().amax(dim=(-1, -2, -3), keepdim=True)

    # The largest normalized HiF4 value is 1.75 * 2 * 2 = 7.
    scale = _round_to_e6m2((absmax / 7.0).clamp_min(2.0**-48))

    for _ in range(GROUP64_SCALE_ITERATIONS):
        _, _, _, _, q = _nearest_hif4_grid(grouped / scale)
        numerator = (grouped * q).sum(dim=(-1, -2, -3), keepdim=True)
        denominator = q.square().sum(dim=(-1, -2, -3), keepdim=True)
        fitted = numerator / denominator.clamp_min(EPS)
        fitted = torch.where(denominator > 0.0, fitted, scale)
        scale = _round_to_e6m2(fitted.clamp_min(2.0**-48))

    lv2, lv3, sign, mantissa, _ = _nearest_hif4_grid(grouped / scale)
    return {
        "scale_factor": scale.to(torch.bfloat16),
        "scale_lv2": lv2.to(torch.bfloat16),
        "scale_lv3": lv3.to(torch.bfloat16),
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
        "weight_params": _quantize_group64_iterative(weight),
        "activation_state": None,
    }


def hif4_dynamic_quantize_activation(
    activation_quant: torch.Tensor,
    activation_scale: torch.Tensor,
    activation_state: Any,
) -> dict[str, torch.Tensor]:
    del activation_state
    activation = dequantize_nvfp4(activation_quant, activation_scale).float()
    return _quantize_group64_iterative(activation)


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
    return _quantize_group64_iterative(dequantize_nvfp4(q_quant, q_scale).float())


def hif4_dynamic_quantize_k(
    k_quant: torch.Tensor,
    k_scale: torch.Tensor,
    kv_num_heads: int,
    head_dim: int,
    k_state: Any,
) -> dict[str, torch.Tensor]:
    del kv_num_heads, head_dim, k_state
    return _quantize_group64_iterative(dequantize_nvfp4(k_quant, k_scale).float())


def hif4_dynamic_quantize_v(
    v_quant: torch.Tensor,
    v_scale: torch.Tensor,
    kv_num_heads: int,
    head_dim: int,
    v_state: Any,
) -> dict[str, torch.Tensor]:
    del kv_num_heads, head_dim, v_state
    return _quantize_group64_iterative(dequantize_nvfp4(v_quant, v_scale).float())
