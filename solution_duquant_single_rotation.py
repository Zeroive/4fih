"""Standalone DuQuant variant with hierarchical 64/8/4 rotations.

Linear path:

    Smooth -> R64 -> global perm8 -> R8 -> global perm4 -> R4 -> HiF4

Calibration statistics retain a per-sample maximum and average those maxima
across samples.  Every rotation level uses exactly one shared Householder; there
is no multi-step rotation search.
"""

from __future__ import annotations

from typing import Any

import torch


EPS = 1e-12
DUQUANT_ALPHA = 0.5


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
        1.0 + mantissa * 0.25
    )


def _hif4_direct(x: torch.Tensor) -> dict[str, torch.Tensor]:
    if x.shape[-1] % 64 != 0:
        raise ValueError(
            f"HiF4 requires last dimension divisible by 64, got {x.shape[-1]}"
        )

    groups = x.shape[-1] // 64
    xg = x.float().unflatten(-1, (groups, 8, 2, 4))
    absolute = xg.abs()

    peak64 = absolute.amax(dim=(-1, -2, -3), keepdim=True)
    scale_factor = _round_to_e6m2(
        (peak64 / 7.0).clamp_min(2.0**-48)
    )

    peak8 = absolute.amax(dim=(-1, -2), keepdim=True)
    scale_lv2 = torch.where(
        peak8 > 4.0 * scale_factor,
        torch.full_like(peak8, 2.0),
        torch.ones_like(peak8),
    )

    peak4 = absolute.amax(dim=-1, keepdim=True)
    scale_lv3 = torch.where(
        peak4 > 2.0 * scale_factor * scale_lv2,
        torch.full_like(peak4, 2.0),
        torch.ones_like(peak4),
    )

    local_scale = (scale_factor * scale_lv2 * scale_lv3).clamp_min(EPS)
    mantissa = (
        torch.round(absolute / local_scale * 4.0) * 0.25
    ).clamp(0.0, 1.75)
    sign = torch.where(
        mantissa > 0.0,
        torch.sign(xg),
        torch.zeros_like(xg),
    )

    return {
        "scale_factor": scale_factor.to(torch.bfloat16),
        "scale_lv2": scale_lv2.to(torch.bfloat16),
        "scale_lv3": scale_lv3.to(torch.bfloat16),
        "sign": sign.to(torch.bfloat16),
        "mant": mantissa.to(torch.bfloat16),
    }


def _activation_stats_and_samples(
    calib_activation_list: list,
    channels: int,
    device: torch.device,
) -> tuple[torch.Tensor, list[torch.Tensor]]:
    samples = []
    sample_channel_maxima = []
    for activation_quant, activation_scale in calib_activation_list:
        activation = dequantize_nvfp4(
            activation_quant, activation_scale
        ).to(device=device, dtype=torch.float32).reshape(-1, channels)
        samples.append(activation)
        sample_channel_maxima.append(activation.abs().amax(dim=0))

    if sample_channel_maxima:
        activation_stat = torch.stack(sample_channel_maxima, dim=0).mean(dim=0)
    else:
        activation_stat = torch.zeros(
            channels, dtype=torch.float32, device=device
        )
    return activation_stat, samples


def _householder(
    width: int,
    position: int,
    device: torch.device,
) -> torch.Tensor:
    source = torch.zeros(width, dtype=torch.float32, device=device)
    source[position] = 1.0
    target = torch.full_like(source, 1.0 / (width**0.5))
    direction = source - target
    return torch.eye(width, dtype=torch.float32, device=device) - (
        2.0
        * torch.outer(direction, direction)
        / direction.dot(direction).clamp_min(EPS)
    )


def _block_rotate(
    x: torch.Tensor,
    rotation: torch.Tensor,
) -> torch.Tensor:
    shape = x.shape
    width = rotation.shape[-1]
    if shape[-1] % width != 0:
        raise ValueError(
            f"last dimension {shape[-1]} is not divisible by rotation width {width}"
        )
    grouped = x.float().reshape(*shape[:-1], -1, width)
    return torch.matmul(grouped, rotation).reshape(shape)


def _worst_local_position(
    samples: list[torch.Tensor],
    width: int,
) -> int:
    sample_lane_maxima = []
    for activation in samples:
        grouped = activation.abs().reshape(
            -1, activation.shape[-1] // width, width
        )
        sample_lane_maxima.append(grouped.amax(dim=(0, 1)))
    if not sample_lane_maxima:
        return 0
    lane_score = torch.stack(sample_lane_maxima, dim=0).mean(dim=0)
    return int(lane_score.argmax().item())


def _zigzag_permutation(
    score: torch.Tensor,
    group_width: int,
) -> torch.Tensor:
    """Globally redistribute channels into groups of ``group_width``.

    This is not restricted to the preceding rotation groups.  For example,
    perm8 may move outputs originating from different R64 groups into the same
    new 8-element group.
    """
    if score.numel() % group_width != 0:
        raise ValueError(
            f"channel count {score.numel()} is not divisible by {group_width}"
        )
    order = torch.argsort(score, descending=True)
    groups = order.numel() // group_width
    table = torch.empty(
        groups, group_width, dtype=order.dtype, device=order.device
    )
    for rank in range(order.numel()):
        lane = rank // groups
        offset = rank % groups
        group = offset if lane % 2 == 0 else groups - 1 - offset
        table[group, lane] = order[rank]
    return table.reshape(-1)


def hif4_calibration_and_quantize_weight(
    weight_quant: torch.Tensor,
    weight_scale: torch.Tensor,
    calib_activation_list: list,
) -> dict[str, Any]:
    weight = dequantize_nvfp4(weight_quant, weight_scale).float()
    activation_stat, samples = _activation_stats_and_samples(
        calib_activation_list, weight.shape[-1], weight.device
    )
    weight_stat = weight.abs().amax(dim=0)
    smooth_scale = (
        activation_stat.clamp_min(EPS).pow(DUQUANT_ALPHA)
        / weight_stat.clamp_min(EPS).pow(1.0 - DUQUANT_ALPHA)
    )

    smoothed_samples = [activation / smooth_scale for activation in samples]
    position64 = _worst_local_position(smoothed_samples, 64)
    rotation64 = _householder(64, position64, weight.device)

    rotated_samples = [
        _block_rotate(activation, rotation64) for activation in smoothed_samples
    ]
    if rotated_samples:
        channel_score = torch.stack(
            [activation.abs().amax(dim=0) for activation in rotated_samples],
            dim=0,
        ).mean(dim=0)
    else:
        channel_score = torch.zeros(
            weight.shape[-1], dtype=torch.float32, device=weight.device
        )
    permutation8 = _zigzag_permutation(channel_score, group_width=8)

    permuted_samples = [
        activation.index_select(-1, permutation8)
        for activation in rotated_samples
    ]
    position8 = _worst_local_position(permuted_samples, 8)
    rotation8 = _householder(8, position8, weight.device)
    rotated8_samples = [
        _block_rotate(activation, rotation8)
        for activation in permuted_samples
    ]

    if rotated8_samples:
        channel_score4 = torch.stack(
            [activation.abs().amax(dim=0) for activation in rotated8_samples],
            dim=0,
        ).mean(dim=0)
    else:
        channel_score4 = torch.zeros(
            weight.shape[-1], dtype=torch.float32, device=weight.device
        )
    permutation4 = _zigzag_permutation(channel_score4, group_width=4)
    permuted4_samples = [
        activation.index_select(-1, permutation4)
        for activation in rotated8_samples
    ]

    position4 = _worst_local_position(permuted4_samples, 4)
    rotation4 = _householder(4, position4, weight.device)

    transformed_weight = _block_rotate(
        weight * smooth_scale, rotation64
    ).index_select(-1, permutation8)
    transformed_weight = _block_rotate(transformed_weight, rotation8)
    transformed_weight = transformed_weight.index_select(-1, permutation4)
    transformed_weight = _block_rotate(transformed_weight, rotation4)
    state = {
        "smooth_scale": smooth_scale.detach().cpu(),
        "rotation64": rotation64.detach().cpu(),
        "rotation64_position": position64,
        "perm8": permutation8.detach().cpu(),
        "rotation8": rotation8.detach().cpu(),
        "rotation8_position": position8,
        "perm4": permutation4.detach().cpu(),
        "rotation4": rotation4.detach().cpu(),
        "rotation4_position": position4,
    }
    return {
        "weight_params": _hif4_direct(transformed_weight),
        "activation_state": state,
    }


def hif4_dynamic_quantize_activation(
    activation_quant: torch.Tensor,
    activation_scale: torch.Tensor,
    activation_state: dict,
) -> dict[str, torch.Tensor]:
    activation = dequantize_nvfp4(
        activation_quant, activation_scale
    ).float()
    smooth_scale = activation_state["smooth_scale"].to(activation.device)
    rotation64 = activation_state["rotation64"].to(activation.device)
    permutation8 = activation_state["perm8"].to(activation.device)
    rotation8 = activation_state["rotation8"].to(activation.device)
    permutation4 = activation_state["perm4"].to(activation.device)
    rotation4 = activation_state["rotation4"].to(activation.device)

    transformed = _block_rotate(
        activation / smooth_scale, rotation64
    ).index_select(-1, permutation8)
    transformed = _block_rotate(transformed, rotation8)
    transformed = transformed.index_select(-1, permutation4)
    transformed = _block_rotate(transformed, rotation4)
    return _hif4_direct(transformed)


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
    return _hif4_direct(dequantize_nvfp4(q_quant, q_scale).float())


def hif4_dynamic_quantize_k(
    k_quant: torch.Tensor,
    k_scale: torch.Tensor,
    kv_num_heads: int,
    head_dim: int,
    k_state: Any,
) -> dict[str, torch.Tensor]:
    del kv_num_heads, head_dim, k_state
    return _hif4_direct(dequantize_nvfp4(k_quant, k_scale).float())


def hif4_dynamic_quantize_v(
    v_quant: torch.Tensor,
    v_scale: torch.Tensor,
    kv_num_heads: int,
    head_dim: int,
    v_state: Any,
) -> dict[str, torch.Tensor]:
    del kv_num_heads, head_dim, v_state
    return _hif4_direct(dequantize_nvfp4(v_quant, v_scale).float())
