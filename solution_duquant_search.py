"""DuQuant with V31 scaler search for the six-function HiF4 demo interface.

A conservative calibration GEMM search selects the channel scaler, followed
by searched multi-step R1, zigzag permutation, and searched multi-step R2.
No techniques from the other solution variants are included.
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

def _hif4_reconstruct(params: dict[str, torch.Tensor]) -> torch.Tensor:
    x = params["sign"].float() * params["mant"].float()
    x = (
        x
        * params["scale_lv3"].float()
        * params["scale_lv2"].float()
        * params["scale_factor"].float()
    )
    return x.flatten(-4, -1)


def _even_indices(n: int, k: int, device: torch.device) -> torch.Tensor:
    if n <= k:
        return torch.arange(n, device=device)
    return torch.linspace(0, n - 1, steps=k, device=device).round().long().unique()


def _activation_absmax_and_samples(calib_activation_list, channels, device):
    stat = torch.zeros(channels, dtype=torch.float32, device=device)
    samples = []
    for activation_quant, activation_scale in calib_activation_list:
        A = dequantize_nvfp4(activation_quant, activation_scale).to(
            device=device, dtype=torch.float32
        ).reshape(-1, channels)
        stat = torch.maximum(stat, A.abs().amax(dim=0))
        samples.append(A)
    return stat, samples


def _householder64(position: int, device: torch.device) -> torch.Tensor:
    source = torch.zeros(64, dtype=torch.float32, device=device)
    source[position] = 1.0
    target = torch.full_like(source, 1.0 / 8.0)
    v = source - target
    return torch.eye(64, device=device) - 2.0 * torch.outer(v, v) / v.dot(v).clamp_min(EPS)


def _block_rotate(x: torch.Tensor, rotation: torch.Tensor) -> torch.Tensor:
    shape = x.shape
    y = x.float().reshape(*shape[:-1], -1, 64)
    return torch.matmul(y, rotation).reshape(shape)


def _worst_local_position(samples: list[torch.Tensor]) -> int:
    local = None
    for A in samples:
        score = A.abs().reshape(-1, A.shape[-1] // 64, 64).amax(dim=(0, 1))
        local = score if local is None else torch.maximum(local, score)
    return 0 if local is None else int(local.argmax().item())


DUQUANT_MAX_ROTATION_STEPS = 8


def _activation_peak(samples: list[torch.Tensor]) -> float:
    """Maximum absolute calibration value after the current rotation."""
    peak = 0.0
    for A in samples:
        if A.numel():
            peak = max(peak, float(A.abs().amax().item()))
    return peak


def _search_multistep_rotation(
    samples: list[torch.Tensor],
    device: torch.device,
    max_steps: int = DUQUANT_MAX_ROTATION_STEPS,
) -> tuple[torch.Tensor, int, float]:
    """Greedily accumulate Householders and retain the lowest-peak depth.

    All 64-channel groups share the cumulative rotation, matching the previous
    implementation.  At every step the worst local lane is recomputed on the
    already-rotated calibration activations.  Candidate depths 1..max_steps are
    compared by their global activation peak.
    """
    identity = torch.eye(64, dtype=torch.float32, device=device)
    if not samples or max_steps <= 0:
        return identity, 0, _activation_peak(samples)

    transformed = [A for A in samples]
    cumulative = identity
    best_rotation = identity
    best_step = 0
    best_peak = float("inf")

    for step in range(1, max_steps + 1):
        position = _worst_local_position(transformed)
        update = _householder64(position, device)
        transformed = [_block_rotate(A, update) for A in transformed]
        cumulative = cumulative @ update
        peak = _activation_peak(transformed)
        if peak < best_peak:
            best_peak = peak
            best_step = step
            best_rotation = cumulative.clone()

    return best_rotation, best_step, best_peak


def _zigzag_permutation(score: torch.Tensor) -> torch.Tensor:
    order = torch.argsort(score, descending=True)
    groups = order.numel() // 64
    table = torch.empty(groups, 64, dtype=order.dtype, device=order.device)
    for rank in range(order.numel()):
        lane = rank // groups
        offset = rank % groups
        group = offset if lane % 2 == 0 else groups - 1 - offset
        table[group, lane] = order[rank]
    return table.reshape(-1)


_SCALER_BETAS = (0.0, 0.25, 0.50)
SCALER_MAX_CALIB_SAMPLES = 4
SCALER_MAX_TOKENS_PER_SAMPLE = 20
SCALER_MAX_WEIGHT_ROWS = 48


def _v31_scaler(
    activation_absmax: torch.Tensor,
    weight_absmax: torch.Tensor,
    beta: float,
) -> torch.Tensor:
    """V31 channel scaler: Activation * s and Weight / s."""
    if beta <= 0.0:
        return torch.ones_like(weight_absmax)
    log_scale = float(beta) * (
        torch.log(weight_absmax.clamp_min(2**-24))
        - torch.log(activation_absmax.clamp_min(2**-24))
    )
    log_scale -= log_scale.median()
    return torch.exp(log_scale).clamp_(2**-6, 2**6)


def _search_v31_scaler(
    weight: torch.Tensor,
    samples: list[torch.Tensor],
    activation_absmax: torch.Tensor,
) -> tuple[torch.Tensor, float]:
    """Select beta by V31's sampled HiF4 GEMM-MSE safety rule."""
    if (
        SCALER_MAX_CALIB_SAMPLES <= 0
        or SCALER_MAX_TOKENS_PER_SAMPLE <= 0
        or SCALER_MAX_WEIGHT_ROWS <= 0
    ):
        raise ValueError("all scaler-search sampling limits must be positive")
    weight_absmax = weight.abs().amax(dim=0)
    if not samples:
        return torch.ones_like(weight_absmax), 0.0

    weight_rows = _even_indices(
        weight.shape[0],
        min(SCALER_MAX_WEIGHT_ROWS, weight.shape[0]),
        weight.device,
    )
    sampled_weight = weight[weight_rows]
    scores = {beta: [] for beta in _SCALER_BETAS}

    for beta in _SCALER_BETAS:
        scaler = _v31_scaler(activation_absmax, weight_absmax, beta)
        transformed_weight = sampled_weight / scaler
        weight_q = _hif4_reconstruct(_hif4_direct(transformed_weight))

        for activation in samples[:SCALER_MAX_CALIB_SAMPLES]:
            token_rows = _even_indices(
                activation.shape[0],
                min(SCALER_MAX_TOKENS_PER_SAMPLE, activation.shape[0]),
                activation.device,
            )
            sampled_activation = activation[token_rows]
            transformed_activation = sampled_activation * scaler
            activation_q = _hif4_reconstruct(
                _hif4_direct(transformed_activation)
            )
            reference = sampled_activation @ sampled_weight.t()
            candidate = activation_q @ weight_q.t()
            scores[beta].append(
                float((candidate - reference).square().mean().item())
            )

    baseline = scores[0.0]
    best_ratio = 1.0
    best_beta = 0.0
    for beta in _SCALER_BETAS[1:]:
        ratios = [
            candidate / max(base, 1e-20)
            for candidate, base in zip(scores[beta], baseline)
        ]
        pooled = sum(scores[beta]) / max(sum(baseline), 1e-20)
        if max(ratios) <= 0.99 and pooled <= 0.92 and pooled < best_ratio:
            best_ratio = pooled
            best_beta = beta

    return (
        _v31_scaler(activation_absmax, weight_absmax, best_beta),
        float(best_beta),
    )


def hif4_calibration_and_quantize_weight(weight_quant, weight_scale, calib_activation_list):
    W = dequantize_nvfp4(weight_quant, weight_scale).float()
    a, samples = _activation_absmax_and_samples(
        calib_activation_list, W.shape[-1], W.device
    )
    scaler, scaler_beta = _search_v31_scaler(W, samples, a)
    smoothed = [A * scaler for A in samples]

    r1, r1_steps, r1_peak = _search_multistep_rotation(smoothed, W.device)
    rotated = [_block_rotate(A, r1) for A in smoothed]
    channel_score = torch.zeros(W.shape[-1], device=W.device)
    for A in rotated:
        channel_score = torch.maximum(channel_score, A.abs().amax(dim=0))
    perm = _zigzag_permutation(channel_score)
    regrouped = [A.index_select(-1, perm) for A in rotated]
    r2, r2_steps, r2_peak = _search_multistep_rotation(regrouped, W.device)

    Wt = _block_rotate(W / scaler, r1).index_select(-1, perm)
    Wt = _block_rotate(Wt, r2)
    state = {
        "activation_scaler": scaler.detach().cpu(),
        "scaler_beta": scaler_beta,
        "rotation1": r1.detach().cpu(),
        "rotation1_steps": r1_steps,
        "rotation1_peak": r1_peak,
        "perm": perm.detach().cpu(),
        "rotation2": r2.detach().cpu(),
        "rotation2_steps": r2_steps,
        "rotation2_peak": r2_peak,
    }
    return {"weight_params": _hif4_direct(Wt), "activation_state": state}


def hif4_dynamic_quantize_activation(activation_quant, activation_scale, activation_state):
    A = dequantize_nvfp4(activation_quant, activation_scale).float()
    scaler = activation_state["activation_scaler"].to(A.device)
    r1 = activation_state["rotation1"].to(A.device)
    perm = activation_state["perm"].to(A.device)
    r2 = activation_state["rotation2"].to(A.device)
    At = _block_rotate(A * scaler, r1).index_select(-1, perm)
    return _hif4_direct(_block_rotate(At, r2))


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
