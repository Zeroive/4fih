"""OmniQuant logic adapted to the six-function HiF4 demo interface.

Implements learnable equivalent per-channel scaling (LET scale) and learnable
Weight clipping (LWC), optimized against HiF4 layer-output reconstruction.
The LET shift is intentionally omitted because this interface has no bias hook.
No techniques from the other solution variants are included.
"D:\Program Files\uv\bin\uv.exe" run D:/DeskTop/26算法大赛/example/.venv/Scripts/python.exe D:\DeskTop\26算法大赛\example\example_0818\self_check_.py --datasets_dir mini_sample --solution_dir solution_test 
Interface check: PASSED (6/6 functions found)

======================== Linear ========================
[Linear][Group 0] calibration: PASSED [4019.96ms] (W=(8192, 2048), num_calib=5)
[Linear][Group 0][Test 0] activation: FAILED [3.44ms] (W=(8192, 2048), A=(10, 2048))
      MatMul MSE 1.6805e-02 exceeds threshold 0.001
[Linear][Group 0][Test 1] activation: FAILED [4.84ms] (W=(8192, 2048), A=(128, 2048))
      MatMul MSE 1.6075e-02 exceeds threshold 0.001
[Linear][Group 0][Test 2] activation: FAILED [6.22ms] (W=(8192, 2048), A=(512, 2048))
      MatMul MSE 1.4444e-02 exceeds threshold 0.001
[Linear][Group 0][Test 3] activation: FAILED [11.90ms] (W=(8192, 2048), A=(1024, 2048))
      MatMul MSE 1.3490e-02 exceeds threshold 0.001
[Linear][Group 0][Test 4] activation: FAILED [11.17ms] (W=(8192, 2048), A=(1024, 2048))
      MatMul MSE 1.3363e-02 exceeds threshold 0.001

====================== Attention ======================
[Attention][Group 0] calibration: PASSED [0.00ms] (q_heads=16, kv_heads=2, head_dim=256, num_calib=5)
[Attention][Group 0][Test 0] FAILED [7.00ms] (Q=(10, 4096), K=(10, 512), V=(10, 512))
      Attention MSE 1.0388e-03 exceeds threshold 0.001
[Attention][Group 0][Test 1] PASSED (MSE=4.0943e-04) [23.84ms] (Q=(128, 4096), K=(128, 512), V=(128, 512))
[Attention][Group 0][Test 2] PASSED (MSE=2.8481e-04) [53.80ms] (Q=(512, 4096), K=(512, 512), V=(512, 512))
[Attention][Group 0][Test 3] PASSED (MSE=2.1662e-04) [136.30ms] (Q=(1024, 4096), K=(1024, 512), V=(1024, 512))
[Attention][Group 0][Test 4] PASSED (MSE=2.3384e-04) [89.29ms] (Q=(1024, 4096), K=(1024, 512), V=(1024, 512))

======================== Summary ========================
Passed checks: 6/12
Failed checks: 6/12
Avg Calibration Time: 2009.98 ms
Avg Inference (Dynamic Quant) Time: 34.78 ms
SOME OUTPUT-FORMAT OR PRECISION CHECKS FAILED

Process finished with exit code 1

"""
from __future__ import annotations

from typing import Any

import torch


EPS = 1e-12
OMNI_STEPS = 20
OMNI_MAX_TOKENS = 64
OMNI_MAX_OUTPUTS = 128
OMNI_LR = 3e-2


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
# OmniQuant fake quantization and learnable parameters
# =============================================================================
def _dequantize_hif4(params: dict[str, torch.Tensor]) -> torch.Tensor:
    x = (
        params["scale_factor"].float()
        * params["scale_lv2"].float()
        * params["scale_lv3"].float()
        * params["sign"].float()
        * params["mant"].float()
    )
    return x.flatten(-4, -1)


def _fake_hif4_ste(x: torch.Tensor) -> torch.Tensor:
    quantized = _dequantize_hif4(_hif4_direct(x))
    return x + (quantized - x).detach()


def _clip_weight(
    weight: torch.Tensor,
    lower_ratio: torch.Tensor,
    upper_ratio: torch.Tensor,
) -> torch.Tensor:
    groups = weight.shape[-1] // 64
    wg = weight.reshape(weight.shape[0], groups, 64)
    lower = wg.amin(dim=-1, keepdim=True) * lower_ratio.reshape(1, groups, 1)
    upper = wg.amax(dim=-1, keepdim=True) * upper_ratio.reshape(1, groups, 1)
    return torch.maximum(torch.minimum(wg, upper), lower).reshape_as(weight)


def _calibration_tokens(calib_activation_list, channels, device):
    chunks = []
    remaining = OMNI_MAX_TOKENS
    for activation_quant, activation_scale in calib_activation_list:
        if remaining <= 0:
            break
        A = dequantize_nvfp4(activation_quant, activation_scale).to(
            device=device, dtype=torch.float32
        ).reshape(-1, channels)[:remaining]
        chunks.append(A)
        remaining -= A.shape[0]
    return None if not chunks else torch.cat(chunks, dim=0)


def hif4_calibration_and_quantize_weight(
    weight_quant: torch.Tensor,
    weight_scale: torch.Tensor,
    calib_activation_list: list,
) -> dict[str, Any]:
    W = dequantize_nvfp4(weight_quant, weight_scale).float()
    channels = W.shape[-1]
    groups = channels // 64
    X = _calibration_tokens(calib_activation_list, channels, W.device)

    log_scale = torch.zeros(channels, device=W.device, requires_grad=True)
    clip_logits = torch.full((groups, 2), 6.0, device=W.device, requires_grad=True)

    if X is not None:
        rows = min(W.shape[0], OMNI_MAX_OUTPUTS)
        W_search = W[:rows]
        reference = X @ W_search.T
        optimizer = torch.optim.Adam((log_scale, clip_logits), lr=OMNI_LR)

        for _ in range(OMNI_STEPS):
            scale = torch.exp(log_scale)
            ratios = torch.sigmoid(clip_logits)
            transformed_w = _clip_weight(
                W_search * scale.unsqueeze(0), ratios[:, 0], ratios[:, 1]
            )
            transformed_x = X / scale
            output = _fake_hif4_ste(transformed_x) @ _fake_hif4_ste(transformed_w).T
            loss = (output - reference).square().mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        scale = torch.exp(log_scale.detach())
        ratios = torch.sigmoid(clip_logits.detach())
    else:
        scale = torch.ones(channels, device=W.device)
        ratios = torch.ones(groups, 2, device=W.device)

    transformed_w = _clip_weight(
        W * scale.unsqueeze(0), ratios[:, 0], ratios[:, 1]
    )
    state = {
        "let_scale": scale.detach().cpu(),
        "lwc_lower_ratio": ratios[:, 0].detach().cpu(),
        "lwc_upper_ratio": ratios[:, 1].detach().cpu(),
    }
    return {"weight_params": _hif4_direct(transformed_w), "activation_state": state}


def hif4_dynamic_quantize_activation(
    activation_quant: torch.Tensor,
    activation_scale: torch.Tensor,
    activation_state: Any,
) -> dict[str, torch.Tensor]:
    A = dequantize_nvfp4(activation_quant, activation_scale).float()
    scale = activation_state["let_scale"].to(A.device)
    return _hif4_direct(A / scale)


# Attention is unchanged in this OmniQuant Linear-layer adaptation.
def hif4_calibration_attention(calib_qkv_list, q_num_heads, kv_num_heads, head_dim):
    return {"q_state": None, "k_state": None, "v_state": None}


def hif4_dynamic_quantize_q(q_quant, q_scale, q_num_heads, head_dim, q_state):
    return _hif4_direct(dequantize_nvfp4(q_quant, q_scale).float())


def hif4_dynamic_quantize_k(k_quant, k_scale, kv_num_heads, head_dim, k_state):
    return _hif4_direct(dequantize_nvfp4(k_quant, k_scale).float())


def hif4_dynamic_quantize_v(v_quant, v_scale, kv_num_heads, head_dim, v_state):
    return _hif4_direct(dequantize_nvfp4(v_quant, v_scale).float())

