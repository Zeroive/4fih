"""HiF4 v1: calibration-selected SmoothQuant scaling.

This module keeps the six public entry points from v0.  The only intentional
algorithmic change is on the Linear path: alpha is selected with an actual
quantized GEMM reconstruction objective instead of being fixed at 0.5.
"""
from __future__ import annotations

from typing import Any, Iterable

import torch

import hif4_solution_v0 as _v0


# None is the no-smoothing ablation.  Keeping it in the search prevents a
# poorly calibrated smooth transform from making a layer worse.
LINEAR_ALPHA_CANDIDATES = (None, 0.0, 0.2, 0.35, 0.5, 0.65, 0.8, 1.0)
SEARCH_MAX_TOKENS = 256
SEARCH_MAX_OUTPUTS = 128


# Unchanged public entry points.
dequantize_nvfp4 = _v0.dequantize_nvfp4
hif4_dynamic_quantize_activation = _v0.hif4_dynamic_quantize_activation
hif4_calibration_attention = _v0.hif4_calibration_attention
hif4_dynamic_quantize_q = _v0.hif4_dynamic_quantize_q
hif4_dynamic_quantize_k = _v0.hif4_dynamic_quantize_k
hif4_dynamic_quantize_v = _v0.hif4_dynamic_quantize_v


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
    if not _v0.USE_LINEAR_PERMUTE:
        return None
    score = 0.70 * torch.log2(w_rms + _v0.EPS) + 0.30 * torch.log2(act_rms + _v0.EPS)
    return torch.argsort(score)


def _smooth_scale(
    act_absmax: torch.Tensor,
    w_absmax: torch.Tensor,
    alpha: float | None,
) -> torch.Tensor | None:
    if alpha is None or not _v0.USE_LINEAR_SMOOTH:
        return None
    s = (
        act_absmax.clamp_min(_v0.EPS).pow(alpha)
        / w_absmax.clamp_min(_v0.EPS).pow(1.0 - alpha)
    )
    s = s.clamp(*_v0.LINEAR_SMOOTH_CLAMP)
    return s / torch.exp(torch.log(s).mean())


def _evaluate_linear_candidate(
    W: torch.Tensor,
    X: torch.Tensor,
    perm: torch.Tensor | None,
    alpha: float | None,
    act_absmax: torch.Tensor,
    act_second: torch.Tensor,
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

    if _v0.USE_LINEAR_HADAMARD64:
        Wt = _v0._hadamard64(Wt)
        Xt = _v0._hadamard64(Xt)
        # Rotation mixes the diagonal statistics; sample statistics are exact
        # for the diagonal proxy and cheap at calibration time.
        asecond = Xt.square().mean(dim=0)

    w_importance = _v0._normalize_importance(asecond)
    wp = _v0._hif4_quantize(Wt, error_weight=w_importance)
    Wq = _v0._dequantize_hif4_params(wp)

    a_importance = _v0._normalize_importance(Wt.square().mean(dim=0))
    xp = _v0._hif4_quantize(Xt, error_weight=a_importance)
    Xq = _v0._dequantize_hif4_params(xp)

    # Output-space loss directly measures the interaction between W and A
    # quantization.  Restricting output rows bounds calibration cost.
    rows = min(Wt.shape[0], SEARCH_MAX_OUTPUTS)
    ref = Xt @ Wt[:rows].T
    got = Xq @ Wq[:rows].T
    loss = (ref - got).square().mean() / ref.square().mean().clamp_min(_v0.EPS)

    state = {
        "perm": None if perm is None else perm.detach().cpu(),
        "smooth_scale": None if s is None else s.detach().cpu(),
        "smooth_alpha": alpha,
        "hadamard64": bool(_v0.USE_LINEAR_HADAMARD64),
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
    act_rms = torch.sqrt(act_second + _v0.EPS)
    w_rms = torch.sqrt(W.square().mean(dim=0) + _v0.EPS)

    best = None
    for perm_name, perm in permutations:
        for alpha in LINEAR_ALPHA_CANDIDATES:
            loss, params, state = _evaluate_linear_candidate(
                W, X, perm, alpha, act_absmax, act_second
            )
            if best is None or bool(loss < best[0]):
                state["permutation_strategy"] = perm_name
                best = (loss, params, state)
    assert best is not None
    return {"weight_params": best[1], "activation_state": best[2]}


def hif4_calibration_and_quantize_weight(
    weight_quant: torch.Tensor,
    weight_scale: torch.Tensor,
    calib_activation_list: list,
) -> dict[str, Any]:
    W = dequantize_nvfp4(weight_quant, weight_scale).float()
    X = _sample_calibration(calib_activation_list, W.device)
    w_rms = torch.sqrt(W.square().mean(dim=0) + _v0.EPS)
    act_rms = torch.sqrt(X.square().mean(dim=0) + _v0.EPS)
    perm = _candidate_perm(w_rms, act_rms)
    return _calibrate_with_permutations(W, X, (("magnitude_sort", perm),))

