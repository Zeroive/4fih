from __future__ import annotations

import math
from typing import Optional, Tuple

import torch

# =============================================================================
# V30-k-multistart
# Strict split-interface quantization. Every public per-tensor API reads ONLY
# its own NVFP4 tensor+scale. No cross-operand statistics, MatMul/SDPA targets,
# or result-baseline fitting are used.
#
# Two general changes over V26:
#   1) exact-structure E6M2 search around max/7: offsets [-1..+4]. Extensive
#      exhaustive checks against all 255 legal E6M2 values found the optimum in
#      this interval for NVFP4-like and broad continuous stress distributions.
#   2) K-only softmax-nullspace optimization. K -> K-c (same c for every key
#      position) changes each query's logits only by a rowwise constant, hence
#      softmax is exactly invariant. We optimize this quotient space using K
#      alone and keep the raw-K quantization as a hard candidate per 64 features.
# =============================================================================

_HIF4_BLOCK = 64
_NVFP4_BLOCK = 16
_SEARCH_CHUNK_BLOCKS = 16384
_E6_ANCHOR_OFFSETS = (-1, 0, 1, 2, 3, 4)
_K_QUOTIENT_TOTAL_ROUNDS = 8  # seven K-only quotient stages after raw baseline
_K_RELAX_GAMMAS = (1.0, 1.5, 2.0, 2.5)
_K_MEDIAN_EXTRA_ROUNDS = 3

_E6_TABLE_CACHE: dict[str, torch.Tensor] = {}


def dequantize_nvfp4(
    quant_float: torch.Tensor,
    scale_float: torch.Tensor,
    blk_size: int = _NVFP4_BLOCK,
) -> torch.Tensor:
    c = int(quant_float.shape[-1])
    if c % blk_size != 0:
        raise ValueError(
            f"Last dimension {c} is not divisible by NVFP4 block size {blk_size}"
        )
    x = quant_float.unflatten(-1, (-1, blk_size))
    result = x * scale_float.unsqueeze(-1)
    # Match the supplied checker domain: NVFP4 dequantization is rounded to BF16.
    return result.flatten(-2, -1).to(torch.bfloat16)


def _build_e6m2_table(device: torch.device) -> torch.Tensor:
    key = str(device)
    cached = _E6_TABLE_CACHE.get(key)
    if cached is not None and cached.device == device:
        return cached
    values = []
    for e in range(-48, 16):
        for m in (1.0, 1.25, 1.5, 1.75):
            v = math.ldexp(m, e)
            if v <= 49152.0:
                values.append(v)
    table = torch.tensor(values, dtype=torch.float32, device=device)
    _E6_TABLE_CACHE[key] = table
    return table


def _nearest_e6m2_index(target: torch.Tensor, table: torch.Tensor) -> torch.Tensor:
    # Avoid table[0].item()/table[-1].item() device synchronizations on accelerators.
    t = target.clamp(min=2.0 ** -48, max=49152.0)
    hi = torch.searchsorted(table, t).clamp(0, table.numel() - 1)
    lo = (hi - 1).clamp(0, table.numel() - 1)
    vlo = table[lo]
    vhi = table[hi]
    choose_hi = (vhi - t).abs() < (t - vlo).abs()
    return torch.where(choose_hi, hi, lo)


def _fixed_scale_self_sse(abs_x: torch.Tensor, sf: torch.Tensor) -> torch.Tensor:
    """Exact minimum unweighted SSE for a fixed E6M2 level-1 scale."""
    errs = []
    for mult in (1.0, 2.0, 4.0):
        denom = sf * mult
        mant = torch.round((abs_x / denom) * 4.0) * 0.25
        mant = mant.clamp_(0.0, 1.75)
        errs.append((mant * denom - abs_x).square().sum(dim=-1, keepdim=True))
    e1, e2, e4 = errs
    # lv2=1 -> each 4-vector may use effective multiplier 1 or 2.
    err_l2_1 = torch.minimum(e1, e2).sum(dim=-2, keepdim=True)
    # lv2=2 -> each 4-vector may use effective multiplier 2 or 4.
    err_l2_2 = torch.minimum(e2, e4).sum(dim=-2, keepdim=True)
    return torch.minimum(err_l2_1, err_l2_2).sum(dim=(-3, -2, -1))


def _materialize_fixed_scale_self(
    x: torch.Tensor,
    sf: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    abs_x = x.abs()
    mant_by_mult = []
    err_by_mult = []
    for mult in (1.0, 2.0, 4.0):
        denom = sf * mult
        mant = torch.round((abs_x / denom) * 4.0) * 0.25
        mant = mant.clamp(0.0, 1.75)
        rec = mant * denom
        mant_by_mult.append(mant)
        err_by_mult.append((rec - abs_x).square().sum(dim=-1, keepdim=True))

    e1, e2, e4 = err_by_mult
    l3_if_l2_1 = torch.where(e2 < e1, 2.0, 1.0)
    l3_if_l2_2 = torch.where(e4 < e2, 2.0, 1.0)
    err_l2_1 = torch.minimum(e1, e2).sum(dim=-2, keepdim=True)
    err_l2_2 = torch.minimum(e2, e4).sum(dim=-2, keepdim=True)
    l2 = torch.where(err_l2_2 < err_l2_1, 2.0, 1.0)
    l3 = torch.where(l2 == 1.0, l3_if_l2_1, l3_if_l2_2)

    mult = l2 * l3
    mant = torch.where(
        mult == 1.0,
        mant_by_mult[0],
        torch.where(mult == 2.0, mant_by_mult[1], mant_by_mult[2]),
    )
    sign = torch.sign(x)
    sign = torch.where(mant == 0.0, torch.zeros_like(sign), sign)
    return sign, mant, l2, l3


def _quantize_tensor_self_mse(
    x: torch.Tensor,
    *,
    return_dequant: bool = False,
) -> Tuple[dict, Optional[torch.Tensor]]:
    """Strict per-tensor HiF4 quantizer; objective is only this tensor's SSE."""
    shape = tuple(int(s) for s in x.shape)
    if not shape:
        raise ValueError("Input must have at least one dimension")
    c = shape[-1]
    if c % _HIF4_BLOCK != 0:
        raise ValueError(f"Last dimension {c} not divisible by HiF4 block size 64")

    x = x.float()
    nblocks = c // _HIF4_BLOCK
    rows = x.numel() // c
    blocks = x.reshape(rows, nblocks, 8, 2, 4).reshape(-1, 8, 2, 4)
    total = int(blocks.shape[0])

    table = _build_e6m2_table(x.device)
    last = int(table.numel() - 1)

    sf_out = torch.empty((total, 1, 1, 1), dtype=torch.bfloat16, device=x.device)
    l2_out = torch.empty((total, 8, 1, 1), dtype=torch.bfloat16, device=x.device)
    l3_out = torch.empty((total, 8, 2, 1), dtype=torch.bfloat16, device=x.device)
    sign_out = torch.empty((total, 8, 2, 4), dtype=torch.bfloat16, device=x.device)
    mant_out = torch.empty((total, 8, 2, 4), dtype=torch.bfloat16, device=x.device)
    dq_out = torch.empty_like(blocks) if return_dequant else None

    for start in range(0, total, _SEARCH_CHUNK_BLOCKS):
        end = min(start + _SEARCH_CHUNK_BLOCKS, total)
        xb = blocks[start:end]
        ax = xb.abs()
        bsz = int(xb.shape[0])

        # Natural anchor: the maximum value just fits 1.75 * 2 * 2 = 7 times sf.
        block_max = ax.amax(dim=(1, 2, 3))
        anchor = _nearest_e6m2_index(block_max / 7.0, table)

        best_err = torch.full((bsz,), float("inf"), dtype=torch.float32, device=x.device)
        best_idx = anchor.clone()
        for off in _E6_ANCHOR_OFFSETS:
            idx = (anchor + off).clamp(0, last)
            sf = table[idx].view(bsz, 1, 1, 1)
            err = _fixed_scale_self_sse(ax, sf)
            better = err < best_err
            best_err = torch.where(better, err, best_err)
            best_idx = torch.where(better, idx, best_idx)

        sf = table[best_idx].view(bsz, 1, 1, 1)
        sign, mant, l2, l3 = _materialize_fixed_scale_self(xb, sf)

        sf_out[start:end] = sf.to(torch.bfloat16)
        l2_out[start:end] = l2.to(torch.bfloat16)
        l3_out[start:end] = l3.to(torch.bfloat16)
        sign_out[start:end] = sign.to(torch.bfloat16)
        mant_out[start:end] = mant.to(torch.bfloat16)
        if dq_out is not None:
            dq_out[start:end] = sign * mant * l2 * l3 * sf

    prefix = shape[:-1]
    params = {
        "scale_factor": sf_out.reshape(*prefix, nblocks, 1, 1, 1),
        "scale_lv2": l2_out.reshape(*prefix, nblocks, 8, 1, 1),
        "scale_lv3": l3_out.reshape(*prefix, nblocks, 8, 2, 1),
        "sign": sign_out.reshape(*prefix, nblocks, 8, 2, 4),
        "mant": mant_out.reshape(*prefix, nblocks, 8, 2, 4),
    }
    dq = None
    if dq_out is not None:
        # Match checker dequantization output precision.
        dq = dq_out.reshape(shape).to(torch.bfloat16).float()
    return params, dq


def _quantize_nvfp4_operand(quant: torch.Tensor, scale: torch.Tensor) -> dict:
    x = dequantize_nvfp4(quant, scale).float()
    params, _ = _quantize_tensor_self_mse(x, return_dequant=False)
    return params


def _merge_k_candidates_per_feature_block(
    x: torch.Tensor,
    params_list: list[dict],
    dq_list: list[torch.Tensor],
) -> dict:
    """
    Select the best K representative modulo sequence-wise translation.

    For each batch-prefix x 64-feature block independently, measure
      min_c ||Kq - (K-c)||_F^2,
    whose closed form is the centered residual SSE. Each selected block may use
    a different translation vector; their concatenation is still one vector
    common to all key positions, hence it remains an exact softmax nullspace.
    """
    if len(params_list) == 1 or x.dim() < 2 or int(x.shape[-2]) <= 1:
        return params_list[0]

    shape = tuple(int(s) for s in x.shape)
    seq = shape[-2]
    hidden = shape[-1]
    nblocks = hidden // 64
    batch_prefix = shape[:-2]
    groups = int(math.prod(batch_prefix)) if batch_prefix else 1
    x4 = x.float().reshape(groups, seq, nblocks, 64)

    scores = []
    for dq in dq_list:
        e = dq.float().reshape(groups, seq, nblocks, 64) - x4
        # Common translation across keys is invisible to softmax; project it out.
        e = e - e.mean(dim=1, keepdim=True)
        scores.append(e.square().sum(dim=(1, 3)))  # [groups, nblocks]
    score_stack = torch.stack(scores, dim=0)  # [candidate, groups, nblocks]
    best = score_stack.argmin(dim=0)  # [groups, nblocks]

    out: dict[str, torch.Tensor] = {}
    for name in params_list[0]:
        base = params_list[0][name]
        # Original parameter shape is [batch_prefix..., seq, nblocks, tail...].
        tail = tuple(int(v) for v in base.shape[len(batch_prefix) + 2 :])
        y = base.reshape(groups, seq, nblocks, *tail).clone()
        for ci in range(1, len(params_list)):
            cand = params_list[ci][name].reshape(groups, seq, nblocks, *tail)
            mask_shape = (groups, 1, nblocks) + (1,) * len(tail)
            mask = (best == ci).reshape(groups, 1, nblocks, *([1] * len(tail)))
            y = torch.where(mask, cand, y)
        out[name] = y.reshape(base.shape)
    return out


def _select_best_k_dq_per_feature_block(
    x: torch.Tensor,
    dq_list: list[torch.Tensor],
) -> torch.Tensor:
    """Return the history-best dequantized K independently per 64-feature block."""
    if len(dq_list) == 1 or x.dim() < 2 or int(x.shape[-2]) <= 1:
        return dq_list[0]
    shape = tuple(int(s) for s in x.shape)
    seq = shape[-2]
    hidden = shape[-1]
    nblocks = hidden // 64
    batch_prefix = shape[:-2]
    groups = int(math.prod(batch_prefix)) if batch_prefix else 1
    x4 = x.float().reshape(groups, seq, nblocks, 64)
    scores = []
    qs = []
    for dq in dq_list:
        q4 = dq.float().reshape(groups, seq, nblocks, 64)
        e = q4 - x4
        e = e - e.mean(dim=1, keepdim=True)
        scores.append(e.square().sum(dim=(1, 3)))
        qs.append(q4)
    best = torch.stack(scores, dim=0).argmin(dim=0)
    out = qs[0].clone()
    for ci in range(1, len(qs)):
        mask = (best == ci).reshape(groups, 1, nblocks, 1)
        out = torch.where(mask, qs[ci], out)
    return out.reshape(shape)


def _quantize_k_softmax_quotient(k_quant: torch.Tensor, k_scale: torch.Tensor) -> dict:
    """K-only softmax-quotient quantization with safeguarded over-relaxation.

    For any feature vector c shared across key positions, K -> K-c changes each
    query's logits only by a rowwise constant and is therefore an exact softmax
    invariance.  V28 used one fixed-point candidate per stage.  V29 evaluates a
    few over-relaxed translations around that fixed-point update to cross discrete
    HiF4 decision boundaries.  The raw/V28 trajectory remains in the candidate
    history and final selection is per 64-feature block by exact quotient SSE.
    """
    x = dequantize_nvfp4(k_quant, k_scale).float()
    if x.dim() < 2 or int(x.shape[-2]) <= 1:
        return _quantize_tensor_self_mse(x, return_dequant=False)[0]

    params_list: list[dict] = []
    dq_list: list[torch.Tensor] = []

    # Raw K is an immutable hard baseline.
    p, q = _quantize_tensor_self_mse(x, return_dequant=True)
    params_list.append(p)
    dq_list.append(q)

    q_best = q
    c_prev = torch.zeros_like(x.mean(dim=-2, keepdim=True))

    for _ in range(1, _K_QUOTIENT_TOTAL_ROUNDS):
        # Exact best translation for the current history-best representative.
        c_star = (x - q_best).mean(dim=-2, keepdim=True)
        delta = c_star - c_prev

        # gamma=1 is exactly the standard V28 update. Larger gammas only add
        # candidates; they can never force a worse block into the final result.
        for gamma in _K_RELAX_GAMMAS:
            c_try = c_prev + float(gamma) * delta
            target = x - c_try
            p, q = _quantize_tensor_self_mse(target, return_dequant=True)
            params_list.append(p)
            dq_list.append(q)

        q_best = _select_best_k_dq_per_feature_block(x, dq_list)
        c_prev = c_star

    # V30: a second deterministic basin seeded only from K itself.  Median
    # centering is robust to outliers and, unlike mean/midrange multi-starts in
    # stress tests, consistently found complementary discrete HiF4 basins.
    # Crucially, all V29 candidates remain in history, so this branch can only
    # win a 64-feature block when its exact softmax-quotient SSE is lower.
    c_prev = x.median(dim=-2, keepdim=True).values
    p, q = _quantize_tensor_self_mse(x - c_prev, return_dequant=True)
    params_list.append(p)
    dq_list.append(q)
    q_best = _select_best_k_dq_per_feature_block(x, dq_list)

    for _ in range(_K_MEDIAN_EXTRA_ROUNDS):
        c_star = (x - q_best).mean(dim=-2, keepdim=True)
        delta = c_star - c_prev
        for gamma in _K_RELAX_GAMMAS:
            c_try = c_prev + float(gamma) * delta
            p, q = _quantize_tensor_self_mse(x - c_try, return_dequant=True)
            params_list.append(p)
            dq_list.append(q)
        q_best = _select_best_k_dq_per_feature_block(x, dq_list)
        c_prev = c_star

    return _merge_k_candidates_per_feature_block(x, params_list, dq_list)



# =============================================================================
# V32: V31 selection + safeguarded Smooth -> DuQuant R1 -> P -> R2 -> H64.
# V30 Attention remains unchanged.
# =============================================================================
_V32_STATE_VERSION = "v32_duquant_then_h64_safe"
_V32_BASE_MODES = (
    (0.0, False), (0.25, False), (0.50, False),
    (0.0, True), (0.25, True), (0.50, True),
)


def _fwht64_v32(x: torch.Tensor) -> torch.Tensor:
    c = int(x.shape[-1])
    if c % 64:
        return x.float()
    original_shape = x.shape
    y = x.float().reshape(-1, c // 64, 64).clone()
    width = 1
    while width < 64:
        z = y.reshape(*y.shape[:-1], -1, 2 * width)
        left = z[..., :width].clone()
        right = z[..., width : 2 * width].clone()
        z[..., :width] = left + right
        z[..., width : 2 * width] = left - right
        y = z.reshape(-1, c // 64, 64)
        width *= 2
    return (y * 0.125).reshape(original_shape)


def _v32_even(n: int, k: int, device: torch.device) -> torch.Tensor:
    if n <= k:
        return torch.arange(n, device=device)
    return torch.linspace(0, n - 1, steps=k, device=device).round().long().unique()


def _v32_smooth(
    activation_absmax: torch.Tensor,
    weight_absmax: torch.Tensor,
    beta: float,
) -> torch.Tensor:
    if beta <= 0:
        return torch.ones_like(weight_absmax)
    log_scale = float(beta) * (
        torch.log(weight_absmax.clamp_min(2**-24))
        - torch.log(activation_absmax.clamp_min(2**-24))
    )
    log_scale -= log_scale.median()
    return torch.exp(log_scale).clamp_(2**-6, 2**6)


def _v32_calib_acts(calib_list: list, channels: int, device: torch.device) -> list:
    result = []
    for pair in calib_list:
        if not isinstance(pair, (list, tuple)) or len(pair) != 2:
            continue
        activation = dequantize_nvfp4(pair[0], pair[1]).float().to(device)
        activation = activation.reshape(-1, channels)
        if activation.numel() and activation.shape[-1] == channels:
            result.append(activation)
    return result


def _v32_choose_base_mode(
    weight: torch.Tensor,
    activations: list[torch.Tensor],
) -> tuple[torch.Tensor, bool, tuple[float, bool]]:
    if not activations:
        return torch.ones(weight.shape[-1], device=weight.device), False, (0.0, False)

    activation_absmax = torch.stack(
        [activation.abs().amax(0) for activation in activations]
    ).amax(0)
    weight_absmax = weight.abs().amax(0)
    weight_rows = _v32_even(weight.shape[0], min(48, weight.shape[0]), weight.device)
    sampled_weight = weight[weight_rows]
    scores = {mode: [] for mode in _V32_BASE_MODES}

    for activation in activations[:4]:
        token_rows = _v32_even(
            activation.shape[0], min(20, activation.shape[0]), activation.device
        )
        sampled_activation = activation[token_rows]
        reference = sampled_activation @ sampled_weight.t()

        for beta, use_hadamard in _V32_BASE_MODES:
            smooth = _v32_smooth(activation_absmax, weight_absmax, beta)
            transformed_activation = sampled_activation * smooth
            transformed_weight = sampled_weight / smooth
            if use_hadamard:
                transformed_activation = _fwht64_v32(transformed_activation)
                transformed_weight = _fwht64_v32(transformed_weight)
            _, activation_q = _quantize_tensor_self_mse(
                transformed_activation, return_dequant=True
            )
            _, weight_q = _quantize_tensor_self_mse(
                transformed_weight, return_dequant=True
            )
            scores[(beta, use_hadamard)].append(
                float((activation_q @ weight_q.t() - reference).square().mean())
            )

    baseline = scores[(0.0, False)]
    best = (1.0, (0.0, False))
    for mode in _V32_BASE_MODES[1:]:
        ratios = [
            candidate / max(base, 1e-20)
            for candidate, base in zip(scores[mode], baseline)
        ]
        pooled = sum(scores[mode]) / max(sum(baseline), 1e-20)
        if max(ratios) <= 0.99 and pooled <= 0.92 and pooled < best[0]:
            best = (pooled, mode)

    beta, use_hadamard = best[1]
    smooth = _v32_smooth(activation_absmax, weight_absmax, beta)
    return smooth, bool(use_hadamard), (float(beta), bool(use_hadamard))


def _v32_householder64(x: torch.Tensor, position: int) -> torch.Tensor:
    """Apply the symmetric 64-wide Householder using its rank-1 form."""
    shape = x.shape
    y = x.float().reshape(*shape[:-1], -1, 64)
    vector = torch.full((64,), -1.0 / 8.0, dtype=torch.float32, device=x.device)
    vector[int(position)] += 1.0
    coefficient = 2.0 * (y * vector).sum(dim=-1, keepdim=True)
    coefficient = coefficient / vector.dot(vector).clamp_min(1e-12)
    return (y - coefficient * vector).reshape(shape)


def _v32_worst_local_position(
    activations: list[torch.Tensor],
    permutation: Optional[torch.Tensor] = None,
) -> int:
    local_score = None
    for activation in activations:
        transformed = activation
        if permutation is not None:
            transformed = transformed.index_select(-1, permutation)
        score = transformed.abs().reshape(
            -1, transformed.shape[-1] // 64, 64
        ).amax(dim=(0, 1))
        local_score = score if local_score is None else torch.maximum(local_score, score)
    return 0 if local_score is None else int(local_score.argmax().item())


def _v32_zigzag_permutation(score: torch.Tensor) -> torch.Tensor:
    order = torch.argsort(score, descending=True)
    groups = order.numel() // 64
    table = torch.empty(groups, 64, dtype=order.dtype, device=order.device)
    for rank in range(order.numel()):
        lane = rank // groups
        offset = rank % groups
        group = offset if lane % 2 == 0 else groups - 1 - offset
        table[group, lane] = order[rank]
    return table.reshape(-1)


def _v32_apply_duquant(
    x: torch.Tensor,
    position1: int,
    permutation: torch.Tensor,
    position2: int,
) -> torch.Tensor:
    x = _v32_householder64(x, position1)
    x = x.index_select(-1, permutation)
    return _v32_householder64(x, position2)


def _v32_build_and_check_duquant_hadamard(
    weight: torch.Tensor,
    activations: list[torch.Tensor],
    smooth: torch.Tensor,
    use_hadamard: bool,
) -> tuple[bool, int, Optional[torch.Tensor], int]:
    """Check Smooth -> R1 -> P -> R2 -> H64 against the selected V31 path."""
    if not activations or weight.shape[-1] % 64:
        return False, 0, None, 0

    smoothed_activations = [activation * smooth for activation in activations]
    smoothed_weight = weight / smooth

    # This is the already-selected V31 candidate and remains the hard fallback.
    base_activations = smoothed_activations
    base_weight = smoothed_weight
    if use_hadamard:
        base_activations = [_fwht64_v32(x) for x in base_activations]
        base_weight = _fwht64_v32(base_weight)

    # The combined candidate intentionally does not use V31's optional pre-H64.
    # It learns DuQuant on the smoothed tensors, then applies one H64 after R2.
    position1 = _v32_worst_local_position(smoothed_activations)
    rotated_activations = [
        _v32_householder64(x, position1) for x in smoothed_activations
    ]
    channel_score = torch.stack(
        [x.abs().amax(dim=0) for x in rotated_activations]
    ).amax(dim=0)
    permutation = _v32_zigzag_permutation(channel_score)
    position2 = _v32_worst_local_position(rotated_activations, permutation)

    weight_rows = _v32_even(weight.shape[0], min(48, weight.shape[0]), weight.device)
    sampled_original_weight = weight[weight_rows]
    sampled_base_weight = base_weight[weight_rows]
    sampled_combined_weight = _fwht64_v32(
        _v32_apply_duquant(
            smoothed_weight[weight_rows], position1, permutation, position2
        )
    )
    _, base_weight_q = _quantize_tensor_self_mse(
        sampled_base_weight, return_dequant=True
    )
    _, combined_weight_q = _quantize_tensor_self_mse(
        sampled_combined_weight, return_dequant=True
    )

    base_errors = []
    combined_errors = []
    for original_activation, base_activation, smoothed_activation in zip(
        activations[:4], base_activations[:4], smoothed_activations[:4]
    ):
        token_rows = _v32_even(
            original_activation.shape[0],
            min(20, original_activation.shape[0]),
            original_activation.device,
        )
        sampled_original_activation = original_activation[token_rows]
        sampled_base_activation = base_activation[token_rows]
        sampled_combined_activation = _fwht64_v32(
            _v32_apply_duquant(
                smoothed_activation[token_rows], position1, permutation, position2
            )
        )
        reference = sampled_original_activation @ sampled_original_weight.t()
        _, base_activation_q = _quantize_tensor_self_mse(
            sampled_base_activation, return_dequant=True
        )
        _, combined_activation_q = _quantize_tensor_self_mse(
            sampled_combined_activation, return_dequant=True
        )
        base_errors.append(
            float((base_activation_q @ base_weight_q.t() - reference).square().mean())
        )
        combined_errors.append(
            float(
                (
                    combined_activation_q @ combined_weight_q.t() - reference
                ).square().mean()
            )
        )

    ratios = [
        candidate / max(base, 1e-20)
        for candidate, base in zip(combined_errors, base_errors)
    ]
    pooled = sum(combined_errors) / max(sum(base_errors), 1e-20)
    enabled = bool(ratios and max(ratios) <= 0.99 and pooled <= 0.92)
    if not enabled:
        return False, 0, None, 0
    return True, position1, permutation, position2


def hif4_calibration_and_quantize_weight(
    weight_quant: torch.Tensor,
    weight_scale: torch.Tensor,
    calib_activation_list: list,
) -> dict:
    weight = dequantize_nvfp4(weight_quant, weight_scale).float()
    activations = _v32_calib_acts(
        calib_activation_list, weight.shape[-1], weight.device
    )
    smooth, use_hadamard, mode = _v32_choose_base_mode(weight, activations)
    use_combined, position1, permutation, position2 = (
        _v32_build_and_check_duquant_hadamard(
            weight, activations, smooth, use_hadamard
        )
    )

    transformed_weight = weight / smooth
    if use_hadamard and not use_combined:
        transformed_weight = _fwht64_v32(transformed_weight)
    if use_combined:
        transformed_weight = _v32_apply_duquant(
            transformed_weight, position1, permutation, position2
        )
        transformed_weight = _fwht64_v32(transformed_weight)

    weight_params, _ = _quantize_tensor_self_mse(
        transformed_weight, return_dequant=False
    )
    state = {
        "version": _V32_STATE_VERSION,
        "smooth": smooth.detach().cpu().float(),
        "hadamard64": bool(use_hadamard and not use_combined),
        "beta": mode[0],
        "duquant_hadamard": use_combined,
    }
    if use_combined:
        state.update(
            {
                "duquant_position1": int(position1),
                "duquant_permutation": permutation.detach().cpu(),
                "duquant_position2": int(position2),
            }
        )
    return {"weight_params": weight_params, "activation_state": state}


def hif4_dynamic_quantize_activation(
    activation_quant: torch.Tensor,
    activation_scale: torch.Tensor,
    activation_state: dict,
) -> dict:
    activation = dequantize_nvfp4(activation_quant, activation_scale).float()
    if isinstance(activation_state, dict) and "smooth" in activation_state:
        smooth = activation_state["smooth"].to(activation.device)
        activation = activation * smooth
        if bool(activation_state.get("duquant_hadamard", False)):
            permutation = activation_state["duquant_permutation"].to(
                activation.device
            )
            activation = _v32_apply_duquant(
                activation,
                int(activation_state["duquant_position1"]),
                permutation,
                int(activation_state["duquant_position2"]),
            )
            activation = _fwht64_v32(activation)
        elif bool(activation_state.get("hadamard64", False)):
            activation = _fwht64_v32(activation)
    return _quantize_tensor_self_mse(activation, return_dequant=False)[0]


def hif4_calibration_attention(
    calib_qkv_list: list,
    q_num_heads: int,
    kv_num_heads: int,
    head_dim: int,
) -> dict:
    common = {"version": _V32_STATE_VERSION, "head_dim": int(head_dim)}
    return {
        "q_state": {**common, "role": "q", "num_heads": int(q_num_heads)},
        "k_state": {**common, "role": "k", "num_heads": int(kv_num_heads)},
        "v_state": {**common, "role": "v", "num_heads": int(kv_num_heads)},
    }


def hif4_dynamic_quantize_q(
    q_quant, q_scale, q_num_heads, head_dim, q_state
):
    return _quantize_nvfp4_operand(q_quant, q_scale)


def hif4_dynamic_quantize_k(
    k_quant, k_scale, kv_num_heads, head_dim, k_state
):
    return _quantize_k_softmax_quotient(k_quant, k_scale)


def hif4_dynamic_quantize_v(
    v_quant, v_scale, kv_num_heads, head_dim, v_state
):
    return _quantize_nvfp4_operand(v_quant, v_scale)
