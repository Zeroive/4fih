"""HiF4 v2: select channel regrouping with real quantized output loss.

In addition to v1's alpha search, v2 compares identity, magnitude sorting and
zigzag balancing.  The winner is selected after quantizing both calibration
activations and weights, not from an RMS/P99 proxy.
"""
from __future__ import annotations

from typing import Any

import torch

import hif4_solution_v0 as _v0
import hif4_solution_v1 as _v1


dequantize_nvfp4 = _v0.dequantize_nvfp4
hif4_dynamic_quantize_activation = _v0.hif4_dynamic_quantize_activation
hif4_calibration_attention = _v0.hif4_calibration_attention
hif4_dynamic_quantize_q = _v0.hif4_dynamic_quantize_q
hif4_dynamic_quantize_k = _v0.hif4_dynamic_quantize_k
hif4_dynamic_quantize_v = _v0.hif4_dynamic_quantize_v


def _zigzag_permutation(order: torch.Tensor) -> torch.Tensor:
    """Distribute alternating low/high-score channels across 64-wide groups."""
    k = order.numel()
    if k % 64 != 0:
        return order
    groups = k // 64
    lo, hi = 0, k - 1
    table = torch.empty(groups, 64, dtype=order.dtype, device=order.device)
    # Fill the same within-group position across groups. Reversing every other
    # lane prevents one group from consistently receiving the extremes first.
    for lane in range(64):
        vals = []
        for _ in range(groups):
            if lane % 2 == 0:
                vals.append(order[hi])
                hi -= 1
            else:
                vals.append(order[lo])
                lo += 1
        lane_values = torch.stack(vals)
        if lane % 4 >= 2:
            lane_values = lane_values.flip(0)
        table[:, lane] = lane_values
    return table.reshape(-1)


def hif4_calibration_and_quantize_weight(
    weight_quant: torch.Tensor,
    weight_scale: torch.Tensor,
    calib_activation_list: list,
) -> dict[str, Any]:
    W = dequantize_nvfp4(weight_quant, weight_scale).float()
    X = _v1._sample_calibration(calib_activation_list, W.device)
    score = (
        0.70 * torch.log2(torch.sqrt(W.square().mean(dim=0) + _v0.EPS))
        + 0.30 * torch.log2(torch.sqrt(X.square().mean(dim=0) + _v0.EPS))
    )
    magnitude = torch.argsort(score)
    if _v0.USE_LINEAR_PERMUTE:
        permutations = (
            ("identity", None),
            ("magnitude_sort", magnitude),
            ("zigzag_balance", _zigzag_permutation(magnitude)),
        )
    else:
        permutations = (("identity", None),)
    return _v1._calibrate_with_permutations(W, X, permutations)
