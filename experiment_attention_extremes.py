#!/usr/bin/env python3
"""Evaluate calibration-only Q/K transform candidates with true GQA."""

from __future__ import annotations

import argparse
import importlib.util
import itertools
import time
from pathlib import Path

import torch

from self_check_ import compute_attention, dequantize_hif4, dequantize_nvfp4


ROOT = Path(__file__).resolve().parent
SOLUTION = ROOT / "solution_collection" / "solution" / "solution.py"


def load_solution():
    spec = importlib.util.spec_from_file_location("extreme_solution", SOLUTION)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def build_state(sol, group, beta: float, k_patterns: tuple[int, ...]):
    q_heads = group["q_num_heads"]
    kv_heads = group["kv_num_heads"]
    head_dim = group["head_dim"]
    decoded = sol._decode_attention_calibration(group["calib"])
    q_scale, k_scale = sol._compute_reciprocal_qk_scales(
        decoded, q_heads, kv_heads, head_dim, beta,
    )
    k_pattern_tensor = torch.tensor(k_patterns, dtype=torch.int8)
    q_pattern_tensor = k_pattern_tensor.repeat_interleave(q_heads // kv_heads)
    common = {
        "version": "attention_extreme_search",
        "enabled": True,
        "head_dim": int(head_dim),
        "transform_kind": "per_head_rot",
        "per_head_rotation": True,
        "beta": float(beta),
    }
    states = {
        "q_state": {
            **common, "role": "q", "scale": q_scale.cpu().float(),
            "head_rotation_patterns": q_pattern_tensor,
        },
        "k_state": {
            **common, "role": "k", "scale": k_scale.cpu().float(),
            "head_rotation_patterns": k_pattern_tensor,
        },
        "v_state": {
            "version": "attention_extreme_search", "enabled": False, "role": "v",
        },
    }
    return sol._attach_attention_covariance_state(
        states, decoded, q_heads, kv_heads, head_dim,
    )


def evaluate(sol, group, states, samples):
    q_heads = group["q_num_heads"]
    kv_heads = group["kv_num_heads"]
    head_dim = group["head_dim"]
    errors = []
    started = time.perf_counter()
    with torch.inference_mode():
        for sample in samples:
            source = {
                role: dequantize_nvfp4(*sample[role]).float()
                for role in ("q", "k", "v")
            }
            params = {
                "q": sol.hif4_dynamic_quantize_q(
                    *sample["q"], q_heads, head_dim, states["q_state"],
                ),
                "k": sol.hif4_dynamic_quantize_k(
                    *sample["k"], kv_heads, head_dim, states["k_state"],
                ),
                "v": sol.hif4_dynamic_quantize_v(
                    *sample["v"], kv_heads, head_dim, states["v_state"],
                ),
            }
            quantized = {
                role: dequantize_hif4(params[role], sample[role][0].shape).float()
                for role in ("q", "k", "v")
            }
            reference = compute_attention(
                source["q"], source["k"], source["v"],
                q_heads, kv_heads, head_dim,
            )
            result = compute_attention(
                quantized["q"], quantized["k"], quantized["v"],
                q_heads, kv_heads, head_dim,
            )
            errors.append(float((reference - result).square().mean()))
    return errors, time.perf_counter() - started


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode", choices=("current", "proxy", "global", "seeds", "per-head", "beta"), required=True,
    )
    args = parser.parse_args()
    sol = load_solution()
    group = torch.load(
        ROOT / "mini_sample" / "attn.pt", weights_only=True, map_location="cpu",
    )[0]
    kv_heads = group["kv_num_heads"]
    patterns = (-1, 0, 1, 2, 3)
    if args.mode == "proxy":
        decoded = sol._decode_attention_calibration(group["calib"])
        q_scale, k_scale = sol._compute_reciprocal_qk_scales(
            decoded, group["q_num_heads"], group["kv_num_heads"],
            group["head_dim"], 0.5,
        )
        for pattern in (-1, 0, 1, 2, 3):
            squared_error = 0.0
            energy = 0.0
            for q, k, _ in decoded:
                for tensor, scale, heads in (
                    (q, q_scale, group["q_num_heads"]),
                    (k, k_scale, group["kv_num_heads"]),
                ):
                    transformed = tensor.float()
                    transformed = transformed * scale if heads == group["q_num_heads"] else transformed / scale
                    transformed = sol._rotate_attention_heads(
                        transformed, heads, group["head_dim"], pattern,
                    )
                    _, dequantized = sol._quantize_tensor_self_mse(
                        transformed, return_dequant=True,
                    )
                    squared_error += float((dequantized - transformed).square().sum())
                    energy += float(transformed.square().sum())
            print(f"pattern={pattern} relative_qk_mse={squared_error / energy:.10e}")
        return
    if args.mode == "current":
        state = sol.hif4_calibration_attention(
            group["calib"], group["q_num_heads"],
            group["kv_num_heads"], group["head_dim"],
        )
        calibration_errors, _ = evaluate(sol, group, state, group["calib"])
        test_errors, elapsed = evaluate(sol, group, state, group["test"])
        print(
            f"current calib_mean={sum(calibration_errors) / len(calibration_errors):.8e} "
            f"test_mean={sum(test_errors) / len(test_errors):.8e} "
            f"test_worst={max(test_errors):.8e} elapsed={elapsed:.3f}s "
            f"test={[f'{value:.8e}' for value in test_errors]}"
        )
        return
    if args.mode == "global":
        candidates = [(0.5, (pattern,) * kv_heads) for pattern in patterns]
    elif args.mode == "seeds":
        candidates = [(0.5, (pattern,) * kv_heads) for pattern in (2, 3)]
    elif args.mode == "per-head":
        candidates = [
            (0.5, combo)
            for combo in itertools.product((0, 1, 2), repeat=kv_heads)
        ]
    else:
        candidates = [(beta, (2,) * kv_heads) for beta in (0.0, 0.25, 0.5, 0.75)]

    for beta, pattern_tuple in candidates:
        state = build_state(sol, group, beta, pattern_tuple)
        calibration_errors, _ = evaluate(sol, group, state, group["calib"])
        test_errors, elapsed = evaluate(sol, group, state, group["test"])
        print(
            f"beta={beta:.2f} patterns={pattern_tuple} "
            f"calib_mean={sum(calibration_errors) / len(calibration_errors):.8e} "
            f"test_mean={sum(test_errors) / len(test_errors):.8e} "
            f"test_worst={max(test_errors):.8e} elapsed={elapsed:.3f}s "
            f"test={[f'{value:.8e}' for value in test_errors]}"
        )


if __name__ == "__main__":
    main()
