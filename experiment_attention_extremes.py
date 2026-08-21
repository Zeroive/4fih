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


def enable_post_rotation_permutation(sol):
    """Teach the experiment module to apply a shared Q/K head permutation."""
    original_query_transform = sol._apply_query_transform
    original_key_transform = sol._apply_key_transform

    def apply_query(x, state, num_heads, head_dim):
        y = original_query_transform(x, state, num_heads, head_dim)
        return sol._apply_per_head_permutation(
            y, num_heads, head_dim, state.get("post_rotation_permutation"),
        )

    def apply_key(x, state, num_heads, head_dim):
        y = original_key_transform(x, state, num_heads, head_dim)
        return sol._apply_per_head_permutation(
            y, num_heads, head_dim, state.get("post_rotation_permutation"),
        )

    sol._apply_query_transform = apply_query
    sol._apply_key_transform = apply_key


def _base_attention_states(sol, decoded, q_heads, kv_heads, head_dim, q_perm, k_perm):
    q_scale, k_scale = sol._compute_reciprocal_qk_scales(
        decoded, q_heads, kv_heads, head_dim, 0.5,
    )
    common = {
        "version": "attention_twopass_permutation",
        "enabled": True,
        "head_dim": int(head_dim),
        "transform_kind": "per_head_rot",
        "per_head_rotation": True,
        "beta": 0.5,
    }
    return {
        "q_state": {
            **common, "role": "q", "scale": q_scale.cpu().float(),
            "head_rotation_patterns": torch.full((q_heads,), 2, dtype=torch.int8),
            "post_rotation_permutation": q_perm.cpu().to(torch.int32),
        },
        "k_state": {
            **common, "role": "k", "scale": k_scale.cpu().float(),
            "head_rotation_patterns": torch.full((kv_heads,), 2, dtype=torch.int8),
            "post_rotation_permutation": k_perm.cpu().to(torch.int32),
        },
        "v_state": {
            "version": "attention_twopass_permutation", "enabled": False, "role": "v",
        },
    }


def _first_pass_outlier_permutation(sol, decoded, states, q_heads, kv_heads, head_dim):
    queries_per_kv = q_heads // kv_heads
    sample_scores = []
    for q, k, _ in decoded:
        qt = sol._apply_query_transform(q, states["q_state"], q_heads, head_dim)
        kt = sol._apply_key_transform(k, states["k_state"], kv_heads, head_dim)
        qh = qt.reshape(-1, kv_heads, queries_per_kv, head_dim)
        kh = kt.reshape(-1, kv_heads, head_dim)
        q_rms = qh.square().mean(dim=(0, 2)).sqrt().clamp_min(2.0 ** -24)
        k_rms = kh.square().mean(dim=0).sqrt().clamp_min(2.0 ** -24)
        q_tail = qh.abs().amax(dim=(0, 2)) / q_rms
        k_tail = kh.abs().amax(dim=0) / k_rms
        sample_scores.append(torch.maximum(q_tail, k_tail).log())
    robust_score = torch.stack(sample_scores).median(dim=0).values
    return torch.argsort(robust_score, dim=-1, descending=True, stable=True)


def _hessian_residual_score(error, hessian):
    gradient = torch.einsum("hde,she->shd", hessian.float(), error.float())
    return (error.float() * gradient).abs().mean(dim=0)


def _second_pass_residual_permutation(
    sol, decoded, states, q_heads, kv_heads, head_dim,
):
    queries_per_kv = q_heads // kv_heads
    q_hessian = states["q_state"]["partner_h256"].float()
    k_hessian = states["k_state"]["partner_h256"].float()
    residual_scores = []
    for q, k, _ in decoded:
        # Re-encode the already decoded calibration tensors through the same
        # internal quantizers used online, without relying on NVFP4 containers.
        qt = sol._apply_query_transform(q, states["q_state"], q_heads, head_dim)
        qp = sol._quantize_dynamic_tensor_hessian(
            qt, states["q_state"]["partner_h64"], q_hessian,
            lv3_iters=1, base_mant_iters=1,
        )
        qdq = sol._dequantize_hif4_params(qp, tuple(qt.shape))
        qe = (qdq - qt).reshape(-1, q_heads, head_dim)
        q_score = _hessian_residual_score(qe, q_hessian).reshape(
            kv_heads, queries_per_kv, head_dim,
        ).mean(dim=1)

        kt = sol._apply_key_transform(k, states["k_state"], kv_heads, head_dim)
        kp = sol._quantize_key_with_partner_metric(
            kt, k_hessian, kv_heads, head_dim,
        )
        kdq = sol._dequantize_hif4_params(kp, tuple(kt.shape))
        ke = (kdq - kt).reshape(-1, kv_heads, head_dim)
        ke = ke - ke.mean(dim=0, keepdim=True)
        k_score = _hessian_residual_score(ke, k_hessian)

        q_score = q_score / q_score.mean(dim=-1, keepdim=True).clamp_min(1e-20)
        k_score = k_score / k_score.mean(dim=-1, keepdim=True).clamp_min(1e-20)
        residual_scores.append(0.5 * (q_score + k_score))
    score = torch.stack(residual_scores).mean(dim=0)
    return torch.stack([
        sol._build_balanced_feature_permutation(head_score)
        for head_score in score
    ])


def build_two_pass_permutation_state(sol, group):
    q_heads = group["q_num_heads"]
    kv_heads = group["kv_num_heads"]
    head_dim = group["head_dim"]
    queries_per_kv = q_heads // kv_heads
    decoded = sol._decode_attention_calibration(group["calib"])
    identity_k = torch.arange(head_dim).repeat(kv_heads, 1)
    identity_q = identity_k.repeat_interleave(queries_per_kv, dim=0)

    identity_states = _base_attention_states(
        sol, decoded, q_heads, kv_heads, head_dim, identity_q, identity_k,
    )
    first_k = _first_pass_outlier_permutation(
        sol, decoded, identity_states, q_heads, kv_heads, head_dim,
    )
    first_q = first_k.repeat_interleave(queries_per_kv, dim=0)
    first_states = _base_attention_states(
        sol, decoded, q_heads, kv_heads, head_dim, first_q, first_k,
    )
    first_states = sol._attach_attention_covariance_state(
        first_states, decoded, q_heads, kv_heads, head_dim,
    )

    second_k = _second_pass_residual_permutation(
        sol, decoded, first_states, q_heads, kv_heads, head_dim,
    )
    final_k = first_k.gather(1, second_k)
    final_q = final_k.repeat_interleave(queries_per_kv, dim=0)
    final_states = _base_attention_states(
        sol, decoded, q_heads, kv_heads, head_dim, final_q, final_k,
    )
    return sol._attach_attention_covariance_state(
        final_states, decoded, q_heads, kv_heads, head_dim,
    )


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
        "--mode", choices=(
            "current", "proxy", "global", "seeds", "per-head", "beta",
            "two-pass-perm",
        ), required=True,
    )
    args = parser.parse_args()
    sol = load_solution()
    group = torch.load(
        ROOT / "mini_sample" / "attn.pt", weights_only=True, map_location="cpu",
    )[0]
    kv_heads = group["kv_num_heads"]
    patterns = (-1, 0, 1, 2, 3)
    if args.mode == "two-pass-perm":
        enable_post_rotation_permutation(sol)
        started = time.perf_counter()
        state = build_two_pass_permutation_state(sol, group)
        calibration_time = time.perf_counter() - started
        calibration_errors, _ = evaluate(sol, group, state, group["calib"])
        test_errors, elapsed = evaluate(sol, group, state, group["test"])
        print(
            f"two_pass_perm calib_time={calibration_time:.3f}s "
            f"calib_mean={sum(calibration_errors) / len(calibration_errors):.8e} "
            f"test_mean={sum(test_errors) / len(test_errors):.8e} "
            f"test_worst={max(test_errors):.8e} elapsed={elapsed:.3f}s "
            f"test={[f'{value:.8e}' for value in test_errors]}"
        )
        return
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
