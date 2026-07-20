"""Activation/curvature-aware discrete matrix reconstruction laboratory."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import numpy as np

from .quantize import binary_rtn, curvature_coordinate_descent, ternary_rtn


def evaluate_matrix(weight: np.ndarray, activations: np.ndarray, group_size: int = 128) -> dict:
    """Compare RTN and curvature-aware discrete reconstruction on one weight matrix.

    `weight` is [out_features, in_features]. `activations` is
    [samples, in_features]. Coordinate descent is intentionally applied to the
    first output row by default; scaling it to every row is a baseline, not the
    proposed production algorithm.
    """
    w = np.asarray(weight, dtype=np.float32)
    x = np.asarray(activations, dtype=np.float32)
    if w.ndim != 2 or x.ndim != 2 or w.shape[1] != x.shape[1]:
        raise ValueError("expected weight[out,in] and activations[samples,in]")
    if w.shape[1] % group_size:
        raise ValueError("in_features must be divisible by group_size")
    binary, _, _ = binary_rtn(w, group_size)
    ternary, _, _ = ternary_rtn(w, group_size)
    teacher = x @ w.T

    def metrics(candidate: np.ndarray) -> dict:
        output = x @ candidate.T
        return {
            "weight_relative_error": float(np.linalg.norm(candidate - w) / np.linalg.norm(w)),
            "output_relative_error": float(np.linalg.norm(output - teacher) / np.linalg.norm(teacher)),
        }

    result = {"binary_rtn": metrics(binary), "ternary_rtn": metrics(ternary)}
    h = (x.T @ x) / max(len(x), 1)
    row = 0
    optimized_binary = binary[row].copy()
    optimized_ternary = ternary[row].copy()
    for start in range(0, w.shape[1], group_size):
        stop = start + group_size
        h_group = h[start:stop, start:stop]
        z, scale, _ = curvature_coordinate_descent(w[row, start:stop], h_group, (-1, 1))
        optimized_binary[start:stop] = z * scale
        z, scale, _ = curvature_coordinate_descent(w[row, start:stop], h_group, (-1, 0, 1))
        optimized_ternary[start:stop] = z * scale
    teacher_row = teacher[:, row]
    result["first_row_curvature_binary"] = {
        "weight_relative_error": float(np.linalg.norm(optimized_binary - w[row]) / np.linalg.norm(w[row])),
        "output_relative_error": float(np.linalg.norm(x @ optimized_binary - teacher_row) / np.linalg.norm(teacher_row)),
    }
    result["first_row_curvature_ternary"] = {
        "weight_relative_error": float(np.linalg.norm(optimized_ternary - w[row]) / np.linalg.norm(w[row])),
        "output_relative_error": float(np.linalg.norm(x @ optimized_ternary - teacher_row) / np.linalg.norm(teacher_row)),
    }
    return result


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("npz", type=Path, help="NPZ containing arrays `weight` and `activations`")
    parser.add_argument("--group-size", type=int, default=128)
    args = parser.parse_args(argv)
    data = np.load(args.npz)
    print(json.dumps(evaluate_matrix(data["weight"], data["activations"], args.group_size), indent=2))


if __name__ == "__main__":
    main()

