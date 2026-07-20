"""Reference group-wise binary and ternary quantizers."""

from __future__ import annotations

import numpy as np


def groups(values: np.ndarray, group_size: int = 128) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    if values.size % group_size:
        raise ValueError("number of values must be divisible by group_size")
    return values.reshape(-1, group_size)


def binary_rtn(values: np.ndarray, group_size: int = 128) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """MSE-optimal symmetric binary RTN: q=s*sign(w), s=mean(abs(w))."""
    w = groups(values, group_size)
    signs = np.where(w >= 0, 1, -1).astype(np.int8)
    scales = np.mean(np.abs(w), axis=1, dtype=np.float32)
    return (signs * scales[:, None]).reshape(np.asarray(values).shape), scales, signs


def ternary_rtn(
    values: np.ndarray,
    group_size: int = 128,
    iterations: int = 8,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Alternating MSE ternarization with alphabet {-s,0,+s}."""
    w = groups(values, group_size)
    scales = np.maximum(np.mean(np.abs(w), axis=1), np.finfo(np.float32).eps)
    codes = np.zeros_like(w, dtype=np.int8)
    for _ in range(iterations):
        codes = np.where(w > scales[:, None] / 2, 1, np.where(w < -scales[:, None] / 2, -1, 0)).astype(np.int8)
        count = np.maximum(np.count_nonzero(codes, axis=1), 1)
        scales = np.sum(np.abs(w) * (codes != 0), axis=1) / count
    return (codes * scales[:, None]).reshape(np.asarray(values).shape), scales.astype(np.float32), codes


def closed_form_scale(w: np.ndarray, codes: np.ndarray, hessian: np.ndarray | None = None) -> float:
    """Optimal scale for fixed discrete codes under Euclidean or Hessian loss."""
    w = np.asarray(w, dtype=np.float64)
    z = np.asarray(codes, dtype=np.float64)
    if hessian is None:
        numerator = float(z @ w)
        denominator = float(z @ z)
    else:
        h = np.asarray(hessian, dtype=np.float64)
        numerator = float(z @ h @ w)
        denominator = float(z @ h @ z)
    return max(0.0, numerator / max(denominator, np.finfo(np.float64).eps))


def curvature_coordinate_descent(
    w: np.ndarray,
    hessian: np.ndarray,
    alphabet: tuple[int, ...] = (-1, 1),
    sweeps: int = 6,
) -> tuple[np.ndarray, float, float]:
    """Small-group discrete baseline for min (w-sz)^T H (w-sz).

    This is deliberately exact-enough rather than fast: it alternates the
    closed-form scale with coordinate search over a binary or ternary alphabet.
    It is useful for 128-value forensic experiments, not full-model conversion.
    """
    w = np.asarray(w, dtype=np.float64)
    h = np.asarray(hessian, dtype=np.float64)
    if h.shape != (w.size, w.size):
        raise ValueError("hessian shape must match weight group")
    if alphabet == (-1, 1):
        z = np.where(w >= 0, 1, -1).astype(np.int8)
    else:
        scale0 = max(np.mean(np.abs(w)), np.finfo(np.float64).eps)
        z = np.where(w > scale0 / 2, 1, np.where(w < -scale0 / 2, -1, 0)).astype(np.int8)
    scale = closed_form_scale(w, z, h)

    def loss(candidate: np.ndarray, candidate_scale: float) -> float:
        delta = w - candidate_scale * candidate
        return float(delta @ h @ delta)

    current = loss(z, scale)
    for _ in range(sweeps):
        changed = False
        for index in range(w.size):
            best_code, best_scale, best_loss = int(z[index]), scale, current
            for code in alphabet:
                if code == z[index]:
                    continue
                candidate = z.copy()
                candidate[index] = code
                candidate_scale = closed_form_scale(w, candidate, h)
                candidate_loss = loss(candidate, candidate_scale)
                if candidate_loss + 1e-12 < best_loss:
                    best_code, best_scale, best_loss = code, candidate_scale, candidate_loss
            if best_code != z[index]:
                z[index] = best_code
                scale, current = best_scale, best_loss
                changed = True
        if not changed:
            break
    return z, float(scale), float(current)

