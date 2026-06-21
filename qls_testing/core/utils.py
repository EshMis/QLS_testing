"""Small numerical utilities."""

from __future__ import annotations

import numpy as np


def residual_metrics(matrix: np.ndarray, solution: np.ndarray, rhs: np.ndarray) -> tuple[float, float]:
    residual = np.asarray(matrix) @ np.asarray(solution) - np.asarray(rhs)
    absolute = float(np.linalg.norm(residual))
    relative = absolute / max(float(np.linalg.norm(rhs)), np.finfo(float).eps)
    return absolute, relative


def uniform_grid(t_final: float, dt: float) -> np.ndarray:
    """Return a grid ending exactly at ``t_final``; reject inconsistent steps."""
    if t_final <= 0 or dt <= 0:
        raise ValueError("t_final and dt must be positive")
    steps = round(t_final / dt)
    if steps < 1 or not np.isclose(steps * dt, t_final, rtol=1e-10, atol=1e-12):
        raise ValueError("t_final must be an integer multiple of dt")
    return np.linspace(0.0, t_final, steps + 1)

