"""Small-system variational linear-solver surrogate."""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize

from qls_testing.core.datatypes import SolveResult
from qls_testing.core.interfaces import LinearSolver
from qls_testing.core.utils import residual_metrics


class VariationalLinearSolver(LinearSolver):
    """Minimize a normalized-state VQLS objective for small real systems.

    The ansatz is an unconstrained real vector normalized inside the objective;
    its optimal scalar is computed analytically. This exercises VQLS loss and
    scaling without claiming to simulate gate expressivity or hardware noise.
    """

    def __init__(self, max_iterations: int = 1000, tolerance: float = 1e-10, seed: int = 42) -> None:
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.seed = seed

    def solve(self, matrix: np.ndarray, rhs: np.ndarray) -> SolveResult:
        a = np.asarray(matrix, dtype=float)
        b = np.asarray(rhs, dtype=float)
        if np.linalg.norm(b) == 0:
            return SolveResult(np.zeros_like(b), 0.0, 0.0, {"method": "vqls_simulator"})
        rng = np.random.default_rng(self.seed)
        initial = rng.normal(size=b.size)

        def reconstruct(parameters: np.ndarray) -> np.ndarray:
            state = parameters / max(np.linalg.norm(parameters), np.finfo(float).eps)
            image = a @ state
            scale = np.dot(image, b) / max(np.dot(image, image), np.finfo(float).eps)
            return scale * state

        def objective(parameters: np.ndarray) -> float:
            residual = a @ reconstruct(parameters) - b
            return float(np.dot(residual, residual) / np.dot(b, b))

        result = minimize(
            objective,
            initial,
            method="BFGS",
            options={"maxiter": self.max_iterations, "gtol": self.tolerance},
        )
        solution = reconstruct(result.x)
        absolute, relative = residual_metrics(a, solution, b)
        return SolveResult(
            solution,
            absolute,
            relative,
            {"method": "vqls_simulator", "success": bool(result.success), "iterations": result.nit, "cost": result.fun},
        )

