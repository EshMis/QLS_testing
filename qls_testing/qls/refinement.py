"""Classical iterative refinement around approximate QLS calls."""

from __future__ import annotations

import numpy as np

from qls_testing.core.datatypes import SolveResult
from qls_testing.core.interfaces import LinearSolver
from qls_testing.core.utils import residual_metrics


class IterativeRefinementSolver(LinearSolver):
    """Repeatedly solve residual equations with an approximate base solver."""

    def __init__(self, base_solver: LinearSolver, max_iterations: int = 3, tolerance: float = 1e-10) -> None:
        self.base_solver = base_solver
        self.max_iterations = max_iterations
        self.tolerance = tolerance

    def solve(self, matrix: np.ndarray, rhs: np.ndarray) -> SolveResult:
        initial = self.base_solver.solve(matrix, rhs)
        solution = np.asarray(initial.solution).copy()
        history = [initial.relative_residual]
        correction_metadata = []
        for _ in range(self.max_iterations):
            residual = np.asarray(rhs) - np.asarray(matrix) @ solution
            _, relative = residual_metrics(matrix, solution, rhs)
            if relative <= self.tolerance:
                break
            correction = self.base_solver.solve(matrix, residual)
            solution += correction.solution
            _, updated = residual_metrics(matrix, solution, rhs)
            history.append(updated)
            correction_metadata.append(correction.metadata)
            if updated >= history[-2]:
                break
        absolute, relative = residual_metrics(matrix, solution, rhs)
        return SolveResult(
            np.real_if_close(solution),
            absolute,
            relative,
            {
                "method": "iterative_refinement",
                "base_method": initial.metadata.get("method"),
                "residual_history": history,
                "iterations": len(history) - 1,
                "correction_metadata": correction_metadata,
            },
        )

