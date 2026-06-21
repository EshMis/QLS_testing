"""Preconditioned wrappers for approximate and QSVT-style solvers."""

from __future__ import annotations

import numpy as np

from qls_testing.core.datatypes import SolveResult
from qls_testing.core.interfaces import LinearSolver
from qls_testing.core.utils import residual_metrics


class DiagonalPreconditionedSolver(LinearSolver):
    """Apply Jacobi left preconditioning before invoking a base solver."""

    def __init__(self, base_solver: LinearSolver, diagonal_tolerance: float = 1e-12) -> None:
        self.base_solver = base_solver
        self.diagonal_tolerance = diagonal_tolerance

    def solve(self, matrix: np.ndarray, rhs: np.ndarray) -> SolveResult:
        operator = np.asarray(matrix)
        diagonal = np.diag(operator)
        if np.any(np.abs(diagonal) <= self.diagonal_tolerance):
            raise np.linalg.LinAlgError("Jacobi preconditioner has a zero diagonal entry")
        inverse_diagonal = 1.0 / diagonal
        conditioned_matrix = inverse_diagonal[:, None] * operator
        conditioned_rhs = inverse_diagonal * np.asarray(rhs)
        base_result = self.base_solver.solve(conditioned_matrix, conditioned_rhs)
        solution = base_result.solution
        absolute, relative = residual_metrics(operator, solution, rhs)
        return SolveResult(
            solution,
            absolute,
            relative,
            {
                "method": "diagonal_preconditioned",
                "base_metadata": base_result.metadata,
                "condition_before": float(np.linalg.cond(operator)),
                "condition_after": float(np.linalg.cond(conditioned_matrix)),
            },
        )

