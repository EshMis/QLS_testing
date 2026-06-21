"""Deterministic reference linear solver."""

from __future__ import annotations

import numpy as np

from qls_testing.core.datatypes import SolveResult
from qls_testing.core.interfaces import LinearSolver
from qls_testing.core.utils import residual_metrics


class ClassicalSolver(LinearSolver):
    """Solve with LAPACK through :func:`numpy.linalg.solve`."""

    def solve(self, matrix: np.ndarray, rhs: np.ndarray) -> SolveResult:
        solution = np.linalg.solve(matrix, rhs)
        absolute, relative = residual_metrics(matrix, solution, rhs)
        return SolveResult(solution, absolute, relative, {"method": "classical"})

