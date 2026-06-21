"""Numerically stable singular-value polynomial inversion simulator."""

from __future__ import annotations

import numpy as np
from numpy.polynomial import Chebyshev

from qls_testing.core.datatypes import SolveResult
from qls_testing.core.interfaces import LinearSolver
from qls_testing.core.utils import residual_metrics


class QSVTPolynomialSimulator(LinearSolver):
    """Approximate ``1/x`` on the matrix singular values with a Chebyshev series.

    SVD is used to apply the polynomial, so this is a circuit-independent QSVT
    surrogate. Unlike the notebook path, it preserves Chebyshev coefficients
    and reports approximation error and residual separately.
    """

    def __init__(self, degree: int = 25, rcond: float = 1e-12) -> None:
        if degree < 1:
            raise ValueError("degree must be positive")
        self.degree = degree
        self.rcond = rcond

    def solve(self, matrix: np.ndarray, rhs: np.ndarray) -> SolveResult:
        a = np.asarray(matrix)
        b = np.asarray(rhs)
        u, singular_values, vh = np.linalg.svd(a, full_matrices=False)
        smax = float(singular_values[0])
        smin = float(singular_values[-1])
        if smin <= self.rcond * smax:
            raise np.linalg.LinAlgError("matrix is singular at the configured QSVT cutoff")
        inverse = Chebyshev.interpolate(lambda value: 1.0 / value, self.degree, domain=[smin, smax])
        inverse_values = inverse(singular_values)
        solution = vh.conj().T @ (inverse_values * (u.conj().T @ b))
        solution = np.real_if_close(solution)
        absolute, relative = residual_metrics(a, solution, b)
        max_inverse_error = float(np.max(np.abs(inverse_values - 1.0 / singular_values)))
        return SolveResult(
            solution,
            absolute,
            relative,
            {
                "method": "qsvt_simulator",
                "degree": self.degree,
                "condition_number": smax / smin,
                "max_inverse_error": max_inverse_error,
            },
        )

