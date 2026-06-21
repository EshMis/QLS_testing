"""Algebraic HHL simulator preserving the notebook's preprocessing semantics."""

from __future__ import annotations

import numpy as np

from qls_testing.core.datatypes import SolveResult
from qls_testing.core.interfaces import LinearSolver
from qls_testing.core.utils import residual_metrics


class SpectralHHLSimulator(LinearSolver):
    """Simulate ideal HHL eigenvalue inversion with Hermitian dilation.

    This is a correctness oracle, not a quantum speedup: ``eigh`` classically
    performs the phase-estimation/eigenvalue-inversion work. It keeps the
    normalization, dilation, and cutoff semantics explicit for future circuit
    backends.
    """

    def __init__(self, rcond: float = 1e-12) -> None:
        self.rcond = rcond

    def solve(self, matrix: np.ndarray, rhs: np.ndarray) -> SolveResult:
        original = np.asarray(matrix)
        b = np.asarray(rhs)
        n = original.shape[0]
        if original.shape != (n, n) or b.shape != (n,):
            raise ValueError("HHL simulator expects a square matrix and matching vector")
        is_hermitian = np.allclose(original, original.conj().T)
        if is_hermitian:
            operator, embedded_rhs = original.astype(complex), b.astype(complex)
        else:
            zero = np.zeros_like(original, dtype=complex)
            operator = np.block([[zero, original], [original.conj().T, zero]])
            embedded_rhs = np.concatenate((b, np.zeros(n, dtype=complex)))
        norm = np.linalg.norm(embedded_rhs)
        if norm == 0:
            solution = np.zeros(n, dtype=complex)
            return SolveResult(solution, 0.0, 0.0, {"method": "hhl_simulator"})

        eigenvalues, eigenvectors = np.linalg.eigh(operator)
        cutoff = self.rcond * max(float(np.max(np.abs(eigenvalues))), 1.0)
        if np.any(np.abs(eigenvalues) <= cutoff):
            raise np.linalg.LinAlgError("operator is singular at the configured HHL cutoff")
        normalized_rhs = embedded_rhs / norm
        amplitudes = eigenvectors.conj().T @ normalized_rhs
        full = norm * (eigenvectors @ (amplitudes / eigenvalues))
        solution = full if is_hermitian else full[n:]
        solution = np.real_if_close(solution)
        absolute, relative = residual_metrics(original, solution, b)
        condition = float(np.max(np.abs(eigenvalues)) / np.min(np.abs(eigenvalues)))
        return SolveResult(
            solution,
            absolute,
            relative,
            {"method": "hhl_simulator", "dilated": not is_hermitian, "condition_number": condition},
        )

