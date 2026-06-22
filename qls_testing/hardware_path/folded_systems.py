"""Sparse all-at-once linear systems for coherent time-history preparation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.sparse import bmat, csr_matrix, eye, kron


@dataclass(frozen=True)
class FoldedSystem:
    """One history-state linear system and its exact Kronecker decomposition."""

    matrix: csr_matrix
    rhs: np.ndarray
    steps: int
    state_dimension: int
    method: str
    kronecker_terms: tuple[tuple[complex | float, csr_matrix, csr_matrix, str], ...]

    @property
    def dimension(self) -> int:
        return self.steps * self.state_dimension

    def reconstruct_from_terms(self) -> csr_matrix:
        total = csr_matrix(self.matrix.shape, dtype=self.matrix.dtype)
        for coefficient, time_operator, state_operator, _label in self.kronecker_terms:
            total = total + coefficient * kron(time_operator, state_operator, format="csr")
        return total


@dataclass(frozen=True)
class TerminalPaddedSystem:
    """Folded system followed by coherent copies of its final state block."""

    matrix: csr_matrix
    rhs: np.ndarray
    original_steps: int
    terminal_copies: int
    state_dimension: int

    @property
    def dimension(self) -> int:
        return (self.original_steps + self.terminal_copies) * self.state_dimension


def _shifts(steps: int) -> tuple[csr_matrix, csr_matrix]:
    first = csr_matrix(np.diag(np.ones(max(steps - 1, 0)), -1))
    second = csr_matrix(np.diag(np.ones(max(steps - 2, 0)), -2))
    return first, second


def build_folded_backward_euler(
    matrix: np.ndarray,
    initial_state: np.ndarray,
    dt: float,
    steps: int,
) -> FoldedSystem:
    """Build ``[I⊗(I-hA)-L⊗I]Y=|0>⊗y0``."""
    if steps < 1:
        raise ValueError("steps must be positive")
    operator = csr_matrix(matrix)
    dimension = operator.shape[0]
    state_identity = eye(dimension, format="csr", dtype=operator.dtype)
    time_identity = eye(steps, format="csr", dtype=operator.dtype)
    first_shift, _ = _shifts(steps)
    terms = (
        (1.0, time_identity, state_identity, "I_time tensor I_state"),
        (-dt, time_identity, operator, "-dt I_time tensor A_C"),
        (-1.0, first_shift, state_identity, "-L1 tensor I_state"),
    )
    folded = sum(
        (coefficient * kron(left, right, format="csr") for coefficient, left, right, _ in terms),
        csr_matrix((steps * dimension, steps * dimension), dtype=operator.dtype),
    )
    rhs = np.zeros(steps * dimension, dtype=np.result_type(matrix, initial_state))
    rhs[:dimension] = initial_state
    return FoldedSystem(folded, rhs, steps, dimension, "folded_backward_euler", terms)


def build_folded_crank_nicolson(
    matrix: np.ndarray,
    initial_state: np.ndarray,
    dt: float,
    steps: int,
) -> FoldedSystem:
    """Build the all-at-once trapezoidal history system."""
    if steps < 1:
        raise ValueError("steps must be positive")
    operator = csr_matrix(matrix)
    dimension = operator.shape[0]
    state_identity = eye(dimension, format="csr", dtype=operator.dtype)
    time_identity = eye(steps, format="csr", dtype=operator.dtype)
    first_shift, _ = _shifts(steps)
    terms = (
        (1.0, time_identity, state_identity, "I_time tensor I_state"),
        (-0.5 * dt, time_identity, operator, "-dt/2 I_time tensor A_C"),
        (-1.0, first_shift, state_identity, "-L1 tensor I_state"),
        (-0.5 * dt, first_shift, operator, "-dt/2 L1 tensor A_C"),
    )
    folded = sum(
        (coefficient * kron(left, right, format="csr") for coefficient, left, right, _ in terms),
        csr_matrix((steps * dimension, steps * dimension), dtype=operator.dtype),
    )
    rhs = np.zeros(steps * dimension, dtype=np.result_type(matrix, initial_state))
    rhs[:dimension] = (state_identity + 0.5 * dt * operator) @ initial_state
    return FoldedSystem(folded, rhs, steps, dimension, "folded_crank_nicolson", terms)


def build_folded_bdf2(
    matrix: np.ndarray,
    initial_state: np.ndarray,
    dt: float,
    steps: int,
) -> FoldedSystem:
    """Build BDF2 history equations with one Backward-Euler bootstrap row.

    The unknown is ``Y=(y1,...,yK)``. The first row is Backward Euler; later
    rows use ``(3/2 I-hA)y[k+1]-2y[k]+1/2y[k-1]=0``.
    """
    if steps < 1:
        raise ValueError("steps must be positive")
    operator = csr_matrix(matrix)
    dimension = operator.shape[0]
    state_identity = eye(dimension, format="csr", dtype=operator.dtype)
    time_identity = eye(steps, format="csr", dtype=operator.dtype)
    first_shift, second_shift = _shifts(steps)
    bootstrap_projector = csr_matrix(([1.0], ([0], [0])), shape=(steps, steps))
    # Base diagonal is 3/2 I-hA. The bootstrap correction changes its first
    # block to I-hA by subtracting 1/2 I.
    terms = (
        (1.5, time_identity, state_identity, "3/2 I_time tensor I_state"),
        (-dt, time_identity, operator, "-dt I_time tensor A_C"),
        (-2.0, first_shift, state_identity, "-2 L1 tensor I_state"),
        (0.5, second_shift, state_identity, "+1/2 L2 tensor I_state"),
        (-0.5, bootstrap_projector, state_identity, "BE bootstrap correction"),
    )
    folded = sum(
        (coefficient * kron(left, right, format="csr") for coefficient, left, right, _ in terms),
        csr_matrix((steps * dimension, steps * dimension), dtype=operator.dtype),
    )
    rhs = np.zeros(steps * dimension, dtype=np.result_type(matrix, initial_state))
    rhs[:dimension] = initial_state
    if steps > 1:
        rhs[dimension : 2 * dimension] = -0.5 * initial_state
    return FoldedSystem(folded, rhs, steps, dimension, "folded_bdf2", terms)


def append_terminal_copies(
    folded: FoldedSystem, terminal_copies: int
) -> TerminalPaddedSystem:
    """Append equations ``z1=yK`` and ``z[j]=z[j-1]``.

    This increases the probability of measuring the final-time block from a
    normalized history state without evolving beyond the requested horizon.
    """
    if terminal_copies < 1:
        raise ValueError("terminal_copies must be positive")
    dimension = folded.state_dimension
    padding_identity = eye(terminal_copies * dimension, format="csr")
    padding_shift = kron(
        csr_matrix(np.diag(np.ones(max(terminal_copies - 1, 0)), -1)),
        eye(dimension, format="csr"),
        format="csr",
    )
    lower_right = padding_identity - padding_shift
    coupling = csr_matrix(
        ([-1.0] * dimension,
         (np.arange(dimension), np.arange((folded.steps - 1) * dimension, folded.steps * dimension))),
        shape=(terminal_copies * dimension, folded.dimension),
    )
    matrix = bmat(
        [[folded.matrix, None], [coupling, lower_right]], format="csr"
    )
    rhs = np.concatenate((folded.rhs, np.zeros(terminal_copies * dimension)))
    return TerminalPaddedSystem(
        matrix, rhs, folded.steps, terminal_copies, dimension
    )
