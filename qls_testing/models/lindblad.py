"""Lindblad master-equation models, deliberately separate from QLS."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import expm_multiply


@dataclass(frozen=True)
class LindbladModel:
    """Finite-dimensional GKSL model ``rho_dot = L(rho)``."""

    hamiltonian: np.ndarray
    jump_operators: tuple[np.ndarray, ...]
    initial_density: np.ndarray
    labels: tuple[str, ...] = ("population_0", "population_1")

    def liouvillian(self) -> csr_matrix:
        """Build the column-vectorized Liouvillian superoperator."""
        h = np.asarray(self.hamiltonian, dtype=complex)
        dimension = h.shape[0]
        identity = np.eye(dimension, dtype=complex)
        operator = -1j * (np.kron(identity, h) - np.kron(h.T, identity))
        for jump in self.jump_operators:
            jump = np.asarray(jump, dtype=complex)
            gram = jump.conj().T @ jump
            operator += np.kron(jump.conj(), jump)
            operator -= 0.5 * np.kron(identity, gram)
            operator -= 0.5 * np.kron(gram.T, identity)
        return csr_matrix(operator)


@dataclass(frozen=True)
class LindbladResult:
    times: np.ndarray
    density_matrices: np.ndarray
    populations: np.ndarray
    trace_error: np.ndarray
    minimum_eigenvalue: np.ndarray
    metadata: dict[str, object]


def amplitude_damping_model(decay_rate: float = 1.0) -> LindbladModel:
    """Qubit amplitude damping from ``|1>`` to ``|0>``."""
    if decay_rate < 0:
        raise ValueError("decay_rate must be nonnegative")
    jump = np.sqrt(decay_rate) * np.asarray([[0.0, 1.0], [0.0, 0.0]])
    return LindbladModel(
        hamiltonian=np.zeros((2, 2)),
        jump_operators=(jump,),
        initial_density=np.asarray([[0.0, 0.0], [0.0, 1.0]], dtype=complex),
    )


def simulate_lindblad(model: LindbladModel, t_final: float, n_points: int) -> LindbladResult:
    """Evolve a density matrix with sparse exponential action.

    This solves a master equation; it does not invoke a quantum linear solver.
    """
    times = np.linspace(0.0, t_final, n_points)
    initial = np.asarray(model.initial_density, dtype=complex).reshape(-1, order="F")
    vectors = expm_multiply(model.liouvillian(), initial, start=0.0, stop=t_final, num=n_points)
    densities = np.asarray([vector.reshape(model.initial_density.shape, order="F") for vector in vectors])
    populations = np.real(np.diagonal(densities, axis1=1, axis2=2))
    trace_error = np.abs(np.trace(densities, axis1=1, axis2=2) - 1.0)
    minimum_eigenvalue = np.asarray([np.min(np.linalg.eigvalsh(0.5 * (rho + rho.conj().T))) for rho in densities])
    return LindbladResult(
        times,
        densities,
        populations,
        trace_error,
        minimum_eigenvalue,
        {"pathway": "lindblad", "liouvillian_dimension": model.liouvillian().shape[0]},
    )

