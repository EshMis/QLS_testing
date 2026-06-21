"""Lindblad master-equation models, deliberately separate from QLS."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import expm_multiply
from scipy.integrate import solve_ivp
from scipy.linalg import block_diag


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


@dataclass(frozen=True)
class NDMELindbladResult:
    """Full-density simulation and extracted linear-ODE state from NDME."""

    times: np.ndarray
    density_matrices: np.ndarray
    encoded_states: np.ndarray
    reference_states: np.ndarray
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


def simulate_ndme_linear_ode(
    matrix: np.ndarray,
    initial_state: np.ndarray,
    times: np.ndarray,
    *,
    shift_epsilon: float = 1e-9,
    rtol: float = 1e-9,
    atol: float = 1e-11,
) -> NDMELindbladResult:
    """Simulate the PDF's nondiagonal density-matrix encoding of ``mu'=A mu``.

    The paper writes ``mu'=-V mu``. Its construction requires
    ``B=(V+V†)/2 >= 0`` and ``H1=(V-V†)/(2i)``. If needed, ``V`` is shifted by
    ``kappa I`` to make ``B`` positive semidefinite; extraction multiplies by
    ``exp(kappa t)`` to recover the original ODE solution.
    """
    operator = np.asarray(matrix, dtype=complex)
    initial = np.asarray(initial_state, dtype=complex)
    dimension = operator.shape[0]
    if operator.shape != (dimension, dimension) or initial.shape != (dimension,):
        raise ValueError("matrix must be square and match initial_state")
    initial_norm = float(np.linalg.norm(initial))
    if initial_norm == 0:
        raise ValueError("NDME requires a nonzero initial state")

    v_operator = -operator
    hermitian_part = 0.5 * (v_operator + v_operator.conj().T)
    antihermitian_hamiltonian = (v_operator - v_operator.conj().T) / (2j)
    eigenvalues = np.linalg.eigvalsh(hermitian_part)
    shift = max(0.0, -float(eigenvalues[0]) + shift_epsilon)
    shifted_part = hermitian_part + shift * np.eye(dimension)
    shifted_eigenvalues, eigenvectors = np.linalg.eigh(shifted_part)
    shifted_eigenvalues = np.clip(shifted_eigenvalues, 0.0, None)
    jump_left = (
        eigenvectors
        @ np.diag(np.sqrt(2.0 * shifted_eigenvalues))
        @ eigenvectors.conj().T
    )

    zero = np.zeros_like(operator)
    hamiltonian = block_diag(antihermitian_hamiltonian, zero)
    jump = block_diag(jump_left, zero)
    normalized_initial = initial / initial_norm
    projector = np.outer(normalized_initial, normalized_initial.conj())
    density_initial = 0.5 * np.block([[projector, projector], [projector, projector]])
    full_dimension = 2 * dimension

    def rhs(_time: float, vector: np.ndarray) -> np.ndarray:
        density = vector.reshape((full_dimension, full_dimension))
        gram = jump.conj().T @ jump
        derivative = -1j * (hamiltonian @ density - density @ hamiltonian)
        derivative += jump @ density @ jump.conj().T
        derivative -= 0.5 * (gram @ density + density @ gram)
        return derivative.reshape(-1)

    solution = solve_ivp(
        rhs,
        (float(times[0]), float(times[-1])),
        density_initial.reshape(-1),
        t_eval=np.asarray(times),
        method="DOP853",
        rtol=rtol,
        atol=atol,
    )
    if not solution.success:
        raise RuntimeError(solution.message)
    densities = solution.y.T.reshape((-1, full_dimension, full_dimension))
    encoded_states = []
    for time, density in zip(times, densities):
        upper_right = density[:dimension, dimension:]
        shifted_state = 2.0 * upper_right @ normalized_initial
        encoded_states.append(initial_norm * np.exp(shift * time) * shifted_state)
    encoded = np.real_if_close(np.asarray(encoded_states))
    reference = np.asarray([expm_multiply(time * csr_matrix(operator), initial) for time in times])
    trace_error = np.abs(np.trace(densities, axis1=1, axis2=2) - 1.0)
    minimum_eigenvalue = np.asarray(
        [np.min(np.linalg.eigvalsh(0.5 * (rho + rho.conj().T))) for rho in densities]
    )
    generator_error = np.linalg.norm(
        (-1j * antihermitian_hamiltonian - 0.5 * jump_left.conj().T @ jump_left)
        - (operator - shift * np.eye(dimension))
    )
    return NDMELindbladResult(
        np.asarray(times),
        densities,
        encoded,
        np.real_if_close(reference),
        trace_error,
        minimum_eigenvalue,
        {
            "pathway": "lindblad_ndme",
            "source": "Shang et al., PRL 135, 120604 (2025), Eqs. (2)-(5)",
            "ode_dimension": dimension,
            "density_dimension": full_dimension,
            "liouville_dimension": full_dimension**2,
            "semidissipative_min_eigenvalue_before_shift": float(eigenvalues[0]),
            "semidissipative_shift": shift,
            "jump_rank": int(np.count_nonzero(shifted_eigenvalues > shift_epsilon)),
            "generator_identity_error": float(generator_error),
            "reduced_model": "Carleman order 1 enzyme pathway",
        },
    )
