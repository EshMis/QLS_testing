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
    trace_error: np.ndarray
    minimum_eigenvalue: np.ndarray
    metadata: dict[str, object]


@dataclass(frozen=True)
class NDMEEncoding:
    """Operators and extraction data for the paper's nondiagonal encoding."""

    operator: np.ndarray
    hamiltonian: np.ndarray
    jump: np.ndarray
    initial_density: np.ndarray
    normalized_initial: np.ndarray
    initial_norm: float
    shift: float
    metadata: dict[str, object]

    def model(self) -> LindbladModel:
        dimension = self.operator.shape[0]
        labels = tuple(f"ndme_{index}" for index in range(2 * dimension))
        return LindbladModel(self.hamiltonian, (self.jump,), self.initial_density, labels)

    def extract_states(self, densities: np.ndarray, times: np.ndarray) -> np.ndarray:
        dimension = self.operator.shape[0]
        states = []
        for time, density in zip(times, densities):
            upper_right = density[:dimension, dimension : 2 * dimension]
            shifted_state = 2.0 * upper_right @ self.normalized_initial
            states.append(self.initial_norm * np.exp(self.shift * time) * shifted_state)
        return np.real_if_close(np.asarray(states))


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


def _first_order_cptp_kraus(model: LindbladModel, step: float) -> tuple[np.ndarray, ...]:
    """Build a normalized first-order Kraus approximation to one GKSL step."""
    hamiltonian = np.asarray(model.hamiltonian, dtype=complex)
    identity = np.eye(hamiltonian.shape[0], dtype=complex)
    gram = sum(
        (np.asarray(jump, dtype=complex).conj().T @ np.asarray(jump, dtype=complex)
         for jump in model.jump_operators),
        np.zeros_like(hamiltonian),
    )
    operators = [identity - step * (1j * hamiltonian + 0.5 * gram)]
    operators.extend(np.sqrt(step) * np.asarray(jump, dtype=complex) for jump in model.jump_operators)
    completeness = sum((operator.conj().T @ operator for operator in operators), np.zeros_like(identity))
    eigenvalues, eigenvectors = np.linalg.eigh(0.5 * (completeness + completeness.conj().T))
    if np.min(eigenvalues) <= 0.0:
        raise ValueError("time step is too large for the normalized first-order channel")
    inverse_root = eigenvectors @ np.diag(eigenvalues**-0.5) @ eigenvectors.conj().T
    return tuple(operator @ inverse_root for operator in operators)


def _pad_lindblad_model(model: LindbladModel) -> tuple[LindbladModel, int]:
    original_dimension = model.hamiltonian.shape[0]
    qubits = max(1, int(np.ceil(np.log2(original_dimension))))
    padded_dimension = 2**qubits
    if padded_dimension == original_dimension:
        return model, original_dimension
    hamiltonian = np.zeros((padded_dimension, padded_dimension), dtype=complex)
    hamiltonian[:original_dimension, :original_dimension] = model.hamiltonian
    jumps = []
    for jump in model.jump_operators:
        padded = np.zeros_like(hamiltonian)
        padded[:original_dimension, :original_dimension] = jump
        jumps.append(padded)
    density = np.zeros_like(hamiltonian)
    density[:original_dimension, :original_dimension] = model.initial_density
    return LindbladModel(hamiltonian, tuple(jumps), density, model.labels), original_dimension


def simulate_lindblad_pennylane(
    model: LindbladModel,
    times: np.ndarray,
    *,
    substeps: int = 4,
    backend: str = "default.mixed",
) -> LindbladResult:
    """Apply short-time CPTP channels through PennyLane's mixed-state device.

    The channel is a normalized first-order Kraus approximation. This is a
    genuine PennyLane channel execution path, while exact Liouvillian evolution
    remains the independent ground truth.
    """
    if substeps < 1:
        raise ValueError("substeps must be positive")
    import pennylane as qml

    padded_model, original_dimension = _pad_lindblad_model(model)
    padded_dimension = padded_model.hamiltonian.shape[0]
    qubits = int(np.log2(padded_dimension))
    wires = tuple(range(qubits))
    device = qml.device(backend, wires=qubits)
    grid = np.asarray(times, dtype=float)
    current = np.asarray(padded_model.initial_density, dtype=complex)
    densities = [current[:original_dimension, :original_dimension].copy()]
    channel_applications = 0

    for interval in np.diff(grid):
        step = float(interval) / substeps
        kraus = _first_order_cptp_kraus(padded_model, step)

        @qml.qnode(device)
        def channel_step(density: np.ndarray) -> np.ndarray:
            qml.QubitDensityMatrix(density, wires=wires)
            qml.QubitChannel(kraus, wires=wires)
            return qml.density_matrix(wires=wires)

        for _ in range(substeps):
            current = np.asarray(channel_step(current))
            channel_applications += 1
        densities.append(current[:original_dimension, :original_dimension].copy())

    density_array = np.asarray(densities)
    populations = np.real(np.diagonal(density_array, axis1=1, axis2=2))
    trace_error = np.abs(np.trace(density_array, axis1=1, axis2=2) - 1.0)
    minimum_eigenvalue = np.asarray(
        [np.min(np.linalg.eigvalsh(0.5 * (rho + rho.conj().T))) for rho in density_array]
    )
    return LindbladResult(
        grid,
        density_array,
        populations,
        trace_error,
        minimum_eigenvalue,
        {
            "pathway": "pennylane_lindblad_channel",
            "backend": backend,
            "qubits": qubits,
            "substeps_per_interval": substeps,
            "channel_applications": channel_applications,
            "channel_approximation": "normalized first-order Kraus",
        },
    )


def build_ndme_encoding(
    matrix: np.ndarray,
    initial_state: np.ndarray,
    *,
    shift_epsilon: float = 1e-9,
) -> NDMEEncoding:
    """Construct the PDF's NDME operators without choosing an evolution method."""
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
    jump_left = eigenvectors @ np.diag(np.sqrt(2.0 * shifted_eigenvalues)) @ eigenvectors.conj().T
    zero = np.zeros_like(operator)
    hamiltonian = block_diag(antihermitian_hamiltonian, zero)
    jump = block_diag(jump_left, zero)
    normalized_initial = initial / initial_norm
    projector = np.outer(normalized_initial, normalized_initial.conj())
    density_initial = 0.5 * np.block([[projector, projector], [projector, projector]])
    generator_error = np.linalg.norm(
        (-1j * antihermitian_hamiltonian - 0.5 * jump_left.conj().T @ jump_left)
        - (operator - shift * np.eye(dimension))
    )
    metadata = {
        "pathway": "lindblad_ndme",
        "source": "Shang et al., PRL 135, 120604 (2025), Eqs. (2)-(5)",
        "ode_dimension": dimension,
        "density_dimension": 2 * dimension,
        "liouville_dimension": (2 * dimension) ** 2,
        "semidissipative_min_eigenvalue_before_shift": float(eigenvalues[0]),
        "semidissipative_shift": shift,
        "jump_rank": int(np.count_nonzero(shifted_eigenvalues > shift_epsilon)),
        "generator_identity_error": float(generator_error),
    }
    return NDMEEncoding(
        operator, hamiltonian, jump, density_initial, normalized_initial,
        initial_norm, shift, metadata,
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
    encoding = build_ndme_encoding(matrix, initial_state, shift_epsilon=shift_epsilon)
    full_dimension = encoding.initial_density.shape[0]
    hamiltonian = encoding.hamiltonian
    jump = encoding.jump

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
        encoding.initial_density.reshape(-1),
        t_eval=np.asarray(times),
        method="DOP853",
        rtol=rtol,
        atol=atol,
    )
    if not solution.success:
        raise RuntimeError(solution.message)
    densities = solution.y.T.reshape((-1, full_dimension, full_dimension))
    encoded = encoding.extract_states(densities, times)
    trace_error = np.abs(np.trace(densities, axis1=1, axis2=2) - 1.0)
    minimum_eigenvalue = np.asarray(
        [np.min(np.linalg.eigvalsh(0.5 * (rho + rho.conj().T))) for rho in densities]
    )
    return NDMELindbladResult(
        np.asarray(times),
        densities,
        encoded,
        trace_error,
        minimum_eigenvalue,
        {
            **encoding.metadata,
            "evolution_backend": "classical DOP853 density integration",
        },
    )


def simulate_ndme_pennylane(
    matrix: np.ndarray,
    initial_state: np.ndarray,
    times: np.ndarray,
    *,
    substeps: int = 4,
    backend: str = "default.mixed",
) -> NDMELindbladResult:
    """Run NDME short-time channels on PennyLane and extract ODE coordinates."""
    encoding = build_ndme_encoding(matrix, initial_state)
    result = simulate_lindblad_pennylane(
        encoding.model(), np.asarray(times), substeps=substeps, backend=backend
    )
    return NDMELindbladResult(
        result.times,
        result.density_matrices,
        encoding.extract_states(result.density_matrices, result.times),
        result.trace_error,
        result.minimum_eigenvalue,
        {**encoding.metadata, **result.metadata},
    )
