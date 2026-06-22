"""Exact small-instance LCU block encoding and scalable decomposition plans."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.linalg import block_diag
from scipy.sparse import kron

from .folded_systems import FoldedSystem


def _next_power_of_two(value: int) -> int:
    return 1 << int(np.ceil(np.log2(max(value, 1))))


def _positive_square_root(matrix: np.ndarray) -> np.ndarray:
    hermitian = 0.5 * (matrix + matrix.conj().T)
    values, vectors = np.linalg.eigh(hermitian)
    if np.min(values) < -2e-10:
        raise ValueError("unitary dilation received a non-contraction")
    return vectors @ np.diag(np.sqrt(np.clip(values, 0.0, None))) @ vectors.conj().T


def _unitary_dilation(contraction: np.ndarray) -> np.ndarray:
    dimension = contraction.shape[0]
    identity = np.eye(dimension, dtype=complex)
    top_right = _positive_square_root(identity - contraction @ contraction.conj().T)
    bottom_left = _positive_square_root(identity - contraction.conj().T @ contraction)
    return np.block(
        [[contraction, top_right], [bottom_left, -contraction.conj().T]]
    )


def _state_preparation_unitary(amplitudes: np.ndarray) -> np.ndarray:
    target = np.asarray(amplitudes, dtype=complex)
    target /= np.linalg.norm(target)
    first = np.zeros_like(target)
    first[0] = 1.0
    difference = first - target
    if np.linalg.norm(difference) < 1e-14:
        return np.eye(len(target), dtype=complex)
    return np.eye(len(target), dtype=complex) - 2.0 * np.outer(
        difference, difference.conj()
    ) / np.vdot(difference, difference)


@dataclass(frozen=True)
class LCUBlockEncoding:
    """Unitary whose all-zero ancilla block equals ``matrix / alpha``."""

    unitary: np.ndarray
    matrix: np.ndarray
    alpha: float
    original_dimension: int
    padded_dimension: int
    selector_dimension: int
    labels: tuple[str, ...]
    term_scales: tuple[float, ...]

    @property
    def ancilla_qubits(self) -> int:
        return 1 + int(np.log2(self.selector_dimension))

    @property
    def system_qubits(self) -> int:
        return int(np.log2(self.padded_dimension))

    def encoded_block(self) -> np.ndarray:
        return self.unitary[: self.padded_dimension, : self.padded_dimension]


def build_lcu_block_encoding(
    matrices: tuple[np.ndarray, ...],
    *,
    labels: tuple[str, ...] | None = None,
) -> LCUBlockEncoding:
    """Build an explicit PREP/SELECT LCU encoding for small validation cases.

    Each term receives an exact unitary dilation. The construction is exponential
    when materialized, so large hardware targets should compile the same terms
    as oracles rather than call this function.
    """
    if not matrices:
        raise ValueError("at least one LCU term is required")
    original_dimension = matrices[0].shape[0]
    if any(matrix.shape != (original_dimension, original_dimension) for matrix in matrices):
        raise ValueError("all LCU terms must be square with equal dimension")
    padded_dimension = _next_power_of_two(original_dimension)
    padded_terms = []
    retained_labels = []
    scales = []
    names = labels or tuple(f"term_{index}" for index in range(len(matrices)))
    for matrix, label in zip(matrices, names):
        padded = np.zeros((padded_dimension, padded_dimension), dtype=complex)
        padded[:original_dimension, :original_dimension] = matrix
        scale = float(np.linalg.norm(padded, ord=2))
        if scale <= 1e-15:
            continue
        padded_terms.append(padded)
        retained_labels.append(label)
        scales.append(scale)
    if not padded_terms:
        raise ValueError("the encoded matrix cannot be zero")
    alpha = float(sum(scales))
    selector_dimension = _next_power_of_two(len(padded_terms))
    selector_amplitudes = np.zeros(selector_dimension)
    selector_amplitudes[: len(scales)] = np.sqrt(np.asarray(scales) / alpha)
    prepare = _state_preparation_unitary(selector_amplitudes)
    dilation_dimension = 2 * padded_dimension
    selected = []
    for term, scale in zip(padded_terms, scales):
        selected.append(_unitary_dilation(term / scale))
    selected.extend(
        np.eye(dilation_dimension, dtype=complex)
        for _ in range(selector_dimension - len(selected))
    )
    select = block_diag(*selected)
    prepare_full = np.kron(prepare, np.eye(dilation_dimension))
    unitary = prepare_full.conj().T @ select @ prepare_full
    matrix = sum(padded_terms, np.zeros_like(padded_terms[0]))
    return LCUBlockEncoding(
        unitary,
        matrix,
        alpha,
        original_dimension,
        padded_dimension,
        selector_dimension,
        tuple(retained_labels),
        tuple(scales),
    )


def folded_lcu_matrices(folded: FoldedSystem) -> tuple[tuple[np.ndarray, ...], tuple[str, ...]]:
    """Materialize the exact Kronecker summands for small encoder tests."""
    matrices = []
    labels = []
    for coefficient, time_operator, state_operator, label in folded.kronecker_terms:
        matrices.append((coefficient * kron(time_operator, state_operator)).toarray())
        labels.append(label)
    return tuple(matrices), tuple(labels)


def projected_lcu_action(
    encoding: LCUBlockEncoding, state: np.ndarray
) -> tuple[np.ndarray, float]:
    """Apply the explicit unitary and return its all-zero-ancilla branch."""
    vector = np.zeros(encoding.unitary.shape[0], dtype=complex)
    normalized = np.asarray(state, dtype=complex) / np.linalg.norm(state)
    vector[: len(normalized)] = normalized
    output = encoding.unitary @ vector
    block = output[: encoding.padded_dimension]
    return block * encoding.alpha, float(np.vdot(block, block).real)


def pennylane_projected_lcu_action(
    encoding: LCUBlockEncoding,
    state: np.ndarray,
    *,
    backend: str = "default.qubit",
) -> tuple[np.ndarray, float]:
    """Execute the explicit validation encoder in a PennyLane QNode.

    This is an actual circuit path for small instances. It deliberately uses a
    dense ``QubitUnitary``; scalable targets must compile PREP/SELECT and sparse
    Carleman/time-shift oracles instead of materializing this unitary.
    """
    import pennylane as qml

    total_qubits = int(np.log2(encoding.unitary.shape[0]))
    system_qubits = encoding.system_qubits
    wires = tuple(range(total_qubits))
    system_wires = wires[-system_qubits:]
    padded_state = np.zeros(encoding.padded_dimension, dtype=complex)
    padded_state[: len(state)] = state
    padded_state /= np.linalg.norm(padded_state)
    device = qml.device(backend, wires=total_qubits)

    @qml.qnode(device)
    def circuit() -> np.ndarray:
        qml.StatePrep(padded_state, wires=system_wires)
        qml.QubitUnitary(encoding.unitary, wires=wires)
        return qml.state()

    output = np.asarray(circuit())
    block = output[: encoding.padded_dimension]
    return block * encoding.alpha, float(np.vdot(block, block).real)
