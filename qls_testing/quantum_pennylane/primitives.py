"""Small, directly testable PennyLane block-encoding primitives."""

from __future__ import annotations

import numpy as np
import pennylane as qml
from scipy.sparse import csr_matrix


def sparse_block_encoding_data(matrix: np.ndarray) -> dict[str, np.ndarray | float | int]:
    """Expose CSR oracle data and normalization for a future sparse circuit.

    The current circuit demo consumes a dense tiny matrix; these arrays define
    the row/column/value oracle boundary needed by scalable sparse encodings.
    """
    sparse = csr_matrix(matrix)
    gram_scale = max(
        float(np.linalg.norm(sparse.toarray() @ sparse.toarray().conj().T, ord=np.inf)),
        float(np.linalg.norm(sparse.toarray().conj().T @ sparse.toarray(), ord=np.inf)),
        1.0,
    )
    return {
        "indptr": sparse.indptr.copy(),
        "indices": sparse.indices.copy(),
        "values": sparse.data.copy(),
        "normalization": float(np.sqrt(gram_scale)),
        "dimension": sparse.shape[0],
        "max_row_sparsity": int(np.max(np.diff(sparse.indptr), initial=0)),
    }


def block_encoding_matrix(matrix: np.ndarray) -> tuple[np.ndarray, float]:
    """Return PennyLane's block-encoding unitary and its normalization scale."""
    operator = np.asarray(matrix, dtype=complex)
    if operator.ndim != 2 or operator.shape[0] != operator.shape[1]:
        raise ValueError("block encoding demo expects a square matrix")
    dimension = operator.shape[0]
    if dimension & (dimension - 1):
        raise ValueError("block encoding demo expects a power-of-two dimension")
    # Match PennyLane's BlockEncode admissibility condition so it does not add
    # a second hidden normalization internally.
    gram_scale = max(
        float(np.linalg.norm(operator @ operator.conj().T, ord=np.inf)),
        float(np.linalg.norm(operator.conj().T @ operator, ord=np.inf)),
        1.0,
    )
    scale = float(np.sqrt(gram_scale))
    normalized = operator / scale
    state_qubits = int(np.log2(dimension))
    wires = list(range(state_qubits + 1))
    unitary = np.asarray(qml.matrix(qml.BlockEncode(normalized, wires=wires)))
    return unitary, scale


def projected_block_encoding_action(matrix: np.ndarray, state: np.ndarray) -> tuple[np.ndarray, float]:
    """Run a block-encoding circuit and postselect its all-zero ancilla block."""
    operator = np.asarray(matrix, dtype=complex)
    vector = np.asarray(state, dtype=complex)
    dimension = operator.shape[0]
    if vector.shape != (dimension,):
        raise ValueError("state dimension must match matrix")
    norm = np.linalg.norm(vector)
    if norm == 0:
        return np.zeros_like(vector), 1.0
    unitary, scale = block_encoding_matrix(operator)
    state_qubits = int(np.log2(dimension))
    wires = list(range(state_qubits + 1))
    system_wires = wires[1:]
    device = qml.device("default.qubit", wires=wires)

    @qml.qnode(device)
    def circuit() -> np.ndarray:
        qml.StatePrep(vector / norm, wires=system_wires)
        qml.QubitUnitary(unitary, wires=wires)
        return qml.state()

    output = np.asarray(circuit())[:dimension]
    return output * norm * scale, float(np.sum(np.abs(output) ** 2))
