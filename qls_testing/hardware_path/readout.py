"""Observable-focused readout accounting for normalized QLS history states."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from math import ceil, log2, pi

import numpy as np


def _preparation_unitary(state: np.ndarray) -> np.ndarray:
    target = np.asarray(state, dtype=complex)
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
class ReadoutPlan:
    folded_dimension: int
    padded_dimension: int
    history_state_qubits: int
    time_qubits: int
    state_qubits: int
    full_history_coordinates: int
    physical_coordinates_per_time: int
    selected_time_count: int
    accepted_terminal_blocks: int
    targeted_real_amplitudes: int
    hadamard_test_settings: int
    sampling_shot_proxy: int
    amplitude_estimation_query_proxy: int
    final_time_postselection_probability: float | None
    amplitude_amplification_query_factor: float | None
    requires_norm_recovery: bool = True

    def to_dict(self) -> dict[str, int | float | bool | None]:
        return asdict(self)


def history_coordinate_index(
    time_step: int, coordinate: int, state_dimension: int
) -> int:
    """Map ``(y[time_step+1], coordinate)`` into the folded solution vector."""
    if time_step < 0 or not 0 <= coordinate < state_dimension:
        raise ValueError("history coordinate is out of range")
    return time_step * state_dimension + coordinate


def build_readout_plan(
    *,
    lifted_dimension: int,
    physical_dimension: int,
    steps: int,
    selected_time_count: int = 1,
    accepted_terminal_blocks: int = 1,
    epsilon: float = 1e-2,
    failure_probability: float = 0.05,
    history_states: np.ndarray | None = None,
) -> ReadoutPlan:
    """Estimate targeted amplitude readout rather than full-history tomography.

    For real concentrations, one interference quadrature per requested amplitude
    is sufficient. Complex states require two. The proxy deliberately separates
    simple repeated Hadamard tests from coherent amplitude estimation.
    """
    if not 1 <= physical_dimension <= lifted_dimension:
        raise ValueError("physical_dimension must be within the lifted state")
    if steps < 1 or not 1 <= selected_time_count <= steps:
        raise ValueError("invalid history/time selection")
    if not 1 <= accepted_terminal_blocks <= steps:
        raise ValueError("accepted_terminal_blocks must lie within the history")
    if not 0.0 < epsilon < 1.0 or not 0.0 < failure_probability < 1.0:
        raise ValueError("epsilon and failure_probability must lie in (0,1)")
    folded_dimension = steps * lifted_dimension
    padded_dimension = 1 << int(ceil(log2(folded_dimension)))
    targets = physical_dimension * selected_time_count
    # Union-bound Hoeffding proxy for signed-amplitude Hadamard tests.
    shots_per_target = ceil(
        np.log(2.0 * targets / failure_probability) / (2.0 * epsilon**2)
    )
    postselection = None
    amplification = None
    if history_states is not None:
        values = np.asarray(history_states)
        if values.shape != (steps, lifted_dimension):
            raise ValueError("history_states must have shape (steps, lifted_dimension)")
        total_norm_squared = float(np.sum(np.abs(values) ** 2))
        postselection = (
            float(
                np.sum(np.abs(values[-accepted_terminal_blocks:]) ** 2)
                / total_norm_squared
            )
            if total_norm_squared
            else 0.0
        )
        amplification = float(1.0 / np.sqrt(postselection)) if postselection else float("inf")
    return ReadoutPlan(
        folded_dimension,
        padded_dimension,
        int(log2(padded_dimension)),
        int(ceil(log2(steps))),
        int(ceil(log2(lifted_dimension))),
        folded_dimension,
        physical_dimension,
        selected_time_count,
        accepted_terminal_blocks,
        targets,
        targets,
        targets * shots_per_target,
        targets * ceil(pi / epsilon),
        postselection,
        amplification,
    )


def pennylane_overlap_quadratures(
    state: np.ndarray,
    reference: np.ndarray,
    *,
    shots: int | None = None,
    backend: str = "default.qubit",
) -> tuple[float, float]:
    """Estimate ``<state|reference>`` through a PennyLane interference test.

    A basis state reference extracts one signed/complex solution amplitude.
    This small explicit-state implementation validates the readout circuit; a
    hardware path must replace dense preparation unitaries with the QSVT state
    preparation and a reversible observable/reference preparation.
    """
    import pennylane as qml

    state_vector = np.asarray(state, dtype=complex)
    reference_vector = np.asarray(reference, dtype=complex)
    if state_vector.shape != reference_vector.shape:
        raise ValueError("state and reference must have equal shape")
    dimension = 1 << int(ceil(log2(max(len(state_vector), 2))))
    padded_state = np.zeros(dimension, dtype=complex)
    padded_reference = np.zeros(dimension, dtype=complex)
    padded_state[: len(state_vector)] = state_vector
    padded_reference[: len(reference_vector)] = reference_vector
    state_unitary = _preparation_unitary(padded_state)
    reference_unitary = _preparation_unitary(padded_reference)
    system_qubits = int(log2(dimension))
    ancilla = 0
    system_wires = tuple(range(1, system_qubits + 1))
    device = qml.device(backend, wires=system_qubits + 1)

    @qml.set_shots(shots=shots)
    @qml.qnode(device)
    def circuit() -> tuple[float, float]:
        qml.Hadamard(wires=ancilla)
        qml.ctrl(qml.QubitUnitary, control=ancilla, control_values=[0])(
            state_unitary, wires=system_wires
        )
        qml.ctrl(qml.QubitUnitary, control=ancilla, control_values=[1])(
            reference_unitary, wires=system_wires
        )
        return qml.expval(qml.PauliX(ancilla)), qml.expval(qml.PauliY(ancilla))

    real, imaginary = circuit()
    return float(real), float(imaginary)
