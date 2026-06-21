"""Small-matrix PennyLane implementation of HHL with explicit QPE."""

from __future__ import annotations

import numpy as np
import pennylane as qml
from scipy.linalg import expm

from qls_testing.core.datatypes import SolveResult
from qls_testing.core.utils import residual_metrics
from qls_testing.quantum_pennylane.base import QuantumLinearSolver


class PennyLaneHHLSolver(QuantumLinearSolver):
    """Run HHL using PennyLane QPE, conditioned rotations, and postselection.

    Non-Hermitian systems are embedded as ``[[0,A],[A†,0]]``. This simulator
    exposes full postselected amplitudes for validation; hardware would require
    repeated state preparation and observable measurements.
    """

    def __init__(
        self,
        n_clock: int = 7,
        phase_scale: float | None = None,
        rotation_safety: float = 0.8,
        backend: str = "default.qubit",
        shots: int | None = None,
    ) -> None:
        if n_clock < 2:
            raise ValueError("n_clock must be >= 2")
        self.n_clock = n_clock
        self.phase_scale = phase_scale
        self.rotation_safety = rotation_safety
        self.backend = backend
        self.shots = shots

    def solve(self, matrix: np.ndarray, rhs: np.ndarray) -> SolveResult:
        original = np.asarray(matrix, dtype=complex)
        original_rhs = np.asarray(rhs, dtype=complex)
        original_dimension = original.shape[0]
        if original.shape != (original_dimension, original_dimension) or original_rhs.shape != (original_dimension,):
            raise ValueError("matrix must be square and match rhs")
        padded_dimension = 1 << int(np.ceil(np.log2(max(original_dimension, 2))))
        padded = np.eye(padded_dimension, dtype=complex)
        padded[:original_dimension, :original_dimension] = original
        padded_rhs = np.zeros(padded_dimension, dtype=complex)
        padded_rhs[:original_dimension] = original_rhs
        hermitian = np.allclose(padded, padded.conj().T)
        if hermitian:
            operator = padded
            embedded_rhs = padded_rhs
        else:
            zero = np.zeros_like(padded)
            operator = np.block([[zero, padded], [padded.conj().T, zero]])
            embedded_rhs = np.concatenate((padded_rhs, np.zeros(padded_dimension)))

        eigenvalues = np.linalg.eigvalsh(operator)
        nonzero = np.abs(eigenvalues[np.abs(eigenvalues) > 1e-12])
        if not nonzero.size or len(nonzero) != len(eigenvalues):
            raise np.linalg.LinAlgError("HHL operator is singular")
        minimum = float(np.min(nonzero))
        maximum = float(np.max(nonzero))
        scale = self.phase_scale or (2.2 * maximum)
        if scale <= 2.0 * maximum:
            raise ValueError("phase_scale must exceed twice the spectral radius")
        rotation_constant = self.rotation_safety * minimum
        rhs_norm = float(np.linalg.norm(embedded_rhs))
        if rhs_norm == 0:
            return SolveResult(np.zeros(original_dimension), 0.0, 0.0, {"method": "pennylane_hhl"})

        system_qubits = int(np.log2(operator.shape[0]))
        clock_wires = list(range(self.n_clock))
        system_wires = list(range(self.n_clock, self.n_clock + system_qubits))
        ancilla_wire = self.n_clock + system_qubits
        all_wires = clock_wires + system_wires + [ancilla_wire]
        evolution = expm(2j * np.pi * operator / scale)
        device = qml.device(self.backend, wires=all_wires, shots=None)

        @qml.qnode(device)
        def circuit() -> np.ndarray:
            qml.StatePrep(embedded_rhs / rhs_norm, wires=system_wires)
            qml.QuantumPhaseEstimation(
                evolution, target_wires=system_wires, estimation_wires=clock_wires
            )
            bins = 2**self.n_clock
            for index in range(1, bins):
                fraction = index / bins
                signed_phase = fraction if fraction <= 0.5 else fraction - 1.0
                eigenvalue_estimate = signed_phase * scale
                if abs(eigenvalue_estimate) <= 1e-12:
                    continue
                bits = [
                    (index >> (self.n_clock - 1 - bit)) & 1
                    for bit in range(self.n_clock)
                ]
                angle = 2.0 * np.arcsin(
                    np.clip(rotation_constant / eigenvalue_estimate, -1.0, 1.0)
                )
                qml.ctrl(qml.RY, control=clock_wires, control_values=bits)(
                    angle, wires=ancilla_wire
                )
            qml.adjoint(qml.QuantumPhaseEstimation)(
                evolution, target_wires=system_wires, estimation_wires=clock_wires
            )
            return qml.state()

        state = np.asarray(circuit()).reshape([2] * len(all_wires))
        selection = tuple([0] * self.n_clock) + (slice(None),) * system_qubits + (1,)
        amplitudes = state[selection].reshape(-1)
        full_solution = amplitudes * rhs_norm / rotation_constant
        padded_solution = full_solution if hermitian else full_solution[padded_dimension:]
        solution = np.real_if_close(padded_solution[:original_dimension])
        absolute, relative = residual_metrics(original, solution, original_rhs)
        success_probability = float(np.sum(np.abs(amplitudes) ** 2))
        sampling_std = 0.0
        if self.shots:
            sampling_std = float(
                np.sqrt(success_probability * (1.0 - success_probability) / self.shots)
            )
        return SolveResult(
            solution,
            absolute,
            relative,
            {
                "method": "pennylane_hhl",
                "backend": self.backend,
                "n_clock": self.n_clock,
                "system_qubits": system_qubits,
                "dilated": not hermitian,
                "condition_number": maximum / minimum,
                "phase_scale": scale,
                "success_probability": success_probability,
                "shots": self.shots or 0,
                "sampling_std": sampling_std,
                "converged": bool(relative < 5e-2),
            },
        )
