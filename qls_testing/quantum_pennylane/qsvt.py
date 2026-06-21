"""Small-matrix inverse QSVT circuit using PennyLane's native QSVT template."""

from __future__ import annotations

import numpy as np
import pennylane as qml
from numpy.polynomial import Chebyshev, Polynomial

from qls_testing.core.datatypes import SolveResult
from qls_testing.core.utils import residual_metrics
from qls_testing.quantum_pennylane.base import QuantumLinearSolver


class PennyLaneQSVTSolver(QuantumLinearSolver):
    """Apply an odd reciprocal polynomial through a block-encoded QSVT circuit."""

    def __init__(
        self,
        degree: int = 11,
        samples: int = 500,
        target_bound: float = 0.45,
        backend: str = "default.qubit",
        shots: int | None = None,
    ) -> None:
        if degree < 3 or degree % 2 == 0:
            raise ValueError("QSVT inverse degree must be odd and >= 3")
        self.degree = degree
        self.samples = samples
        self.target_bound = target_bound
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

        gram_scale = max(
            float(np.linalg.norm(operator @ operator.conj().T, ord=np.inf)),
            float(np.linalg.norm(operator.conj().T @ operator, ord=np.inf)),
            1.0,
        )
        block_scale = float(np.sqrt(gram_scale))
        normalized = operator / block_scale
        eigenvalues = np.linalg.eigvalsh(normalized)
        nonzero = np.abs(eigenvalues[np.abs(eigenvalues) > 1e-12])
        if not nonzero.size or len(nonzero) != len(eigenvalues):
            raise np.linalg.LinAlgError("QSVT operator is singular")
        spectral_gap = float(np.min(nonzero))
        sample_points = np.concatenate(
            (
                np.linspace(-1.0, -spectral_gap, self.samples),
                np.linspace(spectral_gap, 1.0, self.samples),
            )
        )
        rhs_norm = float(np.linalg.norm(embedded_rhs))
        if rhs_norm == 0:
            return SolveResult(np.zeros(original_dimension), 0.0, 0.0, {"method": "pennylane_qsvt"})
        state_qubits = int(np.log2(operator.shape[0]))
        wires = list(range(state_qubits + 1))
        state_wires = wires[1:]
        grid = np.linspace(-1.0, 1.0, 4001)
        qsvt_operator = None
        coefficients = None
        inverse_scale = 0.0
        effective_degree = self.degree
        for candidate_degree in range(self.degree, 2, -2):
            candidate_scale = self.target_bound * spectral_gap
            targets = candidate_scale / sample_points
            chebyshev = Chebyshev.fit(
                sample_points, targets, candidate_degree, domain=[-1.0, 1.0]
            )
            candidate = chebyshev.convert(kind=Polynomial).coef
            candidate = np.pad(
                candidate, (0, max(0, candidate_degree + 1 - len(candidate)))
            )
            candidate[::2] = 0.0
            maximum_polynomial = float(np.max(np.abs(Polynomial(candidate)(grid))))
            if maximum_polynomial >= 0.98:
                reduction = 0.98 / maximum_polynomial
                candidate *= reduction
                candidate_scale *= reduction
            try:
                candidate_operator = qml.qsvt(
                    normalized,
                    candidate,
                    encoding_wires=wires,
                    block_encoding="embedding",
                )
            except ValueError:
                continue
            coefficients = candidate
            inverse_scale = candidate_scale
            effective_degree = candidate_degree
            qsvt_operator = candidate_operator
            break
        if qsvt_operator is None or coefficients is None:
            raise RuntimeError("PennyLane could not synthesize stable QSVT phases")
        device = qml.device(self.backend, wires=wires, shots=None)

        @qml.qnode(device)
        def circuit() -> np.ndarray:
            qml.StatePrep(embedded_rhs / rhs_norm, wires=state_wires)
            qml.apply(qsvt_operator)
            return qml.state()

        full_state = np.asarray(circuit())
        postselected = full_state[: operator.shape[0]]
        transformed = np.real(postselected) * rhs_norm / (inverse_scale * block_scale)
        padded_solution = transformed if hermitian else transformed[padded_dimension:]
        solution = np.real_if_close(padded_solution[:original_dimension])
        absolute, relative = residual_metrics(original, solution, original_rhs)
        success_probability = float(np.sum(np.abs(postselected) ** 2))
        sampling_std = 0.0
        if self.shots:
            sampling_std = float(
                np.sqrt(success_probability * (1.0 - success_probability) / self.shots)
            )
        inverse_values = Polynomial(coefficients)(eigenvalues)
        approximation_error = float(
            np.max(np.abs(inverse_values - inverse_scale / eigenvalues))
        )
        return SolveResult(
            solution,
            absolute,
            relative,
            {
                "method": "pennylane_qsvt",
                "backend": self.backend,
                "degree": effective_degree,
                "requested_degree": self.degree,
                "qubits": state_qubits + 1,
                "dilated": not hermitian,
                "condition_number": float(np.max(nonzero) / spectral_gap),
                "block_scale": block_scale,
                "inverse_scale": inverse_scale,
                "polynomial_error": approximation_error,
                "success_probability": success_probability,
                "shots": self.shots or 0,
                "sampling_std": sampling_std,
                "converged": bool(relative < 5e-2),
            },
        )
