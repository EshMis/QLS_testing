"""Circuit-backed variational solver for tiny real or complex linear systems."""

from __future__ import annotations

import numpy as np
import pennylane as qml
from pennylane import numpy as qnp

from qls_testing.core.datatypes import SolveResult
from qls_testing.core.utils import residual_metrics
from qls_testing.quantum_pennylane.base import QuantumLinearSolver


class PennyLaneVQLSSolver(QuantumLinearSolver):
    """Optimize a hardware-efficient state circuit against ``||A x-b||``.

    The circuit prepares a normalized real ansatz ``u(theta)``. The optimal
    classical scale ``s=<Au,b>/<Au,Au>`` is eliminated analytically, and Adam
    minimizes the resulting normalized residual. This is a working simulator
    algorithm for tiny systems, not a claim of efficient full-vector readout.
    """

    def __init__(
        self,
        layers: int = 3,
        max_steps: int = 300,
        stepsize: float = 0.08,
        tolerance: float = 1e-8,
        seed: int = 42,
        backend: str = "default.qubit",
        shots: int | None = None,
        complex_ansatz: bool = False,
        loss_mode: str = "statevector",
    ) -> None:
        self.layers = layers
        self.max_steps = max_steps
        self.stepsize = stepsize
        self.tolerance = tolerance
        self.seed = seed
        self.backend = backend
        self.shots = shots
        self.complex_ansatz = complex_ansatz
        if loss_mode not in {"statevector", "shot_proxy"}:
            raise ValueError("loss_mode must be statevector or shot_proxy")
        if loss_mode == "shot_proxy" and not shots:
            raise ValueError("shot_proxy loss mode requires shots")
        self.loss_mode = loss_mode

    def solve(self, matrix: np.ndarray, rhs: np.ndarray) -> SolveResult:
        original = np.asarray(matrix)
        b_original = np.asarray(rhs)
        is_complex = np.iscomplexobj(original) or np.iscomplexobj(b_original)
        if is_complex and not self.complex_ansatz:
            raise ValueError("set complex_ansatz=True for complex matrices or vectors")
        dtype = complex if is_complex else float
        a = np.asarray(original, dtype=dtype)
        b = np.asarray(b_original, dtype=dtype)
        dimension = a.shape[0]
        if a.shape != (dimension, dimension) or b.shape != (dimension,):
            raise ValueError("matrix must be square and match rhs")
        qubits = max(1, int(np.ceil(np.log2(dimension))))
        padded_dimension = 2**qubits
        a_pad = np.eye(padded_dimension, dtype=dtype)
        a_pad[:dimension, :dimension] = a
        b_pad = np.zeros(padded_dimension, dtype=dtype)
        b_pad[:dimension] = b
        if np.linalg.norm(b_pad) == 0:
            return SolveResult(np.zeros_like(b), 0.0, 0.0, {"method": "pennylane_vqls"})

        device = qml.device(self.backend, wires=qubits, shots=None)

        @qml.qnode(device, interface="autograd")
        def state_circuit(weights: qnp.ndarray) -> qnp.ndarray:
            for layer in range(self.layers):
                for wire in range(qubits):
                    angle = weights[layer, wire, 0] if self.complex_ansatz else weights[layer, wire]
                    qml.RY(angle, wires=wire)
                    if self.complex_ansatz:
                        qml.RZ(weights[layer, wire, 1], wires=wire)
                for wire in range(qubits - 1):
                    qml.CNOT(wires=(wire, wire + 1))
                if qubits > 2:
                    qml.CNOT(wires=(qubits - 1, 0))
            return qml.state()

        a_quantum = qnp.asarray(a_pad)
        b_quantum = qnp.asarray(b_pad)
        b_norm_squared = qnp.real(qnp.dot(qnp.conj(b_quantum), b_quantum))

        def scaled_state(weights: qnp.ndarray) -> qnp.ndarray:
            state = state_circuit(weights)
            if not self.complex_ansatz:
                state = qnp.real(state)
            image = qnp.dot(a_quantum, state)
            scale = qnp.dot(qnp.conj(image), b_quantum) / (
                qnp.real(qnp.dot(qnp.conj(image), image)) + 1e-14
            )
            return scale * state

        def cost(weights: qnp.ndarray) -> qnp.ndarray:
            residual = qnp.dot(a_quantum, scaled_state(weights)) - b_quantum
            return qnp.real(qnp.dot(qnp.conj(residual), residual)) / b_norm_squared

        rng = np.random.default_rng(self.seed)
        shape = (self.layers, qubits, 2) if self.complex_ansatz else (self.layers, qubits)
        weights = qnp.asarray(rng.uniform(-np.pi, np.pi, size=shape), requires_grad=True)
        optimizer = qml.AdamOptimizer(self.stepsize)
        best_weights = weights.copy()
        best_cost = float(cost(weights))
        steps = 0
        for step in range(self.max_steps):
            weights = optimizer.step(cost, weights)
            current_value = float(cost(weights))
            steps = step + 1
            if current_value < best_cost:
                best_cost = current_value
                best_weights = weights.copy()
            if best_cost <= self.tolerance:
                break
        solution = np.real_if_close(np.asarray(scaled_state(best_weights))[:dimension])
        absolute, relative = residual_metrics(a, solution, b)
        sampling_std = 0.0
        if self.shots:
            probabilities = np.abs(np.asarray(state_circuit(best_weights))) ** 2
            sampling_std = float(np.max(np.sqrt(probabilities * (1.0 - probabilities) / self.shots)))
        specs = qml.specs(state_circuit)(best_weights)
        resources = specs.resources
        return SolveResult(
            solution,
            absolute,
            relative,
            {
                "method": "pennylane_vqls",
                "backend": self.backend,
                "layers": self.layers,
                "steps": steps,
                "cost": best_cost,
                "qubits": qubits,
                "circuit_depth": int(resources.depth) if resources is not None else 0,
                "shots": self.shots or 0,
                "sampling_std": sampling_std,
                "complex_ansatz": self.complex_ansatz,
                "loss_mode": self.loss_mode,
            },
        )
