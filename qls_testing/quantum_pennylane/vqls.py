"""Circuit-backed variational solver for tiny real or complex linear systems."""

from __future__ import annotations

from functools import partial

import numpy as np
import pennylane as qml
from pennylane import numpy as qnp

from qls_testing.core.datatypes import SolveResult
from qls_testing.core.utils import residual_metrics
from qls_testing.quantum_pennylane.base import QuantumLinearSolver


class PennyLaneVQLSSolver(QuantumLinearSolver):
    """Optimize a hardware-efficient state circuit against ``||A x-b||``.

    The circuit prepares a normalized real ansatz ``u(theta)``. Its effective
    depth prevents a simple padded-state parameter-count bottleneck, and
    sequential solves can reuse optimized parameters. The optimal
    classical scale ``s=<Au,b>/<Au,Au>`` is eliminated analytically, and Adam
    minimizes the resulting normalized residual. This is a working simulator
    algorithm for tiny systems, not a claim of efficient full-vector readout.
    Optimizer, gradient, state-change, and sampled-probability histories are
    returned in the solve metadata.
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
        warm_start: bool = True,
        diagnostic_points: int = 25,
    ) -> None:
        self.layers = layers
        self.max_steps = max_steps
        self.stepsize = stepsize
        self.tolerance = tolerance
        self.seed = seed
        self.backend = backend
        self.shots = shots
        self.complex_ansatz = complex_ansatz
        self.warm_start = warm_start
        self.diagnostic_points = max(2, diagnostic_points)
        self._warm_weights: np.ndarray | None = None
        if loss_mode not in {"statevector", "shot_proxy"}:
            raise ValueError("loss_mode must be statevector or shot_proxy")
        if loss_mode == "shot_proxy" and not shots:
            raise ValueError("shot_proxy loss mode requires shots")
        self.loss_mode = loss_mode

    @staticmethod
    def _entropy(probabilities: np.ndarray) -> float:
        nonzero = probabilities[probabilities > 0.0]
        return float(-np.sum(nonzero * np.log2(nonzero)))

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

        # Optimization remains analytic/differentiable. When shots are requested,
        # a second QNode below performs genuine sampled readout diagnostics.
        device = qml.device(self.backend, wires=qubits, shots=None)

        # A shallow hardware-efficient circuit can have fewer parameters than a
        # generic padded state has degrees of freedom. Increase only the effective
        # depth needed to avoid that deterministic expressivity bottleneck.
        parameters_per_layer = qubits * (2 if self.complex_ansatz else 1)
        state_degrees = 2 * padded_dimension - 2 if self.complex_ansatz else padded_dimension - 1
        effective_layers = max(self.layers, int(np.ceil(state_degrees / parameters_per_layer)))

        def apply_ansatz(weights: qnp.ndarray) -> None:
            for layer in range(effective_layers):
                for wire in range(qubits):
                    angle = weights[layer, wire, 0] if self.complex_ansatz else weights[layer, wire]
                    qml.RY(angle, wires=wire)
                    if self.complex_ansatz:
                        qml.RZ(weights[layer, wire, 1], wires=wire)
                for wire in range(qubits - 1):
                    qml.CNOT(wires=(wire, wire + 1))
                if qubits > 2:
                    qml.CNOT(wires=(qubits - 1, 0))

        @qml.qnode(device, interface="autograd")
        def state_circuit(weights: qnp.ndarray) -> qnp.ndarray:
            apply_ansatz(weights)
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
        shape = (effective_layers, qubits, 2) if self.complex_ansatz else (effective_layers, qubits)
        used_warm_start = self.warm_start and self._warm_weights is not None and self._warm_weights.shape == shape
        initial_values = (
            self._warm_weights.copy()
            if used_warm_start
            else rng.uniform(-np.pi, np.pi, size=shape)
        )
        weights = qnp.asarray(initial_values, requires_grad=True)
        initial_weights = np.asarray(weights).copy()
        initial_solution = np.asarray(scaled_state(weights))[:dimension]
        optimizer = qml.AdamOptimizer(self.stepsize)
        best_weights = weights.copy()
        best_cost = float(cost(weights))
        initial_cost = best_cost
        loss_history = [best_cost]
        gradient_norm_history: list[float] = []
        update_norm_history: list[float] = []
        residual_history = [float(np.sqrt(max(best_cost, 0.0)))]
        solution_history = [np.real_if_close(initial_solution).tolist()]
        record_every = max(1, self.max_steps // self.diagnostic_points)
        steps = 0
        for step in range(self.max_steps):
            gradient = qml.grad(cost)(weights)
            gradient_norm_history.append(float(np.linalg.norm(np.asarray(gradient))))
            previous = np.asarray(weights).copy()
            weights = optimizer.step(cost, weights)
            current_value = float(cost(weights))
            update_norm_history.append(float(np.linalg.norm(np.asarray(weights) - previous)))
            loss_history.append(current_value)
            residual_history.append(float(np.sqrt(max(current_value, 0.0))))
            steps = step + 1
            if current_value < best_cost:
                best_cost = current_value
                best_weights = weights.copy()
            if steps % record_every == 0 or steps == self.max_steps or best_cost <= self.tolerance:
                solution_history.append(
                    np.real_if_close(np.asarray(scaled_state(weights))[:dimension]).tolist()
                )
            if best_cost <= self.tolerance:
                break
        if self.warm_start:
            self._warm_weights = np.asarray(best_weights).copy()
        solution = np.real_if_close(np.asarray(scaled_state(best_weights))[:dimension])
        absolute, relative = residual_metrics(a, solution, b)
        exact_probabilities = np.abs(np.asarray(state_circuit(best_weights))) ** 2
        measured_probabilities = exact_probabilities
        sampling_std = 0.0
        if self.shots:
            sampled_device = qml.device(self.backend, wires=qubits)

            @partial(qml.set_shots, shots=self.shots)
            @qml.qnode(sampled_device)
            def probability_circuit(sample_weights: qnp.ndarray) -> qnp.ndarray:
                apply_ansatz(sample_weights)
                return qml.probs(wires=range(qubits))

            measured_probabilities = np.asarray(probability_circuit(best_weights), dtype=float)
            sampling_std = float(
                np.max(np.sqrt(exact_probabilities * (1.0 - exact_probabilities) / self.shots))
            )
        specs = qml.specs(state_circuit)(best_weights)
        resources = specs["resources"]
        return SolveResult(
            solution,
            absolute,
            relative,
            {
                "method": "pennylane_vqls",
                "backend": self.backend,
                "requested_layers": self.layers,
                "layers": effective_layers,
                "steps": steps,
                "cost": best_cost,
                "initial_cost": initial_cost,
                "loss_history": loss_history,
                "gradient_norm_history": gradient_norm_history,
                "parameter_update_norm_history": update_norm_history,
                "residual_history": residual_history,
                "solution_history": solution_history,
                "initial_solution": np.real_if_close(initial_solution).tolist(),
                "final_solution": np.real_if_close(solution).tolist(),
                "parameter_change_norm": float(np.linalg.norm(np.asarray(best_weights) - initial_weights)),
                "state_change_norm": float(np.linalg.norm(solution - initial_solution)),
                "initial_parameter_sample": initial_weights.ravel()[:6].tolist(),
                "final_parameter_sample": np.asarray(best_weights).ravel()[:6].tolist(),
                "used_warm_start": used_warm_start,
                "qubits": qubits,
                "circuit_depth": int(resources.depth) if resources is not None else 0,
                "shots": self.shots or 0,
                "sampling_std": sampling_std,
                "exact_probabilities": exact_probabilities.tolist(),
                "measured_probabilities": measured_probabilities.tolist(),
                "probability_entropy": self._entropy(measured_probabilities),
                "probability_variance": float(np.var(measured_probabilities)),
                "complex_ansatz": self.complex_ansatz,
                "loss_mode": self.loss_mode,
                "loss_evaluation": "analytic statevector (shot circuit used for readout diagnostics)",
            },
        )
