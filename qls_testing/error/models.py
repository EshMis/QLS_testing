"""Per-time error decomposition for the configured pipeline."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import expm_multiply

from qls_testing.core.datatypes import IntegrationResult, LinearizedSystem
from qls_testing.core.interfaces import ErrorModel


@dataclass(frozen=True)
class ErrorReport:
    times: np.ndarray
    components: dict[str, np.ndarray]
    descriptions: dict[str, str]

    @property
    def final(self) -> dict[str, float]:
        return {name: float(values[-1]) for name, values in self.components.items()}

    def to_dict(self) -> dict[str, object]:
        return {
            "times": self.times.tolist(),
            "components": {name: values.tolist() for name, values in self.components.items()},
            "descriptions": self.descriptions,
            "final": self.final,
        }


class PipelineErrorModel(ErrorModel):
    """Separate integration, lifting, solver, and QSSA/model discrepancies.

    The decomposition uses computable reference differences, so components are
    proxies rather than a rigorous additive bound. Their vector sum need not
    equal the norm of total error because norms discard direction.
    """

    def estimate(
        self,
        *,
        lifted: LinearizedSystem,
        integration: IntegrationResult,
        classical_integration: IntegrationResult,
        polynomial_reference: np.ndarray,
        target_reference: np.ndarray,
    ) -> ErrorReport:
        times = integration.times
        sparse_matrix = csr_matrix(lifted.matrix)
        exact_lifted_states = np.asarray(
            [expm_multiply(time * sparse_matrix, lifted.initial_state) for time in times]
        )
        exact_lifted = np.asarray(lifted.project(exact_lifted_states))
        actual = np.asarray(lifted.project(integration.states))
        classical_discrete = np.asarray(lifted.project(classical_integration.states))

        def norm(values: np.ndarray) -> np.ndarray:
            return np.linalg.norm(values, axis=1)

        residuals = [item.relative_residual for item in integration.solve_diagnostics]
        if not residuals:
            residual_by_time = np.zeros(len(times))
        elif len(residuals) == len(times) - 1:
            residual_by_time = np.asarray([0.0, *residuals])
        else:
            residual_by_time = np.linspace(0.0, max(residuals), len(times))
        refinement = np.zeros(len(times))
        for index, diagnostic in enumerate(integration.solve_diagnostics[: len(times) - 1], start=1):
            history = diagnostic.metadata.get("residual_history", [])
            if history:
                refinement[index] = float(history[-1])
        sampling = np.zeros(len(times))
        block_polynomial = np.zeros(len(times))
        for index, diagnostic in enumerate(integration.solve_diagnostics[: len(times) - 1], start=1):
            sampling[index] = float(diagnostic.metadata.get("sampling_std", 0.0))
            block_polynomial[index] = float(
                diagnostic.metadata.get(
                    "polynomial_error", diagnostic.metadata.get("max_inverse_error", 0.0)
                )
            )

        components = {
            "integration_discretization": norm(classical_discrete - exact_lifted),
            "carleman_truncation": norm(exact_lifted - polynomial_reference),
            "qls_approximation": norm(actual - classical_discrete),
            "qls_relative_residual": residual_by_time,
            "qssa_model": norm(polynomial_reference - target_reference),
            "iterative_refinement_residual": refinement,
            "quantum_sampling_std": sampling,
            "block_encoding_polynomial": block_polynomial,
            "total_observed": norm(actual - target_reference),
        }
        descriptions = {
            "integration_discretization": "Classical discrete integrator versus exact lifted exponential action.",
            "carleman_truncation": "Exact finite lift versus direct polynomial-ODE reference.",
            "qls_approximation": "Configured solver trajectory versus the same integrator with exact solves.",
            "qls_relative_residual": "Per-solve ||Mx-b||/||b|| proxy.",
            "qssa_model": "Polynomial QSSA approximation versus rational QSSA reference.",
            "iterative_refinement_residual": "Residual remaining after refinement rounds.",
            "quantum_sampling_std": "Reported shot/sampling standard deviation when available.",
            "block_encoding_polynomial": "QSVT reciprocal-polynomial approximation error on the encoded spectrum.",
            "total_observed": "Pipeline trajectory versus the physical target reference.",
        }
        return ErrorReport(times, components, descriptions)
