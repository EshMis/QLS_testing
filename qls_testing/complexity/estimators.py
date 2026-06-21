"""Transparent operation-count proxies; not wall-clock predictions."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from qls_testing.core.datatypes import IntegrationResult, LinearizedSystem
from qls_testing.core.interfaces import ComplexityEstimator


@dataclass(frozen=True)
class ComplexityReport:
    metrics: dict[str, float | int | str]
    asymptotic: dict[str, str]
    caveat: str = "This is an estimate; actual runtime depends on backend, compilation, precision, and hardware."

    def to_dict(self) -> dict[str, object]:
        return {"metrics": self.metrics, "asymptotic": self.asymptotic, "caveat": self.caveat}


class DefaultComplexityEstimator(ComplexityEstimator):
    """Estimate dense, sparse, and QLS-oriented workload indicators."""

    def estimate(
        self,
        *,
        lifted: LinearizedSystem,
        integration: IntegrationResult,
        integrator_name: str,
        solver_name: str,
    ) -> ComplexityReport:
        dimension = int(lifted.matrix.shape[0])
        nnz = int(np.count_nonzero(lifted.matrix))
        solves = len(integration.solve_diagnostics)
        steps = max(len(integration.times) - 1, 0)
        conditions = [
            float(item.metadata["condition_number"])
            for item in integration.solve_diagnostics
            if "condition_number" in item.metadata
        ]
        degrees = [
            int(item.metadata["degree"])
            for item in integration.solve_diagnostics
            if "degree" in item.metadata
        ]
        clock_qubits = [
            int(item.metadata["n_clock"])
            for item in integration.solve_diagnostics
            if "n_clock" in item.metadata
        ]
        circuit_depths = [
            int(item.metadata["circuit_depth"])
            for item in integration.solve_diagnostics
            if "circuit_depth" in item.metadata
        ]
        success_probabilities = [
            float(item.metadata["success_probability"])
            for item in integration.solve_diagnostics
            if "success_probability" in item.metadata
        ]
        condition = max(conditions, default=float(integration.metadata.get("lhs_condition", 1.0)))
        polynomial_degree = max(degrees, default=0)
        qubits = max(1, int(np.ceil(np.log2(max(dimension, 2)))))
        if polynomial_degree:
            query_proxy = max(solves, 1) * polynomial_degree
        elif clock_qubits:
            query_proxy = max(solves, 1) * (2 ** max(clock_qubits) - 1)
        else:
            query_proxy = max(solves, 1)
        metrics: dict[str, float | int | str] = {
            "lifted_dimension": dimension,
            "nnz": nnz,
            "sparsity": float(1.0 - nnz / max(dimension * dimension, 1)),
            "recorded_steps": steps,
            "linear_solves": solves,
            "dense_solve_flop_proxy": int(solves * dimension**3),
            "sparse_matvec_proxy": int(max(steps, 1) * nnz),
            "condition_number_proxy": condition,
            "qsvt_degree": polynomial_degree,
            "quantum_query_proxy": int(query_proxy),
            "controlled_operation_proxy": int(
                query_proxy * qubits
            ),
            "phase_estimation_clock_qubits": max(clock_qubits, default=0),
            "circuit_depth_proxy": max(circuit_depths, default=0),
            "minimum_postselection_probability": min(success_probabilities, default=1.0),
            "integrator": integrator_name,
            "solver": solver_name,
        }
        asymptotic = {
            "lift_dimension": "D(n,N)=sum_{k=1}^N binomial(n+k-1,k), plus one affine coordinate when needed",
            "dense_time_stepping": "O(n_steps * D^3) without factorization reuse; O(D^3+n_steps*D^2) with reuse",
            "sparse_krylov": "approximately O(n_krylov * nnz(A)) per exponential action",
            "hhl_idealized": "polylog(D) * kappa * sparsity / epsilon under state-preparation and oracle assumptions",
            "qsvt_idealized": "O(kappa log(1/epsilon)) block-encoding queries",
            "iterative_refinement": "base solve cost multiplied by 1 + correction rounds",
        }
        return ComplexityReport(metrics, asymptotic)
