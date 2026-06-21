"""Typed containers passed between pipeline plugins."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import NDArray

Exponent = tuple[int, ...]


@dataclass(frozen=True)
class PolynomialSystem:
    """Autonomous polynomial ODE ``x' = f(x)`` in sparse monomial form.

    ``terms[i][alpha]`` is the coefficient of ``x**alpha`` in ``f_i``.
    Constant terms are deliberately excluded because the lifted basis omits 1.
    """

    variable_names: tuple[str, ...]
    terms: tuple[dict[Exponent, float], ...]
    initial_state: NDArray[np.float64]
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        n = len(self.variable_names)
        if len(self.terms) != n or np.asarray(self.initial_state).shape != (n,):
            raise ValueError("terms and initial_state must match the number of variables")
        for equation in self.terms:
            for exponent in equation:
                if len(exponent) != n or any(value < 0 for value in exponent):
                    raise ValueError(f"invalid monomial exponent {exponent}")
                if sum(exponent) == 0:
                    raise ValueError("constant terms require an affine/homogeneous lift")

    def evaluate(self, state: NDArray[np.float64]) -> NDArray[np.float64]:
        """Evaluate the polynomial vector field."""
        x = np.asarray(state, dtype=float)
        return np.asarray(
            [sum(c * np.prod(x ** np.asarray(a)) for a, c in eq.items()) for eq in self.terms]
        )


@dataclass(frozen=True)
class LinearizedSystem:
    """Finite lifted approximation ``y' = matrix @ y``."""

    matrix: NDArray[np.float64]
    initial_state: NDArray[np.float64]
    physical_dimension: int
    exponents: tuple[Exponent, ...]
    labels: tuple[str, ...]
    metadata: dict[str, Any] = field(default_factory=dict)

    def project(self, lifted_states: NDArray[np.number]) -> NDArray[np.number]:
        """Return degree-one coordinates in original variable order."""
        return np.asarray(lifted_states)[..., : self.physical_dimension]


@dataclass(frozen=True)
class SolveResult:
    solution: NDArray[np.number]
    residual_norm: float
    relative_residual: float
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class IntegrationResult:
    times: NDArray[np.float64]
    states: NDArray[np.number]
    solve_diagnostics: tuple[SolveResult, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ExperimentResult:
    config: Any
    linearized_system: LinearizedSystem
    integration: IntegrationResult
    physical_states: NDArray[np.number]
    reference_times: NDArray[np.float64]
    reference_states: NDArray[np.float64]
    metrics: dict[str, Any]
