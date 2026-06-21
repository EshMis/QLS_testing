"""Abstract interfaces implemented by all registered methods."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np
from numpy.typing import NDArray

from .datatypes import IntegrationResult, LinearizedSystem, PolynomialSystem, SolveResult


class LinearizationMethod(ABC):
    @abstractmethod
    def linearize(self, system: PolynomialSystem, **settings: Any) -> LinearizedSystem:
        """Create a finite linear representation of a nonlinear system."""


class LinearSolver(ABC):
    @abstractmethod
    def solve(self, matrix: NDArray[np.number], rhs: NDArray[np.number]) -> SolveResult:
        """Approximately solve ``matrix @ x = rhs``."""


class Integrator(ABC):
    @abstractmethod
    def integrate(
        self,
        system: LinearizedSystem,
        solver: LinearSolver,
        *,
        t_final: float,
        dt: float,
        n_points: int | None = None,
    ) -> IntegrationResult:
        """Evolve a lifted linear system."""

