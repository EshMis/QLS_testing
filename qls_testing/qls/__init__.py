"""Linear-solver plugins, including stable classical QLS simulators."""

from qls_testing.experiments.registry import QLS_SOLVERS

from .classical import ClassicalSolver
from .hhl import SpectralHHLSimulator
from .qsvt import QSVTPolynomialSimulator
from .refinement import IterativeRefinementSolver
from .preconditioned import DiagonalPreconditionedSolver
from .vqls import VariationalLinearSolver

QLS_SOLVERS.register("classical", ClassicalSolver)
QLS_SOLVERS.register("hhl_simulator", SpectralHHLSimulator)
QLS_SOLVERS.register("qsvt_simulator", QSVTPolynomialSimulator)
QLS_SOLVERS.register("vqls_simulator", VariationalLinearSolver)


def _refinement_factory(
    base_solver: str = "qsvt_simulator",
    base_settings: dict[str, object] | None = None,
    max_iterations: int = 3,
    tolerance: float = 1e-10,
) -> IterativeRefinementSolver:
    base = QLS_SOLVERS.create(base_solver, **(base_settings or {}))
    return IterativeRefinementSolver(base, max_iterations=max_iterations, tolerance=tolerance)


QLS_SOLVERS.register("iterative_refinement", _refinement_factory)


def _pennylane_vqls_factory(**settings: object) -> object:
    from qls_testing.quantum_pennylane import PennyLaneVQLSSolver

    return PennyLaneVQLSSolver(**settings)


QLS_SOLVERS.register("pennylane_vqls", _pennylane_vqls_factory)


def _preconditioned_qsvt_factory(
    qsvt_settings: dict[str, object] | None = None,
    diagonal_tolerance: float = 1e-12,
) -> DiagonalPreconditionedSolver:
    base = QSVTPolynomialSimulator(**(qsvt_settings or {}))
    return DiagonalPreconditionedSolver(base, diagonal_tolerance)


QLS_SOLVERS.register("preconditioned_qsvt", _preconditioned_qsvt_factory)

__all__ = [
    "ClassicalSolver", "DiagonalPreconditionedSolver", "IterativeRefinementSolver",
    "QSVTPolynomialSimulator", "SpectralHHLSimulator", "VariationalLinearSolver",
]
