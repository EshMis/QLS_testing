"""Linear-solver plugins, including stable classical QLS simulators."""

from qls_testing.experiments.registry import QLS_SOLVERS

from .classical import ClassicalSolver
from .hhl import SpectralHHLSimulator
from .qsvt import QSVTPolynomialSimulator
from .vqls import VariationalLinearSolver

QLS_SOLVERS.register("classical", ClassicalSolver)
QLS_SOLVERS.register("hhl_simulator", SpectralHHLSimulator)
QLS_SOLVERS.register("qsvt_simulator", QSVTPolynomialSimulator)
QLS_SOLVERS.register("vqls_simulator", VariationalLinearSolver)

__all__ = ["ClassicalSolver", "QSVTPolynomialSimulator", "SpectralHHLSimulator", "VariationalLinearSolver"]

