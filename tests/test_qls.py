import numpy as np
import pytest

from qls_testing.qls.classical import ClassicalSolver
from qls_testing.qls.hhl import SpectralHHLSimulator
from qls_testing.qls.qsvt import QSVTPolynomialSimulator
from qls_testing.qls.vqls import VariationalLinearSolver


@pytest.mark.parametrize(
    ("solver", "residual_tolerance"),
    [
        (ClassicalSolver(), 1e-12),
        (SpectralHHLSimulator(), 1e-11),
        (QSVTPolynomialSimulator(degree=35), 1e-8),
        (VariationalLinearSolver(max_iterations=2000, tolerance=1e-11, seed=7), 2e-5),
    ],
)
def test_solvers_on_well_conditioned_system(solver, residual_tolerance):
    matrix = np.array([[3.0, 0.4], [0.2, 2.0]])
    rhs = np.array([1.0, -0.5])
    result = solver.solve(matrix, rhs)
    assert result.relative_residual < residual_tolerance, (
        f"{result.metadata['method']} residual {result.relative_residual:.3e} "
        f"exceeds {residual_tolerance:.3e}"
    )
    np.testing.assert_allclose(result.solution, np.linalg.solve(matrix, rhs), rtol=2e-5, atol=2e-6)


def test_hhl_dilation_handles_nonhermitian_matrix():
    matrix = np.array([[2.0, 1.0], [0.0, 3.0]])
    rhs = np.array([1.0, 2.0])
    result = SpectralHHLSimulator().solve(matrix, rhs)
    assert result.metadata["dilated"] is True
    assert result.relative_residual < 1e-12

