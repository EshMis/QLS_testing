import numpy as np
import pytest
from scipy.linalg import expm

from qls_testing.core.datatypes import LinearizedSystem
from qls_testing.integrators.linear import (
    BDF2Integrator,
    BackwardEulerIntegrator,
    CrankNicolsonIntegrator,
    FoldedBDF2Integrator,
    FoldedBackwardEulerIntegrator,
    FoldedCrankNicolsonIntegrator,
    KrylovExponentialIntegrator,
)
from qls_testing.qls.classical import ClassicalSolver


def linear_system(matrix: np.ndarray, initial: np.ndarray) -> LinearizedSystem:
    n = len(initial)
    exponents = tuple(tuple(int(i == j) for i in range(n)) for j in range(n))
    return LinearizedSystem(matrix, initial, n, exponents, tuple(f"x{i}" for i in range(n)))


@pytest.mark.parametrize(
    ("integrator", "dt", "tolerance"),
    [
        (BackwardEulerIntegrator(), 0.005, 4e-3),
        (CrankNicolsonIntegrator(), 0.02, 2e-4),
        (BDF2Integrator(), 0.01, 7e-4),
    ],
)
def test_linear_ode_matches_matrix_exponential(integrator, dt, tolerance):
    matrix = np.array([[-1.0, 0.2], [-0.1, -2.0]])
    initial = np.array([1.0, -0.25])
    result = integrator.integrate(linear_system(matrix, initial), ClassicalSolver(), t_final=1.0, dt=dt)
    expected = expm(matrix) @ initial
    error = np.linalg.norm(result.states[-1] - expected)
    assert error < tolerance, f"final-state error {error:.3e} exceeds {tolerance:.3e}"


def test_crank_nicolson_harmonic_oscillator():
    matrix = np.array([[0.0, 1.0], [-1.0, 0.0]])
    initial = np.array([1.0, 0.0])
    result = CrankNicolsonIntegrator().integrate(linear_system(matrix, initial), ClassicalSolver(), t_final=1.0, dt=0.005)
    expected = np.array([np.cos(1.0), -np.sin(1.0)])
    assert np.linalg.norm(result.states[-1] - expected) < 3e-6


def test_sparse_krylov_matches_analytic_exponential():
    matrix = np.diag([-1.0, -2.0, -3.0])
    initial = np.array([1.0, 2.0, -1.0])
    result = KrylovExponentialIntegrator().integrate(
        linear_system(matrix, initial), ClassicalSolver(), t_final=1.0, dt=0.1, n_points=11
    )
    np.testing.assert_allclose(result.states[-1], np.exp([-1.0, -2.0, -3.0]) * initial, atol=1e-12)
    assert result.metadata["nnz"] == 3


@pytest.mark.parametrize(
    ("sequential", "folded"),
    [
        (BackwardEulerIntegrator(), FoldedBackwardEulerIntegrator()),
        (CrankNicolsonIntegrator(), FoldedCrankNicolsonIntegrator()),
        (BDF2Integrator(), FoldedBDF2Integrator()),
    ],
)
def test_folded_history_system_matches_sequential_integrator(sequential, folded):
    system = linear_system(
        np.asarray([[-1.0, 0.25], [0.1, -0.7]]), np.asarray([1.0, -0.2])
    )
    expected = sequential.integrate(system, ClassicalSolver(), t_final=0.4, dt=0.1)
    actual = folded.integrate(system, ClassicalSolver(), t_final=0.4, dt=0.1)
    np.testing.assert_allclose(actual.states, expected.states, atol=2e-13)
    assert actual.metadata["qls_calls"] == 1
    assert actual.metadata["folded_dimension"] == 8
