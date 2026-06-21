import numpy as np
import pytest

pytest.importorskip("pennylane")

from qls_testing.core.config import Config, MethodConfig, OutputConfig, SystemConfig, TimeConfig
from qls_testing.experiments.run_experiment import run_experiment
from qls_testing.quantum_pennylane import (
    PennyLaneHHLSolver,
    PennyLaneQSVTSolver,
    PennyLaneVQLSSolver,
    projected_block_encoding_action,
)


def test_block_encoding_postselection_applies_matrix():
    matrix = np.array([[2.0, 0.3], [0.1, 1.5]])
    state = np.array([1.0, -0.4])
    state /= np.linalg.norm(state)
    action, probability = projected_block_encoding_action(matrix, state)
    np.testing.assert_allclose(action, matrix @ state, atol=1e-11)
    assert 0.0 < probability <= 1.0


def test_pennylane_vqls_solves_tiny_nonhermitian_system():
    matrix = np.array([[2.0, 0.3], [0.1, 1.5]])
    rhs = np.array([1.0, -0.4])
    result = PennyLaneVQLSSolver(
        layers=2, max_steps=240, stepsize=0.1, tolerance=1e-9, seed=2
    ).solve(matrix, rhs)
    assert result.relative_residual < 2e-4
    assert result.metadata["backend"] == "default.qubit"
    assert result.metadata["circuit_depth"] > 0


@pytest.mark.parametrize(
    "solver",
    (PennyLaneHHLSolver(n_clock=7), PennyLaneQSVTSolver(degree=11)),
)
def test_pennylane_hhl_and_qsvt_solve_small_system(solver):
    matrix = np.array([[1.5, 0.5], [0.5, 1.5]])
    rhs = np.array([1.0, 0.25])
    result = solver.solve(matrix, rhs)
    assert result.relative_residual < 1e-2
    assert result.metadata["converged"] is True
    assert 0.0 < result.metadata["success_probability"] <= 1.0


def test_pennylane_hhl_and_qsvt_handle_nonhermitian_dilation():
    matrix = np.array([[2.0, 0.3], [0.0, 1.0]])
    rhs = np.array([1.0, -0.2])
    for solver in (PennyLaneHHLSolver(n_clock=7), PennyLaneQSVTSolver(degree=9)):
        result = solver.solve(matrix, rhs)
        assert result.relative_residual < 2e-2
        assert result.metadata["dilated"] is True


def test_complex_vqls_ansatz_solves_complex_system():
    matrix = np.array([[1.5 + 0.1j, 0.2], [0.1, 1.0 - 0.1j]])
    rhs = np.array([1.0, 0.25j])
    result = PennyLaneVQLSSolver(
        complex_ansatz=True,
        layers=3,
        max_steps=300,
        stepsize=0.04,
        tolerance=1e-8,
        seed=3,
    ).solve(matrix, rhs)
    assert result.relative_residual < 5e-4
    assert result.metadata["complex_ansatz"] is True


def test_toy_ode_pipeline_runs_with_pennylane_solver():
    config = Config(
        system=SystemConfig(name="toy_linear_ode"),
        linearization=MethodConfig("carleman", {"order": 1}),
        integrator=MethodConfig("backward_euler"),
        qls=MethodConfig(
            "pennylane_vqls",
            {"layers": 2, "max_steps": 350, "stepsize": 0.05, "tolerance": 1e-8},
        ),
        time=TimeConfig(t_final=0.2, dt=0.1, n_points=3),
        output=OutputConfig(save_plot=False),
        random_seed=7,
    )
    result = run_experiment(config)
    assert result.physical_states.shape == (3, 2)
    assert result.metrics["max_relative_linear_solve_residual"] < 1e-3
