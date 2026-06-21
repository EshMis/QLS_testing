import numpy as np
import pytest

pytest.importorskip("pennylane")

from qls_testing.core.config import Config, MethodConfig, OutputConfig, SystemConfig, TimeConfig
from qls_testing.experiments.run_experiment import run_experiment
from qls_testing.quantum_pennylane import PennyLaneVQLSSolver, projected_block_encoding_action


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
