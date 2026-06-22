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
from qls_testing.hardware_path.block_encoding import (
    build_lcu_block_encoding,
    folded_lcu_matrices,
    pennylane_projected_lcu_action,
)
from qls_testing.hardware_path.folded_systems import build_folded_backward_euler
from qls_testing.hardware_path.readout import pennylane_overlap_quadratures


def test_block_encoding_postselection_applies_matrix():
    matrix = np.array([[2.0, 0.3], [0.1, 1.5]])
    state = np.array([1.0, -0.4])
    state /= np.linalg.norm(state)
    action, probability = projected_block_encoding_action(matrix, state)
    np.testing.assert_allclose(action, matrix @ state, atol=1e-11)
    assert 0.0 < probability <= 1.0


def test_pennylane_executes_structured_lcu_encoder_for_tiny_folded_system():
    folded = build_folded_backward_euler(
        np.asarray([[-1.0, 0.2], [0.0, -0.5]]), np.asarray([1.0, 0.0]), 0.1, 2
    )
    matrices, labels = folded_lcu_matrices(folded)
    encoding = build_lcu_block_encoding(matrices, labels=labels)
    state = np.asarray([1.0, -0.2, 0.3, 0.1])
    action, probability = pennylane_projected_lcu_action(encoding, state)
    expected = encoding.matrix @ (state / np.linalg.norm(state))
    np.testing.assert_allclose(action[: len(state)], expected, atol=2e-10)
    assert 0.0 < probability <= 1.0


def test_pennylane_interference_readout_recovers_signed_amplitude():
    state = np.asarray([0.6, -0.8j])
    reference = np.asarray([0.0, 1.0])
    real, imaginary = pennylane_overlap_quadratures(state, reference)
    assert abs(real) < 1e-12
    assert np.isclose(imaginary, 0.8, atol=1e-12)


def test_pennylane_vqls_solves_tiny_nonhermitian_system():
    matrix = np.array([[2.0, 0.3], [0.1, 1.5]])
    rhs = np.array([1.0, -0.4])
    result = PennyLaneVQLSSolver(
        layers=2, max_steps=240, stepsize=0.1, tolerance=1e-9, seed=2
    ).solve(matrix, rhs)
    assert result.relative_residual < 2e-4
    assert result.metadata["backend"] == "default.qubit"
    assert result.metadata["circuit_depth"] > 0
    assert result.metadata["parameter_change_norm"] > 1e-6
    assert result.metadata["state_change_norm"] > 1e-6
    assert min(result.metadata["loss_history"]) < result.metadata["initial_cost"]
    assert max(result.metadata["gradient_norm_history"]) > 1e-8
    assert max(result.metadata["parameter_update_norm_history"]) > 1e-8
    assert len(result.metadata["solution_history"]) >= 2


def test_pennylane_vqls_shots_execute_nontrivial_sampled_readout():
    matrix = np.array([[1.8, 0.2], [0.1, 1.2]])
    rhs = np.array([1.0, 0.4])
    shots = 400
    result = PennyLaneVQLSSolver(
        layers=2, max_steps=120, stepsize=0.08, seed=4, shots=shots
    ).solve(matrix, rhs)
    probabilities = np.asarray(result.metadata["measured_probabilities"])
    assert np.isclose(probabilities.sum(), 1.0)
    assert np.count_nonzero(probabilities) > 1
    np.testing.assert_allclose(probabilities * shots, np.round(probabilities * shots), atol=1e-10)
    assert result.metadata["probability_entropy"] > 0.0


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
    assert np.all(np.linalg.norm(np.diff(result.physical_states, axis=0), axis=1) > 1e-3)
    assert result.integration.solve_diagnostics[1].metadata["used_warm_start"] is True
