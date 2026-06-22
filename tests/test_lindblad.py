import numpy as np

from qls_testing.core.config import Config, MethodConfig, OutputConfig, SystemConfig, TimeConfig
from qls_testing.experiments.run_experiment import run_lindblad_experiment
from qls_testing.models.lindblad import (
    amplitude_damping_model,
    build_ndme_encoding,
    simulate_lindblad,
    simulate_ndme_pennylane,
    simulate_ndme_linear_ode,
)
from qls_testing.models.practice import lindblad_practice_system
from qls_testing.reference import solve_ndme_ground_truth


def test_amplitude_damping_matches_analytic_population_and_preserves_density_matrix():
    result = simulate_lindblad(amplitude_damping_model(1.0), t_final=2.0, n_points=21)
    np.testing.assert_allclose(result.populations[:, 1], np.exp(-result.times), atol=2e-12)
    np.testing.assert_allclose(result.populations[:, 0], 1.0 - np.exp(-result.times), atol=2e-12)
    assert np.max(result.trace_error) < 1e-12
    assert np.min(result.minimum_eigenvalue) > -1e-12


def test_ndme_reproduces_nonhermitian_linear_ode_and_preserves_density_matrix():
    matrix = np.array([[-1.0, 0.4], [0.0, -2.0]])
    initial = np.array([1.0, 0.5])
    times = np.linspace(0.0, 1.0, 11)
    result = simulate_ndme_linear_ode(matrix, initial, times)
    truth = solve_ndme_ground_truth(build_ndme_encoding(matrix, initial), times)
    np.testing.assert_allclose(result.encoded_states, truth.states, atol=2e-8)
    assert np.max(result.trace_error) < 1e-10
    assert np.min(result.minimum_eigenvalue) > -1e-10
    assert result.metadata["generator_identity_error"] < 1e-10


def test_enzyme_ndme_pipeline_compares_all_nine_physical_variables():
    config = Config(
        system=SystemConfig(name="lindblad_enzyme_ndme"),
        linearization=MethodConfig("carleman", {"order": 1}),
        integrator=MethodConfig("lindblad_ndme"),
        qls=MethodConfig("classical"),
        time=TimeConfig(t_final=0.2, dt=0.05, n_points=5, rtol=1e-9, atol=1e-11),
        output=OutputConfig(save_plot=False),
    )
    result = run_lindblad_experiment(config)
    assert result.physical_states.shape == (5, 9)
    assert result.reference_states.shape == (5, 9)
    assert result.metrics["global_rmse"] < 1e-8
    assert result.metrics["reference_scope"] == "Ground-truth Lindbladian solution"
    report = result.complexity_report
    assert report.stages == ("Linearization", "Lindbladian / NDME")
    assert "d_\\rho=2D" in next(
        term.symbolic for term in report.terms if term.label == "Density-matrix embedding"
    )
    assert "Linear solver" not in report.stages


def test_pennylane_lindbladian_practice_channel_tracks_exact_ground_truth():
    practice = lindblad_practice_system()
    times = np.linspace(0.0, 0.2, 5)
    encoding = build_ndme_encoding(practice.matrix, practice.initial_state)
    truth = solve_ndme_ground_truth(encoding, times)
    result = simulate_ndme_pennylane(
        practice.matrix, practice.initial_state, times, substeps=8
    )
    np.testing.assert_allclose(result.encoded_states, truth.states, atol=3.5e-3)
    assert result.metadata["backend"] == "default.mixed"
    assert result.metadata["channel_applications"] == 32
    assert np.max(result.trace_error) < 1e-10
