import numpy as np

from qls_testing.core.config import Config, MethodConfig, OutputConfig, SystemConfig, TimeConfig
from qls_testing.experiments.run_experiment import run_experiment, run_lindblad_experiment


def _toy(integrator: str, solver: str) -> Config:
    settings = {"degree": 11} if solver == "qsvt_simulator" else {}
    return Config(
        system=SystemConfig(name="toy_linear_ode"),
        linearization=MethodConfig("carleman", {"order": 1}),
        integrator=MethodConfig(integrator),
        qls=MethodConfig(solver, settings),
        time=TimeConfig(t_final=0.2, dt=0.1, n_points=3),
        output=OutputConfig(save_plot=False),
    )


def test_ode_ground_truth_is_identical_across_integrators_and_solvers():
    baseline = run_experiment(_toy("backward_euler", "classical"))
    alternatives = (
        run_experiment(_toy("crank_nicolson", "classical")),
        run_experiment(_toy("backward_euler", "hhl_simulator")),
        run_experiment(_toy("backward_euler", "qsvt_simulator")),
    )
    for result in alternatives:
        np.testing.assert_array_equal(result.reference_times, baseline.reference_times)
        np.testing.assert_array_equal(result.reference_states, baseline.reference_states)
        assert result.metrics["reference_scope"] == "Ground-truth ODE solution"


def test_lindbladian_ground_truth_is_identical_across_classical_and_pennylane_paths():
    common = dict(
        linearization=MethodConfig("carleman", {"order": 1}),
        qls=MethodConfig("classical"),
        time=TimeConfig(t_final=0.05, dt=0.05, n_points=2),
        output=OutputConfig(save_plot=False),
    )
    classical = run_lindblad_experiment(
        Config(
            system=SystemConfig(name="lindblad_enzyme_ndme"),
            integrator=MethodConfig("lindblad_ndme"),
            **common,
        )
    )
    pennylane = run_lindblad_experiment(
        Config(
            system=SystemConfig(name="lindblad_enzyme_ndme_pennylane"),
            integrator=MethodConfig("pennylane_lindblad", {"substeps": 1}),
            **common,
        )
    )
    np.testing.assert_array_equal(classical.reference_times, pennylane.reference_times)
    np.testing.assert_array_equal(classical.reference_states, pennylane.reference_states)
    assert classical.metrics["reference_scope"] == "Ground-truth Lindbladian solution"
    assert pennylane.metrics["reference_scope"] == "Ground-truth Lindbladian solution"
    labels = {term.label for term in pennylane.complexity_report.terms}
    assert "PennyLane short-time channel execution" in labels
    assert "Classical Liouvillian integration proxy" not in labels
