from qls_testing.core.config import Config, MethodConfig, OutputConfig, SystemConfig, TimeConfig
from qls_testing.experiments.run_experiment import run_experiment


def test_tiny_full_pipeline_with_hhl_simulator():
    config = Config(
        system=SystemConfig(initial_substrate=0.1),
        linearization=MethodConfig("carleman", {"order": 1}),
        integrator=MethodConfig("backward_euler"),
        qls=MethodConfig("hhl_simulator"),
        time=TimeConfig(t_final=0.2, dt=0.05, n_points=5),
        output=OutputConfig(save_plot=False),
    )
    result = run_experiment(config)
    assert result.physical_states.shape == (5, 9)
    assert result.metrics["max_relative_linear_solve_residual"] < 1e-11
    assert result.metrics["global_rmse"] < 2e-3
