import io

import numpy as np

from qls_testing.core.config import Config, MethodConfig, OutputConfig, SystemConfig, TimeConfig
from qls_testing.experiments.run_experiment import run_experiment
from qls_testing.visualization.exports import result_csv, result_html, result_npz
from qls_testing.visualization.plotters import error_figure, variable_figure


def toy_result():
    return run_experiment(
        Config(
            system=SystemConfig(name="toy_linear_ode"),
            linearization=MethodConfig("carleman", {"order": 1}),
            integrator=MethodConfig("backward_euler"),
            qls=MethodConfig("classical"),
            time=TimeConfig(t_final=0.2, dt=0.1, n_points=3),
            output=OutputConfig(save_plot=False),
        )
    )


def test_error_and_complexity_reports_are_time_resolved():
    result = toy_result()
    assert result.error_report is not None
    assert result.complexity_report is not None
    assert set(result.error_report.components) >= {
        "integration_discretization", "carleman_truncation", "qls_approximation", "total_observed"
    }
    assert all(len(values) == len(result.integration.times) for values in result.error_report.components.values())
    assert result.complexity_report.metrics["lifted_dimension"] == 2


def test_ui_serialization_and_figures_smoke():
    result = toy_result()
    csv_data = result_csv(result)
    assert csv_data.startswith(b"time,lifted:x,lifted:y")
    archive = np.load(io.BytesIO(result_npz(result)))
    assert archive["lifted_states"].shape == (3, 2)
    assert b"plotly" in result_html(result).lower()
    assert len(variable_figure(result, "x").data) >= 1
    assert len(error_figure(result).data) >= 4

