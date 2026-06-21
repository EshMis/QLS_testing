import io

import numpy as np

from qls_testing.core.config import Config, MethodConfig, OutputConfig, SystemConfig, TimeConfig
from qls_testing.experiments.run_experiment import run_experiment
from qls_testing.visualization.exports import result_csv, result_html, result_npz
from qls_testing.visualization.plotters import error_figure, variable_figure
from qls_testing.models.observables import enzyme_observable_groups, observable_groups
from qls_testing.visualization.math_content import active_math_sections
from qls_testing.visualization.plotters import observable_comparison_figure


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


def test_observable_groups_match_notebook_state_order_and_plot_absolute_relative_errors():
    labels = ("S", "X1", "X2", "X3", "P", "C1", "C2", "C3", "C4")
    groups = enzyme_observable_groups(labels)
    assert [group.key for group in groups] == ["S", "Xs", "P", "Cs"]
    assert groups[1].labels == ("X1", "X2", "X3")
    assert groups[3].labels == ("C1", "C2", "C3", "C4")
    result = toy_result()
    practice_group = observable_groups(("x", "y"))[0]
    figure = observable_comparison_figure(result, practice_group)
    assert len(figure.data) == 8  # pipeline, reference, absolute, relative per state


def test_math_sections_are_active_pipeline_specific():
    ndme = Config(
        system=SystemConfig(name="lindblad_enzyme_ndme"),
        linearization=MethodConfig("carleman", {"order": 1}),
        integrator=MethodConfig("lindblad_ndme"),
        qls=MethodConfig("classical"),
    )
    text = " ".join(latex for _, latex, _ in active_math_sections(ndme))
    assert "rho_{01}" in text
    assert "eta_T" in text
