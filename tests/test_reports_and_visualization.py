import io

import numpy as np

from qls_testing.core.config import Config, MethodConfig, OutputConfig, SystemConfig, TimeConfig
from qls_testing.experiments.run_experiment import run_experiment
from qls_testing.visualization.exports import result_csv, result_html, result_npz
from qls_testing.visualization.plotters import error_figure, variable_figure
from qls_testing.models.observables import enzyme_observable_groups, observable_groups
from qls_testing.visualization.math_content import active_math_sections
from qls_testing.visualization.reasoning_content import active_reasoning
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


def test_observable_groups_and_selectable_error_plot_policy():
    labels = ("S", "X1", "X2", "X3", "P", "C1", "C2", "C3", "C4")
    groups = enzyme_observable_groups(labels)
    assert [group.key for group in groups] == ["S", "Xs", "P", "Cs"]
    assert groups[1].labels == ("X1", "X2", "X3")
    assert groups[3].labels == ("C1", "C2", "C3", "C4")
    result = toy_result()
    practice_group = observable_groups(("x", "y"))[0]
    absolute = observable_comparison_figure(result, practice_group, "absolute")
    relative = observable_comparison_figure(result, practice_group, "relative")
    both = observable_comparison_figure(result, practice_group, "both")
    assert len(absolute.data) == 6  # pipeline, reference, selected error per state
    assert len(relative.data) == 6
    assert len(both.data) == 8
    assert all("relative" not in trace.name for trace in absolute.data)
    assert all("absolute" not in trace.name for trace in relative.data)
    # A variable keeps one color for pipeline/reference/error, while variables differ.
    assert absolute.data[0].line.color == absolute.data[1].line.color == absolute.data[2].line.color
    assert absolute.data[0].line.color != absolute.data[3].line.color


def test_complexity_report_contains_only_selected_pipeline_stages_with_provenance():
    result = toy_result()
    report = result.complexity_report
    assert report.stages == ("Linearization", "Integrator", "Linear solver")
    assert all(term.symbolic and term.source and term.evaluated for term in report.terms)
    serialized = str(report.to_dict()).lower()
    assert "dense direct solve" in serialized
    assert "hhl phase estimation" not in serialized
    assert "qsvt inverse polynomial" not in serialized


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


def test_method_reasoning_is_selected_pipeline_specific():
    ode = Config(
        system=SystemConfig(name="toy_linear_ode"),
        integrator=MethodConfig("crank_nicolson"),
        qls=MethodConfig("pennylane_vqls"),
    )
    ode_text = " ".join(text for _, text in active_reasoning(ode))
    assert "not L-stable" in ode_text
    assert "nonzero updates" in ode_text

    lindblad = Config(
        system=SystemConfig(name="lindblad_practice_pennylane"),
        integrator=MethodConfig("pennylane_lindblad"),
        qls=MethodConfig("classical"),
    )
    lindblad_text = " ".join(text for _, text in active_reasoning(lindblad))
    assert "Kraus channels" in lindblad_text
    assert "exact Liouvillian ground truth" in lindblad_text
