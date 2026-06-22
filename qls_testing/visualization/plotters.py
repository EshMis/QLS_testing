"""Reusable Plotly figures for experiment results."""

from __future__ import annotations

from plotly.subplots import make_subplots
import plotly.graph_objects as go
from plotly.colors import qualitative
import numpy as np

from qls_testing.core.datatypes import ExperimentResult
from qls_testing.models.lindblad import LindbladResult
from qls_testing.models.observables import ObservableGroup


def trajectory_figure(result: ExperimentResult) -> go.Figure:
    """Show physical trajectories and absolute errors without rerunning work."""
    names = result.linearized_system.labels[: result.linearized_system.physical_dimension]
    truth_label = str(result.metrics.get("reference_scope", "Ground truth"))
    figure = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.12, subplot_titles=("State trajectories", "Absolute error"))
    for index, name in enumerate(names):
        figure.add_trace(go.Scatter(x=result.integration.times, y=result.physical_states[:, index], name=f"{name} (pipeline)"), row=1, col=1)
        figure.add_trace(go.Scatter(x=result.reference_times, y=result.reference_states[:, index], name=f"{name} ({truth_label})", line={"dash": "dash"}), row=1, col=1)
        figure.add_trace(go.Scatter(x=result.integration.times, y=abs(result.physical_states[:, index] - result.reference_states[:, index]), name=f"|error {name}|", showlegend=False), row=2, col=1)
    figure.update_yaxes(type="log", row=2, col=1)
    figure.update_layout(template="plotly_white", height=760, title=f"{result.config.system.name}: Carleman + {result.config.integrator.name} + {result.config.qls.name}", hovermode="x unified")
    figure.update_xaxes(title_text="Time", row=2, col=1)
    return figure


def variable_figure(result: ExperimentResult, label: str) -> go.Figure:
    """Interactive plot for one physical or lifted coordinate."""
    if label not in result.linearized_system.labels:
        raise KeyError(label)
    index = result.linearized_system.labels.index(label)
    values = result.integration.states[:, index]
    figure = go.Figure()
    figure.add_trace(
        go.Scatter(
            x=result.integration.times,
            y=values.real,
            customdata=values.imag,
            mode="lines+markers",
            marker={"size": 5},
            name=f"Re({label})",
            hovertemplate="t=%{x:.6g}<br>real=%{y:.6g}<br>imag=%{customdata:.6g}<extra></extra>",
        )
    )
    if abs(values.imag).max() > 1e-12:
        figure.add_trace(go.Scatter(x=result.integration.times, y=values.imag, name=f"Im({label})"))
    figure.update_layout(
        template="plotly_white",
        title=f"Lifted coordinate: {label}",
        xaxis_title="Time",
        yaxis_title="Value",
        hovermode="closest",
    )
    return figure


def error_figure(result: ExperimentResult) -> go.Figure:
    """Plot every staged error component on a shared logarithmic axis."""
    figure = go.Figure()
    if result.error_report is not None:
        for name, values in result.error_report.components.items():
            figure.add_trace(go.Scatter(x=result.error_report.times, y=values + 1e-18, name=name))
    figure.update_layout(
        template="plotly_white",
        title="Error decomposition",
        xaxis_title="Time",
        yaxis_title="Error proxy",
        yaxis_type="log",
        hovermode="x unified",
    )
    return figure


def lindblad_figure(result: LindbladResult) -> go.Figure:
    """Plot populations and physicality diagnostics for a Lindblad run."""
    figure = make_subplots(rows=2, cols=1, shared_xaxes=True, subplot_titles=("Populations", "Physicality diagnostics"))
    for index in range(result.populations.shape[1]):
        figure.add_trace(go.Scatter(x=result.times, y=result.populations[:, index], name=f"population_{index}"), row=1, col=1)
    figure.add_trace(go.Scatter(x=result.times, y=result.trace_error + 1e-18, name="trace error"), row=2, col=1)
    figure.add_trace(go.Scatter(x=result.times, y=np.maximum(-result.minimum_eigenvalue, 0.0) + 1e-18, name="negativity violation"), row=2, col=1)
    figure.update_yaxes(type="log", row=2, col=1)
    figure.update_layout(template="plotly_white", height=700, title="Lindblad amplitude damping (separate from QLS)")
    return figure


def observable_comparison_figure(
    result: ExperimentResult,
    group: ObservableGroup,
    error_mode: str = "absolute",
) -> go.Figure:
    """Compare trajectories and show only the user-selected error definition.

    A variable keeps one hue across pipeline, reference, and error traces;
    line style distinguishes the estimate from the reference and error type.
    """
    if error_mode not in {"absolute", "relative", "both"}:
        raise ValueError("error_mode must be 'absolute', 'relative', or 'both'")
    error_title = {
        "absolute": "Absolute error",
        "relative": "Relative error",
        "both": "Absolute and relative error",
    }[error_mode]
    figure = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.12,
        subplot_titles=(f"{group.title}: pipeline vs ground truth", error_title),
    )
    if not group.labels:
        figure.add_annotation(
            text=f"{group.title} is not present in this model.",
            x=0.5,
            y=0.5,
            xref="paper",
            yref="paper",
            showarrow=False,
        )
        return figure
    physical_labels = result.linearized_system.labels[: result.linearized_system.physical_dimension]
    truth_label = str(result.metrics.get("reference_scope", "Ground truth"))
    for label in group.labels:
        index = physical_labels.index(label)
        color = qualitative.Safe[index % len(qualitative.Safe)]
        pipeline = np.asarray(result.physical_states[:, index]).real
        reference = np.asarray(result.reference_states[:, index]).real
        absolute = np.abs(pipeline - reference)
        relative = absolute / np.maximum(np.abs(reference), 1e-12)
        custom = np.column_stack(
            (
                reference,
                absolute,
                relative,
                np.full(len(pipeline), label, dtype=object),
            )
        )
        figure.add_trace(
            go.Scatter(
                x=result.integration.times,
                y=pipeline,
                customdata=custom,
                mode="lines+markers",
                marker={"size": 4},
                line={"color": color},
                name=f"{label} pipeline",
                hovertemplate=(
                    "variable=%{customdata[3]}<br>t=%{x:.6g}<br>pipeline=%{y:.6g}"
                    "<br>reference=%{customdata[0]:.6g}<br>abs error=%{customdata[1]:.3e}"
                    "<br>rel error=%{customdata[2]:.3e}<extra></extra>"
                ),
            ),
            row=1,
            col=1,
        )
        figure.add_trace(
            go.Scatter(
                x=result.reference_times,
                y=reference,
                mode="lines",
                line={"dash": "dash", "color": color},
                name=f"{label} {truth_label}",
                hovertemplate=f"{label} {truth_label}<br>t=%{{x:.6g}}<br>value=%{{y:.6g}}<extra></extra>",
            ),
            row=1,
            col=1,
        )
        if error_mode in {"absolute", "both"}:
            figure.add_trace(
                go.Scatter(
                    x=result.integration.times,
                    y=absolute + 1e-18,
                    customdata=custom,
                    line={"color": color},
                    name=f"{label} absolute",
                    hovertemplate=f"{label}<br>t=%{{x:.6g}}<br>absolute=%{{y:.3e}}<extra></extra>",
                ),
                row=2,
                col=1,
            )
        if error_mode in {"relative", "both"}:
            figure.add_trace(
                go.Scatter(
                    x=result.integration.times,
                    y=relative + 1e-18,
                    customdata=custom,
                    line={"dash": "dot", "color": color},
                    name=f"{label} relative",
                    hovertemplate=f"{label}<br>t=%{{x:.6g}}<br>relative=%{{y:.3e}}<extra></extra>",
                ),
                row=2,
                col=1,
            )
    figure.update_yaxes(type="log", title_text=error_title, autorange=True, row=2, col=1)
    figure.update_xaxes(title_text="Time", row=2, col=1)
    figure.update_layout(
        template="plotly_white",
        height=680,
        hovermode="closest",
        legend={"orientation": "h", "y": -0.18},
    )
    return figure
