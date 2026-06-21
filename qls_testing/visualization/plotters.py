"""Reusable Plotly figures for experiment results."""

from __future__ import annotations

from plotly.subplots import make_subplots
import plotly.graph_objects as go
import numpy as np

from qls_testing.core.datatypes import ExperimentResult
from qls_testing.models.lindblad import LindbladResult


def trajectory_figure(result: ExperimentResult) -> go.Figure:
    """Show physical trajectories and absolute errors without rerunning work."""
    names = result.linearized_system.labels[: result.linearized_system.physical_dimension]
    figure = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.12, subplot_titles=("State trajectories", "Absolute error"))
    for index, name in enumerate(names):
        figure.add_trace(go.Scatter(x=result.integration.times, y=result.physical_states[:, index], name=f"{name} (pipeline)"), row=1, col=1)
        figure.add_trace(go.Scatter(x=result.reference_times, y=result.reference_states[:, index], name=f"{name} (reference)", line={"dash": "dash"}), row=1, col=1)
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
