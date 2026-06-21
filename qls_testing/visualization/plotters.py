"""Reusable Plotly figures for experiment results."""

from __future__ import annotations

from plotly.subplots import make_subplots
import plotly.graph_objects as go

from qls_testing.core.datatypes import ExperimentResult


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

