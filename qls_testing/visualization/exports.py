"""Serialization helpers shared by UI and non-interactive scripts."""

from __future__ import annotations

from io import BytesIO, StringIO

import numpy as np

from qls_testing.core.datatypes import ExperimentResult
from qls_testing.visualization.plotters import trajectory_figure


def result_csv(result: ExperimentResult) -> bytes:
    labels = tuple(f"lifted:{label}" for label in result.linearized_system.labels)
    header = ",".join(("time", *labels))
    stream = StringIO()
    np.savetxt(stream, np.column_stack((result.integration.times, result.integration.states)), delimiter=",", header=header, comments="")
    return stream.getvalue().encode("utf-8")


def result_npz(result: ExperimentResult) -> bytes:
    stream = BytesIO()
    errors = result.error_report.components if result.error_report else {}
    np.savez_compressed(
        stream,
        times=result.integration.times,
        lifted_states=result.integration.states,
        physical_states=result.physical_states,
        reference_states=result.reference_states,
        labels=np.asarray(result.linearized_system.labels),
        **{f"error_{name}": values for name, values in errors.items()},
    )
    return stream.getvalue()


def result_html(result: ExperimentResult) -> bytes:
    return trajectory_figure(result).to_html(include_plotlyjs="cdn").encode("utf-8")
