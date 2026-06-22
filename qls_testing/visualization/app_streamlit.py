"""Streamlit dashboard for physical observables, errors, math, and quantum practice."""

from __future__ import annotations

import numpy as np
import streamlit as st

from qls_testing.core.config import Config, MethodConfig, OutputConfig, SystemConfig, TimeConfig
from qls_testing.experiments.run_experiment import run_experiment, run_lindblad_experiment
from qls_testing.models import PRACTICE_SYSTEMS, observable_groups
from qls_testing.visualization.exports import result_csv, result_html, result_npz
from qls_testing.visualization.math_content import active_math_sections
from qls_testing.visualization.reasoning_content import active_reasoning
from qls_testing.visualization.plotters import (
    error_figure,
    observable_comparison_figure,
    variable_figure,
)


st.set_page_config(page_title="Enzyme Carleman + quantum laboratory", layout="wide")
st.title("Enzyme Carleman + quantum laboratory")
st.caption("Physical-variable comparisons first; lifted coordinates live under Advanced.")

systems = (
    "mass_action_pathway",
    "qssa_taylor_pathway",
    "lindblad_practice_pennylane",
    "lindblad_enzyme_ndme",
    "lindblad_enzyme_ndme_pennylane",
    "toy_linear_ode",
    *PRACTICE_SYSTEMS.keys(),
)
system_name = st.sidebar.selectbox("Pipeline / practice system", systems)
is_ndme = system_name.startswith("lindblad_")
is_pennylane_ndme = system_name.endswith("_pennylane")
order = st.sidebar.slider("Carleman order", 1, 4, 1 if is_ndme else 2, disabled=is_ndme)
integrator_options = (
    (("pennylane_lindblad",) if is_pennylane_ndme else ("lindblad_ndme",))
    if is_ndme
    else (
        "bdf2", "backward_euler", "crank_nicolson", "krylov_exponential",
        "rk45", "folded_backward_euler", "folded_crank_nicolson", "folded_bdf2",
        "pade22",
    )
)
integrator = st.sidebar.selectbox(
    "Integrator",
    integrator_options,
    disabled=is_ndme,
)
solver = st.sidebar.selectbox(
    "Linear solver",
    (
        "classical", "hhl_simulator", "qsvt_simulator", "iterative_refinement",
        "pennylane_hhl", "pennylane_qsvt", "pennylane_vqls",
        "pennylane_complex_vqls",
    ),
    disabled=is_ndme,
)
t_final = st.sidebar.slider("Time horizon", 0.2, 5.0, 1.0 if is_ndme else 5.0, 0.1)
dt = st.sidebar.select_slider("Fixed/output step", options=(0.5, 0.25, 0.1, 0.05, 0.02), value=0.1)
shots = st.sidebar.number_input("Quantum shots (0 = exact statevector)", min_value=0, max_value=100000, value=0, step=100)
initial = st.sidebar.slider("Initial substrate", 0.1, 5.0, 3.0, 0.1)
taylor_degree = st.sidebar.slider("QSSA Taylor degree", 1, 6, 3)
qssa_expansion = st.sidebar.selectbox("QSSA expansion", ("moving_point", "auto", "zero"))

solver_settings: dict[str, object] = {}
if solver in {"pennylane_vqls", "pennylane_complex_vqls"}:
    solver_settings = {"layers": 2, "max_steps": 300, "stepsize": 0.05, "tolerance": 1e-7, "shots": shots or None}
elif solver == "pennylane_hhl":
    solver_settings = {"n_clock": 7, "shots": shots or None}
elif solver == "pennylane_qsvt":
    solver_settings = {"degree": 11, "shots": shots or None}
elif solver == "iterative_refinement":
    solver_settings = {"base_solver": "qsvt_simulator", "max_iterations": 3}

effective_order = 1 if is_ndme else order
config = Config(
    system=SystemConfig(
        name=system_name,
        initial_substrate=initial,
        taylor_degree=taylor_degree,
        qssa_expansion=qssa_expansion,
    ),
    linearization=MethodConfig("carleman", {"order": effective_order}),
    integrator=MethodConfig(
        integrator,
        {"substeps": 8 if system_name == "lindblad_practice_pennylane" else 4}
        if is_pennylane_ndme else {},
    ),
    qls=MethodConfig("classical" if is_ndme else solver, {} if is_ndme else solver_settings),
    time=TimeConfig(t_final=t_final, dt=dt, n_points=max(3, int(round(t_final / dt)) + 1)),
    output=OutputConfig(save_plot=False),
)

if is_ndme:
    pathway = "PennyLane short-time channels" if is_pennylane_ndme else "classical density evolution"
    st.sidebar.info(
        f"NDME uses {pathway}, not a QLS. Order 1 keeps the density matrix interactive."
    )
fixed_step_methods = {
    "bdf2", "backward_euler", "crank_nicolson", "folded_backward_euler",
    "folded_crank_nicolson", "folded_bdf2",
}
grid_valid = is_ndme or integrator not in fixed_step_methods or np.isclose(
    t_final / dt, round(t_final / dt)
)
if not grid_valid:
    st.sidebar.warning("For this fixed-step integrator, choose a horizon divisible by the step size.")
if st.button("Run selected pipeline", type="primary", disabled=not grid_valid):
    with st.spinner("Running simulation…"):
        st.session_state["result"] = (
            run_lindblad_experiment(config) if is_ndme else run_experiment(config)
        )
        st.session_state["result_config"] = config

result = st.session_state.get("result")
result_config = st.session_state.get("result_config")
if result is not None and result_config is not None:
    results_tab, error_tab, math_tab, reasoning_tab, advanced_tab = st.tabs(
        ("Results", "Error & Complexity", "Math", "Method reasoning", "Advanced / Debug")
    )
    physical_labels = result.linearized_system.labels[: result.linearized_system.physical_dimension]
    groups = observable_groups(physical_labels)

    with results_tab:
        st.subheader("Pipeline versus ground truth")
        st.caption(result.metrics.get("reference_scope", "Ground-truth ODE solution"))
        error_mode = st.segmented_control(
            "Error traces",
            options=("absolute", "relative", "both"),
            default="absolute",
            help="The lower plot and summary rescale to the selected error definition.",
        ) or "absolute"
        group_tabs = st.tabs(tuple(group.key for group in groups))
        for group, group_tab in zip(groups, group_tabs):
            with group_tab:
                chart_col, detail_col = st.columns((4, 1))
                with chart_col:
                    event = st.plotly_chart(
                        observable_comparison_figure(result, group, error_mode=error_mode),
                        width="stretch",
                        on_select="rerun",
                        selection_mode="points",
                        key=f"observable_{group.key}",
                    )
                with detail_col:
                    st.markdown(f"#### Inspect {group.key}")
                    points = event.selection.points if event and event.selection else []
                    if points:
                        point = points[0]
                        custom = point.get("customdata") or []
                        st.metric("Selected time", f"{float(point.get('x', 0.0)):.6g}")
                        st.write(f"Pipeline: `{float(point.get('y', 0.0)):.8g}`")
                        if len(custom) >= 4:
                            st.write(f"Variable: `{custom[3]}`")
                            st.write(f"Reference: `{float(custom[0]):.8g}`")
                            if error_mode in {"absolute", "both"}:
                                st.write(f"Absolute error: `{float(custom[1]):.3e}`")
                            if error_mode in {"relative", "both"}:
                                st.write(f"Relative error: `{float(custom[2]):.3e}`")
                        if result.error_report:
                            time_index = int(np.argmin(np.abs(result.error_report.times - float(point.get("x", 0.0)))))
                            st.markdown("**Staged errors**")
                            for name, values in result.error_report.components.items():
                                st.write(f"{name}: `{float(values[time_index]):.3e}`")
                    else:
                        st.info("Click a pipeline or error point to inspect it.")
        error_rows = []
        for index, label in enumerate(physical_labels):
            absolute = np.abs(
                np.asarray(result.physical_states[:, index]).real
                - np.asarray(result.reference_states[:, index]).real
            )
            row: dict[str, object] = {"variable": label}
            if error_mode in {"absolute", "both"}:
                row.update({"final absolute": absolute[-1], "maximum absolute": np.max(absolute)})
            if error_mode in {"relative", "both"}:
                relative = absolute / np.maximum(
                    np.abs(np.asarray(result.reference_states[:, index]).real), 1e-12
                )
                row.update({"final relative": relative[-1], "maximum relative": np.max(relative)})
            error_rows.append(row)
        st.markdown(f"#### {error_mode.title()} error summary")
        st.dataframe(error_rows, width="stretch", hide_index=True)

    with error_tab:
        st.caption(
            "These are staged pipeline diagnostics (truncation, discretization, and solver residual effects), "
            "not the selectable pointwise absolute/relative error shown on Results."
        )
        st.plotly_chart(error_figure(result), width="stretch")
        if result.error_report:
            st.dataframe(
                [
                    {"component": name, "final": value, "meaning": result.error_report.descriptions[name]}
                    for name, value in result.error_report.final.items()
                ],
                width="stretch",
            )
        if result.complexity_report:
            st.warning(result.complexity_report.caveat)
            metrics = result.complexity_report.metrics
            st.subheader("This run's complexity")
            st.markdown(
                f"**Runtime drivers:** lifted dimension `{metrics.get('lifted_dimension')}`, "
                f"nonzeros `{metrics.get('nnz')}`, solves `{metrics.get('linear_solves')}`, "
                f"condition proxy `{float(metrics.get('condition_number_proxy', 0.0)):.3g}`."
            )
            for stage in result.complexity_report.stages:
                with st.expander(stage, expanded=True):
                    for term in result.complexity_report.terms:
                        if term.stage != stage:
                            continue
                        st.markdown(f"**{term.label}**")
                        st.markdown(f"$${term.symbolic}$$")
                        st.write(f"Runtime substitution: {term.evaluated}")
                        st.caption(f"Source: {term.source}")
                        st.caption(f"Assumptions: {term.assumptions}")
            with st.expander("All estimated metrics"):
                st.json(result.complexity_report.to_dict())

    with math_tab:
        st.subheader("Mathematics used by this run")
        for title, latex, explanation in active_math_sections(result_config):
            st.markdown(f"### {title}")
            st.markdown(f"$${latex}$$")
            st.write(explanation)

    with reasoning_tab:
        st.subheader("Why this method / why it wins or fails")
        for title, explanation in active_reasoning(result_config):
            with st.expander(title, expanded=True):
                st.write(explanation)
        st.markdown(
            "[Full system-specific method comparison]"
            "(https://github.com/EshMis/QLS_testing/blob/main/docs/reasoning/"
            "method_comparisons_specific_system.md)"
        )

    with advanced_tab:
        st.caption("Lifted coordinates are retained only for truncation, instability, and solver diagnostics.")
        label = st.selectbox("Lifted coordinate", result.linearized_system.labels)
        st.plotly_chart(variable_figure(result, label), width="stretch")
        st.write("Integration metadata", result.integration.metadata)
        if result.integration.solve_diagnostics:
            latest_result = result.integration.solve_diagnostics[-1]
            latest = latest_result.metadata
            st.write(
                "Latest solver summary",
                {
                    "method": latest.get("method"),
                    "absolute_residual": latest_result.residual_norm,
                    "relative_residual": latest_result.relative_residual,
                    "steps": latest.get("steps"),
                    "condition_number": latest.get("condition_number"),
                },
            )
            if latest.get("method") == "pennylane_vqls":
                with st.expander("VQLS optimizer diagnostics", expanded=True):
                    st.metric("Initial loss", f"{float(latest['initial_cost']):.3e}")
                    st.metric("Best loss", f"{float(latest['cost']):.3e}")
                    st.metric("Parameter change", f"{float(latest['parameter_change_norm']):.3e}")
                    st.line_chart(
                        {
                            "loss": latest["loss_history"],
                            "gradient norm": [np.nan, *latest["gradient_norm_history"]],
                            "update norm": [np.nan, *latest["parameter_update_norm_history"]],
                        }
                    )
                    st.write("Measured probabilities", latest["measured_probabilities"])
            with st.expander("Raw solver metadata"):
                st.json(latest)

    st.download_button("Export CSV", result_csv(result), "qls_results.csv", "text/csv")
    st.download_button("Export NPZ + staged errors", result_npz(result), "qls_results.npz", "application/octet-stream")
    st.download_button("Export HTML plot", result_html(result), "qls_plot.html", "text/html")
