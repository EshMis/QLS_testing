"""Interactive per-variable, error, complexity, and export dashboard."""

from __future__ import annotations

import streamlit as st

from qls_testing.core.config import Config, MethodConfig, OutputConfig, SystemConfig, TimeConfig
from qls_testing.experiments.run_experiment import run_experiment, run_lindblad_experiment
from qls_testing.visualization.exports import result_csv, result_html, result_npz
from qls_testing.visualization.plotters import error_figure, lindblad_figure, trajectory_figure, variable_figure


st.set_page_config(page_title="Carleman + QLS laboratory", layout="wide")
st.title("Carleman + QLS laboratory")
system_name = st.sidebar.selectbox(
    "System",
    ("mass_action_pathway", "qssa_taylor_pathway", "toy_linear_ode", "lindblad_amplitude_damping"),
)
order = st.sidebar.slider("Carleman order", 1, 4, 2)
integrator = st.sidebar.selectbox(
    "Integrator",
    ("bdf2", "backward_euler", "crank_nicolson", "krylov_exponential", "rk45", "folded_backward_euler", "pade22"),
)
solver = st.sidebar.selectbox(
    "Linear solver",
    ("classical", "hhl_simulator", "qsvt_simulator", "iterative_refinement", "pennylane_vqls"),
)
t_final = st.sidebar.slider("Time horizon", 0.5, 10.0, 5.0, 0.5)
dt = st.sidebar.select_slider("Fixed/output step", options=(0.5, 0.25, 0.1, 0.05), value=0.1)
initial = st.sidebar.slider("Initial substrate", 0.1, 5.0, 3.0, 0.1)
taylor_degree = st.sidebar.slider("QSSA Taylor degree", 1, 6, 3)
qssa_expansion = st.sidebar.selectbox("QSSA expansion", ("moving_point", "auto", "zero"))

solver_settings: dict[str, object] = {}
if solver == "pennylane_vqls":
    solver_settings = {"layers": 2, "max_steps": 300, "stepsize": 0.05, "tolerance": 1e-7}
elif solver == "iterative_refinement":
    solver_settings = {"base_solver": "qsvt_simulator", "max_iterations": 3}

config = Config(
    system=SystemConfig(
        name=system_name,
        initial_substrate=initial,
        taylor_degree=taylor_degree,
        qssa_expansion=qssa_expansion,
    ),
    linearization=MethodConfig("carleman", {"order": order}),
    integrator=MethodConfig(integrator),
    qls=MethodConfig(solver, solver_settings),
    time=TimeConfig(t_final=t_final, dt=dt, n_points=101),
    output=OutputConfig(save_plot=False),
)
if st.button("Run experiment", type="primary"):
    with st.spinner("Running simulation…"):
        st.session_state["result"] = (
            run_lindblad_experiment(config)
            if system_name.startswith("lindblad_")
            else run_experiment(config)
        )

result = st.session_state.get("result")
if result is not None:
    if hasattr(result, "density_matrices"):
        st.info("Lindblad evolution is a separate master-equation pathway; no QLS solver is used.")
        st.plotly_chart(lindblad_figure(result), width="stretch")
        st.json(result.metadata)
    else:
        overview, inspect, errors, complexity = st.tabs(("Overview", "Inspect variable", "Error dashboard", "Complexity"))
        with overview:
            st.plotly_chart(trajectory_figure(result), width="stretch")
            st.json(result.metrics)
        with inspect:
            label = st.selectbox("Lifted variable", result.linearized_system.labels)
            event = st.plotly_chart(
                variable_figure(result, label),
                width="stretch",
                on_select="rerun",
                selection_mode="points",
                key="variable_inspector",
            )
            points = event.selection.points if event and event.selection else []
            if points:
                point = points[0]
                st.code(f"t={point.get('x')}\nvalue={point.get('y')}\nvariable={label}")
        with errors:
            st.plotly_chart(error_figure(result), width="stretch")
            if result.error_report:
                st.dataframe([{"component": name, "final": value, "meaning": result.error_report.descriptions[name]} for name, value in result.error_report.final.items()])
        with complexity:
            if result.complexity_report:
                st.warning(result.complexity_report.caveat)
                st.json(result.complexity_report.to_dict())
        st.download_button("Export CSV", result_csv(result), "qls_results.csv", "text/csv")
        st.download_button("Export NPZ + errors", result_npz(result), "qls_results.npz", "application/octet-stream")
        st.download_button("Export HTML plot", result_html(result), "qls_plot.html", "text/html")
