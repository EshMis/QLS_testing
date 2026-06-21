"""Interactive front end backed by the same Config and experiment pipeline."""

from __future__ import annotations

import streamlit as st

from qls_testing.core.config import Config, MethodConfig, OutputConfig, SystemConfig, TimeConfig
from qls_testing.experiments.run_experiment import run_experiment
from qls_testing.visualization.plotters import trajectory_figure


st.set_page_config(page_title="Carleman + QLS laboratory", layout="wide")
st.title("Carleman + QLS laboratory")
system_name = st.sidebar.selectbox("System", ("mass_action_pathway", "qssa_taylor_pathway"))
order = st.sidebar.slider("Carleman order", 1, 4, 2)
integrator = st.sidebar.selectbox("Integrator", ("bdf2", "backward_euler", "crank_nicolson", "folded_backward_euler", "pade22", "rk45", "exponential"))
solver = st.sidebar.selectbox("Linear solver", ("classical", "hhl_simulator", "qsvt_simulator", "vqls_simulator"))
t_final = st.sidebar.slider("Time horizon", 0.5, 10.0, 5.0, 0.5)
dt = st.sidebar.select_slider("Step size", options=(0.5, 0.25, 0.1, 0.05), value=0.1)
initial = st.sidebar.slider("Initial substrate", 0.1, 5.0, 3.0, 0.1)
taylor_degree = st.sidebar.slider("QSSA Taylor degree", 1, 6, 3)

config = Config(
    system=SystemConfig(name=system_name, initial_substrate=initial, taylor_degree=taylor_degree),
    linearization=MethodConfig("carleman", {"order": order}),
    integrator=MethodConfig(integrator),
    qls=MethodConfig(solver),
    time=TimeConfig(t_final=t_final, dt=dt, n_points=101),
    output=OutputConfig(save_plot=False),
)
if st.button("Run experiment", type="primary"):
    with st.spinner("Lifting and integrating…"):
        result = run_experiment(config)
    if result.linearized_system.metadata.get("series_valid_initially") is False:
        st.warning("The QSSA Taylor series is outside its convergence radius at t=0; error includes polynomial-model error.")
    st.plotly_chart(trajectory_figure(result), use_container_width=True)
    st.json(result.metrics)

