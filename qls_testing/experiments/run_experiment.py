"""Run the same pipeline from Python, CLI, tests, or visualization apps."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.integrate import solve_ivp

from qls_testing.core.config import Config, load_config
from qls_testing.core.datatypes import ExperimentResult
from qls_testing.experiments.registry import INTEGRATORS, LINEARIZATIONS, QLS_SOLVERS
from qls_testing.systems.metabolic import build_mass_action_pathway, build_qssa_taylor_pathway, qssa_rhs

# Import modules for their explicit registration side effects.
import qls_testing.integrators  # noqa: F401,E402
import qls_testing.linearization  # noqa: F401,E402
import qls_testing.qls  # noqa: F401,E402


def run_experiment(config: Config) -> ExperimentResult:
    """Build, lift, integrate, and verify one configured experiment."""
    config.validate()
    np.random.seed(config.random_seed)
    if config.system.name == "mass_action_pathway":
        polynomial = build_mass_action_pathway(config.system)

        def reference_rhs(_time: float, state: np.ndarray) -> np.ndarray:
            return polynomial.evaluate(state)

    elif config.system.name == "qssa_taylor_pathway":
        polynomial = build_qssa_taylor_pathway(config.system)

        def reference_rhs(_time: float, state: np.ndarray) -> np.ndarray:
            return qssa_rhs(config.system, state)
    else:
        raise ValueError("system.name must be 'mass_action_pathway' or 'qssa_taylor_pathway'")

    linearizer = LINEARIZATIONS.create(config.linearization.name, **config.linearization.settings)
    lifted = linearizer.linearize(polynomial)
    solver_settings = {**config.qls.settings}
    solver_settings.setdefault("seed", config.random_seed) if config.qls.name == "vqls_simulator" else None
    solver = QLS_SOLVERS.create(config.qls.name, **solver_settings)
    integrator = INTEGRATORS.create(config.integrator.name, **config.integrator.settings)
    integration = integrator.integrate(
        lifted,
        solver,
        t_final=config.time.t_final,
        dt=config.time.dt,
        n_points=config.time.n_points,
    )
    physical = np.asarray(lifted.project(integration.states), dtype=float)
    reference = solve_ivp(
        reference_rhs,
        (0.0, config.time.t_final),
        polynomial.initial_state,
        t_eval=integration.times,
        method="LSODA",
        rtol=1e-10,
        atol=1e-12,
    )
    if not reference.success:
        raise RuntimeError(reference.message)
    difference = physical - reference.y.T
    solve_residuals = [item.relative_residual for item in integration.solve_diagnostics]
    metrics = {
        "rmse_by_state": np.sqrt(np.mean(np.abs(difference) ** 2, axis=0)).tolist(),
        "max_abs_error_by_state": np.max(np.abs(difference), axis=0).tolist(),
        "global_rmse": float(np.sqrt(np.mean(np.abs(difference) ** 2))),
        "max_relative_linear_solve_residual": float(max(solve_residuals, default=0.0)),
        "lifted_dimension": lifted.matrix.shape[0],
        "matrix_sparsity": lifted.metadata["sparsity"],
    }
    return ExperimentResult(config, lifted, integration, physical, reference.t, reference.y.T, metrics)


def _summary(result: ExperimentResult) -> dict[str, object]:
    return {
        "system": result.config.system.name,
        "linearization": result.config.linearization.name,
        "integrator": result.config.integrator.name,
        "solver": result.config.qls.name,
        **result.metrics,
        "linearization_metadata": result.linearized_system.metadata,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--no-plot", action="store_true")
    args = parser.parse_args(argv)
    config = load_config(args.config)
    result = run_experiment(config)
    output = Path(config.output.directory)
    output.mkdir(parents=True, exist_ok=True)
    summary_path = output / "summary.json"
    summary_path.write_text(json.dumps(_summary(result), indent=2), encoding="utf-8")
    if config.output.save_plot and not args.no_plot:
        from qls_testing.visualization.plotters import trajectory_figure

        trajectory_figure(result).write_html(output / "trajectories.html")
    print(json.dumps(_summary(result), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
