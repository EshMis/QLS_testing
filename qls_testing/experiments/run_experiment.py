"""Run the same pipeline from Python, CLI, tests, or visualization apps."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.integrate import solve_ivp

from qls_testing.core.config import Config, load_config
from qls_testing.core.datatypes import ExperimentResult, IntegrationResult
from qls_testing.complexity import DefaultComplexityEstimator
from qls_testing.error import PipelineErrorModel
from qls_testing.error import ErrorReport
from qls_testing.experiments.registry import INTEGRATORS, LINEARIZATIONS, QLS_SOLVERS
from qls_testing.models import (
    LindbladResult,
    amplitude_damping_model,
    build_toy_linear_ode,
    simulate_lindblad,
    simulate_ndme_linear_ode,
    get_practice_system,
)
from qls_testing.qls.classical import ClassicalSolver
from qls_testing.linearization.carleman import CarlemanLinearization, lift_state
from qls_testing.linearization.restarted import AdaptiveRestartedCarleman
from qls_testing.systems.metabolic import build_mass_action_pathway, build_qssa_pathway, qssa_rhs

# Import modules for their explicit registration side effects.
import qls_testing.integrators  # noqa: F401,E402
import qls_testing.linearization  # noqa: F401,E402
import qls_testing.qls  # noqa: F401,E402


def run_experiment(config: Config) -> ExperimentResult:
    """Build, lift, integrate, and verify one configured experiment."""
    config.validate()
    np.random.seed(config.random_seed)
    practice_model = None
    if config.system.name == "mass_action_pathway":
        polynomial = build_mass_action_pathway(config.system)
    elif config.system.name == "qssa_taylor_pathway":
        polynomial = build_qssa_pathway(config.system)
    elif config.system.name == "toy_linear_ode":
        polynomial = build_toy_linear_ode(config.system.toy_coupling)
    elif config.system.name.startswith("practice_"):
        practice_model = get_practice_system(config.system.name)
        polynomial = practice_model.polynomial_system()
    else:
        raise ValueError(
            "unknown system.name; choose an enzyme, toy, or registered practice system"
        )

    solver_settings = {**config.qls.settings}
    if "vqls" in config.qls.name:
        solver_settings.setdefault("seed", config.random_seed)
    solver = QLS_SOLVERS.create(config.qls.name, **solver_settings)
    integrator = INTEGRATORS.create(config.integrator.name, **config.integrator.settings)
    if config.linearization.name == "adaptive_restarted_carleman":
        controller = AdaptiveRestartedCarleman(**config.linearization.settings)
        restarted = controller.evolve(
            polynomial,
            integrator,
            solver,
            t_final=config.time.t_final,
            dt=config.time.dt,
        )
        reporting_order = max(restarted.orders)
        lifted = CarlemanLinearization(reporting_order).linearize(polynomial)
        offset = np.asarray(polynomial.metadata.get("projection_offset", 0.0))
        lifted_states = np.asarray(
            [lift_state(state - offset, lifted.exponents) for state in restarted.physical_states]
        )
        integration = IntegrationResult(
            restarted.times,
            lifted_states,
            restarted.solve_diagnostics,
            {
                "method": "adaptive_restarted_carleman",
                "orders": restarted.orders,
                "local_order_errors": restarted.local_order_errors.tolist(),
            },
        )
    else:
        linearizer = LINEARIZATIONS.create(
            config.linearization.name, **config.linearization.settings
        )
        lifted = linearizer.linearize(polynomial)
        integration = integrator.integrate(
            lifted,
            solver,
            t_final=config.time.t_final,
            dt=config.time.dt,
            n_points=config.time.n_points,
            rtol=config.time.rtol,
            atol=config.time.atol,
            min_step=config.time.min_step,
            max_step=config.time.max_step,
            output_stride=config.time.output_stride,
        )
    physical = np.asarray(lifted.project(integration.states))
    offset = np.asarray(polynomial.metadata.get("projection_offset", 0.0))
    if practice_model is not None:
        polynomial_reference = practice_model.reference(integration.times)
    else:
        polynomial_solution = solve_ivp(
            lambda _time, state: polynomial.evaluate(state),
            (0.0, config.time.t_final),
            polynomial.initial_state,
            t_eval=integration.times,
            method="LSODA",
            rtol=1e-10,
            atol=1e-12,
        )
        if not polynomial_solution.success:
            raise RuntimeError(polynomial_solution.message)
        polynomial_reference = polynomial_solution.y.T + offset
    if config.system.name == "qssa_taylor_pathway":
        physical_initial = np.asarray([config.system.initial_substrate, 0.0, 0.0, 0.0, 0.0])
        target_solution = solve_ivp(
            lambda _time, state: qssa_rhs(config.system, state),
            (0.0, config.time.t_final),
            physical_initial,
            t_eval=integration.times,
            method="LSODA",
            rtol=1e-10,
            atol=1e-12,
        )
        if not target_solution.success:
            raise RuntimeError(target_solution.message)
        target_reference = target_solution.y.T
    else:
        target_reference = polynomial_reference
    difference = physical - target_reference
    solve_residuals = [item.relative_residual for item in integration.solve_diagnostics]
    metrics = {
        "rmse_by_state": np.sqrt(np.mean(np.abs(difference) ** 2, axis=0)).tolist(),
        "max_abs_error_by_state": np.max(np.abs(difference), axis=0).tolist(),
        "global_rmse": float(np.sqrt(np.mean(np.abs(difference) ** 2))),
        "max_relative_linear_solve_residual": float(max(solve_residuals, default=0.0)),
        "lifted_dimension": lifted.matrix.shape[0],
        "matrix_sparsity": lifted.metadata["sparsity"],
    }
    error_report = None
    if config.error.enabled:
        classical_integration = integrator.integrate(
            lifted,
            ClassicalSolver(),
            t_final=config.time.t_final,
            dt=config.time.dt,
            n_points=config.time.n_points,
            rtol=config.time.rtol,
            atol=config.time.atol,
            min_step=config.time.min_step,
            max_step=config.time.max_step,
            output_stride=config.time.output_stride,
        )
        error_report = PipelineErrorModel().estimate(
            lifted=lifted,
            integration=integration,
            classical_integration=classical_integration,
            polynomial_reference=polynomial_reference,
            target_reference=target_reference,
        )
        metrics["error_components_final"] = error_report.final
    complexity_report = None
    if config.complexity.enabled:
        complexity_report = DefaultComplexityEstimator().estimate(
            lifted=lifted,
            integration=integration,
            integrator_name=config.integrator.name,
            solver_name=config.qls.name,
        )
        metrics["complexity"] = complexity_report.to_dict()
    return ExperimentResult(
        config,
        lifted,
        integration,
        physical,
        integration.times,
        target_reference,
        metrics,
        error_report,
        complexity_report,
    )


def run_lindblad_experiment(config: Config) -> LindbladResult | ExperimentResult:
    """Run the separate open-system pathway; no QLS method is involved."""
    if config.system.name == "lindblad_amplitude_damping":
        model = amplitude_damping_model(config.system.lindblad_decay_rate)
        return simulate_lindblad(model, config.time.t_final, config.time.n_points)
    if config.system.name == "lindblad_enzyme_ndme":
        return run_ndme_enzyme_experiment(config)
    raise ValueError("unknown Lindblad system; choose lindblad_enzyme_ndme")


def run_ndme_enzyme_experiment(config: Config) -> ExperimentResult:
    """Run the PDF-based NDME construction on the reduced enzyme Carleman model."""
    requested_order = int(config.linearization.settings.get("order", 1))
    if requested_order != 1:
        raise ValueError(
            "lindblad_enzyme_ndme intentionally uses Carleman order 1; order 2 would create "
            "a 108-dimensional density matrix and is disabled for interactive use"
        )
    polynomial = build_mass_action_pathway(config.system)
    lifted = CarlemanLinearization(order=1).linearize(polynomial)
    times = np.linspace(0.0, config.time.t_final, config.time.n_points)
    ndme = simulate_ndme_linear_ode(
        lifted.matrix,
        lifted.initial_state,
        times,
        rtol=config.time.rtol,
        atol=config.time.atol,
    )
    integration = IntegrationResult(
        times,
        ndme.encoded_states,
        (),
        {"method": "lindblad_ndme", **ndme.metadata},
    )
    physical = np.asarray(lifted.project(ndme.encoded_states), dtype=float)
    reference = np.asarray(lifted.project(ndme.reference_states), dtype=float)
    difference = physical - reference
    physical_error = np.linalg.norm(difference, axis=1)
    positivity_violation = np.maximum(-ndme.minimum_eigenvalue, 0.0)
    error_report = ErrorReport(
        times,
        {
            "lindblad_ndme_integration": physical_error,
            "trace_preservation": ndme.trace_error,
            "positivity_violation": positivity_violation,
            "total_observed": physical_error,
        },
        {
            "lindblad_ndme_integration": "Extracted NDME state versus exp(A t) for the same reduced enzyme model.",
            "trace_preservation": "Absolute density-matrix trace defect.",
            "positivity_violation": "Magnitude of any negative density-matrix eigenvalue.",
            "total_observed": "Pipeline versus reduced-model reference in physical variables.",
        },
    )
    complexity_report = DefaultComplexityEstimator().estimate(
        lifted=lifted,
        integration=integration,
        integrator_name="lindblad_ndme",
        solver_name="not_applicable",
    )
    complexity_report.metrics.update(
        {
            "density_dimension": ndme.metadata["density_dimension"],
            "liouville_dimension": ndme.metadata["liouville_dimension"],
            "semidissipative_shift": ndme.metadata["semidissipative_shift"],
        }
    )
    metrics = {
        "global_rmse": float(np.sqrt(np.mean(np.abs(difference) ** 2))),
        "max_abs_error_by_state": np.max(np.abs(difference), axis=0).tolist(),
        "lifted_dimension": lifted.matrix.shape[0],
        "matrix_sparsity": lifted.metadata["sparsity"],
        "max_trace_error": float(np.max(ndme.trace_error)),
        "minimum_density_eigenvalue": float(np.min(ndme.minimum_eigenvalue)),
        "ndme": ndme.metadata,
        "error_components_final": error_report.final,
        "complexity": complexity_report.to_dict(),
        "reference_scope": "exact exponential of the reduced Carleman-order-1 enzyme model",
    }
    return ExperimentResult(
        config,
        lifted,
        integration,
        physical,
        times,
        reference,
        metrics,
        error_report,
        complexity_report,
    )


def experiment_summary(result: ExperimentResult) -> dict[str, object]:
    return {
        "system": result.config.system.name,
        "linearization": result.config.linearization.name,
        "integrator": result.config.integrator.name,
        "solver": result.config.qls.name,
        **result.metrics,
        "linearization_metadata": result.linearized_system.metadata,
        "errors": result.error_report.to_dict() if result.error_report else None,
        "complexity": result.complexity_report.to_dict() if result.complexity_report else None,
    }


def lindblad_summary(result: LindbladResult) -> dict[str, object]:
    return {
        **result.metadata,
        "final_populations": result.populations[-1].tolist(),
        "max_trace_error": float(np.max(result.trace_error)),
        "minimum_density_eigenvalue": float(np.min(result.minimum_eigenvalue)),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument(
        "--set",
        action="append",
        default=[],
        metavar="PATH=VALUE",
        help="Override a YAML value, for example --set time.dt=0.05",
    )
    parser.add_argument("--no-plot", action="store_true")
    args = parser.parse_args(argv)
    config = load_config(args.config, args.set)
    is_lindblad = config.system.name.startswith("lindblad_")
    result = run_lindblad_experiment(config) if is_lindblad else run_experiment(config)
    output = Path(config.output.directory)
    output.mkdir(parents=True, exist_ok=True)
    summary_path = output / "summary.json"
    is_pipeline_result = isinstance(result, ExperimentResult)
    summary = experiment_summary(result) if is_pipeline_result else lindblad_summary(result)
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    if config.output.save_plot and not args.no_plot:
        from qls_testing.visualization.plotters import lindblad_figure, trajectory_figure

        figure = trajectory_figure(result) if is_pipeline_result else lindblad_figure(result)
        figure.write_html(output / "trajectories.html")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
