"""Time integrators for autonomous linear systems."""

from __future__ import annotations

import numpy as np
from scipy.integrate import solve_ivp
from scipy.linalg import expm
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import expm_multiply

from qls_testing.core.datatypes import IntegrationResult, LinearizedSystem, SolveResult
from qls_testing.core.interfaces import Integrator, LinearSolver
from qls_testing.core.utils import uniform_grid
from qls_testing.hardware_path.folded_systems import (
    FoldedSystem,
    build_folded_backward_euler,
    build_folded_bdf2,
    build_folded_crank_nicolson,
)


def _result(
    times: np.ndarray,
    states: list[np.ndarray] | np.ndarray,
    diagnostics: list[SolveResult],
    *,
    output_stride: int = 1,
    **metadata: object,
) -> IntegrationResult:
    values = np.real_if_close(np.asarray(states))
    indices = list(range(0, len(times), output_stride))
    if indices[-1] != len(times) - 1:
        indices.append(len(times) - 1)
    return IntegrationResult(times[indices], values[indices], tuple(diagnostics), metadata)


class BackwardEulerIntegrator(Integrator):
    """A-stable first-order scheme ``(I-dt A)y[k+1]=y[k]``."""

    def integrate(self, system: LinearizedSystem, solver: LinearSolver, *, t_final: float, dt: float, n_points: int | None = None, **options: object) -> IntegrationResult:
        times = uniform_grid(t_final, dt)
        lhs = np.eye(system.matrix.shape[0]) - dt * system.matrix
        states = [system.initial_state]
        diagnostics = []
        for _ in times[1:]:
            solved = solver.solve(lhs, states[-1])
            diagnostics.append(solved)
            states.append(solved.solution)
        return _result(times, states, diagnostics, output_stride=int(options.get("output_stride", 1)), method="backward_euler", lhs_condition=float(np.linalg.cond(lhs)))


class CrankNicolsonIntegrator(Integrator):
    """A-stable second-order trapezoidal scheme."""

    def integrate(self, system: LinearizedSystem, solver: LinearSolver, *, t_final: float, dt: float, n_points: int | None = None, **options: object) -> IntegrationResult:
        times = uniform_grid(t_final, dt)
        identity = np.eye(system.matrix.shape[0])
        lhs = identity - 0.5 * dt * system.matrix
        rhs_operator = identity + 0.5 * dt * system.matrix
        states = [system.initial_state]
        diagnostics = []
        for _ in times[1:]:
            solved = solver.solve(lhs, rhs_operator @ states[-1])
            diagnostics.append(solved)
            states.append(solved.solution)
        return _result(times, states, diagnostics, output_stride=int(options.get("output_stride", 1)), method="crank_nicolson", lhs_condition=float(np.linalg.cond(lhs)))


class BDF2Integrator(Integrator):
    """Second-order backward differentiation, bootstrapped by backward Euler."""

    def integrate(self, system: LinearizedSystem, solver: LinearSolver, *, t_final: float, dt: float, n_points: int | None = None, **options: object) -> IntegrationResult:
        times = uniform_grid(t_final, dt)
        identity = np.eye(system.matrix.shape[0])
        states = [system.initial_state]
        diagnostics = []
        first = solver.solve(identity - dt * system.matrix, states[0])
        diagnostics.append(first)
        states.append(first.solution)
        lhs = 1.5 * identity - dt * system.matrix
        for _ in times[2:]:
            solved = solver.solve(lhs, 2.0 * states[-1] - 0.5 * states[-2])
            diagnostics.append(solved)
            states.append(solved.solution)
        return _result(times, states, diagnostics, output_stride=int(options.get("output_stride", 1)), method="bdf2", lhs_condition=float(np.linalg.cond(lhs)))


class FoldedBackwardEulerIntegrator(Integrator):
    """Encode all backward-Euler time steps in one block lower-bidiagonal solve."""

    def integrate(self, system: LinearizedSystem, solver: LinearSolver, *, t_final: float, dt: float, n_points: int | None = None, **options: object) -> IntegrationResult:
        times = uniform_grid(t_final, dt)
        steps = len(times) - 1
        folded = build_folded_backward_euler(system.matrix, system.initial_state, dt, steps)
        return _integrate_folded(times, system, folded, solver, options)


class FoldedCrankNicolsonIntegrator(Integrator):
    """Prepare the complete Crank--Nicolson trajectory with one solve/readout state."""

    def integrate(self, system: LinearizedSystem, solver: LinearSolver, *, t_final: float, dt: float, n_points: int | None = None, **options: object) -> IntegrationResult:
        times = uniform_grid(t_final, dt)
        folded = build_folded_crank_nicolson(
            system.matrix, system.initial_state, dt, len(times) - 1
        )
        return _integrate_folded(times, system, folded, solver, options)


class FoldedBDF2Integrator(Integrator):
    """Prepare a BE-bootstrapped BDF2 trajectory with one solve/readout state."""

    def integrate(self, system: LinearizedSystem, solver: LinearSolver, *, t_final: float, dt: float, n_points: int | None = None, **options: object) -> IntegrationResult:
        times = uniform_grid(t_final, dt)
        folded = build_folded_bdf2(system.matrix, system.initial_state, dt, len(times) - 1)
        return _integrate_folded(times, system, folded, solver, options)


def _integrate_folded(
    times: np.ndarray,
    system: LinearizedSystem,
    folded: FoldedSystem,
    solver: LinearSolver,
    options: dict[str, object],
) -> IntegrationResult:
    """Run a folded builder through the existing dense solver plugin boundary."""
    dense = folded.matrix.toarray()
    solved = solver.solve(dense, folded.rhs)
    states = np.vstack(
        (system.initial_state, solved.solution.reshape(folded.steps, folded.state_dimension))
    )
    condition = float(np.linalg.cond(dense)) if folded.dimension <= 1000 else float("nan")
    return _result(
        times,
        states,
        [solved],
        output_stride=int(options.get("output_stride", 1)),
        method=folded.method,
        lhs_condition=condition,
        folded_dimension=folded.dimension,
        folded_nnz=int(folded.matrix.nnz),
        kronecker_term_count=len(folded.kronecker_terms),
        qls_calls=1,
    )


class Pade22Integrator(Integrator):
    """Evaluate the global [2/2] Padé approximation of ``exp(tA)``."""

    def integrate(self, system: LinearizedSystem, solver: LinearSolver, *, t_final: float, dt: float, n_points: int | None = None, **options: object) -> IntegrationResult:
        times = np.linspace(0.0, t_final, n_points or len(uniform_grid(t_final, dt)))
        n = system.matrix.shape[0]
        identity = np.eye(n)
        squared = system.matrix @ system.matrix
        states = [system.initial_state]
        diagnostics = []
        for time in times[1:]:
            numerator = identity + 0.5 * time * system.matrix + (time * time / 12.0) * squared
            denominator = identity - 0.5 * time * system.matrix + (time * time / 12.0) * squared
            solved = solver.solve(denominator, numerator @ system.initial_state)
            diagnostics.append(solved)
            states.append(solved.solution)
        return _result(times, states, diagnostics, output_stride=int(options.get("output_stride", 1)), method="pade22_global")


class RK45Integrator(Integrator):
    """Adaptive SciPy reference; it does not expose QLS subproblems."""

    def integrate(self, system: LinearizedSystem, solver: LinearSolver, *, t_final: float, dt: float, n_points: int | None = None, **options: object) -> IntegrationResult:
        times = np.linspace(0.0, t_final, n_points or len(uniform_grid(t_final, dt)))
        max_step = options.get("max_step")
        solution = solve_ivp(
            lambda _t, y: system.matrix @ y,
            (0.0, t_final),
            system.initial_state,
            t_eval=times,
            rtol=float(options.get("rtol", 1e-7)),
            atol=float(options.get("atol", 1e-9)),
            max_step=np.inf if max_step is None else float(max_step),
        )
        if not solution.success:
            raise RuntimeError(solution.message)
        return _result(solution.t, solution.y.T, [], output_stride=int(options.get("output_stride", 1)), method="rk45_adaptive", function_evaluations=solution.nfev, min_step_requested=options.get("min_step"))


class ExponentialIntegrator(Integrator):
    """Matrix-exponential reference for time-independent lifted systems."""

    def integrate(self, system: LinearizedSystem, solver: LinearSolver, *, t_final: float, dt: float, n_points: int | None = None, **options: object) -> IntegrationResult:
        times = np.linspace(0.0, t_final, n_points or len(uniform_grid(t_final, dt)))
        states = [expm(time * system.matrix) @ system.initial_state for time in times]
        return _result(times, states, [], output_stride=int(options.get("output_stride", 1)), method="matrix_exponential_reference")


class KrylovExponentialIntegrator(Integrator):
    """Apply ``exp(tA)`` through sparse Krylov action without forming ``exp(A)``.

    SciPy chooses the internal scaling/Taylor/Krylov work. ``n_points`` controls
    output sampling rather than the internal stable step size.
    """

    def integrate(self, system: LinearizedSystem, solver: LinearSolver, *, t_final: float, dt: float, n_points: int | None = None, **options: object) -> IntegrationResult:
        count = n_points or len(uniform_grid(t_final, dt))
        times = np.linspace(0.0, t_final, count)
        sparse_matrix = csr_matrix(system.matrix)
        states = expm_multiply(
            sparse_matrix,
            system.initial_state,
            start=0.0,
            stop=t_final,
            num=count,
            endpoint=True,
        )
        return _result(
            times,
            states,
            [],
            output_stride=int(options.get("output_stride", 1)),
            method="krylov_exponential",
            nnz=int(sparse_matrix.nnz),
        )
