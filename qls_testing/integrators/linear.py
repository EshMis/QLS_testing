"""Time integrators for autonomous linear systems."""

from __future__ import annotations

import numpy as np
from scipy.integrate import solve_ivp
from scipy.linalg import expm

from qls_testing.core.datatypes import IntegrationResult, LinearizedSystem, SolveResult
from qls_testing.core.interfaces import Integrator, LinearSolver
from qls_testing.core.utils import uniform_grid


def _result(times: np.ndarray, states: list[np.ndarray] | np.ndarray, diagnostics: list[SolveResult], **metadata: object) -> IntegrationResult:
    return IntegrationResult(times, np.real_if_close(np.asarray(states)), tuple(diagnostics), metadata)


class BackwardEulerIntegrator(Integrator):
    """A-stable first-order scheme ``(I-dt A)y[k+1]=y[k]``."""

    def integrate(self, system: LinearizedSystem, solver: LinearSolver, *, t_final: float, dt: float, n_points: int | None = None) -> IntegrationResult:
        times = uniform_grid(t_final, dt)
        lhs = np.eye(system.matrix.shape[0]) - dt * system.matrix
        states = [system.initial_state]
        diagnostics = []
        for _ in times[1:]:
            solved = solver.solve(lhs, states[-1])
            diagnostics.append(solved)
            states.append(solved.solution)
        return _result(times, states, diagnostics, method="backward_euler", lhs_condition=float(np.linalg.cond(lhs)))


class CrankNicolsonIntegrator(Integrator):
    """A-stable second-order trapezoidal scheme."""

    def integrate(self, system: LinearizedSystem, solver: LinearSolver, *, t_final: float, dt: float, n_points: int | None = None) -> IntegrationResult:
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
        return _result(times, states, diagnostics, method="crank_nicolson", lhs_condition=float(np.linalg.cond(lhs)))


class BDF2Integrator(Integrator):
    """Second-order backward differentiation, bootstrapped by backward Euler."""

    def integrate(self, system: LinearizedSystem, solver: LinearSolver, *, t_final: float, dt: float, n_points: int | None = None) -> IntegrationResult:
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
        return _result(times, states, diagnostics, method="bdf2", lhs_condition=float(np.linalg.cond(lhs)))


class FoldedBackwardEulerIntegrator(Integrator):
    """Encode all backward-Euler time steps in one block lower-bidiagonal solve."""

    def integrate(self, system: LinearizedSystem, solver: LinearSolver, *, t_final: float, dt: float, n_points: int | None = None) -> IntegrationResult:
        times = uniform_grid(t_final, dt)
        steps = len(times) - 1
        n = system.matrix.shape[0]
        step_matrix = np.eye(n) - dt * system.matrix
        folded = np.kron(np.eye(steps), step_matrix) + np.kron(np.diag(-np.ones(steps - 1), -1), np.eye(n))
        rhs = np.zeros(steps * n, dtype=np.result_type(system.matrix, system.initial_state))
        rhs[:n] = system.initial_state
        solved = solver.solve(folded, rhs)
        states = np.vstack((system.initial_state, solved.solution.reshape(steps, n)))
        return _result(times, states, [solved], method="folded_backward_euler", lhs_condition=float(np.linalg.cond(folded)))


class Pade22Integrator(Integrator):
    """Evaluate the global [2/2] Padé approximation of ``exp(tA)``."""

    def integrate(self, system: LinearizedSystem, solver: LinearSolver, *, t_final: float, dt: float, n_points: int | None = None) -> IntegrationResult:
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
        return _result(times, states, diagnostics, method="pade22_global")


class RK45Integrator(Integrator):
    """Adaptive SciPy reference; it does not expose QLS subproblems."""

    def integrate(self, system: LinearizedSystem, solver: LinearSolver, *, t_final: float, dt: float, n_points: int | None = None) -> IntegrationResult:
        times = np.linspace(0.0, t_final, n_points or len(uniform_grid(t_final, dt)))
        solution = solve_ivp(lambda _t, y: system.matrix @ y, (0.0, t_final), system.initial_state, t_eval=times, rtol=1e-10, atol=1e-12)
        if not solution.success:
            raise RuntimeError(solution.message)
        return _result(solution.t, solution.y.T, [], method="rk45_reference", function_evaluations=solution.nfev)


class ExponentialIntegrator(Integrator):
    """Matrix-exponential reference for time-independent lifted systems."""

    def integrate(self, system: LinearizedSystem, solver: LinearSolver, *, t_final: float, dt: float, n_points: int | None = None) -> IntegrationResult:
        times = np.linspace(0.0, t_final, n_points or len(uniform_grid(t_final, dt)))
        states = [expm(time * system.matrix) @ system.initial_state for time in times]
        return _result(times, states, [], method="matrix_exponential_reference")

