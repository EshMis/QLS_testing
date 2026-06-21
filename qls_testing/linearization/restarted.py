"""Adaptive/restarted finite Carleman evolution."""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from qls_testing.core.datatypes import PolynomialSystem, SolveResult
from qls_testing.core.interfaces import Integrator, LinearSolver
from qls_testing.linearization.carleman import CarlemanLinearization


@dataclass(frozen=True)
class RestartedCarlemanResult:
    times: np.ndarray
    physical_states: np.ndarray
    orders: tuple[int, ...]
    local_order_errors: np.ndarray
    solve_diagnostics: tuple[SolveResult, ...] = ()


class AdaptiveRestartedCarleman:
    """Restart the monomial embedding and adapt order segment by segment.

    On each segment, orders ``N`` and ``N+1`` are compared at the endpoint.
    The order grows until the physical-state discrepancy is below tolerance or
    ``max_order`` is reached. Re-embedding the accepted physical endpoint makes
    monomials mutually consistent again and limits accumulated lift drift.
    """

    def __init__(
        self,
        *,
        initial_order: int = 1,
        max_order: int = 4,
        segment_duration: float = 0.5,
        tolerance: float = 1e-3,
    ) -> None:
        if not 1 <= initial_order <= max_order:
            raise ValueError("require 1 <= initial_order <= max_order")
        self.initial_order = initial_order
        self.max_order = max_order
        self.segment_duration = segment_duration
        self.tolerance = tolerance

    def evolve(
        self,
        system: PolynomialSystem,
        integrator: Integrator,
        solver: LinearSolver,
        *,
        t_final: float,
        dt: float,
    ) -> RestartedCarlemanResult:
        if not np.isclose(t_final / self.segment_duration, round(t_final / self.segment_duration)):
            raise ValueError("t_final must be an integer multiple of segment_duration")
        if not np.isclose(self.segment_duration / dt, round(self.segment_duration / dt)):
            raise ValueError("segment_duration must be an integer multiple of dt")
        offset = np.asarray(system.metadata.get("projection_offset", np.zeros(len(system.variable_names))))
        current_coordinates = np.asarray(system.initial_state, dtype=float)
        all_times = [0.0]
        all_states = [current_coordinates + offset]
        orders: list[int] = []
        errors: list[float] = []
        diagnostics: list[SolveResult] = []
        elapsed = 0.0

        while elapsed < t_final - 1e-14:
            local_system = replace(system, initial_state=current_coordinates)
            order = self.initial_order
            accepted = None
            local_error = np.inf
            while True:
                low = CarlemanLinearization(order).linearize(local_system)
                low_result = integrator.integrate(
                    low, solver, t_final=self.segment_duration, dt=dt,
                    n_points=int(round(self.segment_duration / dt)) + 1,
                )
                accepted = low_result
                if order >= self.max_order:
                    break
                high = CarlemanLinearization(order + 1).linearize(local_system)
                high_result = integrator.integrate(
                    high, solver, t_final=self.segment_duration, dt=dt,
                    n_points=int(round(self.segment_duration / dt)) + 1,
                )
                low_end = low.project(low_result.states[-1])
                high_end = high.project(high_result.states[-1])
                local_error = float(np.linalg.norm(high_end - low_end))
                accepted = high_result
                if local_error <= self.tolerance:
                    order += 1
                    break
                order += 1

            assert accepted is not None
            accepted_lift = CarlemanLinearization(order).linearize(local_system)
            physical_segment = accepted_lift.project(accepted.states)
            segment_times = elapsed + accepted.times
            all_times.extend(segment_times[1:])
            all_states.extend(physical_segment[1:])
            current_coordinates = np.asarray(physical_segment[-1]) - offset
            elapsed += self.segment_duration
            orders.append(order)
            errors.append(local_error)
            diagnostics.extend(accepted.solve_diagnostics)

        return RestartedCarlemanResult(
            np.asarray(all_times),
            np.asarray(all_states),
            tuple(orders),
            np.asarray(errors),
            tuple(diagnostics),
        )
