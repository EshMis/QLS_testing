"""Method-independent, high-accuracy ground-truth trajectory generators."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.integrate import solve_ivp
from scipy.sparse.linalg import expm_multiply

from qls_testing.core.config import SystemConfig
from qls_testing.core.datatypes import PolynomialSystem
from qls_testing.models.lindblad import LindbladModel, NDMEEncoding
from qls_testing.models.practice import get_practice_system
from qls_testing.systems.metabolic import qssa_rhs


@dataclass(frozen=True)
class GroundTruth:
    """Physical trajectory that is independent of the tested numerical method."""

    times: np.ndarray
    states: np.ndarray
    label: str
    metadata: dict[str, object]


@dataclass(frozen=True)
class DensityGroundTruth:
    """Exact Liouvillian evolution and derived populations."""

    times: np.ndarray
    density_matrices: np.ndarray
    populations: np.ndarray
    label: str


def _accurate_solve(
    rhs: object,
    initial: np.ndarray,
    times: np.ndarray,
) -> np.ndarray:
    grid = np.asarray(times, dtype=float)
    solution = solve_ivp(
        rhs,
        (float(grid[0]), float(grid[-1])),
        np.asarray(initial),
        t_eval=grid,
        method="DOP853",
        rtol=2e-12,
        atol=2e-14,
    )
    if not solution.success:
        raise RuntimeError(f"ground-truth solve failed: {solution.message}")
    return solution.y.T


def solve_polynomial_ground_truth(
    polynomial: PolynomialSystem,
    times: np.ndarray,
) -> GroundTruth:
    """Solve the exact polynomial model, used to isolate model/truncation error."""
    offset = np.asarray(polynomial.metadata.get("projection_offset", 0.0))
    states = _accurate_solve(
        lambda _time, state: polynomial.evaluate(state),
        polynomial.initial_state,
        times,
    )
    return GroundTruth(
        np.asarray(times),
        np.real_if_close(states + offset),
        "Ground-truth polynomial ODE solution",
        {"generator": "DOP853", "rtol": 2e-12, "atol": 2e-14},
    )


def solve_ode_ground_truth(
    config: SystemConfig,
    times: np.ndarray,
    *,
    polynomial: PolynomialSystem,
) -> GroundTruth:
    """Return the physical ODE truth selected only by the system definition.

    The chosen integrator, linearization, and QLS solver are intentionally not
    accepted as arguments, preventing method-dependent reference drift.
    """
    grid = np.asarray(times, dtype=float)
    if config.name.startswith("practice_"):
        states = get_practice_system(config.name).reference(grid)
        generator = "analytic matrix exponential"
    elif config.name == "qssa_taylor_pathway":
        initial = np.asarray([config.initial_substrate, 0.0, 0.0, 0.0, 0.0])
        states = _accurate_solve(lambda _time, state: qssa_rhs(config, state), initial, grid)
        generator = "DOP853 rational QSSA"
    else:
        result = solve_polynomial_ground_truth(polynomial, grid)
        states = result.states
        generator = "DOP853 underlying ODE"
    return GroundTruth(
        grid,
        np.real_if_close(states),
        "Ground-truth ODE solution",
        {"generator": generator, "system": config.name},
    )


def solve_lindblad_ground_truth(
    model: LindbladModel,
    times: np.ndarray,
) -> DensityGroundTruth:
    """Apply the exact sparse Liouvillian exponential on the requested grid."""
    grid = np.asarray(times, dtype=float)
    initial = np.asarray(model.initial_density, dtype=complex).reshape(-1, order="F")
    if np.allclose(np.diff(grid), np.diff(grid)[0]):
        vectors = expm_multiply(
            model.liouvillian(),
            initial,
            start=float(grid[0]),
            stop=float(grid[-1]),
            num=len(grid),
            endpoint=True,
        )
    else:
        vectors = np.asarray(
            [expm_multiply(time * model.liouvillian(), initial) for time in grid]
        )
    shape = model.initial_density.shape
    densities = np.asarray([vector.reshape(shape, order="F") for vector in vectors])
    populations = np.real(np.diagonal(densities, axis1=1, axis2=2))
    return DensityGroundTruth(grid, densities, populations, "Ground-truth Lindbladian solution")


def solve_ndme_ground_truth(
    encoding: NDMEEncoding,
    times: np.ndarray,
) -> GroundTruth:
    """Evolve the exact NDME Liouvillian and extract its physical ODE block."""
    density_truth = solve_lindblad_ground_truth(encoding.model(), times)
    states = encoding.extract_states(density_truth.density_matrices, density_truth.times)
    return GroundTruth(
        density_truth.times,
        np.real_if_close(states),
        density_truth.label,
        {"generator": "exact sparse Liouvillian exponential", **encoding.metadata},
    )
