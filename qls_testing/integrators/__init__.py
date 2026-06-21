"""Registered linear-system time integrators."""

from qls_testing.experiments.registry import INTEGRATORS

from .linear import (
    BDF2Integrator,
    BackwardEulerIntegrator,
    CrankNicolsonIntegrator,
    ExponentialIntegrator,
    FoldedBackwardEulerIntegrator,
    Pade22Integrator,
    RK45Integrator,
)

INTEGRATORS.register("backward_euler", BackwardEulerIntegrator)
INTEGRATORS.register("crank_nicolson", CrankNicolsonIntegrator)
INTEGRATORS.register("bdf2", BDF2Integrator)
INTEGRATORS.register("folded_backward_euler", FoldedBackwardEulerIntegrator)
INTEGRATORS.register("pade22", Pade22Integrator)
INTEGRATORS.register("rk45", RK45Integrator)
INTEGRATORS.register("exponential", ExponentialIntegrator)

__all__ = [
    "BDF2Integrator", "BackwardEulerIntegrator", "CrankNicolsonIntegrator",
    "ExponentialIntegrator", "FoldedBackwardEulerIntegrator", "Pade22Integrator", "RK45Integrator",
]

