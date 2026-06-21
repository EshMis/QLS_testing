"""Registered nonlinear-to-linear lifting methods."""

from qls_testing.experiments.registry import LINEARIZATIONS

from .carleman import CarlemanLinearization

LINEARIZATIONS.register("carleman", CarlemanLinearization)

__all__ = ["CarlemanLinearization"]

