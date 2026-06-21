"""Tiny ODE used for fast classical and PennyLane verification."""

from __future__ import annotations

import numpy as np

from qls_testing.core.datatypes import PolynomialSystem


def build_toy_linear_ode(coupling: float = 0.5) -> PolynomialSystem:
    """Return ``x'=-x+c y, y'=-2y`` with a nontrivial two-state trajectory."""
    return PolynomialSystem(
        variable_names=("x", "y"),
        terms=(
            {(1, 0): -1.0, (0, 1): float(coupling)},
            {(0, 1): -2.0},
        ),
        initial_state=np.asarray([1.0, 0.5]),
        metadata={"model": "toy_linear_ode", "degree": 1},
    )

