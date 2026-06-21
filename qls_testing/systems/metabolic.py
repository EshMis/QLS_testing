"""Four-reaction enzyme pathway from the original notebook."""

from __future__ import annotations

import numpy as np

from qls_testing.core.config import SystemConfig
from qls_testing.core.datatypes import PolynomialSystem


def _unit(n: int, index: int, power: int = 1) -> tuple[int, ...]:
    values = [0] * n
    values[index] = power
    return tuple(values)


def _product(n: int, first: int, second: int) -> tuple[int, ...]:
    values = [0] * n
    values[first] += 1
    values[second] += 1
    return tuple(values)


def _add(terms: dict[tuple[int, ...], float], exponent: tuple[int, ...], value: float) -> None:
    terms[exponent] = terms.get(exponent, 0.0) + value


def build_mass_action_pathway(config: SystemConfig) -> PolynomialSystem:
    """Return the exact nine-state quadratic mass-action model."""
    n = 9
    equations = [dict() for _ in range(n)]
    substrate_indices = (0, 1, 2, 3)
    complex_indices = (5, 6, 7, 8)
    for reaction, (substrate, complex_) in enumerate(zip(substrate_indices, complex_indices)):
        k1 = config.k1[reaction]
        km1 = config.k_minus_1[reaction]
        kcat = config.kcat[reaction]
        enzyme = config.enzyme_total[reaction]
        binding = _product(n, substrate, complex_)

        # Substrate is S for reaction 1 and X_i for later reactions.
        _add(equations[substrate], _unit(n, substrate), -k1 * enzyme)
        _add(equations[substrate], binding, k1)
        _add(equations[substrate], _unit(n, complex_), km1)
        if reaction > 0:
            _add(equations[substrate], _unit(n, complex_ - 1), config.kcat[reaction - 1])

        _add(equations[complex_], _unit(n, substrate), k1 * enzyme)
        _add(equations[complex_], binding, -k1)
        _add(equations[complex_], _unit(n, complex_), -(km1 + kcat))

    _add(equations[4], _unit(n, 8), config.kcat[3])
    initial = np.zeros(n)
    initial[0] = config.initial_substrate
    return PolynomialSystem(
        variable_names=("S", "X1", "X2", "X3", "P", "C1", "C2", "C3", "C4"),
        terms=tuple(equations),
        initial_state=initial,
        metadata={"model": "mass_action_pathway", "degree": 2},
    )


def build_qssa_taylor_pathway(config: SystemConfig) -> PolynomialSystem:
    """Return a Taylor-polynomial approximation of QSSA Michaelis--Menten rates.

    The series is centered at zero and is only convergent when ``|x/Km| < 1``.
    This limitation is surfaced in metadata and diagnostics rather than hidden.
    """
    n = 5
    equations = [dict() for _ in range(n)]
    vmax = np.asarray(config.kcat) * np.asarray(config.enzyme_total)
    km = (np.asarray(config.k_minus_1) + np.asarray(config.kcat)) / np.asarray(config.k1)
    for reaction in range(4):
        for degree in range(1, config.taylor_degree + 1):
            coefficient = vmax[reaction] * (-1) ** (degree - 1) / km[reaction] ** degree
            exponent = _unit(n, reaction, degree)
            _add(equations[reaction], exponent, -coefficient)
            _add(equations[reaction + 1], exponent, coefficient)
    initial = np.zeros(n)
    initial[0] = config.initial_substrate
    return PolynomialSystem(
        variable_names=("S", "X1", "X2", "X3", "P"),
        terms=tuple(equations),
        initial_state=initial,
        metadata={
            "model": "qssa_taylor_pathway",
            "degree": config.taylor_degree,
            "km": km.tolist(),
            "initial_radius_ratio": float(config.initial_substrate / km[0]),
            "series_valid_initially": bool(config.initial_substrate < km[0]),
        },
    )


def qssa_rhs(config: SystemConfig, state: np.ndarray) -> np.ndarray:
    """Evaluate the unapproximated rational QSSA model for reference solutions."""
    substrate = np.asarray(state[:4])
    vmax = np.asarray(config.kcat) * np.asarray(config.enzyme_total)
    km = (np.asarray(config.k_minus_1) + np.asarray(config.kcat)) / np.asarray(config.k1)
    rates = vmax * substrate / (km + substrate)
    return np.asarray((-rates[0], rates[0] - rates[1], rates[1] - rates[2], rates[2] - rates[3], rates[3]))

