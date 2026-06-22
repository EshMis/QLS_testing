"""Variable-length mass-action chains used for structural scaling experiments."""

from __future__ import annotations

import numpy as np

from qls_testing.core.datatypes import PolynomialSystem


def build_mass_action_chain(
    reactions: int,
    *,
    initial_substrate: float = 3.0,
    k1: tuple[float, ...] | None = None,
    k_minus_1: tuple[float, ...] | None = None,
    kcat: tuple[float, ...] | None = None,
    enzyme_total: tuple[float, ...] | None = None,
) -> PolynomialSystem:
    """Build ``S -> X1 -> ... -> P`` with one complex per reaction."""
    if reactions < 1:
        raise ValueError("reactions must be positive")

    def rates(values: tuple[float, ...] | None, default: tuple[float, ...]) -> np.ndarray:
        source = values or default
        if len(source) != reactions:
            if values is not None:
                raise ValueError("each supplied rate vector must match reactions")
            source = tuple(default[index % len(default)] for index in range(reactions))
        array = np.asarray(source, dtype=float)
        if np.any(array <= 0.0):
            raise ValueError("rates must be positive")
        return array

    binding_rates = rates(k1, (2.0, 1.5, 1.0, 2.0))
    unbinding_rates = rates(k_minus_1, (1.0, 0.5, 0.5, 1.0))
    catalytic_rates = rates(kcat, (3.0, 2.0, 2.5, 4.0))
    enzyme_totals = rates(enzyme_total, (1.0, 1.0, 1.0, 1.0))
    dimension = 2 * reactions + 1
    equations: list[dict[tuple[int, ...], float]] = [dict() for _ in range(dimension)]

    def unit(index: int) -> tuple[int, ...]:
        exponent = [0] * dimension
        exponent[index] = 1
        return tuple(exponent)

    def product(first: int, second: int) -> tuple[int, ...]:
        exponent = [0] * dimension
        exponent[first] += 1
        exponent[second] += 1
        return tuple(exponent)

    def add(row: int, exponent: tuple[int, ...], value: float) -> None:
        equations[row][exponent] = equations[row].get(exponent, 0.0) + value

    complex_offset = reactions + 1
    for reaction in range(reactions):
        substrate = reaction
        complex_index = complex_offset + reaction
        binding = product(substrate, complex_index)
        k_on = binding_rates[reaction]
        k_off = unbinding_rates[reaction]
        k_cat = catalytic_rates[reaction]
        enzyme = enzyme_totals[reaction]
        add(substrate, unit(substrate), -k_on * enzyme)
        add(substrate, binding, k_on)
        add(substrate, unit(complex_index), k_off)
        if reaction:
            add(substrate, unit(complex_index - 1), catalytic_rates[reaction - 1])
        add(complex_index, unit(substrate), k_on * enzyme)
        add(complex_index, binding, -k_on)
        add(complex_index, unit(complex_index), -(k_off + k_cat))
    add(reactions, unit(complex_offset + reactions - 1), catalytic_rates[-1])
    initial = np.zeros(dimension)
    initial[0] = initial_substrate
    labels = (
        "S",
        *(f"X{index}" for index in range(1, reactions)),
        "P",
        *(f"C{index}" for index in range(1, reactions + 1)),
    )
    return PolynomialSystem(
        labels,
        tuple(equations),
        initial,
        {"model": "mass_action_chain", "reactions": reactions, "degree": 2},
    )
