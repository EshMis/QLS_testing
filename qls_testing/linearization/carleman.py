"""Finite Carleman lifting for sparse polynomial vector fields."""

from __future__ import annotations

from itertools import combinations_with_replacement
from math import comb

import numpy as np

from qls_testing.core.datatypes import Exponent, LinearizedSystem, PolynomialSystem
from qls_testing.core.interfaces import LinearizationMethod


def enumerate_monomials(dimension: int, order: int) -> tuple[Exponent, ...]:
    """Enumerate commutative monomials by total degree, matching the notebook."""
    exponents: list[Exponent] = []
    for degree in range(1, order + 1):
        for indices in combinations_with_replacement(range(dimension), degree):
            exponent = [0] * dimension
            for index in indices:
                exponent[index] += 1
            exponents.append(tuple(exponent))
    return tuple(exponents)


def lift_state(state: np.ndarray, exponents: tuple[Exponent, ...]) -> np.ndarray:
    x = np.asarray(state)
    return np.asarray([np.prod(x ** np.asarray(exponent)) for exponent in exponents], dtype=float)


class CarlemanLinearization(LinearizationMethod):
    """Truncate the infinite Carleman system at a requested monomial degree.

    For ``m_alpha(x)=x**alpha``, each row follows directly from
    ``d m_alpha/dt = sum_i alpha_i x**(alpha-e_i) f_i(x)``. Terms above the
    truncation order are omitted and recorded in metadata.
    """

    def __init__(self, order: int = 2) -> None:
        if order < 1:
            raise ValueError("order must be >= 1")
        self.order = order

    def linearize(self, system: PolynomialSystem, **settings: object) -> LinearizedSystem:
        n = len(system.variable_names)
        exponents = enumerate_monomials(n, self.order)
        expected = sum(comb(n + degree - 1, degree) for degree in range(1, self.order + 1))
        assert len(exponents) == expected
        index = {exponent: column for column, exponent in enumerate(exponents)}
        matrix = np.zeros((len(exponents), len(exponents)))
        dropped = 0

        for row, alpha in enumerate(exponents):
            for variable, multiplicity in enumerate(alpha):
                if multiplicity == 0:
                    continue
                base = list(alpha)
                base[variable] -= 1
                for beta, coefficient in system.terms[variable].items():
                    result = tuple(base[i] + beta[i] for i in range(n))
                    if result in index:
                        matrix[row, index[result]] += multiplicity * coefficient
                    else:
                        dropped += 1

        labels = tuple(_label(exponent, system.variable_names) for exponent in exponents)
        return LinearizedSystem(
            matrix=matrix,
            initial_state=lift_state(system.initial_state, exponents),
            physical_dimension=n,
            exponents=exponents,
            labels=labels,
            metadata={
                **system.metadata,
                "linearization": "carleman",
                "order": self.order,
                "lifted_dimension": len(exponents),
                "dropped_term_count": dropped,
                "sparsity": float(1.0 - np.count_nonzero(matrix) / matrix.size),
            },
        )


def _label(exponent: Exponent, names: tuple[str, ...]) -> str:
    parts = []
    for name, power in zip(names, exponent):
        if power == 1:
            parts.append(name)
        elif power > 1:
            parts.append(f"{name}^{power}")
    return "*".join(parts)

