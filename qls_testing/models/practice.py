"""Fast and dashboard-scale linear ODEs with deliberately varied matrices."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.linalg import expm

from qls_testing.core.datatypes import PolynomialSystem


@dataclass(frozen=True)
class PracticeSystem:
    name: str
    matrix: np.ndarray
    initial_state: np.ndarray
    labels: tuple[str, ...]
    speed: str
    properties: tuple[str, ...]

    def polynomial_system(self) -> PolynomialSystem:
        dimension = self.matrix.shape[0]
        equations = []
        for row in range(dimension):
            terms: dict[tuple[int, ...], complex | float] = {}
            for column, coefficient in enumerate(self.matrix[row]):
                if coefficient != 0:
                    exponent = tuple(int(index == column) for index in range(dimension))
                    terms[exponent] = coefficient.item()
            equations.append(terms)
        return PolynomialSystem(
            self.labels,
            tuple(equations),
            self.initial_state.copy(),
            {
                "model": self.name,
                "degree": 1,
                "matrix_structure": self.properties,
                "demo_speed": self.speed,
                "matrix_nnz": int(np.count_nonzero(self.matrix)),
                "matrix_density": float(np.count_nonzero(self.matrix) / self.matrix.size),
            },
        )

    def reference(self, times: np.ndarray) -> np.ndarray:
        return np.asarray([expm(time * self.matrix) @ self.initial_state for time in times])


def _sparse_chain(dimension: int) -> np.ndarray:
    matrix = -np.diag(np.linspace(0.5, 2.0, dimension))
    matrix += np.diag(np.full(dimension - 1, 0.35), 1)
    return matrix


def _dense_non_normal(dimension: int) -> np.ndarray:
    rng = np.random.default_rng(17 + dimension)
    upper = np.triu(rng.uniform(-0.3, 0.5, size=(dimension, dimension)), 1)
    return -np.diag(np.linspace(0.6, 1.8, dimension)) + upper


PRACTICE_SYSTEMS: dict[str, PracticeSystem] = {
    "practice_sparse_stable_3": PracticeSystem(
        "practice_sparse_stable_3",
        np.asarray([[-1.0, 0.4, 0.0], [0.0, -1.5, 0.25], [0.0, 0.0, -2.0]]),
        np.asarray([1.0, -0.25, 0.5]),
        ("u0", "u1", "u2"),
        "fast",
        ("sparse", "non-Hermitian", "stable"),
    ),
    "practice_dense_non_normal_4": PracticeSystem(
        "practice_dense_non_normal_4",
        _dense_non_normal(4),
        np.asarray([1.0, -0.5, 0.25, 0.75]),
        tuple(f"u{i}" for i in range(4)),
        "fast",
        ("dense", "non-Hermitian", "non-normal", "stable"),
    ),
    "practice_indefinite_2": PracticeSystem(
        "practice_indefinite_2",
        np.asarray([[0.35, 0.4], [0.0, -1.2]]),
        np.asarray([0.2, 1.0]),
        ("growing", "decaying"),
        "fast",
        ("non-Hermitian", "non-PSD Hermitian part", "one growing mode"),
    ),
    "practice_skew_oscillator_2": PracticeSystem(
        "practice_skew_oscillator_2",
        np.asarray([[-0.05, -2.0], [2.0, -0.05]]),
        np.asarray([1.0, 0.0]),
        ("position", "momentum"),
        "fast",
        ("dense", "non-PSD", "oscillatory", "normal"),
    ),
    "practice_complex_damped_2": PracticeSystem(
        "practice_complex_damped_2",
        np.asarray([[-0.4 + 1.2j, 0.25], [0.0, -0.9 - 0.7j]], dtype=complex),
        np.asarray([1.0, 0.3j], dtype=complex),
        ("amplitude_0", "amplitude_1"),
        "fast",
        ("complex", "sparse", "non-Hermitian", "stable"),
    ),
    "practice_sparse_chain_16": PracticeSystem(
        "practice_sparse_chain_16",
        _sparse_chain(16),
        np.linspace(1.0, 0.1, 16),
        tuple(f"u{i}" for i in range(16)),
        "dashboard",
        ("sparse", "non-Hermitian", "stable", "larger"),
    ),
    "practice_dense_non_normal_8": PracticeSystem(
        "practice_dense_non_normal_8",
        _dense_non_normal(8),
        np.linspace(1.0, -0.5, 8),
        tuple(f"u{i}" for i in range(8)),
        "dashboard",
        ("dense", "non-Hermitian", "non-normal", "larger"),
    ),
}


def get_practice_system(name: str) -> PracticeSystem:
    try:
        return PRACTICE_SYSTEMS[name]
    except KeyError as exc:
        raise ValueError(f"unknown practice system {name!r}") from exc


def lindblad_practice_system() -> PracticeSystem:
    """Four-coordinate enzyme-like chain small enough for mixed-state circuits."""
    return PracticeSystem(
        "lindblad_practice_pennylane",
        np.asarray(
            [
                [-1.0, 0.0, 0.0, 0.0],
                [0.7, -0.8, 0.0, 0.0],
                [0.0, 0.5, 0.0, 0.0],
                [0.3, 0.3, 0.0, -1.2],
            ]
        ),
        np.asarray([1.0, 0.0, 0.0, 0.0]),
        ("S", "X1", "P", "C1"),
        "fast",
        ("enzyme-like", "non-Hermitian", "stable", "P accumulating"),
    )
