"""Structural diagnostics for order-2 Carleman matrices and sparse encoders."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from qls_testing.core.datatypes import LinearizedSystem


@dataclass(frozen=True)
class CarlemanStructureReport:
    physical_dimension: int
    lifted_dimension: int
    degree_one_dimension: int
    degree_two_dimension: int
    nnz: int
    density: float
    maximum_row_nnz: int
    maximum_column_nnz: int
    unique_nonzero_values: int
    linear_block_nnz: int
    quadratic_coupling_nnz: int
    induced_degree_two_nnz: int
    lower_degree_block_nnz: int
    symmetric_kronecker_residual: float
    quadratic_coupling_rank: int
    sparse_access_normalization_bound: float

    def to_dict(self) -> dict[str, int | float]:
        return asdict(self)


def _induced_symmetric_square(linear: np.ndarray, exponents: tuple[tuple[int, ...], ...]) -> np.ndarray:
    dimension = linear.shape[0]
    pairs = []
    for exponent in exponents:
        indices = []
        for index, power in enumerate(exponent):
            indices.extend([index] * power)
        if len(indices) != 2:
            raise ValueError("degree-two exponent expected")
        pairs.append(tuple(indices))
    pair_index = {pair: index for index, pair in enumerate(pairs)}
    induced = np.zeros((len(pairs), len(pairs)), dtype=linear.dtype)
    for row, (first, second) in enumerate(pairs):
        for column in range(dimension):
            first_pair = tuple(sorted((column, second)))
            second_pair = tuple(sorted((first, column)))
            induced[row, pair_index[first_pair]] += linear[first, column]
            induced[row, pair_index[second_pair]] += linear[second, column]
    return induced


def analyze_carleman_structure(system: LinearizedSystem) -> CarlemanStructureReport:
    """Expose the exact degree block pattern relevant to an order-2 encoder."""
    if int(system.metadata.get("order", 0)) != 2:
        raise ValueError("hardware structure analysis currently requires Carleman order 2")
    matrix = np.asarray(system.matrix)
    physical = system.physical_dimension
    degree_two = matrix.shape[0] - physical
    linear = matrix[:physical, :physical]
    quadratic = matrix[:physical, physical:]
    lower = matrix[physical:, :physical]
    induced = matrix[physical:, physical:]
    expected = _induced_symmetric_square(linear, system.exponents[physical:])
    row_nnz = np.count_nonzero(matrix, axis=1)
    column_nnz = np.count_nonzero(matrix, axis=0)
    nonzero_values = matrix[np.nonzero(matrix)]
    maximum_entry = float(np.max(np.abs(nonzero_values), initial=0.0))
    sparse_degree = max(int(np.max(row_nnz, initial=0)), int(np.max(column_nnz, initial=0)))
    return CarlemanStructureReport(
        physical,
        matrix.shape[0],
        physical,
        degree_two,
        int(np.count_nonzero(matrix)),
        float(np.count_nonzero(matrix) / matrix.size),
        int(np.max(row_nnz, initial=0)),
        int(np.max(column_nnz, initial=0)),
        len(np.unique(np.round(nonzero_values, 14))),
        int(np.count_nonzero(linear)),
        int(np.count_nonzero(quadratic)),
        int(np.count_nonzero(induced)),
        int(np.count_nonzero(lower)),
        float(np.linalg.norm(induced - expected)),
        int(np.linalg.matrix_rank(quadratic)),
        sparse_degree * maximum_entry,
    )
