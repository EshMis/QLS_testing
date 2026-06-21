"""Working PennyLane circuits and LinearSolver-compatible wrappers."""

from .base import QuantumLinearSolver
from .primitives import (
    block_encoding_matrix,
    projected_block_encoding_action,
    sparse_block_encoding_data,
)
from .vqls import PennyLaneVQLSSolver

__all__ = [
    "PennyLaneVQLSSolver", "QuantumLinearSolver", "block_encoding_matrix",
    "projected_block_encoding_action", "sparse_block_encoding_data",
]
