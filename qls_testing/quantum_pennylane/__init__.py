"""Working PennyLane circuits and LinearSolver-compatible wrappers."""

from .base import QuantumLinearSolver
from .primitives import (
    block_encoding_matrix,
    projected_block_encoding_action,
    sparse_block_encoding_data,
)
from .vqls import PennyLaneVQLSSolver
from .hhl import PennyLaneHHLSolver
from .qsvt import PennyLaneQSVTSolver

__all__ = [
    "PennyLaneHHLSolver", "PennyLaneQSVTSolver", "PennyLaneVQLSSolver",
    "QuantumLinearSolver", "block_encoding_matrix",
    "projected_block_encoding_action", "sparse_block_encoding_data",
]
