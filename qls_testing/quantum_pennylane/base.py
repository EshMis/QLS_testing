"""Marker interface for circuit-backed linear solvers."""

from qls_testing.core.interfaces import LinearSolver


class QuantumLinearSolver(LinearSolver):
    """A linear solver whose approximation invokes a quantum circuit backend."""

