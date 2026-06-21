"""User-facing metadata wrapper for lifted coordinates."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from qls_testing.core.datatypes import LinearizedSystem


@dataclass(frozen=True)
class LiftedSystemModel:
    """Manage names, degrees, and lookup for a finite lifted state."""

    system: LinearizedSystem

    @property
    def labels(self) -> tuple[str, ...]:
        return self.system.labels

    @property
    def degrees(self) -> tuple[int, ...]:
        return tuple(sum(exponent) for exponent in self.system.exponents)

    def indices_at_degree(self, degree: int) -> tuple[int, ...]:
        return tuple(index for index, value in enumerate(self.degrees) if value == degree)

    def index(self, label: str) -> int:
        try:
            return self.labels.index(label)
        except ValueError as exc:
            raise KeyError(f"unknown lifted variable {label!r}") from exc

    def trace(self, states: np.ndarray, label: str) -> np.ndarray:
        return np.asarray(states)[..., self.index(label)]

