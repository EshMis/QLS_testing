"""Explicit registries for plugin-like method selection."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any


class Registry:
    def __init__(self, category: str) -> None:
        self.category = category
        self._factories: dict[str, Callable[..., Any]] = {}

    def register(self, name: str, factory: Callable[..., Any]) -> None:
        key = name.lower()
        if key in self._factories:
            raise ValueError(f"{self.category} plugin {name!r} is already registered")
        self._factories[key] = factory

    def create(self, name: str, **settings: Any) -> Any:
        try:
            factory = self._factories[name.lower()]
        except KeyError as exc:
            available = ", ".join(sorted(self._factories))
            raise ValueError(f"unknown {self.category} {name!r}; choose from: {available}") from exc
        return factory(**settings)

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(sorted(self._factories))


LINEARIZATIONS = Registry("linearization")
INTEGRATORS = Registry("integrator")
QLS_SOLVERS = Registry("QLS solver")

