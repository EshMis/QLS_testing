"""Strongly typed YAML experiment configuration."""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any, TypeVar

import yaml

T = TypeVar("T")


def _construct(cls: type[T], values: dict[str, Any] | None) -> T:
    values = values or {}
    allowed = {item.name for item in fields(cls)}
    unknown = set(values) - allowed
    if unknown:
        raise ValueError(f"unknown {cls.__name__} setting(s): {sorted(unknown)}")
    return cls(**values)


@dataclass(frozen=True)
class SystemConfig:
    name: str = "mass_action_pathway"
    initial_substrate: float = 3.0
    k1: tuple[float, ...] = (2.0, 1.5, 1.0, 2.0)
    k_minus_1: tuple[float, ...] = (1.0, 0.5, 0.5, 1.0)
    kcat: tuple[float, ...] = (3.0, 2.0, 2.5, 4.0)
    enzyme_total: tuple[float, ...] = (1.0, 1.0, 1.0, 1.0)
    taylor_degree: int = 3


@dataclass(frozen=True)
class MethodConfig:
    name: str
    settings: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class TimeConfig:
    t_final: float = 5.0
    dt: float = 0.1
    n_points: int = 101


@dataclass(frozen=True)
class OutputConfig:
    directory: str = "outputs"
    save_plot: bool = True
    show_plot: bool = False


@dataclass(frozen=True)
class Config:
    system: SystemConfig = field(default_factory=SystemConfig)
    linearization: MethodConfig = field(default_factory=lambda: MethodConfig("carleman", {"order": 2}))
    integrator: MethodConfig = field(default_factory=lambda: MethodConfig("bdf2"))
    qls: MethodConfig = field(default_factory=lambda: MethodConfig("classical"))
    time: TimeConfig = field(default_factory=TimeConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    random_seed: int = 42

    def validate(self) -> "Config":
        vectors = (self.system.k1, self.system.k_minus_1, self.system.kcat, self.system.enzyme_total)
        if any(len(v) != 4 for v in vectors) or any(value <= 0 for v in vectors for value in v):
            raise ValueError("all four kinetic parameter vectors must contain four positive values")
        if self.system.initial_substrate < 0:
            raise ValueError("initial_substrate must be nonnegative")
        if self.system.taylor_degree < 1:
            raise ValueError("taylor_degree must be >= 1")
        if self.time.t_final <= 0 or self.time.dt <= 0 or self.time.n_points < 2:
            raise ValueError("time values must be positive and n_points >= 2")
        order = self.linearization.settings.get("order", 2)
        if not isinstance(order, int) or order < 1:
            raise ValueError("Carleman order must be a positive integer")
        return self


def load_config(path: str | Path) -> Config:
    """Load and validate a YAML file, rejecting misspelled keys."""
    config_path = Path(path)
    with config_path.open(encoding="utf-8") as stream:
        raw = yaml.safe_load(stream) or {}
    allowed = {"system", "linearization", "integrator", "qls", "time", "output", "random_seed"}
    unknown = set(raw) - allowed
    if unknown:
        raise ValueError(f"unknown top-level config setting(s): {sorted(unknown)}")
    config = Config(
        system=_construct(SystemConfig, raw.get("system")),
        linearization=_construct(MethodConfig, raw.get("linearization") or {"name": "carleman", "settings": {"order": 2}}),
        integrator=_construct(MethodConfig, raw.get("integrator") or {"name": "bdf2"}),
        qls=_construct(MethodConfig, raw.get("qls") or {"name": "classical"}),
        time=_construct(TimeConfig, raw.get("time")),
        output=_construct(OutputConfig, raw.get("output")),
        random_seed=raw.get("random_seed", 42),
    )
    return config.validate()

