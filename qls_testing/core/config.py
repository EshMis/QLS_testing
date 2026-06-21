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
    qssa_expansion: str = "moving_point"
    qssa_fallback: str = "moving_point"
    moving_point: tuple[float, ...] | None = None
    toy_coupling: float = 0.5
    lindblad_decay_rate: float = 1.0


@dataclass(frozen=True)
class MethodConfig:
    name: str
    settings: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class TimeConfig:
    t_final: float = 5.0
    dt: float = 0.1
    n_points: int = 101
    rtol: float = 1e-7
    atol: float = 1e-9
    min_step: float = 1e-6
    max_step: float | None = None
    output_stride: int = 1


@dataclass(frozen=True)
class ErrorConfig:
    enabled: bool = True
    compute_stage_proxies: bool = True


@dataclass(frozen=True)
class ComplexityConfig:
    enabled: bool = True


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
    error: ErrorConfig = field(default_factory=ErrorConfig)
    complexity: ComplexityConfig = field(default_factory=ComplexityConfig)
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
        if self.time.rtol <= 0 or self.time.atol <= 0 or self.time.min_step <= 0:
            raise ValueError("time tolerances and min_step must be positive")
        if self.time.max_step is not None and self.time.max_step < self.time.min_step:
            raise ValueError("max_step must be >= min_step")
        if self.time.output_stride < 1:
            raise ValueError("output_stride must be >= 1")
        if self.system.qssa_expansion not in {"zero", "moving_point", "auto"}:
            raise ValueError("qssa_expansion must be zero, moving_point, or auto")
        order = self.linearization.settings.get("order", 2)
        if not isinstance(order, int) or order < 1:
            raise ValueError("Carleman order must be a positive integer")
        return self


def load_config(path: str | Path, overrides: list[str] | None = None) -> Config:
    """Load and validate a YAML file, rejecting misspelled keys."""
    config_path = Path(path)
    with config_path.open(encoding="utf-8") as stream:
        raw = yaml.safe_load(stream) or {}
    for override in overrides or []:
        if "=" not in override:
            raise ValueError(f"override must use dotted.path=value syntax: {override!r}")
        dotted_path, raw_value = override.split("=", 1)
        keys = dotted_path.split(".")
        target = raw
        for key in keys[:-1]:
            child = target.setdefault(key, {})
            if not isinstance(child, dict):
                raise ValueError(f"override path {dotted_path!r} crosses a non-mapping value")
            target = child
        target[keys[-1]] = yaml.safe_load(raw_value)
    allowed = {
        "system", "linearization", "integrator", "qls", "time", "output",
        "error", "complexity", "random_seed",
    }
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
        error=_construct(ErrorConfig, raw.get("error")),
        complexity=_construct(ComplexityConfig, raw.get("complexity")),
        random_seed=raw.get("random_seed", 42),
    )
    return config.validate()
