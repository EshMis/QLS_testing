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
class PathwaySegmentConfig:
    substrate: str = "S"
    product: str = "P"
    intermediate_count: int = 3
    intermediate_prefix: str = "X"
    k1: float | tuple[float, ...] = (2.0, 1.5, 1.0, 2.0)
    k_minus_1: float | tuple[float, ...] = (1.0, 0.5, 0.5, 1.0)
    kcat: float | tuple[float, ...] = (3.0, 2.0, 2.5, 4.0)
    enzyme_total: float | tuple[float, ...] = (1.0, 1.0, 1.0, 1.0)


@dataclass(frozen=True)
class PathwayConfig:
    mode: str = "chain"
    segments: tuple[PathwaySegmentConfig, ...] = field(
        default_factory=lambda: (PathwaySegmentConfig(),)
    )


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
    pathway: PathwayConfig | None = None
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
        if self.pathway is not None:
            if self.pathway.mode not in {"chain", "chained_segments"}:
                raise ValueError("pathway.mode must be chain or chained_segments")
            if not self.pathway.segments:
                raise ValueError("pathway.segments must contain at least one segment")
            if self.pathway.mode == "chain" and len(self.pathway.segments) != 1:
                raise ValueError("pathway.mode chain accepts exactly one segment")
            for index, segment in enumerate(self.pathway.segments):
                if segment.intermediate_count < 0:
                    raise ValueError("segment intermediate_count must be nonnegative")
                if not segment.substrate or not segment.product:
                    raise ValueError("segment substrate and product names must be nonempty")
                if index and self.pathway.segments[index - 1].product != segment.substrate:
                    raise ValueError(
                        "chained pathway segments must connect previous product to next substrate"
                    )
        order = self.linearization.settings.get("order", 2)
        if not isinstance(order, int) or order < 1:
            raise ValueError("Carleman order must be a positive integer")
        return self


def _construct_pathway(values: dict[str, Any] | None) -> PathwayConfig | None:
    if values is None:
        return None
    allowed = {item.name for item in fields(PathwayConfig)}
    unknown = set(values) - allowed
    if unknown:
        raise ValueError(f"unknown PathwayConfig setting(s): {sorted(unknown)}")
    raw_segments = values.get("segments")
    if raw_segments is None:
        segments = (PathwaySegmentConfig(),)
    else:
        if not isinstance(raw_segments, list):
            raise ValueError("pathway.segments must be a list")
        segment_allowed = {item.name for item in fields(PathwaySegmentConfig)}
        built = []
        for item in raw_segments:
            if not isinstance(item, dict):
                raise ValueError("each pathway segment must be a mapping")
            segment_unknown = set(item) - segment_allowed
            if segment_unknown:
                raise ValueError(
                    f"unknown PathwaySegmentConfig setting(s): {sorted(segment_unknown)}"
                )
            built.append(PathwaySegmentConfig(**item))
        segments = tuple(built)
    return PathwayConfig(mode=values.get("mode", "chain"), segments=segments)


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
        "system", "pathway", "linearization", "integrator", "qls", "time", "output",
        "error", "complexity", "random_seed",
    }
    unknown = set(raw) - allowed
    if unknown:
        raise ValueError(f"unknown top-level config setting(s): {sorted(unknown)}")
    config = Config(
        system=_construct(SystemConfig, raw.get("system")),
        pathway=_construct_pathway(raw.get("pathway")),
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
