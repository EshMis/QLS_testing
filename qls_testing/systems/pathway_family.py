"""Variable-length mass-action chains used for structural scaling experiments."""

from __future__ import annotations

import numpy as np

from qls_testing.core.config import PathwayConfig, PathwaySegmentConfig
from qls_testing.core.datatypes import PolynomialSystem


def _expand_rates(
    values: float | tuple[float, ...],
    reactions: int,
    default: tuple[float, ...],
) -> np.ndarray:
    if isinstance(values, int | float):
        source = (float(values),) * reactions
    else:
        source = tuple(values)
    if len(source) != reactions:
        if len(source) == len(default):
            source = tuple(source[index % len(source)] for index in range(reactions))
        else:
            raise ValueError("each supplied rate vector must match the number of reactions")
    array = np.asarray(source, dtype=float)
    if np.any(array <= 0.0):
        raise ValueError("rates must be positive")
    return array


def build_mass_action_chain(
    reactions: int,
    *,
    initial_substrate: float = 3.0,
    k1: tuple[float, ...] | None = None,
    k_minus_1: tuple[float, ...] | None = None,
    kcat: tuple[float, ...] | None = None,
    enzyme_total: tuple[float, ...] | None = None,
) -> PolynomialSystem:
    """Build ``S -> X1 -> ... -> P`` with one complex per reaction."""
    if reactions < 1:
        raise ValueError("reactions must be positive")
    binding_rates = _expand_rates(k1 or (2.0, 1.5, 1.0, 2.0), reactions, (2.0, 1.5, 1.0, 2.0))
    unbinding_rates = _expand_rates(
        k_minus_1 or (1.0, 0.5, 0.5, 1.0), reactions, (1.0, 0.5, 0.5, 1.0)
    )
    catalytic_rates = _expand_rates(kcat or (3.0, 2.0, 2.5, 4.0), reactions, (3.0, 2.0, 2.5, 4.0))
    enzyme_totals = _expand_rates(enzyme_total or (1.0, 1.0, 1.0, 1.0), reactions, (1.0, 1.0, 1.0, 1.0))
    labels = (
        "S",
        *(f"X{index}" for index in range(1, reactions)),
        "P",
        *(f"C{index}" for index in range(1, reactions + 1)),
    )
    return _build_mass_action_from_rates(
        labels=labels,
        binding_rates=binding_rates,
        unbinding_rates=unbinding_rates,
        catalytic_rates=catalytic_rates,
        enzyme_totals=enzyme_totals,
        initial_substrate=initial_substrate,
        metadata={"model": "mass_action_chain", "reactions": reactions, "degree": 2},
    )


def build_mass_action_intermediate_chain(
    intermediates: int,
    *,
    initial_substrate: float = 3.0,
    substrate: str = "S",
    product: str = "P",
    intermediate_prefix: str = "X",
    k1: float | tuple[float, ...] = (2.0, 1.5, 1.0, 2.0),
    k_minus_1: float | tuple[float, ...] = (1.0, 0.5, 0.5, 1.0),
    kcat: float | tuple[float, ...] = (3.0, 2.0, 2.5, 4.0),
    enzyme_total: float | tuple[float, ...] = (1.0, 1.0, 1.0, 1.0),
) -> PolynomialSystem:
    """Build one segment with ``m`` intermediates and ``m+1`` reactions."""
    if intermediates < 0:
        raise ValueError("intermediates must be nonnegative")
    reactions = intermediates + 1
    metabolite_labels = (
        substrate,
        *(f"{intermediate_prefix}{index}" for index in range(1, intermediates + 1)),
        product,
    )
    labels = (*metabolite_labels, *(f"C{index}" for index in range(1, reactions + 1)))
    return _build_mass_action_from_rates(
        labels=labels,
        binding_rates=_expand_rates(k1, reactions, (2.0, 1.5, 1.0, 2.0)),
        unbinding_rates=_expand_rates(k_minus_1, reactions, (1.0, 0.5, 0.5, 1.0)),
        catalytic_rates=_expand_rates(kcat, reactions, (3.0, 2.0, 2.5, 4.0)),
        enzyme_totals=_expand_rates(enzyme_total, reactions, (1.0, 1.0, 1.0, 1.0)),
        initial_substrate=initial_substrate,
        metadata={
            "model": "mass_action_intermediate_chain",
            "intermediates": intermediates,
            "reactions": reactions,
            "degree": 2,
        },
    )


def build_pathway_from_config(
    pathway: PathwayConfig,
    *,
    initial_substrate: float = 3.0,
) -> PolynomialSystem:
    """Build a configured chain or product-to-next-substrate pathway."""
    if pathway.mode == "chain":
        segment = pathway.segments[0]
        return build_mass_action_intermediate_chain(
            segment.intermediate_count,
            initial_substrate=initial_substrate,
            substrate=segment.substrate,
            product=segment.product,
            intermediate_prefix=segment.intermediate_prefix,
            k1=segment.k1,
            k_minus_1=segment.k_minus_1,
            kcat=segment.kcat,
            enzyme_total=segment.enzyme_total,
        )
    if pathway.mode != "chained_segments":
        raise ValueError("pathway.mode must be chain or chained_segments")
    return build_chained_segments(pathway.segments, initial_substrate=initial_substrate)


def build_chained_segments(
    segments: tuple[PathwaySegmentConfig, ...],
    *,
    initial_substrate: float = 3.0,
) -> PolynomialSystem:
    """Build one monolithic chain while preserving segment boundaries in metadata."""
    if not segments:
        raise ValueError("segments must be nonempty")
    metabolite_labels: list[str] = [segments[0].substrate]
    binding: list[float] = []
    unbinding: list[float] = []
    catalytic: list[float] = []
    enzyme: list[float] = []
    boundaries = []
    reaction_offset = 0
    for index, segment in enumerate(segments):
        if index and segments[index - 1].product != segment.substrate:
            raise ValueError("chained segments must share product/substrate boundary labels")
        reactions = segment.intermediate_count + 1
        start = reaction_offset
        reaction_offset += reactions
        boundaries.append(
            {
                "segment": index,
                "substrate": segment.substrate,
                "product": segment.product,
                "intermediates": segment.intermediate_count,
                "reaction_start": start,
                "reaction_stop": reaction_offset,
            }
        )
        metabolite_labels.extend(
            f"{segment.intermediate_prefix}{item}"
            for item in range(1, segment.intermediate_count + 1)
        )
        metabolite_labels.append(segment.product)
        binding.extend(_expand_rates(segment.k1, reactions, (2.0, 1.5, 1.0, 2.0)))
        unbinding.extend(_expand_rates(segment.k_minus_1, reactions, (1.0, 0.5, 0.5, 1.0)))
        catalytic.extend(_expand_rates(segment.kcat, reactions, (3.0, 2.0, 2.5, 4.0)))
        enzyme.extend(_expand_rates(segment.enzyme_total, reactions, (1.0, 1.0, 1.0, 1.0)))
    reactions = len(binding)
    labels = (*metabolite_labels, *(f"C{index}" for index in range(1, reactions + 1)))
    return _build_mass_action_from_rates(
        labels=labels,
        binding_rates=np.asarray(binding),
        unbinding_rates=np.asarray(unbinding),
        catalytic_rates=np.asarray(catalytic),
        enzyme_totals=np.asarray(enzyme),
        initial_substrate=initial_substrate,
        metadata={
            "model": "mass_action_chained_segments",
            "segments": boundaries,
            "segment_count": len(segments),
            "reactions": reactions,
            "degree": 2,
        },
    )


def _build_mass_action_from_rates(
    *,
    labels: tuple[str, ...],
    binding_rates: np.ndarray,
    unbinding_rates: np.ndarray,
    catalytic_rates: np.ndarray,
    enzyme_totals: np.ndarray,
    initial_substrate: float,
    metadata: dict[str, object],
) -> PolynomialSystem:
    reactions = len(binding_rates)
    dimension = 2 * reactions + 1
    if len(labels) != dimension:
        raise ValueError("labels must contain all metabolites followed by complexes")
    equations: list[dict[tuple[int, ...], float]] = [dict() for _ in range(dimension)]

    def unit(index: int) -> tuple[int, ...]:
        exponent = [0] * dimension
        exponent[index] = 1
        return tuple(exponent)

    def product(first: int, second: int) -> tuple[int, ...]:
        exponent = [0] * dimension
        exponent[first] += 1
        exponent[second] += 1
        return tuple(exponent)

    def add(row: int, exponent: tuple[int, ...], value: float) -> None:
        equations[row][exponent] = equations[row].get(exponent, 0.0) + value

    complex_offset = reactions + 1
    for reaction in range(reactions):
        substrate = reaction
        complex_index = complex_offset + reaction
        binding = product(substrate, complex_index)
        k_on = binding_rates[reaction]
        k_off = unbinding_rates[reaction]
        k_cat = catalytic_rates[reaction]
        enzyme = enzyme_totals[reaction]
        add(substrate, unit(substrate), -k_on * enzyme)
        add(substrate, binding, k_on)
        add(substrate, unit(complex_index), k_off)
        if reaction:
            add(substrate, unit(complex_index - 1), catalytic_rates[reaction - 1])
        add(complex_index, unit(substrate), k_on * enzyme)
        add(complex_index, binding, -k_on)
        add(complex_index, unit(complex_index), -(k_off + k_cat))
    add(reactions, unit(complex_offset + reactions - 1), catalytic_rates[-1])
    initial = np.zeros(dimension)
    initial[0] = initial_substrate
    return PolynomialSystem(
        labels,
        tuple(equations),
        initial,
        metadata,
    )
