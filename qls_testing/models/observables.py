"""Mappings from physical state orderings to dashboard observable groups."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ObservableGroup:
    key: str
    title: str
    labels: tuple[str, ...]
    description: str


def enzyme_observable_groups(labels: tuple[str, ...]) -> tuple[ObservableGroup, ...]:
    """Map the notebook state order to S, intermediates, product, and complexes."""
    available = set(labels)
    definitions = (
        ObservableGroup("S", "Substrate S", ("S",), "Input substrate concentration."),
        ObservableGroup(
            "Xs",
            "Intermediates Xs",
            tuple(label for label in ("X1", "X2", "X3") if label in available),
            "Sequential pathway intermediates.",
        ),
        ObservableGroup("P", "Product P", ("P",) if "P" in available else (), "Final product."),
        ObservableGroup(
            "Cs",
            "Enzyme complexes Cs",
            tuple(label for label in ("C1", "C2", "C3", "C4") if label in available),
            "Enzyme-substrate complexes; absent from the QSSA state.",
        ),
    )
    return definitions


def observable_groups(labels: tuple[str, ...]) -> tuple[ObservableGroup, ...]:
    """Use enzyme groups when available, otherwise expose practice states together."""
    if "S" in labels and "P" in labels:
        return enzyme_observable_groups(labels)
    return (
        ObservableGroup(
            "states",
            "Practice-system states",
            labels,
            "All non-lifted coordinates of the selected practice system.",
        ),
    )
