"""Hardware-oriented QSVT resource proxies for folded pathway systems."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from math import ceil, log, log2


@dataclass(frozen=True)
class QSVTResourceEstimate:
    alpha_bound: float
    condition_number: float | None
    target_precision: float
    inverse_polynomial_degree_proxy: int | None
    block_encoding_query_proxy: int | None
    notes: str

    def to_dict(self) -> dict[str, float | int | str | None]:
        return asdict(self)


@dataclass(frozen=True)
class StructuredBlockEncodingEstimate:
    """Register/oracle-level estimate for the folded Carleman pathway matrix."""

    folded_dimension: int
    lifted_dimension: int
    physical_dimension: int
    reaction_count: int | None
    time_steps: int
    time_qubits: int
    lifted_state_qubits: int
    degree_register_qubits: int
    symmetric_pair_register_qubits: int
    reaction_index_qubits: int | None
    top_level_lcu_selector_qubits: int
    sparse_row_degree_bound: int
    unique_coefficient_values: int
    component_oracles: tuple[str, ...]
    notes: str

    def to_dict(self) -> dict[str, int | str | tuple[str, ...] | None]:
        return asdict(self)


@dataclass(frozen=True)
class OracleGateProxyEstimate:
    """Coarse reversible-gate proxy for one structured block-encoding query."""

    per_query_toffoli_proxy: int
    per_query_depth_proxy: int
    qrom_entries: int
    qrom_address_qubits: int
    qrom_value_register_qubits: int
    arithmetic_workspace_qubits: int
    total_toffoli_proxy: int | None
    total_depth_proxy: int | None
    assumptions: tuple[str, ...]

    def to_dict(self) -> dict[str, int | tuple[str, ...] | None]:
        return asdict(self)


def estimate_structured_block_encoding(
    *,
    folded_dimension: int,
    lifted_dimension: int,
    physical_dimension: int,
    time_steps: int,
    top_level_lcu_terms: int,
    sparse_row_degree_bound: int,
    unique_coefficient_values: int,
    reaction_count: int | None = None,
) -> StructuredBlockEncodingEstimate:
    """Describe the scalable oracle structure without materializing a unitary."""
    if folded_dimension < 1 or lifted_dimension < 1 or physical_dimension < 1:
        raise ValueError("dimensions must be positive")
    if time_steps < 1 or top_level_lcu_terms < 1:
        raise ValueError("time steps and LCU terms must be positive")
    degree_two_dimension = physical_dimension * (physical_dimension + 1) // 2
    return StructuredBlockEncodingEstimate(
        folded_dimension=folded_dimension,
        lifted_dimension=lifted_dimension,
        physical_dimension=physical_dimension,
        reaction_count=reaction_count,
        time_steps=time_steps,
        time_qubits=ceil(log2(time_steps)),
        lifted_state_qubits=ceil(log2(lifted_dimension)),
        degree_register_qubits=1,
        symmetric_pair_register_qubits=ceil(log2(max(degree_two_dimension, 1))),
        reaction_index_qubits=(
            ceil(log2(reaction_count)) if reaction_count and reaction_count > 1 else 0
        ),
        top_level_lcu_selector_qubits=ceil(log2(top_level_lcu_terms)),
        sparse_row_degree_bound=sparse_row_degree_bound,
        unique_coefficient_values=unique_coefficient_values,
        component_oracles=(
            "time_shift_boundary_oracle",
            "degree_one_F1_sparse_oracle",
            "quadratic_F2_reaction_local_oracle",
            "symmetric_pair_rank_unrank_oracle",
            "induced_symmetric_square_reuse_oracle",
            "coefficient_lookup_or_arithmetic_oracle",
        ),
        notes=(
            "Estimate assumes sparse reversible access to reaction-local Carleman rows and "
            "Kronecker time shifts; it is not a compiled gate count."
        ),
    )


def estimate_oracle_gate_proxy(
    structured: StructuredBlockEncodingEstimate,
    *,
    qsvt_queries: int | None = None,
    coefficient_bits: int = 32,
) -> OracleGateProxyEstimate:
    """Estimate reversible arithmetic/qROM work for one block-encoding query.

    The constants are intentionally simple and conservative enough for trend
    comparison across pathway sizes. They are not architecture-specific
    synthesis results.
    """
    if coefficient_bits < 1:
        raise ValueError("coefficient_bits must be positive")
    time_arithmetic = 4 * max(structured.time_qubits, 1)
    pair_rank_unrank = 8 * max(structured.symmetric_pair_register_qubits, 1) ** 2
    reaction_decode = 6 * max(structured.reaction_index_qubits or 0, 1)
    sparse_emit = 3 * max(structured.sparse_row_degree_bound, 1) * (
        max(structured.lifted_state_qubits, 1) + coefficient_bits
    )
    coefficient_lookup = max(structured.unique_coefficient_values, 1) * coefficient_bits
    selector_controls = 4 * max(structured.top_level_lcu_selector_qubits, 1)
    per_query = (
        time_arithmetic
        + pair_rank_unrank
        + reaction_decode
        + sparse_emit
        + coefficient_lookup
        + selector_controls
    )
    depth = (
        time_arithmetic
        + pair_rank_unrank
        + reaction_decode
        + 3 * (max(structured.lifted_state_qubits, 1) + coefficient_bits)
        + coefficient_bits
        + selector_controls
    )
    address_qubits = max(
        structured.reaction_index_qubits or 0,
        ceil(log2(max(structured.unique_coefficient_values, 1))),
    )
    workspace = (
        structured.time_qubits
        + structured.lifted_state_qubits
        + structured.symmetric_pair_register_qubits
        + coefficient_bits
        + max(structured.sparse_row_degree_bound, 1)
    )
    return OracleGateProxyEstimate(
        per_query_toffoli_proxy=per_query,
        per_query_depth_proxy=depth,
        qrom_entries=structured.unique_coefficient_values,
        qrom_address_qubits=address_qubits,
        qrom_value_register_qubits=coefficient_bits,
        arithmetic_workspace_qubits=workspace,
        total_toffoli_proxy=None if qsvt_queries is None else per_query * qsvt_queries,
        total_depth_proxy=None if qsvt_queries is None else depth * qsvt_queries,
        assumptions=(
            "Proxy counts reversible arithmetic/comparison/qROM scale only.",
            "No routing, magic-state factory, fault-tolerant synthesis, or phase-synthesis cost is included.",
            "Coefficient lookup uses one table of distinct rounded Carleman coefficients.",
            "Pair rank/unrank cost is modeled quadratically in the symmetric-pair register size.",
        ),
    )


def estimate_qsvt_resources(
    *,
    alpha_bound: float,
    condition_number: float | None,
    target_precision: float = 1e-3,
    qls_calls: int = 1,
) -> QSVTResourceEstimate:
    """Estimate inverse-polynomial query cost for a normalized block encoding.

    This is intentionally a query-level proxy. It assumes the folded matrix has
    a block encoding with normalization ``alpha_bound`` and that QSVT applies a
    reciprocal polynomial on the normalized singular-value interval.
    """
    if alpha_bound <= 0.0:
        raise ValueError("alpha_bound must be positive")
    if not 0.0 < target_precision < 1.0:
        raise ValueError("target_precision must lie in (0, 1)")
    if condition_number is None or condition_number <= 0.0:
        return QSVTResourceEstimate(
            alpha_bound,
            condition_number,
            target_precision,
            None,
            None,
            "Condition number unavailable; QSVT inverse degree was not estimated.",
        )
    degree = ceil(alpha_bound * condition_number * log(1.0 / target_precision))
    return QSVTResourceEstimate(
        alpha_bound,
        condition_number,
        target_precision,
        degree,
        qls_calls * degree,
        (
            "Proxy uses O(alpha*kappa*log(1/epsilon)) block-encoding queries; "
            "state preparation, oracle synthesis, phase synthesis, and readout are separate."
        ),
    )
