#!/usr/bin/env python3
"""Report Carleman/folded/QSVT scaling for variable pathway sizes."""

from __future__ import annotations

import argparse
import csv
import json
import time
from math import ceil, log2
from pathlib import Path
from typing import Any

import numpy as np
from scipy.sparse import diags
from scipy.sparse.linalg import expm_multiply, svds

from qls_testing.core.config import PathwaySegmentConfig
from qls_testing.core.datatypes import PolynomialSystem
from qls_testing.hardware_path.folded_systems import (
    FoldedSystem,
    build_folded_backward_euler,
    build_folded_bdf2,
    build_folded_crank_nicolson,
)
from qls_testing.hardware_path.readout import build_readout_plan
from qls_testing.hardware_path.resource_estimates import (
    estimate_oracle_gate_proxy,
    estimate_qsvt_resources,
    estimate_structured_block_encoding,
)
from qls_testing.hardware_path.structure import analyze_carleman_structure
from qls_testing.linearization.carleman import CarlemanLinearization
from qls_testing.reference import solve_polynomial_ground_truth
from qls_testing.systems.pathway_family import (
    build_chained_segments,
    build_mass_action_intermediate_chain,
)


def _condition_estimate(folded: FoldedSystem) -> dict[str, float]:
    largest = float(svds(folded.matrix, k=1, which="LM", return_singular_vectors=False, tol=1e-7)[0])
    smallest = float(
        svds(
            folded.matrix,
            k=1,
            which="SM",
            return_singular_vectors=False,
            tol=1e-7,
            maxiter=30000,
        )[0]
    )
    return {
        "largest_singular_value_estimate": largest,
        "smallest_singular_value_estimate": smallest,
        "condition_number_estimate": largest / smallest,
    }


def _equilibrated_condition_estimate(folded: FoldedSystem) -> dict[str, float | str]:
    """Estimate condition after sparse diagonal row/column norm scaling."""
    matrix = folded.matrix.tocsr()
    row_norms = np.sqrt(matrix.multiply(matrix).sum(axis=1)).A.ravel()
    column_norms = np.sqrt(matrix.multiply(matrix).sum(axis=0)).A.ravel()
    if np.any(row_norms <= 1e-15) or np.any(column_norms <= 1e-15):
        return {"equilibration_skipped": "zero row or column norm prevents diagonal scaling"}
    left = diags(1.0 / np.sqrt(row_norms), format="csr")
    right = diags(1.0 / np.sqrt(column_norms), format="csr")
    scaled = left @ matrix @ right
    largest = float(svds(scaled, k=1, which="LM", return_singular_vectors=False, tol=1e-7)[0])
    smallest = float(
        svds(
            scaled,
            k=1,
            which="SM",
            return_singular_vectors=False,
            tol=1e-7,
            maxiter=30000,
        )[0]
    )
    return {
        "equilibrated_largest_singular_value_estimate": largest,
        "equilibrated_smallest_singular_value_estimate": smallest,
        "equilibrated_condition_number_estimate": largest / smallest,
        "left_scaling_min": float(np.min(left.diagonal())),
        "left_scaling_max": float(np.max(left.diagonal())),
        "right_scaling_min": float(np.min(right.diagonal())),
        "right_scaling_max": float(np.max(right.diagonal())),
        "equilibration_notes": (
            "Sparse diagonal row/column norm scaling; hardware usefulness depends on "
            "coherent access to scaling factors and RHS/observable rescaling."
        ),
    }


def folded_report(
    folded: FoldedSystem,
    *,
    structure: dict[str, Any],
    reaction_count: int | None,
    physical_dimension: int,
    estimate_condition: bool,
    condition_max_dimension: int,
    estimate_equilibration: bool,
    qsvt_precision: float,
    terminal_padding_blocks: int,
    history_for_readout: np.ndarray | None,
) -> dict[str, Any]:
    padded = 1 << ceil(log2(folded.dimension))
    dilation = 2 * padded
    term_bounds = []
    for coefficient, time_operator, state_operator, label in folded.kronecker_terms:
        bound = (
            abs(coefficient)
            * np.linalg.norm(time_operator.toarray(), ord=2)
            * np.linalg.norm(state_operator.toarray(), ord=2)
        )
        term_bounds.append({"label": label, "normalization_bound": float(bound)})
    alpha = float(sum(item["normalization_bound"] for item in term_bounds))
    condition: float | None = None
    condition_fields: dict[str, float | str] = {}
    if estimate_condition and folded.dimension <= condition_max_dimension:
        condition_fields = _condition_estimate(folded)
        condition = float(condition_fields["condition_number_estimate"])
    elif estimate_condition:
        condition_fields = {
            "condition_estimate_skipped": (
                f"folded dimension {folded.dimension} exceeds cap {condition_max_dimension}"
            )
        }
    equilibration_fields: dict[str, Any] = {}
    equilibrated_condition: float | None = None
    if estimate_equilibration and folded.dimension <= condition_max_dimension:
        equilibration_fields = _equilibrated_condition_estimate(folded)
        if "equilibrated_condition_number_estimate" in equilibration_fields:
            equilibrated_condition = float(
                equilibration_fields["equilibrated_condition_number_estimate"]
            )
    elif estimate_equilibration:
        equilibration_fields = {
            "equilibration_skipped": (
                f"folded dimension {folded.dimension} exceeds cap {condition_max_dimension}"
            )
        }
    readout = build_readout_plan(
        lifted_dimension=folded.state_dimension,
        physical_dimension=physical_dimension,
        steps=folded.steps,
        selected_time_count=1,
        history_states=history_for_readout,
    )
    accepted = min(max(terminal_padding_blocks, 1), folded.steps)
    terminal_readout = build_readout_plan(
        lifted_dimension=folded.state_dimension,
        physical_dimension=physical_dimension,
        steps=folded.steps,
        selected_time_count=1,
        accepted_terminal_blocks=accepted,
        history_states=history_for_readout,
    )
    structured = estimate_structured_block_encoding(
        folded_dimension=folded.dimension,
        lifted_dimension=folded.state_dimension,
        physical_dimension=physical_dimension,
        time_steps=folded.steps,
        top_level_lcu_terms=len(folded.kronecker_terms),
        sparse_row_degree_bound=int(structure["maximum_row_nnz"]),
        unique_coefficient_values=int(structure["unique_nonzero_values"]),
        reaction_count=reaction_count,
    )
    qsvt_estimate = estimate_qsvt_resources(
        alpha_bound=alpha,
        condition_number=condition,
        target_precision=qsvt_precision,
    )
    equilibrated_qsvt_estimate = estimate_qsvt_resources(
        alpha_bound=alpha,
        condition_number=equilibrated_condition,
        target_precision=qsvt_precision,
    )
    report: dict[str, Any] = {
        "method": folded.method,
        "matrix_dimension": folded.dimension,
        "matrix_nnz": int(folded.matrix.nnz),
        "matrix_density": float(folded.matrix.nnz / folded.dimension**2),
        "maximum_row_nnz": int(np.max(np.diff(folded.matrix.indptr))),
        "kronecker_terms": len(folded.kronecker_terms),
        "term_normalization_bounds": term_bounds,
        "lcu_alpha_bound": alpha,
        "padded_dimension": padded,
        "history_state_qubits": int(log2(padded)),
        "hermitian_dilation_dimension": dilation,
        "qsvt_data_qubits_after_dilation": int(log2(dilation)),
        "selector_qubits_for_top_level_lcu": ceil(log2(len(folded.kronecker_terms))),
        "final_time_readout": readout.to_dict(),
        "terminal_padded_readout_proxy": terminal_readout.to_dict(),
        "structured_block_encoding_estimate": structured.to_dict(),
        "oracle_gate_proxy_estimate": estimate_oracle_gate_proxy(
            structured,
            qsvt_queries=qsvt_estimate.block_encoding_query_proxy,
        ).to_dict(),
        "equilibrated_oracle_gate_proxy_estimate": estimate_oracle_gate_proxy(
            structured,
            qsvt_queries=equilibrated_qsvt_estimate.block_encoding_query_proxy,
        ).to_dict(),
        "qsvt_resource_estimate": qsvt_estimate.to_dict(),
        "equilibrated_qsvt_resource_estimate": equilibrated_qsvt_estimate.to_dict(),
    }
    report.update(condition_fields)
    report.update(equilibration_fields)
    return report


def _history_if_small(
    system: PolynomialSystem,
    lifted_matrix: np.ndarray,
    lifted_initial: np.ndarray,
    *,
    dt: float,
    t_final: float,
    steps: int,
    max_folded_dimension: int,
) -> np.ndarray | None:
    if steps * lifted_matrix.shape[0] > max_folded_dimension:
        return None
    _ = system
    return expm_multiply(
        lifted_matrix,
        lifted_initial,
        start=dt,
        stop=t_final,
        num=steps,
        endpoint=True,
    )


def pathway_case_report(
    *,
    label: str,
    system: PolynomialSystem,
    dt: float,
    t_final: float,
    estimate_condition: bool,
    condition_max_dimension: int,
    estimate_equilibration: bool,
    qsvt_precision: float,
    terminal_padding_blocks: int,
    readout_history_max_folded_dimension: int,
) -> dict[str, Any]:
    lifted = CarlemanLinearization(2).linearize(system)
    structure = analyze_carleman_structure(lifted).to_dict()
    reaction_count = system.metadata.get("reactions")
    reaction_count = int(reaction_count) if reaction_count is not None else None
    steps = int(round(t_final / dt))
    history = _history_if_small(
        system,
        lifted.matrix,
        lifted.initial_state,
        dt=dt,
        t_final=t_final,
        steps=steps,
        max_folded_dimension=readout_history_max_folded_dimension,
    )
    builders = (
        build_folded_backward_euler,
        build_folded_crank_nicolson,
        build_folded_bdf2,
    )
    folded = [
        builder(lifted.matrix, lifted.initial_state, dt, steps)
        for builder in builders
    ]
    return {
        "label": label,
        "model_metadata": system.metadata,
        "variable_names": system.variable_names,
        "steps": steps,
        **structure,
        "folded_systems": [
            folded_report(
                item,
                structure=structure,
                reaction_count=reaction_count,
                physical_dimension=lifted.physical_dimension,
                estimate_condition=estimate_condition,
                condition_max_dimension=condition_max_dimension,
                estimate_equilibration=estimate_equilibration,
                qsvt_precision=qsvt_precision,
                terminal_padding_blocks=terminal_padding_blocks,
                history_for_readout=history,
            )
            for item in folded
        ],
    }


def _default_chained_cases(max_intermediates: int) -> list[tuple[int, ...]]:
    cases = [(0, 0), (1, 1), (2, 2), (3, 3), (1, 3), (3, 1), (5, 5)]
    return [case for case in cases if sum(case) <= max_intermediates * max(len(case), 1)]


def _parse_chained_case(raw: str) -> tuple[int, ...]:
    try:
        values = tuple(int(item.strip()) for item in raw.split(",") if item.strip())
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"invalid chained case {raw!r}; expected comma-separated nonnegative integers"
        ) from exc
    if len(values) < 2:
        raise argparse.ArgumentTypeError("a chained case must contain at least two segments")
    if any(value < 0 for value in values):
        raise argparse.ArgumentTypeError("chained-case intermediate counts must be nonnegative")
    return values


def _segments_from_intermediates(counts: tuple[int, ...]) -> tuple[PathwaySegmentConfig, ...]:
    segments = []
    substrate = "S"
    for index, count in enumerate(counts):
        product = "P" if index == len(counts) - 1 else f"B{index + 1}"
        segments.append(
            PathwaySegmentConfig(
                substrate=substrate,
                product=product,
                intermediate_count=count,
                intermediate_prefix=chr(ord("X") + index),
            )
        )
        substrate = product
    return tuple(segments)


def _chained_comparison(monolithic: dict[str, Any], sequential: list[dict[str, Any]]) -> dict[str, Any]:
    sequential_lifted = sum(int(item["lifted_dimension"]) for item in sequential)
    sequential_folded_be = sum(
        int(item["folded_systems"][0]["matrix_dimension"]) for item in sequential
    )
    sequential_nnz = sum(int(item["nnz"]) for item in sequential)
    mono_lifted = int(monolithic["lifted_dimension"])
    mono_folded_be = int(monolithic["folded_systems"][0]["matrix_dimension"])
    mono_nnz = int(monolithic["nnz"])
    return {
        "monolithic_lifted_dimension": mono_lifted,
        "sum_sequential_lifted_dimensions": sequential_lifted,
        "monolithic_to_sequential_lifted_ratio": mono_lifted / sequential_lifted,
        "monolithic_folded_be_dimension": mono_folded_be,
        "sum_sequential_folded_be_dimensions": sequential_folded_be,
        "monolithic_to_sequential_folded_be_ratio": mono_folded_be / sequential_folded_be,
        "monolithic_carleman_nnz": mono_nnz,
        "sum_sequential_carleman_nnz": sequential_nnz,
        "monolithic_to_sequential_nnz_ratio": mono_nnz / sequential_nnz,
        "interpretation": (
            "Ratio above 1 means the monolithic product-to-next-substrate solve is larger "
            "than solving isolated segments and handing off terminal products. It may still "
            "be useful when coherent correlations across the shared boundary are required."
        ),
    }


def _dynamic_handoff_comparison(
    *,
    segments: tuple[PathwaySegmentConfig, ...],
    monolithic: PolynomialSystem,
    initial_substrate: float,
    t_final: float,
) -> dict[str, Any]:
    """Compare exact monolithic dynamics with segment-by-segment product handoff."""
    times = np.asarray([0.0, t_final])
    substrate = float(initial_substrate)
    segment_results = []
    for index, segment in enumerate(segments):
        system = build_mass_action_intermediate_chain(
            segment.intermediate_count,
            initial_substrate=substrate,
            substrate=segment.substrate,
            product=segment.product,
            intermediate_prefix=segment.intermediate_prefix,
            k1=segment.k1,
            k_minus_1=segment.k_minus_1,
            kcat=segment.kcat,
            enzyme_total=segment.enzyme_total,
        )
        truth = solve_polynomial_ground_truth(system, times)
        product_index = segment.intermediate_count + 1
        product = float(np.real_if_close(truth.states[-1, product_index]))
        segment_results.append(
            {
                "segment": index,
                "substrate": segment.substrate,
                "product": segment.product,
                "intermediates": segment.intermediate_count,
                "initial_substrate_concentration": substrate,
                "terminal_product_concentration": product,
            }
        )
        substrate = product
    monolithic_truth = solve_polynomial_ground_truth(monolithic, times)
    final_product = segments[-1].product
    monolithic_index = monolithic.variable_names.index(final_product)
    monolithic_product = float(np.real_if_close(monolithic_truth.states[-1, monolithic_index]))
    return {
        "method": "exact_polynomial_reference_segment_handoff",
        "segment_time_horizon": t_final,
        "initial_substrate_concentration": initial_substrate,
        "sequential_terminal_product_concentration": substrate,
        "monolithic_terminal_product_concentration": monolithic_product,
        "absolute_terminal_product_difference": abs(substrate - monolithic_product),
        "relative_terminal_product_difference": (
            abs(substrate - monolithic_product) / abs(monolithic_product)
            if abs(monolithic_product) > 1e-15
            else None
        ),
        "segment_results": segment_results,
        "interpretation": (
            "Sequential handoff evolves each segment for the same time horizon and passes only "
            "the terminal product concentration forward. Monolithic dynamics evolve all coupled "
            "segment variables simultaneously, so equality is not expected unless time-scale "
            "separation justifies the handoff approximation."
        ),
    }


def build_report(
    intermediate_min: int,
    intermediate_max: int,
    dt: float,
    t_final: float,
    *,
    continue_until_cap: bool = False,
    max_wall_seconds: float = 300.0,
    max_lifted_dimension: int = 1200,
    max_folded_dimension: int = 100_000,
    estimate_condition: bool = False,
    condition_max_dimension: int = 5000,
    estimate_equilibration: bool = False,
    qsvt_precision: float = 1e-3,
    terminal_padding_blocks: int | None = None,
    readout_history_max_folded_dimension: int = 10_000,
    chained_cases: tuple[tuple[int, ...], ...] | None = None,
) -> dict[str, Any]:
    started = time.monotonic()
    steps = int(round(t_final / dt))
    family = []
    current = intermediate_min
    stop_reason = "completed requested range"
    while True:
        if current > intermediate_max and not continue_until_cap:
            break
        system = build_mass_action_intermediate_chain(current)
        lifted_dimension = (2 * current + 3) * (2 * current + 6) // 2
        folded_dimension = steps * lifted_dimension
        if lifted_dimension > max_lifted_dimension:
            stop_reason = f"lifted dimension cap hit at m={current}"
            break
        if folded_dimension > max_folded_dimension:
            stop_reason = f"folded dimension cap hit at m={current}"
            break
        if time.monotonic() - started > max_wall_seconds:
            stop_reason = f"wall-time cap hit before m={current}"
            break
        family.append(
            {
                "intermediates": current,
                "reactions": current + 1,
                **pathway_case_report(
                    label=f"single_segment_m={current}",
                    system=system,
                    dt=dt,
                    t_final=t_final,
                    estimate_condition=estimate_condition,
                    condition_max_dimension=condition_max_dimension,
                    estimate_equilibration=estimate_equilibration,
                    qsvt_precision=qsvt_precision,
                    terminal_padding_blocks=terminal_padding_blocks or steps,
                    readout_history_max_folded_dimension=readout_history_max_folded_dimension,
                ),
            }
        )
        current += 1
    chained = []
    selected_chained_cases = tuple(chained_cases or _default_chained_cases(intermediate_max))
    for case in selected_chained_cases:
        segments = _segments_from_intermediates(case)
        monolithic = build_chained_segments(segments)
        sequential = [
            pathway_case_report(
                label=f"sequential_segment_{index}_m={segment.intermediate_count}",
                system=build_mass_action_intermediate_chain(
                    segment.intermediate_count,
                    substrate=segment.substrate,
                    product=segment.product,
                    intermediate_prefix=segment.intermediate_prefix,
                ),
                dt=dt,
                t_final=t_final,
                estimate_condition=estimate_condition,
                condition_max_dimension=condition_max_dimension,
                estimate_equilibration=estimate_equilibration,
                qsvt_precision=qsvt_precision,
                terminal_padding_blocks=terminal_padding_blocks or steps,
                readout_history_max_folded_dimension=readout_history_max_folded_dimension,
            )
            for index, segment in enumerate(segments)
        ]
        monolithic_report = pathway_case_report(
            label=f"monolithic_segments_{'_'.join(map(str, case))}",
            system=monolithic,
            dt=dt,
            t_final=t_final,
            estimate_condition=estimate_condition,
            condition_max_dimension=condition_max_dimension,
            estimate_equilibration=estimate_equilibration,
            qsvt_precision=qsvt_precision,
            terminal_padding_blocks=terminal_padding_blocks or steps,
            readout_history_max_folded_dimension=readout_history_max_folded_dimension,
        )
        chained.append(
            {
                "intermediate_counts": case,
                "monolithic": monolithic_report,
                "sequential_segments": sequential,
                "monolithic_vs_sequential": _chained_comparison(monolithic_report, sequential),
                "dynamic_handoff_reference": _dynamic_handoff_comparison(
                    segments=segments,
                    monolithic=monolithic,
                    initial_substrate=3.0,
                    t_final=t_final,
                ),
                "comparison_note": (
                    "Sequential mode solves each segment separately and hands off the terminal product. "
                    "Monolithic mode keeps the product/substrate boundary in one larger sparse chain."
                ),
            }
        )
    return {
        "fixed_choices": {
            "model": "variable-intermediate mass-action pathway",
            "carleman_order": 2,
            "default_integrator": "folded_backward_euler",
            "qsvt_target": "native singular-value transformation over structured block encoding",
            "dt": dt,
            "t_final": t_final,
            "steps": steps,
            "qsvt_precision": qsvt_precision,
        },
        "caps": {
            "max_wall_seconds": max_wall_seconds,
            "max_lifted_dimension": max_lifted_dimension,
            "max_folded_dimension": max_folded_dimension,
            "condition_max_dimension": condition_max_dimension,
            "estimate_equilibration": estimate_equilibration,
            "readout_history_max_folded_dimension": readout_history_max_folded_dimension,
            "chained_cases": [list(case) for case in selected_chained_cases],
        },
        "stop_reason": stop_reason,
        "intermediate_sweep": family,
        "pathway_family": [
            {
                "intermediates": item["intermediates"],
                "reactions": item["reactions"],
                "physical_dimension": item["physical_dimension"],
                "lifted_dimension": item["lifted_dimension"],
                "nnz": item["nnz"],
                "density": item["density"],
                "maximum_row_nnz": item["maximum_row_nnz"],
            }
            for item in family
        ],
        "chained_pathway_sweep": chained,
        "readout_warning": (
            "Computational-basis samples give squared amplitudes only. Signed concentrations "
            "require interference and recovery of the QLS solution norm."
        ),
    }


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0])
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _intermediate_summary_rows(report: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for item in report["intermediate_sweep"]:
        folded_by_name = {folded["method"]: folded for folded in item["folded_systems"]}
        be = folded_by_name["folded_backward_euler"]
        rows.append(
            {
                "intermediates": item["intermediates"],
                "reactions": item["reactions"],
                "physical_dimension": item["physical_dimension"],
                "lifted_dimension": item["lifted_dimension"],
                "carleman_nnz": item["nnz"],
                "carleman_density": item["density"],
                "max_row_nnz": item["maximum_row_nnz"],
                "quadratic_coupling_rank": item["quadratic_coupling_rank"],
                "folded_be_dimension": be["matrix_dimension"],
                "folded_be_nnz": be["matrix_nnz"],
                "folded_be_alpha": be["lcu_alpha_bound"],
                "folded_be_condition": be.get("condition_number_estimate"),
                "folded_be_equilibrated_condition": be.get(
                    "equilibrated_condition_number_estimate"
                ),
                "folded_be_qsvt_degree_proxy": be["qsvt_resource_estimate"][
                    "inverse_polynomial_degree_proxy"
                ],
                "folded_be_equilibrated_qsvt_degree_proxy": be[
                    "equilibrated_qsvt_resource_estimate"
                ]["inverse_polynomial_degree_proxy"],
                "folded_be_oracle_per_query_toffoli_proxy": be[
                    "oracle_gate_proxy_estimate"
                ]["per_query_toffoli_proxy"],
                "folded_be_oracle_total_toffoli_proxy": be[
                    "oracle_gate_proxy_estimate"
                ]["total_toffoli_proxy"],
            }
        )
    return rows


def _chained_summary_rows(report: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for item in report["chained_pathway_sweep"]:
        comparison = item["monolithic_vs_sequential"]
        handoff = item["dynamic_handoff_reference"]
        rows.append(
            {
                "intermediate_counts": "-".join(str(value) for value in item["intermediate_counts"]),
                "segment_count": len(item["intermediate_counts"]),
                "monolithic_lifted_dimension": comparison["monolithic_lifted_dimension"],
                "sum_sequential_lifted_dimensions": comparison[
                    "sum_sequential_lifted_dimensions"
                ],
                "monolithic_to_sequential_lifted_ratio": comparison[
                    "monolithic_to_sequential_lifted_ratio"
                ],
                "monolithic_folded_be_dimension": comparison["monolithic_folded_be_dimension"],
                "sum_sequential_folded_be_dimensions": comparison[
                    "sum_sequential_folded_be_dimensions"
                ],
                "monolithic_to_sequential_folded_be_ratio": comparison[
                    "monolithic_to_sequential_folded_be_ratio"
                ],
                "monolithic_carleman_nnz": comparison["monolithic_carleman_nnz"],
                "sum_sequential_carleman_nnz": comparison["sum_sequential_carleman_nnz"],
                "monolithic_to_sequential_nnz_ratio": comparison[
                    "monolithic_to_sequential_nnz_ratio"
                ],
                "sequential_terminal_product": handoff[
                    "sequential_terminal_product_concentration"
                ],
                "monolithic_terminal_product": handoff[
                    "monolithic_terminal_product_concentration"
                ],
                "absolute_terminal_product_difference": handoff[
                    "absolute_terminal_product_difference"
                ],
                "relative_terminal_product_difference": handoff[
                    "relative_terminal_product_difference"
                ],
            }
        )
    return rows


def write_summary_csvs(report: dict[str, Any], base_path: Path) -> tuple[Path, Path]:
    """Write scan-friendly CSV tables derived from the full JSON report."""
    intermediate_path = base_path.with_name(f"{base_path.stem}_intermediates.csv")
    chained_path = base_path.with_name(f"{base_path.stem}_chained.csv")
    _write_csv(intermediate_path, _intermediate_summary_rows(report))
    _write_csv(chained_path, _chained_summary_rows(report))
    return intermediate_path, chained_path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--intermediate-min", type=int, default=0)
    parser.add_argument("--intermediate-max", type=int, default=10)
    parser.add_argument("--reaction-min", type=int, help="legacy alias: intermediate-min = reaction-min - 1")
    parser.add_argument("--reaction-max", type=int, help="legacy alias: intermediate-max = reaction-max - 1")
    parser.add_argument("--continue-until-cap", action="store_true")
    parser.add_argument("--max-wall-seconds", type=float, default=300.0)
    parser.add_argument("--max-lifted-dimension", type=int, default=1200)
    parser.add_argument("--max-folded-dimension", type=int, default=100_000)
    parser.add_argument("--condition-max-dimension", type=int, default=5000)
    parser.add_argument("--readout-history-max-folded-dimension", type=int, default=10_000)
    parser.add_argument("--dt", type=float, default=0.1)
    parser.add_argument("--t-final", type=float, default=5.0)
    parser.add_argument("--qsvt-precision", type=float, default=1e-3)
    parser.add_argument("--terminal-padding-blocks", type=int)
    parser.add_argument(
        "--chained-case",
        action="append",
        type=_parse_chained_case,
        help=(
            "comma-separated intermediate counts for one product-to-next-substrate chain; "
            "repeat for multiple cases, e.g. --chained-case 1,3 --chained-case 4,1,2"
        ),
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--summary-csv",
        type=Path,
        help=(
            "write two CSV summaries using this base path; suffixes _intermediates.csv "
            "and _chained.csv are added"
        ),
    )
    parser.add_argument("--estimate-condition", action="store_true")
    parser.add_argument("--estimate-equilibration", action="store_true")
    args = parser.parse_args()
    intermediate_min = (
        max(args.reaction_min - 1, 0)
        if args.reaction_min is not None
        else args.intermediate_min
    )
    intermediate_max = (
        max(args.reaction_max - 1, 0)
        if args.reaction_max is not None
        else args.intermediate_max
    )
    report = build_report(
        intermediate_min,
        intermediate_max,
        args.dt,
        args.t_final,
        continue_until_cap=args.continue_until_cap,
        max_wall_seconds=args.max_wall_seconds,
        max_lifted_dimension=args.max_lifted_dimension,
        max_folded_dimension=args.max_folded_dimension,
        estimate_condition=args.estimate_condition,
        condition_max_dimension=args.condition_max_dimension,
        estimate_equilibration=args.estimate_equilibration,
        qsvt_precision=args.qsvt_precision,
        terminal_padding_blocks=args.terminal_padding_blocks,
        readout_history_max_folded_dimension=args.readout_history_max_folded_dimension,
        chained_cases=tuple(args.chained_case) if args.chained_case else None,
    )
    rendered = json.dumps(report, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    if args.summary_csv:
        write_summary_csvs(report, args.summary_csv)
    print(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
