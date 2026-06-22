#!/usr/bin/env python3
"""Report Carleman/folded/QSVT dimensions and structure for the pathway family."""

from __future__ import annotations

import argparse
import json
from math import ceil, log2
from pathlib import Path

import numpy as np
from scipy.sparse.linalg import expm_multiply, svds

from qls_testing.hardware_path.folded_systems import (
    FoldedSystem,
    build_folded_backward_euler,
    build_folded_bdf2,
    build_folded_crank_nicolson,
)
from qls_testing.hardware_path.readout import build_readout_plan
from qls_testing.hardware_path.structure import analyze_carleman_structure
from qls_testing.linearization.carleman import CarlemanLinearization
from qls_testing.systems.pathway_family import build_mass_action_chain


def folded_report(folded: FoldedSystem, estimate_condition: bool) -> dict[str, object]:
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
    report = {
        "method": folded.method,
        "matrix_dimension": folded.dimension,
        "matrix_nnz": int(folded.matrix.nnz),
        "matrix_density": float(folded.matrix.nnz / folded.dimension**2),
        "maximum_row_nnz": int(np.max(np.diff(folded.matrix.indptr))),
        "kronecker_terms": len(folded.kronecker_terms),
        "term_normalization_bounds": term_bounds,
        "lcu_alpha_bound": float(sum(item["normalization_bound"] for item in term_bounds)),
        "padded_dimension": padded,
        "history_state_qubits": int(log2(padded)),
        "hermitian_dilation_dimension": dilation,
        "qsvt_data_qubits_after_dilation": int(log2(dilation)),
        "selector_qubits_for_top_level_lcu": ceil(log2(len(folded.kronecker_terms))),
    }
    if estimate_condition:
        largest = float(
            svds(folded.matrix, k=1, which="LM", return_singular_vectors=False, tol=1e-7)[0]
        )
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
        report.update(
            {
                "largest_singular_value_estimate": largest,
                "smallest_singular_value_estimate": smallest,
                "condition_number_estimate": largest / smallest,
            }
        )
    return report


def build_report(
    reaction_min: int,
    reaction_max: int,
    dt: float,
    t_final: float,
    estimate_condition: bool = False,
) -> dict[str, object]:
    family = []
    for reactions in range(reaction_min, reaction_max + 1):
        lifted = CarlemanLinearization(2).linearize(build_mass_action_chain(reactions))
        family.append({"reactions": reactions, **analyze_carleman_structure(lifted).to_dict()})
    selected = CarlemanLinearization(2).linearize(build_mass_action_chain(4))
    steps = int(round(t_final / dt))
    builders = (
        build_folded_backward_euler,
        build_folded_crank_nicolson,
        build_folded_bdf2,
    )
    folded = [builder(selected.matrix, selected.initial_state, dt, steps) for builder in builders]
    history = expm_multiply(
        selected.matrix,
        selected.initial_state,
        start=dt,
        stop=t_final,
        num=steps,
        endpoint=True,
    )
    readout = build_readout_plan(
        lifted_dimension=selected.matrix.shape[0],
        physical_dimension=selected.physical_dimension,
        steps=steps,
        selected_time_count=1,
        history_states=history,
    )
    padded_history = np.vstack((history, np.repeat(history[-1][None, :], steps, axis=0)))
    padded_readout = build_readout_plan(
        lifted_dimension=selected.matrix.shape[0],
        physical_dimension=selected.physical_dimension,
        steps=2 * steps,
        selected_time_count=1,
        accepted_terminal_blocks=steps,
        history_states=padded_history,
    )
    return {
        "fixed_choices": {
            "model": "four-reaction mass action",
            "carleman_order": 2,
            "dt": dt,
            "t_final": t_final,
            "steps": steps,
        },
        "pathway_family": family,
        "folded_systems": [folded_report(item, estimate_condition) for item in folded],
        "final_time_readout": readout.to_dict(),
        "final_time_readout_with_K_terminal_copies": padded_readout.to_dict(),
        "readout_warning": (
            "Computational-basis samples give squared amplitudes only. Signed concentrations "
            "require interference and recovery of the QLS solution norm."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reaction-min", type=int, default=1)
    parser.add_argument("--reaction-max", type=int, default=6)
    parser.add_argument("--dt", type=float, default=0.1)
    parser.add_argument("--t-final", type=float, default=5.0)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--estimate-condition", action="store_true")
    args = parser.parse_args()
    report = build_report(
        args.reaction_min,
        args.reaction_max,
        args.dt,
        args.t_final,
        args.estimate_condition,
    )
    rendered = json.dumps(report, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
