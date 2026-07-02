import importlib.util
from pathlib import Path

import numpy as np
import pytest

from qls_testing.core.config import (
    Config,
    MethodConfig,
    OutputConfig,
    PathwayConfig,
    PathwaySegmentConfig,
    SystemConfig,
    TimeConfig,
    load_config,
)
from qls_testing.experiments.run_experiment import run_experiment
from qls_testing.hardware_path.folded_systems import (
    append_terminal_copies,
    build_folded_backward_euler,
    build_folded_bdf2,
    build_folded_crank_nicolson,
)
from qls_testing.hardware_path.block_encoding import (
    build_lcu_block_encoding,
    folded_lcu_matrices,
    projected_lcu_action,
)
from qls_testing.hardware_path.structure import analyze_carleman_structure
from qls_testing.hardware_path.readout import build_readout_plan, history_coordinate_index
from qls_testing.linearization.carleman import CarlemanLinearization
from qls_testing.systems.pathway_family import (
    build_chained_segments,
    build_mass_action_chain,
    build_mass_action_intermediate_chain,
    build_pathway_from_config,
)
from qls_testing.systems.metabolic import build_mass_action_pathway


_SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "analyze_hardware_path.py"
_SCRIPT_SPEC = importlib.util.spec_from_file_location("analyze_hardware_path", _SCRIPT_PATH)
assert _SCRIPT_SPEC is not None and _SCRIPT_SPEC.loader is not None
_ANALYZE_HARDWARE_PATH = importlib.util.module_from_spec(_SCRIPT_SPEC)
_SCRIPT_SPEC.loader.exec_module(_ANALYZE_HARDWARE_PATH)
build_report = _ANALYZE_HARDWARE_PATH.build_report
write_summary_csvs = _ANALYZE_HARDWARE_PATH.write_summary_csvs


def test_folded_kronecker_terms_reconstruct_each_matrix_exactly():
    matrix = np.asarray([[-1.0, 0.2], [0.0, -0.5]])
    initial = np.asarray([1.0, 0.0])
    for builder in (
        build_folded_backward_euler,
        build_folded_crank_nicolson,
        build_folded_bdf2,
    ):
        folded = builder(matrix, initial, 0.1, 4)
        difference = folded.matrix - folded.reconstruct_from_terms()
        assert difference.nnz == 0 or np.max(np.abs(difference.data)) < 1e-14
        assert folded.matrix.shape == (8, 8)


def test_explicit_lcu_unitary_block_encodes_folded_matrix():
    folded = build_folded_backward_euler(
        np.asarray([[-1.0, 0.2], [0.0, -0.5]]), np.asarray([1.0, 0.0]), 0.1, 2
    )
    matrices, labels = folded_lcu_matrices(folded)
    encoding = build_lcu_block_encoding(matrices, labels=labels)
    identity = np.eye(encoding.unitary.shape[0])
    np.testing.assert_allclose(encoding.unitary.conj().T @ encoding.unitary, identity, atol=2e-10)
    np.testing.assert_allclose(
        encoding.encoded_block(), encoding.matrix / encoding.alpha, atol=2e-10
    )
    state = np.asarray([1.0, -0.2, 0.3, 0.1])
    action, probability = projected_lcu_action(encoding, state)
    np.testing.assert_allclose(action[:4], encoding.matrix[:4, :4] @ (state / np.linalg.norm(state)), atol=2e-10)
    assert 0.0 < probability <= 1.0


def test_variable_length_four_reaction_chain_matches_primary_model():
    family = build_mass_action_chain(4)
    primary = build_mass_action_pathway(SystemConfig(name="mass_action_pathway"))
    family_lift = CarlemanLinearization(2).linearize(family)
    primary_lift = CarlemanLinearization(2).linearize(primary)
    assert family.variable_names == primary.variable_names
    np.testing.assert_allclose(family_lift.matrix, primary_lift.matrix)
    np.testing.assert_allclose(family_lift.initial_state, primary_lift.initial_state)


def test_three_intermediates_reproduce_primary_four_reaction_model():
    family = build_mass_action_intermediate_chain(3)
    primary = build_mass_action_pathway(SystemConfig(name="mass_action_pathway"))
    family_lift = CarlemanLinearization(2).linearize(family)
    primary_lift = CarlemanLinearization(2).linearize(primary)
    assert family.variable_names == primary.variable_names
    np.testing.assert_allclose(family_lift.matrix, primary_lift.matrix)
    np.testing.assert_allclose(family_lift.initial_state, primary_lift.initial_state)


def test_intermediate_count_dimensions_match_order_two_formula():
    for intermediates in range(6):
        system = build_mass_action_intermediate_chain(intermediates)
        lifted = CarlemanLinearization(2).linearize(system)
        physical = 2 * intermediates + 3
        expected_lifted = physical * (physical + 3) // 2
        assert lifted.physical_dimension == physical
        assert lifted.matrix.shape == (expected_lifted, expected_lifted)


def test_pathway_yaml_config_parses_and_rejects_disconnected_segments(tmp_path):
    config_path = tmp_path / "pathway.yaml"
    config_path.write_text(
        """
system: {name: pathway_graph, initial_substrate: 3.0}
pathway:
  mode: chained_segments
  segments:
    - {substrate: S, product: B1, intermediate_count: 1, intermediate_prefix: X}
    - {substrate: B1, product: P, intermediate_count: 2, intermediate_prefix: Y}
linearization: {name: carleman, settings: {order: 2}}
integrator: {name: folded_backward_euler}
qls: {name: classical}
time: {t_final: 0.1, dt: 0.1, n_points: 2}
output: {save_plot: false}
error: {enabled: false}
complexity: {enabled: false}
""",
        encoding="utf-8",
    )
    config = load_config(config_path)
    assert config.pathway is not None
    assert config.pathway.mode == "chained_segments"
    assert [item.intermediate_count for item in config.pathway.segments] == [1, 2]

    bad = Config(
        system=SystemConfig(name="pathway_graph"),
        pathway=PathwayConfig(
            mode="chained_segments",
            segments=(
                PathwaySegmentConfig(substrate="S", product="B1"),
                PathwaySegmentConfig(substrate="not_B1", product="P"),
            ),
        ),
    )
    with pytest.raises(ValueError, match="connect"):
        bad.validate()


def test_chained_segments_build_monolithic_product_to_next_substrate_pathway():
    pathway = PathwayConfig(
        mode="chained_segments",
        segments=(
            PathwaySegmentConfig(substrate="S", product="B1", intermediate_count=1),
            PathwaySegmentConfig(substrate="B1", product="P", intermediate_count=3, intermediate_prefix="Y"),
        ),
    )
    system = build_pathway_from_config(pathway)
    direct = build_chained_segments(pathway.segments)
    assert system.variable_names == direct.variable_names
    assert system.metadata["segment_count"] == 2
    assert system.metadata["reactions"] == 6
    assert system.variable_names[:7] == ("S", "X1", "B1", "Y1", "Y2", "Y3", "P")


def test_order_two_structure_is_block_upper_triangular_symmetric_kronecker_lift():
    for reactions in (1, 2, 4, 6):
        lifted = CarlemanLinearization(2).linearize(build_mass_action_chain(reactions))
        report = analyze_carleman_structure(lifted)
        assert report.lifted_dimension == report.physical_dimension * (
            report.physical_dimension + 3
        ) // 2
        assert report.lower_degree_block_nnz == 0
        assert report.symmetric_kronecker_residual < 1e-12
        assert report.maximum_row_nnz <= 8


def test_readout_plan_targets_physical_amplitudes_not_full_lifted_history():
    history = np.ones((50, 54))
    plan = build_readout_plan(
        lifted_dimension=54,
        physical_dimension=9,
        steps=50,
        selected_time_count=1,
        history_states=history,
    )
    assert plan.full_history_coordinates == 2700
    assert plan.targeted_real_amplitudes == 9
    assert plan.history_state_qubits == 12
    assert np.isclose(plan.final_time_postselection_probability, 1 / 50)
    assert np.isclose(plan.amplitude_amplification_query_factor, np.sqrt(50))
    assert history_coordinate_index(49, 8, 54) == 2654
    assert history_coordinate_index(49, 53, 54) == 2699


def test_terminal_padding_repeats_final_state_without_extra_evolution():
    folded = build_folded_backward_euler(
        np.asarray([[-1.0, 0.2], [0.0, -0.5]]), np.asarray([1.0, 0.0]), 0.1, 3
    )
    padded = append_terminal_copies(folded, 4)
    solution = np.linalg.solve(padded.matrix.toarray(), padded.rhs).reshape(7, 2)
    np.testing.assert_allclose(solution[3:], np.repeat(solution[2][None, :], 4, axis=0))
    assert padded.dimension == 14


def test_small_intermediate_sweep_reports_qsvt_and_chained_fields():
    report = build_report(
        0,
        2,
        0.1,
        0.2,
        max_wall_seconds=30.0,
        max_folded_dimension=5000,
        estimate_condition=False,
    )
    assert report["stop_reason"] == "completed requested range"
    assert [item["intermediates"] for item in report["intermediate_sweep"]] == [0, 1, 2]
    first_folded = report["intermediate_sweep"][0]["folded_systems"][0]
    assert first_folded["method"] == "folded_backward_euler"
    assert "qsvt_resource_estimate" in first_folded
    block_estimate = first_folded["structured_block_encoding_estimate"]
    assert block_estimate["component_oracles"] == (
        "time_shift_boundary_oracle",
        "degree_one_F1_sparse_oracle",
        "quadratic_F2_reaction_local_oracle",
        "symmetric_pair_rank_unrank_oracle",
        "induced_symmetric_square_reuse_oracle",
        "coefficient_lookup_or_arithmetic_oracle",
    )
    assert block_estimate["degree_register_qubits"] == 1
    gate_proxy = first_folded["oracle_gate_proxy_estimate"]
    assert gate_proxy["per_query_toffoli_proxy"] > 0
    assert gate_proxy["qrom_entries"] == block_estimate["unique_coefficient_values"]
    assert "final_time_readout" in first_folded
    assert report["chained_pathway_sweep"]
    comparison = report["chained_pathway_sweep"][0]["monolithic_vs_sequential"]
    assert comparison["monolithic_folded_be_dimension"] > 0
    assert comparison["sum_sequential_folded_be_dimensions"] > 0
    dynamic = report["chained_pathway_sweep"][0]["dynamic_handoff_reference"]
    assert dynamic["method"] == "exact_polynomial_reference_segment_handoff"
    assert dynamic["sequential_terminal_product_concentration"] >= 0.0
    assert dynamic["monolithic_terminal_product_concentration"] >= 0.0
    assert dynamic["absolute_terminal_product_difference"] >= 0.0
    assert len(dynamic["segment_results"]) == 2


def test_small_sweep_can_estimate_equilibrated_qsvt_proxy():
    report = build_report(
        0,
        0,
        0.1,
        0.2,
        max_wall_seconds=30.0,
        max_folded_dimension=5000,
        estimate_condition=True,
        estimate_equilibration=True,
        condition_max_dimension=1000,
    )
    folded = report["intermediate_sweep"][0]["folded_systems"][0]
    assert folded["condition_number_estimate"] > 0.0
    assert folded["equilibrated_condition_number_estimate"] > 0.0
    assert folded["equilibrated_qsvt_resource_estimate"]["inverse_polynomial_degree_proxy"] > 0
    assert folded["oracle_gate_proxy_estimate"]["total_toffoli_proxy"] > 0
    assert folded["equilibrated_oracle_gate_proxy_estimate"]["total_toffoli_proxy"] > 0
    assert "equilibration_notes" in folded


def test_sweep_accepts_custom_chained_cases():
    report = build_report(
        0,
        1,
        0.1,
        0.2,
        max_wall_seconds=30.0,
        max_folded_dimension=5000,
        chained_cases=((0, 2), (1, 0, 1)),
    )
    assert report["caps"]["chained_cases"] == [[0, 2], [1, 0, 1]]
    assert [item["intermediate_counts"] for item in report["chained_pathway_sweep"]] == [
        (0, 2),
        (1, 0, 1),
    ]
    assert len(report["chained_pathway_sweep"][1]["dynamic_handoff_reference"]["segment_results"]) == 3


def test_sweep_writes_scan_friendly_csv_summaries(tmp_path):
    report = build_report(
        0,
        1,
        0.1,
        0.2,
        max_wall_seconds=30.0,
        max_folded_dimension=5000,
        chained_cases=((0, 2),),
    )
    intermediate_path, chained_path = write_summary_csvs(report, tmp_path / "summary.csv")
    intermediate_text = intermediate_path.read_text(encoding="utf-8")
    chained_text = chained_path.read_text(encoding="utf-8")
    assert intermediate_text.startswith("intermediates,reactions,physical_dimension")
    assert "folded_be_oracle_per_query_toffoli_proxy" in intermediate_text.splitlines()[0]
    assert chained_text.startswith("intermediate_counts,segment_count")
    assert "0-2" in chained_text
    assert "absolute_terminal_product_difference" in chained_text.splitlines()[0]


def test_tiny_pathway_graph_runs_end_to_end_with_qsvt_simulator():
    config = Config(
        system=SystemConfig(name="pathway_graph", initial_substrate=0.1),
        pathway=PathwayConfig(
            mode="chain",
            segments=(PathwaySegmentConfig(intermediate_count=0),),
        ),
        linearization=MethodConfig("carleman", {"order": 2}),
        integrator=MethodConfig("folded_backward_euler"),
        qls=MethodConfig("qsvt_simulator", {"degree": 31}),
        time=TimeConfig(t_final=0.1, dt=0.1, n_points=2),
        output=OutputConfig(save_plot=False),
    )
    result = run_experiment(config)
    assert result.linearized_system.physical_dimension == 3
    assert result.integration.metadata["method"] == "folded_backward_euler"
    assert result.physical_states.shape == (2, 3)
