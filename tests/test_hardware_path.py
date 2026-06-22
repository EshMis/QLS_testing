import numpy as np

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
from qls_testing.systems.pathway_family import build_mass_action_chain
from qls_testing.systems.metabolic import build_mass_action_pathway
from qls_testing.core.config import SystemConfig


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
