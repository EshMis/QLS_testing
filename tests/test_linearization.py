from math import comb

import numpy as np

from qls_testing.core.datatypes import PolynomialSystem
from qls_testing.linearization.carleman import CarlemanLinearization, lift_state
from qls_testing.linearization.restarted import AdaptiveRestartedCarleman
from qls_testing.integrators.linear import KrylovExponentialIntegrator
from qls_testing.qls.classical import ClassicalSolver
from qls_testing.systems.metabolic import build_mass_action_pathway, build_qssa_moving_point_pathway
from qls_testing.core.config import SystemConfig


def test_scalar_quadratic_carleman_matrix_matches_chain_rule():
    system = PolynomialSystem(("x",), ({(1,): -2.0, (2,): 0.5},), np.array([0.3]))
    lifted = CarlemanLinearization(order=2).linearize(system)
    expected = np.array([[-2.0, 0.5], [0.0, -4.0]])
    np.testing.assert_allclose(lifted.matrix, expected)
    np.testing.assert_allclose(lifted.initial_state, [0.3, 0.09])
    assert lifted.metadata["dropped_term_count"] == 1


def test_embedding_derivative_is_exact_for_physical_coordinates():
    system = PolynomialSystem(
        ("x", "y"),
        ({(1, 0): -1.0, (1, 1): 2.0}, {(0, 1): -0.5}),
        np.array([0.2, 0.4]),
    )
    lifted = CarlemanLinearization(order=2).linearize(system)
    lifted_state = lift_state(system.initial_state, lifted.exponents)
    np.testing.assert_allclose((lifted.matrix @ lifted_state)[:2], system.evaluate(system.initial_state))


def test_mass_action_dimension_formula_and_degree_one_order():
    system = build_mass_action_pathway(SystemConfig())
    lifted = CarlemanLinearization(order=2).linearize(system)
    assert lifted.matrix.shape == (comb(9, 1) + comb(10, 2),) * 2
    assert lifted.labels[:9] == system.variable_names


def test_moving_point_qssa_uses_homogeneous_coordinate_and_physical_projection():
    config = SystemConfig(initial_substrate=3.0, taylor_degree=2, qssa_expansion="moving_point")
    system = build_qssa_moving_point_pathway(config)
    lifted = CarlemanLinearization(order=2).linearize(system)
    assert lifted.metadata["homogeneous_coordinate"] is True
    assert lifted.labels[-1] == "1"
    np.testing.assert_allclose(lifted.project(lifted.initial_state), [3.0, 0.0, 0.0, 0.0, 0.0])
    assert lifted.metadata["series_valid_initially"] is True


def test_restarted_carleman_adapts_and_reembeds():
    system = PolynomialSystem(("x",), ({(1,): -1.0, (2,): 0.2},), np.array([0.5]))
    controller = AdaptiveRestartedCarleman(
        initial_order=1, max_order=3, segment_duration=0.1, tolerance=1e-4
    )
    result = controller.evolve(
        system,
        KrylovExponentialIntegrator(),
        ClassicalSolver(),
        t_final=0.2,
        dt=0.05,
    )
    assert result.physical_states.shape == (5, 1)
    assert len(result.orders) == 2
    assert all(1 <= order <= 3 for order in result.orders)
