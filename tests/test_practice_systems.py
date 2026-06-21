import numpy as np

from qls_testing.models.practice import PRACTICE_SYSTEMS
from qls_testing.linearization.carleman import CarlemanLinearization


def test_practice_suite_has_fast_and_dashboard_cases_with_structural_metadata():
    assert len(PRACTICE_SYSTEMS) >= 7
    speeds = {system.speed for system in PRACTICE_SYSTEMS.values()}
    assert speeds == {"fast", "dashboard"}
    properties = {value for system in PRACTICE_SYSTEMS.values() for value in system.properties}
    assert {"sparse", "dense", "non-Hermitian", "non-PSD"} <= properties
    assert "complex" in properties


def test_every_practice_system_has_exact_reference_and_matching_polynomial_rhs():
    for name, practice in PRACTICE_SYSTEMS.items():
        polynomial = practice.polynomial_system()
        np.testing.assert_allclose(polynomial.evaluate(practice.initial_state), practice.matrix @ practice.initial_state)
        reference = practice.reference(np.array([0.0, 0.1]))
        np.testing.assert_allclose(reference[0], practice.initial_state)
        assert reference.shape == (2, len(practice.labels)), name


def test_complex_practice_system_survives_order_one_lifting():
    practice = PRACTICE_SYSTEMS["practice_complex_damped_2"]
    lifted = CarlemanLinearization(order=1).linearize(practice.polynomial_system())
    assert np.iscomplexobj(lifted.matrix)
    np.testing.assert_allclose(lifted.matrix, practice.matrix)
