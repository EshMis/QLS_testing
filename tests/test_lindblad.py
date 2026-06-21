import numpy as np

from qls_testing.models.lindblad import amplitude_damping_model, simulate_lindblad


def test_amplitude_damping_matches_analytic_population_and_preserves_density_matrix():
    result = simulate_lindblad(amplitude_damping_model(1.0), t_final=2.0, n_points=21)
    np.testing.assert_allclose(result.populations[:, 1], np.exp(-result.times), atol=2e-12)
    np.testing.assert_allclose(result.populations[:, 0], 1.0 - np.exp(-result.times), atol=2e-12)
    assert np.max(result.trace_error) < 1e-12
    assert np.min(result.minimum_eigenvalue) > -1e-12

