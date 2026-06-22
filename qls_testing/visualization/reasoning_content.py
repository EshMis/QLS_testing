"""Concise, configuration-specific reasoning shown in the dashboard."""

from __future__ import annotations

from qls_testing.core.config import Config


def active_reasoning(config: Config) -> tuple[tuple[str, str], ...]:
    """Return only explanations relevant to the selected experiment."""
    if config.system.name.startswith("lindblad_"):
        execution = (
            "PennyLane applies normalized first-order Kraus channels. Reduce the channel step "
            "and monitor extracted-observable error, trace, and minimum density eigenvalue."
            if config.system.name.endswith("_pennylane")
            else "DOP853 integrates the density equation. It is a classical verification path, "
            "not a QLS or a quantum-speedup claim."
        )
        return (
            (
                "Why NDME applies",
                "The selected branch embeds the lifted linear ODE in an off-diagonal density "
                "block. The plotted reference is the exact Liouvillian ground truth.",
            ),
            ("Why it may fail", execution),
            (
                "Hardware signal",
                "Trace and positivity are necessary but insufficient: also require small error "
                "in S/Xs/P/Cs and account for channel dilation, shots, and shift amplification.",
            ),
        )

    integrators = {
        "backward_euler": "L-stable and robust for fast lifted decay, but first-order damping raises error.",
        "crank_nicolson": "Second-order and accurate here; very stiff lifted modes can ring because it is not L-stable.",
        "bdf2": "Second-order with useful stiff damping; the backward-Euler startup matters at short horizons.",
        "folded_backward_euler": "One history-state solve, but dimension and conditioning grow with all time steps.",
        "folded_crank_nicolson": "One second-order history solve with better observed conditioning, but it is not L-stable.",
        "folded_bdf2": "One second-order history solve with stiff damping; bootstrap and two time shifts add LCU terms.",
        "pade22": "A global low-order rational exponential; long horizons can leave its approximation region.",
        "rk45": "Adaptive classical validation; fast lifted modes increase RHS evaluations and it exposes no QLS.",
        "krylov_exponential": "Exploits the sparse constant lift and usually reaches the Carleman truncation floor efficiently.",
        "exponential": "Dense small-system reference; accurate for the finite lift but not scalable.",
    }
    solvers = {
        "classical": "Correctness baseline. Cache repeated factorizations before comparing runtime.",
        "hhl_simulator": "Validates Hermitian dilation and scaling; no circuit advantage is implied.",
        "pennylane_hhl": "Needs resolvable eigenvalues, deep controlled evolution, and adequate postselection.",
        "qsvt_simulator": "Tests reciprocal-polynomial error on singular values using classical SVD machinery.",
        "pennylane_qsvt": "Most relevant fault-tolerant direction if sparse block encoding and conditioning are solved.",
        "pennylane_vqls": "Near-term practice path; require loss decrease, nonzero updates, and a small true residual.",
        "pennylane_complex_vqls": "Adds phases for complex systems at higher parameter and measurement cost.",
        "iterative_refinement": "Useful only while correction solves reduce the original residual.",
        "preconditioned_qsvt": "Jacobi scaling may improve the inverse interval; verify condition before and after.",
    }
    return (
        (f"Integrator: {config.integrator.name}", integrators.get(config.integrator.name, "Inspect stability and step sensitivity.")),
        (f"Solver: {config.qls.name}", solvers.get(config.qls.name, "Judge this solver by original-system residual.")),
        (
            "What dominates",
            "Compare staged errors first. Once solve error is below Carleman/model and time error, "
            "additional solver precision cannot improve the physical trajectory.",
        ),
    )
