# Implementation rationale and research trail

This is the “why” record for the June 15 notebook refactor. It complements the
derivations by preserving decisions, observed failures, and what they taught us.

## Decision trail

1. **Keep both physical models.** The nine-variable mass-action equations are
   the faithful quadratic system. QSSA is useful as a reduced comparison, but
   its rational rates require another approximation before Carleman lifting.

2. **Use finite Carleman lifting as the common boundary.** It translates both
   polynomial models into $\dot y=A_Ny$, making integrators and solvers
   interchangeable. The notebook's symbolic chain rule was retained and tested
   on brute-force toy systems rather than treated as matrix magic.

3. **Do not hide the QSSA radius failure.** The notebook default has
   $S(0)/K_{m,1}=1.5$, outside the zero-centred geometric-series radius. The
   implementation records this fact and uses a moving-point expansion in
   `auto` mode. Restarting was added because a locally valid polynomial need not
   remain valid over the full trajectory.

4. **Retain several integrators because they ask different research questions.**
   Backward Euler is the robust one-solve baseline; CN and BDF2 test higher-order
   implicit behavior; folded Euler exposes a history-state solve; Padé tests a
   rational exponential formulation; RK45 and matrix/Krylov exponentials provide
   solver-independent references. They are not ranked by one universal winner.

5. **Make residual and scale non-negotiable.** Quantum linear algorithms output
   normalized states naturally, but the ODE needs magnitudes. Every solver now
   reconstructs scale and is judged by $\|Mx-b\|$, not state overlap alone.

6. **Separate algebraic simulators from circuit-backed experiments.** The HHL
   and QSVT simulators validate dilation, cutoffs, inverse transforms, and
   conditioning. PennyLane variants test actual circuit semantics on tiny
   matrices. This avoids presenting a classical eigendecomposition or SVD as a
   hardware execution.

7. **Repair the notebook's QSVT dispatch and scaling path.** The old combined
   dispatcher could enter QSVT and then fall through to an HHL-only branch.
   Refactored plugins return directly, preserve normalization factors, and report
   original-system residuals. Native QSVT phase synthesis also reduces degree
   when unstable rather than silently returning a malformed result.

8. **Treat VQLS stagnation as a pipeline bug, not merely “bad optimization.”**
   The notebook observation was that most state variables converged toward zero
   or appeared unchanged. The original circuit used complex RY/RZ amplitudes for
   a real ODE and the real integration arrays discarded phase-bearing imaginary
   components. It restarted the identical shallow ansatz each time, had fewer
   real parameters than the padded enzyme state requires, and accepted `shots`
   without executing a shot-backed QNode.

9. **Fix and instrument VQLS.** Real systems now use a real RY ansatz, while the
   complex plugin is explicit. Effective depth grows enough to remove the simple
   parameter-count bottleneck; optimized weights warm-start the next time step.
   Loss, gradient norm, update norm, intermediate solutions, residual proxy,
   parameter samples, probabilities, entropy, and variance are retained. A
   separate finite-shot probability QNode makes the shot setting observable.
   The differentiable training loss remains analytic; genuinely shot-trained
   phase-sensitive objectives remain future work.

10. **Decompose error before comparing algorithms.** A single total curve could
    blame VQLS for a Taylor or Carleman error. The implementation separately
    estimates integration, Carleman, QLS, QSSA/model, and total discrepancies.
    They are diagnostic norms, not additive rigorous bounds. User-facing
    trajectory plots independently select absolute, relative, or both errors.

11. **Report only costs that ran.** The earlier report displayed HHL, QSVT,
    dense, and Krylov formulas together regardless of configuration. The new
    report branches by selected stages and gives every expression a runtime
    substitution, code source, and assumptions. This makes “this run” auditable.

12. **Keep NDME separate.** The supplied Lindbladian paper inspired a
    nondiagonal density-matrix embedding of the order-1 linear ODE. It is not a
    linear solver and therefore has its own trace, positivity, extraction, and
    Liouville-dimension diagnostics. The order-1 limit is deliberate: order 2
    would produce a 108-dimensional density matrix in the interactive path.

13. **Make ground truth method-independent.** Reference curves previously came
    from inline branches and NDME carried its own $e^{At}$ comparison. A single
    `qls_testing.reference` API now generates physical ODE truth or exact
    Liouvillian truth without accepting an integrator or solver argument.
    Regression tests require identical arrays when those methods change.

14. **Add a real PennyLane Lindbladian execution path.** The practice and
    order-1 enzyme modes evolve density matrices through normalized short-time
    Kraus channels on `default.mixed`. Exact Liouvillian exponentiation remains
    outside that pipeline as ground truth. This exposes channel-step error while
    preserving trace and positivity, without pretending `QubitChannel` is
    already a hardware-native implementation.

15. **Use one Markdown math convention.** GitHub documentation and Streamlit
    now use `$...$` and `$$...$$`. A repository script checks that legacy
    delimiters do not quietly return.

## What succeeded, what remains limited

The modular pipeline reproduces simple analytic systems, verifies Carleman rows,
checks small linear systems by residual, and evolves the tiny VQLS ODE with
nonzero parameter/state updates both with and without sampled diagnostics.
Classical and algebraic baselines remain substantially more reliable than the
circuit demos on lifted enzyme dimensions. HHL clock depth, QSVT phase synthesis,
VQLS expressivity/training, dense folded systems, and full-vector readout are
the present practical limits—not details to be hidden behind asymptotic notation.

## Next decisions worth testing

- Cache/factorize repeated classical implicit matrices to establish a stronger
  baseline before interpreting solver overhead.
- Replace dense Carleman construction with sparse combinatorial assembly and
  measure block sparsity after each discretization.
- Use adaptive order/restart criteria informed by a posteriori residuals rather
  than adjacent-order difference alone.
- Test L-stable higher-order schemes or exponential Rosenbrock/Krylov methods
  where stiff lifted spectra make CN unattractive.
- Compare Jacobi with equilibration, incomplete factorizations, and structured
  preconditioners while accounting for quantum-access cost.
- For VQLS, test local/overlap costs, layerwise training, parameter transfer in
  time, natural gradients, and finite-shot gradient estimators. Keep the
  classical residual as the acceptance criterion during simulator research.
- Design observable-level output experiments for $S$, product yield, or
  conserved quantities; avoid tomography when the scientific question does not
  require all amplitudes.
