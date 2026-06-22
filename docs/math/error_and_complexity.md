# Error decomposition and complexity estimates

## Staged error model

Let \(x_*\) be the physical target, \(x_p\) the direct polynomial-model
solution, \(P e^{A_Nt}y_0\) the exact finite lift, \(x_d\) the selected
discretization using exact classical solves, and \(x_q\) the configured solver
trajectory. The dashboard records

\[
e_{int}=\|x_d-Pe^{A_Nt}y_0\|,
\quad e_{Carl}=\|Pe^{A_Nt}y_0-x_p\|,
\]

\[
e_{QLS}=\|x_q-x_d\|,
\quad e_{QSSA}=\|x_p-x_*\|,
\quad e_{total}=\|x_q-x_*\|.
\]

It also records each solve's relative residual \(\|Mx-b\|/\|b\|\), QSVT's
reciprocal-polynomial error, iterative refinement's final residual, and a shot standard-error proxy when a circuit
backend reports one. These are per-time discrepancy estimates, not rigorous
independent bounds. Since vector directions are discarded, component norms are
not expected to add exactly to \(e_{total}\).

## Per-run cost provenance

For \(n\) physical variables and order \(N\),

\[
D(n,N)=\sum_{k=1}^N {n+k-1\choose k}
\]

plus one homogeneous coordinate for affine systems. The report includes \(D\),
nonzeros, sparsity, steps, and solves. It then adds terms only for the selected
linearization, integrator, solver, refinement/preconditioner, or NDME branch.
Every term records its expression, runtime substitution, generating code
operation, and assumptions. Reusing a dense factorization changes repeated
solves from \(O(nD^3)\) to \(O(D^3+nD^2)\). Full derivations and code-symbol
mappings are in [`complexity_derivations.md`](complexity_derivations.md).

## Quantum-oriented proxies

Under efficient state preparation, block-encoding, and favorable readout:

- idealized HHL has representative dependence
  \(\widetilde O(s\kappa\operatorname{polylog}(D)/\epsilon)\);
- inverse QSVT needs degree/query count
  \(O(\kappa\log(1/\epsilon))\);
- refinement multiplies base-solver calls by one plus the correction rounds;
- VQLS optimizer/QNode work and sampled readout are included only for a selected
  VQLS run.

These assumptions are strong. State preparation, postselection, tomography,
noise mitigation, compilation, and backend connectivity can dominate. **This is
an estimate; actual runtime depends on backend.**

For NDME mode the dashboard instead reports density/Liouville dimensions,
the semidissipative shift, trace defect, and positivity violation. Oracle
complexities from the supplied paper are documented in `lindblad.md`.
