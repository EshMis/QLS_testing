# Per-run complexity derivations and code symbols

The dashboard constructs complexity from the selected configuration only. Each
`ComplexityTerm` stores a LaTeX expression, values substituted from the run,
the code operation that generated it, and assumptions. These are transparent
operation proxies rather than wall-clock predictions.

## Symbols

| Symbol | Runtime source |
|---|---|
| \(n\) | `LinearizedSystem.physical_dimension` |
| \(N\) | Carleman `metadata["order"]` |
| \(D\) | `lifted.matrix.shape[0]` |
| \(z=\operatorname{nnz}(A_C)\) | counted from the realized lifted matrix |
| \(N_t\) | `len(integration.times)-1` |
| \(N_{solve}\) | `len(integration.solve_diagnostics)` |
| \(\kappa\) | solver diagnostic or discrete LHS condition proxy |
| \(d_p\) | QSVT diagnostic `degree` |
| \(n_c\) | HHL diagnostic `n_clock` |
| \(N_{opt}\) | sum of VQLS optimizer `steps` |
| \(N_{shots}\) | VQLS/HHL/QSVT shot setting |

## Carleman stage

The number of commutative monomials of degree \(k\) in \(n\) variables is
\({n+k-1\choose k}\), hence

\[
D(n,N)=\sum_{k=1}^N{n+k-1\choose k}+\delta_{affine}.
\]

The current constructor allocates a dense \(D\times D\) array, giving
\(O(D^2)\) storage initialization. A sparse implementation would store
\(O(z)\) entries; row construction is proportional to retained and dropped
monomial contributions.

## Selected integrator stage

Backward Euler, Crank--Nicolson, and BDF2 perform one selected-solver call per
step (including the BDF2 bootstrap):

\[
C=N_{solve}C_{solve}(D)+O(N_tz).
\]

If the fixed left-hand matrix is factorized once, dense direct work changes
from \(O(N_tD^3)\) to \(O(D^3+N_tD^2)\). The current `ClassicalSolver` does not
cache that factorization.

Folded backward Euler has dimension \(D_F=N_tD\). Its block-bidiagonal sparsity
is attractive in principle, but this implementation materializes the matrix
dense before the selected solver, so a direct solve has \(O(D_F^3)\) cost.

Global Padé [2/2] forms \(A_C^2\) once and performs one solve per output time:

\[
C=C(A_C^2)+N_t[C_{matvec}+C_{solve}(D)].
\]

RK45 reports \(N_{fev}\), giving \(O(N_{fev}z)\) under a sparse matvec model
or \(O(N_{fev}D^2)\) for dense execution. Krylov exponential action uses an
internally selected degree \(m\), approximately \(O(N_tmz)\); tolerances,
scaling, norm, and non-normality determine \(m\).

## Selected solver stage

Dense direct solution is \(O(D^3)\) per uncached factorization. HHL's explicit
QPE circuit applies \(2^{n_c}-1\) controlled evolution powers per solve. Its
ideal oracle complexity additionally depends on sparsity, \(\kappa\), precision,
state preparation, postselection, and readout.

An inverse QSVT polynomial of degree \(d_p\) uses \(O(d_p)\) alternating
block-encoding queries, with the ideal approximation scaling
\(d_p=O(\kappa\log(1/\epsilon))\). Phase synthesis and a scalable block encoding
are excluded from that query statement unless explicitly counted.

The implemented PennyLane VQLS diagnostic cost is

\[
C_{VQLS}\sim N_{opt}N_{QNode/step}C_{circuit}
+N_{solve}N_{shots}.
\]

Instrumentation currently incurs about three analytic QNode traversals per
optimizer step: an explicit diagnostic gradient, Adam's update gradient, and a
post-update loss. The final shot QNode samples probabilities. `default.qubit`
statevector simulation itself scales exponentially with qubit count, so circuit
depth alone is not a laptop-runtime estimate.

Iterative refinement costs one initial base solve plus one residual matvec and
base correction solve per successful round. Diagonal preconditioning adds
\(O(D)\) scaling and should be judged by the observed change in \(\kappa\).

## NDME stage

For lifted dimension \(D\), the code uses a two-block ancilla extension so
\(d_\rho=2D\), then vectorizes the density matrix:

\[
d_{\mathcal L}=d_\rho^2.
\]

A dense classical Liouvillian application is \(O(d_{\mathcal L}^2)\) per RHS
evaluation. This must not be confused with the reference paper's oracle query
complexity, documented in `lindblad.md`.

## Reading the UI

`DefaultComplexityEstimator` branches on the active integrator and solver.
Therefore a classical RK45 run has no HHL/QSVT/VQLS term, a VQLS run has no HHL
term, and an NDME run has no QLS stage. The JSON export preserves the same term
provenance shown in each Streamlit accordion.
