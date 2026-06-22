# Method comparisons for the enzyme pathway

## Properties that drive method behavior

The mass-action model has nine physical variables and a quadratic, non-normal
vector field. At the notebook parameters, substrate binding and catalytic release
create different decay scales. The product coordinate accumulates and contributes
a zero eigenvalue. This is not an extreme textbook stiff problem at the tested
horizons, but its lifted fast modes and non-normal couplings make stability,
transient growth, and conditioning relevant.

For the order-1 and order-2 Carleman matrices currently constructed:

| order | dimension $D$ | nonzeros | sparsity | $\kappa_2(I-0.1A)$ | non-normality $\|A^TA-AA^T\|_F$ | real spectral range |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 9 | 20 | 75.3% | 1.76 | 40.3 | $[-5.56,0]$ |
| 2 | 54 | 180 | 93.8% | 2.81 | 178.6 | $[-11.12,0]$ |

Thus lifting makes the matrix larger and sparser, extends the fast negative
spectrum, and increases non-normality. At higher order the dimension follows

$$
D(9,N)=\sum_{k=1}^{N}{9+k-1\choose k},
$$

so dense storage or generic block encoding quickly becomes the wrong abstraction.
Convergence can fail because high-degree terms are truncated, a QSSA Taylor
series leaves its trust region, a discrete solve becomes ill-conditioned, or an
approximate quantum solver returns a normalized direction with incorrect scale.

## Integrator comparison

The following observed values use the mass-action model, Carleman order 2,
$t\in[0,0.5]$, $\Delta t=0.1$, and classical linear solves. Global RMSE is
against the same ground-truth nonlinear ODE. The approximately $0.03$ floor for
the exponential/reference methods is finite-Carleman truncation, not integration
error.

| method | observed global RMSE | relevant observation |
|---|---:|---|
| Backward Euler | 0.0400 | extra first-order damping; LHS condition 2.81 |
| Crank--Nicolson | 0.0301 | low time error here; LHS condition 1.83 |
| BDF2 | 0.0346 | startup error plus second-order stepping; LHS condition 2.14 |
| global Padé [2/2] | 0.0446 | one low-order rational approximation over each requested $t$ |
| RK45 | 0.0315 | 176 lifted RHS evaluations |
| Krylov exponential | 0.0315 | reaches the truncation floor using sparse actions |
| dense exponential | 0.0315 | small-system reference only |
| folded Backward Euler | 0.0400 | same trajectory as stepwise BE; folded condition 18.5 |

### Backward Euler

Backward Euler is L-stable and strongly damps the lifted eigenmodes near
$-11$. It is the safest implicit baseline when order or rates increase. Its
first-order diffusion explains the observed error above the truncation floor.
One time-independent LHS allows factorization reuse classically. On quantum
hardware it creates repeated related right-hand sides, which is useful only if
state preparation and warm starts are exploitable.

### Crank--Nicolson

CN is second-order and A-stable. It performed best among the tested fixed-step
solve formulations at this horizon. It is not L-stable: if a higher lift adds
very fast negative modes, their amplification approaches $-1$ rather than zero,
so oscillatory artifacts can appear. Diagnose this through alternating lifted
coordinates and sensitivity to halving $\Delta t$.

### BDF2

BDF2 combines second-order accuracy with better stiff damping than CN. The
Backward-Euler bootstrap is visible at short horizons and partly explains why
it trails CN in this run. It is a strong default when the lifted spectrum becomes
more separated, provided two-step history is acceptable.

### Folded Backward Euler

The folded system encodes all time steps in one block lower-bidiagonal solve.
It reproduces stepwise BE algebraically, but its dimension is $N_tD$ and its
observed condition number was 18.5 versus 2.81 for one step. This is valuable
for history-state QLS research, not for dense laptop efficiency. Preserve block
sparsity and monitor condition growth before attempting hardware.

### Global Padé [2/2]

This implementation solves a rational approximation to $e^{tA}$ independently
at each output time. It is not a stepwise fourth-order method. Its error grows
when $\|tA\|$ leaves the low-order approximation region, explaining the poorer
result at the end of the observed interval. Scaling-and-squaring or local Padé
steps would be the natural improvement.

### RK45, dense exponential, and Krylov exponential

RK45 is a useful adaptive validation route but does not expose QLS subproblems;
fast lifted modes increase its function evaluations. Dense `expm` reaches the
finite-lift truth but scales cubically. Krylov `expm_multiply` exploits the 94%
sparsity of the order-2 operator and reaches the same truncation floor without
forming $e^{tA}$. For this constant sparse lifted system, Krylov is the strongest
classical numerical direction to develop first.

## Linear-solver and quantum-method comparison

The implicit matrices are real, sparse, non-Hermitian, and non-normal. None of
HHL, QSVT, or VQLS removes Carleman truncation error; once solver residual is
below the truncation/discretization floor, spending more quantum resources does
not improve the physical trajectory.

### Classical direct solve

This is the acceptance baseline. Current calls refactorize repeated matrices;
caching LU should be implemented before using wall time to argue for another
solver. Monitor relative residual, componentwise scale, and factorization reuse.

### HHL

HHL needs Hermitian dynamics for phase estimation. The code uses

$$
H=\begin{bmatrix}0&M\\M^\dagger&0\end{bmatrix},
$$

doubling dimension and mapping singular values to signed eigenvalues. Cost and
rotation accuracy depend strongly on $\kappa$, while success probability can
require amplitude amplification. More clock qubits improve eigenvalue resolution
but increase controlled-evolution depth exponentially in the explicit demo.
Symptoms of failure are low postselection probability, unresolved small
eigenvalues, wrong magnitude after scale restoration, and large original-system
residual. HHL is pedagogically useful here but not the first hardware candidate.

### QSVT

QSVT approximates $1/\sigma$ over a bounded singular-value interval. It handles
non-Hermitian matrices through singular-value transformation more naturally than
HHL dilation, but requires a scalable block encoding of the sparse lifted/discrete
operator and degree roughly $O(\kappa\log(1/\epsilon))$. Native phase synthesis
can become numerically unstable. Watch effective versus requested degree,
polynomial error, encoded singular-value bounds, postselection, and final
residual. Structured sparse block encoding plus preconditioning makes QSVT the
most relevant fault-tolerant direction in this repository.

### VQLS

VQLS replaces deep inverse transformation with a trainable ansatz. It does not
require Hermiticity when the residual $\|Mx-b\|^2$ is evaluated correctly, but
that global cost can itself require expensive matrix decomposition and
measurements. Sequential ODE right-hand sides make parameter transfer/warm starts
attractive. Dominant risks are insufficient expressivity, barren or noisy
gradients, local minima, and probability measurements that lose amplitude phase.
Monitor loss, true residual, gradient norm, update norm, state displacement,
probability entropy, and shot variance. VQLS is the most accessible NISQ practice
path, but a small residual must be demonstrated—not inferred from loss alone.

### Refinement and preconditioning

Jacobi scaling is cheap and can reduce the reciprocal-polynomial interval, but
non-normal matrices may not benefit and a zero diagonal breaks it. Iterative
refinement is useful when the base QLS produces a correction correlated with the
residual. Stop on stagnation. Hardware claims must include the cost of preparing
each residual state and applying the preconditioner.

## Lindbladian pipeline versus QLS

QLS pathways discretize $\dot y=Ay$ into $Mx=b$. NDME instead embeds the same
linear ODE in an off-diagonal density block governed by a GKSL equation. A shift
$\kappa I$ makes the Hermitian part semidissipative, and extraction multiplies by
$e^{\kappa t}$. The plotted $S$, $X_i$, $P$, and $C_i$ values come from that
off-diagonal block, not density populations.

The exact ground truth is generated by the exact sparse Liouvillian exponential.
The classical NDME pipeline integrates the master equation with DOP853. The
PennyLane pipeline applies normalized first-order Kraus channels on
`default.mixed`. Its dominant error is short-time channel/Trotter error, controlled
by substeps, followed by shift-amplified extraction error. Trace and positivity
are necessary diagnostics but do not guarantee correct extracted amplitudes.

Lindbladian simulation is appropriate when open-system structure is itself the
model or when testing the paper's embedding. It is not automatically preferable
for a deterministic enzyme ODE: density dimension is $2D$, Liouville dimension
is $4D^2$, and arbitrary channel realization may require an environment dilation.

## Practical hardware decision guide

1. Start with the four-coordinate PennyLane Lindbladian practice model and tiny
   two-state VQLS systems. Require trace/positivity plus observable error for the
   former, and true residual plus parameter movement for the latter.
2. Establish sparse Krylov and cached-LU classical baselines. Stop quantum-solver
   optimization once its error is below Carleman/time-discretization error.
3. On near-term hardware, test shallow VQLS with warm starts, local costs, finite
   shots, and one or two scientific observables rather than tomography.
4. Treat HHL as a resource-estimation exercise until controlled Hamiltonian
   simulation, clock precision, and postselection are affordable.
5. For fault-tolerant planning, prioritize structured sparse block encoding,
   equilibration/preconditioning, and QSVT. Report oracle construction and
   observable readout, not polynomial query degree alone.
6. Scale NDME only after measuring channel error versus substeps and the recovery
   factor $e^{\kappa T}$. A native dissipative channel or efficient dilation is
   required for a meaningful hardware implementation.

Success means stable physical observables on a fixed ground-truth grid, residuals
or channel errors below the chosen approximation budget, nonvanishing success
probability, and resource estimates that include preparation and readout.

