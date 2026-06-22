# Enzyme ODE, lifted dynamics, and quantum-solver research question

## The physical model

The notebook studies four sequential enzyme-catalysed reactions

\[
S\rightarrow X_1\rightarrow X_2\rightarrow X_3\rightarrow P.
\]

For reaction \(i\), substrate \(Z_i\in(S,X_1,X_2,X_3)\) binds free enzyme
\(E_i^{\rm free}=E_i^T-C_i\), forms complex \(C_i\), and releases the next
metabolite with rate constants \(k_{1i},k_{-1i},k_{{cat},i}\). Define

\[
v_i^+=k_{1i}Z_i(E_i^T-C_i),\qquad
v_i^-=k_{-1i}C_i,\qquad q_i=k_{{cat},i}C_i.
\]

Then

\[
\dot S=-v_1^++v_1^-,\quad
\dot X_i=q_i-v_{i+1}^++v_{i+1}^-\ (i=1,2,3),\quad
\dot P=q_4,
\]

\[
\dot C_i=v_i^+-v_i^- -q_i.
\]

The computed observables are the trajectories grouped in the dashboard as
\(S\), \((X_1,X_2,X_3)\), \(P\), and \((C_1,C_2,C_3,C_4)\). The exact
mass-action vector field is quadratic in nine variables. The alternative QSSA
model removes the complexes and uses

\[
v_i(Z_i)=\frac{V_{\max,i}Z_i}{K_{m,i}+Z_i},\qquad
V_{\max,i}=k_{{cat},i}E_i^T,
\]

which is rational rather than polynomial. The code Taylor-expands this rate
either at zero or around a moving operating point. A zero-centred series only
converges for \(|Z_i/K_{m,i}|<1\); the notebook default starts at
\(S(0)/K_{m,1}=1.5\), so `auto` selects the moving-point expansion.

## Why linearize before using a QLS?

A direct ODE solver is the practical baseline for this small problem. The
Carleman/QLS route is a research decomposition, not a claim that it is faster:

\[
\dot x=f(x)
\xrightarrow{\text{finite Carleman lift}}
\dot y=A_Ny
\xrightarrow{\text{time discretization}}
M_ky_{k+1}=b_k
\xrightarrow{\text{selected solver}}
\widetilde y_{k+1}.
\]

It makes three questions independently testable: how a polynomial nonlinear
system becomes a structured linear one, how stable time schemes expose linear
systems, and what assumptions HHL/QSVT/VQLS would need on those systems.
Carleman dimension grows combinatorially, so the same transformation that makes
the dynamics linear also creates the principal scaling problem.

## Is this an open problem?

Carleman linearization, implicit and exponential integrators, classical linear
solvers, HHL, QSVT, and variational linear solvers are established subjects.
The enzyme ODE itself is not presented here as an unsolved mathematical
problem. Less settled engineering/research questions arise in their
combination:

- selecting or adapting the Carleman order over a trajectory with a useful,
  computable truncation certificate;
- balancing model/Taylor, truncation, time-discretization, and approximate-solve
  errors rather than tuning one in isolation;
- preserving sparsity and obtaining a realistic block encoding of the lifted
  matrices after non-Hermitian dilation and time discretization;
- controlling condition number and success probability as the step size,
  order, and folded history dimension change;
- training shot-based VQLS objectives without assuming free amplitude readout;
- deciding whether a useful observable can be measured without reconstructing
  every concentration; and
- comparing the separate Lindbladian embedding under honest preparation,
  amplification, and readout costs.

The repository therefore reports residuals and operation proxies, never a
blanket quantum advantage.

## Reasoning that transfers to similar systems

### Stability and stiffness

First inspect the spectrum or numerical range of \(A_N\), and monitor the
condition of each discrete left-hand matrix. Backward Euler and BDF2 damp stiff
modes; Crank--Nicolson is A-stable but not L-stable and can preserve unwanted
oscillation. RK45 is useful as a classical check but can take many evaluations
on stiff systems. Krylov exponential action avoids a stability step limit for a
constant linear operator, although approximation work still depends on its
norm and non-normality.

### Truncation and trust regions

Carleman truncation discards derivative monomials above order \(N\). Increasing
\(N\) helps only while finite-precision, conditioning, and cost remain
controlled. Restarting re-embeds monomials from a new physical state, restoring
algebraic consistency. QSSA Taylor expansions require a separate trust-region
test; Carleman order cannot repair a divergent Taylor model.

### Conditioning and QLS assumptions

For implicit methods, changing \(\Delta t\) changes both integration error and
\(\kappa(M)\). HHL and inverse-QSVT cost and accuracy are sensitive to small
singular values. Jacobi preconditioning is inexpensive but can fail on a zero
diagonal or worsen non-normal systems. Iterative refinement is valuable only
when approximate correction solves reduce the true residual.

### Normalization and observables

Quantum linear algorithms naturally prepare a normalized state proportional to
the solution. This code restores magnitude explicitly and verifies
\(\|Mx-b\|\). Full statevector extraction is available only for tiny simulator
validation. A future hardware experiment should define a small set of physical
observables before claiming an end-to-end benefit.

## Why each implemented method is present

### Linearization

- `carleman` exploits polynomial closure in an infinite monomial basis. Finite
  order is transparent and easy to verify, but truncation error and dimension
  growth are the central risks.
- `adaptive_restarted_carleman` compares adjacent orders segment by segment and
  re-embeds the accepted endpoint. It limits lift drift but doubles some trial
  integrations and its adjacent-order discrepancy is an estimator, not a proof.
- QSSA moving-point Taylor expansion reduces physical dimension and keeps the
  rational rate locally polynomial. It is parameter-sensitive and must be
  restarted or rejected outside its radius of validity.

### Integrators

- `backward_euler`: robust, L-stable, and one reusable linear system per step;
  first-order numerical diffusion is the price.
- `crank_nicolson`: second-order and time-symmetric for linear systems; stiff
  decay can ring because it is not L-stable.
- `bdf2`: second-order with good stiff decay; it needs a backward-Euler startup
  and two prior states.
- `folded_backward_euler`: exposes a single history-state linear system, useful
  for QLS studies; dimension becomes \(N_{step}D\) and dense simulation is costly.
- `pade22`: a rational global approximation to \(e^{tA}\), useful for testing
  solve-based exponential ideas; it is not a local fourth-order stepper and can
  deteriorate over long horizons.
- `krylov_exponential`: exploits repeated sparse matrix-vector products and is
  attractive when only \(e^{tA}y_0\) is needed; work depends on spectral and
  non-normal structure.
- `rk45`: adaptive, familiar validation baseline; not a QLS pathway and not the
  preferred stiff method.
- `exponential`: dense exact-matrix-exponential reference for small systems;
  cubic storage/work makes it a benchmark only.

### Linear solvers

- `classical` is the correctness baseline. It is reliable for the tiny dense
  systems but currently refactorizes repeated matrices.
- `hhl_simulator` validates dilation, eigenvalue inversion, scaling, and block
  extraction; classical eigendecomposition means it has no speedup claim.
- `pennylane_hhl` executes QPE and conditioned rotations. Clock precision and
  postselection rapidly make it impractical beyond tiny matrices.
- `qsvt_simulator` stably applies a Chebyshev reciprocal to singular values;
  `pennylane_qsvt` tests native block encoding and phase synthesis. Both become
  difficult as the inverse interval approaches zero.
- `vqls_simulator` isolates the normalized-state objective without gate
  expressivity. `pennylane_vqls` adds a trainable circuit, gradients, optimizer,
  warm starts, and sampled readout diagnostics. Barren plateaus, shallow
  expressivity, shot variance, and local minima remain failure modes.
- `preconditioned_qsvt` tests whether simple diagonal scaling improves the
  inverse-polynomial interval. Always compare condition numbers before/after.
- `iterative_refinement` converts an approximate solver into residual
  corrections. Stagnation is diagnosed and stops the loop.

### Lindbladian branch

`lindblad_ndme` embeds the order-1 linear ODE in a nondiagonal density-matrix
block following the archived reference. It exploits completely positive trace
preserving dynamics and is verified through trace, positivity, and extraction
errors. It is not a QLS solver. The semidissipative shift can amplify recovery
error by \(e^{\kappa t}\), while density-vectorization squares the encoded
dimension for the classical simulator.

