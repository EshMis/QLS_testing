# Quantum linear-solver methods

All plugins solve \(Mx=b\) and return a classical vector only to support research
verification. A hardware implementation would generally prepare a state
proportional to \(M^{-1}b\); tomography/readout costs must be included in any
end-to-end complexity claim.

## Classical baseline

`classical` calls `numpy.linalg.solve` and reports

\[
r_{rel}=\frac{\|Mx-b\|_2}{\max(\|b\|_2,\epsilon)}.
\]

It is the stable baseline used by most integrator tests.

## Idealized HHL simulator

HHL requires a Hermitian operator. For non-Hermitian \(M\), use

\[
H=\begin{bmatrix}0&M\\M^\dagger&0\end{bmatrix},\qquad
\tilde b=\begin{bmatrix}b\\0\end{bmatrix}.
\]

Solving \(H\tilde x=\tilde b\) places the desired solution in the lower half.
The simulator normalizes \(\tilde b\), applies exact spectral weights
\(1/\lambda_i\), restores its norm, and then slices that lower half. It uses a
classical eigendecomposition, standing in for phase estimation and controlled
rotation. Singular values below a configurable cutoff raise an error rather
than being silently pseudo-inverted.

The packaged `pennylane_hhl` backend now pads dimensions, uses signed phase
bins, applies conditioned rotations, runs inverse QPE, postselects the inversion
ancilla, and handles non-Hermitian dilation. It is limited to tiny matrices.

## QSVT polynomial simulator

With \(M=U\Sigma V^\dagger\), singular-value inversion seeks a bounded polynomial
approximating \(1/\sigma\) on \([\sigma_{min},\sigma_{max}]\). The package builds
a Chebyshev interpolant and applies

\[
x\approx Vp(\Sigma)U^\dagger b.
\]

Chebyshev coefficients are never converted to a high-degree power basis. The
reported diagnostics separate maximum reciprocal approximation error from the
actual linear-system residual. This class is a circuit-independent numerical
surrogate. The separate `pennylane_qsvt` backend performs native block encoding,
phase synthesis, QSVT, and postselection for tiny matrices; scalable sparse
oracles remain future work.

## Variational simulator

For a normalized ansatz \(u(\theta)\), define \(y=Mu\). The scalar minimizing
\(\|s y-b\|^2\) is

\[
s_*(\theta)=\frac{\langle y,b\rangle}{\langle y,y\rangle},
\]

and the loss is

\[
L(\theta)=\frac{\|M(s_*u)-b\|^2}{\|b\|^2}.
\]

`vqls_simulator` optimizes this objective over a full small real state vector.
It validates normalization and scale recovery but does not model a gate ansatz.
It is intentionally restricted to small research checks.

## Corrections from the notebook

- QSVT now returns directly instead of falling through into the HHL-only branch.
- Every method restores scale explicitly and is judged by \(\|Mx-b\|\), not only
  state overlap. A correct direction with wrong magnitude is not accepted.
- HHL dilation extracts the lower block and rejects unresolved eigenvalues.
- QSVT's validity interval is based on singular values and its classical
  surrogate works for general matrices; an SPD assumption is not smuggled in.
- “Quantum” backends are named simulators until circuit execution, postselection,
  shot noise, and readout semantics are actually present.

## Preconditioning and iterative refinement

The Jacobi wrapper solves

\[
D^{-1}Mx=D^{-1}b,\qquad D=\operatorname{diag}(M),
\]

then reports residuals in the original system. `preconditioned_qsvt` applies the
Chebyshev/QSVT surrogate to this transformed system and records condition numbers
before and after preconditioning.

Given an approximate solve \(x_k\), refinement forms \(r_k=b-Mx_k\), computes
\(M\delta_k\approx r_k\), and updates \(x_{k+1}=x_k+\delta_k\). It stops at the
requested residual, iteration limit, or residual stagnation. The full residual
history is retained for the error dashboard.

## Circuit-backed PennyLane VQLS

`pennylane_vqls` prepares \(u(\theta)\) with RY layers and entanglers on
`default.qubit`, eliminates the optimal magnitude analytically, and differentiates
the normalized residual loss with PennyLane/Autograd. It is tested on a 2×2
non-Hermitian system and the two-state toy ODE. Its optimization and full
statevector extraction are suitable for validation, not scalable readout.

The block-encoding demo uses PennyLane `BlockEncode` and verifies that projecting
the ancilla-zero branch returns \(A|b\rangle/\alpha\). CSR oracle data forms a
stable interface boundary for a future sparse circuit implementation.
