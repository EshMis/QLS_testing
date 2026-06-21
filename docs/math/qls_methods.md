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

The PennyLane circuit in the source notebook additionally pads dimensions,
uses signed phase bins, and postselects the inversion ancilla. Those details are
preserved in the notebook but are not a stable package backend yet.

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
surrogate; block encoding, QSVT phase synthesis, postselection, and sampling are
future adapters.

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

