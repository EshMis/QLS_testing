# PennyLane quantum components

## Block encoding

For \(\alpha\ge\|A\|\), a unitary \(U_A\) satisfies

\[
(\langle0|\otimes I)U_A(|0\rangle\otimes I)=A/\alpha.
\]

The implementation matches PennyLane's admissibility normalization based on
the infinity norms of \(AA^\dagger\) and \(A^\dagger A\), then runs the unitary
on `default.qubit`. Tests postselect the leading ancilla-zero amplitudes and
recover \(A|b\rangle\) after restoring \(\alpha\). `sparse_block_encoding_data`
exposes CSR row, column, and value arrays for later oracle circuits.

## VQLS circuit

RY layers and nearest-neighbor CNOTs prepare a real normalized state
\(u(\theta)\). For \(y=Au\), the optimal scale is

\[
s(\theta)=\frac{\langle y,b\rangle}{\langle y,y\rangle},
\]

and Adam minimizes \(\|Asu-b\|^2/\|b\|^2\) through a PennyLane QNode. Padding
uses an identity block, preserving the original solution. Diagnostics include
qubits, depth, optimization steps, cost, actual residual, and sampling standard
error proxy.

Limitations: the present ansatz is real, tiny-system oriented, noiseless by
default, and returns a simulator statevector. `pennylane_complex_vqls` adds RZ
phases and a complex optimal scale. `loss_mode: shot_proxy` records binomial
sampling uncertainty but still optimizes the exact statevector loss; a fully
shot-trained likelihood objective remains future work.

## PennyLane HHL

`PennyLaneHHLSolver` runs explicit quantum phase estimation on

\[
U=e^{2\pi iH/s},\qquad s>2\max_j|\lambda_j|,
\]

then applies clock-bin-controlled rotations

\[
R_y\!\left(2\arcsin(C/\widetilde\lambda_j)\right),
\qquad C<\min_j|\lambda_j|.
\]

After inverse QPE it postselects clock zero and inversion ancilla one. General
matrices use \(H=\left[\begin{smallmatrix}0&A\\A^\dagger&0\end{smallmatrix}\right]\)
and read the lower solution block. Diagnostics report clock/system qubits,
condition number, success probability, shot uncertainty, and residual.

## PennyLane QSVT

`PennyLaneQSVTSolver` fits an odd reciprocal polynomial on
\([-1,-1/\kappa]\cup[1/\kappa,1]\), then passes its power-basis coefficients to
PennyLane's native `qml.qsvt` using `BlockEncode`. The ancilla-zero block
approximates

\[
p(A/\alpha)\approx C(A/\alpha)^{-1}.
\]

Phase synthesis can be ill-conditioned even for a bounded polynomial when
power-basis coefficients grow. The implementation reduces the requested odd
degree until PennyLane synthesizes stable phases, records requested/effective
degrees, and always judges success using the original-system residual.

Both HHL and QSVT expose full amplitudes only in laptop-scale demo mode.

