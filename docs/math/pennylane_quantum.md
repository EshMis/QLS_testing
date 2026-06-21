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
default, and returns a simulator statevector. Complex ansätze, shot-derived
losses, hardware noise, and observable-only output are future work.

