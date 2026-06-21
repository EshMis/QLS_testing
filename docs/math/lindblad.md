# Lindblad simulation branch

This pathway simulates an open quantum system; it is **not a quantum linear
solver technique**. For Hamiltonian \(H\) and jumps \(L_j\),

\[
\dot\rho=-i[H,\rho]+
\sum_j\left(L_j\rho L_j^\dagger-	frac12\{L_j^\dagger L_j,\rho\}\right).
\]

Column vectorization gives \(\dot r=\mathcal Lr\), with

\[
\mathcal L=-i(I\otimes H-H^T\otimes I)
+\sum_j\left(L_j^*\otimes L_j
-\tfrac12 I\otimes L_j^\dagger L_j
-\tfrac12(L_j^\dagger L_j)^T\otimes I\right).
\]

The branch uses sparse exponential action on \(\mathcal L\) and reconstructs
density matrices. The UI reports populations, trace error, and minimum
eigenvalue/positivity violation. The amplitude-damping benchmark has
\(p_1(t)=e^{-\gamma t}\) and is tested analytically.

Limitations: the Liouville dimension is \(d^2\); dense observables and positivity
checks become expensive for large Hilbert spaces. Any nonlinear closure or
Carleman approximation for a master equation must be implemented explicitly in
this branch rather than reusing the ODE/QLS path implicitly.

