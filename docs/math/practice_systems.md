# Practice ODE suite

Every practice system is a homogeneous linear ODE $u'=Au$ with exact
reference $u(t)=e^{tA}u_0$. The suite deliberately probes assumptions that a
single friendly SPD matrix would hide.

| Name | Dimension | Structure | Mode |
|---|---:|---|---|
| `practice_sparse_stable_3` | 3 | sparse, triangular, non-Hermitian | fast |
| `practice_dense_non_normal_4` | 4 | dense, stable, non-normal | fast |
| `practice_indefinite_2` | 2 | growing + decaying modes, non-PSD | fast |
| `practice_skew_oscillator_2` | 2 | oscillatory, normal, non-PSD | fast |
| `practice_complex_damped_2` | 2 | complex, sparse, non-Hermitian | fast |
| `practice_sparse_chain_16` | 16 | larger sparse chain | dashboard |
| `practice_dense_non_normal_8` | 8 | larger dense non-normal | dashboard |

Metadata records density, nonzero count, speed class, and structural labels.
The complex case is supported by Carleman order 1, classical/Krylov solvers, and
`pennylane_complex_vqls`. Real-only VQLS rejects it explicitly.

Failure modes are intentional: growing modes violate semidissipative quantum
ODE assumptions; non-normal systems may show transient growth despite stable
eigenvalues; dense systems remove sparse-oracle advantages.

