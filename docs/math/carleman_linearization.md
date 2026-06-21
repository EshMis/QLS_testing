# Carleman linearization

## Polynomial lifting

For an autonomous polynomial system

\[
\dot{x}=f(x),\qquad x\in\mathbb{R}^n,
\]

choose every monomial \(m_\alpha(x)=x^\alpha\) with
\(1\leq |\alpha|\leq N\). The chain rule gives

\[
\frac{d}{dt}x^\alpha
=\sum_{i=1}^n \alpha_i x^{\alpha-e_i}f_i(x).
\]

Expanding the right side and retaining only monomials of degree at most \(N\)
produces \(\dot y=A_Ny\). The implementation uses sparse exponent-to-coefficient
dictionaries, avoiding symbolic float conversion and making each matrix entry
traceable to this formula. The number of lifted coordinates is

\[
D(n,N)=\sum_{k=1}^N {n+k-1\choose k}.
\]

The nine-state mass-action model therefore has 9 coordinates at order 1, 54 at
order 2, 219 at order 3, and 714 at order 4.

## Enzyme pathway

The notebook pathway is

\[
S\xrightarrow{E_1}X_1\xrightarrow{E_2}X_2
\xrightarrow{E_3}X_3\xrightarrow{E_4}P,
\]

with \(E_i+S_i\rightleftharpoons C_i\to E_i+P_i\). Eliminating free enzyme via
\(E_i=E_{i,t}-C_i\) yields nine states
\((S,X_1,X_2,X_3,P,C_1,C_2,C_3,C_4)\) and

\[
\dot C_i=k_{1,i}(E_{i,t}-C_i)S_i-(k_{-1,i}+k_{cat,i})C_i.
\]

All nonlinearities are the four bilinear products \(C_iS_i\). At order two,
\(A_N\) contains the exact degree-one equations and truncated equations for all
quadratic observables. Cubic terms generated while differentiating those
observables are discarded. Thus order two represents the original vector field
but does **not** make its finite lifted trajectory exact.

## QSSA polynomial approximation

QSSA uses

\[
v_i=\frac{V_{max,i}S_i}{K_{M,i}+S_i},\quad
V_{max,i}=k_{cat,i}E_{i,t},\quad
K_{M,i}=\frac{k_{-1,i}+k_{cat,i}}{k_{1,i}}.
\]

Carleman lifting requires a polynomial. The retained notebook approximation is

\[
v_i=V_{max,i}\sum_{j=1}^{d}
(-1)^{j-1}\frac{S_i^j}{K_{M,i}^j}+O(S_i^{d+1}),
\]

whose convergence condition is \(|S_i/K_{M,i}|<1\). With the default initial
condition, \(S(0)/K_{M,1}=3/2\), so the series is initially outside its validity
domain. The code records this fact in linearization metadata and compares the
pipeline against the original rational QSSA system.

## Assumptions and diagnostics

The current basis excludes the constant monomial and therefore accepts systems
without forcing terms. Affine systems can be supported later by adding a fixed
homogeneous coordinate. `dropped_term_count`, matrix sparsity, basis labels, and
the exact exponent ordering are retained in `LinearizedSystem.metadata`.

