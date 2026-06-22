# Integrators and other low-readout formulations

## Why ordinary time marching is a poor quantum interface

Backward Euler produces

$$
(I-hA_C)y_{k+1}=y_k.
$$

If each $y_{k+1}$ is measured to prepare the next classical right-hand side,
every step incurs state readout and preparation. QLS output is a normalized
quantum state, so reconstructing 54 lifted amplitudes per step removes the main
reason to use a quantum solver.

An all-at-once method instead prepares

$$
|Y\rangle\propto\sum_{k=1}^{K}|k\rangle|y_k\rangle
$$

with one QSVT solve. Measurement is deferred until the desired time/observable.

## Implemented folded systems

Let $L_1$ and $L_2$ be first- and second-subdiagonal shifts on the $K$-step
time register.

### Folded Backward Euler

$$
M_{BE}=I_K\otimes(I-hA_C)-L_1\otimes I_D.
$$

It is L-stable and has only three Kronecker terms. It is first order, but its
strong damping can partially cancel Carleman truncation error; that cancellation
must not be mistaken for integrator convergence.

### Folded Crank--Nicolson

$$
M_{CN}=I_K\otimes(I-\tfrac h2A_C)-L_1\otimes(I+\tfrac h2A_C).
$$

Expanding gives four Kronecker terms. It is second order and had the best folded
condition estimate in the measured case. It is not L-stable, so high-order fast
Carleman modes may ring.

### Folded BDF2

With a Backward-Euler first row,

$$
M_{BDF2}=I_K\otimes(\tfrac32I-hA_C)
-2L_1\otimes I+\tfrac12L_2\otimes I
-|0\rangle\!\langle0|\otimes\tfrac12I.
$$

The right-hand side contains $y_0$ in block 0 and $-y_0/2$ in block 1. BDF2
is second order with useful stiff damping, but five LCU terms and the largest
observed condition number make its QSVT inverse more expensive.

## Measured tradeoff at the fixed target

For the four-reaction mass-action lift ($D=54$), $T=5$, and $h=0.1$ ($K=50$):

| formulation | matrix | nonzeros | max row nnz | Kronecker terms | $\kappa_2$ estimate |
|---|---:|---:|---:|---:|---:|
| folded BE | 2700×2700 | 11,746 | 6 | 3 | 536 |
| folded CN | 2700×2700 | 18,018 | 10 | 4 | 337 |
| folded BDF2 | 2700×2700 | 14,338 | 7 | 5 | 879 |

The dimension is the same, but QSVT cost depends on more than dimension:
normalization $\alpha$, condition number, inverse-polynomial degree, block-encoder
queries, and final-time success probability all matter. CN is the most attractive
of these three for a first QSVT resource experiment; BDF2 remains the better
stiffness candidate if preconditioning can control $\kappa$.

At the physical target, reducing $h$ does not monotonically reduce total error
because finite Carleman truncation remains. At $t=5$, the exact order-2 lift has
about 0.129 physical-state truncation error in the staged norm. CN/BDF2 converge
toward that finite-lift trajectory, while BE's numerical damping can accidentally
move closer to the nonlinear truth. Integrator error and truncation error must
therefore remain separate dashboard quantities.

## Other one-readout approaches worth exploring

### Taylor-history linear system

[Berry, Childs, Ostrander, and Wang](https://arxiv.org/abs/1701.03684) encode a
truncated Taylor approximation into a sparse, well-conditioned linear system and
prepare a state proportional to a desired final solution. It avoids finite-
difference stability assumptions and has polylogarithmic precision dependence.
For this constant $A_C$, it is the strongest next alternative to folded CN, but
requires a different block layout and factorial/Taylor-index oracles.

### Global spectral collocation

[Childs and Liu](https://arxiv.org/abs/1901.00961) use spectral differential-
equation methods with polylogarithmic precision dependence. A Chebyshev time
matrix can replace $L_1/L_2$ and offers exponential convergence for smooth
solutions. The time operator becomes dense/structured rather than a simple
shift; condition and state-preparation costs must be tested on this non-normal
lift.

### Direct final-time propagator transformation

If only $y(T)$ is needed, one can approximate $e^{TA_C}$ or a rational
approximation directly from a block encoding. This removes the time register
and its postselection probability. The obstacle is that stable nonunitary
exponential transformation over the relevant non-normal spectrum is not simply
the reciprocal QSVT problem; polynomial boundedness and normalization can be
worse than the folded solve.

### Coherent time marching without measurement

One may compose block encodings of a one-step propagator while retaining the
state coherently. Success amplitudes multiply unless oblivious amplitude
amplification or a larger unitary embedding is used. This exchanges readout cost
for depth and success-probability management.

### Recommendation

Implement hardware resources in this order:

1. folded CN + direct singular-value QSVT block encoding;
2. folded BDF2 after structured preconditioning;
3. Taylor-history final-state algorithm;
4. spectral collocation if high precision becomes the main target.

