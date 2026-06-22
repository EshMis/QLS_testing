# Linear-system integrators

After lifting, every method evolves

$$
\dot y=Ay,\qquad y(0)=y_0.
$$

`rk45` and `exponential` are reference methods. The other methods expose one or
more systems $Mx=b$, allowing any registered QLS solver to replace the
classical solve.

## Backward Euler

$$
(I-\Delta t A)y_{k+1}=y_k.
$$

It is first-order and A-stable. The `folded_backward_euler` variant stacks every
step in a single block lower-bidiagonal system with $I-\Delta tA$ on its block
diagonal and $-I$ below it. This is useful for studying one-shot history-state
QLS formulations, although its dense simulation is expensive.

## Crank–Nicolson

$$
(I-\tfrac{\Delta t}{2}A)y_{k+1}
=(I+\tfrac{\Delta t}{2}A)y_k.
$$

This trapezoidal method is second-order and A-stable, but not L-stable; very
stiff modes can oscillate instead of being strongly damped.

## BDF2

$$
(\tfrac32I-\Delta tA)y_{k+1}=2y_k-\tfrac12y_{k-1}.
$$

BDF2 is second-order and A-stable. A backward-Euler step supplies $y_1$, so
the global error includes a one-step startup contribution.

## Global Padé [2/2]

For each requested time $t$, the notebook approximation is

$$
\left(I-\frac t2A+\frac{t^2}{12}A^2\right)y(t)
=\left(I+\frac t2A+\frac{t^2}{12}A^2\right)y_0.
$$

This is a global approximation to $e^{tA}$, not a stepwise fourth-order
integrator. Error can grow at long horizons because the same low-order rational
approximation spans the whole interval. The method name and metadata now make
that distinction explicit.

Every discrete method records condition numbers and per-solve residuals.
`t_final` must be an integer multiple of `dt`; silently rounding the notebook's
number of steps was removed because it can change the requested final time.

## Sparse Krylov exponential action

`krylov_exponential` evaluates $e^{tA}y_0$ with `scipy.sparse.linalg.expm_multiply`
without forming the dense exponential. For Krylov dimension $m$, one action
has the rough sparse cost

$$
O(m\,\mathrm{nnz}(A))+O(m^3),
$$

with backend-selected scaling and convergence controls. `n_points` controls
recording, not the internal stable step sequence.

## Adaptive RK controls

`rk45` passes `rtol`, `atol`, and `max_step` to SciPy's embedded adaptive
Runge--Kutta method. `min_step` is validated and recorded but SciPy RK45 does not
enforce it directly. `output_stride` thins recorded states while retaining the
final point. Adaptive RK is a reference path rather than a QLS discretization.
