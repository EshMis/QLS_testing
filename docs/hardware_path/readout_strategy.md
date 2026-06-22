# Readout after the folded QSVT solve

## What the quantum state contains

QSVT prepares a normalized approximation

$$
|Y\rangle=\frac{1}{\|Y\|}\sum_{k=1}^{K}\sum_{j=1}^{D}y_{k,j}|k,j\rangle.
$$

For $K=50$, $D=54$, full classical reconstruction has 2700 coordinates and is
not a sensible output contract. The nine physical concentrations are the first
nine coordinates in each Carleman block. If only the final time is needed, the
target set is nine amplitudes, not 2700. If the scientific output is only
substrate depletion and product yield, it is two linear functionals.

## Why computational-basis sampling is insufficient

Measuring $|k,j\rangle$ estimates

$$
p_{k,j}=|y_{k,j}|^2/\|Y\|^2.
$$

This loses sign/phase and the classical magnitude $\|Y\|$. Concentrations cannot
in general be recovered by taking square roots. The QSVT success amplitude,
known right-hand-side norm, block normalization, and inverse-polynomial scale
must be calibrated to recover the solution norm.

For a real amplitude, prepare a coherent basis reference and use an interference/
Hadamard test to estimate $\operatorname{Re}\langle k,j|Y\rangle$. Complex
amplitudes need a second quadrature. Repeated tests scale as $O(1/\epsilon^2)$
shots; coherent amplitude estimation improves query scaling to $O(1/\epsilon)$
but needs controlled state preparation and its inverse, following
[Brassard et al.](https://arxiv.org/abs/quant-ph/0005055).

`pennylane_overlap_quadratures` implements and tests this interference circuit
for small explicit states, including signed complex amplitudes. In the scalable
path, dense state preparation must be replaced by controlled QSVT solution
preparation and a reversible basis/observable reference circuit.

## Estimate linear observables directly

For weights $w$ supported on physical coordinates, prepare

$$
|w\rangle=\frac{1}{\|w\|}\sum_jw_j|j\rangle
$$

and estimate $\langle w|y_k\rangle$. Product yield, total intermediate pool, or
a weighted assay can each be one overlap rather than nine individual amplitudes.
This is the cleanest route to fewer readout settings.

Classical shadows can estimate many expectation values with logarithmic
dependence on their count in favorable shadow norms
([Huang, Kueng, and Preskill](https://arxiv.org/abs/2002.08953)). They are not a
magic amplitude decoder: computational-basis amplitudes are off-diagonal
coherences relative to a reference and can have poor/nonlocal shadow norms.
Benchmark shadows only after defining the actual assay observables.

## Time postselection and terminal padding

The probability of the final-time block is

$$
p_T=\frac{\|y_K\|^2}{\sum_{k=1}^{K}\|y_k\|^2}.
$$

For the order-2 mass-action finite-lift trajectory with $K=50$, $p_T\approx
0.0391$. Naive postselection needs about 26 preparations per accepted final-time
sample; amplitude amplification has a factor $1/\sqrt{p_T}\approx5.05$.

The implemented terminal-padding equations append $K$ copies satisfying
$z_1=y_K$, $z_{j+1}=z_j$. Accepting any copy doubles dimension from 2700 to
5400 but raises the measured final-subspace probability to about 0.662 and lowers
the amplification factor to 1.23. This is often a better exchange than repeatedly
discarding 96% of prepared history states.

## Honest lower bound on the output promise

No method can return all $D$ arbitrary classical amplitudes with a constant
number of measurements. A quantum advantage must promise a small set of
observables, samples, or decision statistics. This project should use:

1. final $S$ and $P$ as the minimum hardware output;
2. optionally all nine physical concentrations at one or a few selected times;
3. no degree-two lifted-coordinate tomography;
4. confidence intervals including QSVT, postselection, shot, and norm-calibration
   errors; and
5. classical ground-truth comparisons only after all quantum measurements are
   complete.
