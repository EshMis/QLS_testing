# Structured block encoding of the Carleman history matrix

## Carleman structure is not arbitrary

Write the quadratic mass-action ODE as

$$
\dot x=F_1x+F_2(x\otimes_s x).
$$

In the repository's symmetric monomial basis, the order-2 lift is

$$
A_C=
\begin{bmatrix}
F_1 & F_2\\
0 & F_1\oplus_s F_1
\end{bmatrix}.
$$

The lower-right block is the action induced by differentiating $x_ix_j$:

$$
\frac{d}{dt}(x_ix_j)
=\sum_k(F_1)_{ik}x_kx_j+\sum_k(F_1)_{jk}x_ix_k.
$$

Tests reconstruct this block from the 9×9 $F_1$ and obtain zero numerical
residual. The lower-left block is exactly zero. The only separate nonlinear data
are the sparse 9×45 couplings $F_2$.

For the actual matrix, $F_1$ has 20 nonzeros, $F_2$ has only 8 nonzeros and rank
4, and the induced degree-two block has 152 nonzeros. There are 27 distinct
nonzero coefficient values overall. This favors rate-table lookup plus reversible
index arithmetic over qROM loading of 180 unrelated entries.

## Scaling across the same pathway family

The experiment varies only the number $r$ of reactions/complexes. Physical
dimension is $n=2r+1$ and order-2 dimension is $D=n+n(n+1)/2$.

| reactions $r$ | physical $n$ | lifted $D$ | nnz$(A_C)$ | max row nnz |
|---:|---:|---:|---:|---:|
| 1 | 3 | 9 | 21 | 3 |
| 2 | 5 | 20 | 58 | 4 |
| 3 | 7 | 35 | 111 | 5 |
| 4 | 9 | 54 | 180 | 5 |
| 5 | 11 | 77 | 265 | 5 |
| 6 | 13 | 104 | 366 | 5 |

The row sparsity saturates at five for this local chain even as dimension grows.
This favors sparse-access oracles. It also means a generic dense 54×54 encoder
throws away the most useful problem structure.

The current sweep script also reports the same scaling in terms of intermediate
count $m$, where $r=m+1$ and $n=2m+3$. For chained product-to-next-substrate
pathways, `monolithic_vs_sequential` compares one larger coherent solve against
separate segment solves with terminal product handoff. This is the first place
to check whether joining pathways creates a meaningful block-encoding penalty.

## Exact time/state Kronecker decomposition

Folded BE is exactly

$$
I_K\otimes I_D-hI_K\otimes A_C-L_1\otimes I_D.
$$

CN needs four and BDF2 five top-level terms. A top-level LCU encoder can therefore
use a 2-qubit selector for BE/CN and a 3-qubit selector for BDF2. PREPARE loads
square roots of term normalization weights; SELECT applies the corresponding
time-shift and Carleman component encoders.

`qls_testing.hardware_path.block_encoding` implements this PREP/SELECT
construction exactly for small instances. Every nonunitary term is embedded by
an exact unitary dilation, the LCU unitary is executed by PennyLane, and tests
verify

$$
(\langle0^a|\otimes I)U_M(|0^a\rangle\otimes I)=M/\alpha.
$$

The explicit unitary is a validation device, not the scalable compiler: its
dense materialization is exponential.

## Proposed scalable component oracles

1. **Time shifts.** $L_1$ and $L_2$ are reversible decrement-with-boundary
   operations on the time register. Their sparse block encodings need arithmetic
   and validity flags, not stored matrices.
2. **Degree register.** One qubit selects degree one versus degree two. Controlled
   blocks implement $F_1$, $F_2$, or the induced symmetric-square action.
3. **Reaction-local sparse oracle.** Given a monomial row, reversible logic emits
   at most five column/value pairs. Coefficients come from the four reaction-rate
   tuples. Pair ranking/unranking maps symmetric indices $(i,j)$ to the 45-state
   degree-two register.
4. **Symmetric-square reuse.** The $F_1\oplus_sF_1$ block calls the same $F_1$
   data twice, once on each member of the pair, with multiplicity handling when
   $i=j$. It should not be loaded as 45×45 independent data.
5. **Quadratic coupling.** $F_2$ is reaction-local and sparse; encode only its
   nonzero binding terms.

This is a plausible oracle design, but reversible gate counts for pair ranking,
coefficient loading, and boundary checks still need compilation before a
hardware claim.

`structured_block_encoding_estimate` in the generated JSON makes this oracle
plan machine-readable for every folded system. It records time qubits, lifted
state qubits, the degree and symmetric-pair register sizes, reaction-index
qubits, the top-level LCU selector size, the sparse row-degree bound, and the
component oracle names. These are register/oracle estimates only; they are not
Toffoli/T-counts.

`oracle_gate_proxy_estimate` adds a first coarse reversible-gate proxy. It
models the cost of time-shift arithmetic, symmetric pair rank/unrank, reaction
decoding, sparse row emission, coefficient lookup, and LCU selector controls.
When a QSVT query estimate is available, the report multiplies the per-query
proxy by that query count. This is still below a hardware gate count: routing,
fault-tolerant synthesis, phase synthesis, state preparation, and readout are
not included.

## Alternatives and normalization

The QSVT framework of
[Gilyén, Su, Low, and Wiebe](https://arxiv.org/abs/1806.01838) assumes a block
encoding with normalization $\alpha$. QSVT query count is not enough: a loose
$\alpha$ shrinks singular values and increases inverse degree/success costs.
The measured top-level LCU normalization bounds are about 3.29 for BE/CN and
5.79 for BDF2 before lower-level oracle normalization.

A generic sparse-access bound $s\|A_C\|_{max}$ is 60 for the present lift and is
far looser than desirable. The degree/reaction decomposition should tighten this
normalization; otherwise the five-sparse advantage is consumed by a poor
$\alpha$.

[FABLE](https://arxiv.org/abs/2205.00081) supplies a useful small structured-
matrix fallback and compression experiment. It is not the preferred full target:
generic FABLE can require $O(N^2)$ gates, whereas this matrix already has sparse
reaction and Kronecker oracles.

Native singular-value QSVT can act directly on a non-Hermitian block encoding.
The current PennyLane demo first performs Hermitian dilation, raising a padded
2700-dimensional history register from 12 to 13 data qubits. A hardware compiler
should avoid that extra dilation and implement the left/right singular-vector
mapping directly. Selector, dilation/flag, value, and arithmetic ancillas remain
additional to those 12 qubits.

## Conditioning is the dominant blocker

At $T=5$, $h=0.1$, observed folded condition estimates are 536, 337, and 879.
An inverse polynomial with degree approximately
$O(\kappa\log(1/\epsilon))$ is therefore thousands of block-encoder queries.
Simple block-Jacobi inversion only reduced BE from 536 to 353 and made the
state blocks dense; it is not an attractive hardware preconditioner.

Promising next experiments are time-circulant preconditioners, equilibration
that preserves sparse access, and problem-specific approximate inverses of the
small 54×54 step block. Their preparation/oracle cost must be included alongside
the improved $\kappa$.

`scripts/analyze_hardware_path.py --estimate-equilibration` now measures the
first of these lightweight options: sparse diagonal row/column norm scaling of
the folded matrix. The generated report includes the scaled condition estimate
and the corresponding QSVT query proxy. This is a screening experiment for
condition improvement, not a proof that the diagonal factors can be loaded
coherently at negligible cost.
