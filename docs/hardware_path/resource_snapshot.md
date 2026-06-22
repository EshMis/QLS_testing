# Resource snapshot for the fixed four-reaction target

Run `python scripts/analyze_hardware_path.py --estimate-condition` to regenerate
these values.

## Fixed inputs

| quantity | value |
|---|---:|
| physical variables | 9 |
| Carleman order | 2 |
| lifted dimension $D$ | 54 |
| nonzeros in $A_C$ | 180 |
| maximum $A_C$ row sparsity | 5 |
| horizon $T$ | 5 |
| step $h$ | 0.1 |
| time steps $K$ | 50 |
| history dimension $KD$ | 2700 |
| padded history dimension | 4096 |
| native history data qubits | 12 |
| current-demo Hermitian-dilated data qubits | 13 |

## Folded matrices

| method | nnz | density | max row nnz | top LCU terms | LCU $\alpha$ bound | $\kappa_2$ |
|---|---:|---:|---:|---:|---:|---:|
| BE | 11,746 | 0.161% | 6 | 3 | 3.29 | 536 |
| CN | 18,018 | 0.247% | 10 | 4 | 3.29 | 337 |
| BDF2 | 14,338 | 0.197% | 7 | 5 | 5.79 | 879 |

The matrix remains sparse after folding; dense QSVT input is an implementation
artifact. Condition growth, not matrix storage, is the dominant QSVT issue.

## Readout

| plan | encoded coordinates | accepted final blocks | final-subspace probability | amplification factor |
|---|---:|---:|---:|---:|
| unpadded history | 2700 | 1 | 3.91% | 5.05 |
| append $K=50$ terminal copies | 5400 | 50 | 66.2% | 1.23 |

At one final time, nine physical real amplitudes are targeted. At error
$\epsilon=10^{-2}$, simple independent interference measurements still have a
large shot proxy; amplitude estimation reduces asymptotic precision dependence
but greatly increases coherent circuit requirements. The analysis JSON reports
both proxies rather than presenting either as free readout.

