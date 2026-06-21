# Implementation status and roadmap

## Fully implemented and tested

- sparse Krylov exponential action;
- adaptive RK tolerances and output stride;
- moving-point QSSA safeguard and fixed-center trust radius;
- adaptive/restarted Carleman order selection;
- iterative refinement as a linear-solver correction loop;
- PennyLane QPE-based HHL, native QSVT, real VQLS, and complex VQLS;
- PDF-based enzyme NDME Lindbladian branch;
- seven-system practice suite;
- staged error/complexity reports and physical-observable Streamlit dashboard.

## Partial but executable

- `loss_mode: shot_proxy` reports finite-shot uncertainty while optimizing the
  exact statevector VQLS loss; likelihood/measurement-trained gradients are not
  yet implemented.
- QSVT runs native PennyLane circuits for small matrices and exposes sparse CSR
  oracle data/cost estimates, but not a scalable coherent sparse-row oracle.
- moving-point QSSA supports explicit/reliable centers; the restarted Carleman
  controller re-embeds states, but does not yet rebuild QSSA Taylor coefficients
  around a new center after every segment.
- HHL/QSVT full-vector extraction is simulator-only; hardware mode should expose
  observables rather than tomography.

## Recommended next additions

1. A shot-derived local VQLS cost based on measured Pauli decompositions.
2. Coherent sparse row/value oracles and hardware-compatible FABLE/QSVT paths.
3. A combined moving-center/restarted-QSSA controller with trust-bound events.
4. Observable-only readout and noise-aware postselection for HHL/QSVT.
5. Sparse density-operator or trajectory methods for NDME order 2.

