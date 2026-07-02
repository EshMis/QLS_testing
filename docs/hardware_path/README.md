# Hardware path: mass action, Carleman order 2, folded solve, QSVT

This offshoot fixes the scientific choices requested for a first hardware-oriented
design:

- the four-reaction mass-action model;
- Carleman truncation order $N=2$, giving 9 physical and 54 lifted coordinates;
- one all-at-once history-state solve rather than one measured solve per time step;
- QSVT as the intended inverse transformation; and
- observable-focused readout rather than full state tomography.

The implementation is deliberately split into two layers. Small instances have
an explicit PREP/SELECT LCU block-encoding unitary that is executed in a
PennyLane QNode. The full $2700\times2700$ target is represented as sparse
Kronecker terms and oracle/resource plans; materializing its unitary would erase
the advantage being studied.

## Reproduce the analysis

```bash
python scripts/analyze_hardware_path.py --estimate-condition \
  --output outputs/hardware_path/resources.json

python scripts/run_from_cli.py \
  --config configs/examples/hardware_mass_action_folded_qsvt.yaml
```

The YAML example is intentionally short ($K=2$) and uses the classical
`qsvt_simulator` to verify the complete Carleman → folded BDF2 → reciprocal
polynomial pipeline. It is not labelled as hardware execution. The explicit
small block-encoding circuit is tested separately in `tests/test_pennylane.py`.

## What is implemented

- sparse folded Backward Euler, Crank--Nicolson, and BE-bootstrapped BDF2 builders;
- matching integrator plugins, each making exactly one solver call;
- exact Kronecker decompositions with 3, 4, and 5 top-level terms respectively;
- variable-length enzyme-chain models for 1–6 reaction structure experiments;
- verification that the degree-two Carleman block is the symmetric Kronecker
  lift of the 9×9 linear block;
- an exact small-instance LCU block encoding and PennyLane execution;
- terminal-copy padding to improve final-time history-state acceptance; and
- targeted signed-amplitude readout estimates for the nine physical coordinates.

## Main conclusion

Readout count is no longer proportional to all $KD=2700$ history coordinates.
At one selected time, the plan targets nine signed physical amplitudes—or fewer
if only substrate/product observables are scientifically required. This does not
make readout free: signs require interference, QLS normalization must be
recovered, and precision still costs repeated preparations or coherent amplitude
estimation.

The immediate bottleneck is not qubit count but condition number and oracle
quality. At $T=5$, $h=0.1$, the folded condition estimates are 536 (BE), 337
(CN), and 879 (BDF2), implying inverse-polynomial degrees far beyond the current
PennyLane demonstrations. Structured preconditioning is therefore a prerequisite
for a credible QSVT hardware run.

- [Variable-intermediate pathway sweeps](pathway_sweeps.md)
- [Scalable pathway solver walkthrough](pathway_solver_walkthrough.md)
- [Integrator and low-readout alternatives](integrator_options.md)
- [Structured block encoding](structured_block_encoding.md)
- [Readout strategy](readout_strategy.md)
- [Measured resource snapshot](resource_snapshot.md)
