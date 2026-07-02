# Variable-intermediate pathway sweeps

This hardware path now supports configurable mass-action pathway chains and
product-to-next-substrate chained segments. The first user-facing input is YAML;
SBML import is intentionally out of scope for v1.

## YAML pathway config

Use `system.name: pathway_graph` and add a top-level `pathway` block:

```yaml
system:
  name: pathway_graph
  initial_substrate: 3.0
pathway:
  mode: chain
  segments:
    - substrate: S
      product: P
      intermediate_count: 3
      intermediate_prefix: X
```

For two chained segments, the previous product must equal the next substrate:

```yaml
pathway:
  mode: chained_segments
  segments:
    - substrate: S
      product: B1
      intermediate_count: 1
      intermediate_prefix: X
    - substrate: B1
      product: P
      intermediate_count: 3
      intermediate_prefix: Y
```

Each segment may also specify `k1`, `k_minus_1`, `kcat`, and `enzyme_total` as a
scalar or a list with one value per enzyme reaction. A segment with `m`
intermediates has `m+1` reactions.

## Dimensions

For one segment with `m` intermediates:

```text
reaction count r = m + 1
physical dimension n = 2r + 1 = 2m + 3
order-2 lifted dimension D = n + n(n+1)/2 = n(n+3)/2
folded dimension = K D, where K = ceil(T / dt)
```

The four-reaction notebook model is the `m = 3` compatibility case.

## Running capped sweeps

Fast default sweep:

```bash
python scripts/analyze_hardware_path.py \
  --intermediate-min 0 \
  --intermediate-max 10 \
  --output outputs/hardware_path/pathway_sweep.json \
  --summary-csv outputs/hardware_path/pathway_sweep.csv
```

Longer capped sweep:

```bash
python scripts/analyze_hardware_path.py \
  --intermediate-min 0 \
  --intermediate-max 20 \
  --continue-until-cap \
  --max-wall-seconds 3600 \
  --max-lifted-dimension 3000 \
  --max-folded-dimension 250000 \
  --estimate-condition \
  --estimate-equilibration \
  --condition-max-dimension 8000 \
  --output outputs/hardware_path/pathway_sweep_long.json
```

The script stops when the requested range completes or when a wall-time, lifted
dimension, folded dimension, or condition-estimation cap is hit.

When `--summary-csv outputs/hardware_path/pathway_sweep.csv` is supplied, the
script writes:

- `outputs/hardware_path/pathway_sweep_intermediates.csv`
- `outputs/hardware_path/pathway_sweep_chained.csv`

These tables contain the main matrix, QSVT, oracle-proxy, and chained-handoff
metrics for quick spreadsheet inspection. The full JSON remains the source of
truth.

Custom chained pathways can be supplied without editing code:

```bash
python scripts/analyze_hardware_path.py \
  --intermediate-min 0 \
  --intermediate-max 5 \
  --chained-case 0,2 \
  --chained-case 4,1,3 \
  --output outputs/hardware_path/custom_chained_cases.json
```

Each comma-separated value is the intermediate count for one segment. For
example, `--chained-case 4,1,3` builds three product-to-next-substrate segments
with 4, 1, and 3 intermediates.

## Report fields

- `intermediate_sweep`: one-segment pathways for `m = 0, 1, 2, ...`.
- `chained_pathway_sweep`: selected product-to-next-substrate cases such as
  `[0,0]`, `[1,1]`, `[1,3]`, and `[5,5]`.
  When `--chained-case` is supplied, those user-specified cases replace the
  defaults and are recorded under `caps.chained_cases`.
- `physical_dimension`, `lifted_dimension`, `nnz`, `density`, row/column nnz,
  `linear_block_nnz`, `quadratic_coupling_nnz`, and `induced_degree_two_nnz`
  describe the order-2 Carleman matrix.
- `folded_systems` compares folded backward Euler, folded Crank-Nicolson, and
  folded BDF2.
- `lcu_alpha_bound` is the top-level LCU normalization bound.
- `structured_block_encoding_estimate` records the register/oracle structure
  intended for scalable compilation:
  - time-shift boundary oracle for folded Euler/CN/BDF2;
  - degree-one sparse `F1` oracle;
  - reaction-local quadratic `F2` oracle;
  - symmetric pair ranking/unranking for degree-two coordinates;
  - reuse of `F1` for the induced symmetric-square block;
  - coefficient lookup or arithmetic oracle for repeated rate values.
- `oracle_gate_proxy_estimate` gives a coarse Toffoli-like proxy per
  block-encoding query and, when a QSVT query estimate is available, a total
  query-scaled proxy. It includes time arithmetic, symmetric pair rank/unrank,
  reaction decoding, sparse row emission, coefficient lookup, and selector
  controls. It excludes routing, fault-tolerant synthesis, magic-state factory
  overhead, and QSVT phase-synthesis cost.
- `qsvt_resource_estimate` is a query proxy using
  `alpha * kappa * log(1/epsilon)`. It is not a compiled gate count.
- `equilibrated_condition_number_estimate` and
  `equilibrated_qsvt_resource_estimate` appear when `--estimate-equilibration`
  is supplied. They use sparse diagonal row/column norm scaling as a
  structure-preserving preconditioning experiment; they are not yet a compiled
  coherent preconditioner.
- `equilibrated_oracle_gate_proxy_estimate` applies the same per-query oracle
  cost to the equilibrated QSVT query proxy when the scaled condition estimate
  is available.
- `final_time_readout` and `terminal_padded_readout_proxy` describe targeted
  physical-coordinate readout instead of full history tomography.
- `monolithic_vs_sequential` compares product-to-next-substrate pathways solved
  as one larger monolithic system versus isolated segment solves with terminal
  product handoff.
- `dynamic_handoff_reference` solves each segment as an exact polynomial ODE,
  passes the terminal product concentration to the next segment, and compares
  the resulting final product to the monolithic exact-polynomial pathway over
  the same time horizon. The two values generally differ because monolithic
  dynamics evolve all segment variables simultaneously.

## Provenance

The evolving reasoning log is kept outside this repo, in the repo owner's
local cross-project workspace (see `AGENTS.md`):

```text
<local workspace>/knowledge/projects/enzyme_qls/provenance/pathway_solver_provenance.md
```
