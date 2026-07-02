# QLS Testing

A reproducible research package for enzyme-pathway dynamics, finite Carleman
linearization, time discretization, and quantum-linear-solver (QLS) experiments.
The original June 15 notebook is preserved unchanged in `notebooks/`; its math
has been separated from tested numerical implementations in `qls_testing/`.

## Quick start

Python 3.10 or newer is supported (3.11+ recommended).

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev,app,quantum]'
python scripts/run_from_cli.py --config configs/default.yaml
```

Override individual YAML values without editing the file:

```bash
python scripts/run_from_cli.py --config configs/default.yaml \
  --set time.dt=0.05 --set linearization.settings.order=3
```

The run writes `outputs/summary.json` and an interactive
`outputs/trajectories.html`. To exercise the complete Carleman → backward Euler
→ idealized HHL path on a tiny case:

```bash
python scripts/run_from_cli.py --config configs/examples/hhl_small_pipeline.yaml
```

For interactive parameter controls:

```bash
streamlit run qls_testing/visualization/app_streamlit.py
```

The dashboard defaults to separate physical `S`, `Xs`, `P`, and `Cs`
pipeline/reference/error graphs. Error display switches between absolute,
relative, or both while preserving one hue per variable. It provides point inspection, hover/zoom,
staged errors, LaTeX math and complexity, CSV/NPZ/HTML exports, and keeps lifted
coordinates under Advanced / Debug.
PDF export is available from the non-interactive script when Kaleido is installed:

```bash
python scripts/generate_plots.py --config configs/default.yaml --pdf
```

Run the fast circuit-backed toy experiment, or switch to its exact baseline:

```bash
python scripts/run_from_cli.py --config configs/examples/toy_pennylane.yaml
python scripts/run_from_cli.py --config configs/examples/toy_classical.yaml
```

The separate Lindblad pathway is:

```bash
python scripts/run_from_cli.py --config configs/examples/lindblad_enzyme_ndme.yaml
```

This implements the supplied PRL paper's nondiagonal density-matrix encoding on
the reduced order-1 nine-variable enzyme model and compares all physical
variables against the exact reduced-model exponential.

The recommended fast PennyLane Lindbladian practice run and optional full
order-1 enzyme channel run are:

```bash
python scripts/run_from_cli.py --config configs/examples/lindblad_practice_pennylane.yaml
python scripts/run_from_cli.py --config configs/examples/lindblad_enzyme_pennylane.yaml
```

Both use the same independent exact-Liouvillian ground truth. The PennyLane path
applies normalized short-time Kraus channels on `default.mixed`.

Working PennyLane HHL and QSVT demos:

```bash
python scripts/run_from_cli.py --config configs/examples/toy_pennylane_hhl.yaml
python scripts/run_from_cli.py --config configs/examples/toy_pennylane_qsvt.yaml
```

Run a small parameter sweep with:

```bash
python scripts/sweep_parameters.py --config configs/default.yaml
```

## Configuration

All important inputs live in YAML: kinetic constants and initial substrate,
Taylor and Carleman orders, time horizon and step, integrator, QLS solver,
solver settings, output behavior, and seed. Unknown keys fail validation instead
of being silently ignored. See `configs/default.yaml` and `configs/examples/`.

Available method names are:

- linearization: `carleman`
- linearization controller: `adaptive_restarted_carleman`
- integrators: `backward_euler`, `crank_nicolson`, `bdf2`,
  `folded_backward_euler`, `folded_crank_nicolson`, `folded_bdf2`, `pade22`,
  `rk45`, `exponential`, `krylov_exponential`
- solvers: `classical`, `hhl_simulator`, `qsvt_simulator`, `vqls_simulator`,
  `pennylane_hhl`, `pennylane_qsvt`, `pennylane_vqls`,
  `pennylane_complex_vqls`, `preconditioned_qsvt`, `iterative_refinement`

The algebraic QLS methods are explicitly called *simulators*: they validate
preprocessing, inverse transformations, scaling, and residuals but make no
quantum-speedup or hardware claim. PennyLane and pyQSP remain optional research
dependencies. PennyLane HHL, QSVT, and real/complex VQLS are working
`default.qubit` backends for tiny systems. Full-vector readout remains a simulator-only verification
convenience, not a quantum complexity claim.

The practice suite provides seven exact-reference systems spanning sparse,
dense, non-normal, growing, oscillatory, complex, and 2--16-dimensional cases.

## Hardware-oriented QSVT path

The focused hardware offshoot fixes mass action + Carleman order 2 and studies
one-readout folded BE/CN/BDF2 systems, structured LCU block encoding, QSVT
conditioning, terminal padding, and observable-level readout:

```bash
python scripts/analyze_hardware_path.py --estimate-condition
python scripts/run_from_cli.py \
  --config configs/examples/hardware_mass_action_folded_qsvt.yaml
```

Variable-intermediate pathway sweeps and product-to-next-substrate chained
pathway comparisons are available with capped resource analysis:

```bash
python scripts/analyze_hardware_path.py \
  --intermediate-min 0 --intermediate-max 10 \
  --output outputs/hardware_path/pathway_sweep.json

python scripts/run_from_cli.py \
  --config configs/examples/pathway_chain_m3_folded_qsvt.yaml
```

See [the hardware-path guide](docs/hardware_path/README.md). The explicit LCU
unitary runs in PennyLane for tiny validation cases; the full 2700-dimensional
target remains an oracle/resource design until its sparse reversible components
are compiled.

For a detailed explanation with diagrams, equations, report-field meanings, and
commands, start with
[the scalable pathway solver walkthrough](docs/hardware_path/pathway_solver_walkthrough.md).

## Architecture and extension

Each method implements `LinearizationMethod`, `Integrator`, `LinearSolver`,
`QuantumLinearSolver`, `ErrorModel`, or `ComplexityEstimator` and is
registered in `qls_testing/experiments/registry.py`. To add a method:

1. implement `LinearizationMethod`, `Integrator`, or `LinearSolver` in a new module;
2. register its factory in the package's `__init__.py`;
3. select that registered name in YAML;
4. add a fast matrix-level test.

The pipeline passes typed `PolynomialSystem`, `LinearizedSystem`, `SolveResult`,
and `IntegrationResult` objects. Visualization consumes `ExperimentResult`, so a
new method automatically works in CLI plots and Streamlit.

## Verification

```bash
pytest
pytest --cov=qls_testing --cov-report=term-missing
ruff check .
```

Tests cover analytic matrix exponentials, a harmonic oscillator, chain-rule
construction of toy Carleman matrices, dimensions of the 54-state metabolic
lift, well-conditioned QLS systems, non-Hermitian HHL dilation, and a tiny full
pipeline suitable for CI.

## Mathematical notes

- [Carleman linearization](docs/math/carleman_linearization.md)
- [Integrators](docs/math/integrators.md)
- [QLS methods](docs/math/qls_methods.md)
- [Error decomposition and complexity](docs/math/error_and_complexity.md)
- [Complexity derivations and runtime provenance](docs/math/complexity_derivations.md)
- [Enzyme ODE and quantum-pipeline problem guide](docs/problem/enzyme_ode_and_quantum_pipeline.md)
- [Implementation rationale and research trail](docs/reasoning/implementation_rationale.md)
- [System-specific method comparisons](docs/reasoning/method_comparisons_specific_system.md)
- [PennyLane circuits](docs/math/pennylane_quantum.md)
- [Lindblad simulation](docs/math/lindblad.md)
- [Practice systems](docs/math/practice_systems.md)
- [Dashboard guide](docs/dashboard.md)
- [Implementation status and roadmap](docs/roadmap.md)
- [API reference](docs/api_reference.md)

Markdown math uses GitHub-compatible `$...$` and `$$...$$` delimiters. Verify it with:

```bash
python scripts/normalize_markdown_math.py --check
```

## Known limitations

- Finite Carleman lifting drops all couplings above the selected degree; order 2
  is the smallest lift representing the original quadratic RHS, not an exact or
  universally minimal-error evolution.
- The QSSA Maclaurin series is invalid initially for the notebook default
  `S(0)/Km1 = 1.5`; its example therefore measures Taylor-model error as well as
  Carleman/integration error.
- Dense lifted matrices scale combinatorially. Order 3 for nine states has 219
  coordinates; hardware-oriented work should exploit sparsity and structured
  block encodings.
- QLS output states do not reveal every classical amplitude for free. Current
  simulators return full vectors strictly for numerical verification.
- Staged errors are computable discrepancy proxies, not rigorous additive upper
  bounds; norms remove signs, so their scalar sum need not equal total error.
- PennyLane quantum modes are tiny-system statevector demonstrations. VQLS uses
  a real sampled probability QNode when `shots` is supplied, while its
  phase-sensitive training loss remains analytic; fully shot-trained losses are
  future work.
