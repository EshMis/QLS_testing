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
python -m pip install -e '.[dev,app]'
python scripts/run_from_cli.py --config configs/default.yaml
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
- integrators: `backward_euler`, `crank_nicolson`, `bdf2`,
  `folded_backward_euler`, `pade22`, `rk45`, `exponential`
- solvers: `classical`, `hhl_simulator`, `qsvt_simulator`, `vqls_simulator`

The three QLS methods are explicitly called *simulators*: they validate
preprocessing, inverse transformations, scaling, and residuals but make no
quantum-speedup or hardware claim. PennyLane and pyQSP remain optional research
dependencies (`pip install -e '.[quantum]'`); the original circuits are retained
in the source notebook while hardware adapters can be added behind `LinearSolver`.

## Architecture and extension

Each method implements one interface in `qls_testing/core/interfaces.py` and is
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
- [API reference](docs/api_reference.md)

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

