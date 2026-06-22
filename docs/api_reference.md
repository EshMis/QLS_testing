# API reference

## Core types

- `PolynomialSystem`: sparse polynomial vector field, variable names, and initial state.
- `LinearizedSystem`: lifted matrix, basis, metadata, and physical projection.
- `SolveResult`: solution plus absolute/relative residual and backend metadata.
- `IntegrationResult`: time grid, lifted states, and all solve diagnostics.
- `ExperimentResult`: configuration, pipeline output, reference solution, and errors.
- `LiftedSystemModel`: lifted coordinate names, degrees, lookup, and trace access.
- `ErrorReport`: time-resolved staged discrepancies and descriptions.
- `ComplexityReport`: numerical/quantum proxies, asymptotics, and caveat.
- `GroundTruth` / `DensityGroundTruth`: method-independent ODE or exact
  Liouvillian references from `qls_testing.reference.reference_solver`.
- `NDMEEncoding`: shared operators and the single off-diagonal extraction map
  used by classical, PennyLane, and ground-truth Lindbladian paths.
- `LindbladModel` / `LindbladResult`: separate master-equation representation.
- `NDMELindbladResult`: PDF-based off-diagonal encoding, extraction, and physicality diagnostics.
- `PracticeSystem`: exact-reference matrix benchmark plus structural metadata.

## Main entry points

```python
from qls_testing import load_config, run_experiment

config = load_config("configs/default.yaml")
result = run_experiment(config)
print(result.metrics)
```

`CarlemanLinearization.linearize(system)` constructs a lift.
Every `Integrator.integrate(...)` accepts that lift and a `LinearSolver`.
`trajectory_figure(result)` returns a Plotly figure without rerunning the model.
`variable_figure`, `error_figure`, and the export serializers back the dashboard.

## Registries

`LINEARIZATIONS`, `INTEGRATORS`, and `QLS_SOLVERS` support `register(name,
factory)`, `create(name, **settings)`, and `.names`. Registrations are explicit
in each component package's `__init__.py`; entry-point discovery is intentionally
deferred until third-party plugins are needed.
